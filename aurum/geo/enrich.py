"""
Enriquecimento comercial dos leads: quem ocupa o telhado e como falar com eles.

Três fontes, em ordem de confiança:

1. **Tags da própria edificação** -- quando o mapeador já colocou nome,
   telefone e site no polígono. É o caso minoritário.
2. **POIs contidos no polígono** -- no OSM brasileiro a loja/empresa costuma
   ser um nó dentro do prédio, com todos os contatos. É a fonte que mais
   rende, e por isso os POIs da região inteira são baixados de uma vez e
   casados espacialmente contra os telhados.
3. **Reverse geocoding (Nominatim)** -- não dá o nome da empresa, mas fecha o
   endereço da rua para o cabeçalho da proposta.

Nada aqui inventa dados: cada campo carrega a fonte de onde veio, e o que não
foi encontrado permanece ausente em vez de ser preenchido com um palpite.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Iterable, Sequence

from shapely.geometry import Point
from shapely.strtree import STRtree

from ..config import Settings, get_settings
from .nominatim import NominatimClient
from .overpass import OverpassClient
from .roofs import RoofLead

LOGGER = logging.getLogger(__name__)

#: Chaves de contato do OSM, da mais específica para a mais genérica.
WEBSITE_KEYS = ("website", "contact:website", "url", "website:official", "brand:website")
PHONE_KEYS = ("phone", "contact:phone", "contact:mobile", "mobile", "telephone")
EMAIL_KEYS = ("email", "contact:email")
NAME_KEYS = ("name", "official_name", "operator", "brand", "alt_name")
CATEGORY_KEYS = ("shop", "office", "amenity", "industrial", "craft", "healthcare", "tourism", "leisure")

#: Distância máxima, em metros, entre o centroide do telhado e um POI para que
#: o POI seja considerado do mesmo estabelecimento quando não está contido no
#: polígono (galpão cujo nó de portaria fica na calçada).
POI_FALLBACK_RADIUS_M = 25.0


def _first_tag(tags: dict[str, Any], keys: Sequence[str]) -> str | None:
    for key in keys:
        value = tags.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def normalizar_telefone(raw: str | None) -> str | None:
    """
    Normaliza para +55 XX XXXXX-XXXX quando reconhece um número brasileiro.
    Números fora desse padrão são devolvidos apenas com espaços colapsados,
    sem tentativa de reformatação -- melhor manter o original do que corromper.
    """
    if not raw:
        return None
    # o OSM permite múltiplos números separados por ';'
    primeiro = str(raw).split(";")[0].strip()
    digitos = re.sub(r"\D", "", primeiro)

    if digitos.startswith("55") and len(digitos) in (12, 13):
        ddd, resto = digitos[2:4], digitos[4:]
    elif len(digitos) in (10, 11):
        ddd, resto = digitos[:2], digitos[2:]
    else:
        return re.sub(r"\s+", " ", primeiro) or None

    if len(resto) == 9:
        return f"+55 {ddd} {resto[:5]}-{resto[5:]}"
    if len(resto) == 8:
        return f"+55 {ddd} {resto[:4]}-{resto[4:]}"
    return re.sub(r"\s+", " ", primeiro) or None


def normalizar_site(raw: str | None) -> str | None:
    """Garante esquema http(s) e remove espaços."""
    if not raw:
        return None
    url = str(raw).split(";")[0].strip()
    if not url:
        return None
    if not url.startswith(("http://", "https://")):
        url = "https://" + url.lstrip("/")
    return url


def _endereco_de_tags(tags: dict[str, Any]) -> str | None:
    """Monta endereço a partir das tags addr:* quando existirem."""
    rua = tags.get("addr:street")
    if not rua:
        return None
    numero = tags.get("addr:housenumber")
    bairro = tags.get("addr:suburb") or tags.get("addr:neighbourhood")
    cidade = tags.get("addr:city")
    uf = tags.get("addr:state")
    cep = tags.get("addr:postcode")

    partes = [f"{rua}, {numero}" if numero else str(rua)]
    if bairro:
        partes.append(str(bairro))
    local = "/".join(str(p) for p in (cidade, uf) if p)
    if local:
        partes.append(local)
    if cep:
        partes.append(f"CEP {cep}")
    return " - ".join(partes)


def perfil_de_tags(tags: dict[str, Any], fonte: str) -> dict[str, Any]:
    """Extrai um perfil comercial de um conjunto de tags do OSM."""
    perfil: dict[str, Any] = {}
    nome = _first_tag(tags, NAME_KEYS)
    if nome:
        perfil["nome"] = nome
    operador = _first_tag(tags, ("operator", "brand"))
    if operador and operador != perfil.get("nome"):
        perfil["operador"] = operador

    site = normalizar_site(_first_tag(tags, WEBSITE_KEYS))
    if site:
        perfil["site"] = site
    telefone = normalizar_telefone(_first_tag(tags, PHONE_KEYS))
    if telefone:
        perfil["telefone"] = telefone
    email = _first_tag(tags, EMAIL_KEYS)
    if email:
        perfil["email"] = email.split(";")[0].strip()

    categoria = _first_tag(tags, CATEGORY_KEYS)
    if categoria:
        perfil["categoria"] = categoria
    endereco = _endereco_de_tags(tags)
    if endereco:
        perfil["endereco"] = endereco
    if tags.get("opening_hours"):
        perfil["horario"] = str(tags["opening_hours"])
    if tags.get("cnpj") or tags.get("ref:vatin") or tags.get("ref:CNPJ"):
        perfil["cnpj"] = str(tags.get("cnpj") or tags.get("ref:vatin") or tags.get("ref:CNPJ"))

    if perfil:
        perfil["fonte"] = fonte
    return perfil


def _mesclar(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    """
    Mescla preservando o que já existe: a primeira fonte a preencher um campo
    vence, porque as fontes são consultadas em ordem decrescente de confiança.
    """
    resultado = dict(base)
    fontes = [f for f in (base.get("fonte"), extra.get("fonte")) if f]
    for chave, valor in extra.items():
        if chave == "fonte":
            continue
        if valor and not resultado.get(chave):
            resultado[chave] = valor
    if fontes:
        resultado["fonte"] = "+".join(dict.fromkeys(fontes))
    return resultado


def _poi_point(element: dict[str, Any]) -> Point | None:
    """Ponto de um elemento POI, aceitando nó (lat/lon) ou way (center)."""
    if "lat" in element and "lon" in element:
        return Point(float(element["lon"]), float(element["lat"]))
    center = element.get("center")
    if isinstance(center, dict) and "lat" in center and "lon" in center:
        return Point(float(center["lon"]), float(center["lat"]))
    return None


def enriquecer_com_pois(
    leads: Sequence[RoofLead],
    pois: Iterable[dict[str, Any]],
) -> int:
    """
    Casa POIs com telhados por contenção espacial, in-place.

    Devolve quantos leads receberam ao menos um campo novo. Um POI contido no
    polígono é considerado do estabelecimento; POIs apenas próximos entram como
    fallback e ficam marcados como tal, para não sugerir mais certeza do que há.
    """
    leads = list(leads)
    if not leads:
        return 0

    pontos: list[Point] = []
    tags_por_ponto: list[dict[str, Any]] = []
    for element in pois:
        ponto = _poi_point(element)
        tags = element.get("tags") or {}
        if ponto is None or not tags:
            continue
        pontos.append(ponto)
        tags_por_ponto.append(tags)

    if not pontos:
        return 0

    arvore = STRtree(pontos)
    enriquecidos = 0

    for lead in leads:
        antes = dict(lead.company)
        perfil = dict(lead.company)

        # 1) POIs efetivamente dentro do polígono do telhado.
        contidos = [
            tags_por_ponto[int(i)]
            for i in arvore.query(lead.geometry)
            if pontos[int(i)].within(lead.geometry)
        ]
        # POI com mais campos de contato primeiro: é o registro mais completo.
        contidos.sort(key=lambda t: -sum(1 for k in (*NAME_KEYS, *PHONE_KEYS, *WEBSITE_KEYS) if t.get(k)))
        for tags in contidos:
            perfil = _mesclar(perfil, perfil_de_tags(tags, "osm_poi_interno"))

        # 2) Fallback por proximidade quando nada foi encontrado dentro.
        if not perfil.get("nome"):
            # O buffer parte da **borda** do polígono, não do centroide: num
            # galpão de 120 m o nó de portaria fica a poucos metros da parede
            # e a dezenas de metros do centro.
            raio_graus = POI_FALLBACK_RADIUS_M / 111_000.0
            vizinhanca = lead.geometry.buffer(raio_graus)
            proximos = [
                tags_por_ponto[int(i)]
                for i in arvore.query(vizinhanca)
                if pontos[int(i)].within(vizinhanca)
            ]
            proximos.sort(key=lambda t: -sum(1 for k in (*NAME_KEYS, *PHONE_KEYS, *WEBSITE_KEYS) if t.get(k)))
            for tags in proximos[:1]:
                perfil = _mesclar(perfil, perfil_de_tags(tags, "osm_poi_proximo"))

        lead.company = perfil
        if perfil != antes:
            enriquecidos += 1

    return enriquecidos


def enriquecer_leads(
    leads: Sequence[RoofLead],
    settings: Settings | None = None,
    overpass: OverpassClient | None = None,
    nominatim: NominatimClient | None = None,
    bbox: Sequence[float] | None = None,
    resolver_endereco: bool = True,
    limite_endereco: int = 50,
) -> dict[str, Any]:
    """
    Enriquece uma lista de leads in-place e devolve estatísticas.

    ``limite_endereco`` existe porque o Nominatim exige 1 req/s: resolver o
    endereço de 500 leads levaria mais de 8 minutos. O padrão cobre os
    primeiros 50 -- na prática, os que o usuário vai de fato selecionar.
    """
    settings = settings or get_settings()
    leads = list(leads)
    stats: dict[str, Any] = {
        "leads": len(leads),
        "com_tags_proprias": 0,
        "enriquecidos_por_poi": 0,
        "com_endereco": 0,
        "com_nome": 0,
        "com_contato": 0,
    }
    if not leads:
        return stats

    # 1) Tags da própria edificação.
    for lead in leads:
        perfil = perfil_de_tags(lead.tags, "osm_edificacao")
        if perfil:
            stats["com_tags_proprias"] += 1
        lead.company = _mesclar(lead.company, perfil)

    # 2) POIs da região.
    overpass = overpass or OverpassClient(settings)
    if bbox is None:
        lats = [lead.centroid_lat for lead in leads]
        lons = [lead.centroid_lon for lead in leads]
        margem = 0.002
        bbox = (min(lons) - margem, min(lats) - margem, max(lons) + margem, max(lats) + margem)
    pois = overpass.fetch_pois_in_bbox(bbox)
    LOGGER.info("POIs baixados para enriquecimento: %s", len(pois))
    stats["pois_baixados"] = len(pois)
    stats["enriquecidos_por_poi"] = enriquecer_com_pois(leads, pois)

    # 3) Endereço por reverse geocoding, apenas para os melhores leads.
    if resolver_endereco and limite_endereco > 0:
        nominatim = nominatim or NominatimClient(settings)
        for lead in leads[:limite_endereco]:
            if lead.company.get("endereco"):
                continue
            endereco = nominatim.reverse(lead.centroid_lat, lead.centroid_lon)
            if endereco is None:
                continue
            lead.company["endereco"] = endereco.linha_curta()
            lead.company.setdefault("cidade", endereco.city)
            lead.company.setdefault("uf", endereco.state)
            lead.company["fonte"] = "+".join(
                dict.fromkeys(filter(None, [lead.company.get("fonte"), "nominatim"]))
            )

    for lead in leads:
        if lead.company.get("endereco"):
            stats["com_endereco"] += 1
        if lead.company.get("nome"):
            stats["com_nome"] += 1
        if lead.company.get("telefone") or lead.company.get("site") or lead.company.get("email"):
            stats["com_contato"] += 1

    return stats
