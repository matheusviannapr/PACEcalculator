"""
Identificação e qualificação de telhados a partir de elementos do OSM.

Fluxo: elementos crus do Overpass -> geometria válida -> métricas métricas ->
filtros de qualidade -> pontuação -> deduplicação -> lista ordenada de
:class:`RoofLead`.

A pontuação é deliberadamente enviesada para **área**, porque em prospecção
comercial de geração solar o tamanho do telhado é o que determina o tamanho do
contrato. Forma e orientação entram como qualificadores.
"""
from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Iterable, Sequence

from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree

from .geometry import (
    ShapeMetrics,
    compute_metrics,
    intersection_over_union,
    normalize,
    project_to_utm,
    to_geojson_geometry,
)
from . import segments
from .overpass import has_explicit_solar

LOGGER = logging.getLogger(__name__)

#: Tipos de edificação que quase nunca rendem um projeto: pequenos, informais
#: ou sem responsável comercial identificável.
EXCLUDED_BUILDINGS = {
    "hut", "shed", "garage", "garages", "carport", "kiosk", "cabin",
    "roof", "ruins", "construction", "static_caravan", "houseboat",
    "bunker", "tent", "container", "toilets", "service", "stable",
}

#: Edificação residencial: entra só se for muito grande (condomínio, prédio).
RESIDENTIAL_BUILDINGS = {
    "residential", "house", "apartments", "detached", "semidetached_house",
    "terrace", "bungalow", "dormitory", "farm",
}

#: Alvos preferenciais: consumo diurno alto e decisor corporativo único.
PRIORITY_BUILDINGS = {
    "industrial", "warehouse", "factory", "manufacture", "commercial",
    "retail", "supermarket", "office", "school", "university", "college",
    "hospital", "clinic", "hotel", "civic", "public", "government",
    "sports_centre", "sports_hall", "stadium", "church", "cathedral",
    "logistics", "storage_tank",
}

#: A taxonomia de segmentos vive em `aurum.geo.segments`. Estes grupos são
#: mantidos só para não quebrar código que ainda os importe.
TARGET_GROUPS: dict[str, set[str]] = {
    "industrial": {"industrial", "warehouse", "factory", "manufacture", "logistics"},
    "comercial": {"commercial", "retail", "supermarket", "office", "hotel", "mall"},
    "institucional": {
        "school", "university", "college", "hospital", "clinic", "civic",
        "public", "government", "church", "cathedral", "sports_centre",
        "sports_hall", "stadium",
    },
}
TARGET_GROUPS["misto"] = set().union(*TARGET_GROUPS.values())


class ContextoUsoSolo:
    """
    Índice espacial das áreas de uso do solo de uma região.

    Existe por causa do ``building=yes``, que é a maioria do mapeamento
    brasileiro: a edificação não diz o que é, mas a área que a contém diz.
    Um galpão sem tag dentro de um ``landuse=farmyard`` é agro; dentro de um
    ``landuse=industrial``, é indústria. Sem isso, o filtro por segmento
    devolveria quase nada fora dos grandes centros.
    """

    def __init__(self, areas: Iterable[dict[str, Any]] | None = None) -> None:
        self._geoms: list[BaseGeometry] = []
        self._usos: list[str] = []
        self._arvore: STRtree | None = None
        if areas:
            self.carregar(areas)

    def carregar(self, elementos: Iterable[dict[str, Any]]) -> int:
        """Indexa elementos do Overpass que carreguem landuse/amenity de área."""
        for elemento in elementos:
            tags = elemento.get("tags") or {}
            uso = (
                tags.get("landuse")
                or tags.get("amenity")
                or tags.get("leisure")
                or tags.get("tourism")
            )
            if not uso:
                continue
            geom = normalize(element_to_geometry(elemento))
            if geom is None or geom.is_empty:
                continue
            self._geoms.append(geom)
            self._usos.append(str(uso).strip().lower())

        self._arvore = STRtree(self._geoms) if self._geoms else None
        return len(self._geoms)

    def __len__(self) -> int:
        return len(self._geoms)

    def uso_em(self, geom: BaseGeometry) -> str | None:
        """
        Uso do solo da menor área que contém a edificação.

        A menor porque as áreas se aninham: um ``landuse=farmyard`` costuma
        estar dentro de um ``landuse=farmland``, e o pátio da fazenda descreve
        melhor o prédio do que a lavoura inteira.
        """
        if self._arvore is None:
            return None
        ponto = geom.representative_point()
        candidatos = [
            indice for indice in self._arvore.query(ponto)
            if self._geoms[int(indice)].contains(ponto)
        ]
        if not candidatos:
            return None
        menor = min(candidatos, key=lambda i: self._geoms[int(i)].area)
        return self._usos[int(menor)]


@dataclass
class RoofFilterConfig:
    """
    Limiares de aceitação de um telhado.

    Os defaults são permissivos de propósito: em prospecção, um falso positivo
    custa uma olhada no satélite, enquanto um falso negativo é um contrato
    perdido que ninguém jamais vê.
    """

    min_area_m2: float = 300.0
    max_area_m2: float | None = None
    compactness_min: float = 0.12
    convexity_min: float = 0.70
    rectangularity_min: float = 0.55
    aspect_ratio_max: float = 10.0
    vertex_count_max: int = 120
    hole_area_ratio_max: float = 0.20
    min_width_m: float = 6.0
    #: Exclui telhados que já declaram solar instalado.
    skip_existing_solar: bool = True
    #: Residencial só passa acima deste limite (condomínio/prédio grande).
    residential_min_area_m2: float = 900.0
    #: IoU acima do qual dois polígonos são o mesmo telhado.
    dedup_iou: float = 0.80

    @classmethod
    def rigoroso(cls, base: "RoofFilterConfig | None" = None) -> "RoofFilterConfig":
        """Perfil conservador: menos leads, geometria mais confiável."""
        base = base or cls()
        return cls(
            min_area_m2=max(base.min_area_m2, 500.0),
            max_area_m2=base.max_area_m2,
            compactness_min=0.20,
            convexity_min=0.82,
            rectangularity_min=0.70,
            aspect_ratio_max=6.0,
            vertex_count_max=60,
            hole_area_ratio_max=0.08,
            min_width_m=8.0,
            skip_existing_solar=base.skip_existing_solar,
            residential_min_area_m2=base.residential_min_area_m2,
            dedup_iou=base.dedup_iou,
        )


@dataclass
class RoofLead:
    """Um telhado qualificado, pronto para virar proposta."""

    osm_type: str
    osm_id: int
    geometry: BaseGeometry = field(repr=False)          # WGS84
    geometry_utm: BaseGeometry = field(repr=False)      # métrico
    utm_epsg: int
    metrics: ShapeMetrics
    tags: dict[str, Any]
    building_type: str
    centroid_lat: float
    centroid_lon: float
    score: float = 0.0
    score_breakdown: dict[str, float] = field(default_factory=dict)
    #: Preenchido por aurum.geo.enrich.
    company: dict[str, Any] = field(default_factory=dict)
    #: Chave do segmento de prospecção (ver aurum.geo.segments).
    segmento: str = segments.SEGMENTO_DESCONHECIDO.chave
    #: Uso do solo da área que contém a edificação, quando conhecido.
    landuse_contexto: str | None = None

    @property
    def segmento_rotulo(self) -> str:
        """Nome legível do segmento, para tabela e proposta."""
        if self.segmento == segments.SEGMENTO_DESCONHECIDO.chave:
            return segments.SEGMENTO_DESCONHECIDO.rotulo
        segmento = segments.POR_CHAVE.get(self.segmento)
        return segmento.rotulo if segmento else self.segmento

    @property
    def tipo_para_calculo(self) -> str:
        """
        Tipo a usar nas tabelas de consumo, autoconsumo e obstáculos.

        Não é o mesmo que ``building_type``: um galpão de fazenda quase sempre
        vem como ``building=yes`` e só é reconhecido como agro pelo uso do
        solo. Dimensioná-lo pelo tipo cru o trataria como prédio urbano
        genérico, com a faixa de consumo e o fator de obstáculos errados.
        """
        return segments.tipo_para_calculo(self.building_type, self.segmento)

    @property
    def lead_id(self) -> str:
        return f"{self.osm_type}/{self.osm_id}"

    @property
    def name(self) -> str:
        """Melhor nome disponível, com fallback legível."""
        for key in ("name", "operator", "brand", "official_name", "alt_name"):
            value = self.tags.get(key)
            if value:
                return str(value)
        empresa = self.company.get("nome")
        if empresa:
            return str(empresa)
        return f"Edificação sem nome ({self.building_type})"

    @property
    def levels(self) -> int:
        """Número de pavimentos, 1 quando não informado."""
        raw = self.tags.get("building:levels") or self.tags.get("levels")
        try:
            return max(1, int(float(str(raw).split(";")[0])))
        except (TypeError, ValueError):
            return 1

    @property
    def osm_url(self) -> str:
        return f"https://www.openstreetmap.org/{self.osm_type}/{self.osm_id}"

    @property
    def maps_url(self) -> str:
        return f"https://www.google.com/maps/search/?api=1&query={self.centroid_lat:.6f},{self.centroid_lon:.6f}"

    def to_record(self) -> dict[str, Any]:
        """Linha achatada para CSV/DataFrame (sem geometria)."""
        record: dict[str, Any] = {
            "lead_id": self.lead_id,
            "nome": self.name,
            "segmento": self.segmento,
            "segmento_rotulo": self.segmento_rotulo,
            "tipo_calculo": self.tipo_para_calculo,
            "landuse_contexto": self.landuse_contexto,
            "tipo_edificacao": self.building_type,
            "area_m2": round(self.metrics.area_m2, 1),
            "pavimentos": self.levels,
            "score": round(self.score, 1),
            "largura_min_m": round(self.metrics.min_width_m, 1),
            "comprimento_m": round(self.metrics.max_length_m, 1),
            "retangularidade": round(self.metrics.rectangularity, 3),
            "azimute_cumeeira": (
                round(self.metrics.ridge_azimuth_deg, 1)
                if self.metrics.ridge_azimuth_deg is not None
                else None
            ),
            "latitude": round(self.centroid_lat, 6),
            "longitude": round(self.centroid_lon, 6),
            "osm_url": self.osm_url,
            "maps_url": self.maps_url,
        }
        for key in ("nome", "site", "telefone", "email", "endereco", "categoria", "fonte"):
            record[f"empresa_{key}"] = self.company.get(key)
        return record

    def to_geojson_feature(self) -> dict[str, Any]:
        props = self.to_record()
        props["tags_osm"] = self.tags
        props["score_breakdown"] = self.score_breakdown
        props["empresa"] = self.company
        return {
            "type": "Feature",
            "geometry": to_geojson_geometry(self.geometry),
            "properties": props,
        }

    @classmethod
    def from_geojson_feature(cls, feature: dict[str, Any]) -> "RoofLead | None":
        """
        Reconstrói um lead a partir do GeoJSON exportado.

        É o que permite separar prospecção e dimensionamento em dois comandos:
        varre-se a região uma vez, olha-se a lista com calma, e só depois
        calcula-se o que foi escolhido -- sem consultar o OpenStreetMap de novo.

        As métricas são recalculadas da geometria em vez de lidas das
        propriedades: são baratas de obter e assim não há risco de o arquivo
        trazer números defasados de uma versão anterior do código.
        """
        from shapely.geometry import shape

        props = feature.get("properties") or {}
        geometria = feature.get("geometry")
        if not geometria:
            return None

        geom = normalize(shape(geometria))
        if geom is None:
            return None
        geom_utm, epsg = project_to_utm(geom)
        metricas = compute_metrics(geom_utm)
        if metricas is None:
            return None

        identificador = str(props.get("lead_id") or "way/0")
        tipo, _, numero = identificador.partition("/")
        tags = props.get("tags_osm") or {}
        centroide = geom.centroid

        empresa = props.get("empresa")
        if not isinstance(empresa, dict):
            # Formato achatado (empresa_nome, empresa_site, ...), como no CSV.
            empresa = {
                chave.removeprefix("empresa_"): valor
                for chave, valor in props.items()
                if chave.startswith("empresa_") and valor
            }

        return cls(
            osm_type=tipo or "way",
            osm_id=int(numero) if numero.isdigit() else 0,
            geometry=geom,
            geometry_utm=geom_utm,
            utm_epsg=epsg,
            metrics=metricas,
            tags=dict(tags),
            building_type=str(props.get("tipo_edificacao") or building_type_of(tags)),
            centroid_lat=float(props.get("latitude", centroide.y)),
            centroid_lon=float(props.get("longitude", centroide.x)),
            score=float(props.get("score") or 0.0),
            score_breakdown=dict(props.get("score_breakdown") or {}),
            company=dict(empresa),
            # Sem restaurar o segmento, a proposta de um galpão de fazenda
            # sairia com os parâmetros genéricos de prédio urbano.
            segmento=str(props.get("segmento") or segments.SEGMENTO_DESCONHECIDO.chave),
            landuse_contexto=props.get("landuse_contexto"),
        )


# ----------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------
def _ring_from_nodes(nodes: Sequence[dict[str, float]]) -> list[tuple[float, float]] | None:
    """Converte a lista de nós do Overpass em um anel fechado (lon, lat)."""
    if not nodes:
        return None
    coords = [(float(n["lon"]), float(n["lat"])) for n in nodes if "lon" in n and "lat" in n]
    if len(coords) < 3:
        return None
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    # Após fechar, ainda é preciso ter 3 vértices distintos.
    if len(coords) < 4:
        return None
    return coords


def _polygon_from_way(element: dict[str, Any]) -> BaseGeometry | None:
    ring = _ring_from_nodes(element.get("geometry") or [])
    if ring is None:
        return None
    try:
        return Polygon(ring)
    except Exception:
        return None


def _polygon_from_relation(element: dict[str, Any]) -> BaseGeometry | None:
    """
    Monta um multipolígono a partir dos membros outer/inner de uma relação.

    Os anéis internos são atribuídos ao anel externo que os contém, e não a
    todos indiscriminadamente -- do contrário um pátio interno de um galpão
    apareceria como vazio em prédios vizinhos da mesma relação.
    """
    outers: list[Polygon] = []
    inners: list[Polygon] = []
    for member in element.get("members", []):
        if member.get("type") != "way":
            continue
        ring = _ring_from_nodes(member.get("geometry") or [])
        if ring is None:
            continue
        try:
            poly = Polygon(ring)
        except Exception:
            continue
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            continue
        (inners if member.get("role") == "inner" else outers).append(poly)

    if not outers:
        return None

    built: list[Polygon] = []
    for outer in outers:
        if outer.geom_type != "Polygon":
            continue
        holes = [
            inner.exterior.coords
            for inner in inners
            if inner.geom_type == "Polygon" and inner.representative_point().within(outer)
        ]
        try:
            built.append(Polygon(outer.exterior.coords, holes))
        except Exception:
            built.append(outer)

    if not built:
        return None
    return built[0] if len(built) == 1 else MultiPolygon(built)


def element_to_geometry(element: dict[str, Any]) -> BaseGeometry | None:
    """Geometria WGS84 de um elemento do Overpass, ou None se inaproveitável."""
    kind = element.get("type")
    if kind == "way":
        return _polygon_from_way(element)
    if kind == "relation":
        return _polygon_from_relation(element)
    return None


def building_type_of(tags: dict[str, Any]) -> str:
    """
    Classe da edificação. Tags de uso (industrial, shop, amenity) têm
    precedência sobre um genérico ``building=yes``, que não diz nada.
    """
    building = str(tags.get("building") or tags.get("building:part") or "").strip().lower()
    if building and building not in {"yes", "true", "1"}:
        return building
    for key in ("building:use", "industrial", "man_made", "amenity", "shop", "office", "landuse"):
        value = tags.get(key)
        if value:
            return str(value).strip().lower()
    return building or "desconhecido"


# ----------------------------------------------------------------------
# Filtros e pontuação
# ----------------------------------------------------------------------
def rejection_reasons(
    metrics: ShapeMetrics,
    tags: dict[str, Any],
    config: RoofFilterConfig,
) -> list[str]:
    """Lista de motivos para descartar um telhado. Vazia = aprovado."""
    reasons: list[str] = []
    kind = building_type_of(tags)

    if kind in EXCLUDED_BUILDINGS:
        reasons.append("tipo_excluido")
    if config.skip_existing_solar and has_explicit_solar(tags):
        reasons.append("solar_existente")
    if metrics.area_m2 < config.min_area_m2:
        reasons.append("area_minima")
    if config.max_area_m2 is not None and metrics.area_m2 > config.max_area_m2:
        reasons.append("area_maxima")
    if kind in RESIDENTIAL_BUILDINGS and metrics.area_m2 < config.residential_min_area_m2:
        reasons.append("residencial_pequeno")
    if metrics.compactness < config.compactness_min:
        reasons.append("compacidade")
    if metrics.convexity < config.convexity_min:
        reasons.append("convexidade")
    if metrics.rectangularity < config.rectangularity_min:
        reasons.append("retangularidade")
    if metrics.aspect_ratio > config.aspect_ratio_max:
        reasons.append("alongamento")
    if metrics.vertex_count > config.vertex_count_max:
        reasons.append("vertices_demais")
    if metrics.hole_area_ratio > config.hole_area_ratio_max:
        reasons.append("vazios_demais")
    if metrics.min_width_m < config.min_width_m:
        reasons.append("largura_minima")
    return reasons


def matches_target(
    tags: dict[str, Any],
    target: str | Sequence[str],
    landuse_contexto: str | None = None,
) -> bool:
    """
    True se a edificação pertence a algum dos segmentos pedidos.

    Aceita um segmento, vários separados por vírgula, uma lista, os nomes
    antigos de grupo e "todos".
    """
    return segments.pertence(tags, segments.normalizar_alvos(target), landuse_contexto)


def score_roof(
    metrics: ShapeMetrics,
    tags: dict[str, Any],
    config: RoofFilterConfig,
    landuse_contexto: str | None = None,
) -> tuple[float, dict[str, float]]:
    """
    Pontua um telhado de 0 a 100, devolvendo também a decomposição para que a
    interface possa explicar ao usuário por que um lead ficou à frente do outro.

    Pesos: área 45, forma 25, tipo de uso 20, qualidade dos dados 10.
    """
    breakdown: dict[str, float] = {}

    # Área: escala logarítmica entre o mínimo e 20.000 m2. Log porque a
    # diferença entre 300 e 3.000 m2 importa muito mais do que entre
    # 17.000 e 20.000 -- ambos já são projetos grandes.
    piso = max(config.min_area_m2, 50.0)
    teto = 20000.0
    if metrics.area_m2 <= piso:
        area_pts = 0.0
    else:
        ratio = math.log(metrics.area_m2 / piso) / math.log(teto / piso)
        area_pts = 45.0 * min(1.0, max(0.0, ratio))
    breakdown["area"] = area_pts

    # Forma: telhado retangular, convexo e sem recortes é mais fácil de
    # ocupar com módulos e de orçar.
    forma = (
        min(1.0, metrics.rectangularity) * 10.0
        + min(1.0, metrics.convexity) * 7.0
        + min(1.0, metrics.compactness / 0.6) * 4.0
        + max(0.0, 1.0 - max(0.0, metrics.aspect_ratio - 2.5) / 6.0) * 4.0
    )
    forma -= min(4.0, metrics.hole_area_ratio * 20.0)
    forma_pts = max(0.0, min(25.0, forma))
    breakdown["forma"] = forma_pts

    # Uso: quanto o segmento costuma render como cliente de geração solar.
    # O peso está declarado em cada segmento, com a justificativa ao lado.
    segmento = segments.classificar(tags, landuse_contexto)
    uso_pts = segmento.peso_comercial
    # Casa isolada não é condomínio: mesmo caindo no segmento residencial,
    # não tem área comum nem síndico para decidir.
    if building_type_of(tags) in {"house", "detached", "semidetached_house", "bungalow"}:
        uso_pts = min(uso_pts, 5.0)
    breakdown["uso"] = uso_pts

    # Qualidade dos dados: um lead com nome, contato e altura é acionável
    # comercialmente hoje; um polígono anônimo exige pesquisa manual.
    dados = 0.0
    if any(tags.get(k) for k in ("name", "operator", "brand")):
        dados += 4.0
    if any(tags.get(k) for k in ("website", "contact:website", "url")):
        dados += 2.0
    if any(tags.get(k) for k in ("phone", "contact:phone", "contact:mobile")):
        dados += 2.0
    if tags.get("addr:street"):
        dados += 1.0
    if tags.get("building:levels") or tags.get("height"):
        dados += 1.0
    dados_pts = min(10.0, dados)
    breakdown["dados"] = dados_pts

    total = area_pts + forma_pts + uso_pts + dados_pts
    return round(min(100.0, max(0.0, total)), 2), breakdown


# ----------------------------------------------------------------------
# Deduplicação
# ----------------------------------------------------------------------
def deduplicate(leads: list[RoofLead], iou_threshold: float = 0.80) -> tuple[list[RoofLead], int]:
    """
    Remove telhados duplicados (mesma edificação mapeada como way e como
    building:part, ou importações sobrepostas), mantendo o de maior score.

    Usa índice espacial STRtree: a versão anterior comparava todos contra
    todos, o que em uma cidade com milhares de candidatos custava minutos.
    """
    if len(leads) < 2:
        return list(leads), 0

    ordered = sorted(leads, key=lambda lead: lead.score, reverse=True)
    geoms = [lead.geometry for lead in ordered]
    tree = STRtree(geoms)

    descartados: set[int] = set()
    for i, lead in enumerate(ordered):
        if i in descartados:
            continue
        for j in tree.query(lead.geometry):
            j = int(j)
            if j <= i or j in descartados:
                continue
            if intersection_over_union(lead.geometry, ordered[j].geometry) >= iou_threshold:
                descartados.add(j)

    mantidos = [lead for i, lead in enumerate(ordered) if i not in descartados]
    return mantidos, len(descartados)


# ----------------------------------------------------------------------
# Pipeline de identificação
# ----------------------------------------------------------------------
def build_leads(
    elements: Iterable[dict[str, Any]],
    config: RoofFilterConfig | None = None,
    target: str | Sequence[str] = "todos",
    aoi: BaseGeometry | None = None,
    contexto_uso_solo: "ContextoUsoSolo | None" = None,
) -> tuple[list[RoofLead], dict[str, Any]]:
    """
    Converte elementos crus do Overpass em leads qualificados e ordenados.

    ``aoi`` restringe a região quando o contorno administrativo é conhecido --
    o bbox de uma cidade sempre inclui território de vizinhas.

    ``contexto_uso_solo`` classifica as edificações sem tag pelo uso do solo
    que as contém. É o que faz um galpão ``building=yes`` dentro de um
    ``landuse=farmyard`` aparecer no filtro de agro.
    """
    config = config or RoofFilterConfig()
    alvos = segments.normalizar_alvos(target)

    # Quem procura condomínio quer prédio residencial; manter o piso de área
    # que existe para descartar casa isolada esvaziaria justamente o alvo.
    if "condominio" in alvos:
        config = replace(config, residential_min_area_m2=config.min_area_m2)

    stats: dict[str, Any] = {
        "elementos": 0,
        "geometria_invalida": 0,
        "fora_da_aoi": 0,
        "fora_do_alvo": 0,
        "descartes_por_motivo": {},
        "aprovados": 0,
        "duplicados_removidos": 0,
        "segmentos": {},
        "alvos": sorted(alvos) or ["todos"],
    }

    leads: list[RoofLead] = []
    for element in elements:
        stats["elementos"] += 1
        tags = element.get("tags") or {}

        geom = normalize(element_to_geometry(element))
        if geom is None:
            stats["geometria_invalida"] += 1
            continue

        if aoi is not None and not geom.representative_point().within(aoi):
            stats["fora_da_aoi"] += 1
            continue

        # O uso do solo é consultado antes do filtro de alvo, e não depois:
        # é justamente ele que decide o segmento dos "building=yes".
        uso_solo = contexto_uso_solo.uso_em(geom) if contexto_uso_solo else None
        segmento = segments.classificar(tags, uso_solo)

        if alvos and segmento.chave not in alvos:
            stats["fora_do_alvo"] += 1
            continue

        geom_utm, epsg = project_to_utm(geom)
        metrics = compute_metrics(geom_utm)
        if metrics is None:
            stats["geometria_invalida"] += 1
            continue

        motivos = rejection_reasons(metrics, tags, config)
        if motivos:
            # Registra apenas o primeiro motivo: é o que efetivamente decidiu.
            chave = motivos[0]
            stats["descartes_por_motivo"][chave] = stats["descartes_por_motivo"].get(chave, 0) + 1
            continue

        score, breakdown = score_roof(metrics, tags, config, uso_solo)
        stats["segmentos"][segmento.chave] = stats["segmentos"].get(segmento.chave, 0) + 1
        centroid = geom.centroid
        leads.append(
            RoofLead(
                osm_type=str(element.get("type", "way")),
                osm_id=int(element.get("id", 0)),
                geometry=geom,
                geometry_utm=geom_utm,
                utm_epsg=epsg,
                metrics=metrics,
                tags=dict(tags),
                building_type=building_type_of(tags),
                centroid_lat=float(centroid.y),
                centroid_lon=float(centroid.x),
                score=score,
                score_breakdown=breakdown,
                segmento=segmento.chave,
                landuse_contexto=uso_solo,
            )
        )

    stats["aprovados"] = len(leads)
    leads, removidos = deduplicate(leads, config.dedup_iou)
    stats["duplicados_removidos"] = removidos

    leads.sort(key=lambda lead: (lead.score, lead.metrics.area_m2), reverse=True)
    stats["leads_finais"] = len(leads)
    return leads, stats


def rank_by_area(leads: Sequence[RoofLead]) -> list[RoofLead]:
    """Ordena estritamente pelos maiores telhados, ignorando a pontuação."""
    return sorted(leads, key=lambda lead: lead.metrics.area_m2, reverse=True)
