"""Geometria, identificação de telhados e enriquecimento comercial."""
from __future__ import annotations

import math

import pytest
from shapely.geometry import Point, Polygon

from aurum.geo.enrich import enriquecer_com_pois, normalizar_site, normalizar_telefone, perfil_de_tags
from aurum.geo.geometry import (
    bbox_area_km2,
    compute_metrics,
    intersection_over_union,
    normalize,
    project_to_utm,
    project_to_wgs84,
    tile_bbox,
    utm_epsg_for,
)
from aurum.geo.nominatim import parse_bbox
from aurum.geo.overpass import has_explicit_solar
from aurum.geo.roofs import RoofFilterConfig, build_leads, deduplicate, score_roof


# ----------------------------------------------------------------------
# Geometria
# ----------------------------------------------------------------------
def test_zona_utm_por_hemisferio():
    assert utm_epsg_for(-49.33, -25.49) == 32722   # sul
    assert utm_epsg_for(-49.33, 25.49) == 32622    # norte
    assert utm_epsg_for(-46.63, -23.55) == 32723


def test_projecao_ida_e_volta_preserva_coordenadas():
    poligono = Polygon([(-46.633, -23.550), (-46.632, -23.550), (-46.632, -23.551)])
    projetado, epsg = project_to_utm(poligono)
    de_volta = project_to_wgs84(projetado, epsg)
    for original, retornado in zip(poligono.bounds, de_volta.bounds):
        assert original == pytest.approx(retornado, abs=1e-7)


def test_area_metrica_de_galpao_conhecido():
    """Retângulo de ~60 x 30 m deve medir ~1.800 m², não graus quadrados."""
    poligono = Polygon([
        (-46.63330, -23.55050), (-46.63271, -23.55050),
        (-46.63271, -23.55077), (-46.63330, -23.55077),
    ])
    metricas = compute_metrics(project_to_utm(poligono)[0])
    assert metricas.area_m2 == pytest.approx(1800, rel=0.02)
    assert metricas.rectangularity == pytest.approx(1.0, abs=0.01)
    assert metricas.aspect_ratio == pytest.approx(2.0, rel=0.05)


def test_azimute_da_cumeeira_aponta_para_o_lado_mais_longo():
    """Galpão alongado no sentido leste-oeste tem cumeeira em ~90° ou ~270°."""
    poligono = Polygon([
        (-46.63330, -23.55050), (-46.63271, -23.55050),
        (-46.63271, -23.55077), (-46.63330, -23.55077),
    ])
    azimute = compute_metrics(project_to_utm(poligono)[0]).ridge_azimuth_deg
    assert azimute is not None
    assert min(abs(azimute - 90), abs(azimute - 270)) < 6


def test_metricas_de_geometria_degenerada_nao_explodem():
    degenerado = Polygon([(0, 0), (1, 0), (2, 0), (0, 0)])
    assert compute_metrics(degenerado) is None
    assert normalize(None) is None


def test_normalize_repara_autointersecao():
    gravata = Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])
    assert not gravata.is_valid
    reparado = normalize(gravata)
    assert reparado is not None and reparado.is_valid


def test_iou():
    a = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    assert intersection_over_union(a, a) == pytest.approx(1.0)
    b = Polygon([(10, 10), (11, 10), (11, 11)])
    assert intersection_over_union(a, b) == 0.0


def test_tiles_cobrem_o_bbox_inteiro():
    bbox = (-46.70, -23.60, -46.60, -23.50)
    tiles = tile_bbox(bbox, 0.02)
    assert len(tiles) == 25
    assert min(t[0] for t in tiles) == pytest.approx(bbox[0])
    assert max(t[2] for t in tiles) == pytest.approx(bbox[2])
    assert max(t[3] for t in tiles) == pytest.approx(bbox[3])


def test_tile_size_invalido():
    with pytest.raises(ValueError):
        tile_bbox((-1, -1, 1, 1), 0)


def test_area_de_bbox_em_km2():
    # ~0,1° x 0,1° na latitude de São Paulo
    assert bbox_area_km2((-46.70, -23.60, -46.60, -23.50)) == pytest.approx(113, rel=0.05)


# ----------------------------------------------------------------------
# Parsing de bbox
# ----------------------------------------------------------------------
def test_parse_bbox_normaliza_cantos_invertidos():
    assert parse_bbox("-46.60,-23.50,-46.70,-23.60") == (-46.70, -23.60, -46.60, -23.50)


@pytest.mark.parametrize("texto", ["1,2,3", "a,b,c,d", "-46.6,-23.5,-46.6,-23.4", "-200,0,-190,10"])
def test_parse_bbox_rejeita_entradas_invalidas(texto):
    with pytest.raises(ValueError):
        parse_bbox(texto)


# ----------------------------------------------------------------------
# Filtro e pontuação
# ----------------------------------------------------------------------
def test_pipeline_de_leads_aplica_todos_os_filtros(elementos_osm):
    leads, stats = build_leads(elementos_osm, RoofFilterConfig())
    identificadores = {lead.osm_id for lead in leads}

    assert 1 in identificadores, "galpão grande deve passar"
    assert 8 in identificadores, "relação multipolígono deve ser lida"
    assert 3 not in identificadores, "casa pequena cai por área"
    assert 4 not in identificadores, "barracão é tipo excluído"
    assert 5 not in identificadores, "telhado com solar já instalado é excluído"
    assert 6 not in identificadores, "duplicata deve ser deduplicada"
    assert stats["duplicados_removidos"] == 1
    assert stats["geometria_invalida"] == 1


def test_relacao_com_patio_desconta_a_area_do_vazio(elementos_osm):
    leads, _ = build_leads(elementos_osm, RoofFilterConfig())
    shopping = next(lead for lead in leads if lead.osm_id == 8)
    # 0,0020 x 0,0012 grau menos o pátio interno
    assert 25_000 < shopping.metrics.area_m2 < 27_000


def test_filtro_por_segmento(elementos_osm):
    industriais, stats = build_leads(elementos_osm, RoofFilterConfig(), target="industrial")
    assert all(lead.building_type == "warehouse" for lead in industriais)
    assert stats["fora_do_alvo"] > 0


def test_deduplicacao_mantem_o_de_maior_score(leads):
    duplicado = leads[0]
    copia = type(duplicado)(**{**duplicado.__dict__, "osm_id": 999, "score": 1.0})
    mantidos, removidos = deduplicate([duplicado, copia], iou_threshold=0.8)
    assert removidos == 1
    assert mantidos[0].osm_id == duplicado.osm_id


def test_score_cresce_com_a_area(leads):
    grande = max(leads, key=lambda l: l.metrics.area_m2)
    pequeno = min(leads, key=lambda l: l.metrics.area_m2)
    assert grande.score_breakdown["area"] > pequeno.score_breakdown["area"]


def test_score_premia_dados_cadastrais(leads):
    """O lead com nome, site e telefone deve pontuar o máximo em dados."""
    com_contato = next(l for l in leads if l.osm_id == 1)
    assert com_contato.score_breakdown["dados"] == 10.0


def test_score_fica_entre_0_e_100(leads):
    assert all(0 <= lead.score <= 100 for lead in leads)


def test_pavimentos_lidos_das_tags(leads):
    assert next(l for l in leads if l.osm_id == 1).levels == 1


@pytest.mark.parametrize("tags,esperado", [
    ({"generator:source": "solar"}, True),
    ({"generator:method": "photovoltaic"}, True),
    ({"rooftop:solar": "qualquer coisa"}, True),
    ({"solar_panel": "yes"}, True),
    ({"building": "industrial"}, False),
    ({"power": "generator"}, False),
])
def test_deteccao_de_solar_existente(tags, esperado):
    assert has_explicit_solar(tags) is esperado


# ----------------------------------------------------------------------
# Enriquecimento
# ----------------------------------------------------------------------
@pytest.mark.parametrize("entrada,esperado", [
    ("1130000000", "+55 11 3000-0000"),
    ("11930000000", "+55 11 93000-0000"),
    ("5511930000000", "+55 11 93000-0000"),
    ("(11) 3000 0000;(11) 99999-9999", "+55 11 3000-0000"),
    (None, None),
])
def test_normalizacao_de_telefone(entrada, esperado):
    assert normalizar_telefone(entrada) == esperado


def test_telefone_estrangeiro_e_preservado():
    """Sem padrão brasileiro reconhecível, devolve o original em vez de corromper."""
    assert normalizar_telefone("+1 555 0100") == "+1 555 0100"


@pytest.mark.parametrize("entrada,esperado", [
    ("alfa.com.br", "https://alfa.com.br"),
    ("https://beta.com", "https://beta.com"),
    ("  gama.com/x  ", "https://gama.com/x"),
    (None, None),
])
def test_normalizacao_de_site(entrada, esperado):
    assert normalizar_site(entrada) == esperado


def test_perfil_monta_endereco_a_partir_das_tags():
    perfil = perfil_de_tags({
        "name": "Frigorífico Delta", "contact:phone": "(11) 4000-1234",
        "website": "delta.ind.br", "addr:street": "Av. das Indústrias",
        "addr:housenumber": "1500", "addr:city": "Guarulhos", "addr:state": "SP",
    }, "teste")
    assert perfil["telefone"] == "+55 11 4000-1234"
    assert perfil["site"] == "https://delta.ind.br"
    assert "Av. das Indústrias, 1500" in perfil["endereco"]
    assert perfil["fonte"] == "teste"


def test_poi_dentro_do_telhado_preenche_a_empresa(leads):
    alvo = next(lead for lead in leads if lead.osm_id == 2)
    centro = alvo.geometry.centroid
    pois = [{
        "type": "node", "lat": centro.y, "lon": centro.x,
        "tags": {"name": "Rede Beta S.A.", "shop": "supermarket", "phone": "1140001111"},
    }]
    enriquecer_com_pois([alvo], pois)
    assert alvo.company["nome"] == "Rede Beta S.A."
    assert alvo.company["telefone"] == "+55 11 4000-1111"
    assert alvo.company["fonte"] == "osm_poi_interno"


def test_poi_proximo_e_marcado_como_tal(leads):
    """POI fora do polígono entra como fallback, com a fonte declarada."""
    alvo = next(lead for lead in leads if lead.osm_id == 2)
    alvo.company = {}
    minx, miny, maxx, maxy = alvo.geometry.bounds
    # ~10 m além da borda oeste
    pois = [{"type": "node", "lat": (miny + maxy) / 2, "lon": minx - 0.0001,
             "tags": {"name": "Loja da Esquina"}}]
    enriquecer_com_pois([alvo], pois)
    assert alvo.company["nome"] == "Loja da Esquina"
    assert alvo.company["fonte"] == "osm_poi_proximo"


def test_poi_distante_nao_contamina(leads):
    alvo = next(lead for lead in leads if lead.osm_id == 2)
    alvo.company = {}
    pois = [{"type": "node", "lat": -1.0, "lon": -50.0, "tags": {"name": "Irrelevante"}}]
    enriquecer_com_pois([alvo], pois)
    assert alvo.company == {}


# ----------------------------------------------------------------------
# Resiliência de rede
# ----------------------------------------------------------------------
def test_overpass_fora_do_ar_da_mensagem_clara():
    """
    Se nenhum tile responde, o problema é o serviço e não a região. Sem essa
    distinção o usuário receberia "nenhum telhado encontrado" e procuraria o
    erro nos filtros.
    """
    from unittest.mock import patch

    from aurum.geo.overpass import OverpassClient, OverpassError

    cliente = OverpassClient()
    with patch.object(cliente, "run_query", side_effect=OverpassError("servidor fora do ar")):
        with pytest.raises(OverpassError, match="indisponível ou sobrecarregado"):
            cliente.fetch_buildings((-49.35, -25.51, -49.33, -25.49))


def test_falha_parcial_de_tiles_nao_aborta_a_varredura():
    """Perder um tile é melhor que perder a prospecção inteira."""
    from unittest.mock import patch

    from aurum.geo.overpass import OverpassClient, OverpassError

    cliente = OverpassClient()
    chamadas = {"n": 0}

    def responder(_query, **_kwargs):
        chamadas["n"] += 1
        if chamadas["n"] % 2:
            raise OverpassError("tile ruim")
        return {"elements": [{"type": "way", "id": chamadas["n"], "tags": {}, "geometry": []}]}

    with patch.object(cliente, "run_query", side_effect=responder):
        elementos, stats = cliente.fetch_buildings((-49.36, -25.52, -49.32, -25.48))

    assert stats["tiles_falhos"] > 0
    assert stats["tiles_ok"] > 0
    assert elementos


def test_timeout_de_conexao_e_menor_que_o_de_leitura():
    """
    Host morto deve ceder a vez em segundos; consulta em execução pode levar
    minutos. Sem essa separação a varredura ficava travada sem sinal de vida.
    """
    from aurum.config import OverpassSettings

    config = OverpassSettings()
    assert config.connect_timeout_s < config.timeout_s
    assert config.connect_timeout_s <= 15


def test_espelho_que_falhou_fica_de_castigo():
    """
    Numa varredura de dezenas de tiles, recomeçar sempre pelo primeiro
    endereço faz pagar o timeout de um espelho fora do ar a cada tile.
    Quem falha sai da rotação por um tempo; quem responde volta a ela.
    """
    from unittest.mock import MagicMock, patch

    from aurum.geo.overpass import OverpassClient

    cliente = OverpassClient()
    cliente.config.min_interval_s = 0.0   # o teste não precisa esperar
    tentados: list[str] = []

    def responder(endpoint, **_kwargs):
        tentados.append(endpoint)
        resposta = MagicMock(status_code=200, ok=True)
        # apenas o segundo espelho responde
        if endpoint != cliente.config.endpoints[1]:
            resposta.status_code = 502
            resposta.ok = False
            return resposta
        resposta.json.return_value = {
            "osm3s": {"timestamp_osm_base": "2026-08-24T01:29:50Z"},
            "elements": [],
        }
        return resposta

    with patch.object(cliente._session, "post", side_effect=responder), \
            patch("aurum.geo.overpass.time.sleep"):
        cliente.run_query("consulta 1", use_cache=False)
        primeira_rodada = len(tentados)
        cliente.run_query("consulta 2", use_cache=False)

    # o espelho bom não está de castigo; o ruim está
    assert 1 not in cliente._castigo
    assert 0 in cliente._castigo
    # a segunda consulta acerta de primeira, sem repetir o espelho ruim
    assert len(tentados) - primeira_rodada == 1
    assert tentados[-1] == cliente.config.endpoints[1]


def test_espelho_dessincronizado_e_recusado():
    """
    Um espelho com o banco quebrado responde HTTP 200, envelope bem formado e
    zero elementos -- indistinguível de área vazia. O carimbo da base denuncia:
    numa instância sadia é ISO-8601; na degradada observada vinha "116619".

    Sem essa checagem, o resultado vazio entrava no cache e a região ficava
    "sem telhados" pelos 30 dias de TTL. Foi exatamente o que aconteceu.
    """
    from aurum.geo.overpass import _instancia_sincronizada

    sadio_vazio = {"osm3s": {"timestamp_osm_base": "2026-08-24T01:29:50Z"}, "elements": []}
    degradado = {"osm3s": {"timestamp_osm_base": "116619"}, "elements": []}
    sem_carimbo = {"elements": []}
    com_dados = {"osm3s": {"timestamp_osm_base": "116619"},
                 "elements": [{"type": "way", "id": 1}]}

    assert _instancia_sincronizada(sadio_vazio), "área realmente vazia é resposta legítima"
    assert not _instancia_sincronizada(degradado)
    assert not _instancia_sincronizada(sem_carimbo)
    assert _instancia_sincronizada(com_dados), "resposta com dados vale mesmo sem carimbo"


def test_resposta_dessincronizada_nao_entra_no_cache(tmp_path):
    from unittest.mock import MagicMock, patch

    from aurum.geo.overpass import OverpassClient, OverpassError

    cliente = OverpassClient()
    cliente.cache.directory = tmp_path
    resposta = MagicMock(status_code=200, ok=True)
    resposta.json.return_value = {"osm3s": {"timestamp_osm_base": "116619"}, "elements": []}

    with patch.object(cliente._session, "post", return_value=resposta),             patch("aurum.geo.overpass.time.sleep"):
        with pytest.raises(OverpassError, match="dessincronizada|falhou"):
            cliente.run_query("consulta qualquer")

    assert not list(tmp_path.glob("*.json")), "resposta suspeita não pode ser cacheada"
