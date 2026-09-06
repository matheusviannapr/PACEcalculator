"""Fixtures compartilhadas. Nenhum teste aqui toca a rede."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from shapely.geometry import Polygon

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from aurum.geo.roofs import RoofFilterConfig, build_leads  # noqa: E402
from aurum.pv.equipment import carregar_base  # noqa: E402
from aurum.pv.solar import PerfilGeracao  # noqa: E402


def retangulo_osm(lon0: float, lat0: float, dlon: float, dlat: float) -> list[dict]:
    """Anel retangular no formato que o Overpass devolve."""
    return [
        {"lon": lon0, "lat": lat0},
        {"lon": lon0 + dlon, "lat": lat0},
        {"lon": lon0 + dlon, "lat": lat0 + dlat},
        {"lon": lon0, "lat": lat0 + dlat},
        {"lon": lon0, "lat": lat0},
    ]


@pytest.fixture(scope="session")
def base_equipamentos():
    return carregar_base()


@pytest.fixture
def galpao_utm() -> Polygon:
    """Galpão retangular de 100 x 50 m, já em coordenadas métricas."""
    return Polygon([(0, 0), (100, 0), (100, 50), (0, 50)])


@pytest.fixture
def telhado_em_l_utm() -> Polygon:
    """Telhado em L com 4.000 m², para comparar contra o retângulo."""
    return Polygon([(0, 0), (100, 0), (100, 25), (40, 25), (40, 62.5), (0, 62.5)])


@pytest.fixture
def elementos_osm() -> list[dict]:
    """Conjunto de elementos cobrindo os casos que o filtro precisa tratar."""
    return [
        {"type": "way", "id": 1,
         "tags": {"building": "warehouse", "name": "Distribuidora Alfa",
                  "website": "https://alfa.com.br", "phone": "+55 11 3000-0000",
                  "addr:street": "Rua Industrial", "building:levels": "1"},
         "geometry": retangulo_osm(-46.6400, -23.5500, 0.0012, 0.0006)},
        {"type": "way", "id": 2, "tags": {"building": "supermarket", "name": "Mercado Beta"},
         "geometry": retangulo_osm(-46.6300, -23.5500, 0.0006, 0.0004)},
        {"type": "way", "id": 3, "tags": {"building": "house"},
         "geometry": retangulo_osm(-46.6200, -23.5500, 0.00010, 0.00008)},
        {"type": "way", "id": 4, "tags": {"building": "shed"},
         "geometry": retangulo_osm(-46.6100, -23.5500, 0.0010, 0.0008)},
        {"type": "way", "id": 5, "tags": {"building": "industrial", "generator:source": "solar"},
         "geometry": retangulo_osm(-46.6000, -23.5500, 0.0010, 0.0008)},
        {"type": "way", "id": 6, "tags": {"building": "warehouse"},
         "geometry": retangulo_osm(-46.64002, -23.55002, 0.0012, 0.0006)},
        {"type": "way", "id": 7, "tags": {"building": "office"},
         "geometry": [{"lon": -46.5, "lat": -23.5}, {"lon": -46.5, "lat": -23.5}]},
        {"type": "relation", "id": 8, "tags": {"building": "commercial", "name": "Shopping Gama"},
         "members": [
             {"type": "way", "role": "outer", "geometry": retangulo_osm(-46.5900, -23.5600, 0.0020, 0.0012)},
             {"type": "way", "role": "inner", "geometry": retangulo_osm(-46.5895, -23.5597, 0.0004, 0.0002)},
         ]},
    ]


@pytest.fixture
def leads(elementos_osm):
    resultado, _ = build_leads(elementos_osm, RoofFilterConfig())
    return resultado


@pytest.fixture
def perfil_curitiba() -> PerfilGeracao:
    """Perfil fixo, para que os testes não dependam de rede."""
    mensal = (114.4, 108.2, 113.8, 103.4, 94.5, 86.8, 101.1, 106.1, 98.5, 98.7, 110.9, 116.3)
    return PerfilGeracao(
        latitude=-25.49,
        longitude=-49.33,
        azimuth_deg=0.0,
        tilt_deg=25.5,
        losses_percent=14.0,
        anual_kwh_por_kwp=sum(mensal),
        mensal_kwh_por_kwp=mensal,
        fonte="pvgis",
        base_dados="PVGIS-SARAH3",
    )


@pytest.fixture(scope="module")
def base_equipamentos_modulo():
    """Igual a base_equipamentos, com escopo compatível com fixtures de módulo."""
    return carregar_base()


@pytest.fixture(scope="module")
def perfil_curitiba_modulo() -> PerfilGeracao:
    mensal = (114.4, 108.2, 113.8, 103.4, 94.5, 86.8, 101.1, 106.1, 98.5, 98.7, 110.9, 116.3)
    return PerfilGeracao(
        latitude=-25.49,
        longitude=-49.33,
        azimuth_deg=0.0,
        tilt_deg=25.5,
        losses_percent=14.0,
        anual_kwh_por_kwp=sum(mensal),
        mensal_kwh_por_kwp=mensal,
        fonte="pvgis",
        base_dados="PVGIS-SARAH3",
    )


# ----------------------------------------------------------------------
# Opções e marcadores
# ----------------------------------------------------------------------
def pytest_addoption(parser):
    parser.addoption(
        "--rede", action="store_true", default=False,
        help="Executa também os testes que acessam OpenStreetMap e PVGIS",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "rede: exige acesso à internet")


def construir_contexto(base, perfil):
    """
    Monta um ContextoProposta completo sem tocar a rede.

    Vive no conftest para que os testes de proposta e os de integração
    compartilhem exatamente o mesmo objeto de referência.
    """
    from aurum.geo.roofs import RoofFilterConfig, build_leads
    from aurum.proposal.context import ContextoProposta, DadosEmissor
    from aurum.pv import consumption, financials
    from aurum.pv.layout import calcular_layout
    from aurum.pv.sizing import melhor_sistema

    elementos = [{
        "type": "way", "id": 42,
        "tags": {"building": "warehouse", "name": "Galpão de teste", "building:levels": "1"},
        "geometry": retangulo_osm(-49.3300, -25.4900, 0.0010, 0.0006),
    }]
    construidos, _ = build_leads(elementos, RoofFilterConfig())
    lead = construidos[0]
    lead.company = {
        "nome": "Indústria Ômega & Filhos Ltda",
        "telefone": "+55 41 3000-1000",
        "site": "https://omega.ind.br?ref=a&b=1",
        "endereco": "Rua 100% Nova, 50 - Curitiba/PR",
        "fonte": "osm_poi_interno",
    }

    quantidades, layouts = {}, {}
    for modulo in base.modulos:
        layout = calcular_layout(
            lead.geometry_utm, modulo, montagem="coplanar", tilt_deg=8.0,
            utm_epsg=lead.utm_epsg, fator_obstaculos=0.85,
        )
        quantidades[modulo.modelo] = layout.quantidade
        layouts[modulo.modelo] = layout

    sistema = melhor_sistema(quantidades, base)
    assert sistema is not None
    sistema.geracao_anual_kwh = perfil.anual(sistema.potencia_cc_kwp)
    sistema.geracao_mensal_kwh = perfil.mensal(sistema.potencia_cc_kwp)

    estimativa = consumption.estimar_de_lead(lead)
    tarifa, _grupo = consumption.tarifa_referencia(lead.building_type, estimativa.consumo_mensal_kwh)
    economia = financials.calcular(
        sistema.geracao_anual_kwh,
        financials.PremissasEconomicas(
            tarifa_brl_kwh=tarifa,
            capex_brl=financials.estimar_capex(sistema.potencia_cc_kwp),
            fracao_autoconsumo=estimativa.fracao_autoconsumo,
            consumo_anual_kwh=estimativa.consumo_anual_kwh,
        ),
    )
    return ContextoProposta(
        lead=lead,
        perfil_solar=perfil,
        layout=layouts[sistema.modulo.modelo],
        sistema=sistema,
        consumo=estimativa,
        economia=economia,
        emissor=DadosEmissor(empresa="PACE Inteligência Energética", responsavel_tecnico="Fulano & Cia"),
    )
