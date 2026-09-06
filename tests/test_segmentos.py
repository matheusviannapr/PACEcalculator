"""
Filtro por tipo de local.

Substitui os quatro grupos grossos anteriores por uma taxonomia que
corresponde a como a prospecção é realmente organizada: agro é segmento
próprio, escola não é hospital, condomínio não é galpão.
"""
from __future__ import annotations

import pytest

from aurum.geo import segments
from aurum.geo.roofs import ContextoUsoSolo, RoofFilterConfig, build_leads, matches_target
from aurum.pv.consumption import faixa_eui, fracao_autoconsumo
from aurum.pv.layout import fator_obstaculos_para
from tests.conftest import retangulo_osm


# ----------------------------------------------------------------------
# Classificação
# ----------------------------------------------------------------------
@pytest.mark.parametrize("tags,esperado", [
    # agro pelas tags de edificação rural
    ({"building": "barn"}, "agro"),
    ({"building": "greenhouse"}, "agro"),
    ({"building": "chicken_coop"}, "agro"),
    ({"man_made": "silo"}, "agro"),
    ({"building": "yes", "landuse": "farmyard"}, "agro"),
    ({"building": "industrial", "industrial": "slaughterhouse"}, "agro"),
    # educação
    ({"building": "school"}, "educacao"),
    ({"amenity": "university"}, "educacao"),
    ({"building": "kindergarten"}, "educacao"),
    # condomínio
    ({"building": "apartments"}, "condominio"),
    ({"building": "residential"}, "condominio"),
    # demais
    ({"building": "hospital"}, "saude"),
    ({"amenity": "clinic"}, "saude"),
    ({"building": "supermarket"}, "supermercado"),
    ({"shop": "wholesale"}, "supermercado"),
    ({"building": "warehouse"}, "logistica"),
    ({"building": "factory"}, "industrial"),
    ({"building": "church"}, "religioso"),
    ({"building": "hotel"}, "hotelaria"),
    ({"building": "office"}, "escritorio"),
    ({"leisure": "stadium"}, "esporte_lazer"),
    ({"amenity": "townhall"}, "publico"),
    # sem informação
    ({"building": "yes"}, "desconhecido"),
    ({}, "desconhecido"),
])
def test_classificacao_por_tags(tags, esperado):
    assert segments.classificar(tags).chave == esperado


def test_agro_vence_logistica_quando_ha_contexto_rural():
    """
    Um galpão dentro de um pátio de fazenda é agro, não centro de distribuição.
    A ordem dos segmentos é que garante isso.
    """
    assert segments.classificar({"building": "warehouse", "landuse": "farmyard"}).chave == "agro"


def test_contexto_de_uso_do_solo_classifica_o_building_yes():
    """
    No mapeamento brasileiro a maioria é ``building=yes``. Sem olhar a área que
    contém o prédio, o filtro por segmento devolveria quase nada.
    """
    sem_contexto = segments.classificar({"building": "yes"})
    com_fazenda = segments.classificar({"building": "yes"}, landuse_contexto="farmyard")
    com_industria = segments.classificar({"building": "yes"}, landuse_contexto="industrial")

    assert sem_contexto.chave == "desconhecido"
    assert com_fazenda.chave == "agro"
    assert com_industria.chave == "industrial"


def test_tag_propria_vence_o_contexto():
    """Uma escola dentro de área industrial continua sendo escola."""
    assert segments.classificar({"building": "school"}, landuse_contexto="industrial").chave == "educacao"


# ----------------------------------------------------------------------
# Normalização do parâmetro de filtro
# ----------------------------------------------------------------------
@pytest.mark.parametrize("entrada,esperado", [
    ("todos", set()),
    ("", set()),
    (None, set()),
    ("agro", {"agro"}),
    ("agro,educacao", {"agro", "educacao"}),
    (["agro", "saude"], {"agro", "saude"}),
    ("  Agro , EDUCACAO ", {"agro", "educacao"}),
    ("inexistente", set()),
])
def test_normalizacao_de_alvos(entrada, esperado):
    assert segments.normalizar_alvos(entrada) == esperado


def test_nomes_antigos_continuam_funcionando():
    """Scripts e comandos que usavam os grupos anteriores não podem quebrar."""
    assert segments.normalizar_alvos("misto") == set()
    comercial = segments.normalizar_alvos("comercial")
    assert "comercio" in comercial and "supermercado" in comercial
    institucional = segments.normalizar_alvos("institucional")
    assert {"educacao", "saude", "publico"} <= institucional


def test_matches_target_aceita_lista_e_string():
    tags = {"building": "barn"}
    assert matches_target(tags, "agro")
    assert matches_target(tags, ["agro", "educacao"])
    assert matches_target(tags, "todos")
    assert not matches_target(tags, "educacao")


# ----------------------------------------------------------------------
# Filtro de ponta a ponta
# ----------------------------------------------------------------------
@pytest.fixture
def elementos_variados():
    """Uma edificação grande de cada segmento que o teste precisa."""
    def bloco(indice, tags):
        return {
            "type": "way", "id": 1000 + indice, "tags": tags,
            "geometry": retangulo_osm(-49.33 + indice * 0.003, -25.49, 0.0010, 0.0006),
        }

    return [
        bloco(0, {"building": "barn", "name": "Granja Boa Vista"}),
        bloco(1, {"building": "school", "name": "Escola Municipal"}),
        bloco(2, {"building": "apartments", "name": "Condomínio Jardins"}),
        bloco(3, {"building": "warehouse", "name": "Transportadora"}),
        bloco(4, {"building": "hospital", "name": "Hospital Central"}),
        bloco(5, {"building": "yes", "name": "Sem tipo"}),
    ]


def test_filtro_por_agro(elementos_variados):
    leads, stats = build_leads(elementos_variados, RoofFilterConfig(), target="agro")
    assert [l.name for l in leads] == ["Granja Boa Vista"]
    assert leads[0].segmento == "agro"
    assert leads[0].segmento_rotulo == "Agro e agroindústria"
    assert stats["fora_do_alvo"] == 5


def test_filtro_por_escola(elementos_variados):
    leads, _ = build_leads(elementos_variados, RoofFilterConfig(), target="educacao")
    assert [l.name for l in leads] == ["Escola Municipal"]


def test_filtro_por_condominio_nao_e_barrado_pelo_piso_residencial(elementos_variados):
    """
    O piso de área residencial existe para descartar casa isolada. Quem pede
    condomínio quer prédio residencial, e manter o piso esvaziaria o alvo.
    """
    leads, _ = build_leads(
        elementos_variados, RoofFilterConfig(min_area_m2=300), target="condominio"
    )
    assert [l.name for l in leads] == ["Condomínio Jardins"]


def test_condominio_sem_o_filtro_cai_pelo_piso_residencial(elementos_variados):
    """Sem pedir condomínio, o prédio residencial pequeno continua sendo descartado."""
    config = RoofFilterConfig(min_area_m2=300, residential_min_area_m2=1_000_000)
    leads, _ = build_leads(elementos_variados, config, target="todos")
    assert "Condomínio Jardins" not in [l.name for l in leads]


def test_varios_segmentos_de_uma_vez(elementos_variados):
    leads, _ = build_leads(elementos_variados, RoofFilterConfig(), target=["agro", "saude"])
    assert sorted(l.name for l in leads) == ["Granja Boa Vista", "Hospital Central"]


def test_sem_filtro_traz_todos(elementos_variados):
    leads, stats = build_leads(elementos_variados, RoofFilterConfig(), target="todos")
    assert len(leads) == len(elementos_variados)
    assert stats["fora_do_alvo"] == 0
    assert set(stats["segmentos"]) == {
        "agro", "educacao", "condominio", "logistica", "saude", "desconhecido"
    }


def test_contexto_de_uso_do_solo_muda_o_resultado_do_filtro():
    """Galpão sem tag dentro de fazenda passa a aparecer no filtro de agro."""
    galpao = {
        "type": "way", "id": 77, "tags": {"building": "yes", "name": "Barracão"},
        "geometry": retangulo_osm(-49.33, -25.49, 0.0010, 0.0006),
    }
    fazenda = {
        "type": "way", "id": 78, "tags": {"landuse": "farmyard"},
        "geometry": retangulo_osm(-49.335, -25.495, 0.0100, 0.0100),
    }

    sem, _ = build_leads([galpao], RoofFilterConfig(), target="agro")
    com, _ = build_leads(
        [galpao], RoofFilterConfig(), target="agro",
        contexto_uso_solo=ContextoUsoSolo([fazenda]),
    )

    assert sem == []
    assert [l.name for l in com] == ["Barracão"]
    assert com[0].landuse_contexto == "farmyard"


def test_contexto_escolhe_a_menor_area_que_contem():
    """
    As áreas se aninham: o pátio da fazenda descreve melhor o prédio do que a
    lavoura inteira que o cerca.
    """
    contexto = ContextoUsoSolo([
        {"type": "way", "id": 1, "tags": {"landuse": "farmland"},
         "geometry": retangulo_osm(-49.40, -25.55, 0.2000, 0.2000)},
        {"type": "way", "id": 2, "tags": {"landuse": "farmyard"},
         "geometry": retangulo_osm(-49.335, -25.495, 0.0100, 0.0100)},
    ])
    from shapely.geometry import Point

    assert contexto.uso_em(Point(-49.330, -25.490).buffer(0.0001)) == "farmyard"
    assert contexto.uso_em(Point(-49.250, -25.400).buffer(0.0001)) == "farmland"
    assert contexto.uso_em(Point(0.0, 0.0).buffer(0.0001)) is None


def test_contexto_vazio_nao_quebra():
    contexto = ContextoUsoSolo([])
    from shapely.geometry import Point

    assert len(contexto) == 0
    assert contexto.uso_em(Point(-49.33, -25.49)) is None


# ----------------------------------------------------------------------
# Reflexo na pontuação e no dimensionamento
# ----------------------------------------------------------------------
def test_peso_do_segmento_entra_na_pontuacao(elementos_variados):
    leads, _ = build_leads(elementos_variados, RoofFilterConfig(), target="todos")
    por_nome = {l.name: l for l in leads}
    assert por_nome["Granja Boa Vista"].score_breakdown["uso"] == 20.0
    assert por_nome["Condomínio Jardins"].score_breakdown["uso"] == 11.0
    assert por_nome["Sem tipo"].score_breakdown["uso"] == 9.0


def test_casa_isolada_nao_entra_no_filtro_de_condominio():
    """Casa não tem área comum nem síndico; não é o mesmo negócio."""
    casa = {
        "type": "way", "id": 90, "tags": {"building": "house"},
        "geometry": retangulo_osm(-49.33, -25.49, 0.0012, 0.0008),
    }
    assert segments.classificar({"building": "house"}).chave != "condominio"
    leads, _ = build_leads([casa], RoofFilterConfig(min_area_m2=300), target="condominio")
    assert leads == []


def test_casa_isolada_pontua_menos_que_predio():
    """
    Se a casa aparecer numa busca sem filtro, ainda assim não pode valer o
    mesmo que um condomínio: não há área comum nem decisor coletivo.
    """
    def bloco(indice, tags):
        return {
            "type": "way", "id": 200 + indice, "tags": tags,
            "geometry": retangulo_osm(-49.33 + indice * 0.003, -25.49, 0.0012, 0.0008),
        }

    leads, _ = build_leads(
        [bloco(0, {"building": "house"}), bloco(1, {"building": "apartments"})],
        RoofFilterConfig(min_area_m2=300, residential_min_area_m2=300),
        target="todos",
    )
    por_tipo = {l.building_type: l for l in leads}
    assert por_tipo["house"].score_breakdown["uso"] <= 5.0
    assert por_tipo["apartments"].score_breakdown["uso"] > por_tipo["house"].score_breakdown["uso"]


def test_pesos_dos_segmentos_estao_na_faixa():
    for segmento in segments.SEGMENTOS:
        assert 0 <= segmento.peso_comercial <= 20, segmento.chave


def test_agro_tem_consumo_e_obstaculos_proprios():
    """
    Sem entradas próprias, todo telhado rural cairia na faixa genérica e num
    fator de obstáculos de prédio urbano.

    A verificação é de presença nas tabelas, não de diferença de valor: um
    tipo pode legitimamente coincidir com o padrão sem deixar de ser explícito.
    """
    from aurum.pv.consumption import EUI_KWH_M2_MES, FRACAO_AUTOCONSUMO
    from aurum.pv.layout import FATOR_OBSTACULOS_PADRAO

    for tipo in ("barn", "cowshed", "greenhouse", "chicken_coop",
                 "slaughterhouse", "cold_storage", "silo", "sty"):
        assert tipo in EUI_KWH_M2_MES, f"{tipo} sem faixa de consumo"
        assert tipo in FRACAO_AUTOCONSUMO, f"{tipo} sem fração de autoconsumo"
        assert tipo in FATOR_OBSTACULOS_PADRAO, f"{tipo} sem fator de obstáculos"


def test_toda_faixa_de_consumo_tem_fracao_de_autoconsumo():
    """As duas tabelas precisam andar juntas; uma sem a outra gera silêncio."""
    from aurum.pv.consumption import EUI_KWH_M2_MES, FRACAO_AUTOCONSUMO

    faltando = sorted(set(EUI_KWH_M2_MES) - set(FRACAO_AUTOCONSUMO))
    assert not faltando, f"sem fração de autoconsumo: {faltando}"


def test_faixas_de_consumo_sao_coerentes():
    """Mínimo <= típico <= máximo, em todas as entradas."""
    from aurum.pv.consumption import EUI_KWH_M2_MES

    for tipo, (minimo, tipico, maximo) in EUI_KWH_M2_MES.items():
        assert 0 < minimo <= tipico <= maximo, tipo


def test_estufa_tem_pouco_espaco_para_modulo():
    """A cobertura de uma estufa precisa passar luz para a lavoura."""
    assert fator_obstaculos_para("greenhouse") < 0.5


def test_camara_fria_consome_muito_e_aproveita_quase_tudo():
    _minimo, tipico, _maximo = faixa_eui("cold_storage")
    assert tipico > faixa_eui("warehouse")[1] * 5
    assert fracao_autoconsumo("cold_storage") >= 0.75


# ----------------------------------------------------------------------
# Metadados para a interface
# ----------------------------------------------------------------------
def test_todo_segmento_tem_rotulo_e_descricao():
    for segmento in segments.SEGMENTOS:
        assert segmento.rotulo and segmento.descricao
        assert segmento.chave == segmento.chave.lower()
        assert " " not in segmento.chave


def test_chaves_nao_se_repetem():
    chaves = [s.chave for s in segments.SEGMENTOS]
    assert len(chaves) == len(set(chaves))


def test_rotulos_e_descricoes_cobrem_todos():
    assert set(segments.rotulos()) == set(segments.POR_CHAVE)
    assert set(segments.descricoes()) == set(segments.POR_CHAVE)


# ----------------------------------------------------------------------
# O segmento precisa chegar ao dimensionamento
# ----------------------------------------------------------------------
def test_segmento_sobrevive_ao_geojson():
    """
    Sem persistir o segmento, a proposta de um galpão de fazenda voltaria a
    ser "não identificado" na segunda etapa, com parâmetros de prédio urbano.
    """
    from aurum.geo.roofs import RoofLead

    galpao = {
        "type": "way", "id": 55, "tags": {"building": "yes"},
        "geometry": retangulo_osm(-49.33, -25.49, 0.0010, 0.0006),
    }
    fazenda = {
        "type": "way", "id": 56, "tags": {"landuse": "farmyard"},
        "geometry": retangulo_osm(-49.335, -25.495, 0.0100, 0.0100),
    }
    leads, _ = build_leads(
        [galpao], RoofFilterConfig(), contexto_uso_solo=ContextoUsoSolo([fazenda])
    )
    original = leads[0]
    assert original.segmento == "agro"

    recuperado = RoofLead.from_geojson_feature(original.to_geojson_feature())
    assert recuperado.segmento == "agro"
    assert recuperado.landuse_contexto == "farmyard"
    assert recuperado.tipo_para_calculo == "farm"


def test_tipo_para_calculo_prefere_o_tipo_do_predio():
    """Quando o prédio declara o que é, isso vence o representante do segmento."""
    from aurum.geo.roofs import RoofLead

    aviario = {
        "type": "way", "id": 57, "tags": {"building": "chicken_coop"},
        "geometry": retangulo_osm(-49.33, -25.49, 0.0010, 0.0006),
    }
    leads, _ = build_leads([aviario], RoofFilterConfig())
    assert leads[0].segmento == "agro"
    assert leads[0].tipo_para_calculo == "chicken_coop", "o específico vence o genérico"


def test_dimensionamento_usa_os_parametros_do_segmento(base_equipamentos_modulo, perfil_curitiba_modulo):
    """
    O ponto todo de classificar por segmento é que o cálculo mude. Um galpão
    de fazenda tem consumo, autoconsumo e obstáculos diferentes de um prédio
    urbano genérico.
    """
    from unittest.mock import patch

    from aurum.pipeline import OpcoesDimensionamento, dimensionar_lead

    def montar(tags, contexto=None):
        elemento = {
            "type": "way", "id": 58, "tags": tags,
            "geometry": retangulo_osm(-49.33, -25.49, 0.0010, 0.0006),
        }
        fazenda = [{
            "type": "way", "id": 59, "tags": {"landuse": contexto},
            "geometry": retangulo_osm(-49.335, -25.495, 0.0100, 0.0100),
        }] if contexto else []
        leads, _ = build_leads(
            [elemento], RoofFilterConfig(),
            contexto_uso_solo=ContextoUsoSolo(fazenda) if fazenda else None,
        )
        return leads[0]

    solar = type("SolarFalso", (), {"perfil": lambda self, *a, **k: perfil_curitiba_modulo})()

    generico = dimensionar_lead(
        montar({"building": "yes"}), base_equipamentos_modulo, solar, OpcoesDimensionamento()
    )
    rural = dimensionar_lead(
        montar({"building": "yes"}, contexto="farmyard"),
        base_equipamentos_modulo, solar, OpcoesDimensionamento(),
    )

    assert generico is not None and rural is not None
    assert rural.lead.segmento == "agro"
    assert rural.consumo.tipo_edificacao == "farm"
    assert generico.consumo.tipo_edificacao != "farm"
    # os três parâmetros que o segmento controla precisam divergir
    assert rural.consumo.eui_kwh_m2_mes != generico.consumo.eui_kwh_m2_mes
    assert rural.consumo.fracao_autoconsumo != generico.consumo.fracao_autoconsumo
    assert rural.layout.fator_obstaculos != generico.layout.fator_obstaculos


def test_todo_segmento_tem_tipo_de_referencia_conhecido():
    """
    O tipo representativo precisa existir nas tabelas, senão o segmento cai no
    padrão genérico em silêncio.
    """
    from aurum.pv.consumption import EUI_KWH_M2_MES, FRACAO_AUTOCONSUMO
    from aurum.pv.layout import FATOR_OBSTACULOS_PADRAO

    for segmento in segments.SEGMENTOS:
        tipo = segmento.tipo_referencia
        assert tipo, f"{segmento.chave} sem tipo de referência"
        assert tipo in EUI_KWH_M2_MES, f"{segmento.chave}: {tipo} sem faixa de consumo"
        assert tipo in FRACAO_AUTOCONSUMO, f"{segmento.chave}: {tipo} sem autoconsumo"
        assert tipo in FATOR_OBSTACULOS_PADRAO, f"{segmento.chave}: {tipo} sem obstáculos"
