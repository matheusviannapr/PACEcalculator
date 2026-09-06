"""
Testes da edição manual do arranjo.

O empacotamento automático é auditado pelo croqui; a edição manual é auditada
por estes testes, porque ela não tem grade que a discipline. O que se garante
aqui são as três coisas que separam um projeto de um desenho bonito:

* nenhum módulo fora do telhado, nem depois de arrastar a grade;
* nenhum módulo em cima de outro;
* a numeração é estável — o índice 12 continua sendo o módulo 12 depois de
  arrastar, senão o que o usuário desligou muda de lugar sozinho.
"""
from __future__ import annotations

import math

import pytest
from shapely.affinity import translate

from aurum.pv.edicao import (
    EdicaoLayout,
    aplicar_edicao,
    croqui_editavel,
    geometria_deslocada,
    indice_no_ponto,
    ponto_para_utm,
)
from aurum.pv.equipment import carregar_base
from aurum.pv.telhado import dimensionar_no_telhado, telhado_de_geojson

LAT, LON = -25.4284, -49.2733


def _retangulo(largura_m: float, altura_m: float) -> dict:
    dlon = largura_m / (111_320 * abs(math.cos(math.radians(LAT))))
    dlat = altura_m / 110_540
    return {
        "type": "Polygon",
        "coordinates": [[
            [LON, LAT], [LON + dlon, LAT], [LON + dlon, LAT + dlat],
            [LON, LAT + dlat], [LON, LAT],
        ]],
    }


@pytest.fixture(scope="module")
def telhado():
    return telhado_de_geojson(_retangulo(30, 20), "Telhado de teste")


@pytest.fixture(scope="module")
def layout(telhado):
    arranjo = dimensionar_no_telhado(telhado, carregar_base())
    assert arranjo is not None and arranjo.quantidade > 20, "o telhado precisa caber módulos"
    return arranjo


# ----------------------------------------------------------------------------
# Remoção e reinclusão
# ----------------------------------------------------------------------------
def test_edicao_vazia_devolve_o_arranjo_intacto(layout, telhado):
    assert aplicar_edicao(layout, telhado, EdicaoLayout()) is layout
    assert aplicar_edicao(layout, telhado, None) is layout


def test_desligar_um_modulo_tira_um_modulo(layout, telhado):
    edicao = EdicaoLayout(desativados={0, 5, 9})
    editado = aplicar_edicao(layout, telhado, edicao)
    assert editado.quantidade == layout.quantidade - 3
    assert editado.potencia_kwp < layout.potencia_kwp
    # E o empacotamento automático continua registrado, para a tela poder
    # dizer quanto a edição custou.
    assert editado.quantidade_geometrica == layout.quantidade_geometrica


def test_alternar_duas_vezes_devolve_o_modulo(layout, telhado):
    edicao = EdicaoLayout()
    edicao.alternar(3)
    assert aplicar_edicao(layout, telhado, edicao).quantidade == layout.quantidade - 1
    edicao.alternar(3)
    assert edicao.vazia
    assert aplicar_edicao(layout, telhado, edicao).quantidade == layout.quantidade


# ----------------------------------------------------------------------------
# Acréscimo manual
# ----------------------------------------------------------------------------
def test_modulo_a_mao_sobre_outro_modulo_e_recusado(layout, telhado):
    """
    Sobreposição some do papel e reaparece na obra.

    O centro do primeiro módulo é, por definição, um lugar ocupado: pôr outro
    ali tem de ser recusado, e o arranjo tem de continuar com a mesma
    quantidade.
    """
    centro = layout.modulos_geom[0].centroid
    edicao = EdicaoLayout(extras_utm=[(centro.x, centro.y)])
    editado = aplicar_edicao(layout, telhado, edicao)
    assert editado.quantidade == layout.quantidade
    assert any("não couberam" in a for a in editado.avisos)


def test_modulo_a_mao_fora_do_telhado_e_recusado(layout, telhado):
    longe = telhado.poligono_utm.centroid
    edicao = EdicaoLayout(extras_utm=[(longe.x + 500.0, longe.y + 500.0)])
    editado = aplicar_edicao(layout, telhado, edicao)
    assert editado.quantidade == layout.quantidade


def test_modulo_a_mao_no_vazio_entra(layout, telhado):
    """
    A beirada que a grade descartou por milímetros é o caso de uso do recurso.

    Desligar um módulo abre um vazio do tamanho exato de um módulo; pôr um
    módulo de volta nesse vazio tem de funcionar, senão a operação "acrescentar"
    não serve para nada.
    """
    centro = layout.modulos_geom[2].centroid
    edicao = EdicaoLayout(desativados={2}, extras_utm=[(centro.x, centro.y)])
    editado = aplicar_edicao(layout, telhado, edicao)
    assert editado.quantidade == layout.quantidade


# ----------------------------------------------------------------------------
# Deslocamento da grade
# ----------------------------------------------------------------------------
def test_deslocar_a_grade_move_todos_os_modulos(layout, telhado):
    edicao = EdicaoLayout(deslocamento_leste_m=0.5, deslocamento_norte_m=0.0)
    movidos = geometria_deslocada(layout, edicao)
    assert len(movidos) == len(layout.modulos_geom)
    for antes, depois in zip(layout.modulos_geom, movidos):
        assert depois.centroid.x == pytest.approx(antes.centroid.x + 0.5)
        assert depois.centroid.y == pytest.approx(antes.centroid.y)


def test_deslocamento_grande_derruba_o_que_saiu_do_telhado(layout, telhado):
    """Arrastar a grade para fora não pode deixar painel pendurado na calha."""
    edicao = EdicaoLayout(deslocamento_leste_m=12.0)
    editado = aplicar_edicao(layout, telhado, edicao)
    assert editado.quantidade < layout.quantidade
    assert any("para fora do telhado" in a for a in editado.avisos)
    permitido = telhado.poligono_utm.buffer(0.01)
    assert all(permitido.contains(g) for g in editado.modulos_geom)


def test_numeracao_e_estavel_sob_deslocamento(layout, telhado):
    """
    O índice 12 tem de continuar sendo o módulo 12 depois de arrastar.

    Se a numeração mudasse, o módulo que o usuário desligou passaria a ser
    outro a cada arrasto — e o croqui mostraria um buraco andando pelo telhado
    sem ninguém ter pedido.
    """
    edicao = EdicaoLayout(deslocamento_leste_m=1.0, deslocamento_norte_m=-0.5)
    movidos = geometria_deslocada(layout, edicao)
    esperado = translate(layout.modulos_geom[12], xoff=1.0, yoff=-0.5)
    assert movidos[12].equals_exact(esperado, 1e-9)


# ----------------------------------------------------------------------------
# Ponte com o mapa
# ----------------------------------------------------------------------------
def test_clique_sobre_um_modulo_acha_o_indice(layout):
    from aurum.geo.geometry import project_to_wgs84

    alvo = 7
    centro = project_to_wgs84(layout.modulos_geom[alvo].centroid, layout.utm_epsg)
    assert indice_no_ponto(layout, centro.y, centro.x) == alvo


def test_clique_no_vazio_nao_acha_nada(layout, telhado):
    from aurum.geo.geometry import project_to_wgs84

    longe = translate(telhado.poligono_utm.centroid, xoff=400.0, yoff=400.0)
    ponto = project_to_wgs84(longe, layout.utm_epsg)
    assert indice_no_ponto(layout, ponto.y, ponto.x) is None


def test_clique_acompanha_o_deslocamento(layout):
    """Com a grade arrastada, clicar onde o módulo ESTÁ é que tem de funcionar."""
    from aurum.geo.geometry import project_to_wgs84

    edicao = EdicaoLayout(deslocamento_leste_m=0.4)
    alvo = 4
    centro_novo = translate(layout.modulos_geom[alvo].centroid, xoff=0.4)
    ponto = project_to_wgs84(centro_novo, layout.utm_epsg)
    assert indice_no_ponto(layout, ponto.y, ponto.x, edicao) == alvo


def test_ida_e_volta_do_ponto_para_utm(layout):
    from aurum.geo.geometry import project_to_wgs84

    centro = layout.modulos_geom[1].centroid
    wgs = project_to_wgs84(centro, layout.utm_epsg)
    leste, norte = ponto_para_utm(layout, wgs.y, wgs.x)
    assert leste == pytest.approx(centro.x, abs=0.01)
    assert norte == pytest.approx(centro.y, abs=0.01)


# ----------------------------------------------------------------------------
# Croqui
# ----------------------------------------------------------------------------
def test_croqui_mostra_o_desligado_em_vez_de_escondê_lo(layout, telhado):
    """Módulo que some do mapa ao ser removido não tem como voltar."""
    edicao = EdicaoLayout(desativados={0, 1})
    croqui = croqui_editavel(layout, telhado, edicao)
    assert len(croqui["features"]) == layout.quantidade
    ativos = [f for f in croqui["features"] if f["properties"]["ativo"]]
    assert len(ativos) == layout.quantidade - 2


def test_croqui_marca_como_inativo_o_que_o_arrasto_jogou_para_fora(layout, telhado):
    edicao = EdicaoLayout(deslocamento_leste_m=12.0)
    croqui = croqui_editavel(layout, telhado, edicao)
    inativos = [f for f in croqui["features"] if not f["properties"]["ativo"]]
    assert inativos, "arrastar 12 m tem de empurrar módulos para fora"
    editado = aplicar_edicao(layout, telhado, edicao)
    assert len(croqui["features"]) - len(inativos) == editado.quantidade


def test_croqui_distingue_o_modulo_posto_a_mao(layout, telhado):
    centro = layout.modulos_geom[3].centroid
    edicao = EdicaoLayout(desativados={3}, extras_utm=[(centro.x, centro.y)])
    croqui = croqui_editavel(layout, telhado, edicao)
    manuais = [f for f in croqui["features"] if f["properties"]["origem"] == "manual"]
    assert len(manuais) == 1
    assert manuais[0]["properties"]["indice"] < 0, "índice manual não entra na numeração da grade"
