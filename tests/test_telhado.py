"""
Testes do telhado marcado no mapa e dos catálogos alimentados por datasheet.

O que estes testes protegem:

* **A medida.** Área, orientação e contagem de módulos saem de um desenho em
  latitude/longitude que precisa passar por UTM antes de virar metro. Um erro
  de projeção não estoura — devolve um número plausível e errado.
* **A ambiguidade da orientação.** O contorno de um telhado de duas águas
  admite **duas** orientações, e o programa tem que oferecer as duas em vez de
  fingir que sabe qual é.
* **A procedência dos catálogos.** Depois de acrescentar os datasheets, o que
  foi lido de documento não pode se misturar com o que é ordem de grandeza de
  mercado. A coluna ``fonte_dado`` é o que separa os dois, e ela some fácil
  numa edição descuidada da planilha.
"""
from __future__ import annotations

import math

import pytest
from shapely.geometry import Polygon

from aurum.pv.equipment import carregar_base
from aurum.pv.telhado import (
    Telhado,
    croqui_geojson,
    dimensionar_no_telhado,
    telhado_de_geojson,
)

LAT, LON = -25.4284, -49.2733


def _retangulo(largura_m: float, altura_m: float, lat: float = LAT, lon: float = LON) -> dict:
    """Retângulo aproximado em graus, com o lado maior no sentido leste-oeste."""
    dlon = largura_m / (111_320 * abs(math.cos(math.radians(lat))))
    dlat = altura_m / 110_540
    return {
        "type": "Polygon",
        "coordinates": [[
            [lon, lat], [lon + dlon, lat], [lon + dlon, lat + dlat],
            [lon, lat + dlat], [lon, lat],
        ]],
    }


# ----------------------------------------------------------------------------
# Medida
# ----------------------------------------------------------------------------
def test_area_do_desenho_bate_com_a_area_pedida():
    telhado = telhado_de_geojson(_retangulo(40, 25), "Galpão")
    assert telhado.area_m2 == pytest.approx(1000.0, rel=0.02)
    assert telhado.perimetro_m == pytest.approx(130.0, rel=0.02)


def test_area_inclinada_supera_a_projecao_horizontal():
    """A projeção vista de cima encolhe pelo cosseno; é a superfície que recebe módulo."""
    telhado = telhado_de_geojson(_retangulo(40, 25), "Galpão", inclinacao_deg=30.0)
    esperado = telhado.area_m2 / math.cos(math.radians(30.0))
    assert telhado.area_inclinada_m2 == pytest.approx(esperado, rel=1e-6)
    assert telhado.area_inclinada_m2 > telhado.area_m2 * 1.15


def test_laje_plana_nao_ganha_area_pela_inclinacao():
    """A estrutura inclina, o telhado não: a área disponível é a da laje."""
    telhado = telhado_de_geojson(_retangulo(40, 25), "Laje", montagem="inclinado")
    assert telhado.area_inclinada_m2 == pytest.approx(telhado.area_m2)


def test_desenho_que_nao_fecha_area_e_recusado_com_mensagem():
    linha = {"type": "LineString", "coordinates": [[LON, LAT], [LON + 0.001, LAT]]}
    with pytest.raises(ValueError, match="polígono|área"):
        telhado_de_geojson(linha, "linha")


def test_aceita_feature_alem_de_geometria():
    """O Draw do folium devolve Feature; a geometria crua também tem que passar."""
    feature = {"type": "Feature", "properties": {}, "geometry": _retangulo(30, 20)}
    telhado = telhado_de_geojson(feature, "Galpão")
    assert telhado.area_m2 == pytest.approx(600.0, rel=0.03)


def test_desenho_minusculo_gera_aviso():
    telhado = telhado_de_geojson(_retangulo(2, 2), "Errado")
    assert any("escala" in a for a in telhado.avisos)


# ----------------------------------------------------------------------------
# Orientação
# ----------------------------------------------------------------------------
def test_cumeeira_leste_oeste_da_duas_aguas_norte_e_sul():
    """O contorno não distingue para que lado a água cai — as duas têm que aparecer."""
    telhado = telhado_de_geojson(_retangulo(40, 25), "Galpão")
    assert telhado.azimute_cumeeira_deg == pytest.approx(90.0, abs=2.0)
    opcoes = sorted(telhado.azimutes_possiveis())
    assert len(opcoes) == 2
    assert opcoes[0] == pytest.approx(0.0, abs=2.0) or opcoes[0] == pytest.approx(180.0, abs=2.0)
    assert abs((opcoes[1] - opcoes[0]) - 180.0) < 4.0


def test_no_hemisferio_sul_a_sugestao_e_a_face_norte():
    telhado = telhado_de_geojson(_retangulo(40, 25), "Galpão")
    azimute, explicacao = telhado.sugerir_azimute()
    assert min(azimute, 360 - azimute) < 10.0  # perto de 0° = Norte
    assert "Norte" in explicacao
    assert "troque" in explicacao, "a explicação tem que admitir a ambiguidade"


def test_laje_plana_usa_o_otimo_da_latitude_sem_ambiguidade():
    telhado = telhado_de_geojson(_retangulo(40, 25), "Laje", montagem="inclinado")
    assert len(telhado.azimutes_possiveis()) == 1
    azimute, explicacao = telhado.sugerir_azimute()
    assert azimute == pytest.approx(0.0)
    assert "livre" in explicacao


def test_no_hemisferio_norte_a_face_otima_inverte():
    telhado = telhado_de_geojson(
        _retangulo(40, 25, lat=40.0, lon=-3.7), "Nave", montagem="inclinado")
    assert telhado.sugerir_azimute()[0] == pytest.approx(180.0)


# ----------------------------------------------------------------------------
# Dimensionamento
# ----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def base():
    return carregar_base()


def test_telhado_maior_cabe_mais_modulos(base):
    pequeno = dimensionar_no_telhado(telhado_de_geojson(_retangulo(20, 12), "P"), base)
    grande = dimensionar_no_telhado(telhado_de_geojson(_retangulo(40, 25), "G"), base)
    assert 0 < pequeno.quantidade < grande.quantidade
    assert pequeno.potencia_kwp < grande.potencia_kwp


def test_laje_plana_cabe_menos_que_telhado_coplanar(base):
    """Fileiras inclinadas precisam de espaçamento para não se sombrearem."""
    coplanar = dimensionar_no_telhado(
        telhado_de_geojson(_retangulo(40, 25), "G", montagem="coplanar"), base)
    laje = dimensionar_no_telhado(
        telhado_de_geojson(_retangulo(40, 25), "G", montagem="inclinado"), base)
    assert laje.quantidade < coplanar.quantidade
    assert laje.densidade_wp_m2 < coplanar.densidade_wp_m2


def test_densidade_fica_na_faixa_plausivel(base):
    """
    Telhado coplanar com módulo moderno rende de 150 a 200 Wp/m² de telhado
    bruto. Fora dessa faixa, ou o empacotamento quebrou ou a projeção falhou.
    """
    layout = dimensionar_no_telhado(telhado_de_geojson(_retangulo(40, 25), "G"), base)
    assert 130 <= layout.densidade_wp_m2 <= 210
    assert 0.5 <= layout.taxa_ocupacao <= 0.95


def test_desconto_de_obstaculos_reduz_a_contagem(base):
    cheio = dimensionar_no_telhado(
        telhado_de_geojson(_retangulo(40, 25), "G", fator_obstaculos=1.0), base)
    descontado = dimensionar_no_telhado(
        telhado_de_geojson(_retangulo(40, 25), "G", fator_obstaculos=0.7), base)
    assert descontado.quantidade < cheio.quantidade


def test_croqui_volta_para_latitude_longitude(base):
    telhado = telhado_de_geojson(_retangulo(40, 25), "G")
    layout = dimensionar_no_telhado(telhado, base)
    croqui = croqui_geojson(layout)
    assert croqui["type"] == "FeatureCollection"
    assert len(croqui["features"]) == layout.quantidade
    # Os módulos têm que cair sobre o telhado, não em outro continente.
    primeiro = Polygon(croqui["features"][0]["geometry"]["coordinates"][0])
    assert primeiro.centroid.distance(telhado.poligono_wgs84.centroid) < 0.001


def test_telhado_vira_dicionario_serializavel(base):
    telhado = telhado_de_geojson(_retangulo(40, 25), "G")
    dados = telhado.as_dict()
    assert dados["area_m2"] > 0
    assert dados["geojson"]["type"] in ("Polygon", "MultiPolygon")
    assert dados["orientacao"]


# ----------------------------------------------------------------------------
# Catálogos alimentados por datasheet
# ----------------------------------------------------------------------------
def test_catalogo_fv_tem_os_modelos_da_pace(base):
    """Inversor GoodWe de 50 kW e módulos de 620 Wp das marcas especificadas."""
    modelos = {m.modelo for m in base.modulos}
    assert "LR7-72HGD-620M" in modelos
    assert any("DHN-78X16" in m for m in modelos)

    inversores = {i.modelo for i in base.inversores}
    assert {"GW50KN-MT", "GW50KS-MT"} <= inversores
    goodwe50 = base.inversor_por_modelo("GW50KN-MT")
    assert goodwe50.potencia_ca_w == 50_000
    assert goodwe50.fases == 3
    assert goodwe50.num_mppt == 4
    assert goodwe50.tensao_max_cc == 1100


def test_modulos_de_620wp_trazem_dimensao_real_de_datasheet(base):
    """Dimensão derivada da eficiência erra a área — e a área decide a contagem."""
    longi = base.modulo_por_modelo("LR7-72HGD-620M")
    assert longi is not None
    assert not longi.dimensoes_derivadas
    assert longi.comprimento_m == pytest.approx(2.382)
    assert longi.largura_m == pytest.approx(1.134)
    assert longi.area_m2 == pytest.approx(2.70, abs=0.02)


def test_catalogo_de_baterias_tem_a_linha_goodwe():
    from aurum.bateria.catalogo import carregar_catalogo

    catalogo = carregar_catalogo()
    baterias = {b.modelo for b in catalogo.baterias}
    inversores = {i.modelo for i in catalogo.inversores}
    assert any(m.startswith("LX F") for m in baterias), "faltam as torres Lynx"
    assert {"GW15K-ET", "GW30K-ET"} <= inversores

    et = catalogo.inversor_por_modelo("GW25K-ET")
    assert et.potencia_ca_nominal_kw == 25.0
    assert et.potencia_ca_pico_kw == 30.0
    assert et.duracao_pico_s == 60.0  # tabelado, não arbitrado
    assert et.fases == 3


def test_torre_lynx_pequena_demais_e_recusada_pelo_et():
    """
    O datasheet avisa: a torre de 6,4 kWh (128 V) não acende os ET de 15-30 kW,
    cuja janela de bateria começa em 200 V. A verificação de tensão tem que
    reproduzir isso sozinha.
    """
    from aurum.bateria.catalogo import carregar_catalogo

    catalogo = carregar_catalogo()
    et = catalogo.inversor_por_modelo("GW15K-ET")
    pequena = catalogo.bateria_por_modelo("LX F6.4-H-20")
    grande = catalogo.bateria_por_modelo("LX F16.0-H-20")
    assert not et.aceita_banco(pequena)
    assert et.aceita_banco(grande)


def test_linhas_de_datasheet_nao_se_misturam_com_ordem_de_grandeza():
    from aurum.bateria.catalogo import carregar_catalogo

    catalogo = carregar_catalogo()
    goodwe = [i for i in catalogo.inversores if i.fabricante == "GoodWe" and "ET" in i.modelo]
    outros = [i for i in catalogo.inversores if i.fabricante != "GoodWe"]
    assert goodwe and outros
    assert all("goodwe.com" in i.fonte_dado for i in goodwe)
    assert all("conferir" in i.fonte_dado.lower() for i in outros)


def test_semear_nao_apaga_ajuste_manual(tmp_path):
    """Reexecutar a semente não pode desfazer correção feita à mão na planilha."""
    import pandas as pd

    from aurum.bateria.catalogo import criar_planilha_modelo, semear_planilha

    caminho = tmp_path / "BDBaterias.xlsx"
    criar_planilha_modelo(caminho)

    abas = pd.read_excel(caminho, sheet_name=None)
    baterias = abas["baterias"]
    alvo = baterias.index[baterias["modelo"] == "US5000"][0]
    baterias.loc[alvo, "preco_brl"] = 99_999.0
    with pd.ExcelWriter(caminho, engine="openpyxl") as escritor:
        for nome, tabela in abas.items():
            tabela.to_excel(escritor, sheet_name=nome, index=False)

    semear_planilha(caminho)
    depois = pd.read_excel(caminho, sheet_name="baterias")
    preco = depois.loc[depois["modelo"] == "US5000", "preco_brl"].iloc[0]
    assert preco == 99_999.0
    assert len(depois[depois["modelo"] == "US5000"]) == 1, "a semente duplicou a linha"
