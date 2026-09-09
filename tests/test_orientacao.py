"""
Testes da comparação entre geometrias de instalação.

Um sistema fora do plano ótimo costuma ser resumido a "perde tanto por cento". A
frase é verdadeira e insuficiente: a perda anual esconde a mudança mais
importante que a geometria produz, que é **em que época do ano a energia
aparece**. O que se garante aqui é que a inversão sazonal é medida, nomeada e
avisada — e que a comparação recusa série que não sabe descrever geometria.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from aurum.bateria.geracao import SerieGeracao
from aurum.pv.orientacao import comparar_orientacoes


def _serie(inclinacao: float, azimute: float, perfil) -> SerieGeracao:
    """
    Uma série de um ano cujo formato mensal é ditado pelo teste.

    ``perfil`` recebe o mês e devolve a potência de pico do dia, em W por kWp.
    Assim o teste controla exatamente a sazonalidade que quer examinar, sem
    depender de rede nem do PVGIS.
    """
    inicio = date(2020, 1, 1)
    datas, matriz = [], []
    for i in range(365):
        dia = inicio + timedelta(days=i)
        datas.append(dia)
        curva = np.zeros(24)
        curva[8:17] = perfil(dia.month)
        matriz.append(curva)
    return SerieGeracao(
        datas=tuple(datas), potencia_w_por_kwp=np.array(matriz),
        latitude=-22.9, longitude=-43.2,
        azimute_deg=azimute, inclinacao_deg=inclinacao, perdas_percent=14.0,
        fonte="pvgis_seriescalc",
    )


def _fabrica(perfis):
    """Devolve um `obter_serie` que serve a série combinada com a geometria."""
    def obter(latitude, longitude, azimute_deg, inclinacao_deg, anos):
        return _serie(inclinacao_deg, azimute_deg,
                      perfis[(inclinacao_deg, azimute_deg)])
    return obter


#: Telhado: mais no verão. Parede: mais no inverno. É a inversão, construída.
_TELHADO = lambda mes: 500.0 if mes in (12, 1, 2) else 380.0   # noqa: E731
_PAREDE = lambda mes: 120.0 if mes in (12, 1, 2) else 400.0    # noqa: E731

CASOS = [("Telhado a 22°, Norte", 22.0, 0.0), ("Parede a 90°, Norte", 90.0, 0.0)]
PERFIS = {(22.0, 0.0): _TELHADO, (90.0, 0.0): _PAREDE}


@pytest.fixture(scope="module")
def comparacao():
    return comparar_orientacoes(
        -22.9, -43.2, CASOS, potencia_kwp=3.05, adotada=1,
        obter_serie=_fabrica(PERFIS),
    )


def test_a_perda_anual_e_medida_contra_o_plano_otimo(comparacao):
    """A referência é a primeira entrada, e é contra ela que a perda vale."""
    assert comparacao.otima.rotulo.startswith("Telhado")
    assert comparacao.escolhida.rotulo.startswith("Parede")
    assert comparacao.perda < 0, "a parede rende menos que o telhado"


def test_a_inversao_sazonal_e_medida_e_nomeada(comparacao):
    """
    O achado que a perda anual esconde.

    No hemisfério sul, um plano vertical voltado ao Norte recebe o sol quase de
    frente no inverno, quando ele cruza o céu baixo, e o vê de raspão no verão,
    quando passa por cima. O telhado inclinado faz o contrário. Duas geometrias
    podem perder o mesmo no ano e servir estações opostas.
    """
    telhado, parede = comparacao.medidas
    assert telhado.razao_inverno_verao < 1.0, "telhado entrega no verão"
    assert parede.razao_inverno_verao > 1.0, "parede entrega no inverno"
    assert not telhado.inverte
    assert parede.inverte


def test_a_inversao_vira_aviso_e_nao_nota_de_rodape(comparacao):
    """
    Quem adota a geometria invertida precisa saber antes de assinar.

    Ela não desqualifica o sistema — muda o cálculo de autoconsumo, porque a
    energia passa a aparecer na estação em que a casa costuma gastar menos com
    climatização.
    """
    assert any("inverte a sazonalidade" in a for a in comparacao.avisos)


def test_sem_inversao_nao_ha_aviso():
    """Aviso que sai sempre deixa de ser aviso."""
    comparacao = comparar_orientacoes(
        -22.9, -43.2, CASOS, potencia_kwp=3.05, adotada=0,
        obter_serie=_fabrica(PERFIS),
    )
    assert not any("inverte a sazonalidade" in a for a in comparacao.avisos)


def test_serie_sintetica_e_denunciada():
    """
    Série sintética descreve o total do ano e descreve mal a geometria.

    E geometria é exatamente o que se está medindo: a inversão sazonal, que é o
    achado, sairia errada. Seguir calado entregaria um gráfico convincente
    construído sobre nada.
    """
    def obter(latitude, longitude, azimute_deg, inclinacao_deg, anos):
        serie = _serie(inclinacao_deg, azimute_deg, PERFIS[(inclinacao_deg, azimute_deg)])
        object.__setattr__(serie, "fonte", "sintetica")
        return serie

    comparacao = comparar_orientacoes(
        -22.9, -43.2, CASOS, potencia_kwp=3.05, obter_serie=obter)
    assert any("não veio do PVGIS" in a for a in comparacao.avisos)


def test_o_pior_dia_entra_na_medida(comparacao):
    """
    Para dimensionar bateria o que importa é o dia mais fraco, não a média.

    Duas geometrias com a mesma perda anual podem ter piores dias muito
    diferentes, e é o pior dia que decide se o banco enche.
    """
    for medida in comparacao.medidas:
        assert medida.diario_p05_kwh <= medida.diario_medio_kwh
        assert medida.diario_medio_kwh > 0


def test_a_tabela_marca_a_referencia_e_a_adotada(comparacao):
    """Sem as duas marcas, o leitor não sabe qual linha é o sistema dele."""
    tabela = comparacao.tabela()
    assert tabela["referencia"].sum() == 1
    assert tabela["adotada"].sum() == 1
    assert bool(tabela.loc[tabela["referencia"], "contra_otima"].iloc[0] == 0.0)
