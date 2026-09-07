"""
Testes do telhado de várias águas.

Um telhado real quase nunca é um plano só, e as águas não somam a uma água
média: a do nascente enche de manhã, a do poente à tarde, e o conjunto é mais
plano que qualquer uma das duas. O que se garante aqui é que a combinação é
feita pela **geração** e não pelos ângulos — porque a média dos ângulos de
uma água a leste e outra a oeste dá uma água ao norte, com um pico ao meio-dia
que o telhado não tem.
"""
from __future__ import annotations

import numpy as np
import pytest

from aurum.bateria.geracao import SerieGeracao, combinar_series, serie_sintetica


def _serie(pico_hora: int, escala: float = 1.0, dias: int = 8) -> SerieGeracao:
    """Uma série artificial com o pico numa hora escolhida."""
    from datetime import date, timedelta

    curva = np.zeros(24)
    for hora in range(24):
        curva[hora] = max(0.0, 1000.0 - 180.0 * abs(hora - pico_hora))
    matriz = np.tile(curva * escala, (dias, 1))
    return SerieGeracao(
        datas=tuple(date(2020, 1, 1) + timedelta(days=i) for i in range(dias)),
        potencia_w_por_kwp=matriz,
        latitude=-25.0, longitude=-49.0,
        azimute_deg=90.0 if pico_hora < 12 else 270.0,
        inclinacao_deg=20.0, perdas_percent=14.0, fonte="teste",
    )


def test_a_combinacao_soma_a_geracao_e_nao_os_angulos():
    """
    Nascente e poente somam a uma curva de dois ombros, não a um pico ao meio-dia.

    É o erro que a média dos ângulos cometeria, e ele não aparece no total
    anual — aparece na hora, que é exatamente onde a bateria e o autoconsumo
    são decididos.
    """
    leste, oeste = _serie(9), _serie(15)
    junta = combinar_series([leste, oeste], [10.0, 10.0])

    dia = junta.potencia_w_por_kwp[0]
    assert dia[9] == pytest.approx((leste.potencia_w_por_kwp[0][9]) / 2, rel=1e-6)
    assert dia[15] == pytest.approx((oeste.potencia_w_por_kwp[0][15]) / 2, rel=1e-6)
    # O meio-dia da soma fica abaixo do pico de cada água — se a combinação
    # fosse pelos ângulos, seria o contrário.
    assert dia[12] < dia[9]
    assert dia[12] < dia[15]


def test_a_ponderacao_e_pela_potencia_de_cada_agua():
    """Uma água de 30 kWp pesa o triplo de uma de 10 kWp."""
    leste, oeste = _serie(9), _serie(15)
    junta = combinar_series([leste, oeste], [30.0, 10.0])
    esperado = (
        leste.potencia_w_por_kwp[0] * 30.0 + oeste.potencia_w_por_kwp[0] * 10.0
    ) / 40.0
    assert junta.potencia_w_por_kwp[0] == pytest.approx(esperado)


def test_a_energia_total_se_conserva():
    """
    A série combinada continua sendo de 1 kWp — do sistema, não de uma água.

    É o que permite tudo que consome :class:`SerieGeracao` continuar
    funcionando sem saber que existem águas.
    """
    leste, oeste = _serie(9), _serie(15, escala=0.8)
    junta = combinar_series([leste, oeste], [10.0, 30.0])
    total_separado = (
        leste.anual_kwh_por_kwp() * 10.0 + oeste.anual_kwh_por_kwp() * 30.0
    )
    assert junta.anual_kwh_por_kwp() * 40.0 == pytest.approx(total_separado, rel=1e-6)


def test_uma_agua_so_devolve_a_propria_serie():
    """O caminho comum não pode pagar nada pela existência do caminho novo."""
    leste = _serie(9)
    assert combinar_series([leste], [12.0]) is leste


def test_a_orientacao_declarada_e_a_da_agua_maior():
    """
    Guardar a média dos azimutes seria guardar uma orientação que não existe.

    A da água de maior potência é a que descreve melhor o conjunto, e é
    honesta: está no telhado.
    """
    leste, oeste = _serie(9), _serie(15)
    assert combinar_series([leste, oeste], [5.0, 30.0]).azimute_deg == oeste.azimute_deg
    assert combinar_series([leste, oeste], [30.0, 5.0]).azimute_deg == leste.azimute_deg


def test_a_composicao_fica_registrada_para_o_relatorio():
    """O dossiê monta a tabela das águas a partir daqui."""
    junta = combinar_series([_serie(9), _serie(15)], [10.0, 30.0])
    aguas = junta.metadados["aguas"]
    assert len(aguas) == 2
    assert junta.metadados["potencia_total_kwp"] == pytest.approx(40.0)
    assert all(a["produtividade_kwh_kwp_ano"] > 0 for a in aguas)


def test_potencia_total_nula_e_recusada():
    """Somar águas de zero kWp devolveria uma divisão por zero silenciosa."""
    with pytest.raises(ValueError):
        combinar_series([_serie(9), _serie(15)], [0.0, 0.0])


def test_series_de_tamanhos_diferentes_sao_recusadas():
    """Combinar dias que não se correspondem misturaria janeiro com julho."""
    with pytest.raises(ValueError):
        combinar_series([_serie(9, dias=8), _serie(15, dias=6)], [10.0, 10.0])


def test_a_serie_sintetica_tambem_combina():
    """Sem rede o estudo ainda roda, e as águas continuam valendo."""
    leste = serie_sintetica(-25.0, -49.0, 90.0, 20.0, 14.0, anos=(2020, 2020))
    oeste = serie_sintetica(-25.0, -49.0, 270.0, 20.0, 14.0, anos=(2020, 2020))
    junta = combinar_series([leste, oeste], [10.0, 10.0])
    assert junta.n_dias == leste.n_dias
    assert junta.anual_kwh_por_kwp() > 0
