"""
Testes do ajuste da curva típica à conta de luz.

O que se fixa aqui é o contrato do motor: a curva ajustada, somada no
calendário do ciclo, **reproduz a conta** — energia do mês, dos postos e, no
que a forma permitir, a demanda medida —, e o ensemble que sai dela carrega a
energia da conta com a variabilidade declarada como hipótese.
"""
from __future__ import annotations

import numpy as np
import pytest

from aurum.demanda.ajuste import (
    FATOR_DEMANDA_MEDIDA,
    AjusteCurva,
    ContaDeLuz,
    _interpolador_minuto,
    ajustar,
    ajustar_melhor_perfil,
    recortar_fracao,
)
from aurum.demanda.biblioteca import perfil_tipico, perfis_tipicos


@pytest.fixture
def comercial():
    return perfil_tipico("comercial_generico")


# ----------------------------------------------------------------------------
# A conta
# ----------------------------------------------------------------------------
def test_conta_do_grupo_a_soma_os_postos():
    conta = ContaDeLuz(consumo_ponta_kwh=900.0, consumo_fora_ponta_kwh=12000.0)
    assert conta.grupo == "A"
    assert conta.consumo_mensal_kwh == pytest.approx(12900.0)


def test_conta_sem_consumo_e_recusada():
    with pytest.raises(ValueError):
        ContaDeLuz()
    with pytest.raises(ValueError):
        ContaDeLuz(consumo_mensal_kwh=-10.0)
    with pytest.raises(ValueError, match="juntos"):
        ContaDeLuz(consumo_mensal_kwh=1000.0, abertura_h=8)


def test_calendario_do_ciclo_fecha():
    conta = ContaDeLuz(consumo_mensal_kwh=1000.0, dias_operacao_semana=6, dias_ciclo=28)
    assert conta.dias_uteis_operacao == pytest.approx(20.0)
    assert conta.dias_fim_de_semana_operacao == pytest.approx(4.0)
    assert conta.dias_fechado == pytest.approx(4.0)


def test_historico_da_os_fatores_por_estacao():
    conta = ContaDeLuz(consumo_por_mes_kwh={
        1: 1200, 2: 1200, 12: 1200,  # verão
        6: 800, 7: 800, 8: 800,      # inverno
    })
    fatores = conta.fatores_sazonais()
    assert fatores["verão"] == pytest.approx(1.2)
    assert fatores["inverno"] == pytest.approx(0.8)
    # Estação sem mês informado fica na média, não inventa.
    assert fatores["outono"] == 1.0 and fatores["primavera"] == 1.0
    assert conta.consumo_mensal_kwh == pytest.approx(1000.0)


# ----------------------------------------------------------------------------
# O ajuste
# ----------------------------------------------------------------------------
def test_grupo_b_fecha_a_energia_do_mes(comercial):
    ajuste = ajustar(ContaDeLuz(consumo_mensal_kwh=9000.0), comercial)
    assert ajuste.obtidos["energia_mensal_kwh"] == pytest.approx(9000.0)
    assert ajuste.fechou
    # Sete dias por semana: o dia de operação é a conta dividida pelo ciclo.
    assert ajuste.consumo_diario_operacao_kwh == pytest.approx(300.0)
    assert ajuste.expoente == 1.0


def test_dias_fechados_ficam_na_carga_de_base_e_a_conta_ainda_fecha(comercial):
    conta = ContaDeLuz(consumo_mensal_kwh=9000.0, dias_operacao_semana=5,
                       abertura_h=8, fechamento_h=18)
    ajuste = ajustar(conta, comercial)
    assert ajuste.obtidos["energia_mensal_kwh"] == pytest.approx(9000.0)
    # O dia fechado é plano, na hora mais vazia do dia aberto.
    assert np.allclose(ajuste.curva_fechado_kw, ajuste.curva_operacao_kw.min())
    # Fora do horário a curva está na base.
    fora = ~conta.mascara_aberto
    assert np.allclose(ajuste.curva_operacao_kw[fora], ajuste.carga_base_kw)
    # E o dia de operação é maior que a média diária, porque há dias fechados.
    assert ajuste.consumo_diario_operacao_kwh > 9000.0 / 30


def test_grupo_a_fecha_ponta_e_fora_de_ponta_exatamente(comercial):
    conta = ContaDeLuz(consumo_ponta_kwh=900.0, consumo_fora_ponta_kwh=12000.0,
                       dias_operacao_semana=6, abertura_h=7, fechamento_h=22)
    ajuste = ajustar(conta, comercial)
    assert ajuste.obtidos["energia_ponta_kwh"] == pytest.approx(900.0)
    assert ajuste.obtidos["energia_fora_ponta_kwh"] == pytest.approx(12000.0)
    assert ajuste.obtidos["energia_mensal_kwh"] == pytest.approx(12900.0)
    assert ajuste.fator_ponta != ajuste.fator_fora


def test_sabado_conta_como_fora_de_ponta(comercial):
    """Com 5 e com 6 dias, a ponta só vale nos dias úteis — o sábado vai para fora."""
    base = dict(consumo_ponta_kwh=900.0, consumo_fora_ponta_kwh=12000.0)
    cinco = ajustar(ContaDeLuz(dias_operacao_semana=5, **base), comercial)
    seis = ajustar(ContaDeLuz(dias_operacao_semana=6, **base), comercial)
    # A ponta vale nos mesmos cinco dias úteis, então o fator de ponta é igual;
    # a energia fora de ponta se espalha por mais um dia, então cai por dia.
    assert cinco.fator_ponta == pytest.approx(seis.fator_ponta)
    assert cinco.curva_operacao_kw[12] > seis.curva_operacao_kw[12]
    for ajuste in (cinco, seis):
        assert ajuste.obtidos["energia_ponta_kwh"] == pytest.approx(900.0)


def test_demanda_medida_afina_o_pico_sem_mudar_a_energia(comercial):
    sem = ajustar(ContaDeLuz(consumo_mensal_kwh=9000.0), comercial)
    alvo = sem.demanda_maxima_kw * 1.25 * FATOR_DEMANDA_MEDIDA
    com = ajustar(ContaDeLuz(consumo_mensal_kwh=9000.0, demanda_fora_ponta_kw=alvo), comercial)
    assert com.expoente > 1.0
    assert com.demanda_maxima_kw == pytest.approx(alvo / FATOR_DEMANDA_MEDIDA, rel=0.02)
    assert com.obtidos["energia_mensal_kwh"] == pytest.approx(9000.0)
    assert com.fechou
    # A madrugada não se move: a carga de base é a mesma.
    assert com.carga_base_kw == pytest.approx(sem.carga_base_kw)
    assert (com.curva_operacao_kw >= com.carga_base_kw - 1e-9).all()


def test_demanda_medida_baixa_achata(comercial):
    sem = ajustar(ContaDeLuz(consumo_mensal_kwh=9000.0), comercial)
    alvo = sem.demanda_maxima_kw * 0.85 * FATOR_DEMANDA_MEDIDA
    com = ajustar(ContaDeLuz(consumo_mensal_kwh=9000.0, demanda_fora_ponta_kw=alvo), comercial)
    assert com.expoente < 1.0
    assert com.demanda_maxima_kw == pytest.approx(alvo / FATOR_DEMANDA_MEDIDA, rel=0.02)


def test_demanda_impossivel_avisa_em_vez_de_inventar(comercial):
    sem = ajustar(ContaDeLuz(consumo_mensal_kwh=9000.0), comercial)
    absurda = sem.demanda_maxima_kw * 4.0
    com = ajustar(ContaDeLuz(consumo_mensal_kwh=9000.0, demanda_fora_ponta_kw=absurda), comercial)
    assert not com.fechou
    assert com.erros_pct["demanda_fora_ponta_kw"] > 5.0
    assert any("não cabe na forma" in a for a in com.avisos)
    assert any("não fechou" in a for a in com.avisos)
    # Mesmo assim a energia continua sendo a da conta.
    assert com.obtidos["energia_mensal_kwh"] == pytest.approx(9000.0)


def test_consumo_na_ponta_sem_forma_na_ponta_cai_para_o_total(comercial):
    """Fecha às 17 h e a conta diz que há ponta: o ajuste fecha a conta e avisa."""
    conta = ContaDeLuz(consumo_ponta_kwh=500.0, consumo_fora_ponta_kwh=8000.0,
                       abertura_h=8, fechamento_h=17)
    ajuste = ajustar(conta, comercial)
    # A conta manda: a energia da ponta entra sobre a carga de base dessas
    # horas, e o ajuste diz que o horário e a conta se contradizem.
    assert ajuste.obtidos["energia_mensal_kwh"] == pytest.approx(8500.0)
    assert ajuste.obtidos["energia_ponta_kwh"] == pytest.approx(500.0)
    assert any("fecha antes dela" in a for a in ajuste.avisos)


def test_sazonalidade_entra_no_ensemble(comercial):
    conta = ContaDeLuz(consumo_por_mes_kwh={m: (1300.0 if m in (12, 1, 2) else 900.0) for m in range(1, 13)})
    ajuste = ajustar(conta, comercial)
    assert ajuste.sazonalidade_aplicada
    ensemble = ajuste.ensemble(num_simulacoes=200, semente=1)
    verao = ensemble.energia_diaria_kwh("verão").mean()
    inverno = ensemble.energia_diaria_kwh("inverno").mean()
    assert verao / inverno == pytest.approx(1300.0 / 900.0, rel=0.08)
    assert ensemble.metadados["sazonalidade_aplicada"] is True
    # E o ano inteiro soma o histórico.
    assert ajuste.consumo_anual_kwh == pytest.approx(3 * 1300.0 + 9 * 900.0)


def test_perfil_sintetico_e_declarado(comercial):
    assert not comercial.confiavel
    ajuste = ajustar(ContaDeLuz(consumo_mensal_kwh=9000.0), comercial)
    assert any("sintética" in a for a in ajuste.avisos)


# ----------------------------------------------------------------------------
# A edição
# ----------------------------------------------------------------------------
def test_curva_editada_mantem_a_energia_da_conta(comercial):
    conta = ContaDeLuz(consumo_mensal_kwh=9000.0, dias_operacao_semana=6,
                       abertura_h=8, fechamento_h=18)
    ajuste = ajustar(conta, comercial)
    curva = ajuste.curva_operacao_kw.copy()
    curva[7] = 40.0  # o chuveiro das sete
    editado = ajuste.com_curva_editada(curva)
    assert editado.editada
    assert editado.obtidos["energia_mensal_kwh"] == pytest.approx(9000.0)
    assert editado.fechou
    # A forma é a digitada, reescalada; a hora sete ficou proporcionalmente alta.
    assert editado.curva_operacao_kw[7] / editado.curva_operacao_kw[12] == pytest.approx(
        40.0 / curva[12])
    # O dia fechado acompanha a hora mais vazia da curva editada.
    assert np.allclose(editado.curva_fechado_kw, editado.curva_operacao_kw.min())
    assert any("editada pelo operador" in a for a in editado.avisos)
    assert editado.resumo()["editada"] is True
    # O ajuste original não foi tocado.
    assert not ajuste.editada and ajuste.curva_operacao_kw[7] != 40.0


def test_curva_editada_sem_manter_energia_vale_em_kw(comercial):
    ajuste = ajustar(ContaDeLuz(consumo_mensal_kwh=9000.0), comercial)
    curva = ajuste.curva_operacao_kw * 1.2
    editado = ajuste.com_curva_editada(curva, manter_energia=False)
    assert np.allclose(editado.curva_operacao_kw, curva)
    assert editado.obtidos["energia_mensal_kwh"] == pytest.approx(9000.0 * 1.2)
    assert not editado.fechou


def test_curva_editada_invalida_e_recusada(comercial):
    ajuste = ajustar(ContaDeLuz(consumo_mensal_kwh=9000.0), comercial)
    with pytest.raises(ValueError):
        ajuste.com_curva_editada([1.0] * 23)
    with pytest.raises(ValueError):
        ajuste.com_curva_editada([0.0] * 24)
    with pytest.raises(ValueError):
        ajuste.com_curva_editada([-1.0] + [1.0] * 23)


def test_curva_editada_chega_ao_ensemble(comercial):
    ajuste = ajustar(ContaDeLuz(consumo_mensal_kwh=9000.0), comercial)
    curva = np.full(24, 1.0)
    curva[20] = 30.0
    editado = ajuste.com_curva_editada(curva)
    ensemble = editado.ensemble(num_simulacoes=50, dispersao_diaria=0.0, dispersao_horaria=0.0)
    medio = ensemble.perfil_medio_w() / 1000.0
    assert int(np.argmax(medio)) // 60 == 20
    assert ensemble.energia_diaria_kwh().mean() * 30 == pytest.approx(9000.0)


# ----------------------------------------------------------------------------
# O ensemble
# ----------------------------------------------------------------------------
def test_interpolador_preserva_energia_e_nao_passa_do_pico():
    matriz = _interpolador_minuto()
    assert matriz.shape == (24, 1440)
    # Cada minuto é combinação convexa de duas horas.
    assert np.allclose(matriz.sum(axis=0), 1.0)
    # Cada hora distribui exatamente 60 minutos.
    assert np.allclose(matriz.sum(axis=1), 60.0)
    horaria = np.random.default_rng(0).uniform(1, 10, size=24)
    minuto = horaria @ matriz
    assert minuto.sum() / 60.0 == pytest.approx(horaria.sum())
    assert minuto.max() <= horaria.max() + 1e-9
    assert minuto.min() >= horaria.min() - 1e-9


def test_ensemble_carrega_a_energia_da_conta_com_cauda(comercial):
    conta = ContaDeLuz(consumo_mensal_kwh=9000.0, dias_operacao_semana=5,
                       abertura_h=8, fechamento_h=18)
    ajuste = ajustar(conta, comercial)
    ensemble = ajuste.ensemble(num_simulacoes=400, semente=7)
    assert ensemble.metadados["origem"] == "curva_tipica_ajustada"
    assert ensemble.passos_por_dia == 1440
    # Média dos dias × dias do ciclo = a conta (ruído de média um).
    assert ensemble.energia_diaria_kwh().mean() * 30 == pytest.approx(9000.0, rel=0.03)
    # Há cauda: os picos variam, e o P95 fica acima da curva média.
    picos = ensemble.picos_diarios_w() / 1000.0
    assert picos.std() > 0
    assert np.percentile(picos, 95) > ajuste.demanda_maxima_kw
    # Dias fechados entram na proporção da semana: 2/7 dos dias ficam na base.
    energias = ensemble.energia_diaria_kwh("verão")
    fechados = (energias < ajuste.curva_fechado_kw.sum() * 1.6).sum()
    assert fechados == pytest.approx(400 * 2 / 7, abs=1)


def test_ensemble_sem_dispersao_e_a_propria_curva(comercial):
    ajuste = ajustar(ContaDeLuz(consumo_mensal_kwh=9000.0), comercial)
    ensemble = ajuste.ensemble(num_simulacoes=10, dispersao_diaria=0.0, dispersao_horaria=0.0)
    assert ensemble.energia_diaria_kwh().std() == pytest.approx(0.0)
    assert ensemble.picos_diarios_w().max() / 1000.0 == pytest.approx(ajuste.demanda_maxima_kw)


def test_recortar_fracao_escala_perfis_e_picos(comercial):
    ajuste = ajustar(ContaDeLuz(consumo_mensal_kwh=9000.0), comercial)
    ensemble = ajuste.ensemble(num_simulacoes=20)
    metade = recortar_fracao(ensemble, 0.5)
    assert metade.energia_diaria_kwh().mean() == pytest.approx(ensemble.energia_diaria_kwh().mean() / 2)
    assert metade.picos_diarios_w().mean() == pytest.approx(ensemble.picos_diarios_w().mean() / 2)
    assert metade.metadados["fracao_backup"] == 0.5
    assert recortar_fracao(ensemble, 1.0) is ensemble
    with pytest.raises(ValueError):
        recortar_fracao(ensemble, 0.0)


# ----------------------------------------------------------------------------
# A escolha do perfil
# ----------------------------------------------------------------------------
def test_so_com_o_total_vale_o_perfil_do_segmento(comercial):
    ajuste = ajustar_melhor_perfil(ContaDeLuz(consumo_mensal_kwh=9000.0), preferido=comercial)
    assert ajuste.perfil.id == comercial.id
    assert not any("melhor explica" in a for a in ajuste.avisos)


def test_com_demanda_o_ajuste_pode_trocar_o_perfil_e_diz(comercial):
    """
    Uma conta com a forma de um frigorífico — quase plana — não é de comércio.

    O ajuste tem que preferir a forma que reproduz a demanda sem deformação e
    avisar que trocou.
    """
    plano = perfil_tipico("frigorifico")
    referencia = ajustar(ContaDeLuz(consumo_mensal_kwh=30000.0), plano)
    conta = ContaDeLuz(
        consumo_mensal_kwh=30000.0,
        demanda_fora_ponta_kw=referencia.demanda_maxima_kw * FATOR_DEMANDA_MEDIDA,
    )
    escolhido = ajustar_melhor_perfil(conta, preferido=comercial)
    assert escolhido.perfil.id != comercial.id
    assert escolhido.fechou
    assert any("melhor explica" in a for a in escolhido.avisos)
    assert abs(escolhido.expoente - 1.0) < abs(ajustar(conta, comercial).expoente - 1.0)


def test_todos_os_perfis_da_base_ajustam_sem_erro():
    conta = ContaDeLuz(consumo_ponta_kwh=1000.0, consumo_fora_ponta_kwh=15000.0,
                       demanda_ponta_kw=40.0, demanda_fora_ponta_kw=70.0,
                       dias_operacao_semana=6, abertura_h=6, fechamento_h=23)
    for perfil in perfis_tipicos().values():
        ajuste = ajustar(conta, perfil)
        assert isinstance(ajuste, AjusteCurva)
        assert ajuste.obtidos["energia_mensal_kwh"] == pytest.approx(16000.0)
        assert (ajuste.curva_operacao_kw >= 0).all()
        assert np.isfinite(ajuste.curva_operacao_kw).all()


# ----------------------------------------------------------------------------
# O estudo
# ----------------------------------------------------------------------------
def test_estudo_com_ajuste_nao_cai_no_cenario_de_demonstracao(comercial):
    """
    O defeito que motivou o módulo: sem cômodos, o estudo simulava o hotel
    de demonstração e o dossiê saía com ele no lugar da conta do cliente.
    """
    from aurum.bateria.apagao import MalhaApagao
    from aurum.bateria.estudo import ConfiguracaoEstudo, executar_estudo

    conta = ContaDeLuz(consumo_mensal_kwh=9000.0, dias_operacao_semana=6,
                       abertura_h=8, fechamento_h=20)
    ajuste = ajustar(conta, comercial)
    cfg = ConfiguracaoEstudo(
        latitude=-25.43, longitude=-49.27, nome="Loja",
        ajuste_conta=ajuste, fracao_backup=0.4,
        simulacoes=30, potencia_fv_kwp=10.0, anos_serie=(2019, 2020),
        max_candidatos=2, autonomia_alvo_h=2.0,
        malha=MalhaApagao(duracoes_h=(1.0, 2.0), horas_inicio=(0, 12, 18), amostras=10, passo_min=15),
    )
    estudo = executar_estudo(cfg)
    total = estudo.ensemble_total.energia_diaria_kwh().mean() * 30
    assert total == pytest.approx(9000.0, rel=0.06)
    backup = estudo.ensemble_backup.energia_diaria_kwh().mean() * 30
    assert backup == pytest.approx(total * 0.4, rel=0.01)
    assert estudo.ensemble_total.metadados["origem"] == "curva_tipica_ajustada"
    assert estudo.configuracao.consumo_anual_kwh == pytest.approx(9000.0 * 12)
    assert any("hipótese" in a for a in estudo.avisos)
    assert any("40%" in a for a in estudo.avisos)
    assert not estudo.ranking.empty
