"""
Testes da comparação de fontes: solar, bateria e gerador.

O que se testa aqui não é aritmética de fluxo de caixa — é o conjunto de
afirmações que o quadro de cenários faz ao cliente, e que estariam erradas em
silêncio se o modelo mudasse:

* solar sozinho não dá backup nenhum;
* bateria sem sol não economiza nada na tarifa simples;
* gerador atravessa apagão longo, e falha por potência, não por energia;
* o cenário de referência é o que o cliente paga hoje, e as economias são
  todas medidas contra ele.
"""
from __future__ import annotations

import numpy as np
import pytest

from aurum.bateria.apagao import MalhaApagao
from aurum.bateria.catalogo import carregar_catalogo
from aurum.bateria.despacho import CAUSA_POTENCIA, Gerador, LimitesDespacho, simular_ilhamento
from aurum.bateria.economia import PremissasBateria
from aurum.bateria.fontes import (
    Composicao,
    Fatura,
    combinacoes,
    comparar_fontes,
)
from aurum.bateria.geracao import serie_sintetica
from aurum.demanda import cenario_exemplo, simular_ensemble


# ----------------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ensembles():
    comodos, instancias = cenario_exemplo("residencia")
    total = simular_ensemble(comodos, instancias, 30).reamostrar(15)
    essencial = comodos[:1]
    backup = simular_ensemble(
        essencial, {c.nome: instancias.get(c.nome, 1) for c in essencial}, 30
    ).reamostrar(15)
    return total, backup


@pytest.fixture(scope="module")
def serie():
    return serie_sintetica(-25.43, -49.27, 0.0, 20.0, 14.0, anos=(2019, 2020))


@pytest.fixture(scope="module")
def conjunto():
    base = carregar_catalogo()
    return next(c for c in base.combinacoes(modulos=(2,)) if c.energia_util_kwh > 5)


@pytest.fixture(scope="module")
def malha():
    return MalhaApagao(duracoes_h=(1.0, 3.0, 12.0), horas_inicio=(0, 8, 18), amostras=20, passo_min=15)


@pytest.fixture(scope="module")
def comparacao(ensembles, serie, conjunto, malha):
    total, backup = ensembles
    return comparar_fontes(
        ensemble_total=total,
        ensemble_backup=backup,
        serie=serie,
        # 3 kWp de propósito: com 8 kWp a residência gera tanto que a conta
        # bate no piso do custo de disponibilidade nos dois cenários, e a
        # diferença que o teste mede desaparece atrás do piso.
        potencia_fv_kwp=3.0,
        fatura=Fatura(tarifa_brl_kwh=0.95, custo_disponibilidade_kwh=50.0),
        conjunto=conjunto,
        gerador=Gerador(potencia_kw=15.0, custo_energia_brl_kwh=2.4, capex_brl=25_000.0),
        premissas=PremissasBateria(custo_interrupcao_brl_kwh=12.0),
        malha=malha,
        dias_por_estacao=12,
    )


# ----------------------------------------------------------------------------
# Enumeração dos arranjos
# ----------------------------------------------------------------------------
def test_oito_arranjos_com_as_tres_fontes():
    todas = combinacoes()
    assert len(todas) == 8
    assert todas[0] == Composicao(), "o referencial vem primeiro"
    assert Composicao(True, True, True) in todas


def test_fonte_indisponivel_nao_gera_cenario():
    # Cliente que não quer gerador não deve ver quatro linhas sobre gerador.
    sem_grupo = combinacoes(gerador=False)
    assert len(sem_grupo) == 4
    assert all(not c.gerador for c in sem_grupo)


# ----------------------------------------------------------------------------
# As afirmações do quadro
# ----------------------------------------------------------------------------
def test_solar_sozinho_nao_da_backup(comparacao):
    so_solar = comparacao.por_chave("solar")
    assert so_solar is not None
    assert so_solar.autonomia_garantida_h == 0.0
    assert all(p == 0.0 for p in so_solar.prob_por_duracao.values())
    assert any("anti-ilhamento" in a for a in so_solar.avisos)


def test_solar_reduz_a_conta(comparacao):
    base = comparacao.base
    so_solar = comparacao.por_chave("solar")
    assert so_solar.conta_anual_brl < base.conta_anual_brl
    assert so_solar.economia_anual_brl > 0


def test_bateria_sem_sol_nao_economiza_na_tarifa_simples(comparacao):
    """
    Sem diferença de tarifa entre horários e sem excedente para guardar, a
    bateria não tem de onde tirar economia. Um modelo que devolvesse economia
    aqui estaria dando ao banco uma carga inicial de graça em cada dia
    simulado — foi exatamente esse o defeito que a passada de aquecimento
    corrigiu.
    """
    so_bateria = comparacao.por_chave("bateria")
    assert so_bateria.economia_anual_brl == pytest.approx(0.0, abs=1.0)
    assert so_bateria.autonomia_garantida_h > 0, "mas backup ela dá"


def test_bateria_com_sol_economiza_pouco_e_por_um_motivo_so(comparacao):
    """O ganho é o Fio B recuperado, e por isso é pequeno perto do solar."""
    so_solar = comparacao.por_chave("solar")
    com_banco = comparacao.por_chave("solar+bateria")
    assert com_banco.economia_anual_brl > so_solar.economia_anual_brl
    assert com_banco.economia_anual_brl < so_solar.economia_anual_brl * 1.5


def test_conta_tem_piso_no_custo_de_disponibilidade():
    fatura = Fatura(tarifa_brl_kwh=1.0, custo_disponibilidade_kwh=50.0)
    # Sistema que zera a compra e injeta um caminhão de energia: a conta para
    # no mínimo, nunca fica negativa.
    assert fatura.conta_anual_brl(0.0, 50_000.0) == pytest.approx(50 * 12 * 1.0)


def test_economia_e_sempre_medida_contra_a_rede(comparacao):
    base = comparacao.base
    assert base.composicao.n_fontes == 0
    assert base.capex_brl == 0.0
    assert base.economia_anual_brl == 0.0
    for cenario in comparacao.cenarios:
        esperado = base.conta_anual_brl - cenario.conta_anual_brl
        assert cenario.economia_anual_brl == pytest.approx(esperado)


def test_todo_cenario_com_investimento_tem_fluxo_de_caixa(comparacao):
    for cenario in comparacao.cenarios:
        assert len(cenario.fluxo) == comparacao.premissas.anos_analise
        if cenario.capex_brl > 0:
            assert cenario.vpl_brl != 0.0


# ----------------------------------------------------------------------------
# O gerador no despacho
# ----------------------------------------------------------------------------
def _janela(carga_kw: float, potencia_kw: float, passos: int = 48) -> tuple:
    carga = np.full((1, passos), carga_kw)
    return carga, carga, np.zeros((1, passos))


def test_gerador_sozinho_atravessa_apagao_longo():
    """
    Energia ilimitada é o ponto do gerador: 12 h de carga constante passam sem
    banco nenhum, o que nenhuma bateria deste catálogo faria.
    """
    carga, pico, sol = _janela(5.0, 15.0, passos=48)  # 12 h em passos de 15 min
    resultado = simular_ilhamento(
        carga, pico, sol, LimitesDespacho.sem_bateria(), passo_min=15,
        gerador=Gerador(potencia_kw=15.0, atraso_partida_min=0.0),
    )
    assert bool(resultado.atendido[0])
    assert resultado.ens_kwh[0] == pytest.approx(0.0)
    assert resultado.energia_gerador_kwh[0] == pytest.approx(5.0 * 12.0)


def test_gerador_falha_por_potencia_e_nao_por_energia():
    """
    Carga acima da placa do grupo não é atendida por mais tanque que haja — e
    o estudo precisa dizer "potência", porque comprar combustível não resolve.
    """
    carga, pico, sol = _janela(20.0, 15.0, passos=8)
    resultado = simular_ilhamento(
        carga, pico, sol, LimitesDespacho.sem_bateria(), passo_min=15,
        gerador=Gerador(potencia_kw=15.0, atraso_partida_min=0.0),
    )
    assert not bool(resultado.atendido[0])
    assert int(resultado.causa[0]) == CAUSA_POTENCIA


def test_transferencia_custa_energia_mas_nao_reprova_o_arranjo():
    """
    Os segundos entre a queda e o grupo assumir não são gratuitos — e também
    não são defeito.

    Um grupo com QTA leva de 10 a 30 s para assumir: é dado de catálogo. Se
    esse intervalo reprovasse o cenário, todo grupo gerador do mundo teria
    autonomia zero, e a pergunta "quanto custa rodar o gerador" nunca seria
    respondida. A energia perdida entra no ENS, separada, e a autonomia
    continua de pé.
    """
    carga, pico, sol = _janela(6.0, 15.0, passos=8)  # 2 h em passos de 15 min
    resultado = simular_ilhamento(
        carga, pico, sol, LimitesDespacho.sem_bateria(), passo_min=15,
        gerador=Gerador(potencia_kw=15.0, atraso_partida_min=10.0),
    )
    assert bool(resultado.atendido[0]), "a transferência não reprova o arranjo"
    # No primeiro passo de 15 min o grupo cobre 5 dos 15 minutos: entrega 1/3
    # da placa em média, e o que falta para os 6 kW é o custo da transferência.
    assert resultado.ens_transferencia_kwh[0] == pytest.approx(1.0 * 0.25)
    assert resultado.ens_kwh[0] == pytest.approx(resultado.ens_transferencia_kwh[0])


def test_grupo_dimensionado_para_a_carga_atravessa_qualquer_duracao():
    """
    É a promessa do gerador, e ela precisa de teste.

    Dimensionado pelo pico, o grupo atravessa 36 h tão bem quanto 1 h: o que
    muda entre as duas não é a probabilidade de atender, é a conta de
    combustível.
    """
    grupo = Gerador.para_carga(20.0, custo_energia_brl_kwh=2.0, atraso_partida_min=0.0)
    assert grupo.potencia_kw >= 20.0
    for horas in (1, 12, 36):
        carga, pico, sol = _janela(20.0, grupo.potencia_kw, passos=horas * 4)
        resultado = simular_ilhamento(
            carga, pico, sol, LimitesDespacho.sem_bateria(), passo_min=15, gerador=grupo,
        )
        assert bool(resultado.atendido[0]), f"falhou em {horas} h"
        gerado = float(resultado.energia_gerador_kwh[0])
        assert gerado == pytest.approx(20.0 * horas)
        assert grupo.custo_da_energia_brl(gerado) == pytest.approx(2.0 * 20.0 * horas)


def test_potencia_comercial_arredonda_para_cima():
    """Ninguém fabrica um grupo de 137 kW: a placa vem da lista comercial."""
    # 100 kW a 0,8 de fator de potência pedem 125 kVA, que sobem para 130.
    assert Gerador.para_carga(100.0, folga=1.0).potencia_kw == pytest.approx(130 * 0.8)
    # Com 15% de folga são 143,75 kVA, que sobem para 150.
    assert Gerador.para_carga(100.0, folga=1.15).potencia_kw == pytest.approx(150 * 0.8)


def test_gerador_recarrega_o_banco_com_a_folga_de_potencia(conjunto):
    """
    É o que faz a dupla gerador + bateria atravessar dias: o grupo enche o
    banco na folga, e o banco devolve nos picos que o grupo não aguenta.
    """
    limites = LimitesDespacho.do_conjunto(conjunto)
    passos = 16
    carga = np.full((1, passos), 0.5)
    sol = np.zeros((1, passos))
    grupo = Gerador(potencia_kw=10.0, atraso_partida_min=0.0, recarrega_bateria=True)
    com = simular_ilhamento(carga, carga, sol, limites, 15, soc_inicial_frac=0.2, gerador=grupo)
    sem = simular_ilhamento(
        carga, carga, sol, limites, 15, soc_inicial_frac=0.2,
        gerador=Gerador(potencia_kw=10.0, atraso_partida_min=0.0, recarrega_bateria=False),
    )
    assert com.soc_final_frac[0] > sem.soc_final_frac[0]
    assert com.energia_recarregada_kwh[0] > 0.0
