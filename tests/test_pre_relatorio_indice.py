"""
Índice de oportunidade: a régua de 0 a 100 do pré-relatório.

Os pesos são arbitrados (ver o cabeçalho de aurum.pre_relatorio.indice); estes
testes travam o que não pode mudar sem querer: os tetos, a ordem dos casos
óbvios e os limites.
"""
import pytest

from aurum.pre_relatorio.indice import (TETO_ECONOMIA, TETO_ENCAIXE, TETO_FALTA, TETO_NOITE,
                                         calcular, faixa)


def test_tetos_somam_cem():
    assert TETO_ECONOMIA + TETO_FALTA + TETO_NOITE + TETO_ENCAIXE == 100


def test_caso_maximo_e_minimo():
    tudo = calcular(4000, 3600, "semanal", "dia", "grave", 0.95, 1.0)
    nada = calcular(150, None, "rara", "minutos", "nenhum", 0.20, None)
    assert tudo.valor == 100 and tudo.faixa == "muito alta"
    assert nada.valor < 20 and nada.faixa == "baixa"


def test_ordem_dos_casos_obvios():
    base = dict(conta_atual_brl=800, economia_mensal_brl=600, frequencia="mensal", duracao="horas",
                nivel_critico="critico", fracao_sem_sol=0.6, cobertura_da_falta=0.9)
    v = calcular(**base).valor
    assert calcular(**{**base, "nivel_critico": "grave"}).valor > v           # algo não pode cair
    assert calcular(**{**base, "frequencia": "semanal"}).valor > v           # falta mais vezes
    assert calcular(**{**base, "economia_mensal_brl": 200}).valor < v        # economia menor
    assert calcular(**{**base, "cobertura_da_falta": 0.3}).valor < v         # bateria não segura
    assert calcular(**{**base, "fracao_sem_sol": 0.3}).valor < v             # consumo de dia


def test_conta_grande_vale_mais_em_reais():
    # Mesma fração da conta (75%), valores diferentes: o negócio grande pontua mais.
    pequeno = calcular(400, 300, "mensal", "horas", "critico", 0.6, 0.9)
    grande = calcular(4000, 3000, "mensal", "horas", "critico", 0.6, 0.9)
    assert grande.partes[0].pontos > pequeno.partes[0].pontos


@pytest.mark.parametrize("valor,esperado", [(0, "baixa"), (39, "baixa"), (40, "moderada"),
                                            (60, "alta"), (79, "alta"), (80, "muito alta")])
def test_faixas(valor, esperado):
    assert faixa(valor) == esperado
