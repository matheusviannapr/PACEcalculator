"""
O estresse dos três níveis de uso contra a curva residencial de referência.

Os três cenários são construídos por transformação — janelas reescritas,
refeições desdobradas, madrugada arrastada — e transformação sem verificação é
palpite com casas decimais. Este arquivo é a verificação: simula os três sobre
uma casa real e confere a distribuição da energia ao longo do dia contra o
perfil ``residencial`` da base.

O que ele pegou, na ordem em que apareceu:

* **Pico ao meio-dia.** Espalhar toda carga de presença pela janela acordada
  apagava a concentração natural do jantar, e a casa aparecia com o pico às
  12 h contra as 20 h da referência. Espalhar passou a valer só para a janela
  que cai dentro do expediente, que é a suspeita de vir do formulário.
* **Madrugada em 1%, contra 12% da referência.** O formulário pede uma janela
  por equipamento, o morador responde "das 18 às 23", e o ar-condicionado some
  às 23:00 em ponto. Nenhum ar-condicionado faz isso.
* **Refeição cortada mais que o uso geral.** Quem fica em casa sozinho deixa
  de usar metade da casa e não deixa de almoçar; o fator de refeição tinha de
  ficar acima do diurno, e estava abaixo.

O limite é o mesmo dos demais segmentos da biblioteca. Ele é folgado de
propósito: a curva de referência é sintética, e esta é uma casa de alto padrão
no Rio, com climatização noturna que a média nacional não tem. Apertar mais
seria ajustar um modelo físico a um valor de referência provisório.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aurum.demanda import ocupacao, simular_ensemble
from aurum.demanda.biblioteca import (
    COLUNAS,
    LIMITE_DIVERGENCIA_PP,
    comparar_com_perfil,
    linha_de_planilha,
    perfil_tipico,
)
from aurum.demanda.cenario import Cenario

#: Simulações por estação. Baixo de propósito: o que se mede aqui é a
#: distribuição da energia entre quatro períodos do dia, e ela converge muito
#: antes da cauda que dimensiona o inversor.
SIMULACOES = 120


def _casa_alto_padrao() -> Cenario:
    """
    Uma casa parecida com o caso real que originou estes testes.

    Refrigeração contínua, chuveiros de manhã, cozinha à noite, climatização
    nos quartos e a janela de escritório que o formulário insiste em sugerir —
    é essa última que a casa cheia tem o direito de reescrever.
    """
    def tabela(itens):
        return pd.DataFrame(
            [linha_de_planilha(nome, quantidade, **ajustes)
             for nome, quantidade, ajustes in itens],
            columns=COLUNAS,
        )

    return Cenario(
        nome="Casa de estresse", segmento="residencia",
        comodos={
            "Cozinha": tabela([
                ("Geladeira doméstica", 1, {}),
                ("Freezer horizontal", 1, {}),
                ("Forno de micro-ondas", 1,
                 {"intervalo": "18:30 as 21:30", "probabilidade": 0.85}),
                ("Coifa / exaustão de cozinha", 1,
                 {"intervalo": "18:30 as 21:30", "probabilidade": 0.6}),
                ("Lâmpada LED bulbo 9 W", 4, {"intervalo": "18:00 as 22:30"}),
            ]),
            "Quarto": tabela([
                ("Ar-condicionado split 9.000 BTU", 1,
                 {"intervalo": "20:00 as 23:30", "duracao_min": 3.0,
                  "duracao_max": 6.0, "probabilidade": 0.7}),
                ("Lâmpada LED bulbo 9 W", 2, {"intervalo": "18:30 as 23:00"}),
            ]),
            # Sem estes, a casa vira três quartos com ar e mais nada, e a
            # climatização noturna passa a responder por metade da energia —
            # o que reprovava o próprio teste por defeito do exemplo, e não do
            # modelo. Uma casa real tem sala, área de serviço e área externa.
            "Sala": tabela([
                ('TV LED 50"', 1, {"intervalo": "18:00 as 23:30"}),
                ("Lâmpada LED bulbo 9 W", 6, {"intervalo": "18:00 as 23:00"}),
                ("Ventilador de teto", 1,
                 {"intervalo": "12:00 as 22:00", "probabilidade": 0.5}),
                ("Sistema de segurança / CFTV", 1, {}),
            ]),
            "Área de serviço": tabela([
                ("Máquina de lavar doméstica", 1,
                 {"intervalo": "08:00 as 19:00", "duracao_min": 0.8,
                  "duracao_max": 1.5, "probabilidade": 0.6}),
                ("Lâmpada LED bulbo 9 W", 2, {"intervalo": "17:30 as 21:00"}),
            ]),
            "Área externa": tabela([
                ("Bomba d'água 1 CV", 1, {}),
                ("Bomba de piscina", 1,
                 {"intervalo": "09:00 as 16:00", "duracao_min": 3.0,
                  "duracao_max": 5.0, "probabilidade": 0.8}),
                ("Luminária LED alta potência 150 W", 2,
                 {"intervalo": "18:00 as 06:00"}),
            ]),
            "Banheiro": tabela([
                ("Chuveiro elétrico 5.500 W", 1,
                 {"intervalo": "06:00 as 09:00", "duracao_min": 0.12,
                  "duracao_max": 0.25, "probabilidade": 0.9}),
            ]),
            "Escritório": tabela([
                # A janela que o formulário sugere e que ninguém cumpre em casa.
                ("Notebook", 1, {"intervalo": "08:00 as 18:00", "probabilidade": 0.7}),
                ("Nobreak / rack de rede", 1, {}),
            ]),
        },
        instancias={"Cozinha": 1, "Quarto": 3, "Banheiro": 2, "Escritório": 1,
                    "Sala": 1, "Área de serviço": 1, "Área externa": 1},
    )


def _curva_do_cenario(chave: str, medidos: dict) -> np.ndarray:
    """A curva média ponderada dos perfis que compõem o cenário."""
    pesos = ocupacao.ESTUDOS[chave]["perfis"]
    dias = sum(pesos.values())
    curvas = [np.asarray(medidos[p]["curva"], dtype=float) for p in pesos]
    tamanho = min(len(c) for c in curvas)
    return sum(c[:tamanho] * n for c, n in zip(curvas, pesos.values())) / dias


@pytest.fixture(scope="module")
def medidos() -> dict:
    casa = _casa_alto_padrao()
    resultado = {}
    for identificador in {p for e in ocupacao.ESTUDOS.values() for p in e["perfis"]}:
        ajustado = ocupacao.aplicar(casa, identificador)
        ensemble = simular_ensemble(
            ajustado.para_comodos(), ajustado.instancias_de(), SIMULACOES)
        suave = ensemble.reamostrar(5) if ensemble.passo_min == 1 else ensemble
        resultado[identificador] = {
            "curva": suave.perfil_medio_w(),
            "resumo": ensemble.resumo()["geral"],
        }
    return resultado


# ----------------------------------------------------------------------------
def test_os_tres_niveis_batem_com_a_curva_residencial(medidos):
    """
    A distribuição da energia ao longo do dia, nos três, contra a referência.

    É o teste que pegou o pico ao meio-dia e a madrugada vazia — dois defeitos
    que não apareciam em número nenhum do estudo, só no formato da curva.
    """
    referencia = perfil_tipico("residencial")
    for chave in ocupacao.ORDEM_DOS_ESTUDOS:
        comparacao = comparar_com_perfil(_curva_do_cenario(chave, medidos), referencia, 5)
        assert comparacao["coerente"], (
            f"{ocupacao.ESTUDOS[chave]['nome']}: "
            f"{comparacao['divergencia_do_pior_periodo_pp']:.1f} pp de desvio na "
            f"{comparacao['pior_periodo']} (limite {LIMITE_DIVERGENCIA_PP} pp)"
        )


def test_o_pico_da_casa_e_o_jantar_nos_tres(medidos):
    """
    Numa casa brasileira o pico é à noite, e nos três níveis.

    Quando a casa cheia espalhava toda carga de presença pela janela acordada,
    o cume do almoço passava o do jantar e o pico ia para as 12 h — plausível
    à primeira vista, e errado.
    """
    referencia = perfil_tipico("residencial")
    for chave in ocupacao.ORDEM_DOS_ESTUDOS:
        comparacao = comparar_com_perfil(_curva_do_cenario(chave, medidos), referencia, 5)
        assert 18 <= comparacao["hora_de_pico_simulada"] <= 22, (
            f"{ocupacao.ESTUDOS[chave]['nome']}: pico às "
            f"{comparacao['hora_de_pico_simulada']} h")


def test_a_madrugada_nao_fica_vazia(medidos):
    """
    O ar-condicionado não desliga às 23:00 em ponto porque o formulário
    terminou ali.

    Antes da regra de arrasto noturno, a simulação punha 1% da energia na
    madrugada contra 12% da referência — e a diferença não era da casa, era do
    formulário.
    """
    for chave in ocupacao.ORDEM_DOS_ESTUDOS:
        curva = _curva_do_cenario(chave, medidos)
        passos_por_hora = len(curva) / 24
        madrugada = curva[: int(5 * passos_por_hora)].sum() / curva.sum()
        assert madrugada > 0.04, (
            f"{ocupacao.ESTUDOS[chave]['nome']}: só {madrugada:.1%} da energia na "
            "madrugada — o que estava ligado na hora de dormir sumiu")


def test_os_tres_niveis_se_ordenam(medidos):
    """Três níveis que não se ordenam não são níveis."""
    consumos = []
    for chave in ocupacao.ORDEM_DOS_ESTUDOS:
        pesos = ocupacao.ESTUDOS[chave]["perfis"]
        dias = sum(pesos.values())
        consumos.append(
            sum(medidos[p]["resumo"]["energia_diaria_media_kwh"] * n
                for p, n in pesos.items()) / dias
        )
    assert consumos == sorted(consumos), dict(zip(ocupacao.ORDEM_DOS_ESTUDOS, consumos))


def test_a_refeicao_e_menos_cortada_que_o_uso_geral():
    """
    Quem fica em casa sozinho deixa de usar metade da casa e não deixa de
    almoçar.

    O primeiro valor arbitrado cortava a refeição **mais** que o uso geral do
    dia, o que nenhuma casa faz — e o teste do jantar foi quem apontou.
    """
    perfil = ocupacao.PERFIS["casa_quase_vazia"]
    assert perfil.fator_refeicao > perfil.fator_diurno
    assert perfil.fator_refeicao < 1.0, "menor que a da casa cheia, ainda assim"
