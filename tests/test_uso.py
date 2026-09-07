"""
Testes dos três cenários de uso de uma residência.

Uma casa não tem uma curva de carga, e escolher uma delas antes de mostrar as
alternativas é justamente o que não se pode fazer: o sistema que atende um dia
de semana de casa vazia é metade do que atende uma casa cheia o ano inteiro.

O que se garante aqui é que cada cenário recebe o **seu** dimensionamento, e
que as duas grandezas não se misturam — o solar vem da energia do ano, o banco
vem do pior dia. Trocar uma pela outra é o erro clássico: solar dimensionado
pelo fim de semana fica 40% grande demais, banco dimensionado pela média não
atravessa o sábado.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aurum.bateria.apagao import MalhaApagao
from aurum.bateria.catalogo import carregar_catalogo
from aurum.bateria.geracao import serie_sintetica
from aurum.bateria.uso import comparar_cenarios_de_uso
from aurum.demanda import ocupacao
from aurum.demanda.biblioteca import COLUNAS, linha_de_planilha
from aurum.demanda.cenario import Cenario


def _casa() -> Cenario:
    """Uma casa pequena, com criticidade, para os três cenários morderem."""
    def tabela(itens):
        linhas = []
        for nome, quantidade, criticidade, ajustes in itens:
            linha = linha_de_planilha(nome, quantidade, **ajustes)
            linha["criticidade"] = criticidade
            linhas.append(linha)
        return pd.DataFrame(linhas, columns=[*COLUNAS, "criticidade"])

    cenario = Cenario(
        nome="Casa de teste", segmento="residencia",
        comodos={
            "Cozinha": tabela([
                ("Geladeira doméstica", 1, "C", {}),
                ("Forno de micro-ondas", 1, "NC",
                 {"intervalo": "18:30 as 21:30", "probabilidade": 0.8}),
            ]),
            "Sala": tabela([
                ("Nobreak / rack de rede", 1, "MC", {}),
                ('TV LED 50"', 1, "P", {"intervalo": "18:00 as 23:00"}),
                ("Notebook", 1, "NC", {"intervalo": "08:00 as 18:00"}),
            ]),
        },
        instancias={"Cozinha": 1, "Sala": 1},
    )
    cenario.criticidades_essenciais = ("MC", "C")
    return cenario


@pytest.fixture(scope="module")
def comparacao():
    return comparar_cenarios_de_uso(
        _casa(),
        serie_sintetica(-25.4, -49.3, 0.0, 20.0, 14.0, anos=(2020, 2020)),
        carregar_catalogo(),
        malha=MalhaApagao(duracoes_h=(1.0, 2.0, 4.0, 6.0), amostras=30),
        autonomia_alvo_h=4.0, simulacoes=50,
    )


# ----------------------------------------------------------------------------
# Os três cenários
# ----------------------------------------------------------------------------
def test_os_tres_cenarios_aparecem(comparacao):
    assert len(comparacao.cenarios) == 3
    assert {c.chave for c in comparacao.cenarios} == set(ocupacao.ESTUDOS)


def test_o_pouco_uso_e_o_piso(comparacao):
    """
    Os três são níveis, e níveis se ordenam. Se não se ordenassem, não seriam
    níveis — seriam três leituras avulsas, e a comparação não diria nada.
    """
    piso = comparacao.piso
    assert piso is not None and piso.chave == "pouco_uso"
    for outro in comparacao.cenarios:
        assert outro.consumo_anual_kwh >= piso.consumo_anual_kwh * 0.999


def test_o_muito_uso_e_o_teto(comparacao):
    teto = comparacao.teto
    assert teto is not None and teto.chave == "muito_uso"


def test_o_do_meio_e_a_rotina_de_quem_trabalha_fora(comparacao):
    """
    O uso comum fica entre os outros dois — o que **não** faz dele a resposta.

    O estudo apresenta os três e não escolhe: quem conhece a casa é o cliente,
    e escolher por ele transformaria uma premissa em conclusão.
    """
    meio = comparacao.intermediario
    assert meio is not None and meio.chave == "uso_comum"
    assert comparacao.piso.consumo_anual_kwh < meio.consumo_anual_kwh
    assert meio.consumo_anual_kwh < comparacao.teto.consumo_anual_kwh


# ----------------------------------------------------------------------------
# O dimensionamento de cada um
# ----------------------------------------------------------------------------
def test_o_solar_sai_da_energia_do_ano_de_cada_cenario(comparacao):
    """Mais consumo, mais kWp — e na mesma proporção, porque é uma divisão."""
    piso, teto = comparacao.piso, comparacao.teto
    assert teto.potencia_fv_kwp > piso.potencia_fv_kwp
    razao_consumo = teto.consumo_anual_kwh / piso.consumo_anual_kwh
    razao_kwp = teto.potencia_fv_kwp / piso.potencia_fv_kwp
    assert razao_kwp == pytest.approx(razao_consumo, rel=1e-6)


def test_o_banco_sai_do_pior_dia_e_nao_da_media(comparacao):
    """
    O cenário de fim de semana é dimensionado pelo dia cheio, e não pela média.

    Banco escolhido pela média não atravessa o sábado — e o sábado é o dia em
    que a família está em casa para notar.
    """
    meio = comparacao.intermediario
    teto = comparacao.teto
    assert meio.dimensionante == teto.dimensionante == "Casa cheia"
    assert meio.pico_p95_kw == pytest.approx(teto.pico_p95_kw, rel=1e-9)


def test_a_curva_e_a_media_ponderada_e_nao_a_do_dimensionante(comparacao):
    """
    O defeito que saltava aos olhos no gráfico: com a curva do perfil que
    dimensiona, o cenário de fim de semana saía **idêntico** ao de casa cheia,
    porque os dois são dimensionados pelo mesmo dia. A energia da curva tem de
    bater com a energia declarada do cenário.
    """
    meio, teto = comparacao.intermediario, comparacao.teto
    assert not np.allclose(meio.curva_w, teto.curva_w)
    # A energia sob a curva bate com o consumo diário declarado.
    for cenario in comparacao.cenarios:
        horas = cenario.passo_min / 60.0
        energia = float(np.sum(cenario.curva_w) * horas / 1000.0)
        assert energia == pytest.approx(cenario.energia_diaria_kwh, rel=0.05), cenario.nome


def test_cada_cenario_tem_o_seu_investimento(comparacao):
    """Comparar um sistema só sob três consumos responderia outra pergunta."""
    for cenario in comparacao.cenarios:
        assert cenario.capex_sem_bateria_brl > 0, cenario.nome
        assert cenario.capex_com_bateria_brl >= cenario.capex_sem_bateria_brl
    assert comparacao.teto.capex_com_bateria_brl > comparacao.piso.capex_com_bateria_brl


def test_a_amplitude_mede_o_quanto_a_escolha_importa(comparacao):
    amplitude = comparacao.amplitude()
    assert amplitude["consumo"] > 1.0
    assert amplitude["solar_kwp"] == pytest.approx(amplitude["consumo"], rel=1e-6)
    # O investimento varia menos que o consumo: o banco quase não muda entre os
    # cenários, porque o quadro de backup é refrigeração e rede.
    assert amplitude["capex"] < amplitude["consumo"]


def test_a_tabela_sai_pronta_para_o_relatorio(comparacao):
    tabela = comparacao.tabela()
    assert len(tabela) == 3
    for coluna in ("cenario", "consumo_kwh_dia", "solar_kwp", "banco_kwh",
                   "capex_com_bateria_brl"):
        assert coluna in tabela.columns


# ----------------------------------------------------------------------------
# O perfil de pouco uso
# ----------------------------------------------------------------------------
def test_o_dia_de_pouco_uso_ainda_almoca_e_janta():
    """
    Almoço e jantar acontecem em qualquer dia — mudam de tamanho, não de
    existência.

    A vistoria costuma anotar só o jantar, e sem o desdobramento o dia de
    semana saía com um vale ao meio-dia que nenhuma casa tem — justamente na
    hora em que o sol está no máximo, que é onde o autoconsumo se decide.
    """
    perfil = ocupacao.PERFIS["casa_quase_vazia"]
    assert perfil.desdobrar_refeicoes, "o dia de pouco uso também tem refeições"
    assert 0.0 < perfil.fator_refeicao < 1.0, "menores que as da casa cheia, e não nulas"

    ajustado = ocupacao.aplicar(_casa(), perfil)
    micro = ajustado.comodos["Cozinha"].iloc[1]["intervalo"]
    assert " e " in micro, "duas refeições, duas janelas"
    assert micro.startswith(ocupacao._hhmm(ocupacao._ALMOCO[0]))
