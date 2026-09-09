"""
Testes da comparação entre metas de autonomia.

A meta é a premissa mais silenciosa de um estudo de armazenamento: alguém
escreve 6 h no começo e o documento inteiro sai dali. O que se garante aqui é
que as duas metas são medidas com o mesmo rigor, que o degrau entre elas é o
número que a conversa comercial pede, e que a saturação da malha aparece como
ressalva em vez de virar promessa.
"""
from __future__ import annotations

import numpy as np
import pytest

from aurum.bateria.apagao import MalhaApagao
from aurum.bateria.autonomia import comparar_metas_de_autonomia
from aurum.bateria.catalogo import candidatos_em_blocos, carregar_catalogo
from aurum.bateria.economia import PremissasBateria
from aurum.bateria.geracao import serie_sintetica
import pandas as pd

from aurum.demanda import simular_ensemble
from aurum.demanda.biblioteca import COLUNAS, linha_de_planilha
from aurum.demanda.cenario import Cenario

MALHA = MalhaApagao(duracoes_h=(1.0, 3.0, 6.0, 12.0, 24.0), amostras=40)


def _casa() -> Cenario:
    """
    Uma residência, e não um hotel.

    A carga de backup de uma casa é pequena por construção — geladeira, rede,
    bomba —, e é aí que a comparação entre 6 h e 24 h tem resposta. Com a carga
    de um hotel nenhum banco do catálogo atravessa 24 h, e o teste que deveria
    guardar a comparação simplesmente pula.
    """
    tabela = pd.DataFrame(
        [linha_de_planilha(nome, quantidade, criticidade=crit)
         for nome, quantidade, crit in (
             ("Geladeira doméstica", 1, "C"),
             ("Nobreak / rack de rede", 1, "MC"),
         )],
        columns=COLUNAS,
    )
    cenario = Cenario(
        nome="Casa de teste", segmento="residencia",
        comodos={"Casa": tabela}, instancias={"Casa": 1},
    )
    cenario.criticidades_essenciais = ("MC", "C")
    return cenario


@pytest.fixture(scope="module")
def cenario():
    casa = _casa()
    ensemble = simular_ensemble(
        casa.para_comodos(True), casa.instancias_de(True), 40, semente=7,
    ).reamostrar(MALHA.passo_min)
    serie = serie_sintetica(-22.9, -43.2, 0.0, 20.0, 14.0, anos=(2020, 2020))
    candidatos = candidatos_em_blocos(carregar_catalogo())
    return candidatos, ensemble, serie


@pytest.fixture(scope="module")
def comparacao(cenario):
    candidatos, ensemble, serie = cenario
    return comparar_metas_de_autonomia(
        candidatos, ensemble, serie, MALHA,
        metas_h=(6.0, 24.0),
        premissas=PremissasBateria(bloco_bateria_kwh=5.0, bloco_bateria_brl=12_000.0),
        max_candidatos=6,
    )


def test_a_meta_maior_nunca_custa_menos(comparacao):
    """
    Uma inversão aqui seria erro de seleção, não resultado.

    Se o banco que atravessa 24 h aparecesse mais barato que o de 6 h, seria
    porque a busca escolheu por energia em vez de preço — o mesmo erro que já
    devolveu 189 kWh e R$ 872 mil para uma residência.
    """
    seis, vinte_quatro = comparacao.para(6.0), comparacao.para(24.0)
    if seis is None or vinte_quatro is None:
        pytest.skip("o catálogo não alcança uma das metas neste cenário")
    assert vinte_quatro.capex_brl >= seis.capex_brl
    assert vinte_quatro.conjunto.energia_util_kwh >= seis.conjunto.energia_util_kwh


def test_cada_meta_recebe_o_mais_barato_que_a_cumpre(comparacao):
    """
    Ordenar por preço, e não por energia, é o que equilibra banco e inversor.

    O preço carrega as duas dimensões: módulos e inversor entram na mesma conta,
    e o mais barato que cumpre é justamente o que as equilibra.
    """
    for meta in comparacao.metas_h:
        escolhido = comparacao.para(meta)
        if escolhido is None:
            continue
        cumprem = [m for m in comparacao.medidas if m.autonomia_h >= meta]
        assert escolhido.capex_brl == min(m.capex_brl for m in cumprem)


def test_o_degrau_e_o_numero_da_conversa(comparacao):
    """
    Quanto custa subir de meta, em blocos e em reais.

    Ele quase nunca é proporcional: a autonomia cresce em degraus de bloco, e um
    bloco a mais pode dobrar a autonomia ou não mover nada, conforme onde a meta
    cai entre dois degraus. É por isso que o número tem de ser medido e não
    estimado por regra de três.
    """
    degrau = comparacao.degrau(6.0, 24.0)
    if degrau is None:
        pytest.skip("o catálogo não alcança uma das metas neste cenário")
    assert degrau["blocos_a_mais"] >= 0
    assert degrau["capex_a_mais_brl"] >= 0
    seis = comparacao.para(6.0)
    vinte_quatro = comparacao.para(24.0)
    assert degrau["capex_a_mais_brl"] == pytest.approx(
        vinte_quatro.capex_brl - seis.capex_brl)


def test_a_saturacao_da_malha_vira_ressalva(comparacao):
    """
    Um banco que aparece com 24 h pode ir além, e a malha não sabe dizer quanto.

    Para a decisão isso basta — ninguém compra bateria para 17 h. Para a leitura
    não basta, e calar seria transformar o limite do método em promessa.
    """
    assert any("satura" in a for a in comparacao.avisos)


def test_meta_inalcancavel_e_dita_e_nao_arredondada(cenario):
    """
    "Nenhum banco chega lá" é resposta, e uma resposta útil.

    Devolver o maior banco disponível como se ele cumprisse a meta entregaria um
    sistema que o cliente descobre insuficiente no primeiro temporal.
    """
    candidatos, ensemble, serie = cenario
    comparacao = comparar_metas_de_autonomia(
        candidatos, ensemble, serie, MALHA,
        metas_h=(240.0,),
        premissas=PremissasBateria(bloco_bateria_kwh=5.0, bloco_bateria_brl=12_000.0),
        max_candidatos=3,
    )
    assert comparacao.para(240.0) is None
    assert any("Nenhum banco" in a for a in comparacao.avisos)
    linha = comparacao.tabela().iloc[0]
    assert linha["atendida"] is False or not bool(linha["atendida"])


def test_a_autonomia_e_do_pior_par_estacao_hora(cenario):
    """
    Medir na média é sempre mais generoso e não descreve apagão nenhum.

    O apagão acontece numa hora e numa estação, e é contra essa hora que o banco
    precisa ter sido comprado.
    """
    candidatos, ensemble, serie = cenario
    premissas = PremissasBateria(bloco_bateria_kwh=5.0, bloco_bateria_brl=12_000.0)
    rigorosa = comparar_metas_de_autonomia(
        candidatos, ensemble, serie, MALHA, metas_h=(6.0,),
        premissas=premissas, exigir_pior_caso=True, max_candidatos=4)
    media = comparar_metas_de_autonomia(
        candidatos, ensemble, serie, MALHA, metas_h=(6.0,),
        premissas=premissas, exigir_pior_caso=False, max_candidatos=4)

    for a, b in zip(rigorosa.medidas, media.medidas):
        assert a.autonomia_h <= b.autonomia_h, (
            "o pior par não pode ser mais generoso que a média"
        )
