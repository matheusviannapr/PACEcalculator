"""
Testes do quadro de backup em dois níveis.

A vistoria classifica em MC, C, P e NC. O estudo usava só a linha de corte, e
a pergunta que sobrava era comercial: quanto custa levar junto o que seria bom
ter? O que se garante aqui é que os dois quadros são medidos **separados** —
porque a coincidência entre as cargas não é aditiva — e que a escolha do banco
não cai nas duas armadilhas que o caso real revelou.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aurum.bateria.apagao import MalhaApagao
from aurum.bateria.catalogo import carregar_catalogo
from aurum.bateria.escopos import ESCOPOS, comparar_escopos
from aurum.bateria.geracao import serie_sintetica
from aurum.demanda.biblioteca import COLUNAS, linha_de_planilha


def _tabela(itens) -> pd.DataFrame:
    """``(equipamento, quantidade, criticidade, ajustes)`` → tabela do cenário."""
    # A criticidade vai como ajuste, e não como coluna acrescentada: ela passou
    # a fazer parte de COLUNAS, e somá-la de novo criava uma coluna duplicada —
    # que o pandas devolve como Series e derruba tudo adiante.
    return pd.DataFrame(
        [linha_de_planilha(nome, quantidade, criticidade=criticidade, **ajustes)
         for nome, quantidade, criticidade, ajustes in itens],
        columns=COLUNAS,
    )


@pytest.fixture(scope="module")
def casa() -> dict[str, pd.DataFrame]:
    """
    Uma casa pequena com os quatro níveis representados.

    A geladeira é crítica; a TV e o ar são preferíveis — conforto, e é
    exatamente sobre eles que a pergunta comercial recai; o chuveiro de
    5.500 W é não crítico e nunca deve aparecer em quadro nenhum.
    """
    return {
        "Cozinha": _tabela([
            ("Geladeira doméstica", 1, "C", {}),
            ("Forno de micro-ondas", 1, "NC", {}),
        ]),
        "Sala": _tabela([
            ("Nobreak / rack de rede", 1, "MC", {}),
            ('TV LED 50"', 1, "P", {"intervalo": "18:00 as 23:00"}),
            ("Ar-condicionado split 12.000 BTU", 1, "P",
             {"intervalo": "19:00 as 23:30", "duracao_min": 2.0, "duracao_max": 4.0}),
        ]),
        "Banheiro": _tabela([
            ("Chuveiro elétrico 5.500 W", 1, "NC", {}),
        ]),
    }


@pytest.fixture(scope="module")
def comparacao(casa):
    catalogo = carregar_catalogo()
    serie = serie_sintetica(-25.4, -49.3, 0.0, 20.0, 14.0, anos=(2020, 2020))
    return comparar_escopos(
        casa, {"Cozinha": 1, "Sala": 1, "Banheiro": 1},
        candidatos=[], serie=serie,
        malha=MalhaApagao(duracoes_h=(1.0, 2.0, 4.0, 6.0), amostras=40),
        autonomia_alvo_h=4.0, simulacoes=60, catalogo=catalogo,
    )


# ----------------------------------------------------------------------------
# O recorte
# ----------------------------------------------------------------------------
def test_o_nao_critico_nunca_entra_em_quadro_nenhum(comparacao):
    """
    O chuveiro de 5.500 W é NC, e é ele que decide se o inversor é grande.

    Um NC que vaza para o quadro ampliado multiplicaria o pico e mataria a
    comparação — o quadro ampliado passaria a pedir um inversor que a casa não
    precisa, e o preço marginal viraria ficção.
    """
    base, ampliado = comparacao.base, comparacao.ampliado
    assert base.equipamentos == 2, "geladeira (C) e rack (MC)"
    assert ampliado.equipamentos == 4, "os dois acima, mais TV e ar (P)"
    # 5.500 W do chuveiro fora dos dois quadros.
    assert ampliado.potencia_instalada_w < 5000.0


def test_os_dois_escopos_sao_medidos_separados(comparacao):
    """
    Cada quadro tem a sua simulação, e não uma soma da outra.

    A coincidência entre as duas cargas não é aditiva: o pico do conjunto é
    menor que a soma dos picos, e estimar o ampliado somando o essencial ao
    preferível superestimaria o inversor.
    """
    base, ampliado = comparacao.base, comparacao.ampliado
    assert ampliado.energia_diaria_kwh > base.energia_diaria_kwh
    assert ampliado.pico_p95_kw > base.pico_p95_kw
    assert len(base.curva_w) == len(ampliado.curva_w)


def test_sem_preferiveis_nao_ha_o_que_comparar():
    """
    Duas linhas iguais são piores que nenhuma: sugerem uma escolha inexistente.
    """
    sem_p = {
        "Cozinha": _tabela([
            ("Geladeira doméstica", 1, "C", {}),
            ("Forno de micro-ondas", 1, "NC", {}),
        ]),
    }
    catalogo = carregar_catalogo()
    comparacao = comparar_escopos(
        sem_p, {"Cozinha": 1}, [], serie_sintetica(-25.4, -49.3, 0.0, 20.0, 14.0, anos=(2020, 2020)),
        MalhaApagao(duracoes_h=(1.0, 4.0), amostras=20),
        autonomia_alvo_h=4.0, simulacoes=40, catalogo=catalogo,
    )
    assert not comparacao.tem_preferiveis


# ----------------------------------------------------------------------------
# A escolha do banco
# ----------------------------------------------------------------------------
def test_o_banco_escolhido_e_o_mais_barato_que_cumpre(comparacao):
    """
    O primeiro critério tentado — menor por energia — devolvia absurdo.

    Quando o quadro ampliado falha por potência, todo banco pequeno falha por
    mais energia que tenha, e o primeiro a passar numa varredura por energia
    crescente é um banco enorme que veio junto de um inversor grande. No caso
    real isso deu 189 kWh e R$ 872 mil para uma residência. Ordenar por preço
    resolve porque o preço carrega as duas dimensões.
    """
    for medida in comparacao.medidas:
        assert medida.conjunto is not None, medida.escopo.nome
        assert medida.capex_brl > 0
    # O ampliado nunca é mais barato, e não é uma ordem de grandeza maior.
    # **Igual é o caso comum**: com o banco contado em blocos de 5 kWh e 5 kW,
    # os dois quadros de uma residência costumam caber no mesmo bloco, e aí
    # levar os preferíveis não custa nada. Antes, com o banco saindo da
    # varredura do catálogo, o essencial cabia num módulo de 2,2 kWh com
    # 1,28 kW e o ampliado exigia outro equipamento.
    base, ampliado = comparacao.base, comparacao.ampliado
    assert ampliado.capex_brl >= base.capex_brl
    assert ampliado.capex_brl < base.capex_brl * 10


def test_cada_escopo_monta_a_propria_lista_de_candidatos(comparacao):
    """
    Reaproveitar a lista do estudo deixava o ampliado sem candidato na faixa.

    A lista do estudo é ancorada na energia do quadro essencial; o ampliado
    precisa de bancos que não estão nela. Com `catalogo` passado, cada escopo
    ancora a busca na energia que ele mesmo exige — e é por isso que o
    resultado abaixo é um banco intermediário, e não o único grande que
    sobrava.
    """
    base, ampliado = comparacao.base, comparacao.ampliado
    assert ampliado.energia_util_kwh >= base.energia_util_kwh
    assert ampliado.autonomia_h > 0


# ----------------------------------------------------------------------------
# O que a comparação responde
# ----------------------------------------------------------------------------
def test_o_marginal_e_o_numero_que_decide(comparacao):
    """
    Não é o total do orçamento: o quadro essencial já foi aprovado antes.
    """
    marginal = comparacao.marginal()
    assert marginal["energia_diaria_kwh"] > 0
    # Zero é resposta legítima, e é a mais interessante: significa que o bloco
    # que o quadro essencial já exige carrega também os preferíveis.
    assert marginal["capex_brl"] >= 0
    assert marginal["capex_por_kwh_dia"] == pytest.approx(
        marginal["capex_brl"] / marginal["energia_diaria_kwh"])


def test_o_bloco_padrao_costuma_zerar_o_custo_de_levar_o_desejavel(comparacao):
    """
    O achado que o bloco de 5 kWh / 5 kW trouxe.

    Enquanto o banco saía da varredura do catálogo, o quadro essencial de uma
    residência cabia num módulo de 2,2 kWh com 1,28 kW de descarga, e o
    ampliado — com pico de 1,75 kW — falhava por potência e exigia outro
    equipamento. Com o bloco padrão, a descarga é 5 kW e a restrição some: os
    dois quadros cabem no mesmo bloco.

    Isso muda a conversa comercial. Não se está pedindo ao cliente que pague
    mais para levar o desejável; está-se pedindo que ele decida agora quais
    circuitos entram no quadro de backup, porque refazer isso depois custa.
    """
    base, ampliado = comparacao.base, comparacao.ampliado
    if base.conjunto is None or ampliado.conjunto is None:
        pytest.skip("sem banco dimensionado nos dois escopos")
    if base.conjunto.modulos != ampliado.conjunto.modulos:
        pytest.skip("os quadros exigiram números de blocos diferentes")
    assert comparacao.marginal()["capex_brl"] == pytest.approx(0.0)


def test_a_ociosidade_diz_o_que_ja_esta_pago(comparacao):
    """Banco vem em degraus, e a folga do degrau já foi comprada."""
    ociosidade = comparacao.ociosidade()
    assert ociosidade["energia_util_kwh"] > 0
    assert ociosidade["folga_kwh"] >= 0
    assert 0.0 <= ociosidade["fracao_da_autonomia_prometida"] <= 1.5


def test_o_limitante_separa_potencia_de_energia(comparacao):
    """
    A distinção decide o que comprar, e a autonomia sozinha não a responde.

    Faltar energia se resolve com módulo de bateria no mesmo inversor — barato
    e linear. Faltar potência exige inversor maior: outro equipamento, outro
    preço, e nenhum módulo extra resolve.
    """
    assert comparacao.limitante in {"potencia", "energia", "nenhum", "indeterminado"}
    if comparacao.limitante == "potencia":
        base, ampliado = comparacao.base, comparacao.ampliado
        assert ampliado.pico_p95_kw > base.conjunto.potencia_descarga_kw


def test_a_tabela_sai_pronta_para_o_relatorio(comparacao):
    tabela = comparacao.tabela()
    assert len(tabela) == len(ESCOPOS)
    for coluna in ("escopo", "niveis", "equipamentos", "energia_diaria_kwh",
                   "pico_p95_kw", "energia_util_kwh", "capex_brl"):
        assert coluna in tabela.columns


# ----------------------------------------------------------------------------
# Que tabelas chegam aqui
# ----------------------------------------------------------------------------


def test_os_escopos_recebem_o_dia_que_dimensiona():
    """
    Equipamento sai do pior dia — inclusive o que os escopos escolhem.

    `tabelas_cenario` serve a dois consumidores com necessidades opostas: os
    cenários de uso precisam do levantamento cru, porque aplicam cada perfil de
    ocupação por cima; os escopos escolhem banco, e medidos no levantamento cru
    dimensionam contra uma carga mais leve que a do resto do estudo. Num caso
    real a recomendação vinha com 9,3 kWh e a seção de escopos dizia que o mesmo
    recorte, com o mesmo alvo, cabe em 4,6 kWh — dois bancos no mesmo documento.
    """
    import inspect

    from aurum.bateria.estudo import ConfiguracaoEstudo, executar_estudo

    padrao = ConfiguracaoEstudo(latitude=-22.9, longitude=-43.2)
    assert padrao.tabelas_dimensionantes is None, (
        "o padrão tem de preservar o comportamento de todo estudo existente"
    )

    fonte = inspect.getsource(executar_estudo)
    chamada = fonte[fonte.index("_comparar_escopos("):]
    chamada = chamada[:chamada.index(")\n")]
    assert "cfg.tabelas_dimensionantes or cfg.tabelas_cenario" in chamada, (
        "os escopos têm de receber o dia que dimensiona, quando declarado"
    )


def test_cada_argumento_novo_cai_na_chamada_certa():
    """
    O fio, e não o número: duas vezes um argumento caiu na chamada errada.

    `ajustes_sazonais=cfg.ajustes_sazonais` aparece nas chamadas de escopos e de
    cenários de uso, e ancorar uma edição nele acerta a primeira. Foi assim que
    a tarifa foi parar em escopos uma vez, e `considerar_solar` outra — este
    último derrubou a comparação inteira com `unexpected keyword argument`.

    `considerar_solar` pertence aos cenários de uso, que dimensionam solar;
    escopos não dimensiona solar nenhum e não conhece o argumento.
    """
    import inspect

    from aurum.bateria.escopos import comparar_escopos
    from aurum.bateria.estudo import executar_estudo
    from aurum.bateria.uso import comparar_cenarios_de_uso

    assert "considerar_solar" in inspect.signature(comparar_cenarios_de_uso).parameters
    assert "considerar_solar" not in inspect.signature(comparar_escopos).parameters

    fonte = inspect.getsource(executar_estudo)
    escopos = fonte[fonte.index("_comparar_escopos("):]
    escopos = escopos[:escopos.index(")\n")]
    assert "considerar_solar" not in escopos, "argumento na chamada errada"

    uso = fonte[fonte.index("comparar_cenarios_de_uso("):]
    uso = uso[:uso.index(")\n")]
    assert "considerar_solar=cfg.considerar_solar" in uso
