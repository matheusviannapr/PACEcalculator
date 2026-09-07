"""
Testes da ponte entre a vistoria técnica e o estudo.

O que se garante aqui é o contrato: que o backup em JSON é lido inteiro, que
a criticidade recorta o quadro de backup **equipamento a equipamento**, e que
a conferência diz o que falta antes de o estudo rodar com hipótese no lugar
de dado.
"""
from __future__ import annotations

import json

import pytest

from aurum.demanda import ocupacao
from aurum.demanda.contrato_vistoria import CAMPOS, Gravidade, conferir, contrato_markdown
from aurum.demanda.vistoria import CRITICIDADES, ler_backup


def _backup(**vistoria) -> dict:
    """Um backup mínimo, no formato que a vistoria exporta."""
    base = {
        "cliente": "Casa de Teste", "imovel": "T1", "uf": "PR",
        "data": "2026-09-06", "vistoriador": "Fulano",
        "tensao": 127, "sistema": "127/220 V",
        "fv": False, "gerDiesel": False, "ups": False,
    }
    base.update(vistoria)
    return {
        "formato": "vistoria-eletrica/1",
        "vistoria": base,
        "ambientes": [
            {"nome": "Cozinha", "ordem": 20},
            {"nome": "Sala", "ordem": 10},
        ],
        "itens": [
            {"comodo": "Cozinha", "equipamento": "Geladeira", "presente": True,
             "simular": True, "quantidade": 1, "criticidade": "C", "potTipica": 180,
             "tipoInt": "dinâmico", "intervalo": "00:00 as 23:59", "prob": 0.45,
             "fd": 1.0, "durMin": 0.25, "durMax": 0.75, "fp": 5.0,
             "tensao": 127, "tempoMax": 15},
            {"comodo": "Cozinha", "equipamento": "Forno elétrico", "presente": True,
             "simular": True, "quantidade": 1, "criticidade": "NC", "potTipica": 4000,
             "tipoInt": "dinâmico", "intervalo": "18:00 as 22:00", "prob": 0.3,
             "fd": 0.9, "durMin": 0.3, "durMax": 1.0, "fp": 1.0,
             "tensao": 220, "tempoMax": 1440},
            {"comodo": "Sala", "equipamento": "Roteador", "presente": True,
             "simular": True, "quantidade": 1, "criticidade": "MC", "potTipica": 20,
             "tipoInt": "fixo", "intervalo": "00:00 as 23:59", "prob": 1.0,
             "fd": 1.0, "modo": "FIXO_100%", "fp": 1.0,
             "tensao": 127, "tempoMax": 0},
            {"comodo": "Sala", "equipamento": "Notebook", "presente": True,
             "simular": True, "quantidade": 1, "criticidade": "P", "potTipica": 100,
             "tipoInt": "fixo", "intervalo": "08:00 as 18:00", "prob": 0.8,
             "fd": 0.8, "modo": "FIXO_DURACAO_INTERVALAR", "fp": 1.0,
             "tensao": 127, "tempoMax": 120},
        ],
    }


# ----------------------------------------------------------------------------
# Leitura
# ----------------------------------------------------------------------------
def test_le_o_que_as_planilhas_jogam_fora():
    """
    O JSON é a fonte porque carrega o que as planilhas perdem.

    A do simulador tem 10 colunas e descarta cômodo e criticidade — que é o
    que define o quadro de backup. Sem isso o estudo volta a recortar por
    ambiente, e a geladeira crítica arrasta o forno de 4 kW junto.
    """
    v = ler_backup(_backup())
    assert v.cliente == "Casa de Teste"
    assert v.tensao_rede_v == 220.0, "127/220 é rede de 220 V de linha"
    assert v.tem_fv is False
    assert set(v.cenario.comodos) == {"Sala", "Cozinha"}
    assert v.cenario.tem_criticidade


def test_a_ordem_dos_ambientes_e_a_da_vistoria():
    """O vistoriador percorreu a casa numa ordem; o relatório a preserva."""
    assert list(ler_backup(_backup()).cenario.comodos) == ["Sala", "Cozinha"]


def test_tensao_de_linha_e_o_maior_do_par():
    """`tensao: 127` sozinho levaria a procurar inversor de 127 V, que não existe."""
    assert ler_backup(_backup(sistema="127/220 V")).tensao_rede_v == 220.0
    assert ler_backup(_backup(sistema="220/380 V")).tensao_rede_v == 380.0
    # Sem o campo `sistema`, a tensão de fase decide a rede.
    assert ler_backup(_backup(sistema="", tensao=127)).tensao_rede_v == 220.0
    assert ler_backup(_backup(sistema="", tensao=220)).tensao_rede_v == 380.0


def test_item_ausente_ou_sem_potencia_fica_de_fora():
    dados = _backup()
    dados["itens"][1]["presente"] = False
    dados["itens"][3]["potTipica"] = None
    v = ler_backup(dados)
    assert v.cenario.total_de_equipamentos() == 2
    assert len(v.descartados) == 2
    assert any("ausente" in d for d in v.descartados)


# ----------------------------------------------------------------------------
# Criticidade por equipamento
# ----------------------------------------------------------------------------
def test_o_backup_recorta_equipamento_e_nao_comodo():
    """
    A diferença que muda o inversor.

    Na cozinha, a geladeira é crítica e o forno de 4 kW não. Recortando por
    cômodo, os dois entram e o backup pede 4,2 kW; por equipamento, entra só a
    geladeira e o backup pede 0,2 kW.
    """
    v = ler_backup(_backup())
    cenario = v.cenario
    cenario.criticidades_essenciais = ("MC", "C")

    assert cenario.potencia_instalada_w() == pytest.approx(4300.0)
    assert cenario.potencia_instalada_w(True) == pytest.approx(200.0)

    nomes = {
        eq.nome
        for comodo in cenario.para_comodos(True)
        for eq in comodo.equipamentos
    }
    assert nomes == {"Geladeira", "Roteador"}
    assert "Forno elétrico" not in nomes


def test_o_corte_na_escala_muda_o_que_entra():
    v = ler_backup(_backup())
    cenario = v.cenario
    for corte, esperado in ((("MC",), 20.0), (("MC", "C"), 200.0), (CRITICIDADES, 4300.0)):
        cenario.criticidades_essenciais = corte
        assert cenario.potencia_instalada_w(True) == pytest.approx(esperado), corte


def test_a_tabela_por_criticidade_soma_a_instalacao():
    v = ler_backup(_backup())
    tabela = v.cenario.por_criticidade()
    assert len(tabela) == 4
    assert tabela["potencia_w"].sum() == pytest.approx(4300.0)


# ----------------------------------------------------------------------------
# Perfis de ocupação
# ----------------------------------------------------------------------------
def test_a_casa_cheia_espalha_e_a_quase_vazia_nao():
    """
    As duas alavancas do modelo fazem coisas diferentes, e é isso que se testa.

    Na casa cheia todo mundo está em casa e acordado: manter o notebook das
    08:00 às 18:00 seria repetir o horário de escritório num dia em que
    ninguém foi trabalhar. Na casa quase vazia a janela levantada já descreve
    o dia -- o que muda é a intensidade, não o horário.
    """
    base = ler_backup(_backup()).cenario
    assert base.comodos["Sala"].iloc[1]["intervalo"] == "08:00 as 18:00"

    cheia = ocupacao.aplicar(base, "casa_cheia").comodos["Sala"].iloc[1]
    vazia = ocupacao.aplicar(base, "casa_quase_vazia").comodos["Sala"].iloc[1]

    assert cheia["intervalo"] == "07:00 as 23:30", "a casa cheia usa o dia todo"
    assert vazia["intervalo"] == "08:00 as 18:00", "a janela levantada é o dado"


def test_a_casa_quase_vazia_reduz_o_dia_sem_zerar():
    """
    O erro que este módulo cometeu primeiro foi zerar o meio do dia.

    Uma casa em que a família trabalha fora não fica vazia: sobra alguém, ou
    alguém volta para almoçar. Zerar produzia um vale que nenhuma casa tem, e
    fazia o dimensionamento parecer mais folgado do que é.
    """
    base = ler_backup(_backup()).cenario
    antes = float(base.comodos["Sala"].iloc[1]["probabilidade"])
    depois = float(
        ocupacao.aplicar(base, "casa_quase_vazia").comodos["Sala"].iloc[1]["probabilidade"]
    )
    assert 0.0 < depois < antes, "o uso diurno cai"
    # 08:00 às 18:00 cai inteiro no expediente, então leva o fator cheio.
    assert depois == pytest.approx(antes * ocupacao.PERFIS["casa_quase_vazia"].fator_diurno)


def test_o_fator_diurno_nao_alcanca_o_jantar():
    """
    Ninguém trabalha fora às 20 h, então o fator diurno não pode chegar lá.

    O jantar **é** reduzido na casa quase vazia — alguém cozinha para um, e
    não para a família —, mas por um fator próprio, o de refeição. Confundir
    os dois apagaria a refeição do dia de semana ou a deixaria do tamanho da
    do fim de semana; nenhum dos dois é a casa.
    """
    base = ler_backup(_backup()).cenario
    perfil = ocupacao.PERFIS["casa_quase_vazia"]
    forno = base.comodos["Cozinha"].iloc[1]
    assert forno["intervalo"] == "18:00 as 22:00"

    ajustado = ocupacao.aplicar(base, perfil).comodos["Cozinha"].iloc[1]
    esperado = float(forno["probabilidade"]) * perfil.fator_refeicao
    assert float(ajustado["probabilidade"]) == pytest.approx(esperado), (
        "o jantar leva o fator de refeição")
    # E não o diurno, que seria bem menor.
    assert float(ajustado["probabilidade"]) > float(forno["probabilidade"]) * perfil.fator_diurno


def test_a_cozinha_da_casa_cheia_ganha_o_almoco():
    """
    A vistoria levanta o jantar; numa casa cheia almoça-se em casa.

    O formulário aceita uma janela por item e o morador cita a refeição de que
    lembra -- o jantar. Sem o desdobramento a curva sai com um cume só, e o
    pico do meio-dia, que é quando o sol está no máximo, não aparece.
    """
    base = ler_backup(_backup()).cenario
    forno = ocupacao.aplicar(base, "casa_cheia").comodos["Cozinha"].iloc[1]["intervalo"]
    assert " e " in forno, "duas refeições, duas janelas"
    assert forno.startswith("11:30"), "o almoço entra antes do jantar"
    # A TV da sala às 20 h não é carga de refeição, e não ganha almoço nenhum.
    sala = ocupacao.aplicar(base, "casa_cheia").comodos["Sala"].iloc[1]["intervalo"]
    assert " e " not in sala


def test_carga_continua_nao_e_tocada():
    """Geladeira e roteador não sabem que dia é hoje."""
    base = ler_backup(_backup()).cenario
    for perfil in ocupacao.PERFIS:
        ajustado = ocupacao.aplicar(base, perfil)
        assert ajustado.comodos["Cozinha"].iloc[0]["intervalo"] == "00:00 as 23:59"
        assert ajustado.comodos["Sala"].iloc[0]["intervalo"] == "00:00 as 23:59"


def test_casa_cheia_levanta_a_probabilidade_sem_chegar_a_um():
    """
    Um modelo em que tudo liga todo dia perde a coincidência — que é o que o
    Monte Carlo existe para medir.
    """
    base = ler_backup(_backup()).cenario
    perfil = ocupacao.PERFIS["casa_cheia"]
    ajustado = ocupacao.aplicar(base, perfil)
    # O teto vale para as cargas de presença. As contínuas passam intactas —
    # o roteador tem probabilidade 1,0 e continua com ela, porque não é a
    # ocupação que decide se ele liga.
    notebook = ajustado.comodos["Sala"].iloc[1]
    assert 0.8 < float(notebook["probabilidade"]) <= perfil.probabilidade_max
    roteador = ajustado.comodos["Sala"].iloc[0]
    assert float(roteador["probabilidade"]) == 1.0


def test_espalhar_compensa_a_diluicao_da_janela():
    """
    Alargar a janela sem mexer na probabilidade **conserva** a energia do dia.

    O sorteio de uso vem antes do sorteio de horário, e a duração não depende
    da janela: uma janela três vezes maior devolve a mesma energia numa curva
    três vezes mais rasa. Sem compensar, a casa cheia terminava a noite abaixo
    da casa quase vazia — o contrário do que qualquer casa faz.
    """
    base = ler_backup(_backup()).cenario
    perfil = ocupacao.PERFIS["casa_cheia"]
    notebook = base.comodos["Sala"].iloc[1]
    prob = float(notebook["probabilidade"])

    ajustado = float(
        ocupacao.aplicar(base, perfil).comodos["Sala"].iloc[1]["probabilidade"]
    )
    # A janela vai de 10 h para 16,5 h: o alargamento é de 1,65.
    esperado = min(prob * perfil.fator_probabilidade * 1.65, perfil.probabilidade_max)
    assert ajustado == pytest.approx(esperado, rel=1e-3)


def test_os_dois_estudos_pesam_a_semana_inteira():
    """Sete dias, nem mais nem menos — a média ponderada sai errada se não."""
    for chave, estudo in ocupacao.ESTUDOS.items():
        assert sum(estudo["perfis"].values()) == 7, chave
        assert set(estudo["perfis"]) <= set(ocupacao.PERFIS), chave
    assert ocupacao.PERFIL_DIMENSIONANTE in ocupacao.PERFIS


def test_o_original_nao_e_alterado():
    """O levantado em campo é o dado; os perfis são leituras dele."""
    base = ler_backup(_backup()).cenario
    antes = base.comodos["Sala"].iloc[1]["intervalo"]
    ocupacao.aplicar(base, "casa_cheia")
    assert base.comodos["Sala"].iloc[1]["intervalo"] == antes


# ----------------------------------------------------------------------------
# O contrato
# ----------------------------------------------------------------------------
def test_backup_completo_passa_sem_ressalva_de_item():
    conferencia = conferir(_backup(cidade="Curitiba", endereco="Rua X, 1"))
    assert conferencia.serve
    dos_itens = [a for a in conferencia.achados if a.campo.onde == "item"]
    assert not dos_itens, [str(a) for a in dos_itens]


def test_falta_de_criticidade_bloqueia():
    """
    Sem criticidade não há quadro de backup, e o estudo perde a razão de ser.

    É obrigatório de propósito: rodar assim produziria um dimensionamento
    plausível e errado, que é pior que uma recusa.
    """
    dados = _backup()
    for item in dados["itens"]:
        item["criticidade"] = ""
    conferencia = conferir(dados)
    assert not conferencia.serve
    assert any("criticidade" in a.campo.caminho for a in conferencia.bloqueios)


def test_falta_de_tarifa_avisa_mas_nao_bloqueia():
    conferencia = conferir(_backup())
    assert conferencia.serve
    importantes = {a.campo.caminho for a in conferencia.por_gravidade(Gravidade.IMPORTANTE)}
    assert "fatura.tarifaBrlKwh" in importantes
    assert "vistoria.cidade" in importantes


def test_todo_campo_do_contrato_explica_o_custo_da_falta():
    """
    Um contrato que só lista campos vira discussão; um que diz o que se perde
    vira implementação.
    """
    for campo in CAMPOS:
        assert len(campo.para_que) > 15, campo.caminho
        assert len(campo.sem_ele) > 20, campo.caminho


def test_o_contrato_sai_em_markdown_para_colar_no_chamado():
    texto = contrato_markdown([Gravidade.OBRIGATORIO])
    assert texto.startswith("| Campo |")
    assert "itens[].criticidade" in texto
    assert "fatura.tarifaBrlKwh" not in texto, "só os obrigatórios foram pedidos"
