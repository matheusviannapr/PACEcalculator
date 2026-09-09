"""
Testes do deslocamento de rotina.

O Monte Carlo sorteia se um equipamento é usado e por quanto tempo, e até aqui
não sorteava **quando a rotina acontece**: toda simulação punha o almoço na
mesma janela e a hora de dormir na mesma hora. O que se garante aqui é que a
correção faz o que promete e não faz o que não promete — deslocar rotina não
cria nem destrói energia, e não desloca o que não tem hora.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aurum.demanda import simular_ensemble
from aurum.demanda.biblioteca import COLUNAS, linha_de_planilha
from aurum.demanda.cenario import Cenario
from aurum.demanda.rotina import (
    ANCORAS,
    PerfilDeRotina,
    ROTINA_ALTO_PADRAO,
    ROTINA_PADRAO,
    Rotina,
    ancora_da_janela,
    deslocar,
    simular_com_rotina,
    sortear_rotinas,
)


def _casa() -> Cenario:
    def tabela(itens):
        return pd.DataFrame(
            [linha_de_planilha(nome, quantidade, criticidade=crit, **ajustes)
             for nome, quantidade, crit, ajustes in itens],
            columns=COLUNAS,
        )

    cenario = Cenario(
        nome="Casa de teste", segmento="residencia",
        comodos={
            "Cozinha": tabela([
                ("Geladeira doméstica", 1, "C", {}),
                ("Forno de micro-ondas", 1, "NC",
                 {"intervalo": "11:30 as 13:30 e 19:00 as 21:00",
                  "probabilidade": 0.8}),
            ]),
            "Quarto": tabela([
                ("Ar-condicionado split 12.000 BTU", 1, "P",
                 {"intervalo": "21:00 as 06:00", "probabilidade": 0.7}),
                ('TV LED 50"', 1, "NC", {"intervalo": "19:00 as 23:00"}),
            ]),
        },
        instancias={"Cozinha": 1, "Quarto": 1},
    )
    cenario.criticidades_essenciais = ("MC", "C")
    return cenario


# ----------------------------------------------------------------------------
# As âncoras
# ----------------------------------------------------------------------------
def test_cada_janela_cai_no_momento_do_dia_a_que_pertence():
    assert ancora_da_janela(6.0, 8.5) == "manha"
    assert ancora_da_janela(11.5, 13.5) == "almoco"
    assert ancora_da_janela(19.0, 21.0) == "jantar"
    assert ancora_da_janela(21.0, 6.0) == "noite", "a janela que atravessa a noite"


def test_carga_continua_nao_tem_hora_e_nao_se_desloca():
    """
    Uma geladeira não sabe que dia é hoje.

    Deslocar o que não tem momento seria inventar variação onde não há, e o
    efeito apareceria como ruído na base do quadro de backup — que é justamente
    a parte do perfil que precisa ser estável para o banco ser dimensionado.
    """
    assert ancora_da_janela(0.0, 23.99) is None
    assert ancora_da_janela(8.0, 22.0) is None, "janela larga demais para ter momento"


def test_o_deslocamento_preserva_a_duracao_de_cada_janela():
    """
    Deslocar é mover no tempo, não esticar. Se a duração mudasse, a energia
    mudaria junto, e a correção de coincidência viraria uma correção de consumo
    disfarçada — o pior tipo, porque ninguém a veria.
    """
    rotina = Rotina(deslocamentos={"almoco": 90.0, "jantar": -45.0, "noite": 0.0,
                                   "manha": 0.0, "tarde": 0.0})
    cenario = deslocar(_casa(), rotina)
    micro = cenario.comodos["Cozinha"].set_index("Equipamento").loc[
        "Forno de micro-ondas", "intervalo"]
    assert micro == "13:00 as 15:00 e 18:15 as 20:15"


def test_o_deslocamento_nao_altera_o_original():
    """O levantamento é o dado; a rotina é uma leitura dele."""
    cenario = _casa()
    antes = cenario.comodos["Quarto"].copy(deep=True)
    deslocar(cenario, sortear_rotinas(1, ROTINA_PADRAO, 1)[0])
    pd.testing.assert_frame_equal(cenario.comodos["Quarto"], antes)


def test_a_janela_deslocada_pode_atravessar_a_meia_noite():
    """
    Uma rotina empurrada para depois das 24 h está na madrugada, e isso é
    físico. O motor já sabe tratar janela circular; recusá-la aqui seria
    truncar o dia de quem dorme tarde.
    """
    rotina = Rotina(deslocamentos={"noite": 120.0, "manha": 0.0, "almoco": 0.0,
                                   "tarde": 0.0, "jantar": 0.0})
    cenario = deslocar(_casa(), rotina)
    ar = cenario.comodos["Quarto"].set_index("Equipamento").loc[
        "Ar-condicionado split 12.000 BTU", "intervalo"]
    assert ar == "23:00 as 08:00"


# ----------------------------------------------------------------------------
# O sorteio
# ----------------------------------------------------------------------------
def test_o_atraso_do_dia_e_do_domicilio_e_nao_do_aparelho():
    """
    A correlação é o ponto, e é onde é fácil errar.

    Sortear um deslocamento independente por equipamento seria pior que não
    sortear nada: destruiria a coincidência que de fato existe, porque a família
    janta junta. O componente comum é o que faz um dia atrasado ser atrasado em
    tudo, e ele aparece como correlação positiva entre as âncoras.
    """
    rotinas = sortear_rotinas(400, PerfilDeRotina(sigma_comum_min=40.0), semente=3)
    almoco = np.array([r.de("almoco") for r in rotinas])
    jantar = np.array([r.de("jantar") for r in rotinas])
    assert np.corrcoef(almoco, jantar)[0, 1] > 0.2, (
        "um dia de almoço tarde tende a ter jantar tarde"
    )


def test_o_jantar_empurra_a_noite_e_a_noite_nao_puxa_o_jantar():
    """A causalidade tem um sentido só: dormir tarde não adianta o jantar."""
    assert ANCORAS["noite"].arrasto_do_jantar > 0
    assert ANCORAS["jantar"].arrasto_do_jantar == 0

    rotinas = sortear_rotinas(400, ROTINA_PADRAO, semente=5)
    jantar = np.array([r.de("jantar") for r in rotinas])
    noite = np.array([r.de("noite") for r in rotinas])
    manha = np.array([r.de("manha") for r in rotinas])
    assert np.corrcoef(jantar, noite)[0, 1] > np.corrcoef(jantar, manha)[0, 1]


def test_a_madrugada_e_regime_e_nao_cauda():
    """
    Quem vira a noite não janta 40 minutos mais tarde: dorme quatro horas mais
    tarde. Modelar isso como cauda da normal daria um dia levemente estranho no
    lugar de um dia de outra natureza, e a diferença aparece no perfil de
    madrugada, que é o que dimensiona banco.
    """
    perfil = PerfilDeRotina(prob_madrugada=0.25)
    rotinas = sortear_rotinas(400, perfil, semente=11)
    madrugadas = [r for r in rotinas if r.madrugada]
    comuns = [r for r in rotinas if not r.madrugada]

    assert 0.15 < len(madrugadas) / len(rotinas) < 0.35
    assert np.mean([r.de("noite") for r in madrugadas]) > \
        np.mean([r.de("noite") for r in comuns]) + 100.0
    # E só a noite vira: quem trabalha de madrugada almoça na hora de sempre.
    assert abs(np.mean([r.de("almoco") for r in madrugadas])
               - np.mean([r.de("almoco") for r in comuns])) < 30.0


def test_o_deslocamento_e_limitado():
    """Sem limite, a cauda da normal produz jantar às 3 da manhã."""
    rotinas = sortear_rotinas(500, PerfilDeRotina(sigma_comum_min=200.0,
                                                  prob_madrugada=0.0), semente=2)
    for rotina in rotinas:
        for nome, ancora in ANCORAS.items():
            assert abs(rotina.de(nome)) <= ancora.limite_min + 1e-6


# ----------------------------------------------------------------------------
# A simulação
# ----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def comparadas():
    casa = _casa()
    sem = simular_ensemble(casa.para_comodos(), casa.instancias_de(), 160, semente=9)
    com = simular_com_rotina(casa, 160, ROTINA_ALTO_PADRAO, sorteios=16, semente=9)
    return sem, com


def test_deslocar_rotina_nao_cria_nem_destroi_energia(comparadas):
    """
    A promessa que separa esta correção de uma calibração disfarçada.

    Deslocar uma janela no tempo não muda quanto o equipamento consome no dia.
    Se a energia caísse, seria porque alguma janela saiu do dia — bug, não
    modelo — e o número menor pareceria uma melhoria.
    """
    sem, com = comparadas
    energia_sem = float(sem.perfis().sum(axis=1).mean())
    energia_com = float(com.perfis().sum(axis=1).mean())
    assert abs(energia_com / energia_sem - 1.0) < 0.05


def test_deslocar_rotina_espalha_o_perfil_medio(comparadas):
    """
    O que a correção de fato faz: derruba o cume do perfil **médio**.

    Com janelas fixas, os equipamentos começam no mesmo minuto do relógio em
    todas as simulações, e o perfil médio sai mais pontudo que qualquer casa
    real. Quem lê esse perfil dimensiona para uma sincronia que não existe.
    """
    sem, com = comparadas
    cume_sem = float(sem.perfis().mean(axis=0).max())
    cume_com = float(com.perfis().mean(axis=0).max())
    assert cume_com < cume_sem


def test_o_pico_do_dia_nao_e_suavizado(comparadas):
    """
    O dia continua coerente, e é isso que impede a correção de virar maquiagem.

    Espalhar o **conjunto** é certo; espalhar o **dia** seria errado, porque a
    família continua jantando junta. O inversor tem de servir o pior dia, e o
    pior dia não fica menor por a rotina variar — em alguns casos fica maior,
    quando jantar tarde e ar-condicionado cedo passam a se sobrepor, o que o
    modelo de janela fixa não conseguia produzir.
    """
    sem, com = comparadas
    pico_sem = float(np.percentile(sem.perfis().max(axis=1), 95))
    pico_com = float(np.percentile(com.perfis().max(axis=1), 95))
    assert pico_com > 0.9 * pico_sem


def test_o_ensemble_tem_o_numero_de_dias_pedido():
    """Um ensemble menor que o pedido mudaria a estatística sem ninguém notar."""
    casa = _casa()
    com = simular_com_rotina(casa, 100, ROTINA_PADRAO, sorteios=13, semente=4)
    assert com.perfis("verão").shape[0] == 100
    assert com.metadados["rotina_sorteios"] == 13


def test_o_contrato_do_ensemble_nao_muda():
    """Quem consome não precisa saber que a rotina variou."""
    casa = _casa()
    com = simular_com_rotina(casa, 40, ROTINA_PADRAO, sorteios=8, semente=4)
    assert set(com.estacoes) == {"verão", "outono", "inverno", "primavera"}
    assert com.passos_por_dia == 1440
    assert com.perfil_medio_w().shape == (1440,)


# ----------------------------------------------------------------------------
# Sazonalidade: frequência, e não intensidade
# ----------------------------------------------------------------------------
def test_no_inverno_o_ar_nao_liga_mais_fraco_ele_nao_liga():
    """
    O modo sazonal muda a **forma** da curva, e não a energia.

    Aplicado sobre o fator de demanda, −45% de inverno vira "roda as mesmas
    horas todas as noites, a 55% da potência", e o modelo põe um platô de
    ar-condicionado em todas as madrugadas de julho. Aplicado sobre a
    frequência, a maioria das noites de julho fica vazia e as poucas em que o
    aparelho liga têm potência de gente com calor.

    A energia é parecida nos dois — multiplicação é comutativa —, e é
    justamente por isso que o erro passava despercebido: só a curva denuncia.
    """
    from aurum.demanda.nucleo import aplicar_ajuste_sazonal

    casa = _casa()
    comodos = casa.para_comodos()
    chave = "Quarto::Ar-condicionado split 12.000 BTU"
    ajuste = {chave: {"ativo": True, "modo": "probabilidade", "inverno": -45.0}}

    quarto = [c for c in aplicar_ajuste_sazonal(comodos, ajuste, "inverno")
              if c.nome == "Quarto"][0]
    ar = [e for e in quarto.equipamentos if e.nome.startswith("Ar-condicionado")][0]
    assert ar.probabilidade == pytest.approx(0.7 * 0.55)
    assert ar.fator_demanda == pytest.approx(
        [e for e in casa.para_comodos() if e.nome == "Quarto"][0]
        .equipamentos[0].fator_demanda
    ), "a intensidade não é tocada quando o modo é frequência"


def test_a_sazonalidade_por_frequencia_alcanca_o_intervalo_dinamico():
    """
    O defeito que fazia a correção não corrigir nada.

    Um item de intervalo dinâmico com duração não guarda a probabilidade num
    campo: ela está dentro do gerador de janelas, que sorteia se o dia tem
    utilização antes de sortear a hora. Mexer em `probabilidade` depois de
    construído não muda nada — e era exatamente o caso do ar-condicionado, o
    equipamento para o qual o modo foi escrito. Medido antes da correção: as
    quatro estações davam energia idêntica com e sem o modo.
    """
    from aurum.demanda.nucleo import aplicar_ajuste_sazonal

    casa = _casa()
    chave = "Quarto::Ar-condicionado split 12.000 BTU"
    ajuste = {chave: {"ativo": True, "modo": "probabilidade", "inverno": -80.0}}

    quarto = [c for c in aplicar_ajuste_sazonal(casa.para_comodos(), ajuste, "inverno")
              if c.nome == "Quarto"][0]
    ar = [e for e in quarto.equipamentos if e.nome.startswith("Ar-condicionado")][0]
    assert ar.probabilisticado_no_intervalo, "o ar é dinâmico com duração"
    assert ar.fabrica_intervalos is not None, "sem fábrica, o ajuste não alcança"

    # O gerador refeito tem de produzir dia vazio na maioria das vezes. A janela
    # do ar atravessa a meia-noite, e sortear fora do contexto circular é o que
    # o simulador nunca faz — por isso ele entra aqui também.
    from aurum.demanda.correcoes import intervalos_circulares

    with intervalos_circulares(True):
        vazios = sum(1 for _ in range(300) if not ar.intervalos())
    assert vazios > 200, "com −80% de frequência, a maioria das noites fica vazia"


def test_o_modo_padrao_nao_muda_nada():
    """Nenhum ajuste já escrito muda de comportamento sem declarar o modo."""
    from aurum.demanda.nucleo import aplicar_ajuste_sazonal

    casa = _casa()
    chave = "Quarto::Ar-condicionado split 12.000 BTU"
    ajuste = {chave: {"ativo": True, "inverno": -45.0}}

    quarto = [c for c in aplicar_ajuste_sazonal(casa.para_comodos(), ajuste, "inverno")
              if c.nome == "Quarto"][0]
    ar = [e for e in quarto.equipamentos if e.nome.startswith("Ar-condicionado")][0]
    assert ar.probabilidade == pytest.approx(0.7), "a frequência fica intacta"
    assert ar.fator_demanda < 1.0
