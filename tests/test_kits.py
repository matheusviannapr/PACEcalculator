"""
Testes da tabela de preço de kit até 40 kWp.

O que se garante aqui é que a tabela é lida como tabela — com degraus, com
buracos e com limite — e não como se fosse mais uma curva. Os três erros que
custariam caro: extrapolar acima de 40 kWp, inventar preço onde a fonte traz
um traço, e somar a bateria duas vezes.
"""
from __future__ import annotations

import pytest

from aurum.pv.financials import estimar_capex
from aurum.pv.kits import (
    MAO_DE_OBRA_BRL_KWP,
    MATERIAL_CA_BRL_KWP,
    POTENCIA_MAXIMA_KWP,
    TABELA_KIT,
    TOPOLOGIAS,
    capex_de_kit,
    composicao_de_kit,
    potencia_maxima,
    preco_kit,
    preco_kit_com_bateria,
    topologia_para_rede,
)


# ----------------------------------------------------------------------------
# A tabela
# ----------------------------------------------------------------------------
def test_os_degraus_da_tabela_saem_exatos():
    """Na potência de um degrau, o preço é o da fonte — sem interpolação."""
    for potencia, (_, valores) in TABELA_KIT.items():
        for topologia in TOPOLOGIAS:
            esperado = valores[topologia]
            if esperado is None:
                continue
            assert preco_kit(potencia, topologia) == pytest.approx(esperado), (
                f"{potencia} kWp / {topologia}")


def test_entre_degraus_interpola_em_linha_reta():
    """
    O estudo dimensiona 13,6 kWp, e a tabela só tem 12,5 e 15.

    A reta entre os vizinhos não inventa desconto de escala que a tabela não
    mostra, e não arredonda o cliente para cima.
    """
    meio = preco_kit(13.75, "mono_bifasico")
    assert meio == pytest.approx((35_380.0 + 41_944.0) / 2)
    assert 35_380.0 < preco_kit(13.0, "mono_bifasico") < 41_944.0


def test_acima_da_tabela_nao_extrapola():
    """
    Estender a reta acima de 40 kWp entregaria cotação onde não há cotação.

    ``None`` é a resposta certa: quem chama volta para a curva de escala.
    """
    teto = potencia_maxima("mono_bifasico")
    assert teto == 40.0, "o mono/bifásico para em 40 kWp; só o trifásico segue"
    assert preco_kit(teto, "mono_bifasico") is not None
    assert preco_kit(teto + 0.1, "mono_bifasico") is None
    assert preco_kit(80.0, "mono_bifasico") is None
    # O trifásico vai mais longe, e o teto global é o dele.
    assert potencia_maxima("trifasico") == POTENCIA_MAXIMA_KWP == 125.0
    assert preco_kit(125.0, "trifasico") == pytest.approx(313_681.0)
    assert preco_kit(125.1, "trifasico") is None


def test_abaixo_da_tabela_o_preco_tem_piso():
    """Um kit de 2 kWp não custa 40% de um de 5 — inversor e projeto têm piso."""
    assert preco_kit(2.0, "mono_bifasico") == preco_kit(5.0, "mono_bifasico")


def test_o_traco_da_fonte_nao_vira_preco():
    """
    O trifásico só começa em 20 kWp, e o traço não é zero nem "consulte".

    Interpolar por cima dele inventaria um produto que a fonte diz não existir.
    """
    assert TABELA_KIT[10.0][1]["trifasico"] is None
    assert preco_kit(10.0, "trifasico") == pytest.approx(56_244.0), (
        "abaixo do primeiro degrau da coluna vale o piso dela, que é o de 20 kWp")
    assert preco_kit(25.0, "trifasico") == pytest.approx(68_866.0)


def test_topologia_desconhecida_e_recusada():
    with pytest.raises(ValueError, match="topologia desconhecida"):
        preco_kit(10.0, "monofasico_com_bateria_e_cafe")


# ----------------------------------------------------------------------------
# Do kit ao investimento
# ----------------------------------------------------------------------------
def test_a_obra_soma_ao_kit_e_nao_multiplica():
    """
    A equipe leva o mesmo tempo para montar um kit caro e um barato.

    Um percentual sobre o kit cobraria 50% a mais de mão de obra pelo
    split-phase, que é a mesma instalação com equipamento diferente.
    """
    kwp = 10.0
    adicional = (MAO_DE_OBRA_BRL_KWP + MATERIAL_CA_BRL_KWP) * kwp
    for topologia in ("mono_bifasico", "splitphase"):
        assert capex_de_kit(kwp, topologia) == pytest.approx(
            preco_kit(kwp, topologia) + adicional)


def test_as_parcelas_de_obra_sao_manipulaveis():
    """São premissas do instalador, e mudam de uma empresa para outra."""
    barato = capex_de_kit(10.0, "mono_bifasico", 0.0, 0.0)
    caro = capex_de_kit(10.0, "mono_bifasico", 800.0, 400.0)
    assert barato == pytest.approx(preco_kit(10.0, "mono_bifasico"))
    assert caro == pytest.approx(barato + 12_000.0)


def test_a_composicao_fecha_no_total():
    partes = composicao_de_kit(13.6, "splitphase", 500.0, 300.0)
    assert sum(partes.values()) == pytest.approx(
        capex_de_kit(13.6, "splitphase", 500.0, 300.0))
    assert partes["mao_de_obra"] == pytest.approx(500.0 * 13.6)
    assert partes["material_ca"] == pytest.approx(300.0 * 13.6)


# ----------------------------------------------------------------------------
# A ligação com o estudo
# ----------------------------------------------------------------------------
def test_a_tabela_manda_abaixo_de_40_kwp():
    da_tabela = estimar_capex(10.0, topologia="mono_bifasico")
    da_curva = estimar_capex(10.0, padrao="padrao")
    assert da_tabela == pytest.approx(capex_de_kit(10.0, "mono_bifasico"))
    assert da_tabela != pytest.approx(da_curva)


def test_acima_de_40_kwp_a_curva_volta_sozinha():
    """A tabela não fala, e pedir topologia não pode virar erro nem zero."""
    com = estimar_capex(60.0, padrao="padrao", topologia="mono_bifasico")
    sem = estimar_capex(60.0, padrao="padrao")
    assert com == pytest.approx(sem)
    assert com > 0


def test_a_fronteira_da_tabela_e_avisada_e_nao_disfarcada():
    """
    As duas bases de preço não são comparáveis, e o degrau é grande.

    Abaixo do alcance da tabela o número é **equipamento posto**; acima dela
    vale a curva de R$/kWp, que é **obra entregue**. Enquanto a mão de obra
    entrava com algum valor, a diferença ficava em poucos por cento e ninguém
    precisava saber; com a obra em zero, o degrau passa de 30%.

    Disfarçar isso — interpolando entre as duas, ou estendendo a tabela — seria
    o pior dos mundos, porque o degrau passaria a parecer um resultado. O que
    se garante aqui é que ele existe e que vem acompanhado de aviso.
    """
    from aurum.pv.kits import fora_da_tabela

    dentro = estimar_capex(40.0, padrao="padrao", topologia="mono_bifasico") / 40.0
    fora = estimar_capex(40.5, padrao="padrao", topologia="mono_bifasico") / 40.5
    assert fora > dentro, "a curva de obra entregue tem de ficar acima do kit"

    assert fora_da_tabela(40.0, "mono_bifasico") is None
    aviso = fora_da_tabela(40.5, "mono_bifasico")
    assert aviso and "não são comparáveis" in aviso


def test_referencia_explicita_vence_a_tabela():
    """
    A ordem de precedência é a da confiança na origem do número.

    `referencia_brl_kwp` continua sendo referência **em 100 kWp**, e por isso
    ainda atravessa a curva de escala — o que se garante aqui é só que a
    tabela de kit sai do caminho quando alguém informa a própria referência.
    """
    com = estimar_capex(10.0, referencia_brl_kwp=2_000.0, topologia="mono_bifasico")
    sem = estimar_capex(10.0, referencia_brl_kwp=2_000.0)
    assert com == pytest.approx(sem)
    assert com != pytest.approx(capex_de_kit(10.0, "mono_bifasico"))


# ----------------------------------------------------------------------------
# A coluna com bateria
# ----------------------------------------------------------------------------
def test_a_coluna_com_bateria_fica_fora_do_capex_solar():
    """
    "Split + 5kWh" traz um banco embutido, e o estudo já precifica o dele.

    Somar as duas coisas cobraria a bateria duas vezes — por isso a coluna não
    é uma topologia escolhível, e sim uma referência de conferência.
    """
    assert "com_bateria" not in TOPOLOGIAS
    referencia = preco_kit_com_bateria(10.0)
    assert referencia == pytest.approx(47_480.0)
    assert referencia > preco_kit(10.0, "splitphase")


# ----------------------------------------------------------------------------
# A sugestão de topologia
# ----------------------------------------------------------------------------
def test_a_rede_sugere_a_topologia():
    assert topologia_para_rede(380.0) == "trifasico"
    assert topologia_para_rede(220.0) == "splitphase"
    assert topologia_para_rede(220.0, monofasico=True) == "mono_bifasico"


# ----------------------------------------------------------------------------
# O bloco de bateria
# ----------------------------------------------------------------------------
def test_a_coluna_com_bateria_e_o_splitphase_mais_um_bloco():
    """
    A hipótese que valida o modelo inteiro, conferida linha a linha.

    Se "Split + 5kWh" é o kit split-phase mais um banco de 5 kWh, então o
    preço do bloco sai da própria tabela — e a regra de campo ("a cada 5 kWh,
    +R$ 10 mil") deixa de ser palpite e vira leitura. A diferença é de
    R$ 9.900 até 20 kWp e cai para R$ 7.900 acima disso, que é desconto de
    volume sobre o mesmo bloco.
    """
    from aurum.pv.kits import BATERIA_BLOCO_KWH, preco_da_bateria

    for kwp in (5.0, 10.0, 15.0, 20.0, 30.0, 40.0):
        reconstruido = preco_kit(kwp, "splitphase") + preco_da_bateria(BATERIA_BLOCO_KWH)
        da_tabela = preco_kit_com_bateria(kwp)
        assert reconstruido == pytest.approx(da_tabela, rel=0.03), (
            f"{kwp} kWp: reconstruído {reconstruido:,.0f} contra {da_tabela:,.0f}")


def test_a_bateria_e_cobrada_em_blocos_inteiros():
    """
    Meio módulo de bateria não se vende, e cobrar 1,2 bloco seria cobrar um
    produto que não existe.
    """
    from aurum.pv.kits import BATERIA_BLOCO_BRL, blocos_de_bateria, preco_da_bateria

    assert blocos_de_bateria(0.0) == 0
    assert blocos_de_bateria(2.2) == 1, "quem precisa de 2,2 kWh compra um bloco de 5"
    assert blocos_de_bateria(5.0) == 1
    assert blocos_de_bateria(5.1) == 2
    assert blocos_de_bateria(9.9) == 2
    assert preco_da_bateria(9.9) == pytest.approx(2 * BATERIA_BLOCO_BRL)


def test_o_preco_do_bloco_e_manipulavel():
    """É cotação, e cotação muda toda semana."""
    from aurum.pv.kits import preco_da_bateria

    assert preco_da_bateria(10.0, bloco_kwh=5.0, bloco_brl=12_000.0) == pytest.approx(24_000.0)
    assert preco_da_bateria(10.0, bloco_kwh=10.0, bloco_brl=18_000.0) == pytest.approx(18_000.0)


def test_a_bateria_entra_na_composicao_e_no_capex():
    """
    O inversor não é cobrado de novo: ele veio no kit split-phase.

    Somar um sistema de armazenamento inteiro sobre um kit que já traz o
    híbrido é o erro que dobra o orçamento — e era o que acontecia antes de a
    bateria passar a ser contada em blocos.
    """
    from aurum.pv.kits import preco_da_bateria

    sem = capex_de_kit(10.0, "splitphase")
    com = capex_de_kit(10.0, "splitphase", bateria_kwh=9.9)
    assert com - sem == pytest.approx(preco_da_bateria(9.9))

    partes = composicao_de_kit(10.0, "splitphase", bateria_kwh=9.9)
    assert "bateria" in partes
    assert sum(partes.values()) == pytest.approx(com)
    # Sem bateria, a parcela nem aparece — uma linha de R$ 0 é ruído.
    assert "bateria" not in composicao_de_kit(10.0, "splitphase")


def test_toda_topologia_explica_para_que_serve():
    """
    A diferença de 50% entre um mono e um split-phase só faz sentido ao lado
    do motivo. Preço sem o porquê vira discussão de desconto.
    """
    from aurum.pv.kits import DESCRICAO_TOPOLOGIA, TOPOLOGIA_COM_BATERIA

    for chave in TOPOLOGIAS:
        assert len(DESCRICAO_TOPOLOGIA.get(chave, "")) > 40, chave
    assert TOPOLOGIA_COM_BATERIA in TOPOLOGIAS, "bateria exige híbrido, e híbrido é split-phase"
