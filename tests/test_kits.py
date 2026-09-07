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


def test_a_transicao_em_40_kwp_nao_da_salto():
    """
    Um degrau grande em 40 kWp faria o estudo recomendar 39,9 kWp por preço.

    Não se exige continuidade perfeita — são duas fontes diferentes —, mas o
    salto tem de ser pequeno o bastante para não distorcer a escolha.
    """
    antes = estimar_capex(40.0, padrao="padrao", topologia="mono_bifasico") / 40.0
    depois = estimar_capex(40.5, padrao="padrao", topologia="mono_bifasico") / 40.5
    assert abs(depois / antes - 1.0) < 0.10


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
