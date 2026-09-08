"""
Testes da tabela de preço de kit, de 5 a 125 kWp.

O que se garante aqui é que a tabela é lida como tabela — com degraus, com
buracos e com limite por coluna — e não como se fosse mais uma curva. Os erros
que custariam caro: extrapolar acima do teto da coluna, inventar preço onde a
fonte traz um traço, somar a bateria duas vezes, e apresentar como cotação um
banco que acima de 40 kWp o fornecedor monta caso a caso.
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
    baixo = TABELA_KIT[12.5][1]["mono_bifasico"]
    alto = TABELA_KIT[15.0][1]["mono_bifasico"]
    assert preco_kit(13.75, "mono_bifasico") == pytest.approx((baixo + alto) / 2)
    assert baixo < preco_kit(13.0, "mono_bifasico") < alto


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
    assert preco_kit(125.0, "trifasico") == pytest.approx(377_744.0)
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
    assert preco_kit(10.0, "trifasico") == pytest.approx(
        TABELA_KIT[20.0][1]["trifasico"]
    ), "abaixo do primeiro degrau da coluna vale o piso dela, que é o de 20 kWp"
    assert preco_kit(25.0, "trifasico") == pytest.approx(TABELA_KIT[25.0][1]["trifasico"])


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
    assert referencia == pytest.approx(TABELA_KIT[10.0][1]["com_bateria"])
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

    "Split + 5kWh" é o kit split-phase mais um banco de 5 kWh, e a diferença
    entre as duas colunas é o próprio bloco — é isso que faz o preço do banco
    sair da tabela em vez de sair de um R$/kWh genérico.
    """
    from aurum.pv.kits import BATERIA_BLOCO_KWH

    for potencia, (_, valores) in TABELA_KIT.items():
        split, com = valores["splitphase"], valores["com_bateria"]
        if split is None or com is None:
            continue
        bloco = com - split
        assert bloco > 0, f"{potencia} kWp: a coluna com bateria não é mais cara"
        # A faixa é larga porque o desconto de volume é real: o mesmo bloco de
        # 5 kWh sai por menos num sistema de 40 kWp do que num de 5 kWp.
        assert 8_000.0 <= bloco <= 14_000.0, (
            f"{potencia} kWp: bloco implícito de R$ {bloco:,.0f} — fora da faixa "
            "esperada para um módulo de 5 kWh")


def test_o_bloco_adotado_acompanha_o_que_a_tabela_implica():
    """
    A distância entre a regra de campo e a tabela, medida e travada.

    A tabela traz impressa a regra "a cada 5 kWh, adicionar + ~R$ 10 mil", e é
    ela que o módulo adota — é a declaração do fornecedor, e contrariá-la seria
    substituir o dado por uma dedução nossa. Mas a diferença entre as colunas,
    na faixa residencial, está em torno de R$ 11.900 desde a revisão de
    setembro: 19% acima da regra.

    Enquanto a distância for essa, adotar a regra é conservador de menos e
    consciente. Se ela crescer, as duas se separaram de vez e o valor adotado
    precisa mudar — é isso que este teste vigia.
    """
    from aurum.pv.kits import BATERIA_BLOCO_BRL

    implicitos = [
        v["com_bateria"] - v["splitphase"]
        for potencia, (_, v) in TABELA_KIT.items()
        if v["splitphase"] and v["com_bateria"] and potencia <= 20.0
    ]
    mediana = sorted(implicitos)[len(implicitos) // 2]
    distancia = mediana / BATERIA_BLOCO_BRL - 1.0
    assert 0.0 <= distancia <= 0.30, (
        f"o bloco adotado (R$ {BATERIA_BLOCO_BRL:,.0f}) está a {distancia:.0%} da "
        f"mediana implícita pela tabela (R$ {mediana:,.0f}) — revise o valor adotado"
    )


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


def test_a_faixa_alta_e_a_mesma_revisao_da_baixa():
    """
    A tabela deixou de estar pela metade, e a razão entre as revisões prova.

    Por um tempo os preços até 40 kWp eram de setembro e os de 50 a 125 kWp da
    revisão anterior — 20% mais baratos, num estudo que apresentaria
    investimento subestimado. O reajuste é de lista, não de produto: sobe
    uniforme degrau a degrau e não mexe na contagem de módulos. Se um dia a
    faixa alta ficar para trás de novo, é este teste que acusa.
    """
    from aurum.pv.kits import POTENCIA_REVISADA_ATE, revisao_defasada

    anterior = {50.0: 131_623.0, 60.0: 155_461.0, 70.0: 178_123.0,
                80.0: 207_966.0, 90.0: 231_000.0, 100.0: 258_568.0,
                110.0: 281_468.0, 125.0: 313_681.0}
    for kwp, velho in anterior.items():
        razao = TABELA_KIT[kwp][1]["trifasico"] / velho
        assert 1.18 < razao < 1.22, f"{kwp} kWp fora do reajuste uniforme"

    assert TABELA_KIT[125.0][0] == 202, "reajuste de lista não mexe em módulo"
    assert POTENCIA_REVISADA_ATE == POTENCIA_MAXIMA_KWP
    assert revisao_defasada(125.0, "trifasico") is None, "nada mais está defasado"


def test_bateria_de_tabela_para_em_40_kwp():
    """
    Acima de 40 kWp o kit com bateria é montado caso a caso, e a fonte diz isso.

    O banco continua dimensionado — quanta energia a carga essencial exige não
    depende do tamanho do sistema. O que não vale é apresentar o preço dele como
    cotação: um banco trifásico de 90 kWp não é o split-phase residencial
    multiplicado por blocos de R$ 12 mil.
    """
    from aurum.pv.kits import BATERIA_EM_TABELA_ATE, bateria_caso_a_caso

    assert BATERIA_EM_TABELA_ATE == 40.0
    assert preco_kit_com_bateria(40.0) is not None
    assert preco_kit_com_bateria(50.0) is None, "a coluna com bateria para em 40"

    assert bateria_caso_a_caso(30.0) is None
    assert bateria_caso_a_caso(90.0, tem_bateria=False) is None, "sem banco, sem ruído"
    aviso = bateria_caso_a_caso(90.0)
    assert aviso and "caso a caso" in aviso and "cotação" in aviso
