"""
Tabela de preço de kit fotovoltaico: até 40 kWp, e o trifásico até 125.

Uma curva de R$/kWp com ganho de escala descreve bem obra grande e descreve
mal o varejo. Abaixo de 40 kWp o preço não vem de uma curva: vem de uma
**tabela de kit** do distribuidor, com degraus por potência e colunas por
topologia de inversor. A diferença entre um kit monofásico e um split-phase de
mesma potência passa de 50%, e nenhuma curva de escala captura isso — é
topologia, não tamanho.

**O que esta tabela é e o que ela não é.** O cabeçalho da fonte diz "Preço Kit
Fotovoltaico": é **equipamento posto**, não obra entregue. Não estão aqui a
mão de obra, a estrutura além da que vem no kit, o projeto, a ART, a
homologação na distribuidora, o frete até a obra nem a margem. Usar estes
números como CAPEX total subestima o investimento; usá-los como o que são —
o custo do material — é exatamente o que eles servem para fazer.

Por isso :func:`preco_kit` devolve o preço do kit, e o que falta entra em
**duas parcelas por kWp** -- mão de obra e material CA -- que são as duas
coisas que o instalador sabe de cor e que um multiplicador único esconderia.
Elas são somadas, e não multiplicadas: mão de obra não escala com o preço do
kit, escala com o tamanho do telhado. Um kit split-phase custa 50% mais que um
mono da mesma potência, e a equipe leva o mesmo tempo para instalar os dois --
um fator percentual cobraria 50% a mais de mão de obra por nada.

**A coluna com bateria fica de fora do CAPEX solar.** "Split + 5kWh" traz um
banco de 5 kWh embutido, e o estudo já precifica armazenamento por conta
própria a partir do banco que ele dimensionou. Somar as duas coisas cobraria a
bateria duas vezes. Ela permanece na tabela como referência de conferência —
:func:`preco_kit_com_bateria` — e é útil justamente para checar se o
armazenamento do estudo está saindo caro perto do que o distribuidor cobra.
"""
from __future__ import annotations

import math
from datetime import date
from typing import Iterable

__all__ = [
    "BATERIA_BLOCO_BRL",
    "BATERIA_BLOCO_KWH",
    "DESCRICAO_TOPOLOGIA",
    "TOPOLOGIA_COM_BATERIA",
    "blocos_de_bateria",
    "preco_da_bateria",
    "MAO_DE_OBRA_BRL_KWP",
    "MATERIAL_CA_BRL_KWP",
    "POTENCIA_MAXIMA_KWP",
    "TABELA_KIT",
    "TOPOLOGIAS",
    "APURADO_EM",
    "capex_de_kit",
    "composicao_de_kit",
    "fora_da_tabela",
    "revisao_defasada",
    "potencia_maxima",
    "preco_kit",
    "preco_kit_com_bateria",
    "topologia_para_rede",
]

#: Quando a tabela foi apurada. Preço de kit muda toda semana; sem a data, o
#: número vira uma verdade permanente que ele não é.
APURADO_EM = date(2026, 9, 8)

FONTE = "Tabela de kit fotovoltaico do distribuidor, faixa até 40 kWp"

#: Para que serve cada topologia, na linguagem de quem vende.
#:
#: A escolha não é técnica no sentido de haver uma certa: as quatro funcionam.
#: O que muda é o problema que cada uma resolve, e é isso que decide. Guardar
#: esta descrição junto do preço é deliberado — a diferença de 50% entre um
#: mono e um split-phase de mesma potência só faz sentido ao lado do motivo.
DESCRICAO_TOPOLOGIA: dict[str, str] = {
    "mono_bifasico":
        "Cliente focado em economizar. Sistema mais simples, que cumpre o seu "
        "papel: injeta na rede e abate a conta.",
    "microinversor":
        "Clientes com sombreamento severo. Cada módulo trabalha por conta "
        "própria, então a sombra num deles não arrasta a série inteira.",
    "splitphase":
        "Inversor híbrido: funciona de dia mesmo sem a concessionária, e "
        "atende 127 V e 220 V no mesmo equipamento, o que simplifica a "
        "separação de cargas do quadro de backup.",
    "trifasico":
        "Entrada trifásica, a partir de 20 kWp. É o caminho de qualquer "
        "sistema maior, e o único disponível acima de 40 kWp.",
}

#: A topologia que um sistema com bateria pede.
#:
#: Bateria exige inversor híbrido, e híbrido no varejo é split-phase. Escolher
#: mono/bifásico e pedir bateria é especificar um equipamento que não existe.
TOPOLOGIA_COM_BATERIA = "splitphase"

#: O bloco de expansão de bateria: quanto de energia, e quanto custa.
#:
#: Vem da própria tabela. A coluna "Split + 5kWh" é o kit split-phase mais um
#: banco de 5 kWh, e a diferença entre as duas colunas, linha a linha, é de
#: R$ 11.900 até 20 kWp e cai para R$ 9.800 acima disso — desconto de volume
#: sobre o mesmo bloco. R$ 12.000 é a cotação de expansão em vigor e o padrão
#: aqui; quem tiver a do dia deve trocá-la.
#:
#: Bloco, e não R$/kWh contínuo, porque é assim que se compra: bateria vem em
#: módulo, e meio módulo não existe.
BATERIA_BLOCO_KWH = 5.0
BATERIA_BLOCO_BRL = 12_000.0

#: As topologias da tabela, na ordem em que a fonte as apresenta.
#:
#: A coluna com bateria não entra aqui: ela não é uma alternativa de topologia
#: para o CAPEX solar, é um kit que já traz armazenamento e seria contado duas
#: vezes. Ver :func:`preco_kit_com_bateria`.
TOPOLOGIAS: dict[str, str] = {
    "mono_bifasico": "Mono/Bifásico",
    "microinversor": "Microinversor",
    "splitphase": "SplitPhase",
    "trifasico": "Trifásico",
}

#: ``potencia_kwp -> (módulos, {topologia: preço})``. ``None`` onde a fonte traz
#: um traço: o traço não é zero nem "consulte" — é a topologia não existir
#: naquela potência, e interpolar por cima dele inventaria um produto.
TABELA_KIT: dict[float, tuple[int, dict[str, float | None]]] = {
    5.0:  (8,  {"mono_bifasico": 19_802.0, "microinversor": 20_550.0,
                "splitphase": 30_653.0, "trifasico": None, "com_bateria": 41_918.0}),
    7.5:  (12, {"mono_bifasico": 27_498.0, "microinversor": 28_230.0,
                "splitphase": 36_232.0, "trifasico": None, "com_bateria": 48_346.0}),
    10.0: (16, {"mono_bifasico": 35_397.0, "microinversor": 38_260.0,
                "splitphase": 45_196.0, "trifasico": None, "com_bateria": 57_538.0}),
    12.5: (20, {"mono_bifasico": 42_542.0, "microinversor": 47_074.0,
                "splitphase": 52_531.0, "trifasico": None, "com_bateria": 64_398.0}),
    15.0: (24, {"mono_bifasico": 50_465.0, "microinversor": 56_113.0,
                "splitphase": 61_741.0, "trifasico": None, "com_bateria": 73_808.0}),
    17.5: (28, {"mono_bifasico": 62_505.0, "microinversor": 66_679.0,
                "splitphase": 70_574.0, "trifasico": None, "com_bateria": 82_695.0}),
    20.0: (32, {"mono_bifasico": 67_274.0, "microinversor": 73_034.0,
                "splitphase": 75_367.0, "trifasico": 67_527.0, "com_bateria": 87_475.0}),
    25.0: (40, {"mono_bifasico": 83_649.0, "microinversor": 91_703.0,
                "splitphase": 91_660.0, "trifasico": 82_756.0, "com_bateria": 101_705.0}),
    30.0: (48, {"mono_bifasico": 97_952.0, "microinversor": 108_938.0,
                "splitphase": 105_728.0, "trifasico": 97_972.0, "com_bateria": 115_711.0}),
    35.0: (56, {"mono_bifasico": 113_808.0, "microinversor": 125_423.0,
                "splitphase": 121_895.0, "trifasico": 110_305.0, "com_bateria": 131_768.0}),
    40.0: (64, {"mono_bifasico": 128_901.0, "microinversor": 141_960.0,
                "splitphase": 138_924.0, "trifasico": 128_607.0, "com_bateria": 148_713.0}),
    # ---- revisão anterior, não reenviada -----------------------------------
    # A coluna trifásica de 20 a 40 kWp subiu 20% na revisão nova (R$ 56.244
    # para R$ 67.527 em 20 kWp), então estas linhas certamente também. Aplicar
    # o reajuste médio a elas seria inventar cotação, e por isso elas ficam
    # como estão, marcadas em `POTENCIA_REVISADA_ATE`: o estudo avisa quando
    # as usa. Reenviada a tabela de 50 a 125 kWp, é só substituir aqui.
    50.0:  (80,  {"mono_bifasico": None, "microinversor": None,
                  "splitphase": None, "trifasico": 131_623.0, "com_bateria": None}),
    60.0:  (96,  {"mono_bifasico": None, "microinversor": None,
                  "splitphase": None, "trifasico": 155_461.0, "com_bateria": None}),
    70.0:  (112, {"mono_bifasico": None, "microinversor": None,
                  "splitphase": None, "trifasico": 178_123.0, "com_bateria": None}),
    80.0:  (128, {"mono_bifasico": None, "microinversor": None,
                  "splitphase": None, "trifasico": 207_966.0, "com_bateria": None}),
    90.0:  (144, {"mono_bifasico": None, "microinversor": None,
                  "splitphase": None, "trifasico": 231_000.0, "com_bateria": None}),
    100.0: (160, {"mono_bifasico": None, "microinversor": None,
                  "splitphase": None, "trifasico": 258_568.0, "com_bateria": None}),
    110.0: (176, {"mono_bifasico": None, "microinversor": None,
                  "splitphase": None, "trifasico": 281_468.0, "com_bateria": None}),
    120.0: (192, {"mono_bifasico": None, "microinversor": None,
                  "splitphase": None, "trifasico": 305_731.0, "com_bateria": None}),
    125.0: (202, {"mono_bifasico": None, "microinversor": None,
                  "splitphase": None, "trifasico": 313_681.0, "com_bateria": None}),
}

#: Até onde a tabela é da revisão atual. Acima disso os preços são da anterior.
#:
#: A revisão nova subiu 20,3% de forma uniforme, degrau a degrau — assinatura
#: de reajuste de lista, e não de mudança de produto. As linhas acima deste
#: limite não foram reenviadas e estão, portanto, defasadas na mesma ordem de
#: grandeza. O estudo avisa quando as usa em vez de calar.
POTENCIA_REVISADA_ATE = 40.0

#: Acima disto nenhuma coluna fala, e a curva de escala volta a valer.
#:
#: O limite útil é **por coluna**, e não global: o mono/bifásico para em
#: 40 kWp e o trifásico segue até 125. Use :func:`potencia_maxima` para saber
#: o de cada topologia; este valor é o teto do conjunto.
POTENCIA_MAXIMA_KWP = max(TABELA_KIT)

#: Menor potência da tabela. Abaixo dela o preço não cai proporcionalmente --
#: um kit de 2 kWp não custa 40% de um de 5 kWp, porque inversor, estrutura e
#: projeto têm piso. O preço de 5 kWp é adotado como piso.
POTENCIA_MINIMA_KWP = min(TABELA_KIT)

#: Mão de obra de instalação, em R$/kWp.
#:
#: Cobre equipe, estrutura fora do kit, projeto, ART e homologação -- o que se
#: gasta para transformar material posto em usina ligada. Escala com o
#: tamanho do telhado, e não com o preço do equipamento, que é a razão de ser
#: uma parcela somada em vez de um percentual.
#:
#: **Nasce em zero, de propósito.** Era o número mais arbitrado do módulo, e
#: arbitrado é o que não deveria entrar num orçamento sem alguém ter dito de
#: onde veio. Em zero, o investimento que sai do estudo é cotação de ponta a
#: ponta — o kit da tabela mais os blocos de bateria — e quem conhece a
#: própria estrutura de custo informa o valor na tela.
MAO_DE_OBRA_BRL_KWP = 0.0

#: Material do lado CA, em R$/kWp.
#:
#: Cabo CA do inversor ao quadro, disjuntores, DPS, quadro de proteção,
#: eletroduto e aterramento -- tudo que o kit não traz porque depende da
#: distância até o padrão de entrada. Também por kWp: numa instalação com o
#: inversor longe do quadro, sobe.
#:
#: Nasce em zero pela mesma razão que a mão de obra.
MATERIAL_CA_BRL_KWP = 0.0


def topologia_para_rede(tensao_rede_v: float, monofasico: bool = False) -> str:
    """
    A topologia que a rede da instalação pede.

    Rede de 380 V é trifásica e o kit trifásico é o que se liga nela. Em 220 V
    o padrão do varejo é o mono/bifásico; o split-phase existe para atender
    127 V e 220 V no mesmo inversor, que é o caso das redes 127/220 em que há
    carga de 220 V a segurar no backup.

    A escolha é um ponto de partida razoável, não um veredito: quem monta o kit
    sabe da instalação coisas que a tensão nominal não conta.
    """
    if float(tensao_rede_v) >= 300.0:
        return "trifasico"
    return "mono_bifasico" if monofasico else "splitphase"


def potencia_maxima(topologia: str) -> float:
    """
    Até onde a coluna vai.

    Acima de 40 kWp a entrada é trifásica e as outras topologias somem do
    catálogo, então cada coluna tem o seu próprio teto. Perguntar pelo teto
    global levaria a tela a prometer preço de kit monofásico em 100 kWp.
    """
    potencias = [
        potencia for potencia, (_, valores) in TABELA_KIT.items()
        if valores.get(topologia) is not None
    ]
    return max(potencias) if potencias else 0.0


def _interpolar(kwp: float, coluna: str) -> float | None:
    """
    O preço da coluna na potência pedida, interpolando entre os degraus.

    Entre 12,5 e 15 kWp não existe produto, existe degrau -- mas o estudo
    dimensiona 13,6 kWp e precisa de um número. A reta entre os dois degraus
    vizinhos é a leitura mais honesta disponível: não inventa desconto de
    escala que a tabela não mostra, e não arredonda para cima o cliente.
    """
    pontos = [
        (potencia, valores[coluna])
        for potencia, (_, valores) in sorted(TABELA_KIT.items())
        if valores.get(coluna) is not None
    ]
    if not pontos:
        return None

    menor, maior = pontos[0][0], pontos[-1][0]
    if kwp > maior:
        return None
    if kwp <= menor:
        # Piso: inversor, estrutura e projeto não encolhem com a potência.
        return pontos[0][1]

    anterior = pontos[0]
    for atual in pontos[1:]:
        if kwp <= atual[0]:
            largura = atual[0] - anterior[0]
            fracao = (kwp - anterior[0]) / largura if largura else 0.0
            return anterior[1] + fracao * (atual[1] - anterior[1])
        anterior = atual
    return anterior[1]


def preco_kit(potencia_kwp: float, topologia: str = "mono_bifasico") -> float | None:
    """
    Preço do **kit** — equipamento posto, sem obra — na potência pedida.

    Devolve ``None`` acima de 40 kWp, onde a tabela não fala, e para topologia
    que não existe na faixa (o trifásico só começa em 20 kWp). ``None`` é a
    resposta certa nesses casos: quem chama volta para a curva de escala em vez
    de receber um número extrapolado que parece cotação.
    """
    if potencia_kwp <= 0:
        return 0.0
    chave = str(topologia).strip().lower()
    if chave not in TOPOLOGIAS and chave != "com_bateria":
        conhecidas = ", ".join(sorted(TOPOLOGIAS))
        raise ValueError(f"topologia desconhecida: {topologia!r}. Conhecidas: {conhecidas}")
    return _interpolar(float(potencia_kwp), chave)


def preco_kit_com_bateria(potencia_kwp: float) -> float | None:
    """
    O kit split-phase com 5 kWh embutidos, exatamente como a tabela traz.

    Serve de conferência: o split-phase somado a um bloco de bateria deve dar
    aproximadamente isto, e dá — a diferença entre as duas colunas é o próprio
    bloco. Quando o estudo se afasta muito deste número na mesma potência, ou
    o banco ficou grande, ou o preço do bloco está velho.
    """
    return _interpolar(float(potencia_kwp), "com_bateria")


def blocos_de_bateria(energia_kwh: float, bloco_kwh: float = BATERIA_BLOCO_KWH) -> int:
    """
    Quantos blocos cobrem a energia pedida.

    Arredonda para cima porque bateria vem em módulo: quem precisa de 6 kWh
    compra dois blocos de 5, e cobrar 1,2 bloco seria cobrar um produto que
    não se vende.
    """
    if energia_kwh <= 0 or bloco_kwh <= 0:
        return 0
    return int(math.ceil(float(energia_kwh) / float(bloco_kwh)))


def preco_da_bateria(
    energia_kwh: float,
    bloco_kwh: float = BATERIA_BLOCO_KWH,
    bloco_brl: float = BATERIA_BLOCO_BRL,
) -> float:
    """
    O preço do banco, contado em blocos de expansão.

    É o valor que a tabela implica e a referência de campo confirma: a cada
    5 kWh, mais ou menos R$ 10 mil. O inversor **não** entra aqui — ele já veio
    no kit split-phase, e cobrá-lo de novo é o erro que dobra o orçamento de um
    sistema com bateria.
    """
    return blocos_de_bateria(energia_kwh, bloco_kwh) * max(0.0, float(bloco_brl))


def revisao_defasada(potencia_kwp: float, topologia: str) -> str | None:
    """
    O aviso de quem caiu nas linhas que não foram reenviadas.

    Acima de :data:`POTENCIA_REVISADA_ATE` a tabela é da revisão anterior. A
    revisão nova subiu 20,3% de forma uniforme na faixa que foi reenviada, e
    não há razão para supor que a faixa de cima tenha ficado parada. Aplicar o
    reajuste médio a ela seria inventar cotação; avisar é o que resta.
    """
    if not topologia or potencia_kwp <= POTENCIA_REVISADA_ATE:
        return None
    return (
        f"O sistema tem {potencia_kwp:.1f} kWp, acima dos "
        f"{POTENCIA_REVISADA_ATE:.0f} kWp até onde a tabela de kit foi atualizada. "
        "Os preços desta faixa são da revisão anterior, e a faixa reenviada subiu "
        "20% entre uma revisão e outra — o investimento deste estudo está, "
        "portanto, provavelmente subestimado. Confirme a cotação antes da proposta."
    )


def fora_da_tabela(potencia_kwp: float, topologia: str) -> str | None:
    """
    O aviso de quem cruzou a fronteira das duas bases de preço.

    Abaixo do alcance da coluna o preço é **equipamento posto** — o kit do
    distribuidor. Acima dela vale a curva de R$/kWp, que é **obra entregue**.
    Enquanto a mão de obra entrava com algum valor, a diferença entre as duas
    ficava em poucos por cento e ninguém precisava saber disso; com a obra em
    zero, o degrau passa de 30%, e um sistema de 41 kWp aparece um terço mais
    caro que um de 40. A diferença não é do sistema — é do que cada número
    inclui, e é isso que o aviso diz.

    Devolve ``None`` quando não há fronteira cruzada, que é o caso comum.
    """
    if not topologia:
        return None
    teto = potencia_maxima(topologia)
    if not teto or potencia_kwp <= teto:
        return None
    return (
        f"O sistema tem {potencia_kwp:.1f} kWp e a tabela de kit vai até "
        f"{teto:.0f} kWp em {TOPOLOGIAS.get(topologia, topologia)}. Acima disso o "
        "investimento vem da curva de R$/kWp, que é obra entregue — inclui mão de "
        "obra, projeto, ART e margem —, enquanto abaixo dela o número é equipamento "
        "posto. As duas bases não são comparáveis entre si."
    )


def capex_de_kit(
    potencia_kwp: float,
    topologia: str = "mono_bifasico",
    mao_de_obra_brl_kwp: float = MAO_DE_OBRA_BRL_KWP,
    material_ca_brl_kwp: float = MATERIAL_CA_BRL_KWP,
    bateria_kwh: float = 0.0,
    bloco_kwh: float = BATERIA_BLOCO_KWH,
    bloco_brl: float = BATERIA_BLOCO_BRL,
) -> float | None:
    """
    O CAPEX instalado: o kit da tabela mais o que a obra acrescenta.

    Kit é material posto; CAPEX é usina ligada. As duas parcelas que faltam
    entram **somadas por kWp**, e não como percentual sobre o kit: a equipe
    leva o mesmo tempo para instalar um kit mono e um split-phase de mesma
    potência, e um multiplicador cobraria 50% a mais de mão de obra só porque
    o equipamento é mais caro.

    Devolve ``None`` quando a tabela não cobre a potência ou a topologia --
    quem chama volta para a curva de escala.
    """
    kit = preco_kit(potencia_kwp, topologia)
    if kit is None:
        return None
    adicional = max(0.0, float(mao_de_obra_brl_kwp)) + max(0.0, float(material_ca_brl_kwp))
    bateria = preco_da_bateria(bateria_kwh, bloco_kwh, bloco_brl)
    return kit + adicional * float(potencia_kwp) + bateria


def composicao_de_kit(
    potencia_kwp: float,
    topologia: str = "mono_bifasico",
    mao_de_obra_brl_kwp: float = MAO_DE_OBRA_BRL_KWP,
    material_ca_brl_kwp: float = MATERIAL_CA_BRL_KWP,
    bateria_kwh: float = 0.0,
    bloco_kwh: float = BATERIA_BLOCO_KWH,
    bloco_brl: float = BATERIA_BLOCO_BRL,
) -> dict[str, float] | None:
    """
    As três parcelas do investimento, separadas.

    Serve para o relatório dizer de onde vem cada real em vez de mostrar um
    total. É a diferença entre um número que se discute e um que se aceita: o
    cliente que acha o kit caro pesquisa preço de kit, e o que acha a obra cara
    pede outro instalador.
    """
    kit = preco_kit(potencia_kwp, topologia)
    if kit is None:
        return None
    kwp = float(potencia_kwp)
    # Parcela zerada não vira linha: uma linha de R$ 0 numa tabela de
    # composição não informa nada e ainda sugere que alguém esqueceu de
    # preencher. Quem não informou mão de obra não a vê no documento.
    candidatas = {
        "kit": kit,
        "mao_de_obra": max(0.0, float(mao_de_obra_brl_kwp)) * kwp,
        "material_ca": max(0.0, float(material_ca_brl_kwp)) * kwp,
        "bateria": preco_da_bateria(bateria_kwh, bloco_kwh, bloco_brl),
    }
    return {chave: valor for chave, valor in candidatas.items() if valor > 0}


def linhas_da_tabela(topologias: Iterable[str] | None = None) -> list[dict[str, object]]:
    """A tabela em forma de linhas, para o relatório e para a tela."""
    escolhidas = list(topologias or TOPOLOGIAS)
    linhas = []
    for potencia, (modulos, valores) in sorted(TABELA_KIT.items()):
        linha: dict[str, object] = {"potencia_kwp": potencia, "modulos": modulos}
        for chave in escolhidas:
            linha[chave] = valores.get(chave)
            preco = valores.get(chave)
            linha[f"{chave}_brl_kwp"] = (preco / potencia) if preco else None
        linhas.append(linha)
    return linhas
