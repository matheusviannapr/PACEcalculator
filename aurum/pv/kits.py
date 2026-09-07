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

from datetime import date
from typing import Iterable

__all__ = [
    "MAO_DE_OBRA_BRL_KWP",
    "MATERIAL_CA_BRL_KWP",
    "POTENCIA_MAXIMA_KWP",
    "TABELA_KIT",
    "TOPOLOGIAS",
    "APURADO_EM",
    "capex_de_kit",
    "composicao_de_kit",
    "potencia_maxima",
    "preco_kit",
    "preco_kit_com_bateria",
    "topologia_para_rede",
]

#: Quando a tabela foi apurada. Preço de kit muda toda semana; sem a data, o
#: número vira uma verdade permanente que ele não é.
APURADO_EM = date(2026, 9, 7)

FONTE = "Tabela de kit fotovoltaico do distribuidor, faixa até 40 kWp"

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
    5.0:  (8,  {"mono_bifasico": 16_485.0, "microinversor": 17_088.0,
                "splitphase": 25_227.0, "trifasico": None, "com_bateria": 34_508.0}),
    7.5:  (12, {"mono_bifasico": 22_883.0, "microinversor": 23_472.0,
                "splitphase": 29_918.0, "trifasico": None, "com_bateria": 39_883.0}),
    10.0: (16, {"mono_bifasico": 29_438.0, "microinversor": 31_744.0,
                "splitphase": 37_332.0, "trifasico": None, "com_bateria": 47_480.0}),
    12.5: (20, {"mono_bifasico": 35_380.0, "microinversor": 39_031.0,
                "splitphase": 43_427.0, "trifasico": None, "com_bateria": 53_194.0}),
    15.0: (24, {"mono_bifasico": 41_944.0, "microinversor": 46_495.0,
                "splitphase": 51_028.0, "trifasico": None, "com_bateria": 60_956.0}),
    17.5: (28, {"mono_bifasico": 51_821.0, "microinversor": 55_183.0,
                "splitphase": 58_320.0, "trifasico": None, "com_bateria": 68_291.0}),
    20.0: (32, {"mono_bifasico": 55_834.0, "microinversor": 60_474.0,
                "splitphase": 62_354.0, "trifasico": 56_244.0, "com_bateria": 72_314.0}),
    25.0: (40, {"mono_bifasico": 69_584.0, "microinversor": 76_073.0,
                "splitphase": 76_038.0, "trifasico": 68_866.0, "com_bateria": 84_130.0}),
    30.0: (48, {"mono_bifasico": 81_450.0, "microinversor": 90_300.0,
                "splitphase": 87_714.0, "trifasico": 81_466.0, "com_bateria": 95_756.0}),
    35.0: (56, {"mono_bifasico": 94_556.0, "microinversor": 103_913.0,
                "splitphase": 101_071.0, "trifasico": 91_734.0, "com_bateria": 109_025.0}),
    40.0: (64, {"mono_bifasico": 107_037.0, "microinversor": 117_558.0,
                "splitphase": 115_112.0, "trifasico": 106_800.0, "com_bateria": 122_997.0}),
    # A partir daqui só o trifásico continua: acima de 40 kWp a entrada da
    # instalação é trifásica, e as outras topologias deixam de existir no
    # catálogo do distribuidor. As colunas vazias são `None` de propósito --
    # `_interpolar` trabalha coluna a coluna, então cada uma tem o seu próprio
    # alcance e nenhuma é estendida por causa das vizinhas.
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
#: **É arbitrado**, e é o número deste módulo que mais move o payback. Quem
#: conhece a própria estrutura de custo deve trocá-lo.
MAO_DE_OBRA_BRL_KWP = 500.0

#: Material do lado CA, em R$/kWp.
#:
#: Cabo CA do inversor ao quadro, disjuntores, DPS, quadro de proteção,
#: eletroduto e aterramento -- tudo que o kit não traz porque depende da
#: distância até o padrão de entrada. Também arbitrado, e também por kWp: numa
#: instalação com o inversor longe do quadro, sobe.
MATERIAL_CA_BRL_KWP = 300.0


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
    O kit split-phase com 5 kWh de bateria embutidos, para conferência.

    **Não entra no CAPEX do estudo.** O armazenamento é dimensionado e
    precificado à parte, a partir do banco que a simulação de apagão exigiu;
    somar os dois cobraria a bateria duas vezes. Serve para a conferência que
    importa: se o solar mais o banco do estudo saem muito acima disto na mesma
    potência, ou o banco ficou grande, ou o preço de armazenamento está velho.
    """
    return _interpolar(float(potencia_kwp), "com_bateria")


def capex_de_kit(
    potencia_kwp: float,
    topologia: str = "mono_bifasico",
    mao_de_obra_brl_kwp: float = MAO_DE_OBRA_BRL_KWP,
    material_ca_brl_kwp: float = MATERIAL_CA_BRL_KWP,
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
    return kit + adicional * float(potencia_kwp)


def composicao_de_kit(
    potencia_kwp: float,
    topologia: str = "mono_bifasico",
    mao_de_obra_brl_kwp: float = MAO_DE_OBRA_BRL_KWP,
    material_ca_brl_kwp: float = MATERIAL_CA_BRL_KWP,
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
    return {
        "kit": kit,
        "mao_de_obra": max(0.0, float(mao_de_obra_brl_kwp)) * kwp,
        "material_ca": max(0.0, float(material_ca_brl_kwp)) * kwp,
    }


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
