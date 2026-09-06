"""
Preços de referência de mercado — datados, e nunca confundidos com datasheet.

Um datasheet é verdade permanente: a Voc de um módulo não muda porque o dólar
subiu. Preço é o contrário — muda toda semana, varia entre distribuidores e
depende de volume. Misturar os dois no mesmo catálogo produz um documento em
que o cliente não distingue "o inversor entrega 24 kVA por 10 s" (fato) de "o
inversor custa R$ 18.000" (chute de dois meses atrás).

Por isso este módulo é separado, tem **data** e diz de onde cada faixa veio. O
catálogo de equipamento continua com ``preco_brl=None`` no que não tem cotação,
e é aqui que esse vazio é preenchido — marcado como estimativa.

**Como usar bem:** cotação real do seu distribuidor vence tudo isto. Preencha a
coluna de preço na planilha do catálogo e estas faixas deixam de ser
consultadas. Elas existem para o estudo não travar por falta de preço, não para
substituir orçamento.

O mercado brasileiro de 2026 tem uma particularidade que vale registrar: os
preços **subiram**, contrariando a série histórica. Módulos acumularam algo em
torno de 30% de alta entre dezembro de 2025 e março de 2026, com o imposto de
importação sobre painel chinês escalonando até 35% em julho de 2026 e o fim do
reembolso de VAT na China. Quem calibrou premissa em 2024 está errando para
baixo.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

__all__ = [
    "FaixaPreco",
    "PRECOS",
    "preco_bateria_brl",
    "preco_inversor_hibrido_brl",
    "preco_modulo_brl",
]


@dataclass(frozen=True)
class FaixaPreco:
    """
    Uma faixa de preço de mercado, com a data e a origem à vista.

    A faixa é publicada inteira, e não só a média, de propósito: a distância
    entre o mínimo e o máximo é a informação mais honesta que existe aqui.
    Quando ela é larga, o número do meio não merece a confiança que um valor
    único aparenta ter.
    """

    minimo: float
    tipico: float
    maximo: float
    unidade: str
    fonte: str
    apurado_em: date

    def __post_init__(self) -> None:
        if not (self.minimo <= self.tipico <= self.maximo):
            raise ValueError(
                f"faixa inconsistente: {self.minimo} / {self.tipico} / {self.maximo}"
            )

    @property
    def dispersao(self) -> float:
        """Quanto o máximo supera o mínimo — a medida da incerteza da faixa."""
        return self.maximo / self.minimo - 1.0 if self.minimo > 0 else float("inf")

    @staticmethod
    def _br(valor: float) -> str:
        """Ponto de milhar e vírgula decimal, sem tocar no resto da frase."""
        return f"{valor:,.2f}".replace(",", "\x00").replace(".", ",").replace("\x00", ".")

    def descricao(self) -> str:
        # Cada número é formatado sozinho: aplicar o replace na frase inteira
        # comia a vírgula antes de "apurado" e transformava 3.200,00 em
        # 3.200.00 — o mesmo defeito que já mordeu o relatório duas vezes.
        return (
            f"{self._br(self.tipico)} {self.unidade} "
            f"(faixa {self._br(self.minimo)}–{self._br(self.maximo)}), "
            f"apurado em {self.apurado_em:%m/%Y}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "minimo": self.minimo,
            "tipico": self.tipico,
            "maximo": self.maximo,
            "unidade": self.unidade,
            "fonte": self.fonte,
            "apurado_em": self.apurado_em.isoformat(),
        }


_APURACAO = date(2026, 9, 1)

#: As faixas em si. Cada uma carrega a origem porque, sem ela, um número
#: destes vira folclore de planilha em três meses.
PRECOS: dict[str, FaixaPreco] = {
    # Módulo de 550-620 Wp de marca conhecida, no varejo de distribuidor.
    # A ponta alta é bifacial tipo N; a baixa, policristalino de saldo.
    "modulo_brl_wp": FaixaPreco(
        1.36, 1.75, 2.18, "R$/Wp",
        "Levantamento de mercado brasileiro; painel de 550 Wp entre R$ 750 e "
        "R$ 1.200 no varejo, e a faixa de R$ 1,36 a R$ 2,18/Wp praticada em "
        "fevereiro de 2026, corrigida pela alta acumulada do primeiro semestre",
        _APURACAO,
    ),
    # Inversor híbrido trifásico de 5 a 20 kW. Ancorado num preço de tabela
    # observado: GoodWe GW10K-ES-LD a R$ 21.525 cheio e R$ 18.296 com desconto,
    # que dão R$ 2,15 e R$ 1,83 por watt.
    "inversor_hibrido_brl_w": FaixaPreco(
        1.60, 2.00, 2.60, "R$/W",
        "Preço de tabela de distribuidor para híbrido GoodWe de 10 kW "
        "(R$ 21.525 cheio, R$ 18.296 com desconto = R$ 2,15 e R$ 1,83/W), "
        "alargado para cobrir a faixa de 5 a 20 kW e outras marcas",
        _APURACAO,
    ),
    # Banco de lítio LFP residencial e comercial pequeno, só a bateria.
    "bateria_lfp_brl_kwh": FaixaPreco(
        2_500.0, 3_200.0, 5_000.0, "R$/kWh",
        "Faixa de R$ 2.500 a R$ 5.000 por kWh armazenado praticada no mercado "
        "residencial brasileiro em 2026; banco de 10 kWh entre R$ 25.000 e "
        "R$ 50.000 só em baterias",
        _APURACAO,
    ),
    # Grupo gerador a diesel, custo da energia gerada: combustível,
    # lubrificante e manutenção proporcional às horas rodadas.
    "gerador_diesel_brl_kwh": FaixaPreco(
        1.80, 2.30, 3.20, "R$/kWh",
        "Consumo específico de 0,28 a 0,35 L/kWh a preço de diesel S-10 entre "
        "R$ 6,00 e R$ 7,50/L, mais lubrificante e manutenção proporcional",
        _APURACAO,
    ),
}


def _faixa(chave: str) -> FaixaPreco:
    if chave not in PRECOS:
        raise KeyError(f"faixa de preço desconhecida: {chave!r}. Conhecidas: {sorted(PRECOS)}")
    return PRECOS[chave]


def preco_modulo_brl(potencia_wp: float, nivel: str = "tipico") -> float:
    """Preço de um módulo pela potência, na faixa de mercado."""
    faixa = _faixa("modulo_brl_wp")
    return max(0.0, float(potencia_wp)) * float(getattr(faixa, nivel))


def preco_inversor_hibrido_brl(potencia_kw: float, nivel: str = "tipico") -> float:
    """Preço de um híbrido pela potência nominal."""
    faixa = _faixa("inversor_hibrido_brl_w")
    return max(0.0, float(potencia_kw)) * 1000.0 * float(getattr(faixa, nivel))


def preco_bateria_brl(capacidade_kwh: float, nivel: str = "tipico") -> float:
    """
    Preço de um banco pela capacidade nominal.

    Pela capacidade **nominal**, e não pela útil: é assim que o mercado
    anuncia, e converter aqui embutiria a profundidade de descarga duas vezes.
    """
    faixa = _faixa("bateria_lfp_brl_kwh")
    return max(0.0, float(capacidade_kwh)) * float(getattr(faixa, nivel))


def procedencia() -> str:
    """Uma linha para o relatório dizer de quando são os preços."""
    return (
        f"Preços de referência apurados em {_APURACAO:%m/%Y}. São faixas de "
        "mercado, não cotação: orçamento do distribuidor substitui qualquer um "
        "destes números."
    )
