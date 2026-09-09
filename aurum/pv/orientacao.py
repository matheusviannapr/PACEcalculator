"""
O que a orientação e a inclinação custam — e o que elas invertem.

Um sistema fora do plano ótimo é quase sempre discutido por uma única frase:
"perde tanto por cento". A frase é verdadeira e insuficiente, porque a perda
anual esconde a mudança mais importante que a geometria produz: **em que época
do ano a energia aparece**.

No hemisfério sul, um plano vertical voltado ao Norte inverte a sazonalidade da
geração. No verão o sol passa alto, quase por cima, e roça a superfície vertical
com ângulo de incidência ruim. No inverno ele cruza o céu baixo, ao Norte, e bate
quase de frente na parede. O telhado inclinado faz o contrário. Medido no
Flamengo, Rio de Janeiro: o telhado a 22 graus gera no inverno 85% do que gera no
verão; a parede a 90 graus gera no inverno **três vezes** o que gera no verão.

Isso muda o que o sistema serve, e não só quanto ele rende:

* um telhado entrega no verão, que é quando o ar-condicionado pede;
* uma parede entrega no inverno, que é quando ele não pede — mas é também
  quando os dias são curtos e a bateria tem menos chance de encher;
* para dimensionar armazenamento o que importa é o **dia mais fraco**, e não a
  média do ano. Duas geometrias com a mesma perda anual podem ter piores dias
  muito diferentes.

Este módulo mede as duas coisas a partir das séries horárias reais, e não de
regra de bolso. Ele não escolhe: apresenta, como o resto do estudo.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .solar import nome_orientacao

__all__ = [
    "MedidaDeOrientacao",
    "ComparacaoDeOrientacao",
    "comparar_orientacoes",
]

#: Meses do verão e do inverno no hemisfério sul, em índice de 0 a 11.
_VERAO = (11, 0, 1)
_INVERNO = (5, 6, 7)


@dataclass
class MedidaDeOrientacao:
    """Uma geometria de instalação, medida."""

    rotulo: str
    inclinacao_deg: float
    azimute_deg: float
    anual_kwh_por_kwp: float
    mensal_kwh_por_kwp: np.ndarray
    #: kWh por dia do sistema inteiro, para o dia médio e para os piores.
    diario_medio_kwh: float = 0.0
    diario_p05_kwh: float = 0.0
    fonte: str = ""

    @property
    def orientacao(self) -> str:
        return nome_orientacao(self.azimute_deg)

    @property
    def verao_kwh(self) -> float:
        return float(np.mean([self.mensal_kwh_por_kwp[m] for m in _VERAO]))

    @property
    def inverno_kwh(self) -> float:
        return float(np.mean([self.mensal_kwh_por_kwp[m] for m in _INVERNO]))

    @property
    def razao_inverno_verao(self) -> float:
        """
        A assinatura da inversão.

        Abaixo de 1, a geometria entrega mais no verão — é o telhado. Acima de
        1, ela entrega mais no inverno, e aí a curva de geração está invertida
        em relação ao que quase todo mundo espera de um sistema solar.
        """
        return self.inverno_kwh / self.verao_kwh if self.verao_kwh else float("nan")

    @property
    def inverte(self) -> bool:
        return self.razao_inverno_verao > 1.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "rotulo": self.rotulo,
            "inclinacao_deg": self.inclinacao_deg,
            "azimute_deg": self.azimute_deg,
            "orientacao": self.orientacao,
            "anual_kwh_por_kwp": round(self.anual_kwh_por_kwp, 0),
            "verao_kwh_por_kwp": round(self.verao_kwh, 0),
            "inverno_kwh_por_kwp": round(self.inverno_kwh, 0),
            "razao_inverno_verao": round(self.razao_inverno_verao, 2),
            "diario_medio_kwh": round(self.diario_medio_kwh, 1),
            "diario_p05_kwh": round(self.diario_p05_kwh, 1),
        }


@dataclass
class ComparacaoDeOrientacao:
    """As geometrias lado a lado, com a de referência marcada."""

    medidas: list[MedidaDeOrientacao]
    #: Índice da geometria de referência — o plano ótimo.
    referencia: int = 0
    #: Índice da geometria efetivamente adotada.
    adotada: int = 0
    potencia_kwp: float = 1.0
    avisos: list[str] = field(default_factory=list)

    @property
    def otima(self) -> MedidaDeOrientacao:
        return self.medidas[self.referencia]

    @property
    def escolhida(self) -> MedidaDeOrientacao:
        return self.medidas[self.adotada]

    @property
    def perda(self) -> float:
        """Quanto a geometria adotada perde contra a ótima, em fração."""
        base = self.otima.anual_kwh_por_kwp
        return (self.escolhida.anual_kwh_por_kwp / base - 1.0) if base else float("nan")

    def tabela(self) -> pd.DataFrame:
        linhas = []
        for i, medida in enumerate(self.medidas):
            base = self.otima.anual_kwh_por_kwp
            linhas.append({
                **medida.as_dict(),
                "contra_otima": (medida.anual_kwh_por_kwp / base - 1.0) if base else 0.0,
                "referencia": i == self.referencia,
                "adotada": i == self.adotada,
            })
        return pd.DataFrame(linhas)


def comparar_orientacoes(
    latitude: float,
    longitude: float,
    casos: Sequence[tuple[str, float, float]],
    potencia_kwp: float = 1.0,
    anos: tuple[int, int] = (2016, 2020),
    referencia: int = 0,
    adotada: int | None = None,
    obter_serie=None,
) -> ComparacaoDeOrientacao:
    """
    Mede cada geometria na série horária real do local.

    ``casos`` é uma lista de ``(rótulo, inclinação, azimute)`` na convenção
    interna — 0 grau de azimute é Norte. A primeira entrada é a referência por
    padrão, e deve ser o plano ótimo: é contra ele que a perda faz sentido.

    Sem série confiável a comparação não é feita. Uma série sintética descreve
    razoavelmente o total do ano e descreve mal a geometria, que é exatamente o
    que se está medindo aqui — e a inversão sazonal, que é o achado, sairia
    errada.
    """
    if obter_serie is None:
        from ..bateria.geracao import obter_serie_horaria as obter_serie

    medidas: list[MedidaDeOrientacao] = []
    avisos: list[str] = []
    for rotulo, inclinacao, azimute in casos:
        serie = obter_serie(
            latitude, longitude, azimute_deg=azimute, inclinacao_deg=inclinacao,
            anos=anos,
        )
        if not serie.confiavel:
            avisos.append(
                f"A série de {rotulo} não veio do PVGIS. A comparação entre "
                "geometrias precisa de série medida: a sintética descreve o total "
                "do ano razoavelmente e descreve mal a inclinação, que é o que se "
                "está medindo."
            )
        meses = np.array([d.month for d in serie.datas])
        n_anos = max(1, len({d.year for d in serie.datas}))
        diario_kwp = serie.potencia_w_por_kwp.sum(axis=1) / 1000.0
        mensal = np.array([
            diario_kwp[meses == m].sum() / n_anos for m in range(1, 13)
        ])
        diario = diario_kwp * float(potencia_kwp)
        medidas.append(MedidaDeOrientacao(
            rotulo=rotulo,
            inclinacao_deg=float(inclinacao),
            azimute_deg=float(azimute),
            anual_kwh_por_kwp=float(serie.anual_kwh_por_kwp()),
            mensal_kwh_por_kwp=mensal,
            diario_medio_kwh=float(diario.mean()),
            diario_p05_kwh=float(np.percentile(diario, 5)),
            fonte=serie.fonte,
        ))

    comparacao = ComparacaoDeOrientacao(
        medidas=medidas,
        referencia=referencia,
        adotada=referencia if adotada is None else adotada,
        potencia_kwp=float(potencia_kwp),
        avisos=avisos,
    )
    escolhida = comparacao.escolhida
    if escolhida.inverte:
        avisos.append(
            f"A geometria adotada ({escolhida.rotulo}) inverte a sazonalidade da "
            f"geração: ela produz {escolhida.razao_inverno_verao:.1f} vezes mais no "
            "inverno que no verão, enquanto o plano ótimo produz menos no inverno "
            "que no verão. Isso muda o que o sistema serve, e não só quanto ele "
            "rende — vale conferir contra a estação em que a casa mais consome."
        )
    return comparacao
