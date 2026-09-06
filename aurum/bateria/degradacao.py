"""
Envelhecimento do banco — e o que ele faz com a confiabilidade.

Uma bateria não falha de repente: ela encolhe. Um estudo que declara "atende
36 h" e não diz em que ano está falando entrega uma promessa que expira sem
avisar. No ano 10 o mesmo banco pode estar em 78% da capacidade original, e a
autonomia que fechava com folga passa a não fechar.

Dois mecanismos, somados porque agem ao mesmo tempo e por caminhos físicos
distintos:

* **Calendário** — perda por tempo, independente de uso. Cresce com a raiz do
  tempo (mecanismo difusivo, dominado pelo crescimento da SEI), o que explica
  por que o primeiro ano tira mais capacidade que o décimo.
* **Ciclagem** — perda por energia processada. Cresce linearmente com o
  throughput acumulado, medido em **ciclos equivalentes**: a energia total
  descarregada dividida pela energia útil de um ciclo completo.

Os dois somam. O modelo é deliberadamente simples e explícito nos parâmetros
— não há ganho em sofisticar a curva de fade quando a incerteza dominante está
no throughput anual, que depende de como o cliente vai operar o sistema.

O que este módulo **não** faz: temperatura. A dependência é forte (a perda de
calendário praticamente dobra a cada 10 °C acima de 25 °C) e modelá-la sem
saber onde o banco vai ser instalado daria uma precisão falsa. Se o banco fica
em casa de máquinas sem ventilação, ``fade_calendario_ano`` é o parâmetro a
dobrar, e o efeito disso aparece inteiro no resultado.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .catalogo import ConjuntoArmazenamento

__all__ = ["ModeloDegradacao", "trajetoria_de_vida"]


@dataclass(frozen=True)
class ModeloDegradacao:
    """Perda de capacidade por calendário e por ciclagem."""

    ciclos_nominais: int = 6000
    retencao_fim_vida: float = 0.80
    #: Perda de calendário no **primeiro** ano, em fração. LFP a 25 °C fica
    #: entre 0,8% e 1,5%; 1% é o valor de catálogo mais comum.
    fade_calendario_ano: float = 0.01
    #: Expoente do tempo. 0,5 = raiz, o comportamento medido em célula.
    expoente_calendario: float = 0.5

    @classmethod
    def do_conjunto(cls, conjunto: ConjuntoArmazenamento, **ajustes: Any) -> "ModeloDegradacao":
        return cls(
            ciclos_nominais=conjunto.bateria.ciclos_vida,
            retencao_fim_vida=conjunto.bateria.retencao_fim_vida_percent / 100.0,
            **ajustes,
        )

    # -- mecanismos --------------------------------------------------------
    def fade_calendario(self, anos: float) -> float:
        return self.fade_calendario_ano * float(max(0.0, anos)) ** self.expoente_calendario

    def fade_ciclagem(self, ciclos_equivalentes: float) -> float:
        perda_total = 1.0 - self.retencao_fim_vida
        return perda_total * float(max(0.0, ciclos_equivalentes)) / max(1, self.ciclos_nominais)

    def retencao(self, anos: float, ciclos_equivalentes: float) -> float:
        """Capacidade restante como fração da original, no piso de zero."""
        perda = self.fade_calendario(anos) + self.fade_ciclagem(ciclos_equivalentes)
        return float(np.clip(1.0 - perda, 0.0, 1.0))

    # -- vida útil ---------------------------------------------------------
    def vida_util_anos(self, ciclos_por_ano: float, limite_anos: int = 30) -> float:
        """
        Ano em que a retenção cruza o fim de vida declarado.

        Interpolado entre anos inteiros: dizer "morre no ano 12" quando o
        cruzamento acontece em março do ano 12 embute quase um ano de erro no
        cronograma de substituição, que é dinheiro no fluxo de caixa.
        """
        anterior = 1.0
        for ano in range(1, limite_anos + 1):
            atual = self.retencao(ano, ciclos_por_ano * ano)
            if atual <= self.retencao_fim_vida:
                if anterior == atual:
                    return float(ano)
                fracao = (anterior - self.retencao_fim_vida) / (anterior - atual)
                return float(ano - 1 + fracao)
            anterior = atual
        return float(limite_anos)


def trajetoria_de_vida(
    modelo: ModeloDegradacao,
    ciclos_por_ano: float,
    energia_util_inicial_kwh: float,
    anos: int = 20,
) -> pd.DataFrame:
    """Tabela ano a ano: ciclos acumulados, retenção e energia útil restante."""
    linhas = []
    for ano in range(0, int(anos) + 1):
        ciclos = ciclos_por_ano * ano
        retencao = modelo.retencao(ano, ciclos)
        linhas.append(
            {
                "ano": ano,
                "ciclos_equivalentes_acumulados": ciclos,
                "retencao": retencao,
                "energia_util_kwh": energia_util_inicial_kwh * retencao,
                "fade_calendario": modelo.fade_calendario(ano),
                "fade_ciclagem": modelo.fade_ciclagem(ciclos),
                "fim_de_vida": retencao <= modelo.retencao_fim_vida,
            }
        )
    return pd.DataFrame(linhas)
