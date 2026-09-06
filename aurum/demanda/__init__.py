"""
Modelagem probabilística de demanda elétrica — núcleo do D² tornado biblioteca.

* :mod:`aurum.demanda.nucleo` — extração literal do simulador Monte Carlo do
  repositório ``matheusviannapr/DemandaDados``, sem a camada Streamlit.
* :mod:`aurum.demanda.ensemble` — o que o estudo de baterias consome: curvas
  diárias por estação, curvas de excedência e análise de duração de eventos.
"""
from .correcoes import intervalos_circulares
from .ensemble import (
    CurvaExcedencia,
    EnsembleCarga,
    carregar_cenario_excel,
    cenario_exemplo,
    estatisticas_de_eventos,
    simular_ensemble,
)

__all__ = [
    "CurvaExcedencia",
    "EnsembleCarga",
    "carregar_cenario_excel",
    "cenario_exemplo",
    "estatisticas_de_eventos",
    "intervalos_circulares",
    "simular_ensemble",
]
