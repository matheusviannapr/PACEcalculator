"""
Quanto banco cada meta de autonomia exige — e o que a meta seguinte custa.

O estudo dimensiona para uma meta só, e a meta é uma premissa que quase nunca é
discutida: alguém escreve 6 h no começo e o documento inteiro sai dali. Mas 6 h
e 24 h respondem a apagões diferentes, e a diferença entre elas é a conversa
comercial que de fato acontece.

* **6 horas** atravessa o apagão comum brasileiro — o poste, o transformador, a
  manobra da distribuidora. É o que resolve a esmagadora maioria das faltas.
* **24 horas** é o temporal que derruba a rede de uma região inteira. Não é uma
  meta maior: é outro evento, com outra frequência e outro valor para o cliente.

Apresentar as duas, com o preço de cada uma, transforma uma premissa escondida
numa escolha informada. É a mesma lógica dos escopos de criticidade e dos
cenários de uso: o estudo não escolhe, mostra.

**A resolução da malha limita a resposta.** A autonomia garantida é medida nos
degraus da malha de apagões — tipicamente 1, 2, 3, 6, 12 e 24 h. Um banco que
"garante 12 h" pode garantir 17, e a malha não sabe dizer. Para a decisão isso
basta, porque ninguém compra bateria para 17 h; para a leitura, não: por isso a
tabela declara os degraus que usou.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import pandas as pd

from .apagao import MalhaApagao, avaliar_conjunto
from .catalogo import ConjuntoArmazenamento
from .economia import PremissasBateria, _capex

__all__ = [
    "MedidaDeAutonomia",
    "ComparacaoDeAutonomia",
    "comparar_metas_de_autonomia",
]


@dataclass
class MedidaDeAutonomia:
    """Um banco, e até onde ele leva."""

    conjunto: ConjuntoArmazenamento
    autonomia_h: float
    capex_brl: float
    #: A menor meta, entre as pedidas, que este banco cumpre.
    meta_h: float | None = None

    @property
    def blocos(self) -> int:
        return int(self.conjunto.modulos)

    def as_dict(self) -> dict[str, Any]:
        return {
            "blocos": self.blocos,
            "energia_util_kwh": round(self.conjunto.energia_util_kwh, 1),
            "potencia_kw": round(self.conjunto.potencia_descarga_kw, 1),
            "autonomia_h": round(self.autonomia_h, 1),
            "capex_brl": round(self.capex_brl, 0),
            "meta_h": self.meta_h,
        }


@dataclass
class ComparacaoDeAutonomia:
    """As metas lado a lado, com o degrau de preço entre elas."""

    medidas: list[MedidaDeAutonomia]
    metas_h: tuple[float, ...]
    duracoes_da_malha: tuple[float, ...] = ()
    avisos: list[str] = field(default_factory=list)

    def para(self, meta_h: float) -> MedidaDeAutonomia | None:
        """O banco mais barato que cumpre esta meta."""
        candidatos = [m for m in self.medidas if m.autonomia_h >= meta_h]
        return min(candidatos, key=lambda m: m.capex_brl) if candidatos else None

    def tabela(self) -> pd.DataFrame:
        linhas = []
        for meta in self.metas_h:
            medida = self.para(meta)
            if medida is None:
                linhas.append({"meta_h": meta, "atendida": False})
                continue
            linhas.append({"meta_h": meta, "atendida": True, **medida.as_dict()})
        return pd.DataFrame(linhas)

    def degrau(self, de_h: float, para_h: float) -> dict[str, Any] | None:
        """
        O que custa subir de uma meta para a outra.

        É o número da conversa comercial, e ele quase nunca é proporcional: a
        autonomia cresce em degraus de bloco, e um bloco a mais pode dobrar a
        autonomia ou não mover nada, conforme onde a meta cai entre dois
        degraus.
        """
        a, b = self.para(de_h), self.para(para_h)
        if a is None or b is None:
            return None
        return {
            "de_h": de_h,
            "para_h": para_h,
            "blocos_a_mais": b.blocos - a.blocos,
            "capex_a_mais_brl": b.capex_brl - a.capex_brl,
            "percentual": (b.capex_brl / a.capex_brl - 1.0) if a.capex_brl else float("nan"),
        }


def comparar_metas_de_autonomia(
    candidatos: Sequence[ConjuntoArmazenamento],
    ensemble_backup,
    serie,
    malha: MalhaApagao,
    metas_h: Sequence[float] = (6.0, 24.0),
    confiabilidade: float = 0.95,
    exigir_pior_caso: bool = True,
    premissas: PremissasBateria | None = None,
    semente: int = 20260902,
    max_candidatos: int = 8,
) -> ComparacaoDeAutonomia:
    """
    Mede cada banco candidato e diz qual é o mais barato para cada meta.

    A autonomia é a garantida com ``confiabilidade`` **no pior par estação/hora**
    quando ``exigir_pior_caso``. Medir na média é sempre mais generoso e não
    descreve apagão nenhum: o apagão acontece numa hora e numa estação, e é
    contra essa hora que o banco precisa ter sido comprado.

    Os candidatos entram ordenados por preço e são truncados em
    ``max_candidatos`` — a avaliação de apagões é a parte cara do estudo, e
    bancos muito acima da maior meta não acrescentam informação nenhuma.
    """
    premissas = premissas or PremissasBateria()
    avisos: list[str] = []

    ordenados = sorted(candidatos, key=lambda c: _capex(c, premissas)[0])
    if max_candidatos > 0:
        ordenados = ordenados[:max_candidatos]

    medidas: list[MedidaDeAutonomia] = []
    for conjunto in ordenados:
        resultado = avaliar_conjunto(conjunto, ensemble_backup, serie, malha, semente)
        medidas.append(MedidaDeAutonomia(
            conjunto=conjunto,
            autonomia_h=resultado.autonomia_garantida_h(confiabilidade, exigir_pior_caso),
            capex_brl=_capex(conjunto, premissas)[0],
        ))

    metas = tuple(sorted(float(m) for m in metas_h))
    comparacao = ComparacaoDeAutonomia(
        medidas=medidas, metas_h=metas,
        duracoes_da_malha=tuple(float(d) for d in malha.duracoes_h),
        avisos=avisos,
    )
    for medida in medidas:
        atendidas = [m for m in metas if medida.autonomia_h >= m]
        medida.meta_h = max(atendidas) if atendidas else None

    for meta in metas:
        if comparacao.para(meta) is None:
            avisos.append(
                f"Nenhum banco avaliado atravessa {meta:g} h com "
                f"{confiabilidade:.0%} de confiabilidade"
                f"{' no pior par estação/hora' if exigir_pior_caso else ''}. "
                "Ou a carga essencial precisa encolher, ou o banco precisa passar "
                "do que o catálogo cobre."
            )

    maior = max(malha.duracoes_h) if malha.duracoes_h else 0.0
    if metas and max(metas) >= maior:
        avisos.append(
            f"A malha de apagões vai até {maior:g} h, então a autonomia medida "
            f"satura aí: um banco que aparece com {maior:g} h pode ir além, e a "
            "malha não tem como dizer quanto. Para a decisão isso basta; para a "
            "leitura, é ressalva."
        )
    return comparacao
