"""
Excedência de pico contra o limite do inversor.

Este é o módulo que responde a pergunta direta: *"com este inversor, qual a
probabilidade de a carga pedir mais do que ele entrega — e, quando pedir, por
quanto tempo?"*

Três grandezas diferentes, que costumam ser confundidas numa só:

1. **P(pico diário > limite)** — a chance de o dia ter *algum* instante acima
   do limite. É a estatística de dimensionamento clássica, e é a que responde
   "com que frequência isso acontece": um valor de 0,30 quer dizer que três em
   cada dez dias têm pelo menos um momento de saturação.
2. **Fração do tempo acima do limite** — quanto do dia, em média, a carga fica
   estourada. Costuma ser uma ordem de grandeza menor que a anterior, e é a que
   diz se o problema é um pico afiado ou um patamar.
3. **Duração do evento** — quantos minutos seguidos dura cada estouro. É o que
   decide entre "o inversor corta um chuveiro por 4 minutos" e "o inversor não
   dá conta do jantar inteiro". Também é o que se compara contra
   ``duracao_pico_s``: um evento de 40 minutos não é surto, é regime.

A distinção entre nominal e pico é o coração da coisa. Um inversor de 5 kW com
sobrecarga de 10 kW por 10 s atende sem tossir uma partida de motor que puxa
8 kW por 2 s, e não atende um chuveiro de 5,5 kW ligado por 8 minutos — o
segundo caso tem potência menor e é o que derruba. Nenhum número único de
"potência" separa os dois; por isso a avaliação sai em três colunas.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..demanda.ensemble import CurvaExcedencia, EnsembleCarga, estatisticas_de_eventos
from .catalogo import BaseBaterias, InversorHibrido

__all__ = [
    "DiagnosticoInversor",
    "avaliar_inversores",
    "diagnosticar_inversor",
    "potencia_para_excedencia",
    "tabela_por_faixa_horaria",
]

#: Acima desta probabilidade de estourar a **sobrecarga**, o inversor é
#: reprovado: significa desarme em mais de 1% dos dias, o que num backup se
#: traduz em "a luz caiu mesmo com a bateria cheia" algumas vezes por ano.
LIMITE_REPROVACAO = 0.01

#: Abaixo desta probabilidade de estourar o **nominal**, considera-se folga.
LIMITE_FOLGA = 0.02


@dataclass(frozen=True)
class DiagnosticoInversor:
    """O que se sabe sobre um inversor diante de uma curva de carga."""

    inversor: InversorHibrido
    prob_pico_diario_acima_nominal: float
    prob_pico_diario_acima_surto: float
    fracao_do_tempo_acima_nominal: float
    eventos_por_dia: float
    duracao_media_min: float
    duracao_p95_min: float
    duracao_max_min: float
    energia_media_por_evento_kwh: float
    dias_com_evento_percent: float
    margem_sobre_p95_percent: float
    fracao_eventos_dentro_do_surto: float

    @property
    def veredito(self) -> str:
        """Uma frase que diz o que fazer, não só o que aconteceu."""
        if self.prob_pico_diario_acima_surto > LIMITE_REPROVACAO:
            return (
                f"reprovado — estoura a sobrecarga de {self.inversor.potencia_ca_pico_kw:.1f} kW "
                f"em {self.prob_pico_diario_acima_surto:.1%} dos dias"
            )
        if self.prob_pico_diario_acima_nominal <= LIMITE_FOLGA:
            return f"atende com folga — margem de {self.margem_sobre_p95_percent:+.0f}% sobre o pico P95"
        if self.fracao_eventos_dentro_do_surto >= 0.95:
            return (
                f"atende pela sobrecarga — satura em {self.prob_pico_diario_acima_nominal:.0%} dos dias, "
                f"eventos de {self.duracao_media_min:.0f} min em média, todos cobertos pelo surto"
            )
        return (
            f"atende com corte — {self.fracao_do_tempo_acima_nominal:.1%} do tempo acima do nominal, "
            f"eventos de {self.duracao_media_min:.0f} min (P95 {self.duracao_p95_min:.0f} min); "
            f"exige seletividade de carga"
        )

    @property
    def aprovado(self) -> bool:
        return self.prob_pico_diario_acima_surto <= LIMITE_REPROVACAO

    def as_dict(self) -> dict[str, Any]:
        return {
            "fabricante": self.inversor.fabricante,
            "modelo": self.inversor.modelo,
            "nominal_kw": self.inversor.potencia_ca_nominal_kw,
            "pico_kw": self.inversor.potencia_ca_pico_kw,
            "duracao_pico_s": self.inversor.duracao_pico_s,
            "fases": self.inversor.fases,
            "prob_pico_diario_acima_nominal": self.prob_pico_diario_acima_nominal,
            "prob_pico_diario_acima_surto": self.prob_pico_diario_acima_surto,
            "fracao_do_tempo_acima_nominal": self.fracao_do_tempo_acima_nominal,
            "eventos_por_dia": self.eventos_por_dia,
            "duracao_media_min": self.duracao_media_min,
            "duracao_p95_min": self.duracao_p95_min,
            "duracao_max_min": self.duracao_max_min,
            "energia_media_por_evento_kwh": self.energia_media_por_evento_kwh,
            "dias_com_evento_percent": self.dias_com_evento_percent,
            "margem_sobre_p95_percent": self.margem_sobre_p95_percent,
            "fracao_eventos_dentro_do_surto": self.fracao_eventos_dentro_do_surto,
            "aprovado": self.aprovado,
            "veredito": self.veredito,
        }


def diagnosticar_inversor(
    inversor: InversorHibrido,
    ensemble: EnsembleCarga,
    estacao: str | None = None,
) -> DiagnosticoInversor:
    """Cruza um inversor com o ensemble de carga."""
    serie = ensemble.picos_janela(estacao)
    curva_pico = ensemble.curva_excedencia_pico(estacao)
    nominal_w = inversor.potencia_ca_nominal_kw * 1000.0
    surto_w = inversor.potencia_ca_pico_kw * 1000.0

    eventos = estatisticas_de_eventos(serie, nominal_w, ensemble.passo_min)
    p95 = curva_pico.percentil(95)

    # Dos eventos que passam do nominal, quantos ficam abaixo do surto e são
    # curtos o bastante para o surto cobrir? A segunda condição é o que impede
    # de creditar ao inversor uma sobrecarga que ele não sustenta pelo tempo
    # que o evento dura.
    eventos_surto = estatisticas_de_eventos(serie, surto_w, ensemble.passo_min)
    duracao_surto_min = inversor.duracao_pico_s / 60.0
    if eventos["eventos_por_dia"] <= 0:
        dentro = 1.0
    else:
        # Fração dos eventos acima do nominal que nem estouram o surto nem
        # duram mais do que a sobrecarga aguenta.
        fora_por_amplitude = eventos_surto["eventos_por_dia"] / eventos["eventos_por_dia"]
        fora_por_duracao = 0.0 if eventos["duracao_media_min"] <= duracao_surto_min else 1.0
        dentro = float(np.clip(1.0 - max(fora_por_amplitude, fora_por_duracao), 0.0, 1.0))

    return DiagnosticoInversor(
        inversor=inversor,
        prob_pico_diario_acima_nominal=curva_pico.prob_excedencia(nominal_w),
        prob_pico_diario_acima_surto=curva_pico.prob_excedencia(surto_w),
        fracao_do_tempo_acima_nominal=eventos["fracao_do_tempo"],
        eventos_por_dia=eventos["eventos_por_dia"],
        duracao_media_min=eventos["duracao_media_min"],
        duracao_p95_min=eventos["duracao_p95_min"],
        duracao_max_min=eventos["duracao_max_min"],
        energia_media_por_evento_kwh=eventos["energia_media_por_evento_kwh"],
        dias_com_evento_percent=eventos["dias_com_evento_percent"],
        margem_sobre_p95_percent=(nominal_w / p95 - 1.0) * 100.0 if p95 > 0 else float("inf"),
        fracao_eventos_dentro_do_surto=dentro,
    )


def avaliar_inversores(
    base: BaseBaterias,
    ensemble: EnsembleCarga,
    estacao: str | None = None,
    apenas: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Tabela de diagnóstico para o catálogo inteiro, do menor para o maior."""
    inversores = base.inversores_ordenados()
    if apenas:
        alvos = {m.strip().lower() for m in apenas}
        inversores = [i for i in inversores if i.modelo.lower() in alvos]
    linhas = [diagnosticar_inversor(i, ensemble, estacao).as_dict() for i in inversores]
    return pd.DataFrame(linhas)


def potencia_para_excedencia(
    ensemble: EnsembleCarga,
    probabilidades: Sequence[float] = (0.50, 0.20, 0.10, 0.05, 0.02, 0.01, 0.001),
    estacao: str | None = None,
) -> pd.DataFrame:
    """
    A leitura inversa da curva: qual potência é excedida com cada probabilidade.

    É a tabela de dimensionamento propriamente dita. "0,05" na coluna de pico
    diário significa: um inversor desta potência é insuficiente em cinco de
    cada cem dias.
    """
    pico = ensemble.curva_excedencia_pico(estacao)
    instantanea = ensemble.curva_excedencia_instantanea(estacao)
    return pd.DataFrame(
        [
            {
                "prob_excedencia": float(p),
                "pico_diario_kw": pico.valor_para_probabilidade(p) / 1000.0,
                "carga_instantanea_kw": instantanea.valor_para_probabilidade(p) / 1000.0,
            }
            for p in probabilidades
        ]
    )


def tabela_por_faixa_horaria(
    ensemble: EnsembleCarga,
    potencias_kw: Sequence[float],
    estacao: str | None = None,
) -> pd.DataFrame:
    """
    Excedência hora a hora — "se faltar luz às 19 h, o que este inversor faz?"

    Cada célula é a fração do tempo daquela hora em que a carga passa da
    potência da coluna. A leitura útil é a linha: um inversor que fica limpo o
    dia todo e satura das 18 h às 21 h é um caso de deslocamento de carga, não
    de inversor maior.
    """
    linhas = []
    for hora in range(24):
        curva = ensemble.curva_excedencia_por_hora(hora, estacao)
        linha: dict[str, Any] = {"hora": hora, "carga_media_kw": curva.media / 1000.0}
        for potencia in potencias_kw:
            linha[f"P>{potencia:g}kW"] = curva.prob_excedencia(float(potencia) * 1000.0)
        linhas.append(linha)
    return pd.DataFrame(linhas)


def curvas_para_grafico(
    ensemble: EnsembleCarga,
    por_estacao: bool = True,
) -> dict[str, CurvaExcedencia]:
    """Curvas prontas para o relatório: a agregada e, opcionalmente, uma por estação."""
    curvas = {"todas as estações": ensemble.curva_excedencia_pico()}
    if por_estacao:
        for estacao in ensemble.estacoes:
            curvas[estacao] = ensemble.curva_excedencia_pico(estacao)
    return curvas
