"""
A malha de apagões: duração × hora de início × estação.

A pergunta "essa bateria aguenta uma falta de luz?" não tem resposta, porque
não é uma pergunta só. Aguentar 1 h às 3 da manhã de um domingo de junho e
aguentar 36 h a partir das 18 h de uma quinta de janeiro são problemas
diferentes por três motivos simultâneos:

* **a duração** determina a energia pedida;
* **a hora de início** determina quanto sol vem antes do banco esvaziar — um
  apagão que começa às 6 h tem o dia inteiro de recarga pela frente, um que
  começa às 18 h tem a noite inteira antes de ver o primeiro raio;
* **a estação** mexe nos dois lados ao mesmo tempo: o inverno encurta o dia
  solar e (dependendo da carga) muda o perfil de consumo.

Este módulo varre as três dimensões, roda o despacho para cada combinação com
``amostras`` sorteios de Monte Carlo e devolve uma tabela. O resultado que
interessa raramente é a média sobre tudo: é a célula específica — "36 h
começando às 18 h no inverno" — que decide a venda.

Todos os cenários de uma mesma (estação, duração) são simulados numa chamada
só, com as 24 horas de início empilhadas no mesmo lote. É o que faz a varredura
inteira de um conjunto caber em segundos em vez de minutos.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..demanda.ensemble import EnsembleCarga
from .catalogo import ConjuntoArmazenamento
from .despacho import (
    CAUSA_ENERGIA,
    CAUSA_POTENCIA,
    Gerador,
    LimitesDespacho,
    ResultadoDespacho,
    simular_ilhamento,
)
from .geracao import ESTACOES, SerieGeracao

__all__ = [
    "MalhaApagao",
    "ResultadoResiliencia",
    "avaliar_conjunto",
    "avaliar_limites",
    "varrer_conjuntos",
]

#: Durações pedidas no escopo do estudo. 6 h entra entre 3 e 12 porque é onde
#: a curva de atendimento costuma virar: até 3 h quase tudo passa, de 12 h em
#: diante quase nada passa sem sol.
DURACOES_PADRAO = (1.0, 2.0, 3.0, 6.0, 12.0, 24.0, 36.0)


@dataclass(frozen=True)
class MalhaApagao:
    """Definição da varredura."""

    duracoes_h: tuple[float, ...] = DURACOES_PADRAO
    horas_inicio: tuple[int, ...] = tuple(range(24))
    estacoes: tuple[str, ...] = ESTACOES
    amostras: int = 200
    passo_min: int = 5
    soc_inicial_frac: float = 1.0

    @property
    def cenarios_totais(self) -> int:
        return (
            len(self.duracoes_h) * len(self.horas_inicio) * len(self.estacoes) * self.amostras
        )

    def descricao(self) -> str:
        return (
            f"{len(self.duracoes_h)} durações × {len(self.horas_inicio)} horas × "
            f"{len(self.estacoes)} estações × {self.amostras} amostras "
            f"= {self.cenarios_totais:,} cenários (passo de {self.passo_min} min)"
        ).replace(",", ".")


# ----------------------------------------------------------------------------
# Montagem do lote
# ----------------------------------------------------------------------------
def _lote_de_carga(
    ensemble: EnsembleCarga,
    estacao: str,
    horas: Sequence[int],
    n_passos: int,
    amostras: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Empilha ``len(horas) × amostras`` janelas de carga, em kW.

    Apagões que passam da meia-noite recebem **um novo dia sorteado**, não a
    repetição do primeiro. O D² trata cada dia como um sorteio independente; um
    apagão de 36 h que reaproveitasse o mesmo dia duas vezes duplicaria o pior
    caso e o melhor caso junto, estreitando artificialmente a distribuição.
    """
    passos_dia = ensemble.passos_por_dia
    por_hora = passos_dia // 24
    lotes_carga: list[np.ndarray] = []
    lotes_pico: list[np.ndarray] = []

    for hora in horas:
        inicio = int(hora) * por_hora
        dias = int(math.ceil((inicio + n_passos) / passos_dia))
        media, pico = ensemble.amostrar_dias(estacao, amostras * dias, rng)
        media = media.reshape(amostras, dias * passos_dia)
        pico = pico.reshape(amostras, dias * passos_dia)
        lotes_carga.append(media[:, inicio : inicio + n_passos])
        lotes_pico.append(pico[:, inicio : inicio + n_passos])

    return (
        np.concatenate(lotes_carga, axis=0) / 1000.0,
        np.concatenate(lotes_pico, axis=0) / 1000.0,
    )


def _lote_de_geracao(
    serie: SerieGeracao,
    estacao: str,
    horas: Sequence[int],
    n_passos: int,
    passo_min: int,
    amostras: int,
    potencia_fv_kwp: float,
    rng: np.random.Generator,
) -> np.ndarray:
    lotes = [
        serie.amostrar_janelas(estacao, int(hora), n_passos, passo_min, amostras, rng)
        for hora in horas
    ]
    return np.concatenate(lotes, axis=0) * float(potencia_fv_kwp) / 1000.0


# ----------------------------------------------------------------------------
# Resultado
# ----------------------------------------------------------------------------
@dataclass
class ResultadoResiliencia:
    """Tabela da varredura, com os cortes que respondem as perguntas usuais."""

    #: ``None`` nos cenários que não têm banco -- só gerador, ou nada. A
    #: varredura é a mesma; o que muda é quem entrega a energia.
    conjunto: ConjuntoArmazenamento | None
    malha: MalhaApagao
    tabela: pd.DataFrame
    limites: LimitesDespacho
    gerador: Gerador | None = None
    metadados: dict[str, Any] = field(default_factory=dict)

    # -- cortes ------------------------------------------------------------
    def por_duracao(self) -> pd.DataFrame:
        """Média sobre horas e estações — a visão de folheto, útil como abertura."""
        return (
            self.tabela.groupby("duracao_h")
            .agg(
                prob_atendimento=("prob_atendimento", "mean"),
                pior_prob=("prob_atendimento", "min"),
                ens_medio_kwh=("ens_medio_kwh", "mean"),
                ens_p95_kwh=("ens_p95_kwh", "max"),
                energia_gerador_kwh=("energia_gerador_media_kwh", "mean"),
                falhas_por_potencia=("falhas_por_potencia", "mean"),
                falhas_por_energia=("falhas_por_energia", "mean"),
                minutos_ate_falha_mediano=("minutos_ate_falha_mediano", "mean"),
            )
            .reset_index()
        )

    def por_duracao_e_estacao(self) -> pd.DataFrame:
        return (
            self.tabela.groupby(["estacao", "duracao_h"])
            .agg(
                prob_atendimento=("prob_atendimento", "mean"),
                pior_prob=("prob_atendimento", "min"),
                ens_medio_kwh=("ens_medio_kwh", "mean"),
            )
            .reset_index()
        )

    def matriz_hora_duracao(self, estacao: str | None = None, metrica: str = "prob_atendimento") -> pd.DataFrame:
        """Mapa de calor cru: linhas = hora de início, colunas = duração."""
        base = self.tabela if estacao is None else self.tabela[self.tabela["estacao"] == estacao]
        return base.pivot_table(index="hora_inicio", columns="duracao_h", values=metrica, aggfunc="mean")

    def pior_janela(self, duracao_h: float) -> pd.Series:
        """A combinação (estação, hora) de menor probabilidade para a duração."""
        recorte = self.tabela[np.isclose(self.tabela["duracao_h"], duracao_h)]
        if recorte.empty:
            raise ValueError(f"duração {duracao_h} h não está na malha")
        return recorte.loc[recorte["prob_atendimento"].idxmin()]

    def autonomia_garantida_h(self, confiabilidade: float = 0.95, pior_caso: bool = True) -> float:
        """
        A maior duração que o conjunto atravessa com a confiabilidade pedida.

        Com ``pior_caso=True`` a exigência vale para **toda** combinação de hora
        e estação — que é o critério defensável numa proposta. Com ``False``,
        vale para a média, que é sempre mais generosa e não descreve nenhum
        apagão em particular.
        """
        coluna = "pior_prob" if pior_caso else "prob_atendimento"
        resumo = self.por_duracao().sort_values("duracao_h")
        atendidas = resumo[resumo[coluna] >= float(confiabilidade)]["duracao_h"]
        return float(atendidas.max()) if not atendidas.empty else 0.0

    def resumo(self) -> dict[str, Any]:
        por_duracao = self.por_duracao()
        return {
            "conjunto": self.conjunto.descricao() if self.conjunto else "sem banco",
            **(self.conjunto.as_dict() if self.conjunto else {}),
            "gerador": self.gerador.as_dict() if self.gerador else None,
            "autonomia_garantida_95_h": self.autonomia_garantida_h(0.95),
            "autonomia_garantida_99_h": self.autonomia_garantida_h(0.99),
            "autonomia_media_95_h": self.autonomia_garantida_h(0.95, pior_caso=False),
            "prob_por_duracao": {
                float(linha.duracao_h): float(linha.prob_atendimento)
                for linha in por_duracao.itertuples()
            },
            "pior_prob_por_duracao": {
                float(linha.duracao_h): float(linha.pior_prob)
                for linha in por_duracao.itertuples()
            },
            "desarmes_por_surto_medio": float(self.tabela["desarmes_por_surto_medio"].mean()),
            "cenarios_com_desarme": float(self.tabela["cenarios_com_desarme"].mean()),
        }


# ----------------------------------------------------------------------------
# Execução
# ----------------------------------------------------------------------------
def avaliar_limites(
    limites: LimitesDespacho,
    potencia_fv_kw: float,
    ensemble: EnsembleCarga,
    serie: SerieGeracao,
    malha: MalhaApagao | None = None,
    semente: int = 20260902,
    gerador: Gerador | None = None,
    conjunto: ConjuntoArmazenamento | None = None,
    progresso=None,
    rotulo: str = "",
) -> ResultadoResiliencia:
    """
    Varre a malha para uma combinação qualquer de fontes.

    É o corpo que :func:`avaliar_conjunto` usa, exposto sem a exigência de um
    conjunto de armazenamento: a comparação de fontes precisa medir a
    resiliência de arranjos que não têm banco nenhum -- só gerador, ou nada --
    e não faz sentido inventar um ``ConjuntoArmazenamento`` de zero kWh só
    para atravessar a assinatura.

    ``potencia_fv_kw`` é a potência FV que o ilhamento pode usar. Zero
    representa tanto a ausência de sol quanto o inversor conectado à rede que
    desliga no apagão -- em backup, dá na mesma.
    """
    malha = malha or MalhaApagao()
    if ensemble.passo_min != malha.passo_min:
        ensemble = ensemble.reamostrar(malha.passo_min)

    rng = np.random.default_rng(semente)
    horas = tuple(int(h) for h in malha.horas_inicio)
    linhas: list[dict[str, Any]] = []
    total = len(malha.estacoes) * len(malha.duracoes_h)
    feitos = 0

    for estacao in malha.estacoes:
        for duracao in malha.duracoes_h:
            n_passos = int(round(duracao * 60 / malha.passo_min))
            carga, pico = _lote_de_carga(ensemble, estacao, horas, n_passos, malha.amostras, rng)
            geracao = _lote_de_geracao(
                serie, estacao, horas, n_passos, malha.passo_min,
                malha.amostras, potencia_fv_kw, rng,
            )
            resultado = simular_ilhamento(
                carga, pico, geracao, limites, malha.passo_min, malha.soc_inicial_frac,
                gerador=gerador,
            )
            linhas.extend(_dividir_por_hora(resultado, horas, malha, estacao, duracao))
            feitos += 1
            if progresso is not None:
                progresso(f"{rotulo} — {estacao}, {duracao:g} h", feitos / total)

    return ResultadoResiliencia(
        conjunto=conjunto,
        malha=malha,
        tabela=pd.DataFrame(linhas),
        limites=limites,
        gerador=gerador,
        metadados={"fonte_geracao": serie.fonte, "semente": semente},
    )


def avaliar_conjunto(
    conjunto: ConjuntoArmazenamento,
    ensemble: EnsembleCarga,
    serie: SerieGeracao,
    malha: MalhaApagao | None = None,
    semente: int = 20260902,
    fator_capacidade: float = 1.0,
    progresso=None,
    gerador: Gerador | None = None,
) -> ResultadoResiliencia:
    """
    Varre a malha inteira para um conjunto e devolve a tabela de resiliência.

    ``fator_capacidade`` reduz a energia útil para representar o banco
    envelhecido — é por onde :mod:`aurum.bateria.degradacao` reavalia a
    confiabilidade daqui a cinco ou dez anos sem duplicar nada deste código.
    """
    limites = LimitesDespacho.do_conjunto(conjunto)
    if fator_capacidade != 1.0:
        limites = replace(
            limites, energia_util_dc_kwh=limites.energia_util_dc_kwh * float(fator_capacidade)
        )

    resultado = avaliar_limites(
        limites,
        conjunto.potencia_fv_aproveitavel_kw,
        ensemble,
        serie,
        malha,
        semente,
        gerador=gerador,
        conjunto=conjunto,
        progresso=progresso,
        rotulo=conjunto.inversor.modelo,
    )
    resultado.metadados["fator_capacidade"] = float(fator_capacidade)
    return resultado


def _dividir_por_hora(
    resultado: ResultadoDespacho,
    horas: Sequence[int],
    malha: MalhaApagao,
    estacao: str,
    duracao: float,
) -> list[dict[str, Any]]:
    """Desempilha o lote: cada bloco de ``amostras`` linhas é uma hora de início."""
    linhas: list[dict[str, Any]] = []
    for i, hora in enumerate(horas):
        fatia = slice(i * malha.amostras, (i + 1) * malha.amostras)
        atendido = resultado.atendido[fatia]
        ens = resultado.ens_kwh[fatia]
        causa = resultado.causa[fatia]
        minutos = resultado.minutos_ate_falha[fatia]
        falhou = ~atendido
        demandada = resultado.energia_demandada_kwh[fatia]
        linhas.append({
            "estacao": estacao,
            "duracao_h": float(duracao),
            "hora_inicio": int(hora),
            "prob_atendimento": float(atendido.mean()),
            "ens_medio_kwh": float(ens.mean()),
            "ens_p95_kwh": float(np.percentile(ens, 95)),
            "ens_max_kwh": float(ens.max()),
            "energia_demandada_media_kwh": float(demandada.mean()),
            "fracao_energia_atendida": (
                1.0 - float(ens.sum()) / float(demandada.sum()) if demandada.sum() > 0 else 1.0
            ),
            "minutos_ate_falha_mediano": float(np.median(minutos[falhou])) if falhou.any() else float("nan"),
            "minutos_ate_falha_p05": float(np.percentile(minutos[falhou], 5)) if falhou.any() else float("nan"),
            "falhas_por_potencia": float((causa == CAUSA_POTENCIA).mean()),
            "falhas_por_energia": float((causa == CAUSA_ENERGIA).mean()),
            "desarmes_por_surto_medio": float(resultado.eventos_desarme[fatia].mean()),
            "cenarios_com_desarme": float((resultado.eventos_desarme[fatia] > 0).mean()),
            "soc_final_medio": float(resultado.soc_final_frac[fatia].mean()),
            "soc_min_medio": float(resultado.soc_min_frac[fatia].mean()),
            "energia_pv_media_kwh": float(resultado.energia_pv_kwh[fatia].mean()),
            "energia_gerador_media_kwh": float(resultado.energia_gerador_kwh[fatia].mean()),
            "energia_recarregada_media_kwh": float(resultado.energia_recarregada_kwh[fatia].mean()),
            "energia_desperdicada_media_kwh": float(resultado.energia_desperdicada_kwh[fatia].mean()),
        })
    return linhas


def varrer_conjuntos(
    conjuntos: Sequence[ConjuntoArmazenamento],
    ensemble: EnsembleCarga,
    serie: SerieGeracao,
    malha: MalhaApagao | None = None,
    semente: int = 20260902,
    progresso=None,
) -> list[ResultadoResiliencia]:
    """Avalia vários conjuntos com a mesma malha e a mesma semente."""
    malha = malha or MalhaApagao()
    # Reamostrar uma vez só: cada chamada de `reamostrar` copia todo o ensemble.
    ensemble = ensemble.reamostrar(malha.passo_min) if ensemble.passo_min != malha.passo_min else ensemble
    resultados: list[ResultadoResiliencia] = []
    for i, conjunto in enumerate(conjuntos):
        resultados.append(avaliar_conjunto(conjunto, ensemble, serie, malha, semente))
        if progresso is not None:
            progresso(conjunto.descricao(), (i + 1) / len(conjuntos))
    return resultados
