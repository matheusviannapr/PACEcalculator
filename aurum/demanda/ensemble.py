"""
Ponte entre o Monte Carlo do D² e o estudo de baterias.

O D² entrega, por rodada, uma matriz ``(n_simulações, 1440)`` de potência em
watts: uma curva de carga diária por simulação. O estudo de baterias precisa
de três coisas que essa matriz crua não dá de imediato:

1. **Separação por estação.** A bateria que atravessa 12 h de apagão em julho
   não é a mesma que atravessa 12 h em janeiro, porque nem a carga nem o sol
   são os mesmos. O ensemble guarda uma matriz por estação.
2. **Curvas de excedência.** Duas, e elas respondem perguntas diferentes:
   a de **pico diário** dimensiona (qual potência o dia pede, no pior minuto);
   a **instantânea** avalia o inversor (que fração do tempo a carga passa de
   um dado limite, e por quanto tempo seguido).
3. **Reamostragem no passo do despacho.** Simular 36 h de apagão minuto a
   minuto para dezenas de milhares de cenários é caro sem necessidade. Ao
   reamostrar, a energia é preservada pela média e o pico pela máxima da
   janela — as duas séries são mantidas lado a lado, porque a média subestima
   a solicitação do inversor e a máxima superestima a energia.

Unidades: este módulo fala **watts** na fronteira com o D² e expõe
conversões para kW, que é a moeda do resto do :mod:`aurum.bateria`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .correcoes import intervalos_circulares
from .nucleo import (
    ESTACOES_ANO,
    TEMPO_TOTAL_PADRAO_MIN,
    Comodo,
    aplicar_ajuste_sazonal,
    cria_comodos_do_dataframe,
    cria_comodos_do_excel,
    criar_cenario_exemplo,
    simula_carga_total,
)

__all__ = [
    "CurvaExcedencia",
    "EnsembleCarga",
    "carregar_cenario_excel",
    "cenario_exemplo",
    "estatisticas_de_eventos",
    "simular_ensemble",
]


# ----------------------------------------------------------------------------
# Curva de excedência
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class CurvaExcedencia:
    """
    Distribuição complementar empírica: ``P(X > x)``.

    Guardamos as amostras, não uma curva ajustada. Ajustar uma normal ou uma
    Gumbel ao pico de um Monte Carlo de carga é tentador e engana na cauda,
    que é justamente a região que decide o inversor.
    """

    amostras: np.ndarray
    unidade: str = "W"
    rotulo: str = "pico diário"

    @classmethod
    def de_amostras(
        cls,
        amostras: Iterable[float],
        unidade: str = "W",
        rotulo: str = "pico diário",
    ) -> "CurvaExcedencia":
        vetor = np.asarray(amostras, dtype=float).ravel()
        vetor = vetor[np.isfinite(vetor)]
        if vetor.size == 0:
            raise ValueError("curva de excedência exige ao menos uma amostra finita")
        return cls(amostras=np.sort(vetor), unidade=unidade, rotulo=rotulo)

    @property
    def n(self) -> int:
        return int(self.amostras.size)

    def prob_excedencia(self, valor: float) -> float:
        """``P(X > valor)`` — a probabilidade de o limite ser estourado."""
        return float(np.mean(self.amostras > float(valor)))

    def prob_excedencia_vetor(self, valores: Sequence[float]) -> np.ndarray:
        alvos = np.asarray(valores, dtype=float)
        # searchsorted no vetor já ordenado: O(log n) por alvo em vez de O(n).
        acima = self.amostras.size - np.searchsorted(self.amostras, alvos, side="right")
        return acima / self.amostras.size

    def valor_para_probabilidade(self, prob_excedencia: float) -> float:
        """Inverso: o valor que é excedido com a probabilidade dada."""
        p = float(prob_excedencia)
        if not 0.0 <= p <= 1.0:
            raise ValueError("probabilidade de excedência deve estar em [0, 1]")
        return float(np.quantile(self.amostras, 1.0 - p))

    def percentil(self, q: float) -> float:
        return float(np.percentile(self.amostras, q))

    @property
    def media(self) -> float:
        return float(np.mean(self.amostras))

    @property
    def desvio(self) -> float:
        return float(np.std(self.amostras))

    def pontos(self, n: int = 200) -> tuple[np.ndarray, np.ndarray]:
        """Eixo x e ``P(X > x)`` prontos para plotar."""
        x = np.quantile(self.amostras, np.linspace(0.0, 1.0, n))
        return x, self.prob_excedencia_vetor(x)

    def resumo(self) -> dict[str, float]:
        return {
            "n": float(self.n),
            "media": self.media,
            "desvio": self.desvio,
            "p50": self.percentil(50),
            "p90": self.percentil(90),
            "p95": self.percentil(95),
            "p99": self.percentil(99),
            "maximo": float(self.amostras[-1]),
        }


# ----------------------------------------------------------------------------
# Ensemble sazonal
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class EnsembleCarga:
    """
    Biblioteca de curvas diárias de carga, uma matriz por estação.

    ``perfis_w[estacao]`` tem forma ``(n_simulacoes, n_passos)`` em watts.
    ``picos_janela_w`` guarda, no mesmo formato, a **máxima** de cada janela de
    reamostragem; com ``passo_min == 1`` as duas matrizes são idênticas.
    """

    perfis_w: Mapping[str, np.ndarray]
    picos_janela_w: Mapping[str, np.ndarray] = field(default_factory=dict)
    passo_min: int = 1
    metadados: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.perfis_w:
            raise ValueError("ensemble vazio")
        if not self.picos_janela_w:
            object.__setattr__(self, "picos_janela_w", dict(self.perfis_w))

    # -- acesso ------------------------------------------------------------
    @property
    def estacoes(self) -> tuple[str, ...]:
        return tuple(self.perfis_w.keys())

    @property
    def passos_por_dia(self) -> int:
        return int(next(iter(self.perfis_w.values())).shape[1])

    @property
    def horas_por_passo(self) -> float:
        return self.passo_min / 60.0

    def perfis(self, estacao: str | None = None) -> np.ndarray:
        if estacao is None:
            return np.concatenate([self.perfis_w[e] for e in self.estacoes], axis=0)
        return self.perfis_w[estacao]

    def picos_janela(self, estacao: str | None = None) -> np.ndarray:
        if estacao is None:
            return np.concatenate([self.picos_janela_w[e] for e in self.estacoes], axis=0)
        return self.picos_janela_w[estacao]

    # -- estatística -------------------------------------------------------
    def picos_diarios_w(self, estacao: str | None = None) -> np.ndarray:
        """Um valor por simulação: a maior potência instantânea do dia."""
        return self.picos_janela(estacao).max(axis=1)

    def energia_diaria_kwh(self, estacao: str | None = None) -> np.ndarray:
        return self.perfis(estacao).sum(axis=1) * self.horas_por_passo / 1000.0

    def perfil_medio_w(self, estacao: str | None = None) -> np.ndarray:
        return self.perfis(estacao).mean(axis=0)

    def curva_excedencia_pico(self, estacao: str | None = None) -> CurvaExcedencia:
        """Excedência do **pico diário** — a curva que dimensiona."""
        rotulo = "pico diário" if estacao is None else f"pico diário — {estacao}"
        return CurvaExcedencia.de_amostras(self.picos_diarios_w(estacao), rotulo=rotulo)

    def curva_excedencia_instantanea(self, estacao: str | None = None) -> CurvaExcedencia:
        """
        Excedência **instantânea**: cada ponto de cada curva é uma amostra.

        ``P(X > x)`` aqui é a fração do tempo em que a carga passa de ``x``, e
        é ela que diz quanto tempo por dia um inversor de dado porte ficaria
        saturado — não a curva de pico, que só olha o pior minuto.
        """
        rotulo = "carga instantânea" if estacao is None else f"carga instantânea — {estacao}"
        return CurvaExcedencia.de_amostras(self.picos_janela(estacao).ravel(), rotulo=rotulo)

    def curva_excedencia_por_hora(self, hora: int, estacao: str | None = None) -> CurvaExcedencia:
        """Excedência instantânea restrita a uma hora do dia (0-23)."""
        por_hora = max(1, int(round(60 / self.passo_min)))
        inicio, fim = int(hora) * por_hora, (int(hora) + 1) * por_hora
        fatia = self.picos_janela(estacao)[:, inicio:fim]
        return CurvaExcedencia.de_amostras(fatia.ravel(), rotulo=f"carga às {int(hora):02d}h")

    # -- amostragem --------------------------------------------------------
    def amostrar_dias(
        self,
        estacao: str,
        quantidade: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Sorteia ``quantidade`` dias com reposição da estação.

        Devolve ``(media, pico)`` no mesmo formato — a primeira para o balanço
        de energia, a segunda para o teste de potência do inversor.
        """
        matriz = self.perfis_w[estacao]
        indices = rng.integers(0, matriz.shape[0], size=int(quantidade))
        return matriz[indices], self.picos_janela_w[estacao][indices]

    # -- transformação -----------------------------------------------------
    def reamostrar(self, passo_min: int) -> "EnsembleCarga":
        """
        Agrega para um passo maior preservando energia (média) e pico (máxima).

        Só aceita múltiplos do passo atual que dividam o dia inteiro; qualquer
        outra coisa deixaria a última janela com peso diferente das demais e
        distorceria silenciosamente a energia diária.
        """
        novo = int(passo_min)
        if novo == self.passo_min:
            return self
        if novo < self.passo_min or novo % self.passo_min != 0:
            raise ValueError(f"passo {novo} min não é múltiplo do passo atual ({self.passo_min} min)")
        fator = novo // self.passo_min
        if self.passos_por_dia % fator != 0:
            raise ValueError(f"passo {novo} min não divide o dia em janelas iguais")

        medias: dict[str, np.ndarray] = {}
        picos: dict[str, np.ndarray] = {}
        for estacao in self.estacoes:
            n_sim = self.perfis_w[estacao].shape[0]
            medias[estacao] = self.perfis_w[estacao].reshape(n_sim, -1, fator).mean(axis=2)
            picos[estacao] = self.picos_janela_w[estacao].reshape(n_sim, -1, fator).max(axis=2)
        return EnsembleCarga(
            perfis_w=medias,
            picos_janela_w=picos,
            passo_min=novo,
            metadados=dict(self.metadados, reamostrado_de_min=self.passo_min),
        )

    def resumo(self) -> dict[str, Any]:
        dados: dict[str, Any] = {
            "passo_min": self.passo_min,
            "simulacoes_por_estacao": {e: int(self.perfis_w[e].shape[0]) for e in self.estacoes},
            "geral": {
                "pico_medio_kw": float(np.mean(self.picos_diarios_w())) / 1000.0,
                "pico_p95_kw": float(np.percentile(self.picos_diarios_w(), 95)) / 1000.0,
                "energia_diaria_media_kwh": float(np.mean(self.energia_diaria_kwh())),
            },
            "por_estacao": {},
        }
        for estacao in self.estacoes:
            dados["por_estacao"][estacao] = {
                "pico_medio_kw": float(np.mean(self.picos_diarios_w(estacao))) / 1000.0,
                "pico_p95_kw": float(np.percentile(self.picos_diarios_w(estacao), 95)) / 1000.0,
                "energia_diaria_media_kwh": float(np.mean(self.energia_diaria_kwh(estacao))),
            }
        return dados


# ----------------------------------------------------------------------------
# Execução do Monte Carlo
# ----------------------------------------------------------------------------
def simular_ensemble(
    comodos: Sequence[Comodo],
    instancias_por_comodo: Mapping[str, int] | None = None,
    num_simulacoes: int = 300,
    ajustes_sazonais: Mapping[str, dict] | None = None,
    estacoes: Sequence[str] = tuple(ESTACOES_ANO),
    tempo_total: int = TEMPO_TOTAL_PADRAO_MIN,
    semente: int | None = 42,
    corrigir_intervalos: bool = True,
    progresso=None,
) -> EnsembleCarga:
    """
    Roda o Monte Carlo do D² uma vez por estação.

    ``ajustes_sazonais`` tem a forma esperada por
    :func:`aurum.demanda.nucleo.aplicar_ajuste_sazonal`: chave
    ``"Cômodo::Equipamento"`` apontando para ``{"ativo": True, "verão": +20, ...}``
    em variação percentual. Sem ajuste, as quatro estações saem estatisticamente
    iguais e o estudo ainda funciona — só perde a sazonalidade da carga, não a
    do sol, que vem da série de geração.

    ``corrigir_intervalos`` liga a correção de janela que atravessa a
    meia-noite (ver :mod:`aurum.demanda.correcoes`). Fica ligada por padrão
    porque, desligada, toda carga noturna some do perfil — e é a carga noturna
    que decide o dimensionamento de um backup. Desligue apenas para reproduzir
    o número original do D².
    """
    if not comodos:
        raise ValueError("nenhum cômodo para simular")
    instancias = dict(instancias_por_comodo or {c.nome: 1 for c in comodos})
    ajustes = dict(ajustes_sazonais or {})

    perfis: dict[str, np.ndarray] = {}
    with intervalos_circulares(corrigir_intervalos):
        for i, estacao in enumerate(estacoes):
            if semente is not None:
                # O núcleo do D² usa random e np.random globais; semeamos os dois
                # para que a rodada seja reprodutível, com semente distinta por
                # estação (mesma semente daria quatro estações idênticas).
                import random as _random

                _random.seed(semente + i)
                np.random.seed((semente + i) % (2**32))
            comodos_estacao = (
                aplicar_ajuste_sazonal(list(comodos), dict(ajustes), estacao)
                if ajustes
                else list(comodos)
            )
            _, matriz, _, _ = simula_carga_total(
                comodos_estacao,
                instancias,
                num_simulacoes=int(num_simulacoes),
                tempo_total=int(tempo_total),
            )
            perfis[estacao] = np.asarray(matriz, dtype=float)
            if progresso is not None:
                progresso(f"Monte Carlo — {estacao}", (i + 1) / len(estacoes))

    return EnsembleCarga(
        perfis_w=perfis,
        passo_min=1,
        metadados={
            "num_simulacoes": int(num_simulacoes),
            "tempo_total_min": int(tempo_total),
            "sazonalidade_aplicada": bool(ajustes),
            "intervalos_circulares": bool(corrigir_intervalos),
            "semente": semente,
        },
    )


def ensemble_de_curva_tipica(
    curva_w_por_minuto: Sequence[float],
    estacoes: Sequence[str] = tuple(ESTACOES_ANO),
    num_simulacoes: int = 300,
    dispersao_diaria: float = 0.15,
    semente: int | None = 42,
) -> EnsembleCarga:
    """
    Ensemble a partir de uma curva típica calibrada pela conta de luz.

    O caminho de quem não tem levantamento de equipamentos. A forma do dia vem
    do perfil do segmento; o tamanho, da fatura. Falta a variabilidade, e é
    ela que produz a cauda que decide o inversor — então ela é **assumida**,
    não derivada: um fator multiplicativo diário log-normal de desvio
    ``dispersao_diaria`` (15% por padrão, a ordem de grandeza da variação dia a
    dia de carga comercial).

    Isso precisa ser dito em voz alta em qualquer relatório que use este
    caminho: a distribuição de picos aqui é uma hipótese sobre a instalação,
    não uma medição dela. O levantamento de equipamentos deriva essa
    distribuição do comportamento de cada aparelho; este atalho a arbitra.
    """
    base = np.asarray(curva_w_por_minuto, dtype=float).ravel()
    if base.size % 1440 != 0 and base.size != 1440:
        raise ValueError("a curva precisa ter 1.440 pontos (um por minuto do dia)")
    rng = np.random.default_rng(semente)
    perfis = {
        estacao: base[None, :] * rng.lognormal(0.0, dispersao_diaria, size=(num_simulacoes, 1))
        for estacao in estacoes
    }
    return EnsembleCarga(
        perfis_w=perfis,
        passo_min=1,
        metadados={
            "origem": "curva_tipica_calibrada",
            "num_simulacoes": int(num_simulacoes),
            "dispersao_diaria": float(dispersao_diaria),
            "sazonalidade_aplicada": False,
            "aviso": (
                "Ensemble gerado de curva típica: a forma vem do perfil do segmento e o "
                "tamanho, da conta de luz. A dispersão dia a dia é uma hipótese "
                f"({dispersao_diaria:.0%}), não uma medição — e é ela que forma a cauda "
                "que dimensiona o inversor."
            ),
        },
    )


# ----------------------------------------------------------------------------
# Entrada de cenário
# ----------------------------------------------------------------------------
def carregar_cenario_excel(caminho) -> list[Comodo]:
    """Lê a planilha de cenário do D² (uma aba por cômodo)."""
    return cria_comodos_do_excel(caminho)


#: Quantas vezes cada cômodo do cenário de demonstração se repete no prédio.
#: O D² não guarda essa contagem junto do cenário — ela é dada na interface —,
#: então o exemplo traz um prédio plausível para que o estudo rode de ponta a
#: ponta sem configuração: 40 quartos e uma área comum.
INSTANCIAS_EXEMPLO: dict[str, dict[str, int]] = {
    "hotel": {"Quarto Standard": 40, "Área Comum": 1},
    "escritorio": {"Sala de Reunião": 3, "Estação de Trabalho": 30},
}


def cenario_exemplo(tipo: str = "hotel") -> tuple[list[Comodo], dict[str, int]]:
    """
    Cenário de demonstração do próprio D², pronto para rodar sem planilha.

    Serve de fixture nos testes e de caminho de primeira execução para quem
    ainda não montou o levantamento de cargas.
    """
    comodos = cria_comodos_do_dataframe(criar_cenario_exemplo(tipo))
    padrao = INSTANCIAS_EXEMPLO.get(tipo, {})
    instancias = {c.nome: int(padrao.get(c.nome, 1)) for c in comodos}
    return comodos, instancias


# ----------------------------------------------------------------------------
# Análise de eventos (duração de excedência)
# ----------------------------------------------------------------------------
def estatisticas_de_eventos(
    serie: np.ndarray,
    limite: float,
    passo_min: int,
) -> dict[str, float]:
    """
    Quanto tempo, seguido, a carga fica acima de ``limite``.

    ``serie`` é ``(n_dias, n_passos)``. A conta é feita por dia e não emenda o
    fim de um dia no começo do seguinte — dias distintos do ensemble são
    sorteios independentes, e colar um no outro inventaria eventos longos que
    o modelo não afirma existir.

    Devolve a frequência dos eventos, sua duração média/p95/máxima em minutos
    e a energia média excedente por evento (kWh acima do limite).
    """
    matriz = np.atleast_2d(np.asarray(serie, dtype=float))
    acima = matriz > float(limite)
    if not acima.any():
        return {
            "eventos_por_dia": 0.0,
            "fracao_do_tempo": 0.0,
            "duracao_media_min": 0.0,
            "duracao_p95_min": 0.0,
            "duracao_max_min": 0.0,
            "energia_media_por_evento_kwh": 0.0,
            "dias_com_evento_percent": 0.0,
        }

    # Bordas com padding: uma transição False->True inicia um evento.
    pad = np.zeros((matriz.shape[0], 1), dtype=bool)
    marcado = np.concatenate([pad, acima, pad], axis=1)
    inicios = np.argwhere(~marcado[:, :-1] & marcado[:, 1:])
    fins = np.argwhere(marcado[:, :-1] & ~marcado[:, 1:])

    duracoes = (fins[:, 1] - inicios[:, 1]).astype(float) * passo_min
    excesso = np.clip(matriz - float(limite), 0.0, None)
    energias = np.array(
        [
            excesso[linha, ini:fim].sum() * (passo_min / 60.0) / 1000.0
            for (linha, ini), (_, fim) in zip(inicios, fins)
        ],
        dtype=float,
    )

    n_dias = matriz.shape[0]
    return {
        "eventos_por_dia": float(len(duracoes) / n_dias),
        "fracao_do_tempo": float(acima.mean()),
        "duracao_media_min": float(duracoes.mean()),
        "duracao_p95_min": float(np.percentile(duracoes, 95)),
        "duracao_max_min": float(duracoes.max()),
        "energia_media_por_evento_kwh": float(energias.mean()) if energias.size else 0.0,
        "dias_com_evento_percent": float(acima.any(axis=1).mean() * 100.0),
    }
