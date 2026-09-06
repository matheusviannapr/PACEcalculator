"""
Análise estatística completa da demanda — o que o D² entrega, e mais.

O D² produz seis figuras e um resumo estatístico. Esta é a mesma análise, com
as contas refeitas sobre o ensemble sazonal e com o que faltava lá:

**O que veio do D²**

* distribuição dos picos diários, com média e P95 marcados;
* curva de probabilidade de excedência;
* curva de duração de carga;
* perfil médio ao longo do dia;
* fator de carga hora a hora;
* composição da potência por cômodo, empilhada.

**O que foi acrescentado, e por quê**

* **Por estação.** No D² a rodada é uma só; aqui cada grandeza existe quatro
  vezes, porque o mesmo prédio tem picos diferentes em janeiro e em julho, e é
  o pior deles que dimensiona.
* **Composição do pico por equipamento.** Saber que o pico é 32 kW não diz o
  que fazer; saber que 60% dele são três chuveiros ligados ao mesmo tempo diz.
  Vem de ``simula_carga_total(coletar_detalhes_pico=True)``, que o D² já
  calculava e não usava no relatório.
* **Coincidência e diversidade.** A razão entre o pico do conjunto e a soma
  dos picos individuais mede quanto a operação real economiza em relação à
  potência instalada. É o número que separa "instalei 337 kW" de "a demanda é
  de 32 kW", e a diferença entre os dois é o que paga ou inviabiliza um projeto.
* **Intervalo de confiança do próprio Monte Carlo.** Com poucas simulações o
  P95 tem incerteza própria, e reportá-lo sem essa margem sugere uma precisão
  que o número não tem.

Unidades: watts na entrada, e as saídas declaram a unidade em cada campo.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .ensemble import EnsembleCarga
from .nucleo import Comodo, cria_comodos_individualizados

LOGGER = logging.getLogger(__name__)

__all__ = [
    "AnaliseDemanda",
    "ComposicaoPico",
    "EstatisticasPico",
    "IndicadoresEnergia",
    "analisar",
    "composicao_do_pico",
    "curva_de_duracao",
]

#: Percentis reportados na distribuição de picos. O 95 é o de dimensionamento
#: consagrado; o 99 existe porque a diferença entre os dois costuma ser o
#: argumento a favor ou contra um degrau de inversor.
PERCENTIS = (50, 75, 90, 95, 99)


# ----------------------------------------------------------------------------
# Estatística dos picos
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class EstatisticasPico:
    """Distribuição do pico diário, em watts."""

    n: int
    media: float
    mediana: float
    minimo: float
    maximo: float
    desvio: float
    percentis: Mapping[int, float]
    ic_inferior: float
    ic_superior: float
    erro_padrao_p95: float

    @property
    def coeficiente_variacao(self) -> float:
        """Desvio sobre média, em fração. Mede previsibilidade da operação."""
        return self.desvio / self.media if self.media > 0 else 0.0

    @property
    def interpretacao(self) -> str:
        cv = self.coeficiente_variacao
        if cv < 0.15:
            return "baixa variabilidade: operação previsível, dimensionamento confortável"
        if cv < 0.30:
            return "variabilidade moderada, típica de instalação comercial"
        return "alta variabilidade: o pico depende de coincidências, e a cauda importa"

    @classmethod
    def de_amostras(cls, picos: np.ndarray) -> "EstatisticasPico":
        amostras = np.asarray(picos, dtype=float).ravel()
        n = amostras.size
        p95 = float(np.percentile(amostras, 95))
        # Erro padrão de um percentil pela aproximação normal de ordem: mede a
        # incerteza que o próprio Monte Carlo carrega. Sem ele, um P95 obtido
        # com 50 rodadas parece tão firme quanto um obtido com 5.000.
        densidade = max(1e-9, _densidade_local(amostras, p95))
        erro = float(np.sqrt(0.95 * 0.05 / n) / densidade) if n > 1 else float("nan")
        return cls(
            n=int(n),
            media=float(amostras.mean()),
            mediana=float(np.median(amostras)),
            minimo=float(amostras.min()),
            maximo=float(amostras.max()),
            desvio=float(amostras.std(ddof=1)) if n > 1 else 0.0,
            percentis={q: float(np.percentile(amostras, q)) for q in PERCENTIS},
            ic_inferior=float(np.percentile(amostras, 2.5)),
            ic_superior=float(np.percentile(amostras, 97.5)),
            erro_padrao_p95=erro,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "simulacoes": self.n,
            "pico_medio_kw": self.media / 1000.0,
            "pico_mediano_kw": self.mediana / 1000.0,
            "pico_minimo_kw": self.minimo / 1000.0,
            "pico_maximo_kw": self.maximo / 1000.0,
            "desvio_padrao_kw": self.desvio / 1000.0,
            "coeficiente_variacao": self.coeficiente_variacao,
            **{f"p{q}_kw": v / 1000.0 for q, v in self.percentis.items()},
            "ic95_inferior_kw": self.ic_inferior / 1000.0,
            "ic95_superior_kw": self.ic_superior / 1000.0,
            "erro_padrao_p95_kw": self.erro_padrao_p95 / 1000.0,
            "interpretacao": self.interpretacao,
        }


def _densidade_local(amostras: np.ndarray, ponto: float) -> float:
    """Densidade empírica em torno de um ponto, por janela de largura de Silverman."""
    n = amostras.size
    if n < 2:
        return 0.0
    largura = 1.06 * amostras.std(ddof=1) * n ** (-1 / 5)
    if largura <= 0:
        return 0.0
    dentro = np.abs(amostras - ponto) <= largura
    return float(dentro.sum() / (2 * largura * n))


# ----------------------------------------------------------------------------
# Energia e fatores
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class IndicadoresEnergia:
    """Consumo, fatores de carga e de diversidade."""

    consumo_diario_kwh: float
    consumo_mensal_kwh: float
    consumo_anual_kwh: float
    demanda_media_kw: float
    demanda_maxima_kw: float
    energia_ponta_kwh: float
    energia_fora_ponta_kwh: float
    demanda_maxima_ponta_kw: float
    demanda_maxima_fora_kw: float
    potencia_instalada_kw: float
    soma_dos_picos_individuais_kw: float

    @property
    def fator_de_carga(self) -> float:
        """Demanda média sobre demanda máxima. Quanto mais alto, melhor a infra é usada."""
        return self.demanda_media_kw / self.demanda_maxima_kw if self.demanda_maxima_kw else 0.0

    @property
    def fator_de_demanda(self) -> float:
        """Demanda máxima sobre potência instalada — o clássico da NBR 5410."""
        return self.demanda_maxima_kw / self.potencia_instalada_kw if self.potencia_instalada_kw else 0.0

    @property
    def fator_de_coincidencia(self) -> float:
        """
        Pico do conjunto sobre a soma dos picos individuais.

        Mede quanto a operação real economiza por os equipamentos não ligarem
        todos juntos. É a diferença entre dimensionar pela placa e dimensionar
        pelo comportamento — e num prédio com muitas unidades iguais ela chega
        a um fator de cinco.
        """
        if self.soma_dos_picos_individuais_kw <= 0:
            return 0.0
        return self.demanda_maxima_kw / self.soma_dos_picos_individuais_kw

    @property
    def fracao_na_ponta(self) -> float:
        total = self.energia_ponta_kwh + self.energia_fora_ponta_kwh
        return self.energia_ponta_kwh / total if total else 0.0

    @property
    def interpretacao_fator_carga(self) -> str:
        fator = self.fator_de_carga
        if fator > 0.70:
            return "excelente aproveitamento da infraestrutura instalada"
        if fator > 0.50:
            return "bom aproveitamento"
        if fator > 0.30:
            return "aproveitamento baixo: o pico é bem maior que a média"
        return "aproveitamento muito baixo: carga concentrada em poucas horas"

    def as_dict(self) -> dict[str, Any]:
        return {
            "consumo_diario_kwh": self.consumo_diario_kwh,
            "consumo_mensal_kwh": self.consumo_mensal_kwh,
            "consumo_anual_kwh": self.consumo_anual_kwh,
            "demanda_media_kw": self.demanda_media_kw,
            "demanda_maxima_kw": self.demanda_maxima_kw,
            "energia_ponta_kwh_dia": self.energia_ponta_kwh,
            "energia_fora_ponta_kwh_dia": self.energia_fora_ponta_kwh,
            "fracao_na_ponta": self.fracao_na_ponta,
            "demanda_maxima_ponta_kw": self.demanda_maxima_ponta_kw,
            "demanda_maxima_fora_kw": self.demanda_maxima_fora_kw,
            "potencia_instalada_kw": self.potencia_instalada_kw,
            "fator_de_carga": self.fator_de_carga,
            "fator_de_demanda": self.fator_de_demanda,
            "fator_de_coincidencia": self.fator_de_coincidencia,
            "interpretacao_fator_carga": self.interpretacao_fator_carga,
        }


# ----------------------------------------------------------------------------
# Composição do pico
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class ComposicaoPico:
    """Quem estava ligado quando o pico aconteceu."""

    por_equipamento: pd.DataFrame
    por_comodo: pd.DataFrame
    hora_mais_provavel: int
    distribuicao_horaria: np.ndarray

    def as_dict(self) -> dict[str, Any]:
        return {
            "hora_mais_provavel_do_pico": self.hora_mais_provavel,
            "principais_equipamentos": self.por_equipamento.head(8).to_dict("records"),
            "por_comodo": self.por_comodo.to_dict("records"),
        }


def _agrupador(comodos: Sequence[Comodo]):
    """
    Devolve a função que traz o nome da instância de volta ao nome do cômodo.

    ``cria_comodos_individualizados`` numera as cópias como ``"Cozinha.3"``.
    Um regex de ``\\.\\d+$`` resolveria quase sempre e erraria num cômodo
    chamado "Bloco 2.1"; casar contra a lista de nomes originais não erra.
    """
    conhecidos = sorted((c.nome for c in comodos), key=len, reverse=True)

    def base(nome: str) -> str:
        for original in conhecidos:
            if nome == original or nome.startswith(original + "."):
                return original
        return nome

    return base


def composicao_do_pico(
    comodos: Sequence[Comodo],
    instancias_por_comodo: Mapping[str, int],
    num_simulacoes: int = 200,
    tempo_total: int = 1440,
    semente: int | None = 42,
) -> ComposicaoPico:
    """
    Roda o Monte Carlo pedindo o detalhamento do instante de pico.

    O D² já coletava esses detalhes e não os levava ao relatório. São eles que
    transformam "o pico é 32 kW" em "o pico é 32 kW e três quartos dele são
    chuveiros às 7 h" — a segunda frase sugere o que fazer, a primeira não.
    """
    from .correcoes import intervalos_circulares
    from .nucleo import simula_carga_total

    if semente is not None:
        import random as _random

        _random.seed(semente)
        np.random.seed(semente % (2**32))

    with intervalos_circulares():
        _, _, _, detalhes = simula_carga_total(
            list(comodos), dict(instancias_por_comodo),
            num_simulacoes=int(num_simulacoes), tempo_total=int(tempo_total),
            coletar_detalhes_pico=True,
        )

    if not detalhes:
        vazio = pd.DataFrame(columns=["equipamento", "comodo", "carga_media_kw", "participacao"])
        return ComposicaoPico(vazio, vazio.copy(), 0, np.zeros(24))

    # As 40 instâncias de "Apartamento" são o mesmo cômodo repetido; somá-las
    # de volta é o que transforma quarenta linhas de meio quilowatt numa linha
    # de vinte, que é a informação útil.
    base = _agrupador(comodos)
    por_equipamento: dict[tuple[str, str], float] = {}
    por_comodo: dict[str, float] = {}
    horas = np.zeros(24)
    for detalhe in detalhes:
        horas[int(detalhe["minuto_pico"]) // 60] += 1
        for ativo in detalhe["equipamentos_ativos"]:
            chave = (ativo["equipamento"], base(ativo["comodo"]))
            por_equipamento[chave] = por_equipamento.get(chave, 0.0) + ativo["carga_w"]
            nome = base(ativo["comodo"])
            por_comodo[nome] = por_comodo.get(nome, 0.0) + ativo["carga_w"]

    # Quantas vezes cada equipamento apareceu em algum pico, para separar
    # "sempre ligado, potência pequena" de "raro, potência enorme".
    aparicoes: dict[tuple[str, str], int] = {}
    for detalhe in detalhes:
        vistos = {(a["equipamento"], base(a["comodo"])) for a in detalhe["equipamentos_ativos"]}
        for chave in vistos:
            aparicoes[chave] = aparicoes.get(chave, 0) + 1

    pico_medio = float(np.mean([d["valor_pico"] for d in detalhes]))

    total = len(detalhes)
    equipamentos = pd.DataFrame(
        [
            {
                "equipamento": nome,
                "comodo": comodo,
                # Média sobre TODAS as simulações, não só as em que apareceu:
                # um equipamento presente em 10% dos picos não contribui como
                # um presente em todos, e dividir só pelas aparições esconderia
                # essa diferença.
                "carga_media_kw": soma / total / 1000.0,
                "presenca": aparicoes.get((nome, comodo), 0) / total,
                "participacao": (soma / total / pico_medio) if pico_medio else 0.0,
            }
            for (nome, comodo), soma in por_equipamento.items()
        ]
    ).sort_values("carga_media_kw", ascending=False).reset_index(drop=True)

    comodos_df = pd.DataFrame(
        [
            {
                "comodo": nome,
                "carga_media_kw": soma / total / 1000.0,
                "participacao": (soma / total / pico_medio) if pico_medio else 0.0,
            }
            for nome, soma in por_comodo.items()
        ]
    ).sort_values("carga_media_kw", ascending=False).reset_index(drop=True)

    return ComposicaoPico(
        por_equipamento=equipamentos,
        por_comodo=comodos_df,
        hora_mais_provavel=int(np.argmax(horas)),
        distribuicao_horaria=horas / horas.sum() if horas.sum() else horas,
    )


# ----------------------------------------------------------------------------
# Curvas
# ----------------------------------------------------------------------------
def curva_de_duracao(perfis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Curva de duração de carga: potência ordenada contra fração do tempo.

    Lê-se da direita para a esquerda: a área sob a curva é a energia, e o
    joelho perto da origem é a parcela de carga que só existe em poucas horas —
    exatamente a que um banco de baterias consegue cortar.
    """
    valores = np.sort(np.asarray(perfis, dtype=float).ravel())[::-1]
    fracao = np.arange(1, valores.size + 1) / valores.size
    return fracao, valores


def fator_de_carga_por_hora(perfis: np.ndarray, passo_min: int) -> np.ndarray:
    """Média sobre máxima, hora a hora — mostra onde a carga é firme e onde é pico."""
    por_hora = max(1, int(round(60 / passo_min)))
    blocos = np.asarray(perfis, dtype=float).reshape(perfis.shape[0], 24, por_hora)
    media = blocos.mean(axis=(0, 2))
    maxima = blocos.max(axis=(0, 2))
    return np.divide(media, maxima, out=np.zeros_like(media), where=maxima > 0)


def composicao_por_comodo(
    comodos: Sequence[Comodo],
    instancias_por_comodo: Mapping[str, int],
    num_simulacoes: int = 60,
    tempo_total: int = 1440,
    semente: int | None = 42,
) -> pd.DataFrame:
    """Perfil médio de cada cômodo ao longo do dia, para o gráfico empilhado."""
    from .correcoes import intervalos_circulares

    if semente is not None:
        import random as _random

        _random.seed(semente)
        np.random.seed(semente % (2**32))

    colunas: dict[str, np.ndarray] = {}
    base = _agrupador(comodos)
    with intervalos_circulares():
        individualizados = cria_comodos_individualizados(
            list(comodos), dict(instancias_por_comodo))
        acumulado: dict[str, np.ndarray] = {}
        for _ in range(int(num_simulacoes)):
            for comodo in individualizados:
                nome = base(comodo.nome)
                curva = comodo.simula_carga(tempo_total)
                acumulado[nome] = acumulado.get(nome, np.zeros(tempo_total)) + curva
        for nome, soma in acumulado.items():
            colunas[nome] = soma / num_simulacoes / 1000.0
    return pd.DataFrame(colunas, index=pd.Index(np.arange(tempo_total) / 60.0, name="hora"))


# ----------------------------------------------------------------------------
# A análise
# ----------------------------------------------------------------------------
@dataclass
class AnaliseDemanda:
    """Tudo que se pode dizer sobre o consumo, a partir do ensemble."""

    ensemble: EnsembleCarga
    estatisticas: EstatisticasPico
    indicadores: IndicadoresEnergia
    por_estacao: pd.DataFrame
    composicao: ComposicaoPico | None = None
    perfis_por_comodo: pd.DataFrame | None = None
    inicio_ponta_h: int = 18
    fim_ponta_h: int = 21
    metadados: dict[str, Any] = field(default_factory=dict)

    def curva_de_duracao(self, estacao: str | None = None) -> tuple[np.ndarray, np.ndarray]:
        return curva_de_duracao(self.ensemble.perfis(estacao))

    def fator_de_carga_horario(self, estacao: str | None = None) -> np.ndarray:
        return fator_de_carga_por_hora(self.ensemble.perfis(estacao), self.ensemble.passo_min)

    def tabela_percentis(self) -> pd.DataFrame:
        """Percentis do pico com o intervalo de confiança do próprio Monte Carlo."""
        linhas = []
        for q, valor in self.estatisticas.percentis.items():
            linhas.append({
                "percentil": f"P{q}",
                "pico_kw": valor / 1000.0,
                "excedido_em": f"{100 - q}% dos dias",
            })
        return pd.DataFrame(linhas)

    def resumo(self) -> dict[str, Any]:
        return {
            "estatisticas_do_pico": self.estatisticas.as_dict(),
            "indicadores": self.indicadores.as_dict(),
            "por_estacao": self.por_estacao.to_dict("records"),
            "composicao_do_pico": self.composicao.as_dict() if self.composicao else None,
            "janela_de_ponta": f"{self.inicio_ponta_h}h às {self.fim_ponta_h}h",
        }


def analisar(
    ensemble: EnsembleCarga,
    comodos: Sequence[Comodo] | None = None,
    instancias_por_comodo: Mapping[str, int] | None = None,
    potencia_instalada_w: float = 0.0,
    inicio_ponta_h: int = 18,
    fim_ponta_h: int = 21,
    detalhar_pico: bool = True,
    num_simulacoes_detalhe: int = 200,
) -> AnaliseDemanda:
    """
    Roda a análise completa sobre um ensemble já simulado.

    ``comodos`` é opcional: sem ele, a análise perde a composição do pico e a
    decomposição por cômodo, mas todo o resto continua. É o caso do caminho
    que parte da conta de luz, onde não existe lista de equipamentos.
    """
    picos = ensemble.picos_diarios_w()
    estatisticas = EstatisticasPico.de_amostras(picos)

    perfis = ensemble.perfis()
    passo_h = ensemble.horas_por_passo
    por_hora = max(1, int(round(60 / ensemble.passo_min)))
    indices_hora = np.arange(ensemble.passos_por_dia) // por_hora
    mascara_ponta = _mascara_ponta(indices_hora, inicio_ponta_h, fim_ponta_h)

    medio = perfis.mean(axis=0)
    consumo_diario = float(perfis.sum(axis=1).mean() * passo_h / 1000.0)
    soma_picos = _soma_dos_picos_individuais(comodos, instancias_por_comodo)

    indicadores = IndicadoresEnergia(
        consumo_diario_kwh=consumo_diario,
        consumo_mensal_kwh=consumo_diario * 30.0,
        consumo_anual_kwh=consumo_diario * 365.0,
        demanda_media_kw=float(medio.mean()) / 1000.0,
        demanda_maxima_kw=float(estatisticas.percentis[95]) / 1000.0,
        energia_ponta_kwh=float(medio[mascara_ponta].sum() * passo_h / 1000.0),
        energia_fora_ponta_kwh=float(medio[~mascara_ponta].sum() * passo_h / 1000.0),
        demanda_maxima_ponta_kw=float(perfis[:, mascara_ponta].max()) / 1000.0,
        demanda_maxima_fora_kw=float(perfis[:, ~mascara_ponta].max()) / 1000.0,
        potencia_instalada_kw=potencia_instalada_w / 1000.0,
        soma_dos_picos_individuais_kw=soma_picos / 1000.0,
    )

    linhas_estacao = []
    for estacao in ensemble.estacoes:
        picos_estacao = ensemble.picos_diarios_w(estacao)
        energia = ensemble.energia_diaria_kwh(estacao)
        linhas_estacao.append({
            "estacao": estacao,
            "pico_medio_kw": float(picos_estacao.mean()) / 1000.0,
            "pico_p95_kw": float(np.percentile(picos_estacao, 95)) / 1000.0,
            "pico_maximo_kw": float(picos_estacao.max()) / 1000.0,
            "consumo_diario_kwh": float(energia.mean()),
            "fator_de_carga": float(
                ensemble.perfil_medio_w(estacao).mean() / ensemble.perfil_medio_w(estacao).max()
            ) if ensemble.perfil_medio_w(estacao).max() > 0 else 0.0,
        })

    composicao = perfis_comodo = None
    if detalhar_pico and comodos:
        instancias = dict(instancias_por_comodo or {c.nome: 1 for c in comodos})
        composicao = composicao_do_pico(comodos, instancias, num_simulacoes_detalhe)
        perfis_comodo = composicao_por_comodo(comodos, instancias)

    return AnaliseDemanda(
        ensemble=ensemble,
        estatisticas=estatisticas,
        indicadores=indicadores,
        por_estacao=pd.DataFrame(linhas_estacao),
        composicao=composicao,
        perfis_por_comodo=perfis_comodo,
        inicio_ponta_h=inicio_ponta_h,
        fim_ponta_h=fim_ponta_h,
        metadados=dict(ensemble.metadados),
    )


def _mascara_ponta(indices_hora: np.ndarray, inicio: int, fim: int) -> np.ndarray:
    if inicio == fim:
        return np.zeros_like(indices_hora, dtype=bool)
    if inicio < fim:
        return (indices_hora >= inicio) & (indices_hora < fim)
    return (indices_hora >= inicio) | (indices_hora < fim)


def _soma_dos_picos_individuais(
    comodos: Sequence[Comodo] | None,
    instancias: Mapping[str, int] | None,
) -> float:
    """
    Soma dos picos de cada cômodo, isolado dos demais.

    É o denominador do fator de coincidência. Não é a potência instalada: cada
    cômodo já traz dentro de si a diversidade dos próprios equipamentos, e o
    que se quer medir aqui é só a diversidade *entre* cômodos.
    """
    if not comodos:
        return 0.0
    from .correcoes import intervalos_circulares

    instancias = dict(instancias or {c.nome: 1 for c in comodos})
    total = 0.0
    with intervalos_circulares():
        for comodo in comodos:
            curvas = np.array([comodo.simula_carga(1440) for _ in range(30)])
            total += float(curvas.max(axis=1).mean()) * instancias.get(comodo.nome, 1)
    return total
