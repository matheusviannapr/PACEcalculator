"""
A conta da bateria: operação anual, valor da resiliência e fluxo de caixa.

Bateria conectada à rede ganha dinheiro de três formas, e elas competem entre
si pelo mesmo estado de carga:

1. **Arbitragem tarifária** — carrega no fora-ponta (ou com o excedente solar)
   e descarrega na ponta. O ganho por kWh é a diferença de tarifa, e no Brasil
   ela é grande na tarifa branca e no Grupo A, quase nula no B convencional.
2. **Autoconsumo** — guarda o excedente solar do meio-dia para a noite em vez
   de injetar. Com a Lei 14.300, injetar deixou de valer tarifa cheia: o ganho
   da bateria é exatamente a parte que o Fio B come da energia injetada.
3. **Resiliência** — a energia que deixa de faltar quando a rede cai. Não
   aparece na conta de luz e é, em muitos clientes, a maior das três. Entra
   aqui pelo produto ``energia não suprida evitada × custo da interrupção``,
   com o custo declarado pelo cliente — porque só ele sabe quanto vale uma
   hora de câmara fria parada ou de hotel sem elevador.

A competição entre 1 e 3 é real e o modelo a representa: ``reserva_backup_frac``
é a fração do banco que não é ciclada, guardada para o apagão. Reserva alta dá
resiliência e mata a arbitragem; reserva zero faz o contrário. Ver a fronteira
entre as duas é metade do valor deste módulo.

Não modelado, e de propósito: mercado livre, resposta da demanda e serviços
ancilares. Nenhum deles está acessível ao cliente típico de geração distribuída
no Brasil hoje, e incluí-los inflaria o retorno com receita que não existe.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..demanda.ensemble import EnsembleCarga
from ..precos import PRECOS
from ..pv.financials import _tir
from .apagao import ResultadoResiliencia
from .catalogo import ConjuntoArmazenamento
from .degradacao import ModeloDegradacao, trajetoria_de_vida
from .despacho import LimitesDespacho
from .geracao import SerieGeracao

__all__ = [
    "OperacaoAnual",
    "PremissasBateria",
    "ResultadoEconomicoBateria",
    "avaliar_economia",
    "simular_operacao_anual",
]

MODOS = ("autoconsumo", "ponta", "hibrido")


@dataclass(frozen=True)
class PremissasBateria:
    """Tarifas, custos e hipóteses de operação."""

    # -- tarifas -----------------------------------------------------------
    tarifa_fora_ponta_brl_kwh: float = 0.78
    tarifa_ponta_brl_kwh: float = 1.35
    #: Valor recebido por kWh injetado, já líquido do Fio B. Zero significa
    #: que a injeção não vale nada (ilha, ou compensação esgotada).
    valor_injecao_brl_kwh: float = 0.55
    inicio_ponta_h: int = 18
    fim_ponta_h: int = 21
    #: Tarifa de demanda contratada, R$/kW/mês. Zero para clientes do Grupo B.
    tarifa_demanda_brl_kw_mes: float = 0.0

    # -- custos ------------------------------------------------------------
    #: Preço do kWh de bateria e do kW de inversor quando o catálogo não traz
    #: cotação. Os padrões vêm de :mod:`aurum.precos`, que é datado e diz de
    #: onde a faixa saiu -- ter o número aqui, solto, fazia dele folclore de
    #: planilha em três meses.
    capex_bateria_brl_kwh: float = PRECOS["bateria_lfp_brl_kwh"].tipico
    capex_inversor_brl_kw: float = PRECOS["inversor_hibrido_brl_w"].tipico * 1000.0
    #: Projeto, instalação, proteções e quadro de backup, sobre o custo de
    #: equipamento. Bateria puxa mais que FV porque exige quadro dedicado.
    instalacao_percent_do_equipamento: float = 0.35
    opex_percent_do_capex_ano: float = 0.01
    #: Queda real de preço do kWh de bateria ao ano — o que barateia a troca
    #: futura. 5% a.a. é conservador diante da série histórica de LFP.
    queda_preco_bateria_ano: float = 0.05

    # -- resiliência -------------------------------------------------------
    #: Quanto vale para o cliente 1 kWh que faltou. É o parâmetro mais
    #: sensível do modelo e o único que não se estima de fora: pergunte.
    custo_interrupcao_brl_kwh: float = 0.0
    #: Frequência e duração das faltas. Os padrões são a ordem de grandeza dos
    #: limites de DEC/FEC da ANEEL para área urbana; use os índices reais do
    #: alimentador quando houver.
    interrupcoes_por_ano: float = 8.0
    duracao_media_interrupcao_h: float = 2.5

    # -- operação e finanças ----------------------------------------------
    modo: str = "hibrido"
    reserva_backup_frac: float = 0.30
    permitir_carga_da_rede: bool = True
    #: Horas imediatamente antes da ponta em que a carga pela rede é permitida.
    #: Sem essa janela o modelo carrega e descarrega no mesmo dia dezenas de
    #: vezes — matematicamente possível, fisicamente absurdo e destruidor de
    #: bateria. Controlador nenhum opera assim: todos completam o banco pouco
    #: antes da ponta e param.
    horas_preparo_ponta: float = 3.0
    taxa_desconto_ano: float = 0.10
    escalada_tarifaria_ano: float = 0.05
    anos_analise: int = 15

    def __post_init__(self) -> None:
        if self.modo not in MODOS:
            raise ValueError(f"modo deve ser um de {MODOS}")

    def mascara_ponta(self, passos_por_dia: int) -> np.ndarray:
        """Máscara booleana da janela de ponta no passo do despacho."""
        por_hora = passos_por_dia / 24.0
        indices = np.arange(passos_por_dia) / por_hora
        if self.inicio_ponta_h == self.fim_ponta_h:
            return np.zeros(passos_por_dia, dtype=bool)
        if self.inicio_ponta_h < self.fim_ponta_h:
            return (indices >= self.inicio_ponta_h) & (indices < self.fim_ponta_h)
        return (indices >= self.inicio_ponta_h) | (indices < self.fim_ponta_h)

    def mascara_preparo(self, passos_por_dia: int) -> np.ndarray:
        """As horas antes da ponta em que se completa o banco pela rede."""
        por_hora = passos_por_dia / 24.0
        indices = np.arange(passos_por_dia) / por_hora
        inicio = (self.inicio_ponta_h - self.horas_preparo_ponta) % 24
        fim = float(self.inicio_ponta_h)
        if self.horas_preparo_ponta <= 0:
            return np.zeros(passos_por_dia, dtype=bool)
        if inicio < fim:
            return (indices >= inicio) & (indices < fim)
        return (indices >= inicio) | (indices < fim)


# ----------------------------------------------------------------------------
# Operação anual
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class OperacaoAnual:
    """Balanço energético de um ano típico, com e sem bateria."""

    import_ponta_sem_kwh: float
    import_fora_sem_kwh: float
    export_sem_kwh: float
    import_ponta_com_kwh: float
    import_fora_com_kwh: float
    export_com_kwh: float
    descarga_bateria_kwh: float
    recarga_bateria_kwh: float
    recarga_da_rede_kwh: float
    ciclos_equivalentes: float
    reducao_pico_ponta_kw: float

    @property
    def deslocamento_ponta_kwh(self) -> float:
        return self.import_ponta_sem_kwh - self.import_ponta_com_kwh

    @property
    def variacao_fora_ponta_kwh(self) -> float:
        """Negativo quando a bateria fez o cliente importar **mais** fora da ponta."""
        return self.import_fora_sem_kwh - self.import_fora_com_kwh

    @property
    def exportacao_convertida_kwh(self) -> float:
        """Excedente solar que deixou de ser injetado e virou autoconsumo."""
        return self.export_sem_kwh - self.export_com_kwh

    def as_dict(self) -> dict[str, Any]:
        return {
            "deslocamento_ponta_kwh_ano": self.deslocamento_ponta_kwh,
            "variacao_fora_ponta_kwh_ano": self.variacao_fora_ponta_kwh,
            "exportacao_convertida_kwh_ano": self.exportacao_convertida_kwh,
            "descarga_bateria_kwh_ano": self.descarga_bateria_kwh,
            "recarga_da_rede_kwh_ano": self.recarga_da_rede_kwh,
            "ciclos_equivalentes_ano": self.ciclos_equivalentes,
            "reducao_pico_ponta_kw": self.reducao_pico_ponta_kw,
        }


def simular_operacao_anual(
    conjunto: ConjuntoArmazenamento,
    ensemble: EnsembleCarga,
    serie: SerieGeracao,
    premissas: PremissasBateria,
    dias_por_estacao: int = 60,
    semente: int = 4242,
    fator_capacidade: float = 1.0,
) -> OperacaoAnual:
    """
    Simula um ano típico com a rede disponível, dia a dia, com e sem bateria.

    O "sem bateria" roda no mesmo sorteio de dias, e não numa média — comparar
    um cenário estocástico contra a média do outro produziria uma economia
    fantasma só pela diferença de variância.
    """
    limites = LimitesDespacho.do_conjunto(conjunto)
    capacidade = limites.energia_util_dc_kwh * float(fator_capacidade)
    piso = capacidade * float(np.clip(premissas.reserva_backup_frac, 0.0, 0.95))

    rng = np.random.default_rng(semente)
    passos_dia = ensemble.passos_por_dia
    dt_h = ensemble.passo_min / 60.0
    ponta = premissas.mascara_ponta(passos_dia)
    preparo = premissas.mascara_preparo(passos_dia)
    # Só vale trazer energia da rede se a diferença de tarifa pagar as duas
    # conversões. Avaliado uma vez, fora do laço: não depende do passo.
    arbitragem_paga = premissas.tarifa_ponta_brl_kwh > premissas.tarifa_fora_ponta_brl_kwh / max(
        1e-6, limites.eficiencia_carga * limites.eficiencia_descarga
    )
    carrega_da_rede = (
        premissas.permitir_carga_da_rede and arbitragem_paga and premissas.modo != "autoconsumo"
    )

    totais = dict.fromkeys(
        [
            "imp_ponta_sem", "imp_fora_sem", "exp_sem",
            "imp_ponta_com", "imp_fora_com", "exp_com",
            "descarga", "recarga", "recarga_rede",
        ],
        0.0,
    )
    picos_ponta_sem: list[float] = []
    picos_ponta_com: list[float] = []

    for estacao in ensemble.estacoes:
        carga, _ = ensemble.amostrar_dias(estacao, dias_por_estacao, rng)
        carga = carga / 1000.0
        geracao = (
            serie.amostrar_janelas(estacao, 0, passos_dia, ensemble.passo_min, dias_por_estacao, rng)
            * conjunto.potencia_fv_aproveitavel_kw
            / 1000.0
        )
        # Estado inicial no meio da faixa útil: o dia começa nem cheio nem
        # vazio, que é o regime permanente de um banco em uso diário.
        soc = np.full(dias_por_estacao, (capacidade + piso) / 2.0)
        pico_sem = np.zeros(dias_por_estacao)
        pico_com = np.zeros(dias_por_estacao)

        for t in range(passos_dia):
            pv = np.minimum(geracao[:, t], limites.potencia_fv_kw)
            demanda = carga[:, t]
            liquido = pv - demanda
            deficit = np.maximum(0.0, -liquido)
            sobra = np.maximum(0.0, liquido)

            # -- sem bateria ------------------------------------------------
            if ponta[t]:
                totais["imp_ponta_sem"] += deficit.sum() * dt_h
                pico_sem = np.maximum(pico_sem, deficit)
            else:
                totais["imp_fora_sem"] += deficit.sum() * dt_h
            totais["exp_sem"] += sobra.sum() * dt_h

            # -- com bateria ------------------------------------------------
            descarrega = _quer_descarregar(premissas.modo, ponta[t])
            disponivel = np.maximum(0.0, soc - piso) * limites.eficiencia_descarga / dt_h
            descarga = (
                np.minimum(np.minimum(deficit, limites.potencia_descarga_kw), disponivel)
                if descarrega
                else np.zeros_like(deficit)
            )
            soc -= descarga * dt_h / limites.eficiencia_descarga

            espaco = np.maximum(0.0, capacidade - soc) / dt_h / limites.eficiencia_carga
            recarga = np.minimum(np.minimum(sobra, limites.potencia_carga_kw), espaco)
            recarga_rede = np.zeros_like(recarga)
            if carrega_da_rede and preparo[t]:
                # Completa o banco nas horas que antecedem a ponta, e só nelas.
                folga = np.maximum(0.0, limites.potencia_carga_kw - recarga)
                recarga_rede = np.minimum(folga, np.maximum(0.0, espaco - recarga))
            soc += (recarga + recarga_rede) * dt_h * limites.eficiencia_carga
            soc = np.clip(soc, 0.0, capacidade)

            importado = deficit - descarga + recarga_rede
            exportado = sobra - recarga
            if ponta[t]:
                totais["imp_ponta_com"] += importado.sum() * dt_h
                pico_com = np.maximum(pico_com, importado)
            else:
                totais["imp_fora_com"] += importado.sum() * dt_h
            totais["exp_com"] += exportado.sum() * dt_h
            totais["descarga"] += descarga.sum() * dt_h
            totais["recarga"] += recarga.sum() * dt_h
            totais["recarga_rede"] += recarga_rede.sum() * dt_h

        picos_ponta_sem.extend(pico_sem.tolist())
        picos_ponta_com.extend(pico_com.tolist())

    dias_simulados = dias_por_estacao * len(ensemble.estacoes)
    escala = 365.0 / dias_simulados
    energia_por_ciclo = max(1e-6, limites.energia_util_ca_kwh * float(fator_capacidade))

    return OperacaoAnual(
        import_ponta_sem_kwh=totais["imp_ponta_sem"] * escala,
        import_fora_sem_kwh=totais["imp_fora_sem"] * escala,
        export_sem_kwh=totais["exp_sem"] * escala,
        import_ponta_com_kwh=totais["imp_ponta_com"] * escala,
        import_fora_com_kwh=totais["imp_fora_com"] * escala,
        export_com_kwh=totais["exp_com"] * escala,
        descarga_bateria_kwh=totais["descarga"] * escala,
        recarga_bateria_kwh=totais["recarga"] * escala,
        recarga_da_rede_kwh=totais["recarga_rede"] * escala,
        ciclos_equivalentes=totais["descarga"] * escala / energia_por_ciclo,
        # Redução do pico de demanda na ponta: o P95 dos dias, não o máximo,
        # porque a demanda faturada é medida em intervalos de 15 min e um único
        # dia extremo não define o contrato.
        reducao_pico_ponta_kw=float(
            np.percentile(picos_ponta_sem, 95) - np.percentile(picos_ponta_com, 95)
        ),
    )


def _quer_descarregar(modo: str, em_ponta: bool) -> bool:
    if modo == "autoconsumo":
        return True
    if modo == "ponta":
        return em_ponta
    return True  # híbrido: descarrega sempre, mas respeitando a reserva de backup


# ----------------------------------------------------------------------------
# Fluxo de caixa
# ----------------------------------------------------------------------------
@dataclass
class ResultadoEconomicoBateria:
    """Fluxo de caixa e indicadores do investimento em armazenamento."""

    capex_brl: float
    operacao: OperacaoAnual
    fluxo: pd.DataFrame
    vpl_brl: float
    tir: float | None
    payback_anos: float | None
    economia_ano1_brl: float
    valor_resiliencia_ano1_brl: float
    vida_util_anos: float
    trocas: list[int] = field(default_factory=list)
    premissas: PremissasBateria | None = None

    @property
    def custo_nivelado_brl_kwh(self) -> float | None:
        """LCOS: custo por kWh que passa pela bateria ao longo da vida."""
        total = float(self.fluxo["energia_bateria_kwh"].sum())
        if total <= 0:
            return None
        custos = self.capex_brl + float(self.fluxo["opex_brl"].sum())
        return custos / total

    def as_dict(self) -> dict[str, Any]:
        return {
            "capex_brl": self.capex_brl,
            "economia_ano1_brl": self.economia_ano1_brl,
            "valor_resiliencia_ano1_brl": self.valor_resiliencia_ano1_brl,
            "vpl_brl": self.vpl_brl,
            "tir": self.tir,
            "payback_anos": self.payback_anos,
            "lcos_brl_kwh": self.custo_nivelado_brl_kwh,
            "vida_util_anos": self.vida_util_anos,
            "trocas_no_horizonte": self.trocas,
            **self.operacao.as_dict(),
        }


def _capex(conjunto: ConjuntoArmazenamento, premissas: PremissasBateria) -> tuple[float, float]:
    """Devolve (capex total, custo só das baterias) — o segundo é o que se troca."""
    baterias = (
        conjunto.bateria.preco_brl * conjunto.modulos
        if conjunto.bateria.preco_brl
        else conjunto.capacidade_nominal_kwh * premissas.capex_bateria_brl_kwh
    )
    inversor = (
        conjunto.inversor.preco_brl
        if conjunto.inversor.preco_brl
        else conjunto.inversor.potencia_ca_nominal_kw * premissas.capex_inversor_brl_kw
    )
    equipamento = baterias + inversor
    return equipamento * (1.0 + premissas.instalacao_percent_do_equipamento), baterias


def _energia_nao_suprida_evitada_kwh(
    resiliencia: ResultadoResiliencia | None,
    premissas: PremissasBateria,
    ensemble: EnsembleCarga,
) -> float:
    """
    Quanto de energia deixa de faltar por ano, graças ao banco.

    Sem bateria, um apagão de duração ``d`` custa toda a energia que a carga
    teria consumido nele. Com bateria, custa a ENS que a simulação mediu. A
    diferença, multiplicada pela frequência anual de faltas, é o benefício.
    """
    demanda_media_kw = float(np.mean(ensemble.perfis())) / 1000.0
    sem_bateria_kwh = demanda_media_kw * premissas.duracao_media_interrupcao_h
    if resiliencia is None:
        com_bateria_kwh = 0.0
    else:
        duracoes = resiliencia.tabela["duracao_h"].to_numpy()
        alvo = duracoes[np.argmin(np.abs(duracoes - premissas.duracao_media_interrupcao_h))]
        recorte = resiliencia.tabela[np.isclose(resiliencia.tabela["duracao_h"], alvo)]
        com_bateria_kwh = float(recorte["ens_medio_kwh"].mean())
        # Reescala se a duração da malha não bate com a média de interrupção.
        if alvo > 0:
            com_bateria_kwh *= premissas.duracao_media_interrupcao_h / float(alvo)
    return max(0.0, sem_bateria_kwh - com_bateria_kwh) * premissas.interrupcoes_por_ano


def avaliar_economia(
    conjunto: ConjuntoArmazenamento,
    ensemble: EnsembleCarga,
    serie: SerieGeracao,
    premissas: PremissasBateria | None = None,
    resiliencia: ResultadoResiliencia | None = None,
    degradacao: ModeloDegradacao | None = None,
    dias_por_estacao: int = 60,
    semente: int = 4242,
) -> ResultadoEconomicoBateria:
    """
    Monta o fluxo de caixa do armazenamento ao longo do horizonte de análise.

    A degradação entra duas vezes, e as duas contam: encolhe o benefício anual
    (menos energia ciclada) e marca o ano da troca do banco, que volta como
    desembolso — a um preço menor, corrigido pela queda histórica do kWh de
    lítio. Um estudo que ignora a troca faz uma bateria de 6.000 ciclos parecer
    eterna num horizonte de 15 anos.
    """
    premissas = premissas or PremissasBateria()
    modelo = degradacao or ModeloDegradacao.do_conjunto(conjunto)

    operacao = simular_operacao_anual(
        conjunto, ensemble, serie, premissas, dias_por_estacao, semente
    )
    capex_total, capex_baterias = _capex(conjunto, premissas)
    vida = modelo.vida_util_anos(operacao.ciclos_equivalentes)
    ens_evitada = _energia_nao_suprida_evitada_kwh(resiliencia, premissas, ensemble)

    trajetoria = trajetoria_de_vida(
        modelo, operacao.ciclos_equivalentes, conjunto.energia_util_kwh, premissas.anos_analise
    )
    retencao_por_ano = {int(l.ano): float(l.retencao) for l in trajetoria.itertuples()}

    linhas: list[dict[str, Any]] = []
    trocas: list[int] = []
    idade = 0.0
    ciclos_do_banco = 0.0
    for ano in range(1, premissas.anos_analise + 1):
        idade += 1.0
        ciclos_do_banco += operacao.ciclos_equivalentes
        retencao = modelo.retencao(idade, ciclos_do_banco)

        inflacao = (1.0 + premissas.escalada_tarifaria_ano) ** (ano - 1)
        # O benefício acompanha a capacidade restante: banco em 80% cicla 80%
        # da energia, e a economia cai na mesma proporção.
        economia_ponta = operacao.deslocamento_ponta_kwh * premissas.tarifa_ponta_brl_kwh
        custo_recarga = operacao.recarga_da_rede_kwh * premissas.tarifa_fora_ponta_brl_kwh
        ganho_autoconsumo = operacao.exportacao_convertida_kwh * (
            premissas.tarifa_fora_ponta_brl_kwh - premissas.valor_injecao_brl_kwh
        )
        ganho_demanda = (
            operacao.reducao_pico_ponta_kw * premissas.tarifa_demanda_brl_kw_mes * 12.0
        )
        beneficio_tarifario = (
            economia_ponta - custo_recarga + ganho_autoconsumo + ganho_demanda
        ) * retencao * inflacao
        beneficio_resiliencia = (
            ens_evitada * premissas.custo_interrupcao_brl_kwh * retencao * inflacao
        )
        opex = capex_total * premissas.opex_percent_do_capex_ano

        troca = 0.0
        if retencao <= modelo.retencao_fim_vida:
            troca = capex_baterias * (1.0 - premissas.queda_preco_bateria_ano) ** (ano - 1)
            trocas.append(ano)
            idade, ciclos_do_banco = 0.0, 0.0

        linhas.append(
            {
                "ano": ano,
                "retencao": retencao,
                "beneficio_tarifario_brl": beneficio_tarifario,
                "beneficio_resiliencia_brl": beneficio_resiliencia,
                "opex_brl": opex,
                "troca_brl": troca,
                "fluxo_brl": beneficio_tarifario + beneficio_resiliencia - opex - troca,
                "energia_bateria_kwh": operacao.descarga_bateria_kwh * retencao,
            }
        )

    fluxo = pd.DataFrame(linhas)
    caixa = [-capex_total, *fluxo["fluxo_brl"].tolist()]
    taxa = premissas.taxa_desconto_ano
    vpl = float(sum(v / (1.0 + taxa) ** i for i, v in enumerate(caixa)))

    acumulado = np.cumsum(caixa)
    payback: float | None = None
    for i in range(1, len(acumulado)):
        if acumulado[i] >= 0 and acumulado[i - 1] < 0:
            fatia = caixa[i]
            payback = float(i - 1 + (-acumulado[i - 1] / fatia)) if fatia else float(i)
            break

    return ResultadoEconomicoBateria(
        capex_brl=capex_total,
        operacao=operacao,
        fluxo=fluxo,
        vpl_brl=vpl,
        tir=_tir(caixa),
        payback_anos=payback,
        economia_ano1_brl=float(fluxo.loc[0, "beneficio_tarifario_brl"]),
        valor_resiliencia_ano1_brl=float(fluxo.loc[0, "beneficio_resiliencia_brl"]),
        vida_util_anos=vida,
        trocas=trocas,
        premissas=premissas,
    )
