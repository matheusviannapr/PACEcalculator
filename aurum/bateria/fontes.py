"""
Comparação de fontes: solar, bateria e gerador, ligados e desligados.

A pergunta que este módulo responde não é "qual bateria comprar", e sim a que
vem antes dela: **vale a pena comprar alguma coisa, e o quê**. O cliente tem
uma conta de luz e um problema de falta de energia; entre não fazer nada e
comprar tudo existem seis arranjos intermediários, e cada um resolve uma parte
diferente do problema por um preço diferente.

As três fontes têm naturezas distintas, e é isso que faz a comparação valer:

* **Solar** reduz a conta e não dá backup nenhum. Inversor conectado à rede
  desliga quando a rede cai -- é exigência de anti-ilhamento da ABNT NBR
  16149, não limitação de marca. Vender solar como segurança energética é o
  engano mais comum do setor, e o quadro de cenários o desfaz sozinho: a
  coluna de autonomia do cenário "só solar" é zero.
* **Bateria** dá backup e, com tarifa simples, quase nada de economia -- não
  há diferença de tarifa para arbitrar. O que ela economiza é o pedaço que o
  Fio B come da energia injetada, guardando o excedente do meio-dia para a
  noite. É pouco. Quem paga a bateria, na tarifa simples, é a resiliência.
* **Gerador** é o inverso da bateria: energia praticamente ilimitada, potência
  limitada, e um custo por kWh que só aparece quando ele roda. Atravessa 36 h
  sem esforço e não ajuda em nada na conta de luz.

Todo cenário é medido contra o mesmo referencial -- **só a rede**, com a conta
que o cliente paga hoje -- e as três respostas saem lado a lado: quanto custa,
quanto economiza por ano, e o que acontece quando a luz cai.

Tarifa simples apenas, de propósito. Tarifa branca e Grupo A mudam a conta da
bateria por completo (a arbitragem passa a existir) e exigem a modelagem de
posto tarifário e demanda contratada, que :mod:`aurum.bateria.economia` já
tem. Aqui o objetivo é a decisão de arranjo, não a otimização do despacho.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..demanda.ensemble import EnsembleCarga
from ..pv.financials import _tir, estimar_capex
from .apagao import MalhaApagao, ResultadoResiliencia, avaliar_limites
from .catalogo import ConjuntoArmazenamento
from .degradacao import ModeloDegradacao
from .despacho import Gerador, LimitesDespacho
from .economia import PremissasBateria
from .geracao import SerieGeracao

LOGGER = logging.getLogger(__name__)

#: Confiabilidade exigida para declarar autonomia no quadro de cenários.
#: A mesma do estudo principal, e pelo mesmo motivo: prometer autonomia
#: média é prometer o que não se cumpre na noite em que a luz cai.
CONFIABILIDADE_AUTONOMIA = 0.95

__all__ = [
    "Composicao",
    "ComparacaoFontes",
    "Fatura",
    "Gerador",
    "ResultadoCenario",
    "comparar_fontes",
    "estudo_do_gerador",
]


# ----------------------------------------------------------------------------
# Entradas
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Fatura:
    """
    A conta de luz do cliente, na tarifa simples (convencional, Grupo B).

    ``consumo_mensal_kwh`` é o que a fatura diz. Quando informado, ele manda:
    a simulação de Monte Carlo é reescalada para bater com ele, porque o
    levantamento de cargas erra e a fatura não. Quando ausente, vale o consumo
    que a própria simulação produziu -- e o estudo diz que foi assim.
    """

    tarifa_brl_kwh: float = 0.98
    consumo_mensal_kwh: float | None = None
    #: Custo de disponibilidade em kWh: 30 monofásico, 50 bifásico, 100
    #: trifásico. É o piso da conta -- nenhum sistema leva a fatura a zero.
    custo_disponibilidade_kwh: float = 100.0
    #: Fração da tarifa retida como Fio B na energia injetada (Lei 14.300). O
    #: valor cresce até 2029; 45% do Fio B em 2026 dá algo perto de 0,15 da
    #: tarifa cheia para o consumidor típico do Grupo B.
    fio_b_frac_da_tarifa: float = 0.15
    escalada_tarifaria_ano: float = 0.05

    @property
    def valor_injecao_brl_kwh(self) -> float:
        """Quanto vale 1 kWh injetado, já descontado o Fio B."""
        return self.tarifa_brl_kwh * (1.0 - float(np.clip(self.fio_b_frac_da_tarifa, 0.0, 1.0)))

    @property
    def consumo_anual_kwh(self) -> float | None:
        return None if self.consumo_mensal_kwh is None else self.consumo_mensal_kwh * 12.0

    def conta_anual_brl(self, compra_kwh_ano: float, injecao_kwh_ano: float = 0.0) -> float:
        """
        A conta do ano, com o piso do custo de disponibilidade.

        O piso é aplicado mês a mês, não sobre o total do ano: um sistema que
        zera o consumo de dez meses e deixa dois acima do piso paga doze
        mínimos, e não a média deles.
        """
        faturado_mes = max(compra_kwh_ano / 12.0, float(self.custo_disponibilidade_kwh))
        bruto = faturado_mes * 12.0 * self.tarifa_brl_kwh
        return max(
            float(self.custo_disponibilidade_kwh) * 12.0 * self.tarifa_brl_kwh,
            bruto - injecao_kwh_ano * self.valor_injecao_brl_kwh,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "tarifa_brl_kwh": self.tarifa_brl_kwh,
            "consumo_mensal_kwh": self.consumo_mensal_kwh,
            "custo_disponibilidade_kwh": self.custo_disponibilidade_kwh,
            "valor_injecao_brl_kwh": self.valor_injecao_brl_kwh,
            "escalada_tarifaria_ano": self.escalada_tarifaria_ano,
        }


@dataclass(frozen=True)
class Composicao:
    """Quais fontes existem num cenário."""

    solar: bool = False
    bateria: bool = False
    gerador: bool = False

    @property
    def chave(self) -> str:
        partes = [n for n, on in (("solar", self.solar), ("bateria", self.bateria), ("gerador", self.gerador)) if on]
        return "+".join(partes) if partes else "rede"

    @property
    def nome(self) -> str:
        rotulos = [n for n, on in (("solar", self.solar), ("bateria", self.bateria), ("gerador", self.gerador)) if on]
        if not rotulos:
            return "Somente a rede"
        return "Rede + " + " + ".join(rotulos)

    @property
    def n_fontes(self) -> int:
        return int(self.solar) + int(self.bateria) + int(self.gerador)

    def as_dict(self) -> dict[str, Any]:
        return {"solar": self.solar, "bateria": self.bateria, "gerador": self.gerador}


def combinacoes(
    solar: bool = True, bateria: bool = True, gerador: bool = True
) -> list[Composicao]:
    """
    Todos os arranjos possíveis com as fontes que o cliente aceita considerar.

    O cenário vazio -- só a rede -- entra sempre, porque é o referencial contra
    o qual todos os outros são medidos. Sem ele não existe economia: existe
    apenas um custo.
    """
    disponiveis = []
    for s in (False, True) if solar else (False,):
        for b in (False, True) if bateria else (False,):
            for g in (False, True) if gerador else (False,):
                disponiveis.append(Composicao(s, b, g))
    return sorted(disponiveis, key=lambda c: (c.n_fontes, c.chave))


# ----------------------------------------------------------------------------
# Saída
# ----------------------------------------------------------------------------
@dataclass
class ResultadoCenario:
    """Um arranjo de fontes, medido por dinheiro e por resiliência."""

    composicao: Composicao
    capex_brl: float

    # -- energia do ano típico --------------------------------------------
    consumo_kwh_ano: float
    geracao_fv_kwh_ano: float
    autoconsumo_kwh_ano: float
    injecao_kwh_ano: float
    compra_da_rede_kwh_ano: float
    gerador_em_paralelo_kwh_ano: float

    # -- dinheiro ----------------------------------------------------------
    conta_anual_brl: float
    custo_operacional_brl_ano: float
    economia_anual_brl: float
    valor_resiliencia_brl_ano: float

    # -- resiliência -------------------------------------------------------
    autonomia_garantida_h: float
    prob_por_duracao: dict[float, float]
    ens_por_evento_kwh: float
    ens_evitada_kwh_ano: float
    combustivel_em_apagao_brl_ano: float

    # -- finanças ----------------------------------------------------------
    fluxo: pd.DataFrame
    vpl_brl: float
    tir: float | None
    payback_anos: float | None
    resiliencia: ResultadoResiliencia | None = None
    avisos: list[str] = field(default_factory=list)

    @property
    def nome(self) -> str:
        return self.composicao.nome

    @property
    def beneficio_anual_brl(self) -> float:
        """Economia de conta mais valor da resiliência, menos o que custa operar."""
        return (
            self.economia_anual_brl
            + self.valor_resiliencia_brl_ano
            - self.custo_operacional_brl_ano
            - self.combustivel_em_apagao_brl_ano
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "cenario": self.nome,
            "chave": self.composicao.chave,
            **self.composicao.as_dict(),
            "capex_brl": self.capex_brl,
            "consumo_kwh_ano": self.consumo_kwh_ano,
            "geracao_fv_kwh_ano": self.geracao_fv_kwh_ano,
            "autoconsumo_kwh_ano": self.autoconsumo_kwh_ano,
            "injecao_kwh_ano": self.injecao_kwh_ano,
            "compra_da_rede_kwh_ano": self.compra_da_rede_kwh_ano,
            "conta_anual_brl": self.conta_anual_brl,
            "economia_anual_brl": self.economia_anual_brl,
            "custo_operacional_brl_ano": self.custo_operacional_brl_ano,
            "valor_resiliencia_brl_ano": self.valor_resiliencia_brl_ano,
            "beneficio_anual_brl": self.beneficio_anual_brl,
            "autonomia_garantida_h": self.autonomia_garantida_h,
            "ens_por_evento_kwh": self.ens_por_evento_kwh,
            "vpl_brl": self.vpl_brl,
            "tir": self.tir,
            "payback_anos": self.payback_anos,
        }


@dataclass
class ComparacaoFontes:
    """Os cenários lado a lado, com o referencial destacado."""

    cenarios: list[ResultadoCenario]
    base: ResultadoCenario
    fatura: Fatura
    premissas: PremissasBateria
    gerador: Gerador | None
    conjunto: ConjuntoArmazenamento | None
    potencia_fv_kwp: float
    avisos: list[str] = field(default_factory=list)

    def tabela(self) -> pd.DataFrame:
        return pd.DataFrame([c.as_dict() for c in self.cenarios])

    def por_chave(self, chave: str) -> ResultadoCenario | None:
        return next((c for c in self.cenarios if c.composicao.chave == chave), None)

    @property
    def melhor_vpl(self) -> ResultadoCenario | None:
        """
        O arranjo de maior valor presente entre os que investem alguma coisa.

        O referencial fica de fora porque o VPL dele é zero por construção, e
        um empate técnico com o "não faça nada" precisa ser lido na tabela, não
        resolvido por arredondamento.
        """
        investem = [c for c in self.cenarios if c.capex_brl > 0]
        return max(investem, key=lambda c: c.vpl_brl) if investem else None

    @property
    def melhor_resiliencia(self) -> ResultadoCenario | None:
        investem = [c for c in self.cenarios if c.capex_brl > 0]
        if not investem:
            return None
        # A autonomia garantida empata em zero sempre que nenhum arranjo
        # atravessa a falta inteira em 95% dos casos — comum em carga
        # essencial pesada. O desempate vai para a energia que efetivamente
        # deixa de faltar, que é o estrago real, e só depois para o preço.
        return max(
            investem,
            key=lambda c: (c.autonomia_garantida_h, -c.ens_por_evento_kwh, -c.capex_brl),
        )

    def resumo(self) -> dict[str, Any]:
        return {
            "fatura": self.fatura.as_dict(),
            "potencia_fv_kwp": self.potencia_fv_kwp,
            "gerador": self.gerador.as_dict() if self.gerador else None,
            "conjunto": self.conjunto.descricao() if self.conjunto else None,
            "conta_atual_brl_ano": self.base.conta_anual_brl,
            "cenarios": [c.as_dict() for c in self.cenarios],
            "melhor_vpl": self.melhor_vpl.nome if self.melhor_vpl else None,
            "melhor_resiliencia": self.melhor_resiliencia.nome if self.melhor_resiliencia else None,
            "avisos": self.avisos,
        }


# ----------------------------------------------------------------------------
# Balanço anual conectado à rede
# ----------------------------------------------------------------------------
def _balanco_anual(
    ensemble: EnsembleCarga,
    serie: SerieGeracao,
    potencia_fv_kwp: float,
    limites: LimitesDespacho | None,
    gerador: Gerador | None,
    fatura: Fatura,
    reserva_backup_frac: float,
    dias_por_estacao: int,
    semente: int,
) -> dict[str, float]:
    """
    Um ano típico com a rede disponível, dia a dia, para um arranjo de fontes.

    A ordem de despacho é a única que faz sentido economicamente com tarifa
    simples: o sol atende a carga, o excedente carrega o banco, o que sobra é
    injetado, e a bateria devolve à noite o que guardou. Não há carga pela
    rede, porque sem diferença de tarifa entre horários a arbitragem só perde
    dinheiro nas duas conversões.

    O gerador só entra em paralelo à rede se o kWh dele for mais barato que o
    da distribuidora -- o que quase nunca acontece com diesel a preço de 2026,
    e é exatamente por isso que a regra é explícita em vez de suposta.
    """
    rng = np.random.default_rng(semente)
    passos_dia = ensemble.passos_por_dia
    dt_h = ensemble.passo_min / 60.0

    tem_banco = limites is not None and limites.energia_util_dc_kwh > 0
    capacidade = limites.energia_util_dc_kwh if tem_banco else 0.0
    piso = capacidade * float(np.clip(reserva_backup_frac, 0.0, 0.95))
    ger_paralelo_kw = (
        float(gerador.potencia_kw)
        if gerador is not None and gerador.custo_energia_brl_kwh < fatura.tarifa_brl_kwh
        else 0.0
    )

    totais = dict.fromkeys(
        ["consumo", "geracao", "autoconsumo", "injecao", "compra", "gerador", "descarga"],
        0.0,
    )

    for estacao in ensemble.estacoes:
        carga, _ = ensemble.amostrar_dias(estacao, dias_por_estacao, rng)
        carga = carga / 1000.0
        geracao = (
            serie.amostrar_janelas(
                estacao, 0, passos_dia, ensemble.passo_min, dias_por_estacao, rng
            )
            * float(potencia_fv_kwp)
            / 1000.0
        )
        if limites is not None and limites.potencia_fv_kw > 0:
            geracao = np.minimum(geracao, limites.potencia_fv_kw)

        def rodar(soc: np.ndarray, acumular: bool) -> np.ndarray:
            """Roda o dia inteiro; só a segunda passada entra na conta."""
            for t in range(passos_dia):
                demanda = carga[:, t]
                pv = geracao[:, t]

                direto = np.minimum(pv, demanda)
                sobra = pv - direto
                falta = demanda - direto

                descarga = np.zeros_like(falta)
                recarga = np.zeros_like(sobra)
                if tem_banco:
                    disponivel = np.maximum(0.0, soc - piso) * limites.eficiencia_descarga / dt_h
                    descarga = np.minimum(
                        np.minimum(falta, limites.potencia_descarga_kw), disponivel
                    )
                    soc = soc - descarga * dt_h / limites.eficiencia_descarga
                    falta = falta - descarga

                    espaco = np.maximum(0.0, capacidade - soc) / dt_h / limites.eficiencia_carga
                    recarga = np.minimum(np.minimum(sobra, limites.potencia_carga_kw), espaco)
                    soc = np.clip(soc + recarga * dt_h * limites.eficiencia_carga, 0.0, capacidade)
                    sobra = sobra - recarga

                do_grupo = np.minimum(falta, ger_paralelo_kw)
                falta = falta - do_grupo

                if acumular:
                    totais["consumo"] += demanda.sum() * dt_h
                    totais["geracao"] += pv.sum() * dt_h
                    totais["autoconsumo"] += (direto + descarga).sum() * dt_h
                    totais["injecao"] += sobra.sum() * dt_h
                    totais["compra"] += falta.sum() * dt_h
                    totais["gerador"] += do_grupo.sum() * dt_h
                    totais["descarga"] += descarga.sum() * dt_h
            return soc

        # Uma passada de aquecimento antes de contar. O banco não pode começar
        # o dia com energia que ninguém pagou: sem sol e sem carga pela rede,
        # um estado inicial arbitrário vira economia fantasma, e o cenário
        # "bateria sem solar" apareceria economizando o que não economiza.
        # A segunda passada começa onde a primeira terminou, que é o regime
        # permanente de um banco em uso diário.
        soc_inicial = np.full(dias_por_estacao, piso)
        rodar(rodar(soc_inicial, acumular=False), acumular=True)

    escala = 365.0 / (dias_por_estacao * len(ensemble.estacoes))
    return {chave: valor * escala for chave, valor in totais.items()}


def _fator_de_fatura(consumo_simulado_kwh_ano: float, fatura: Fatura) -> tuple[float, str | None]:
    """
    Quanto a simulação precisa ser esticada para bater com a conta de luz.

    O levantamento de cargas é uma estimativa; a fatura é medida. Quando as
    duas discordam em mais de 10%, o estudo escala a simulação para a fatura e
    diz que fez isso -- calar seria apresentar como consumo do cliente um
    número que a conta dele desmente.
    """
    alvo = fatura.consumo_anual_kwh
    if alvo is None or consumo_simulado_kwh_ano <= 0:
        return 1.0, None
    fator = alvo / consumo_simulado_kwh_ano
    if abs(fator - 1.0) < 0.10:
        return fator, None
    # O separador de milhar vira ponto só no número: aplicar o replace na
    # frase inteira comeria as vírgulas do texto.
    simulado = f"{consumo_simulado_kwh_ano:,.0f}".replace(",", ".")
    medido = f"{alvo:,.0f}".replace(",", ".")
    return fator, (
        f"O levantamento de cargas dá {simulado} kWh/ano e a fatura informada, "
        f"{medido} kWh/ano — diferença de {abs(fator - 1) * 100:.0f}%. Os valores de "
        "conta e economia foram escalados para a fatura; a análise de potência e de "
        "resiliência continua na curva simulada, que é quem descreve o pico."
    )


# ----------------------------------------------------------------------------
# Resiliência do arranjo
# ----------------------------------------------------------------------------
def _resiliencia_do_cenario(
    composicao: Composicao,
    conjunto: ConjuntoArmazenamento | None,
    gerador: Gerador | None,
    ensemble_backup: EnsembleCarga,
    serie: SerieGeracao,
    malha: MalhaApagao,
    semente: int,
) -> ResultadoResiliencia | None:
    """
    Varre a malha de apagões para o arranjo, ou devolve ``None`` quando é óbvio.

    Sem bateria e sem gerador não há o que simular: a carga fica sem energia do
    primeiro minuto ao último, seja qual for a potência instalada em painéis.
    """
    if not composicao.bateria and not composicao.gerador:
        return None

    if composicao.bateria and conjunto is not None:
        limites = LimitesDespacho.do_conjunto(conjunto)
        fv_kw = conjunto.potencia_fv_aproveitavel_kw if composicao.solar else 0.0
        limites = replace(limites, potencia_fv_kw=fv_kw)
    else:
        # Sem banco, o FV não ilha: o inversor conectado à rede desliga junto
        # com ela. O gerador é a única fonte, e ele entra pelo parâmetro.
        limites = LimitesDespacho.sem_bateria(0.0)

    return avaliar_limites(
        limites,
        limites.potencia_fv_kw,
        ensemble_backup,
        serie,
        malha,
        semente,
        gerador=gerador if composicao.gerador else None,
        conjunto=conjunto if composicao.bateria else None,
        rotulo=composicao.nome,
    )


def _interpolar_por_duracao(
    resiliencia: ResultadoResiliencia | None, coluna: str, duracao_h: float
) -> float:
    """Lê uma coluna da tabela por duração na duração média de interrupção."""
    if resiliencia is None or resiliencia.tabela.empty:
        return float("nan")
    resumo = resiliencia.por_duracao().sort_values("duracao_h")
    return float(
        np.interp(
            float(duracao_h),
            resumo["duracao_h"].to_numpy(dtype=float),
            resumo[coluna].to_numpy(dtype=float),
        )
    )


# ----------------------------------------------------------------------------
# Fluxo de caixa do cenário
# ----------------------------------------------------------------------------
def _fluxo_do_cenario(
    capex_brl: float,
    economia_anual_brl: float,
    valor_resiliencia_brl_ano: float,
    custo_operacional_brl_ano: float,
    combustivel_brl_ano: float,
    premissas: PremissasBateria,
    fatura: Fatura,
    degradacao: ModeloDegradacao | None,
    ciclos_ano: float,
    capex_baterias_brl: float,
) -> tuple[pd.DataFrame, float, float | None, float | None]:
    """
    Fluxo de caixa incremental do arranjo contra o cenário "só a rede".

    Tudo que acompanha a tarifa -- economia de conta e valor da resiliência --
    sobe com a escalada tarifária. O que é custo de operação sobe junto: um
    fluxo que inflaciona só o benefício fabrica retorno.

    A troca do banco entra como desembolso no ano em que a retenção cruza o
    fim de vida útil, com o preço do kWh de bateria corrigido pela queda
    histórica. Sem ela, um horizonte de 15 anos faz qualquer bateria parecer
    eterna.
    """
    linhas: list[dict[str, Any]] = []
    idade = 0.0
    ciclos = 0.0
    for ano in range(1, premissas.anos_analise + 1):
        inflacao = (1.0 + fatura.escalada_tarifaria_ano) ** (ano - 1)
        retencao = 1.0
        troca = 0.0
        if degradacao is not None:
            idade += 1.0
            ciclos += ciclos_ano
            retencao = degradacao.retencao(idade, ciclos)
            if retencao <= degradacao.retencao_fim_vida:
                troca = capex_baterias_brl * (1.0 - premissas.queda_preco_bateria_ano) ** (ano - 1)
                idade, ciclos = 0.0, 0.0

        # A economia de conta que vem do sol não depende do banco; a que vem
        # do banco, sim. Aplicar a retenção ao total seria pessimista demais,
        # e ignorá-la, otimista. O meio-termo honesto é aplicá-la ao valor da
        # resiliência, que é inteiramente da bateria.
        beneficio = economia_anual_brl * inflacao
        resiliencia = valor_resiliencia_brl_ano * retencao * inflacao
        custos = (custo_operacional_brl_ano + combustivel_brl_ano) * inflacao
        linhas.append({
            "ano": ano,
            "retencao": retencao,
            "economia_conta_brl": beneficio,
            "valor_resiliencia_brl": resiliencia,
            "custo_operacional_brl": custos,
            "troca_brl": troca,
            "fluxo_brl": beneficio + resiliencia - custos - troca,
        })

    fluxo = pd.DataFrame(linhas)
    caixa = [-capex_brl, *fluxo["fluxo_brl"].tolist()]
    taxa = premissas.taxa_desconto_ano
    vpl = float(sum(v / (1.0 + taxa) ** i for i, v in enumerate(caixa)))

    payback: float | None = None
    acumulado = np.cumsum(caixa)
    for i in range(1, len(acumulado)):
        if acumulado[i] >= 0 and acumulado[i - 1] < 0:
            fatia = caixa[i]
            payback = float(i - 1 + (-acumulado[i - 1] / fatia)) if fatia else float(i)
            break

    return fluxo, vpl, (_tir(caixa) if capex_brl > 0 else None), payback


# ----------------------------------------------------------------------------
# O custo de rodar o gerador
# ----------------------------------------------------------------------------
def estudo_do_gerador(comparacao: "ComparacaoFontes") -> pd.DataFrame:
    """
    Quanto combustível o grupo queima, e quanto isso custa, por duração.

    É a pergunta que sobra depois de dimensionar o grupo para suprir a carga:
    a potência deixa de ser incerteza e o custo passa a ser o assunto. A
    tabela sai por duração de falta porque é assim que a conta muda -- 1 h de
    gerador é troco, 36 h é uma decisão de orçamento.

    A comparação entre "só gerador" e "gerador com bateria" aparece de graça:
    com banco, o grupo entra depois do sol e da bateria, e queima menos. É o
    argumento econômico da dupla, e ele não existe sem esta tabela.
    """
    grupo = comparacao.gerador
    if grupo is None:
        return pd.DataFrame()

    premissas = comparacao.premissas
    linhas: list[dict[str, Any]] = []
    for cenario in comparacao.cenarios:
        if not cenario.composicao.gerador or cenario.resiliencia is None:
            continue
        for linha in cenario.resiliencia.por_duracao().itertuples():
            energia = float(getattr(linha, "energia_gerador_kwh", 0.0))
            horas = energia / grupo.potencia_kw if grupo.potencia_kw > 0 else float("nan")
            linhas.append({
                "cenario": cenario.nome,
                "duracao_h": float(linha.duracao_h),
                "energia_gerada_kwh": energia,
                "horas_equivalentes_a_plena_carga": horas,
                "custo_do_evento_brl": grupo.custo_da_energia_brl(energia),
                "prob_atendimento": float(linha.prob_atendimento),
                "energia_nao_suprida_kwh": float(linha.ens_medio_kwh),
            })

    tabela = pd.DataFrame(linhas)
    if tabela.empty:
        return tabela
    # O custo anual usa a frequência de interrupções declarada: é o número que
    # entra no orçamento, e ele só existe quando alguém diz quantas faltas por
    # ano a região tem.
    tabela["custo_anual_brl"] = (
        tabela["custo_do_evento_brl"] * premissas.interrupcoes_por_ano
    )
    return tabela


# ----------------------------------------------------------------------------
# Orquestração
# ----------------------------------------------------------------------------
def comparar_fontes(
    ensemble_total: EnsembleCarga,
    ensemble_backup: EnsembleCarga,
    serie: SerieGeracao,
    potencia_fv_kwp: float,
    fatura: Fatura,
    conjunto: ConjuntoArmazenamento | None = None,
    gerador: Gerador | None = None,
    premissas: PremissasBateria | None = None,
    malha: MalhaApagao | None = None,
    capex_fv_brl: float | None = None,
    padrao_capex: str | None = None,
    topologia_kit: str | None = None,
    mao_de_obra_brl_kwp: float | None = None,
    material_ca_brl_kwp: float | None = None,
    composicoes: Sequence[Composicao] | None = None,
    dias_por_estacao: int = 45,
    semente: int = 20260902,
    progresso=None,
) -> ComparacaoFontes:
    """
    Avalia cada arranjo de fontes e devolve os cenários lado a lado.

    ``padrao_capex`` escolhe a referência de R$/kWp do sistema fotovoltaico
    entre os :data:`aurum.pv.financials.PADROES_CAPEX`. O mesmo kWp custa
    coisas muito diferentes num galpão e num prédio ocupado de alto padrão, e
    tratar os dois pela mesma curva erra o payback em quase 50%.

    ``conjunto`` é o armazenamento a considerar nos cenários com bateria --
    tipicamente o recomendado pelo estudo. ``gerador`` é o grupo a considerar
    nos cenários com gerador. Quando um deles é ``None``, os cenários que o
    exigiriam simplesmente não são gerados: é melhor um quadro com quatro
    linhas verdadeiras que um com oito e metade inventada.
    """
    premissas = premissas or PremissasBateria()
    malha = malha or MalhaApagao()
    avisos: list[str] = []

    lista = list(
        composicoes
        if composicoes is not None
        else combinacoes(
            solar=potencia_fv_kwp > 0, bateria=conjunto is not None, gerador=gerador is not None
        )
    )
    if Composicao() not in lista:
        lista.insert(0, Composicao())

    # -- referencial: consumo simulado × fatura ----------------------------
    base_balanco = _balanco_anual(
        ensemble_total, serie, 0.0, None, None, fatura,
        premissas.reserva_backup_frac, dias_por_estacao, semente,
    )
    fator, aviso_fatura = _fator_de_fatura(base_balanco["consumo"], fatura)
    if aviso_fatura:
        avisos.append(aviso_fatura)

    limites_conjunto = LimitesDespacho.do_conjunto(conjunto) if conjunto is not None else None
    # Cotação de verdade passa na frente de qualquer curva; sem ela, o padrão
    # construtivo escolhe a referência de R$/kWp.
    capex_fv = (
        float(capex_fv_brl)
        if capex_fv_brl is not None
        else estimar_capex(
            potencia_fv_kwp, padrao=padrao_capex,
            topologia=topologia_kit,
            mao_de_obra_brl_kwp=mao_de_obra_brl_kwp,
            material_ca_brl_kwp=material_ca_brl_kwp,
        )
    )
    # Duas contas para o mesmo banco, porque o inversor pertence a uma ou a
    # outra conforme o arranjo. Num sistema com solar, o híbrido veio no kit e
    # o banco custa só os blocos de expansão; num sistema **só** de bateria,
    # não há kit nenhum, e o inversor é compra à parte. Uma conta só erraria
    # um dos dois cenários, e os dois aparecem lado a lado no quadro final.
    capex_bat_com_kit, capex_baterias = _capex_do_conjunto(
        conjunto, premissas, topologia_kit)
    capex_bat_sozinho, _ = _capex_do_conjunto(conjunto, premissas, None)
    modelo_degradacao = (
        ModeloDegradacao.do_conjunto(conjunto) if conjunto is not None else None
    )

    resultados: list[ResultadoCenario] = []
    base: ResultadoCenario | None = None

    for i, composicao in enumerate(lista):
        if progresso is not None:
            progresso(f"Cenário: {composicao.nome}", (i + 1) / len(lista))

        balanco = (
            base_balanco
            if composicao.n_fontes == 0
            else _balanco_anual(
                ensemble_total,
                serie,
                potencia_fv_kwp if composicao.solar else 0.0,
                limites_conjunto if composicao.bateria else None,
                gerador if composicao.gerador else None,
                fatura,
                premissas.reserva_backup_frac,
                dias_por_estacao,
                semente,
            )
        )
        balanco = {k: v * fator for k, v in balanco.items()}
        conta = fatura.conta_anual_brl(balanco["compra"], balanco["injecao"])

        capex_bat = capex_bat_com_kit if composicao.solar else capex_bat_sozinho
        capex = (
            (capex_fv if composicao.solar else 0.0)
            + (capex_bat if composicao.bateria else 0.0)
            + (float(gerador.capex_brl) if composicao.gerador and gerador else 0.0)
        )
        opex = capex * premissas.opex_percent_do_capex_ano + (
            float(gerador.opex_fixo_brl_ano) if composicao.gerador and gerador else 0.0
        )
        custo_grupo_paralelo = (
            balanco["gerador"] * float(gerador.custo_energia_brl_kwh)
            if composicao.gerador and gerador
            else 0.0
        )

        resiliencia = _resiliencia_do_cenario(
            composicao, conjunto, gerador, ensemble_backup, serie, malha, semente
        )
        autonomia = (
            resiliencia.autonomia_garantida_h(CONFIABILIDADE_AUTONOMIA, pior_caso=True)
            if resiliencia is not None
            else 0.0
        )
        duracao = premissas.duracao_media_interrupcao_h
        ens_evento = _interpolar_por_duracao(resiliencia, "ens_medio_kwh", duracao)
        ger_evento = _interpolar_por_duracao(resiliencia, "energia_gerador_kwh", duracao)

        # Sem nenhuma fonte de backup, tudo que a carga essencial pediria no
        # apagão é energia não suprida: é o referencial da resiliência.
        carga_backup_media_kw = float(np.mean(ensemble_backup.perfis())) / 1000.0
        ens_sem_nada = carga_backup_media_kw * duracao
        if resiliencia is None:
            ens_evento = ens_sem_nada
            ger_evento = 0.0
        evitada_ano = max(0.0, ens_sem_nada - ens_evento) * premissas.interrupcoes_por_ano
        # Por evento quando informado, por kWh quando não — a mesma regra da
        # análise econômica. Sem esta linha, o quadro de arranjos continuava
        # valorando por energia enquanto a seção econômica valorava por evento,
        # e o mesmo banco aparecia com R$ 18 numa página e R$ 4.730 na outra.
        if premissas.custo_interrupcao_brl_evento > 0:
            # A fração do evento que o backup cobre é a fração de energia que
            # ele supre: atravessar metade do apagão evita metade do
            # transtorno, e não o transtorno inteiro.
            fracao = (
                max(0.0, ens_sem_nada - ens_evento) / ens_sem_nada
                if ens_sem_nada > 0 else 0.0
            )
            valor_resiliencia = (
                fracao * premissas.interrupcoes_por_ano
                * premissas.custo_interrupcao_brl_evento
            )
        else:
            valor_resiliencia = evitada_ano * premissas.custo_interrupcao_brl_kwh
        combustivel_apagao = (
            float(np.nan_to_num(ger_evento))
            * premissas.interrupcoes_por_ano
            * float(gerador.custo_energia_brl_kwh)
            if composicao.gerador and gerador
            else 0.0
        )

        economia = (base.conta_anual_brl - conta) if base is not None else 0.0
        ciclos_ano = _ciclos_estimados(balanco, limites_conjunto) if composicao.bateria else 0.0
        fluxo, vpl, tir, payback = _fluxo_do_cenario(
            capex,
            economia,
            valor_resiliencia,
            opex + custo_grupo_paralelo,
            combustivel_apagao,
            premissas,
            fatura,
            modelo_degradacao if composicao.bateria else None,
            ciclos_ano,
            capex_baterias,
        )

        avisos_cenario: list[str] = []
        if composicao.solar and not composicao.bateria:
            avisos_cenario.append(
                "Solar sem bateria não fornece backup: o inversor conectado à rede "
                "desliga quando a rede cai, por exigência de anti-ilhamento."
            )

        cenario = ResultadoCenario(
            composicao=composicao,
            capex_brl=capex,
            consumo_kwh_ano=balanco["consumo"],
            geracao_fv_kwh_ano=balanco["geracao"],
            autoconsumo_kwh_ano=balanco["autoconsumo"],
            injecao_kwh_ano=balanco["injecao"],
            compra_da_rede_kwh_ano=balanco["compra"],
            gerador_em_paralelo_kwh_ano=balanco["gerador"],
            conta_anual_brl=conta,
            custo_operacional_brl_ano=opex + custo_grupo_paralelo,
            economia_anual_brl=economia,
            valor_resiliencia_brl_ano=valor_resiliencia,
            autonomia_garantida_h=autonomia,
            prob_por_duracao=(
                {
                    float(l.duracao_h): float(l.prob_atendimento)
                    for l in resiliencia.por_duracao().itertuples()
                }
                if resiliencia is not None
                else {float(d): 0.0 for d in malha.duracoes_h}
            ),
            ens_por_evento_kwh=float(np.nan_to_num(ens_evento)),
            ens_evitada_kwh_ano=evitada_ano,
            combustivel_em_apagao_brl_ano=combustivel_apagao,
            fluxo=fluxo,
            vpl_brl=vpl,
            tir=tir,
            payback_anos=payback,
            resiliencia=resiliencia,
            avisos=avisos_cenario,
        )
        resultados.append(cenario)
        if composicao.n_fontes == 0:
            base = cenario

    assert base is not None, "o cenário de referência é sempre montado primeiro"
    # As duas unidades contam. O custo por evento existe porque o por kWh serve
    # mal em residência — quanto melhor o recorte de criticidade, menor o
    # quadro, menos energia falta, e menor o valor que o modelo dá à bateria —,
    # e olhar só para o segundo fazia o aviso disparar exatamente em quem usa o
    # modelo bom, dizendo que a resiliência não entra em payback nenhum
    # enquanto o payback ao lado tinha sido calculado com ela dentro.
    if (premissas.custo_interrupcao_brl_kwh <= 0
            and premissas.custo_interrupcao_brl_evento <= 0):
        avisos.append(
            "O custo da interrupção foi deixado em zero, nas duas unidades — por kWh "
            "não suprido e por evento evitado —, então o valor da resiliência não "
            "entra em nenhum payback. Com tarifa simples é ele que paga a bateria: sem "
            "informá-lo, o quadro compara apenas economia de conta, e a bateria aparece "
            "pior do que é para quem não pode ficar sem energia."
        )

    return ComparacaoFontes(
        cenarios=resultados,
        base=base,
        fatura=fatura,
        premissas=premissas,
        gerador=gerador,
        conjunto=conjunto,
        potencia_fv_kwp=potencia_fv_kwp,
        avisos=avisos,
    )


def _capex_do_conjunto(
    conjunto: ConjuntoArmazenamento | None,
    premissas: PremissasBateria,
    topologia_kit: str | None = None,
) -> tuple[float, float]:
    """
    (CAPEX instalado, custo só das baterias) — o segundo é o que se troca.

    Com o kit split-phase escolhido, o banco é contado em **blocos de
    expansão** e o inversor **não** entra: ele já veio no kit fotovoltaico, e
    cobrá-lo outra vez dobrava o preço do armazenamento num sistema que tem as
    duas coisas. A própria tabela de preço confirma o modelo — a diferença
    entre a coluna split-phase e a coluna com 5 kWh é o bloco, e nada mais.

    Fora desse caminho vale a conta de sempre: bateria por R$/kWh mais
    inversor, porque aí o inversor é mesmo uma compra à parte.
    """
    if conjunto is None:
        return 0.0, 0.0
    from ..pv.kits import TOPOLOGIA_COM_BATERIA, preco_da_bateria
    from .economia import _capex

    if topologia_kit == TOPOLOGIA_COM_BATERIA:
        # Pela capacidade **nominal**, e não pela útil: o bloco é vendido pela
        # placa, e um banco de 5 kWh nominais entrega 4,6 kWh úteis. Contar
        # pelos úteis pedia um segundo bloco onde um basta.
        baterias = preco_da_bateria(conjunto.capacidade_nominal_kwh)
        if baterias > 0:
            # Sem os 35% de instalação: o bloco já é preço de módulo pronto,
            # com caixa, BMS e a instalação do módulo dentro.
            return baterias, baterias

    # Sem topologia, o inversor é compra à parte — e dizer isso aqui é
    # necessário. `premissas.inversor_no_kit_fv` é global do estudo e fica
    # ligada quando há solar; sem esta linha, o arranjo "Rede + bateria", que
    # não tem kit fotovoltaico nenhum, recebia o híbrido de graça e aparecia
    # por R$ 12.000 num quadro em que ele custa o banco mais o inversor.
    if topologia_kit is None and premissas.inversor_no_kit_fv:
        from dataclasses import replace as _replace

        premissas = _replace(premissas, inversor_no_kit_fv=False)
    return _capex(conjunto, premissas)


def _ciclos_estimados(balanco: dict[str, float], limites: LimitesDespacho | None) -> float:
    """Ciclos equivalentes por ano, a partir do que a bateria deslocou."""
    if limites is None or limites.energia_util_ca_kwh <= 0:
        return 0.0
    return max(0.0, balanco["descarga"]) / limites.energia_util_ca_kwh
