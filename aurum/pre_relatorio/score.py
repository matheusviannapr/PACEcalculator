"""
Score PACE: de 0 a 100, quão bem o imóvel usa a energia — e até onde dá para
chegar naquele local.

A modelagem é a da planilha "Score PACE": cinco critérios, cada um com um
critério bruto ("CRIT 1"), uma correção e um teto para o local.

========================  ====  =====================================================
Critério                  Peso  Como pontua
========================  ====  =====================================================
Aproveitamento econômico     5  faixa de kWh por pessoa × (tarifa mínima ÷ real)
Autonomia                   20  faixa de horas sem rede × "curva T" (chance de
                                segurar a falta que o cliente descreveu)
Eficiência de equipamento   30  30 × consumo possível ÷ consumo estimado
Uso na hora do sol          15  encaixe do consumo na curva do sol × 0,5 sem
(load placement)                geração ou × 1 com geração; o teto do local é o
                                encaixe com o que dá para mudar de horário
Aproveitamento de área      30  30 × produção ÷ consumo; o teto do local é o que
                                o telhado produziria ÷ consumo
========================  ====  =====================================================

A soma dos tetos é o **máximo do local**: um apartamento não tem telhado, e
uma casa que não pode mudar o horário de nada não chega a 15 no uso no sol.

**O que está arbitrado** (e marcado ``A_CALIBRAR`` abaixo): as faixas de kWh
por pessoa, os multiplicadores da tarifa branca, as faixas de horas da
autonomia, o encaixe de referência e a produtividade do telhado por m². A
estrutura é a da planilha; os números são o primeiro palpite, para recalibrar
com casos reais.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .eficiencia import AvaliacaoEficiencia

#: Os números moram em score_parametros.json, que o app também lê (o build
#: copia para vistoria/cliente/score-parametros.js). Aqui só viram constantes.
PARAMETROS = json.loads((Path(__file__).with_name("score_parametros.json")).read_text(encoding="utf-8"))
_P = PARAMETROS

PESOS = {k: float(v) for k, v in _P["pesos"].items()}
NOMES = dict(_P["nomes"])

# --- A_CALIBRAR (score_parametros.json) ------------------------------------
FAIXAS_KWH_PESSOA = tuple((math.inf if c is None else float(c), float(p)) for c, p in _P["economico"]["faixasKwhPessoa"])
CRIT_ECONOMICO_NEGOCIO = float(_P["economico"]["critNegocio"])
_B = _P["tarifaBranca"]
BRANCA_FORA_PONTA, BRANCA_INTERMEDIARIO, BRANCA_PONTA = float(_B["foraPonta"]), float(_B["intermediario"]), float(_B["ponta"])
HORAS_PONTA, HORAS_INTERMEDIARIO = tuple(_B["horasPonta"]), tuple(_B["horasIntermediario"])
FAIXAS_AUTONOMIA_H = tuple((float(c), float(p)) for c, p in _P["autonomia"]["faixasHoras"])
HORAS_GERADOR, HORAS_NOBREAK = float(_P["autonomia"]["horasGerador"]), float(_P["autonomia"]["horasNobreak"])
ENCAIXE_REFERENCIA = float(_P["usoNoSol"]["encaixeReferencia"])
CORRECAO_SEM_GERACAO = float(_P["usoNoSol"]["correcaoSemGeracao"])
_A = _P["area"]
AREA_TELHADO_M2 = {k: float(v) for k, v in _A["m2"].items()}
FRACAO_UTIL_TELHADO, KWP_POR_M2 = float(_A["fracaoUtil"]), float(_A["kwpPorM2"])
FATOR_SOMBRA = {k: float(v) for k, v in _A["sombra"].items()}
KWP_POR_PLACA, PRODUTIVIDADE_PADRAO = float(_A["kwpPorPlaca"]), float(_A["produtividadePadrao"])
# ---------------------------------------------------------------------------


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


@dataclass(frozen=True)
class EntradaScore:
    """Os números de um cenário. Tudo já em unidades físicas: o score não sabe de app."""
    consumo_mensal_kwh: float
    pessoas: int | None = None              # None em negócio
    tarifa_real_brl_kwh: float | None = None
    tarifa_minima_brl_kwh: float | None = None
    horas_sem_rede: float = 0.0             # o que o sistema segura
    curva_t: float | None = None            # chance de segurar a falta declarada (0..1)
    consumo_possivel_mensal_kwh: float | None = None   # com todas as trocas de aparelho
    #: O que se gasta depois das trocas deste cenário (None = o consumo de hoje).
    #: A eficiência é possível ÷ isto: 1 quando todas as trocas foram feitas.
    consumo_eficiencia_mensal_kwh: float | None = None
    encaixe_sol: float = 0.0                # 0..1
    encaixe_sol_maximo: float | None = None
    tem_geracao: bool = False
    producao_anual_kwh: float = 0.0
    producao_possivel_anual_kwh: float | None = None   # None = telhado desconhecido


@dataclass(frozen=True)
class Criterio:
    id: str
    nome: str
    peso: float
    crit: float             # "CRIT 1" da planilha, em pontos
    correcao: float | None  # "CORR."
    pontos: float
    teto_local: float
    detalhe: str = ""

    def as_dict(self) -> dict:
        return {"id": self.id, "nome": self.nome, "peso": self.peso, "crit": round(self.crit, 2),
                "correcao": None if self.correcao is None else round(self.correcao, 3),
                "pontos": round(self.pontos, 1), "teto_local": round(self.teto_local, 1), "detalhe": self.detalhe}


@dataclass(frozen=True)
class Score:
    criterios: tuple[Criterio, ...]

    @property
    def valor(self) -> float:
        return sum(c.pontos for c in self.criterios)

    @property
    def maximo_local(self) -> float:
        return sum(c.teto_local for c in self.criterios)

    def criterio(self, id_: str) -> Criterio:
        return next(c for c in self.criterios if c.id == id_)

    def as_dict(self) -> dict:
        return {"valor": round(self.valor, 1), "maximo_local": round(self.maximo_local, 1),
                "criterios": [c.as_dict() for c in self.criterios]}


# ----------------------------------------------------------------------------
# Os cinco critérios
# ----------------------------------------------------------------------------
def _faixa_ate(valor: float, faixas) -> float:
    """Faixas "até X": a primeira cujo corte o valor não passa (FAIXAS_KWH_PESSOA)."""
    return next(p for corte, p in faixas if valor <= corte)


def _faixa_desde(valor: float, faixas) -> float:
    """Faixas "a partir de X", da maior para a menor (FAIXAS_AUTONOMIA_H); abaixo de todas, 0."""
    return next((p for corte, p in faixas if valor >= corte), 0.0)


def economico(e: EntradaScore) -> Criterio:
    if e.pessoas:
        por_pessoa = e.consumo_mensal_kwh / e.pessoas
        crit = _faixa_ate(por_pessoa, FAIXAS_KWH_PESSOA)
        detalhe = f"{por_pessoa:.0f} kWh por pessoa por mês"
    else:
        crit, detalhe = CRIT_ECONOMICO_NEGOCIO, "faixa neutra (negócio)"
    if e.tarifa_real_brl_kwh and e.tarifa_minima_brl_kwh:
        corr = _clip01(e.tarifa_minima_brl_kwh / e.tarifa_real_brl_kwh)
        if corr < 0.995:
            detalhe += f"; a tarifa branca sairia {1 - corr:.0%} mais barata"
    else:
        corr = 1.0
    return Criterio("economico", NOMES["economico"], PESOS["economico"], crit, corr, crit * corr,
                    PESOS["economico"], detalhe)


def autonomia(e: EntradaScore) -> Criterio:
    crit = _faixa_desde(e.horas_sem_rede, FAIXAS_AUTONOMIA_H)
    corr = 1.0 if e.curva_t is None else _clip01(e.curva_t)
    if crit <= 0:
        detalhe = "nada mantém a energia quando a luz cai"
    else:
        detalhe = f"segura ~{e.horas_sem_rede:.0f} h sem rede"
        if e.curva_t is not None:
            detalhe += f", ~{corr:.0%} das faltas declaradas"
    return Criterio("autonomia", NOMES["autonomia"], PESOS["autonomia"], crit, corr, crit * corr,
                    PESOS["autonomia"], detalhe)


def eficiencia(e: EntradaScore) -> Criterio:
    gasto = e.consumo_mensal_kwh if e.consumo_eficiencia_mensal_kwh is None else e.consumo_eficiencia_mensal_kwh
    possivel = e.consumo_possivel_mensal_kwh if e.consumo_possivel_mensal_kwh is not None else gasto
    razao = _clip01(possivel / gasto) if gasto > 0 else 1.0
    pontos = PESOS["eficiencia"] * razao
    detalhe = ("os aparelhos já são os eficientes" if razao >= 0.995 else
               f"dá para gastar {1 - razao:.0%} menos com aparelhos eficientes")
    return Criterio("eficiencia", NOMES["eficiencia"], PESOS["eficiencia"], pontos, None, pontos,
                    PESOS["eficiencia"], detalhe)


def uso_no_sol(e: EntradaScore) -> Criterio:
    peso = PESOS["uso_no_sol"]
    crit = peso * _clip01(e.encaixe_sol / ENCAIXE_REFERENCIA)
    maximo = e.encaixe_sol if e.encaixe_sol_maximo is None else max(e.encaixe_sol, e.encaixe_sol_maximo)
    teto = peso * _clip01(maximo / ENCAIXE_REFERENCIA)
    corr = 1.0 if e.tem_geracao else CORRECAO_SEM_GERACAO
    detalhe = f"{e.encaixe_sol:.0%} do consumo encaixa na curva do sol"
    if not e.tem_geracao:
        detalhe += "; sem geração, vale metade"
    return Criterio("uso_no_sol", NOMES["uso_no_sol"], peso, crit, corr, crit * corr, teto, detalhe)


def area(e: EntradaScore) -> Criterio:
    peso = PESOS["area"]
    anual = e.consumo_mensal_kwh * 12
    crit = peso * _clip01(e.producao_anual_kwh / anual) if anual > 0 else 0.0
    if e.producao_possivel_anual_kwh is None:
        teto, onde = peso, "telhado não informado: teto cheio até a vistoria"
    else:
        teto = peso * _clip01(e.producao_possivel_anual_kwh / anual) if anual > 0 else 0.0
        onde = (f"o telhado cobre ~{e.producao_possivel_anual_kwh / anual:.0%} do consumo"
                if e.producao_possivel_anual_kwh > 0 else "sem telhado próprio")
    teto = max(teto, crit)
    detalhe = (f"o solar produz ~{e.producao_anual_kwh / anual:.0%} do consumo; " if e.producao_anual_kwh > 0
               else "nada é produzido hoje; ") + onde
    return Criterio("area", NOMES["area"], peso, crit, None, crit, teto, detalhe)


def calcular(e: EntradaScore) -> Score:
    return Score((economico(e), autonomia(e), eficiencia(e), uso_no_sol(e), area(e)))


# ----------------------------------------------------------------------------
# As medidas a partir das curvas e das respostas
# ----------------------------------------------------------------------------
def encaixe_com_sol(consumo_kw: np.ndarray, sol: np.ndarray) -> float:
    """
    Quanto do consumo cai dentro da curva do sol, com o solar do tamanho do
    consumo: Σ min(consumo_h, sol_h) ÷ Σ consumo, com o sol escalado para a
    mesma energia do dia. 1 = o consumo tem a forma do sol.
    """
    c = np.clip(np.asarray(consumo_kw, dtype=float), 0, None)
    s = np.clip(np.asarray(sol, dtype=float), 0, None)
    if c.sum() <= 0 or s.sum() <= 0:
        return 0.0
    s = s * c.sum() / s.sum()
    return float(np.minimum(c, s).sum() / c.sum())


def deslocar_para_o_sol(consumo_kw: np.ndarray, sol: np.ndarray, energia_dia_kwh: float) -> np.ndarray:
    """
    Leva ``energia_dia_kwh`` das horas sem sol para as horas de sol, na forma
    do sol. Tira de cada hora fora do sol na proporção do que ela tem — não
    sabe em que hora o aparelho estava, então é um teto otimista para a
    máquina que já rodava de dia.
    """
    c = np.clip(np.asarray(consumo_kw, dtype=float), 0, None).copy()
    s = np.clip(np.asarray(sol, dtype=float), 0, None)
    if energia_dia_kwh <= 0 or s.sum() <= 0:
        return c
    escuro = s < 0.05 * s.max()
    disponivel = float(c[escuro].sum())
    mover = min(energia_dia_kwh, 0.9 * disponivel)
    if mover <= 0:
        return c
    c[escuro] *= 1 - mover / disponivel
    c += mover * s / s.sum()
    return c


def razao_tarifa_branca(consumo_kw: np.ndarray) -> float:
    """
    Quanto a conta sairia na tarifa branca, em fração da convencional, com
    este perfil de consumo. Dias úteis com ponta e intermediário; fim de
    semana todo fora-ponta.
    """
    c = np.clip(np.asarray(consumo_kw, dtype=float), 0, None)
    if c.sum() <= 0:
        return 1.0
    mult = np.full(24, BRANCA_FORA_PONTA)
    for h in HORAS_INTERMEDIARIO:
        mult[h] = BRANCA_INTERMEDIARIO
    for h in HORAS_PONTA:
        mult[h] = BRANCA_PONTA
    util = float((c * mult).sum() / c.sum())
    return 5 / 7 * util + 2 / 7 * BRANCA_FORA_PONTA


def kwp_do_telhado(imovel: dict | None) -> float | None:
    """kWp que cabem no telhado; 0 sem telhado próprio; None se não se sabe."""
    imovel = imovel or {}
    acesso = imovel.get("acesso")
    if acesso == "predio":
        return 0.0
    if acesso not in ("proprio", "cobertura"):
        return None
    m2 = AREA_TELHADO_M2.get(str(imovel.get("tamanho")))
    if m2 is None:
        return None
    return m2 * FRACAO_UTIL_TELHADO * KWP_POR_M2 * FATOR_SOMBRA.get(str(imovel.get("sombra")), FATOR_SOMBRA["nao_sei"])


def kwp_existente(fontes: dict | None, imovel: dict | None) -> float | None:
    """O solar que já existe: placas declaradas × 0,55 kWp. None = tem, mas não sabe quanto."""
    if not (fontes or {}).get("solar"):
        return 0.0
    placas = str((imovel or {}).get("placas") or "")
    return int(placas) * KWP_POR_PLACA if placas.isdigit() else None


def horas_existentes(fontes: dict | None) -> float:
    fontes = fontes or {}
    if fontes.get("gerador"):
        return HORAS_GERADOR
    if fontes.get("nobreak"):
        return HORAS_NOBREAK
    return 0.0


@dataclass(frozen=True)
class Cenario:
    id: str
    nome: str
    score: Score
    investimento_brl: float | None = None
    payback_anos: float | None = None
    o_que_inclui: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict:
        return {"id": self.id, "nome": self.nome, "valor": round(self.score.valor, 1),
                "investimento_brl": None if self.investimento_brl is None else round(self.investimento_brl),
                "payback_anos": None if self.payback_anos is None else round(self.payback_anos, 1),
                "o_que_inclui": list(self.o_que_inclui), "criterios": [c.as_dict() for c in self.score.criterios]}


def com(e: EntradaScore, **mudancas) -> EntradaScore:
    """O mesmo cenário com algumas medidas trocadas."""
    return replace(e, **mudancas)


# ----------------------------------------------------------------------------
# Hoje: o imóvel como o cliente descreveu
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class MedidasDoImovel:
    """O que o score de hoje mediu, e o que os outros cenários reaproveitam."""
    hoje: EntradaScore
    eficiencia: "AvaliacaoEficiencia"
    sol_forma: np.ndarray            # 24 h, W por kWp
    produtividade_kwh_kwp_ano: float
    kwp_telhado: float | None
    kwp_existente: float | None
    energia_deslocavel_kwh_mes: float
    avisos: tuple[str, ...] = ()


def medir_hoje(entrada, curva, sol_w_por_kwp: np.ndarray | None = None,
               produtividade_kwh_kwp_ano: float | None = None) -> MedidasDoImovel:
    """
    ``entrada`` e ``curva`` são as do pré-relatório. ``sol_w_por_kwp`` é a
    média de 24 h da série solar do local (W por kWp); sem ela, uma senoide
    das 6 às 18 h. A produtividade anual sai da mesma série quando vem.
    """
    from .eficiencia import avaliar, energia_deslocavel_kwh_mes

    q = entrada.questionario
    avisos: list[str] = []
    consumo_kw = np.asarray(curva.curva_total_kw, dtype=float)
    if sol_w_por_kwp is None:
        horas = np.arange(24) + 0.5
        sol = np.clip(np.sin((horas - 6) / 12 * np.pi), 0, None) * 800.0
    else:
        sol = np.asarray(sol_w_por_kwp, dtype=float)
    produtividade = produtividade_kwh_kwp_ano or (float(sol.sum()) * 365 / 1000.0 if sol_w_por_kwp is not None
                                                  else PRODUTIVIDADE_PADRAO)

    pessoas = None
    if not entrada.negocio:
        p = str((q.get("rotina") or {}).get("pessoas") or "")
        pessoas = int(p) if p.isdigit() else None

    tarifa = float(entrada.tarifa_brl_kwh)
    if q.get("tarifaModalidade") == "branca":
        minima = tarifa
    else:
        minima = tarifa * min(1.0, razao_tarifa_branca(consumo_kw)) if _B.get("ativa", True) else tarifa

    ef = avaliar(curva.cargas, entrada.pacote)
    deslocavel = energia_deslocavel_kwh_mes(ef, q.get("aparelhos"))
    dias_mes = 30.0
    maximo = encaixe_com_sol(deslocar_para_o_sol(consumo_kw, sol, deslocavel / dias_mes), sol)

    fontes, imovel = q.get("fontesExistentes") or {}, q.get("imovel") or {}
    kwp_ja = kwp_existente(fontes, imovel)
    consumo_anual = float(entrada.conta.consumo_mensal_kwh) * 12
    if kwp_ja is None:
        producao = 0.5 * consumo_anual
        avisos.append("O cliente já tem solar mas não disse quantas placas: contei metade do consumo.")
    else:
        producao = kwp_ja * produtividade
    kwp_teto = kwp_do_telhado(imovel)
    if kwp_teto is None:
        avisos.append("Telhado não informado: o teto de área fica cheio até a vistoria.")

    hoje = EntradaScore(
        consumo_mensal_kwh=float(entrada.conta.consumo_mensal_kwh), pessoas=pessoas,
        tarifa_real_brl_kwh=tarifa, tarifa_minima_brl_kwh=minima,
        horas_sem_rede=horas_existentes(fontes), curva_t=None,
        consumo_possivel_mensal_kwh=float(entrada.conta.consumo_mensal_kwh) * ef.razao,
        encaixe_sol=encaixe_com_sol(consumo_kw, sol), encaixe_sol_maximo=maximo,
        tem_geracao=kwp_ja != 0.0, producao_anual_kwh=producao,
        producao_possivel_anual_kwh=None if kwp_teto is None else max(kwp_teto * produtividade, producao),
    )
    return MedidasDoImovel(hoje=hoje, eficiencia=ef, sol_forma=sol, produtividade_kwh_kwp_ano=produtividade,
                           kwp_telhado=kwp_teto, kwp_existente=kwp_ja, energia_deslocavel_kwh_mes=deslocavel,
                           avisos=tuple(avisos))
