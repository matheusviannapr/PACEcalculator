"""
Do questionário ao estudo, e do estudo ao resumo da imagem.

O pré-relatório não tem conta própria. Ele roda o mesmo
:func:`aurum.bateria.estudo.executar_estudo` do dossiê completo, com duas
entradas trocadas:

* ``ensemble_total`` — a curva híbrida (:mod:`aurum.pre_relatorio.curva`), no
  lugar do Monte Carlo do levantamento ou da curva típica pura;
* ``comodos_backup`` — só as cargas críticas (MC e C) que o cliente declarou.

O resto — catálogo de baterias, despacho, varredura de apagões, financeiro da
proposta — é o de sempre. Por isso o número da imagem e o do PDF completo são o
mesmo número: se a imagem diz "economia de R$ 900", é o ``FinanceiroProposta``
que diz, e o dossiê vai dizer igual. Ver a nota "financeiro único" do projeto.

O estudo aqui roda mais leve que o do dossiê (menos simulações e amostras de
apagão), porque o pré-relatório é faixa, não proposta. O dossiê roda com as
opções completas a partir da mesma configuração.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from datetime import date
from typing import Any

import numpy as np

from . import indice as indice_mod
from .curva import CRITICAS, CurvaHibrida, montar_curva
from .entrada import Entrada
from .resumo import Cobertura, Consumidor, ResumoPreRelatorio

#: As durações de falta que a imagem mostra. A declarada pelo cliente entra junto.
DURACOES_IMAGEM_H = (1.0, 3.0, 6.0, 12.0)
#: Confiabilidade da autonomia "garantida": a mesma das propostas.
CONFIABILIDADE = 0.95
#: Meia-largura da faixa de conta com solar. É faixa porque o pré-relatório não
#: viu o telhado nem confirmou a tarifa.
FOLGA_FAIXA = 0.12


@dataclass
class OpcoesPreRelatorio:
    tensao_v: float = 220.0
    disponibilidade_kwh: float = 50.0
    autonomia_alvo_h: float | None = None     # None: a sugerida pelo questionário
    simulacoes: int = 150
    amostras_apagao: int = 60
    candidatos: int = 8
    permitir_serie_sintetica: bool = True
    mostrar_investimento: bool = False


@dataclass
class PreRelatorio:
    entrada: Entrada
    curva: CurvaHibrida
    estudo: Any                      # ResultadoEstudo
    resumo: ResumoPreRelatorio
    avisos: list[str] = field(default_factory=list)


def configurar(entrada: Entrada, curva: CurvaHibrida, opcoes: OpcoesPreRelatorio):
    """A ``ConfiguracaoEstudo`` com a curva híbrida e o quadro de backup das críticas."""
    from ..bateria.apagao import MalhaApagao
    from ..bateria.economia import PremissasBateria
    from ..bateria.estudo import ConfiguracaoEstudo
    from ..bateria.fontes import Fatura
    from ..demanda.vistoria import DESCRICAO_CRITICIDADE

    dados = entrada.dados
    c = replace(dados.cenario, criticidades_essenciais=CRITICAS)
    q = entrada.questionario
    autonomia = opcoes.autonomia_alvo_h or float(q.get("autonomiaSugeridaH") or 6.0)
    duracoes = sorted({*DURACOES_IMAGEM_H, *(d for d in [entrada.falta_declarada_h, autonomia] if d)})
    consumo_mes = curva.consumo_mensal_kwh
    backup = c.para_comodos(True) if c.tem_criticidade and _tem_critica(c) else None
    return ConfiguracaoEstudo(
        latitude=dados.latitude, longitude=dados.longitude,
        nome=f"{dados.cliente} — {dados.local}", tensao_rede_v=opcoes.tensao_v,
        comodos=c.para_comodos(), instancias_por_comodo=c.instancias_de(),
        ensemble_total=curva.ensemble_total,
        comodos_backup=backup, instancias_backup=c.instancias_de(True) if backup else None,
        criticidade={"tabela": c.por_criticidade(), "corte": CRITICAS, "descricoes": DESCRICAO_CRITICIDADE}
        if backup else None,
        tabelas_cenario=c.comodos,
        consumo_anual_kwh=float(curva.ajuste.consumo_anual_kwh),
        fatura=Fatura(tarifa_brl_kwh=entrada.tarifa_brl_kwh, consumo_mensal_kwh=consumo_mes,
                      custo_disponibilidade_kwh=opcoes.disponibilidade_kwh),
        premissas=PremissasBateria(tarifa_fora_ponta_brl_kwh=entrada.tarifa_brl_kwh),
        simulacoes=opcoes.simulacoes, max_candidatos=opcoes.candidatos,
        autonomia_alvo_h=autonomia, confiabilidade_alvo=CONFIABILIDADE,
        permitir_serie_sintetica=opcoes.permitir_serie_sintetica,
        malha=MalhaApagao(amostras=opcoes.amostras_apagao, duracoes_h=tuple(duracoes)),
    )


def _tem_critica(cenario) -> bool:
    try:
        cenario.para_comodos(True)
        return True
    except ValueError:
        return False


# ----------------------------------------------------------------------------
# Do resultado ao resumo
# ----------------------------------------------------------------------------
def _cidade_uf(dados) -> str:
    return "/".join(x for x in (dados.cidade, dados.uf) if x)


def _criticos(entrada: Entrada, curva: CurvaHibrida) -> list[str]:
    """Como o cliente disse; sem isso no pacote (versão antiga), os rótulos das críticas."""
    textos = entrada.questionario.get("faltasTexto")
    if textos:
        return list(textos)[:4]
    crit = curva.cargas[curva.cargas["criticidade"].isin(CRITICAS)]
    return list(dict.fromkeys(r.lower() for r in crit["rotulo"]))[:4]


def _curva_solar_kw(estudo) -> list[float]:
    if not getattr(estudo, "com_solar", False):
        return [0.0] * 24
    media = np.asarray(estudo.serie.potencia_w_por_kwp, dtype=float).mean(axis=0)
    return [round(float(v) * estudo.potencia_fv_kwp / 1000.0, 3) for v in media]


def _fracao_sem_sol(consumo: list[float], solar: list[float]) -> float | None:
    if not consumo or max(solar or [0]) <= 0:
        return None
    pico = max(solar)
    sem_sol = sum(c for c, s in zip(consumo, solar) if s < 0.05 * pico)
    total = sum(consumo)
    return sem_sol / total if total > 0 else None


#: Prejuízos diretos declarados pelo negócio: pesam como "nem um minuto".
_IMPACTOS_GRAVES = {"mercadoria", "medicamentos", "atendimentos", "servico", "hospedes",
                    "trabalho", "aulas", "cultos", "fecha", "vendas"}


def _nivel_critico(q: dict) -> str:
    """'grave', 'critico' ou 'nenhum', pelo que o cliente disse que fez falta."""
    falta = q.get("fezFalta") or {}
    tolerancia = q.get("tolerancia") or {}
    impacto = (q.get("negocio") or {}).get("impacto") or {}
    if (any(tolerancia.get(k) == "MC" for k in falta) or "saude" in falta
            or any(v and k in _IMPACTOS_GRAVES for k, v in impacto.items())):
        return "grave"
    return "critico" if falta else "nenhum"


def _arredonda_reais(v: float, passo: float = 10.0) -> float:
    return float(passo * round(v / passo))


def resumir(entrada: Entrada, curva: CurvaHibrida, estudo, opcoes: OpcoesPreRelatorio) -> ResumoPreRelatorio:
    dados = entrada.dados
    q = entrada.questionario
    v = entrada.pacote.get("vistoria") or {}

    # bloco 1 — a resiliência do conjunto recomendado
    rec = estudo.recomendado
    bateria_kwh = autonomia = None
    cobertura: list[Cobertura] = []
    if rec is not None and rec.conjunto is not None:
        bateria_kwh = round(float(rec.conjunto.capacidade_nominal_kwh), 1)
        autonomia = float(rec.autonomia_garantida_h(CONFIABILIDADE))
        tabela = rec.por_duracao()
        for linha in tabela.itertuples():
            if any(math.isclose(float(linha.duracao_h), d) for d in DURACOES_IMAGEM_H):
                cobertura.append(Cobertura(float(linha.duracao_h), float(linha.prob_atendimento)))

    # bloco 2 — o dia em energia
    consumo = [round(float(x), 3) for x in curva.curva_total_kw]
    solar = _curva_solar_kw(estudo)

    # bloco 3 — para onde vai a conta
    consumidores = [Consumidor(str(r.rotulo), float(r.fracao), estimado=True)
                    for r in curva.principais_consumidores(3).itertuples()]

    # bloco 4 — a conta, pelo financeiro único da proposta
    consumo_mes = curva.consumo_mensal_kwh
    fatura = estudo.configuracao.fatura
    conta_atual = float(fatura.conta_anual_brl(consumo_mes * 12)) / 12.0
    fin = estudo.financeiro_do_cenario() if hasattr(estudo, "financeiro_do_cenario") else None
    conta_solar_min = conta_solar_max = investimento = payback = None
    if fin is not None and fin.economia_mensal_brl > 0:
        com_solar = max(conta_atual - float(fin.economia_mensal_brl),
                        fatura.tarifa_brl_kwh * fatura.custo_disponibilidade_kwh)
        conta_solar_min = _arredonda_reais(com_solar * (1 - FOLGA_FAIXA))
        conta_solar_max = _arredonda_reais(com_solar * (1 + FOLGA_FAIXA))
        if opcoes.mostrar_investimento:
            investimento = _arredonda_reais(float(fin.capex_brl), 100.0)
            payback = fin.payback_anos

    # índice de oportunidade
    economia = float(fin.economia_mensal_brl) if fin is not None else None
    cobertura_falta = None
    if rec is not None and rec.conjunto is not None:
        alvo_h = entrada.falta_declarada_h or 3.0
        tabela = rec.por_duracao().sort_values("duracao_h")
        cabem = tabela[tabela["duracao_h"] >= alvo_h - 1e-9]
        if not cabem.empty:
            cobertura_falta = float(cabem.iloc[0]["prob_atendimento"])
    apagao = q.get("apagao") or {}
    ind = indice_mod.calcular(
        conta_atual_brl=conta_atual, economia_mensal_brl=economia,
        frequencia=apagao.get("frequencia"), duracao=apagao.get("duracao"),
        nivel_critico=_nivel_critico(q), fracao_sem_sol=_fracao_sem_sol(consumo, solar),
        cobertura_da_falta=cobertura_falta,
    )

    setor_nome = str(v.get("imovel") or "") if entrada.negocio else ""
    return ResumoPreRelatorio(
        cliente=dados.cliente, protocolo=str(entrada.pacote.get("protocolo") or v.get("protocolo") or ""),
        cidade_uf=_cidade_uf(dados), tipo=entrada.tipo, empresa=str(v.get("empresa") or ""), setor=setor_nome,
        data=date.today().strftime("%d/%m/%Y"),
        criticos=_criticos(entrada, curva), bateria_kwh=bateria_kwh, autonomia_h=autonomia, cobertura=cobertura,
        falta_declarada=entrada.falta_declarada, falta_declarada_h=entrada.falta_declarada_h,
        curva_consumo_kw=consumo, curva_solar_kw=solar,
        curva_critica_kw=[round(float(x), 3) for x in curva.curva_critica_kw],
        fracao_sem_sol=_fracao_sem_sol(consumo, solar),
        consumidores=consumidores,
        consumo_mensal_kwh=round(consumo_mes),
        conta_atual_brl=_arredonda_reais(conta_atual),
        conta_solar_min_brl=conta_solar_min, conta_solar_max_brl=conta_solar_max,
        potencia_fv_kwp=round(float(estudo.potencia_fv_kwp), 1) if getattr(estudo, "com_solar", False) else None,
        investimento_brl=investimento, payback_anos=payback,
        origem_consumo=entrada.origem_consumo,
        indice=ind.valor, indice_faixa=ind.faixa, indice_partes=ind.as_dict()["partes"],
        avisos=list(dict.fromkeys(entrada.avisos + curva.avisos)),
    )


def gerar(entrada: Entrada, opcoes: OpcoesPreRelatorio | None = None, progresso=None) -> PreRelatorio:
    """Curva híbrida → estudo → resumo. A imagem é :func:`aurum.pre_relatorio.imagem.desenhar`."""
    from ..bateria.estudo import executar_estudo

    opcoes = opcoes or OpcoesPreRelatorio()
    curva = montar_curva(entrada.dados.cenario, entrada.conta, entrada.perfil_id, simulacoes=opcoes.simulacoes,
                         rotulos=entrada.rotulos)
    cfg = configurar(entrada, curva, opcoes)
    estudo = executar_estudo(cfg, progresso=progresso)
    resumo = resumir(entrada, curva, estudo, opcoes)
    avisos = list(dict.fromkeys(resumo.avisos + list(estudo.avisos or [])))
    return PreRelatorio(entrada=entrada, curva=curva, estudo=estudo, resumo=resumo, avisos=avisos)
