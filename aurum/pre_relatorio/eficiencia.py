"""
Eficiência dos aparelhos: quanto a casa gasta hoje e quanto gastaria fazendo
a mesma coisa com o aparelho eficiente.

É a régua do critério "eficiência de equipamento" do Score PACE
(consumo possível ÷ consumo estimado) e a lista das trocas que aparecem nas
sugestões. O app pergunta uma coisa por tipo de aparelho ("é inverter?",
"quantos anos tem a geladeira?") e a resposta escolhe a linha da tabela.

**A tabela é arbitrada.** Os fatores são a fração do consumo atual que sobra
depois da troca, tirados de ordens de grandeza de mercado (inverter gasta
~35% menos que o convencional, geladeira de mais de 10 anos ~45% mais que uma
Procel A, aquecimento solar cobre ~70% do banho). Os custos são preço de
balcão com instalação, para ordenar as sugestões — não são orçamento.
Recalibrar quando houver obras reais para comparar.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from .curva import DEMAIS

from .score import PARAMETROS

#: Classe do aparelho pelo cartão que o cliente tocou (casa e negócio), e dos
#: itens sem cartão (a luz estimada). Em score_parametros.json, junto da
#: tabela: o app usa os mesmos.
CLASSE_DO_CARTAO: dict[str, str] = dict(PARAMETROS["classes"]["cartoes"])
CLASSE_DO_EQUIPAMENTO: dict[str, str] = dict(PARAMETROS["classes"]["equipamentos"])


@dataclass(frozen=True)
class Troca:
    """Uma linha da tabela: o que fazer, e quanto do consumo sobra por resposta."""
    classe: str
    acao: str
    #: resposta do app -> fração do consumo atual que sobra depois da troca
    fator: dict[str, float]
    #: custo por unidade (ou do sistema, quando ``por_unidade`` é falso)
    custo_brl: float
    por_unidade: bool = True
    #: resposta -> fração das unidades que precisam trocar
    unidades: dict[str, float] = field(default_factory=dict)
    curta: str = ""


TROCAS: dict[str, Troca] = {
    classe: Troca(classe=classe, acao=t["acao"], curta=t.get("curta", t["acao"]),
                  fator={k: float(v) for k, v in t["fator"].items()}, custo_brl=float(t["custo"]),
                  por_unidade=bool(t.get("porUnidade", True)),
                  unidades={k: float(v) for k, v in (t.get("unidades") or {}).items()})
    for classe, t in PARAMETROS["trocas"].items() if not classe.startswith("_")
}
#: Classes sem troca mas com horário que dá para levar para o sol.
DESLOCAVEIS = {k: (None if v is None else tuple(v)) for k, v in PARAMETROS["deslocaveis"].items()
               if not k.startswith("_")}
#: Sem resposta, vale a hipótese do meio.
PADRAO = "nao_sei"


def normalizar(respostas: dict | None) -> dict[str, str]:
    """'sim'/'nao' de um aparelho só valem 'todos'/'nenhum'."""
    r = {}
    for k, v in (respostas or {}).items():
        r[k] = {"sim": "todos", "nao": "nenhum"}.get(v, v) if k == "ar" else v
    return r


def classes_dos_itens(pacote: dict) -> dict[str, str]:
    """Nome do equipamento no pacote -> classe do aparelho (só os que têm classe)."""
    classes: dict[str, str] = {}
    for item in pacote.get("itens") or []:
        classe = CLASSE_DO_CARTAO.get(str(item.get("cartaoId") or "")) or CLASSE_DO_EQUIPAMENTO.get(str(item.get("equipId") or ""))
        if classe:
            classes[str(item.get("equipamento"))] = classe
    return classes


@dataclass(frozen=True)
class TrocaAplicada:
    classe: str
    acao: str
    consumo_kwh_mes: float          # o que a classe gasta hoje
    economia_kwh_mes: float
    unidades: int
    custo_brl: float

    def as_dict(self) -> dict:
        return {"classe": self.classe, "acao": self.acao, "consumo_kwh_mes": round(self.consumo_kwh_mes, 1),
                "economia_kwh_mes": round(self.economia_kwh_mes, 1), "unidades": self.unidades,
                "custo_brl": round(self.custo_brl)}


@dataclass(frozen=True)
class AvaliacaoEficiencia:
    consumo_estimado_kwh_mes: float
    #: consumo por classe de aparelho, inclusive as que não têm troca
    consumo_por_classe: dict[str, float]
    trocas: tuple[TrocaAplicada, ...]

    @property
    def consumo_possivel_kwh_mes(self) -> float:
        return self.consumo_estimado_kwh_mes - sum(t.economia_kwh_mes for t in self.trocas)

    def possivel_com(self, classes: set[str] | None) -> float:
        """O consumo fazendo só as trocas das ``classes`` (None = todas)."""
        return self.consumo_estimado_kwh_mes - sum(
            t.economia_kwh_mes for t in self.trocas if classes is None or t.classe in classes)

    @property
    def razao(self) -> float:
        """Consumo possível ÷ estimado: 1 quando não há o que trocar."""
        if self.consumo_estimado_kwh_mes <= 0:
            return 1.0
        return self.consumo_possivel_kwh_mes / self.consumo_estimado_kwh_mes


def avaliar(cargas: pd.DataFrame, pacote: dict, respostas: dict | None = None) -> AvaliacaoEficiencia:
    """
    ``cargas`` é a tabela da curva híbrida (já fechada com a conta): a energia
    de cada aparelho é a mesma que o pré-relatório mostra. As demais cargas
    entram no consumo estimado e não têm troca.
    """
    respostas = normalizar(respostas if respostas is not None else (pacote.get("questionario") or {}).get("aparelhos"))
    classes = classes_dos_itens(pacote)
    total = float(cargas["energia_kwh_mes"].sum())
    linhas = cargas[cargas["equipamento"] != DEMAIS].copy()
    linhas["classe"] = linhas["equipamento"].map(classes)
    por_classe = linhas.dropna(subset=["classe"]).groupby("classe").agg(
        energia=("energia_kwh_mes", "sum"), quantidade=("quantidade", "sum"))

    trocas = []
    for classe, linha in por_classe.iterrows():
        troca = TROCAS.get(str(classe))
        if troca is None:
            continue
        resposta = respostas.get(str(classe)) or PADRAO
        fator = troca.fator.get(resposta, troca.fator.get(PADRAO, 1.0))
        economia = float(linha["energia"]) * (1.0 - fator)
        if economia <= 0.05:
            continue
        if troca.por_unidade:
            # A madrugada do ar é o mesmo aparelho: conta as unidades pela maior linha.
            qtd = float(linhas.loc[linhas["classe"] == classe, "quantidade"].max())
            unidades = max(1, math.ceil(qtd * troca.unidades.get(resposta, 1.0) - 1e-9))
        else:
            unidades = 1
        trocas.append(TrocaAplicada(classe=str(classe), acao=troca.acao, consumo_kwh_mes=float(linha["energia"]),
                                    economia_kwh_mes=economia, unidades=unidades,
                                    custo_brl=troca.custo_brl * unidades))
    trocas.sort(key=lambda t: t.economia_kwh_mes, reverse=True)
    return AvaliacaoEficiencia(consumo_estimado_kwh_mes=total,
                               consumo_por_classe={str(k): float(v) for k, v in por_classe["energia"].items()},
                               trocas=tuple(trocas))


def energia_deslocavel_kwh_mes(avaliacao: AvaliacaoEficiencia, respostas: dict | None) -> float:
    """O que dá para levar para a hora do sol: máquinas, piscina, boiler, carro."""
    respostas = normalizar(respostas)
    total = 0.0
    for classe, aceitas in DESLOCAVEIS.items():
        if aceitas is not None and respostas.get(classe) not in aceitas and respostas.get(classe) is not None:
            continue
        total += avaliacao.consumo_por_classe.get(classe, 0.0)
    return total
