"""
O quadro de backup em dois níveis: o essencial e o desejável.

A vistoria classifica cada equipamento em quatro níveis — MC, C, P e NC — e
até aqui o estudo usava só a linha de corte: o que estava acima entrava no
quadro, o que estava abaixo ficava de fora. Isso responde "o que a bateria
segura" e deixa sem resposta a pergunta que vem logo depois, que é comercial
e não técnica: **quanto custa levar junto o que seria bom ter?**

O nível ``P`` existe justamente para essa pergunta. Não é crítico — a casa
não para sem ele — mas o cliente sente falta. Trata-se de decidir se vale
pagar por ele, e essa decisão precisa de três números que este módulo produz:

* **O que ele consome.** Energia por dia e pico, medidos separados do
  essencial, para se saber o tamanho do problema antes de precificá-lo.
* **O banco que ele exige.** Não o banco do conjunto, mas a diferença entre o
  banco do escopo ampliado e o do essencial. É o custo marginal de promover
  os preferíveis, e é o número que fecha ou não fecha a venda.
* **A ociosidade que já está paga.** Bancos vêm em degraus: um banco
  dimensionado para 1,9 kW de carga essencial quase nunca fica exatamente em
  1,9 kW — sobra capacidade, e essa sobra já foi comprada. Medir por quanto
  tempo o banco do essencial atravessaria **também** os preferíveis responde
  quanto do desejável sai de graça, e a partir de onde começa a custar.

A ordem importa: primeiro o essencial sozinho, com o seu banco; depois o
ampliado, com o dele. Dimensionar o conjunto todo de uma vez e depois tentar
separar o que é de quem não funciona, porque a coincidência entre as duas
cargas não é aditiva — o pico do conjunto é menor que a soma dos picos.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..demanda import simular_ensemble
from ..demanda.ensemble import EnsembleCarga
from .apagao import MalhaApagao, ResultadoResiliencia, avaliar_conjunto
from .catalogo import ConjuntoArmazenamento
from .economia import PremissasBateria, _capex
from .geracao import SerieGeracao

__all__ = [
    "ESCOPOS",
    "ComparacaoEscopos",
    "Escopo",
    "MedidaEscopo",
    "comparar_escopos",
]


@dataclass(frozen=True)
class Escopo:
    """Um recorte do quadro de backup, definido pelos níveis que aceita."""

    id: str
    nome: str
    criticidades: tuple[str, ...]
    pergunta: str

    @property
    def rotulo_dos_niveis(self) -> str:
        return " + ".join(self.criticidades)


#: Os dois recortes que fazem sentido comparar.
#:
#: Um terceiro — só MC — foi tentado e descartado: numa residência o nível MC
#: costuma ser uma lâmpada de emergência, e o banco que o atende é pequeno
#: demais para existir no catálogo. A comparação útil é entre o que não pode
#: faltar e o que seria bom não faltar.
ESCOPOS: dict[str, Escopo] = {
    "essencial": Escopo(
        "essencial", "Só o essencial", ("MC", "C"),
        "O que não pode faltar. É este escopo que define o banco mínimo, e é "
        "a promessa que a proposta assina.",
    ),
    "ampliado": Escopo(
        "ampliado", "Essencial + preferível", ("MC", "C", "P"),
        "O que seria bom não faltar, junto com o que não pode. A diferença "
        "entre os dois escopos é o preço de levar o desejável.",
    ),
}

#: O escopo que dimensiona e que a proposta promete.
ESCOPO_BASE = "essencial"


@dataclass
class MedidaEscopo:
    """O que um escopo consome, e o banco que ele exige."""

    escopo: Escopo
    equipamentos: int
    potencia_instalada_w: float
    energia_diaria_kwh: float
    pico_p95_kw: float
    curva_w: np.ndarray
    passo_min: int
    ensemble: EnsembleCarga
    conjunto: ConjuntoArmazenamento | None = None
    resiliencia: ResultadoResiliencia | None = None
    autonomia_h: float = 0.0
    capex_brl: float = 0.0
    avisos: list[str] = field(default_factory=list)

    @property
    def energia_util_kwh(self) -> float:
        return float(self.conjunto.energia_util_kwh) if self.conjunto else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "escopo": self.escopo.nome,
            "niveis": self.escopo.rotulo_dos_niveis,
            "equipamentos": self.equipamentos,
            "potencia_instalada_kw": round(self.potencia_instalada_w / 1000.0, 2),
            "energia_diaria_kwh": round(self.energia_diaria_kwh, 1),
            "pico_p95_kw": round(self.pico_p95_kw, 2),
            "energia_util_kwh": round(self.energia_util_kwh, 2),
            "autonomia_h": round(self.autonomia_h, 1),
            "capex_brl": round(self.capex_brl, 2),
        }


@dataclass
class ComparacaoEscopos:
    """Os dois escopos lado a lado, e o que separa um do outro."""

    medidas: list[MedidaEscopo]
    #: Autonomia do banco do escopo base carregando o escopo ampliado — a
    #: leitura da ociosidade: quanto do desejável a sobra já atende.
    autonomia_do_base_no_ampliado_h: float = 0.0
    #: A varredura inteira desse cruzamento. Guardada porque a autonomia
    #: sozinha não diz **por que** o banco não aguenta, e a resposta muda a
    #: recomendação: faltar kWh se resolve com mais módulo de bateria; faltar
    #: kW exige inversor maior, que é outro equipamento e outro preço.
    cruzado: ResultadoResiliencia | None = None
    confiabilidade: float = 0.95
    autonomia_alvo_h: float = 0.0
    avisos: list[str] = field(default_factory=list)

    def de(self, identificador: str) -> MedidaEscopo | None:
        return next((m for m in self.medidas if m.escopo.id == identificador), None)

    @property
    def base(self) -> MedidaEscopo | None:
        return self.de(ESCOPO_BASE)

    @property
    def ampliado(self) -> MedidaEscopo | None:
        return self.de("ampliado")

    @property
    def tem_preferiveis(self) -> bool:
        """
        Há o que comparar?

        Quando a vistoria não classificou nada como ``P``, os dois escopos são
        o mesmo quadro e a comparação vira uma tabela de duas linhas iguais —
        pior que não mostrar nada, porque sugere uma escolha que não existe.
        """
        base, ampliado = self.base, self.ampliado
        if base is None or ampliado is None:
            return False
        return ampliado.equipamentos > base.equipamentos

    def marginal(self) -> dict[str, float]:
        """
        O preço de promover os preferíveis, em cada moeda que importa.

        Energia e potência são o que muda na carga; energia útil e CAPEX são o
        que muda no equipamento. As quatro juntas respondem se vale a pena, e
        nenhuma delas responde sozinha.
        """
        base, ampliado = self.base, self.ampliado
        if base is None or ampliado is None:
            return {}
        capex = ampliado.capex_brl - base.capex_brl
        return {
            "energia_diaria_kwh": ampliado.energia_diaria_kwh - base.energia_diaria_kwh,
            "pico_kw": ampliado.pico_p95_kw - base.pico_p95_kw,
            "energia_util_kwh": ampliado.energia_util_kwh - base.energia_util_kwh,
            "capex_brl": capex,
            "capex_por_kwh_dia": (
                capex / (ampliado.energia_diaria_kwh - base.energia_diaria_kwh)
                if ampliado.energia_diaria_kwh > base.energia_diaria_kwh else float("nan")
            ),
        }

    def ociosidade(self) -> dict[str, float]:
        """
        Quanto do desejável a sobra do banco essencial já atende.

        Banco vem em degraus: quem dimensiona para 1,9 kW quase nunca compra
        exatamente 1,9 kW. A folga que sobrou já está paga, e é ela que
        atende os preferíveis nas primeiras horas sem custo nenhum. Depois
        dela é que começa a conta.
        """
        base, ampliado = self.base, self.ampliado
        if base is None or ampliado is None or not base.conjunto:
            return {}
        # A folga é a energia que o banco tem além do que o escopo essencial
        # consome na autonomia prometida.
        consumo_na_autonomia = base.energia_diaria_kwh / 24.0 * max(base.autonomia_h, 0.0)
        medida = {
            "energia_util_kwh": base.energia_util_kwh,
            "potencia_kw": float(base.conjunto.potencia_descarga_kw),
            "folga_kwh": max(0.0, base.energia_util_kwh - consumo_na_autonomia),
            "autonomia_no_ampliado_h": self.autonomia_do_base_no_ampliado_h,
            "fracao_da_autonomia_prometida": (
                self.autonomia_do_base_no_ampliado_h / base.autonomia_h
                if base.autonomia_h > 0 else 0.0
            ),
        }
        if self.cruzado is not None:
            tabela = self.cruzado.tabela
            medida["falhas_por_potencia"] = float(tabela["falhas_por_potencia"].mean())
            medida["falhas_por_energia"] = float(tabela["falhas_por_energia"].mean())
        return medida

    @property
    def limitante(self) -> str:
        """
        O que impede o banco do essencial de carregar também o desejável.

        A distinção decide o que comprar, e é a pergunta que a autonomia
        sozinha não responde. Faltar **energia** se resolve acrescentando
        módulo de bateria ao mesmo inversor — é barato e é linear. Faltar
        **potência** exige inversor maior: outro equipamento, outro preço, e
        nenhum módulo extra resolve. Quando o pico do quadro ampliado passa da
        descarga do banco, ele desarma no primeiro instante, e a folga de kWh
        que sobrou não serve para nada.
        """
        base = self.base
        if self.cruzado is None or base is None or base.conjunto is None:
            return "indeterminado"
        ampliado = self.ampliado
        if ampliado is not None and ampliado.pico_p95_kw > base.conjunto.potencia_descarga_kw:
            return "potencia"
        tabela = self.cruzado.tabela
        por_potencia = float(tabela["falhas_por_potencia"].mean())
        por_energia = float(tabela["falhas_por_energia"].mean())
        if por_potencia <= 0 and por_energia <= 0:
            return "nenhum"
        return "potencia" if por_potencia >= por_energia else "energia"

    def tabela(self) -> pd.DataFrame:
        return pd.DataFrame([m.as_dict() for m in self.medidas])


# ----------------------------------------------------------------------------
def _mais_barato_que_cumpre(
    candidatos: Sequence[ConjuntoArmazenamento],
    ensemble: EnsembleCarga,
    serie: SerieGeracao,
    malha: MalhaApagao,
    autonomia_alvo_h: float,
    confiabilidade: float,
    semente: int,
    premissas: PremissasBateria,
) -> tuple[ConjuntoArmazenamento | None, ResultadoResiliencia | None, float, float]:
    """
    O banco **mais barato** que cumpre a meta — e, se nenhum cumpre, o melhor.

    O primeiro critério tentado foi o menor por energia útil, e ele produziu um
    resultado absurdo: quando o quadro ampliado falha por *potência*, todos os
    bancos pequenos falham por mais energia que tenham, e o primeiro a passar
    na varredura por energia crescente é um banco enorme que veio junto de um
    inversor grande. Num caso real isso devolveu 189 kWh e R$ 872 mil para uma
    residência, quando um banco de 10 kWh com inversor maior resolveria.

    Ordenar por preço resolve porque é o preço que carrega as duas dimensões:
    inversor e módulos entram na mesma conta, e o mais barato que cumpre é
    justamente o que equilibra os dois. Devolver o melhor quando nenhum cumpre
    é deliberado — um estudo que responde "nenhum serve" e para não ajuda
    ninguém a decidir.
    """
    def preco(conjunto: ConjuntoArmazenamento) -> float:
        return _capex(conjunto, premissas)[0]

    melhor: tuple[ConjuntoArmazenamento, ResultadoResiliencia, float, float] | None = None
    for conjunto in sorted(candidatos, key=preco):
        resultado = avaliar_conjunto(conjunto, ensemble, serie, malha, semente)
        autonomia = resultado.autonomia_garantida_h(confiabilidade)
        custo = preco(conjunto)
        if melhor is None or autonomia > melhor[2]:
            melhor = (conjunto, resultado, autonomia, custo)
        if autonomia >= autonomia_alvo_h:
            return conjunto, resultado, autonomia, custo
    if melhor is None:
        return None, None, 0.0, 0.0
    return melhor


def _candidatos_do_escopo(
    catalogo: Any,
    energia_alvo_kwh: float,
    quantos: int = 14,
    modulos: tuple[int, ...] = (1, 2, 3, 4, 6, 8),
) -> list[ConjuntoArmazenamento]:
    """
    Uma lista de candidatos ancorada na energia que **este** escopo exige.

    Reaproveitar a lista do estudo não funciona, e o caso real mostrou por quê:
    ela é montada em torno da energia do quadro essencial, e o quadro ampliado
    precisa de bancos que simplesmente não estão nela. O resultado foi o estudo
    devolver o único candidato grande que sobrava — 189 kWh e R$ 872 mil para
    uma residência — quando havia solução no meio da faixa que a lista não
    continha.

    A âncora é a energia do escopo, e a busca pega candidatos dos dois lados
    dela: abaixo, porque o sol pode encurtar o banco; acima, porque a potência
    pode obrigar a subir.
    """
    from .catalogo import candidatos_em_blocos

    # Blocos, e não a varredura do catálogo: é a mesma unidade em que o preço
    # é cotado e em que o cliente decide. A lista fica curta o bastante para
    # dispensar amostragem — doze blocos, e não trezentas combinações.
    todos = candidatos_em_blocos(catalogo) or list(catalogo.combinacoes(modulos=modulos))
    if not todos:
        return []
    if energia_alvo_kwh <= 0 or len(todos) <= quantos:
        return sorted(todos, key=lambda c: c.energia_util_kwh)[:quantos]

    fatores = (0.4, 0.6, 0.8, 1.0, 1.25, 1.6, 2.0, 2.6, 3.4, 4.5, 0.25, 6.0, 8.0, 12.0)
    escolhidos: dict[str, ConjuntoArmazenamento] = {}
    for fator in fatores[:quantos]:
        alvo = energia_alvo_kwh * fator
        perto = min(todos, key=lambda c: abs(c.energia_util_kwh - alvo))
        escolhidos.setdefault(perto.descricao(), perto)
    return sorted(escolhidos.values(), key=lambda c: c.energia_util_kwh)


def comparar_escopos(
    tabelas_por_comodo: dict[str, pd.DataFrame],
    instancias: dict[str, int],
    candidatos: Sequence[ConjuntoArmazenamento],
    serie: SerieGeracao,
    malha: MalhaApagao,
    autonomia_alvo_h: float = 6.0,
    confiabilidade: float = 0.95,
    simulacoes: int = 200,
    premissas: PremissasBateria | None = None,
    semente: int = 20260902,
    ajustes_sazonais: dict | None = None,
    catalogo: Any = None,
) -> ComparacaoEscopos:
    """
    Mede os dois escopos e dimensiona o banco de cada um.

    Recebe as tabelas do cenário — com a coluna ``criticidade`` que a vistoria
    trouxe — e não os objetos já convertidos do núcleo, porque é a coluna de
    criticidade que define o recorte e ela se perde na conversão.
    """
    from ..demanda.nucleo import cria_comodo_da_planilha

    premissas = premissas or PremissasBateria()
    medidas: list[MedidaEscopo] = []
    avisos: list[str] = []

    for escopo in ESCOPOS.values():
        aceitos = {c.upper() for c in escopo.criticidades}
        comodos, instancias_escopo, equipamentos, potencia = [], {}, 0, 0.0
        for nome, tabela in tabelas_por_comodo.items():
            if "criticidade" not in tabela.columns:
                continue
            recorte = tabela[tabela["criticidade"].astype(str).str.upper().isin(aceitos)]
            if recorte.empty:
                continue
            n = int(instancias.get(nome, 1))
            comodos.append(cria_comodo_da_planilha(recorte.reset_index(drop=True), nome))
            instancias_escopo[nome] = n
            equipamentos += len(recorte)
            potencia += float(
                (recorte["Potência"].astype(float) * recorte["Quantidade"].astype(float)).sum()
            ) * n

        if not comodos:
            avisos.append(
                f"Nenhum equipamento classificado como {escopo.rotulo_dos_niveis}; "
                f"o escopo “{escopo.nome}” ficou de fora da comparação."
            )
            continue

        ensemble = simular_ensemble(
            comodos, instancias_escopo, simulacoes, ajustes_sazonais, semente=semente)
        suave = ensemble.reamostrar(15) if ensemble.passo_min == 1 else ensemble
        geral = ensemble.resumo()["geral"]

        # Cada escopo monta a sua própria lista, ancorada na energia que ele
        # exige. Compartilhar a lista do estudo — feita para o quadro
        # essencial — deixava o quadro ampliado sem candidato no meio da faixa.
        energia_alvo = float(np.mean(ensemble.perfis())) / 1000.0 * autonomia_alvo_h
        do_escopo = (
            _candidatos_do_escopo(catalogo, energia_alvo)
            if catalogo is not None else list(candidatos)
        ) or list(candidatos)

        conjunto, resiliencia, autonomia, capex = _mais_barato_que_cumpre(
            do_escopo, ensemble, serie, malha, autonomia_alvo_h,
            confiabilidade, semente, premissas)

        medidas.append(MedidaEscopo(
            escopo=escopo,
            equipamentos=equipamentos,
            potencia_instalada_w=potencia,
            energia_diaria_kwh=float(geral["energia_diaria_media_kwh"]),
            pico_p95_kw=float(geral["pico_p95_kw"]),
            curva_w=suave.perfil_medio_w(),
            passo_min=suave.passo_min,
            ensemble=ensemble,
            conjunto=conjunto,
            resiliencia=resiliencia,
            autonomia_h=autonomia,
            capex_brl=capex,
        ))

    comparacao = ComparacaoEscopos(
        medidas=medidas, confiabilidade=confiabilidade,
        autonomia_alvo_h=autonomia_alvo_h, avisos=avisos,
    )

    # A leitura da ociosidade: o banco comprado para o essencial, carregando o
    # quadro ampliado. É o que diz quanto do desejável já vem junto.
    base, ampliado = comparacao.base, comparacao.ampliado
    if base is not None and ampliado is not None and base.conjunto is not None:
        cruzado = avaliar_conjunto(
            base.conjunto, ampliado.ensemble, serie, malha, semente)
        comparacao.cruzado = cruzado
        comparacao.autonomia_do_base_no_ampliado_h = cruzado.autonomia_garantida_h(
            confiabilidade)
    return comparacao
