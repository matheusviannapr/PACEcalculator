"""
A entrada do pré-relatório: o JSON do questionário mais o que o operador leu
na conta de luz.

O pacote ``pace-coleta/1`` do questionário do cliente (``vistoria/cliente``)
é o mesmo que a calculadora já lê pela coleta residencial. Ele não traz o que
só a fatura diz — kWh faturados, tarifa, tensão —, e por isso o operador
informa. Quando não informa, a entrada estima e **diz que estimou**:

* kWh lidos da conta → ``origem_consumo = "conta"`` (o certo);
* sem kWh, mas com o valor em R$ que o cliente digitou → divide pela tarifa,
  ``"valor"`` (impostos e disponibilidade fazem isso errar para cima);
* sem nada → a própria simulação das respostas, ``"respostas"`` (pior caso).

Duas correções de segurança entram aqui, antes de o motor ver o pacote:

* **Refrigeração contínua.** O catálogo da vistoria descreve geladeira e
  freezer como "dinâmico, 0,25 a 0,75 h, 45%" — o motor lê isso como meia hora
  de uso por dia, e a geladeira sai com 1 kWh/mês. Aqui ela vira carga de
  24 h com o fator de ciclo no FD, como a biblioteca do motor modela.
* **Horário de funcionamento** do negócio vira o horário da conta
  (:class:`aurum.demanda.ajuste.ContaDeLuz`), para o dia fechado cair na
  carga de base. As faixas abaixo espelham ``vistoria/cliente/negocio.js``.
"""
from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..coleta import DadosColeta, ler_pacote
from ..demanda.ajuste import ContaDeLuz

# -- espelho de negocio.js (ABRE, FECHA, TURNOS) ------------------------------
ABRE_H = {"madrugada": 5, "cedo": 7, "manha": 9, "almoco": 11, "tarde": 15}
FECHA_H = {"cedo": 15, "tarde": 19, "noite": 22, "tarde_noite": 24}
TURNOS_H = {"manha": (7, 12), "tarde": (13, 18), "noite": (18, 22)}
DIAS_SEMANA = {"seg_sex": 5, "seg_sab": 6, "todos": 7, "fim_semana": 2, "alguns": 3}

#: A falta de luz que o cliente descreveu, em horas — a mesma escala da
#: autonomia sugerida no pacote.
DURACAO_FALTA_H = {"minutos": 0.5, "horas": 2.0, "meio_dia": 6.0, "dia": 12.0}
DURACAO_FALTA_TEXTO = {"minutos": "alguns minutos", "horas": "uma ou duas horas",
                       "meio_dia": "até meio dia", "dia": "mais de meio dia"}

#: Fator de ciclo do compressor por tipo, alinhado a aurum.demanda.biblioteca.
FD_REFRIGERACAO = (("frigobar", 0.30), ("adega", 0.30), ("bebedouro", 0.30), ("purificador", 0.30),
                   ("geladeira", 0.35), ("freezer", 0.40), ("cervejeira", 0.40),
                   ("expositor", 0.50), ("gelo", 0.50))


@dataclass
class Entrada:
    pacote: dict
    dados: DadosColeta
    conta: ContaDeLuz
    perfil_id: str
    tipo: str                     # "casa" | "negocio"
    setor: str | None
    tarifa_brl_kwh: float
    origem_consumo: str           # "conta" | "valor" | "respostas"
    falta_declarada: str = ""
    falta_declarada_h: float | None = None
    #: Equipamento do catálogo -> nome do cartão que o cliente tocou ("Balcão refrigerado").
    rotulos: dict[str, str] = field(default_factory=dict)
    avisos: list[str] = field(default_factory=list)

    @property
    def questionario(self) -> dict:
        return self.pacote.get("questionario") or {}

    @property
    def negocio(self) -> bool:
        return self.tipo == "negocio"


def carregar(origem: str | Path | dict) -> dict:
    if isinstance(origem, dict):
        return copy.deepcopy(origem)
    return json.loads(Path(origem).read_text(encoding="utf-8-sig"))


def corrigir_refrigeracao(pacote: dict) -> int:
    """Refrigeração "dinâmica de meia hora" vira carga contínua com FD. Devolve quantos itens mudaram."""
    mudados = 0
    for item in pacote.get("itens") or []:
        if str(item.get("categoria") or "") != "Refrigeração":
            continue
        if str(item.get("tipoInt") or "").startswith("fixo") or float(item.get("durMax") or 0) > 2:
            continue
        nome = str(item.get("equipamento") or "").lower()
        fd = next((v for chave, v in FD_REFRIGERACAO if chave in nome), 0.4)
        item.update(tipoInt="fixo", intervalo="00:00 as 23:59", prob=1, fd=fd,
                    durMin=None, durMax=None, modo="FIXO_100%")
        mudados += 1
    return mudados


def _horario_da_conta(questionario: dict) -> tuple[int | None, int | None, int]:
    """(abertura, fechamento, dias por semana) do negócio; 24 h e 7 dias para a casa."""
    if questionario.get("tipo") != "negocio":
        return None, None, 7
    n = questionario.get("negocio") or {}
    setor = questionario.get("setor")
    dias = DIAS_SEMANA.get(n.get("dias"), 7)
    if setor == "pousada":
        return None, None, 7
    if setor == "escola":
        turnos = [TURNOS_H[t] for t, v in (n.get("turnos") or {}).items() if v and t in TURNOS_H] or [(7, 12), (13, 18)]
        return min(t[0] for t in turnos), max(t[1] for t in turnos), dias
    abre, fecha = ABRE_H.get(n.get("abre"), 8), FECHA_H.get(n.get("fecha"), 19)
    if fecha <= abre:
        fecha = min(24, abre + 8)
    return abre, fecha, dias


def montar_entrada(
    origem: str | Path | dict,
    consumo_mensal_kwh: float | None = None,
    tarifa_brl_kwh: float | None = None,
    consumo_por_mes_kwh: dict[int, float] | None = None,
) -> Entrada:
    """
    Lê o pacote e junta o que a conta diz.

    ``tarifa_brl_kwh`` é a tarifa **efetiva** — valor total da conta dividido
    pelos kWh faturados —, não a de tabela da distribuidora.
    """
    pacote = carregar(origem)
    avisos: list[str] = []
    n = corrigir_refrigeracao(pacote)
    if n:
        avisos.append(f"{n} item(ns) de refrigeração tratado(s) como carga contínua com fator de ciclo "
                      "(o catálogo da vistoria os descreve como uso de meia hora por dia).")
    dados = ler_pacote(pacote)
    q = pacote.get("questionario") or {}
    tipo = q.get("tipo") or "casa"
    tarifa = float(tarifa_brl_kwh) if tarifa_brl_kwh else 1.0
    if not tarifa_brl_kwh:
        avisos.append("Tarifa não informada: usado R$ 1,00/kWh. Leia a tarifa efetiva na conta "
                      "(valor total ÷ kWh) para a economia sair certa.")

    valor = (pacote.get("vistoria") or {}).get("contaValorMensal")
    if consumo_mensal_kwh:
        kwh, origem_consumo = float(consumo_mensal_kwh), "conta"
    elif consumo_por_mes_kwh:
        kwh, origem_consumo = None, "conta"
    elif valor:
        kwh, origem_consumo = float(valor) / tarifa, "valor"
        avisos.append(f"Consumo estimado pelo valor que o cliente digitou (R$ {float(valor):,.0f} ÷ tarifa): "
                      "leia os kWh na conta anexada.".replace(",", "."))
    else:
        kwh, origem_consumo = None, "respostas"
        avisos.append("Sem kWh da conta e sem valor digitado: o consumo vem das respostas, "
                      "que costumam errar para baixo.")

    if kwh is None and not consumo_por_mes_kwh:
        from ..demanda.ensemble import simular_ensemble
        c = dados.cenario
        ens = simular_ensemble(c.para_comodos(), c.instancias_de(), num_simulacoes=80, semente=3)
        kwh = max(float(ens.energia_diaria_kwh().mean()) * 30.0, 30.0)

    abre, fecha, dias = _horario_da_conta(q)
    conta = ContaDeLuz(consumo_mensal_kwh=kwh, consumo_por_mes_kwh=consumo_por_mes_kwh,
                       abertura_h=abre, fechamento_h=fecha if fecha is None else min(int(math.ceil(fecha)), 24),
                       dias_operacao_semana=dias)

    rotulos = {str(i.get("equipamento")): str(i["cartao"]) for i in pacote.get("itens") or [] if i.get("cartao")}
    duracao = (q.get("apagao") or {}).get("duracao")
    return Entrada(
        pacote=pacote, dados=dados, conta=conta,
        perfil_id=str(q.get("perfilTipico") or "residencial"),
        tipo=tipo, setor=q.get("setor"), tarifa_brl_kwh=tarifa, origem_consumo=origem_consumo,
        falta_declarada=DURACAO_FALTA_TEXTO.get(duracao, ""), falta_declarada_h=DURACAO_FALTA_H.get(duracao),
        rotulos=rotulos, avisos=avisos,
    )
