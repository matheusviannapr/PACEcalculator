"""
Leitura do backup da vistoria técnica.

A vistoria em campo produz três arquivos: um inventário em planilha, uma
planilha reduzida "de simulador" e um backup em JSON. **O JSON é a fonte**, e
os outros dois são derivados empobrecidos dele:

* a planilha do simulador tem 10 colunas e joga fora o cômodo e a criticidade
  — que são justamente o que decide o que entra no quadro de backup;
* o inventário tem as 31 colunas, mas sai com o texto corrompido (``Ã§Ã£`` no
  lugar de ``çã``), porque é escrito em UTF-8 e lido como Latin-1 pelo Excel;
* o JSON tem tudo, com o texto certo, e ainda traz o que nenhuma das duas
  planilhas carrega: a tensão da rede, se existe FV, gerador ou nobreak, e o
  fator de partida de cada equipamento.

Por isso este módulo lê o JSON por padrão e trata as planilhas como plano B.

**A criticidade é por equipamento, não por ambiente.** É a diferença que
importa: numa cozinha, a geladeira é crítica e o forno elétrico não, e um
quadro de backup montado por cômodo levaria os dois. A escala da vistoria tem
quatro níveis, e eles vêm com o tempo máximo sem energia e a fonte admitida,
que é o que dá sentido a cada um:

===  ==========================  ===============  ==========================
Cód  Significado                 Tempo máximo     Fonte admitida
===  ==========================  ===============  ==========================
MC   Muito crítico               0 min            Bateria/UPS + gerador
C    Crítico                     0 a 120 min      Qualquer gerador
P    Postergável                 120 min          Gerador se houver margem
NC   Não crítico                 1440 min         Somente rede
===  ==========================  ===============  ==========================

Só o que a vistoria marcou ``simular`` e ``presente`` entra na simulação — um
equipamento levantado mas ausente não tem por que virar carga.
"""
from __future__ import annotations

import json
import logging
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .biblioteca import COLUNAS
from .cenario import Cenario

LOGGER = logging.getLogger(__name__)

__all__ = [
    "CRITICIDADES",
    "DadosVistoria",
    "ler_backup",
    "ler_inventario",
]

#: Os quatro níveis, do mais para o menos crítico. A ordem é significativa:
#: ``CRITICIDADES.index`` dá a severidade, e o quadro de backup é definido por
#: um corte nessa escala.
CRITICIDADES: tuple[str, ...] = ("MC", "C", "P", "NC")

#: O que cada nível quer dizer, para o relatório não publicar sigla solta.
DESCRICAO_CRITICIDADE: dict[str, str] = {
    "MC": "muito crítico — não admite interrupção nenhuma",
    "C": "crítico — admite minutos, e exige gerador ou banco",
    "P": "postergável — atende se houver margem",
    "NC": "não crítico — pode ficar sem energia até a rede voltar",
}

#: Coluna que o cenário passa a carregar, além das dez do D².
COLUNA_CRITICIDADE = "criticidade"
COLUNA_COMODO = "comodo"
COLUNA_FATOR_PARTIDA = "fator_partida"


@dataclass
class DadosVistoria:
    """
    Uma vistoria lida, pronta para virar estudo.

    Junta o que o :class:`~aurum.demanda.cenario.Cenario` precisa com o que
    só a vistoria sabe — tensão da rede, se já existe geração ou nobreak, e
    quem levantou. Sem isso o operador teria que redigitar na tela dados que
    o vistoriador já anotou em campo.
    """

    cenario: Cenario
    cliente: str
    imovel: str = ""
    cidade: str = ""
    uf: str = ""
    data: str = ""
    vistoriador: str = ""
    #: Tensão de linha em volts: 220 para rede 127/220, 380 para 220/380.
    tensao_rede_v: float = 380.0
    tem_fv: bool = False
    tem_gerador: bool = False
    tem_nobreak: bool = False
    observacoes: str = ""
    #: Itens levantados que **não** entram na simulação, e por quê.
    descartados: list[str] = field(default_factory=list)
    avisos: list[str] = field(default_factory=list)

    @property
    def nome(self) -> str:
        """Como o estudo se chama: cliente e imóvel, quando há os dois."""
        if self.imovel and self.imovel not in self.cliente:
            return f"{self.cliente} — {self.imovel}"
        return self.cliente

    @property
    def local(self) -> str:
        return " / ".join(x for x in (self.cidade, self.uf) if x)

    def resumo(self) -> dict[str, Any]:
        return {
            "cliente": self.cliente,
            "imovel": self.imovel,
            "local": self.local,
            "data": self.data,
            "vistoriador": self.vistoriador,
            "tensao_rede_v": self.tensao_rede_v,
            "tem_fv": self.tem_fv,
            "tem_gerador": self.tem_gerador,
            "tem_nobreak": self.tem_nobreak,
            "comodos": len(self.cenario.comodos),
            "equipamentos": self.cenario.total_de_equipamentos(),
            "descartados": len(self.descartados),
        }


# ----------------------------------------------------------------------------
# Conversão de um item
# ----------------------------------------------------------------------------
def _texto(valor: Any) -> str:
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return ""
    return str(valor).strip()


def _num(valor: Any) -> float | None:
    try:
        numero = float(valor)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(numero) else numero


def _normalizar_tipo(tipo: str) -> str:
    """
    ``dinâmico`` com ou sem acento, e com ou sem corrupção de encoding.

    O inventário chega com ``dinÃ¢mico`` quando o Excel lê UTF-8 como
    Latin-1. Comparar sem acento resolve os dois casos de uma vez, e é mais
    barato que tentar adivinhar a codificação do arquivo.
    """
    limpo = unicodedata.normalize("NFKD", tipo.lower())
    limpo = "".join(c for c in limpo if not unicodedata.combining(c))
    return "dinâmico" if limpo.startswith("din") else "fixo"


def _linha_do_item(item: dict[str, Any]) -> dict[str, Any] | None:
    """
    Um item da vistoria vira uma linha no formato do D².

    Devolve ``None`` para o que não deve ser simulado. A potência adotada é a
    informada quando existe e a típica quando não — a vistoria já resolve essa
    precedência ao gravar, e repeti-la aqui é a garantia de que um backup
    antigo, sem o campo, continue a ser lido.
    """
    potencia = _num(item.get("potInformada")) or _num(item.get("potTipica"))
    if not potencia or potencia <= 0:
        return None

    tipo = _normalizar_tipo(_texto(item.get("tipoInt")) or "fixo")
    linha: dict[str, Any] = {
        "Equipamento": _texto(item.get("equipamento")) or "sem nome",
        "Potência": float(potencia),
        "Quantidade": int(_num(item.get("quantidade")) or 1),
        "Tipo de intervalo": tipo,
        "intervalo": _texto(item.get("intervalo")),
        "probabilidade": _num(item.get("prob")),
        "FD": _num(item.get("fd")),
        "duracao_min": _num(item.get("durMin")) if tipo == "dinâmico" else None,
        "duracao_max": _num(item.get("durMax")) if tipo == "dinâmico" else None,
        "modo_fixo": _texto(item.get("modo")) or None,
        COLUNA_CRITICIDADE: (_texto(item.get("criticidade")) or "NC").upper(),
        COLUNA_COMODO: _texto(item.get("comodo")),
        # O fator de partida não entra na simulação de energia, mas entra na
        # conversa sobre surto: um compressor com fp 5 pede do inversor cinco
        # vezes a corrente nominal por alguns ciclos.
        COLUNA_FATOR_PARTIDA: _num(item.get("fp")) or 1.0,
    }
    return linha


def _tabela_do_comodo(linhas: Iterable[dict[str, Any]]) -> pd.DataFrame:
    tabela = pd.DataFrame(list(linhas))
    for coluna in COLUNAS:
        if coluna not in tabela.columns:
            tabela[coluna] = np.nan
    extras = [COLUNA_CRITICIDADE, COLUNA_COMODO, COLUNA_FATOR_PARTIDA]
    return tabela[[*COLUNAS, *[c for c in extras if c in tabela.columns]]]


# ----------------------------------------------------------------------------
# Backup em JSON — a fonte
# ----------------------------------------------------------------------------
def ler_backup(caminho: str | Path | dict[str, Any]) -> DadosVistoria:
    """
    Lê o backup em JSON da vistoria técnica.

    É o caminho preferido: traz o texto sem corrupção, a criticidade de cada
    item, a tensão da rede e a informação de já existir FV, gerador ou
    nobreak — que a planilha do simulador descarta e o operador teria que
    redigitar na tela.
    """
    if isinstance(caminho, dict):
        dados = caminho
    else:
        dados = json.loads(Path(caminho).read_text(encoding="utf-8"))

    formato = str(dados.get("formato", ""))
    avisos: list[str] = []
    if not formato.startswith("vistoria-eletrica/"):
        avisos.append(
            f"O arquivo se declara como formato {formato!r}, e não como "
            "'vistoria-eletrica/1'. A leitura seguiu, mas confira o resultado."
        )

    vistoria = dados.get("vistoria") or {}
    itens = dados.get("itens") or []
    if not itens:
        raise ValueError("o backup não tem nenhum item levantado")

    # A ordem dos ambientes vem da vistoria e é a que o vistoriador percorreu;
    # preservá-la faz o relatório contar a história na ordem em que ela foi
    # levantada, em vez de em ordem alfabética.
    ordem = {
        _texto(a.get("nome")): int(a.get("ordem") or 0)
        for a in (dados.get("ambientes") or [])
    }

    por_comodo: dict[str, list[dict[str, Any]]] = {}
    instancias: dict[str, int] = {}
    descartados: list[str] = []

    for item in itens:
        nome = _texto(item.get("equipamento")) or "sem nome"
        comodo = _texto(item.get("comodo")) or "Sem ambiente"
        if not item.get("presente", True):
            descartados.append(f"{comodo} · {nome}: marcado como ausente")
            continue
        if not item.get("simular", True):
            descartados.append(f"{comodo} · {nome}: marcado para não simular")
            continue
        linha = _linha_do_item(item)
        if linha is None:
            descartados.append(f"{comodo} · {nome}: sem potência informada nem típica")
            continue
        por_comodo.setdefault(comodo, []).append(linha)
        instancias.setdefault(comodo, 1)

    if not por_comodo:
        raise ValueError(
            "nenhum item do backup pôde ser simulado — todos ausentes, marcados "
            "para não simular, ou sem potência"
        )

    comodos = {
        nome: _tabela_do_comodo(linhas)
        for nome, linhas in sorted(por_comodo.items(), key=lambda kv: ordem.get(kv[0], 999))
    }

    cenario = Cenario(
        nome=_texto(vistoria.get("cliente")) or "instalação",
        segmento="residencia",
        comodos=comodos,
        instancias=instancias,
        # Os essenciais saem da criticidade de cada equipamento, e não de uma
        # lista de cômodos: numa cozinha a geladeira é crítica e o forno não.
        essenciais=[],
    )

    tensao = _num(vistoria.get("tensao"))
    sistema = _texto(vistoria.get("sistema"))
    tensao_linha = _tensao_de_linha(tensao, sistema)

    return DadosVistoria(
        cenario=cenario,
        cliente=_texto(vistoria.get("cliente")) or "instalação",
        imovel=_texto(vistoria.get("imovel")),
        cidade=_texto(vistoria.get("cidade")),
        uf=_texto(vistoria.get("uf")),
        data=_texto(vistoria.get("data")),
        vistoriador=_texto(vistoria.get("vistoriador")),
        tensao_rede_v=tensao_linha,
        tem_fv=bool(vistoria.get("fv")),
        tem_gerador=bool(vistoria.get("gerDiesel") or vistoria.get("gerGNV")),
        tem_nobreak=bool(vistoria.get("ups")),
        observacoes=_texto(vistoria.get("obs")),
        descartados=descartados,
        avisos=avisos,
    )


def _tensao_de_linha(tensao: float | None, sistema: str) -> float:
    """
    A tensão **de linha**, que é a que o catálogo de inversor usa.

    A vistoria grava ``tensao=127`` e ``sistema='127/220 V'``: o primeiro é a
    tensão de fase, e o segundo carrega o par. Quem dimensiona pensa em linha,
    então é o maior do par que vale — e o campo ``tensao`` sozinho levaria a
    procurar inversor de 127 V, que não existe.
    """
    if sistema:
        numeros = [float(n) for n in ("".join(
            c if (c.isdigit() or c == "/") else " " for c in sistema
        )).replace("/", " ").split() if n]
        if numeros:
            return max(numeros)
    if not tensao:
        return 380.0
    # O campo `tensao` da vistoria é a tensão de FASE. 127 de fase é a rede
    # 127/220; 220 de fase é a 220/380. Tratar 220 como se já fosse de linha
    # — o atalho óbvio — devolvia 220 para uma instalação de 380, e o filtro
    # de catálogo passava a recusar justamente os inversores certos.
    if tensao >= 300:
        return float(tensao)  # aí só pode ser de linha
    return 220.0 if tensao < 180 else 380.0


# ----------------------------------------------------------------------------
# Inventário em planilha — plano B
# ----------------------------------------------------------------------------
#: Como as colunas do inventário se chamam, sem acento e em minúsculas, para
#: sobreviver à corrupção de encoding do Excel.
_DE_PARA_INVENTARIO = {
    "comodo": COLUNA_COMODO,
    "equipamento": "Equipamento",
    "quantidade": "Quantidade",
    "criticidade": COLUNA_CRITICIDADE,
    "potencia_adotada_w": "Potência",
    "tipo_intervalo": "Tipo de intervalo",
    "intervalo": "intervalo",
    "probabilidade": "probabilidade",
    "fd": "FD",
    "duracao_min_h": "duracao_min",
    "duracao_max_h": "duracao_max",
    "modo_fixo": "modo_fixo",
    "fator_partida": COLUNA_FATOR_PARTIDA,
    "presente": "presente",
}


def _chave(coluna: str) -> str:
    """Nome de coluna sem acento, sem BOM e em minúsculas."""
    limpo = unicodedata.normalize("NFKD", str(coluna).lstrip("﻿").strip().lower())
    return "".join(c for c in limpo if not unicodedata.combining(c))


def ler_inventario(caminho: str | Path, cliente: str = "instalação") -> DadosVistoria:
    """
    Lê o inventário em planilha — o plano B, quando não há o JSON.

    Funciona, e vem com uma ressalva que o relatório precisa carregar: o Excel
    grava o arquivo em UTF-8 e a maioria das instalações o reabre como
    Latin-1, então acentos chegam corrompidos (``Ã§Ã£``). O que dá para
    consertar aqui é a comparação — nomes de coluna e de tipo são lidos sem
    acento — mas o **texto** dos nomes de equipamento e de cômodo chega como
    veio, e vai assim para o documento.
    """
    tabela = pd.read_excel(caminho)
    renomear = {c: _DE_PARA_INVENTARIO[_chave(c)] for c in tabela.columns
                if _chave(c) in _DE_PARA_INVENTARIO}
    tabela = tabela.rename(columns=renomear)

    faltando = [c for c in ("Equipamento", "Potência", COLUNA_CRITICIDADE, COLUNA_COMODO)
                if c not in tabela.columns]
    if faltando:
        raise ValueError(
            f"o inventário não tem as colunas {faltando}. Se este for o arquivo do "
            "simulador, ele não serve: descarta cômodo e criticidade, que são o "
            "que define o quadro de backup."
        )

    por_comodo: dict[str, list[dict[str, Any]]] = {}
    descartados: list[str] = []
    for _, linha in tabela.iterrows():
        nome = _texto(linha.get("Equipamento")) or "sem nome"
        comodo = _texto(linha.get(COLUNA_COMODO)) or "Sem ambiente"
        if _texto(linha.get("presente")).upper() in {"NAO", "NÃO", "FALSE", "0"}:
            descartados.append(f"{comodo} · {nome}: marcado como ausente")
            continue
        potencia = _num(linha.get("Potência"))
        if not potencia or potencia <= 0:
            descartados.append(f"{comodo} · {nome}: sem potência")
            continue
        tipo = _normalizar_tipo(_texto(linha.get("Tipo de intervalo")) or "fixo")
        por_comodo.setdefault(comodo, []).append({
            "Equipamento": nome,
            "Potência": float(potencia),
            "Quantidade": int(_num(linha.get("Quantidade")) or 1),
            "Tipo de intervalo": tipo,
            "intervalo": _texto(linha.get("intervalo")),
            "probabilidade": _num(linha.get("probabilidade")),
            "FD": _num(linha.get("FD")),
            "duracao_min": _num(linha.get("duracao_min")) if tipo == "dinâmico" else None,
            "duracao_max": _num(linha.get("duracao_max")) if tipo == "dinâmico" else None,
            "modo_fixo": _texto(linha.get("modo_fixo")) or None,
            COLUNA_CRITICIDADE: (_texto(linha.get(COLUNA_CRITICIDADE)) or "NC").upper(),
            COLUNA_COMODO: comodo,
            COLUNA_FATOR_PARTIDA: _num(linha.get(COLUNA_FATOR_PARTIDA)) or 1.0,
        })

    if not por_comodo:
        raise ValueError("nenhuma linha do inventário pôde ser simulada")

    cenario = Cenario(
        nome=cliente, segmento="residencia",
        comodos={n: _tabela_do_comodo(linhas) for n, linhas in por_comodo.items()},
        instancias={n: 1 for n in por_comodo},
        essenciais=[],
    )
    return DadosVistoria(
        cenario=cenario, cliente=cliente, descartados=descartados,
        avisos=[
            "Lido do inventário em planilha, e não do backup em JSON. A planilha "
            "não traz a tensão da rede nem a existência de FV, gerador ou nobreak, "
            "e o texto pode chegar com acentos corrompidos."
        ],
    )
