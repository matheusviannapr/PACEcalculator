"""
Cenário de cargas editável, com validação antes de simular.

O D² lê a planilha e simula. Entre as duas coisas falta uma que decide se o
estudo presta: **conferir o que foi digitado**. Sem isso, três erros passam sem
ruído nenhum e contaminam tudo que vem depois:

* **Janela que atravessa a meia-noite** — some do perfil (corrigido em
  :mod:`aurum.demanda.correcoes`, mas o usuário merece saber que o caso existe).
* **Tipo "dinâmico" sem duração** — o D² cai no parser legado, que espera o
  formato ``"Início entre 10:30-14:00, duração 2"``. Recebendo
  ``"08:00 as 18:00"``, ele não casa a expressão regular e devolve o padrão
  ``[(0, 60)]``: **uma hora de operação à meia-noite**, em silêncio, para um
  equipamento que devia rodar o dia inteiro. É o erro mais caro dos três,
  porque o número resultante parece plausível.
* **Duração maior que a janela** — o D² encurta a duração para caber
  (``on_overflow="clamp"``) sem avisar. O equipamento roda menos do que o
  usuário pediu, e a energia sai menor.

:meth:`Cenario.validar` levanta os três, mais os erros triviais de digitação,
e aponta cômodo, linha e campo. Nada aqui muda o comportamento do D²: só diz
antes o que ele faria depois.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .biblioteca import COLUNAS, MODELOS, linha_de_planilha
from .nucleo import Comodo, cria_comodo_da_planilha, parse_intervalo_fixo, parse_time

__all__ = ["Cenario", "Problema", "TIPOS_INTERVALO", "MODOS_FIXOS"]

TIPOS_INTERVALO = ("fixo", "dinâmico")
MODOS_FIXOS = ("FIXO_100%", "FIXO_DURACAO_INTERVALAR")

#: O formato que `parse_intervalo_dinamico_split` do D² sabe ler.
_FORMATO_LEGADO = re.compile(r"(?i)entre\s*[\d:]+\s*-\s*[\d:]+.*dura[çc]ão\s*\d")


@dataclass(frozen=True)
class Problema:
    """Um achado da validação, endereçado ao lugar exato da planilha."""

    comodo: str
    linha: int  # 1-based, como o usuário vê
    equipamento: str
    campo: str
    gravidade: str  # "erro" impede simular; "aviso" só alerta
    mensagem: str

    @property
    def impede(self) -> bool:
        return self.gravidade == "erro"

    def __str__(self) -> str:
        marca = "❌" if self.impede else "⚠️"
        return f"{marca} {self.comodo} · linha {self.linha} ({self.equipamento}) — {self.mensagem}"


def _num(valor: Any) -> float | None:
    if valor is None:
        return None
    if isinstance(valor, (int, float)):
        return None if (isinstance(valor, float) and np.isnan(valor)) else float(valor)
    texto = str(valor).strip().replace(",", ".")
    if not texto or texto.lower() in {"nan", "none"}:
        return None
    try:
        return float(texto)
    except ValueError:
        return None


def _texto(valor: Any) -> str:
    """
    Célula de texto como string, tratando toda forma de ausência.

    ``valor or ""`` parece resolver e não resolve: ``pd.NA or ""`` levanta
    ``TypeError`` porque o valor ausente do pandas recusa ser avaliado como
    booleano, e ``float('nan') or ""`` devolve o próprio NaN, que é verdadeiro.
    """
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return ""
    try:
        if pd.isna(valor):
            return ""
    except (TypeError, ValueError):
        pass
    return str(valor).strip()


def _janela_minutos(intervalo: str) -> tuple[int, int] | None:
    """(início, fim) em minutos, com o fim estendido quando vira o dia."""
    partes = parse_intervalo_fixo(str(intervalo))
    if not partes:
        return None
    inicio, fim = partes[0]
    if fim <= inicio:
        fim += 1440
    return inicio, fim


#: O que o Excel não aceita no nome de uma aba.
_PROIBIDO_NA_ABA = str.maketrans({c: "-" for c in "\\/*?:[]"})

#: E o limite de tamanho, que é do formato e não do openpyxl.
LIMITE_NOME_DE_ABA = 31


def nome_de_aba(nome: str, usados: set[str] | None = None) -> str:
    """
    O nome de cômodo convertido para um nome de aba que o Excel aceita.

    Três regras do formato, e nenhuma delas é opcional: nada de ``\\ / * ? :
    [ ]``, no máximo 31 caracteres, e sem repetir. A vistoria devolve cômodos
    como "Sala de TV / Cinema", e a barra sozinha derrubava a exportação
    inteira com um erro que não dizia qual cômodo era o culpado.

    Quando ``usados`` é passado, o nome é desambiguado com um sufixo numérico e
    registrado no conjunto. Sem isso, dois cômodos longos que só diferem depois
    do trigésimo primeiro caractere virariam a mesma aba, e o Excel recusaria a
    segunda — ou pior, dependendo da versão, a sobrescreveria em silêncio.
    """
    limpo = str(nome or "").translate(_PROIBIDO_NA_ABA).strip() or "comodo"
    limpo = limpo[:LIMITE_NOME_DE_ABA]
    if usados is None:
        return limpo

    candidato, n = limpo, 2
    while candidato.lower() in usados:
        sufixo = f" {n}"
        candidato = f"{limpo[:LIMITE_NOME_DE_ABA - len(sufixo)]}{sufixo}"
        n += 1
    usados.add(candidato.lower())
    return candidato


@dataclass
class Cenario:
    """
    O levantamento de cargas, do jeito que a interface edita.

    ``comodos`` guarda uma tabela por cômodo, nas colunas que o D² espera.
    ``instancias`` diz quantas vezes cada cômodo se repete no prédio, e
    ``essenciais`` quais deles ficam no quadro de backup — a escolha que mais
    muda o dimensionamento do inversor.
    """

    nome: str = "instalação"
    segmento: str | None = None
    comodos: dict[str, pd.DataFrame] = field(default_factory=dict)
    instancias: dict[str, int] = field(default_factory=dict)
    essenciais: list[str] = field(default_factory=list)
    #: Quando a origem é uma vistoria técnica, cada equipamento traz a sua
    #: criticidade e é ela que define o quadro de backup. A lista de cômodos
    #: essenciais continua valendo para os cenários montados a partir de um
    #: modelo de segmento, onde criticidade por equipamento não existe.
    criticidades_essenciais: tuple[str, ...] = ()
    ajustes_sazonais: dict[str, dict] = field(default_factory=dict)

    # -- construção --------------------------------------------------------
    @classmethod
    def de_segmento(cls, segmento: str, nome: str = "instalação") -> "Cenario":
        """Monta um rascunho a partir do modelo do segmento."""
        modelo = MODELOS[segmento]
        _, tabelas, instancias, essenciais = modelo.montar()
        return cls(
            nome=nome, segmento=segmento,
            comodos={n: cls._normalizar(t) for n, t in tabelas.items()},
            instancias=instancias, essenciais=list(essenciais),
        )

    @classmethod
    def de_planilha(cls, arquivo, nome: str = "instalação") -> "Cenario":
        """Lê a planilha do D²: uma aba por cômodo."""
        abas = pd.read_excel(arquivo, sheet_name=None)
        comodos: dict[str, pd.DataFrame] = {}
        instancias: dict[str, int] = {}
        for aba, tabela in abas.items():
            if aba.strip().lower() in {"instancias", "instâncias", "leia-me", "config"}:
                continue
            comodos[aba] = cls._normalizar(tabela)
            instancias[aba] = 1
        if not comodos:
            raise ValueError("a planilha não tem nenhuma aba de cômodo")
        # Aba opcional de instâncias: duas colunas, cômodo e quantidade.
        for chave in ("instancias", "instâncias"):
            if chave in {a.lower() for a in abas}:
                bruta = next(v for k, v in abas.items() if k.lower() == chave)
                for _, linha in bruta.iterrows():
                    alvo = str(linha.iloc[0]).strip()
                    if alvo in instancias:
                        instancias[alvo] = int(_num(linha.iloc[1]) or 1)
        return cls(nome=nome, comodos=comodos, instancias=instancias)

    @staticmethod
    def _normalizar(tabela: pd.DataFrame) -> pd.DataFrame:
        """Garante as colunas do D², preservando o que já existe."""
        saida = tabela.copy()
        for coluna in COLUNAS:
            if coluna not in saida.columns:
                saida[coluna] = np.nan
        saida = saida[COLUNAS]
        # Colunas numéricas precisam ser numéricas de fato: uma coluna `object`
        # com `None` dentro aparece como o texto "None" na grade de edição, e
        # o usuário não tem como distinguir isso de um valor de verdade.
        for coluna in ("Potência", "Quantidade", "probabilidade", "FD",
                       "duracao_min", "duracao_max"):
            saida[coluna] = pd.to_numeric(saida[coluna], errors="coerce")
        for coluna, padrao in (
            ("Quantidade", 1), ("probabilidade", 1.0), ("FD", 1.0),
            ("Tipo de intervalo", "fixo"),
        ):
            saida[coluna] = saida[coluna].fillna(padrao)
        # `modo_fixo` só é lido pelo D² quando o tipo é "fixo". Preencher a
        # linha dinâmica com "FIXO_100%" não muda nada no cálculo e confunde
        # quem lê a grade, então fica em branco onde não se aplica.
        e_fixo = saida["Tipo de intervalo"].astype(str).str.strip().str.lower() == "fixo"
        saida["modo_fixo"] = saida["modo_fixo"].astype("object")
        saida.loc[e_fixo, "modo_fixo"] = saida.loc[e_fixo, "modo_fixo"].fillna("FIXO_100%")
        # `pd.NA`, e não `None`: numa coluna de texto o `None` é desenhado como
        # a palavra "None" na grade de edição, enquanto `pd.NA` sai em branco.
        saida.loc[~e_fixo, "modo_fixo"] = pd.NA
        return saida.reset_index(drop=True)

    # -- edição ------------------------------------------------------------
    def tabela_vazia(self) -> pd.DataFrame:
        return pd.DataFrame(columns=COLUNAS)

    def adicionar_comodo(self, nome: str, instancias: int = 1) -> None:
        if nome in self.comodos:
            raise ValueError(f"já existe um cômodo chamado '{nome}'")
        self.comodos[nome] = self.tabela_vazia()
        self.instancias[nome] = int(instancias)

    def remover_comodo(self, nome: str) -> None:
        self.comodos.pop(nome, None)
        self.instancias.pop(nome, None)
        if nome in self.essenciais:
            self.essenciais.remove(nome)

    def adicionar_equipamento(self, comodo: str, nome_catalogo: str, quantidade: int = 1) -> None:
        linha = pd.DataFrame([linha_de_planilha(nome_catalogo, quantidade)], columns=COLUNAS)
        atual = self.comodos[comodo]
        self.comodos[comodo] = pd.concat([atual, linha], ignore_index=True) if len(atual) else linha

    # -- números -----------------------------------------------------------
    def potencia_instalada_w(self, apenas_essenciais: bool = False) -> float:
        total = 0.0
        for nome, tabela in self._selecao(apenas_essenciais).items():
            for _, linha in tabela.iterrows():
                potencia = _num(linha.get("Potência")) or 0.0
                quantidade = _num(linha.get("Quantidade")) or 0.0
                total += potencia * quantidade * self.instancias.get(nome, 1)
        return total

    def total_de_equipamentos(self, apenas_essenciais: bool = False) -> int:
        return sum(len(t) for t in self._selecao(apenas_essenciais).values())

    @property
    def tem_criticidade(self) -> bool:
        """Se os equipamentos carregam criticidade — só a vistoria traz."""
        return any(
            "criticidade" in tabela.columns and tabela["criticidade"].notna().any()
            for tabela in self.comodos.values()
        )

    def _selecao(self, apenas_essenciais: bool) -> dict[str, pd.DataFrame]:
        """
        Os cômodos que entram, e com quais equipamentos.

        Duas formas de recortar o backup, e a diferença entre elas importa:

        * **por criticidade**, quando a vistoria a levantou. Recorta *linhas*:
          numa cozinha, a geladeira entra e o forno elétrico não.
        * **por cômodo**, no cenário montado de um modelo de segmento, onde
          criticidade por equipamento não existe. Recorta *cômodos* inteiros.

        A primeira é sempre melhor quando disponível. Um quadro de backup
        montado por cômodo leva junto tudo que estava naquele ambiente, e é
        assim que se compra um inversor três vezes maior que o necessário.
        """
        if not apenas_essenciais:
            return self.comodos

        if self.criticidades_essenciais and self.tem_criticidade:
            aceitas = {c.upper() for c in self.criticidades_essenciais}
            selecao: dict[str, pd.DataFrame] = {}
            for nome, tabela in self.comodos.items():
                if "criticidade" not in tabela.columns:
                    continue
                recorte = tabela[
                    tabela["criticidade"].astype(str).str.upper().isin(aceitas)
                ]
                if not recorte.empty:
                    selecao[nome] = recorte.reset_index(drop=True)
            return selecao

        if not self.essenciais:
            return self.comodos
        return {n: t for n, t in self.comodos.items() if n in self.essenciais}

    def por_criticidade(self) -> "pd.DataFrame":
        """
        Quantos equipamentos e quanta potência há em cada nível.

        É a tabela que justifica o corte: ver que 62% da potência instalada
        está em "não crítico" é o argumento de por que o quadro de backup é
        pequeno, e ele vale mais que qualquer explicação.
        """
        linhas = []
        for nome, tabela in self.comodos.items():
            if "criticidade" not in tabela.columns:
                continue
            instancias = int(self.instancias.get(nome, 1))
            for _, linha in tabela.iterrows():
                potencia = float(linha.get("Potência") or 0) * float(
                    linha.get("Quantidade") or 1
                ) * instancias
                linhas.append({
                    "comodo": nome,
                    "criticidade": str(linha.get("criticidade") or "NC").upper(),
                    "equipamento": linha.get("Equipamento"),
                    "potencia_w": potencia,
                })
        return pd.DataFrame(linhas)

    def resumo(self) -> dict[str, Any]:
        return {
            "nome": self.nome,
            "segmento": self.segmento,
            "comodos": len(self.comodos),
            "equipamentos": self.total_de_equipamentos(),
            "potencia_instalada_kw": self.potencia_instalada_w() / 1000.0,
            "potencia_backup_kw": self.potencia_instalada_w(True) / 1000.0,
            "essenciais": list(self.essenciais),
            "instancias": dict(self.instancias),
        }

    # -- validação ---------------------------------------------------------
    def validar(self) -> list[Problema]:
        """Todos os problemas do cenário, do mais grave para o mais leve."""
        achados: list[Problema] = []
        if not self.comodos:
            return [Problema("—", 0, "—", "cenário", "erro", "nenhum cômodo cadastrado")]

        for comodo, tabela in self.comodos.items():
            if self.instancias.get(comodo, 1) < 1:
                achados.append(Problema(comodo, 0, "—", "instâncias", "erro",
                                        "o número de instâncias precisa ser 1 ou mais"))
            if tabela.empty:
                achados.append(Problema(comodo, 0, "—", "cômodo", "aviso",
                                        "cômodo sem equipamento: não contribui com carga nenhuma"))
                continue
            for posicao, (_, linha) in enumerate(tabela.iterrows(), start=1):
                achados.extend(self._validar_linha(comodo, posicao, linha))

        ordem = {"erro": 0, "aviso": 1}
        return sorted(achados, key=lambda p: (ordem[p.gravidade], p.comodo, p.linha))

    def _validar_linha(self, comodo: str, posicao: int, linha: pd.Series) -> list[Problema]:
        nome = _texto(linha.get("Equipamento"))
        achados: list[Problema] = []

        def erro(campo: str, mensagem: str) -> None:
            achados.append(Problema(comodo, posicao, nome or "sem nome", campo, "erro", mensagem))

        def aviso(campo: str, mensagem: str) -> None:
            achados.append(Problema(comodo, posicao, nome or "sem nome", campo, "aviso", mensagem))

        if not nome:
            erro("Equipamento", "sem nome")

        potencia = _num(linha.get("Potência"))
        if potencia is None or potencia <= 0:
            erro("Potência", "potência precisa ser maior que zero (em watts)")
        elif potencia > 500_000:
            aviso("Potência", f"{potencia:,.0f} W é muito alto — confira se não está em VA ou kW"
                  .replace(",", "."))

        quantidade = _num(linha.get("Quantidade"))
        if quantidade is None or quantidade < 1:
            erro("Quantidade", "quantidade precisa ser 1 ou mais")

        probabilidade = _num(linha.get("probabilidade"))
        if probabilidade is None or not 0.0 <= probabilidade <= 1.0:
            erro("probabilidade", "probabilidade precisa estar entre 0 e 1")

        fd = _num(linha.get("FD"))
        if fd is None or fd <= 0:
            erro("FD", "fator de demanda precisa ser maior que zero")
        elif fd > 1.0:
            aviso("FD", f"fator de demanda {fd:g} acima de 1 faz o equipamento consumir mais "
                        "que a potência de placa")

        tipo = _texto(linha.get("Tipo de intervalo")).lower()
        if tipo not in TIPOS_INTERVALO:
            erro("Tipo de intervalo", f"tipo '{tipo}' inválido — use {' ou '.join(TIPOS_INTERVALO)}")

        intervalo = _texto(linha.get("intervalo"))
        janela = _janela_minutos(intervalo)
        dmin, dmax = _num(linha.get("duracao_min")), _num(linha.get("duracao_max"))
        modo = _texto(linha.get("modo_fixo")).upper()

        if dmin is not None and dmax is not None and dmin > dmax:
            erro("duracao_min", f"duração mínima ({dmin:g} h) maior que a máxima ({dmax:g} h)")

        # O caso silencioso: dinâmico sem duração cai no parser legado.
        if tipo == "dinâmico" and dmin is None:
            if not _FORMATO_LEGADO.search(intervalo):
                erro(
                    "intervalo",
                    "tipo 'dinâmico' sem duração exige o formato legado "
                    "\"Início entre 10:30-14:00, duração 2\". Com o formato "
                    f"\"{intervalo}\" o D² não reconhece nada e assume **1 hora à "
                    "meia-noite**, em silêncio. Preencha duracao_min e duracao_max.",
                )
        elif janela is None:
            erro("intervalo", f"não consegui ler \"{intervalo}\" — use o formato \"08:00 as 18:00\"")
        else:
            inicio, fim = janela
            duracao_janela_h = (fim - inicio) / 60.0
            if duracao_janela_h <= 0:
                erro("intervalo", "a janela tem duração zero")
            elif duracao_janela_h > 24:
                aviso("intervalo", "a janela cobre mais de 24 h e será truncada em um dia")
            # `_janela_minutos` já somou 24 h quando o fim vinha antes do
            # início: é exatamente esse deslocamento que denuncia a virada.
            if fim > 1440:
                aviso("intervalo", "janela atravessa a meia-noite — tratada aqui como dia "
                                   "cíclico (o D² original perderia essa carga inteira)")
            if dmin is not None and dmax is not None and dmax > duracao_janela_h:
                aviso("duracao_max", f"duração de até {dmax:g} h não cabe na janela de "
                                     f"{duracao_janela_h:g} h e será encurtada para caber")

        if modo == "FIXO_DURACAO_INTERVALAR" and dmin is None:
            aviso("modo_fixo", "FIXO_DURACAO_INTERVALAR sem duração se comporta como FIXO_100%")

        return achados

    @property
    def valido(self) -> bool:
        return not any(p.impede for p in self.validar())

    # -- conversão ---------------------------------------------------------
    def para_comodos(self, apenas_essenciais: bool = False) -> list[Comodo]:
        """Converte para os objetos do núcleo do D², pronto para simular."""
        problemas = [p for p in self.validar() if p.impede]
        if problemas:
            resumo = "; ".join(str(p) for p in problemas[:5])
            raise ValueError(f"cenário com {len(problemas)} erro(s): {resumo}")
        selecao = self._selecao(apenas_essenciais)
        if not selecao:
            raise ValueError("nenhum cômodo selecionado")
        return [cria_comodo_da_planilha(tabela, nome) for nome, tabela in selecao.items()]

    def instancias_de(self, apenas_essenciais: bool = False) -> dict[str, int]:
        return {n: int(self.instancias.get(n, 1)) for n in self._selecao(apenas_essenciais)}

    def para_planilha(self, destino: str | Path | io.BytesIO | None = None) -> bytes:
        """
        Grava no formato que o D² lê — e devolve os bytes, para download.

        Os nomes das abas passam por :func:`nome_de_aba`: o Excel é mais
        restritivo que o cadastro de cômodos, e um "Sala de TV / Cinema" vindo
        da vistoria derrubava o download inteiro.
        """
        buffer = io.BytesIO()
        usados: set[str] = {"instancias"}
        with pd.ExcelWriter(buffer, engine="openpyxl") as escritor:
            for nome, tabela in self.comodos.items():
                tabela.to_excel(
                    escritor, sheet_name=nome_de_aba(nome, usados), index=False)
            pd.DataFrame(
                [{"comodo": n, "instancias": q} for n, q in self.instancias.items()]
            ).to_excel(escritor, sheet_name="instancias", index=False)
        dados = buffer.getvalue()
        if destino is not None and not isinstance(destino, io.BytesIO):
            Path(destino).write_bytes(dados)
        return dados
