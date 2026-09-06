r"""
Primitivas de LaTeX: escape correto e construção de tabelas e figuras.

Os dois bugs que este módulo existe para não repetir:

1. **Escape com barra dupla.** A versão anterior declarava as substituições
   como ``r"\\&"``, que em Python é a string de dois caracteres ``\\`` seguida
   de ``&`` -- ou seja, uma quebra de linha do LaTeX colada num E comercial.
   O correto é ``r"\&"``. Todos os dez caracteres especiais estavam assim.

2. **Escape duplo de conteúdo já formatado.** Células eram montadas como
   ``f"R\\$ {v:,.2f}"`` e depois passadas pelo escapador, que transformava a
   barra em ``\textbackslash{}``. Toda tabela de valores saía corrompida.
   A solução aqui é o tipo :class:`Raw`: o que já é LaTeX válido é marcado
   como tal e atravessa o escapador intacto.

Além disso, todos os números saem no padrão pt-BR (1.234,56).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from ..util.numfmt import br_float, br_int, br_money, br_percent

#: Substituições de caractere especial. Uma barra invertida simples em cada
#: sequência -- este é o ponto exato que estava errado antes.
_SUBSTITUICOES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}

# A barra invertida é tratada primeiro e seu substituto contém chaves, que
# seriam reescapadas numa segunda passada. Usar uma única varredura por regex
# elimina o problema: cada caractere do original é visitado exatamente uma vez.
_PADRAO = re.compile("|".join(re.escape(c) for c in _SUBSTITUICOES))

#: Caracteres que o ``pdflatex`` com ``inputenc utf8`` não sabe desenhar em
#: modo texto, e o que pôr no lugar.
#:
#: Um único ``η`` perdido numa observação de catálogo derruba a compilação
#: inteira com "Unicode character not set up for use with LaTeX" — sem PDF
#: nenhum, e a mensagem não diz de onde veio. Já aconteceu duas vezes neste
#: repositório: primeiro com um ``β`` na memória de cálculo, depois com o
#: ``ηRT`` da ressalva das baterias Lynx.
#:
#: A tradução acontece aqui, no ``esc``, e não em cada lugar que monta texto:
#: ``esc`` é a fronteira por onde todo dado externo passa, e é o único ponto
#: em que a garantia vale para o que ainda não foi escrito.
_TRANSLITERACAO = {
    # Gregas, que aparecem em coeficiente de temperatura e rendimento.
    "α": "alfa", "β": "beta", "γ": "gama", "Δ": "delta", "δ": "delta",
    "η": "eta", "θ": "teta", "λ": "lambda", "μ": "u", "π": "pi",
    "ρ": "rho", "σ": "sigma", "τ": "tau", "φ": "fi", "Φ": "Fi", "Ω": "ohm",
    # Matemáticas.
    "×": "x", "÷": "/", "≤": "<=", "≥": ">=", "≠": "!=", "≈": "~",
    "√": "raiz", "∑": "soma", "∆": "delta", "∞": "infinito", "±": "+/-",
    "−": "-", "⌊": "[", "⌋": "]", "⌈": "[", "⌉": "]",
}

_PADRAO_UNICODE = re.compile("|".join(re.escape(c) for c in _TRANSLITERACAO))


@dataclass(frozen=True)
class Raw:
    """
    Conteúdo que já é LaTeX válido e não deve ser escapado.

    Use para comandos montados no código (``\\textbf{...}``, fórmulas,
    quebras de linha). Tudo que vier de dado externo -- nome de empresa,
    endereço, tag do OSM -- deve passar pelo escape.
    """

    texto: str

    def __str__(self) -> str:
        return self.texto


def esc(valor: Any) -> str:
    """
    Escapa um valor para uso seguro em LaTeX.

    :class:`Raw` atravessa intacto; None vira travessão; números são
    formatados no padrão pt-BR.

    Letras gregas e símbolos matemáticos são transliterados antes do escape:
    o ``pdflatex`` não os desenha em modo texto, e um só deles derruba a
    compilação inteira sem dizer de onde veio.
    """
    if isinstance(valor, Raw):
        return valor.texto
    if valor is None:
        return "---"
    if isinstance(valor, bool):
        return "sim" if valor else "não"
    if isinstance(valor, float):
        valor = br_float(valor)
    # Inteiros saem sem separador de milhar de propósito: anos, identificadores
    # do OSM e números de página não levam ponto. Para contagens grandes que
    # devem levar (quantidade de módulos), use `inteiro()` explicitamente.
    texto = _PADRAO_UNICODE.sub(lambda m: _TRANSLITERACAO[m.group()], str(valor))
    return _PADRAO.sub(lambda m: _SUBSTITUICOES[m.group()], texto)


# ----------------------------------------------------------------------
# Formatadores que devolvem LaTeX pronto (já marcados como Raw)
# ----------------------------------------------------------------------
def dinheiro(valor: Any, casas: int = 2) -> Raw:
    """R$ 1.234,56 com o cifrão escapado."""
    texto = br_money(valor, casas=casas, simbolo="")
    if texto == "—":
        return Raw("---")
    return Raw(rf"R\$ {texto}")


def numero(valor: Any, casas: int = 2, unidade: str = "") -> Raw:
    texto = br_float(valor, casas=casas)
    if texto == "—":
        return Raw("---")
    return Raw(f"{texto}{(' ' + esc(unidade)) if unidade else ''}")


def inteiro(valor: Any, unidade: str = "") -> Raw:
    texto = br_int(valor)
    if texto == "—":
        return Raw("---")
    return Raw(f"{texto}{(' ' + esc(unidade)) if unidade else ''}")


def percentual(valor: Any, casas: int = 1, fracao: bool = False) -> Raw:
    texto = br_percent(valor, casas=casas, fracao=fracao)
    if texto == "—":
        return Raw("---")
    return Raw(texto.replace("%", r"\%"))


def area(valor: Any, casas: int = 0) -> Raw:
    texto = br_float(valor, casas=casas)
    if texto == "—":
        return Raw("---")
    return Raw(rf"{texto}\,m\textsuperscript{{2}}")


def potencia(valor: Any, casas: int = 2, unidade: str = "kWp") -> Raw:
    texto = br_float(valor, casas=casas)
    if texto == "—":
        return Raw("---")
    return Raw(rf"{texto}\,{esc(unidade)}")


def energia(valor: Any, casas: int = 0, unidade: str = "kWh") -> Raw:
    texto = br_float(valor, casas=casas)
    if texto == "—":
        return Raw("---")
    return Raw(rf"{texto}\,{esc(unidade)}")


def negrito(valor: Any) -> Raw:
    return Raw(rf"\textbf{{{esc(valor)}}}")


def italico(valor: Any) -> Raw:
    return Raw(rf"\textit{{{esc(valor)}}}")


def link(url: str | None, rotulo: str | None = None) -> Raw:
    """
    Hiperlink clicável. URLs não passam pelo escape normal: ``\\url`` já lida
    com os caracteres especiais, e escapá-los quebraria o endereço.
    """
    if not url:
        return Raw("---")
    url_limpa = str(url).strip()
    if rotulo:
        return Raw(rf"\href{{{url_limpa}}}{{{esc(rotulo)}}}")
    return Raw(rf"\url{{{url_limpa}}}")


# ----------------------------------------------------------------------
# Estruturas
# ----------------------------------------------------------------------
def tabela(
    cabecalho: Sequence[Any],
    linhas: Iterable[Sequence[Any]],
    alinhamento: str | None = None,
    legenda: str | None = None,
    rotulo: str | None = None,
    largura_primeira_coluna: str | None = None,
    tamanho_fonte: str | None = None,
) -> str:
    """
    Tabela com booktabs.

    ``largura_primeira_coluna`` (ex.: ``"6cm"``) transforma a primeira coluna
    em parágrafo, para que descrições longas quebrem em vez de estourar a
    margem -- problema recorrente com nomes de empresa e endereços.
    """
    linhas = [list(linha) for linha in linhas]
    n_colunas = len(cabecalho)

    if alinhamento is None:
        primeira = f"p{{{largura_primeira_coluna}}}" if largura_primeira_coluna else "l"
        alinhamento = primeira + "r" * (n_colunas - 1)

    partes = [r"\begin{table}[H]", r"\centering"]
    if tamanho_fonte:
        partes.append(f"\\{tamanho_fonte}")
    if legenda:
        partes.append(rf"\caption{{{esc(legenda)}}}")
    if rotulo:
        partes.append(rf"\label{{{esc(rotulo)}}}")
    partes.append(rf"\begin{{tabular}}{{{alinhamento}}}")
    partes.append(r"\toprule")
    partes.append(" & ".join(rf"\textbf{{{esc(c)}}}" for c in cabecalho) + r" \\")
    partes.append(r"\midrule")
    for linha in linhas:
        # Preenche células faltantes para não gerar "Extra alignment tab".
        celulas = list(linha) + [""] * (n_colunas - len(linha))
        partes.append(" & ".join(esc(c) for c in celulas[:n_colunas]) + r" \\")
    partes.append(r"\bottomrule")
    partes.append(r"\end{tabular}")
    partes.append(r"\end{table}")
    return "\n".join(partes)


def tabela_longa(
    cabecalho: Sequence[Any],
    linhas: Iterable[Sequence[Any]],
    alinhamento: str | None = None,
    legenda: str | None = None,
    rotulo: str | None = None,
    tamanho_fonte: str = "footnotesize",
) -> str:
    """
    Tabela que atravessa páginas (longtable), com cabeçalho repetido.

    Necessária para o fluxo de caixa de 25 anos, que não cabe numa página.
    """
    linhas = [list(linha) for linha in linhas]
    n_colunas = len(cabecalho)
    if alinhamento is None:
        alinhamento = "l" + "r" * (n_colunas - 1)

    linha_cabecalho = " & ".join(rf"\textbf{{{esc(c)}}}" for c in cabecalho) + r" \\"
    partes = [
        rf"{{\{tamanho_fonte}",
        rf"\begin{{longtable}}{{{alinhamento}}}",
    ]
    if legenda:
        partes.append(rf"\caption{{{esc(legenda)}}}" + (rf"\label{{{esc(rotulo)}}}" if rotulo else "") + r" \\")
    partes.extend([
        r"\toprule",
        linha_cabecalho,
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        linha_cabecalho,
        r"\midrule",
        r"\endhead",
        r"\midrule",
        rf"\multicolumn{{{n_colunas}}}{{r}}{{\footnotesize continua na próxima página}} \\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ])
    for linha in linhas:
        celulas = list(linha) + [""] * (n_colunas - len(linha))
        partes.append(" & ".join(esc(c) for c in celulas[:n_colunas]) + r" \\")
    partes.append(r"\end{longtable}")
    partes.append("}")
    return "\n".join(partes)


def caixa(titulo: str, texto: str, cor: str = "pretopace") -> str:
    """Caixa destacada para avisos e notas técnicas."""
    return (
        rf"\begin{{center}}\fcolorbox{{{cor}}}{{{cor}!5}}{{%"
        "\n"
        rf"\parbox{{0.94\linewidth}}{{\vspace{{2pt}}\textbf{{{esc(titulo)}}}\\[2pt] {esc(texto)}\vspace{{2pt}}}}}}"
        "\n"
        r"\end{center}"
    )


def aviso(texto: str) -> str:
    return caixa("Aviso técnico", texto, cor="vermelhoaviso")


def nota(texto: str) -> str:
    return caixa("Nota técnica", texto, cor="pretopace")


def lista(itens: Iterable[Any], numerada: bool = False) -> str:
    itens = list(itens)
    if not itens:
        return ""
    ambiente = "enumerate" if numerada else "itemize"
    corpo = "\n".join(rf"  \item {esc(i)}" for i in itens)
    return f"\\begin{{{ambiente}}}\n{corpo}\n\\end{{{ambiente}}}"


def secao(titulo: str, nivel: int = 1) -> str:
    comando = {1: "section", 2: "subsection", 3: "subsubsection"}.get(nivel, "paragraph")
    return f"\\{comando}{{{esc(titulo)}}}"


def grafico_barras(
    rotulos: Sequence[str],
    valores: Sequence[float],
    titulo: str,
    rotulo_y: str,
    cor: str = "pretopace",
    altura: str = "6.5cm",
) -> str:
    """
    Gráfico de barras em pgfplots.

    Os rótulos são sanitizados para conter apenas o que ``symbolic x coords``
    aceita: vírgulas e chaves dentro de um rótulo quebram o eixo.
    """
    rotulos_limpos = [re.sub(r"[^\w\-/áéíóúâêôãõçÁÉÍÓÚÂÊÔÃÕÇ ]", "", str(r)).strip() or "-" for r in rotulos]
    if not rotulos_limpos or len(rotulos_limpos) != len(valores):
        return ""
    coords = " ".join(f"({r},{v:.2f})" for r, v in zip(rotulos_limpos, valores))
    simbolicos = ",".join(rotulos_limpos)
    return (
        "\\begin{figure}[H]\n\\centering\n\\begin{tikzpicture}\n"
        "\\begin{axis}[\n"
        "  ybar,\n"
        f"  symbolic x coords={{{simbolicos}}},\n"
        "  xtick=data,\n"
        "  x tick label style={rotate=45,anchor=east,font=\\scriptsize},\n"
        "  y tick label style={font=\\scriptsize},\n"
        # Números do eixo no padrão pt-BR e sem fator "10^n" no topo, que
        # obriga o leitor a multiplicar de cabeça.
        "  scaled y ticks=false,\n"
        "  yticklabel style={/pgf/number format/.cd,fixed,precision=0,"
        "use comma,1000 sep={.}},\n"
        "  bar width=8pt,\n"
        "  enlarge x limits=0.06,\n"
        "  ymin=0,\n"
        "  width=0.95\\linewidth,\n"
        f"  height={altura},\n"
        f"  ylabel={{{rotulo_y}}},\n"
        "  ylabel style={font=\\small},\n"
        "  grid=major,\n"
        "  grid style={dotted,gray!40},\n"
        f"  every axis plot/.append style={{fill={cor}!70,draw={cor}}},\n"
        "]\n"
        f"\\addplot coordinates {{{coords}}};\n"
        "\\end{axis}\n\\end{tikzpicture}\n"
        f"\\caption{{{esc(titulo)}}}\n"
        "\\end{figure}"
    )


def grafico_linha(
    rotulos: Sequence[str],
    valores: Sequence[float],
    titulo: str,
    rotulo_y: str,
    cor: str = "verdepace",
    linha_zero: bool = True,
    altura: str = "6.5cm",
) -> str:
    """
    Gráfico de linha em pgfplots, com marcação opcional do zero.

    A linha do zero é desenhada com as coordenadas do primeiro e do último
    rótulo. Na versão anterior essa linha era escrita numa string comum que
    parecia f-string, e o texto ``{len(sym_coords)}`` ia literalmente para o
    arquivo .tex, impedindo a compilação.
    """
    rotulos_limpos = [re.sub(r"[^\w\-/ ]", "", str(r)).strip() or "-" for r in rotulos]
    if not rotulos_limpos or len(rotulos_limpos) != len(valores):
        return ""
    coords = " ".join(f"({r},{v:.2f})" for r, v in zip(rotulos_limpos, valores))
    simbolicos = ",".join(rotulos_limpos)
    zero = ""
    if linha_zero and min(valores) < 0 < max(valores):
        zero = (
            f"\\addplot[dashed,red!70,thick,mark=none] coordinates "
            f"{{({rotulos_limpos[0]},0) ({rotulos_limpos[-1]},0)}};\n"
        )
    return (
        "\\begin{figure}[H]\n\\centering\n\\begin{tikzpicture}\n"
        "\\begin{axis}[\n"
        f"  symbolic x coords={{{simbolicos}}},\n"
        "  xtick=data,\n"
        "  x tick label style={rotate=45,anchor=east,font=\\scriptsize},\n"
        "  y tick label style={font=\\scriptsize},\n"
        # Números do eixo no padrão pt-BR e sem fator "10^n" no topo, que
        # obriga o leitor a multiplicar de cabeça.
        "  scaled y ticks=false,\n"
        "  yticklabel style={/pgf/number format/.cd,fixed,precision=0,"
        "use comma,1000 sep={.}},\n"
        "  width=0.95\\linewidth,\n"
        f"  height={altura},\n"
        f"  ylabel={{{rotulo_y}}},\n"
        "  ylabel style={font=\\small},\n"
        "  grid=major,\n"
        "  grid style={dotted,gray!40},\n"
        "]\n"
        f"\\addplot[color={cor},thick,mark=*,mark size=1.2pt] coordinates {{{coords}}};\n"
        f"{zero}"
        "\\end{axis}\n\\end{tikzpicture}\n"
        f"\\caption{{{esc(titulo)}}}\n"
        "\\end{figure}"
    )
