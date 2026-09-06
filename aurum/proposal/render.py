"""
Escrita dos arquivos da proposta e compilação em PDF.

Gera, para cada telhado, um .tex autocontido, o JSON com todos os dados de
entrada e, quando há uma distribuição LaTeX instalada, o PDF compilado. Sem
LaTeX na máquina, o .tex continua sendo entregue e pode ser compilado no
Overleaf -- o pacote inclui instruções para isso.
"""
from __future__ import annotations

import io
import json
import logging
import re
import shutil
import subprocess
import unicodedata
import zipfile
from pathlib import Path
from typing import Iterable, Sequence

from .context import ContextoProposta
from .sections import montar_documento

LOGGER = logging.getLogger(__name__)

#: Executáveis testados, em ordem de preferência. lualatex e xelatex lidam
#: melhor com acentuação; pdflatex é o mais comum.
COMPILADORES = ("pdflatex", "lualatex", "xelatex")

#: Locais onde MiKTeX e TeX Live costumam ficar no Windows sem entrar no PATH.
#: A instalação por usuário do MiKTeX, em particular, não altera o PATH do
#: sistema, e sem esta busca o usuário receberia só o .tex mesmo tendo LaTeX.
DIRETORIOS_LATEX_WINDOWS = (
    r"%LOCALAPPDATA%\Programs\MiKTeX\miktex\bin\x64",
    r"%LOCALAPPDATA%\Programs\MiKTeX\miktex\bin",
    r"%APPDATA%\..\Local\Programs\MiKTeX\miktex\bin\x64",
    r"C:\Program Files\MiKTeX\miktex\bin\x64",
    r"C:\Program Files (x86)\MiKTeX\miktex\bin",
    r"C:\texlive\2026\bin\windows",
    r"C:\texlive\2025\bin\windows",
    r"C:\texlive\2024\bin\win32",
)


def slug(texto: str, tamanho_max: int = 60) -> str:
    """Nome de arquivo seguro a partir de texto livre, sem acento nem espaço."""
    normalizado = unicodedata.normalize("NFKD", str(texto))
    ascii_only = normalizado.encode("ascii", "ignore").decode("ascii").lower()
    limpo = re.sub(r"[^a-z0-9]+", "-", ascii_only).strip("-")
    return (limpo[:tamanho_max].rstrip("-")) or "sem-nome"


def encontrar_compilador() -> str | None:
    """
    Primeiro compilador LaTeX disponível, ou None.

    Procura no PATH e, em seguida, nos diretórios padrão de instalação do
    Windows -- o instalador do MiKTeX por usuário não mexe no PATH, e sem
    esta segunda busca uma máquina com LaTeX instalado entregaria só o .tex.
    """
    import os
    import sys

    for nome in COMPILADORES:
        caminho = shutil.which(nome)
        if caminho:
            return caminho

    if sys.platform != "win32":
        return None

    for diretorio in DIRETORIOS_LATEX_WINDOWS:
        expandido = Path(os.path.expandvars(diretorio))
        if "%" in str(expandido) or not expandido.is_dir():
            continue
        for nome in COMPILADORES:
            executavel = expandido / f"{nome}.exe"
            if executavel.is_file():
                return str(executavel)
    return None


def compilar_pdf(caminho_tex: Path, passadas: int = 2, timeout_s: int = 180) -> Path | None:
    """
    Compila o .tex em PDF.

    Duas passadas por padrão: a primeira resolve o conteúdo, a segunda acerta
    o sumário e as referências cruzadas, que na primeira ainda não existem.
    """
    compilador = encontrar_compilador()
    if compilador is None:
        LOGGER.info("Nenhuma distribuição LaTeX encontrada; o PDF não será gerado.")
        return None

    diretorio = caminho_tex.parent
    for passada in range(passadas):
        try:
            resultado = subprocess.run(
                [compilador, "-interaction=nonstopmode", "-halt-on-error", caminho_tex.name],
                cwd=diretorio,
                capture_output=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            LOGGER.warning("Compilação de %s excedeu %ss.", caminho_tex.name, timeout_s)
            return None
        except OSError as exc:
            LOGGER.warning("Falha ao executar %s: %s", compilador, exc)
            return None

        if resultado.returncode != 0 and passada == 0:
            saida = (resultado.stdout or b"").decode("utf-8", "replace")
            # As linhas de erro do LaTeX começam com "!" -- só elas interessam.
            erros = [l for l in saida.splitlines() if l.startswith("!")][:8]
            LOGGER.warning(
                "LaTeX retornou erro em %s: %s", caminho_tex.name, " | ".join(erros) or "ver .log"
            )
            return None

    pdf = caminho_tex.with_suffix(".pdf")
    return pdf if pdf.exists() else None


def limpar_auxiliares(diretorio: Path) -> None:
    """Remove os arquivos intermediários do LaTeX."""
    for extensao in (".aux", ".log", ".out", ".toc", ".lof", ".lot", ".synctex.gz"):
        for arquivo in diretorio.glob(f"*{extensao}"):
            arquivo.unlink(missing_ok=True)


def escrever_proposta(
    ctx: ContextoProposta,
    diretorio_saida: Path | str,
    compilar: bool = True,
    manter_auxiliares: bool = False,
) -> dict[str, Path | None]:
    """
    Escreve os arquivos de uma proposta e devolve os caminhos gerados.

    Cada proposta ganha sua própria pasta: o LaTeX cria vários auxiliares com
    nomes derivados do .tex, e misturá-los seria confuso num lote de dezenas.
    """
    destino = Path(diretorio_saida)
    destino.mkdir(parents=True, exist_ok=True)

    nome = f"{slug(ctx.referencia)}-{slug(ctx.titulo_cliente, 40)}"
    caminho_tex = destino / f"{nome}.tex"
    caminho_tex.write_text(montar_documento(ctx), encoding="utf-8")

    caminho_json = destino / f"{nome}.json"
    caminho_json.write_text(
        json.dumps(ctx.as_dict(), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    caminho_pdf = compilar_pdf(caminho_tex) if compilar else None
    if not manter_auxiliares:
        limpar_auxiliares(destino)

    return {"tex": caminho_tex, "json": caminho_json, "pdf": caminho_pdf}


LEIA_ME = """PROPOSTAS DE GERAÇÃO SOLAR — AURUM
===================================

Este pacote contém, para cada telhado prospectado:

  <referencia>-<cliente>.tex    proposta em LaTeX
  <referencia>-<cliente>.json   todos os dados de entrada e cálculo
  <referencia>-<cliente>.pdf    proposta compilada (quando disponível)

Também acompanham:

  indice.csv       tabela com todos os leads e seus indicadores
  indice.md        mesma tabela em formato legível
  leads.geojson    geometria dos telhados, para abrir em QGIS ou geojson.io

COMO COMPILAR O LaTeX
---------------------

Com LaTeX instalado (TeX Live ou MiKTeX), na pasta da proposta:

    pdflatex arquivo.tex
    pdflatex arquivo.tex

A segunda passada é necessária para montar o sumário.

Sem LaTeX instalado, use o Overleaf (overleaf.com): crie um projeto em branco
e envie o arquivo .tex. Os pacotes usados são todos padrão.

RESSALVA
--------

Os dados de edificação vêm do OpenStreetMap, mapeamento colaborativo cuja
cobertura e precisão variam por região. O consumo de energia é estimado por
segmento, não obtido de fatura. Toda proposta traz uma seção de limitações
que deve ser lida antes do contato comercial.
"""


def montar_zip(
    diretorio: Path | str,
    nome_arquivo: str = "propostas-aurum.zip",
) -> Path:
    """Empacota uma pasta de propostas num único ZIP para envio."""
    origem = Path(diretorio)
    destino = origem / nome_arquivo
    # O próprio ZIP não pode entrar no ZIP.
    arquivos = [
        p for p in sorted(origem.rglob("*"))
        if p.is_file() and p.resolve() != destino.resolve() and p.suffix != ".zip"
    ]
    with zipfile.ZipFile(destino, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("LEIA-ME.txt", LEIA_ME)
        for arquivo in arquivos:
            zf.write(arquivo, arquivo.relative_to(origem).as_posix())
    return destino


def zip_em_memoria(contextos: Sequence[ContextoProposta]) -> bytes:
    """
    Gera o ZIP das propostas direto na memória, sem tocar o disco.

    Usado pela interface Streamlit, onde o resultado vira um botão de download
    e não faz sentido deixar arquivos espalhados pela máquina do usuário.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("LEIA-ME.txt", LEIA_ME)
        for ctx in contextos:
            nome = f"{slug(ctx.referencia)}-{slug(ctx.titulo_cliente, 40)}"
            zf.writestr(f"{nome}/{nome}.tex", montar_documento(ctx))
            zf.writestr(
                f"{nome}/{nome}.json",
                json.dumps(ctx.as_dict(), ensure_ascii=False, indent=2, default=str),
            )
    buffer.seek(0)
    return buffer.getvalue()
