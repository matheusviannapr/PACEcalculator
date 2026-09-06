"""
Testes da identidade visual.

Marca não se testa por gosto — se testa por consistência. O que se garante
aqui é que a cor está num lugar só, que ela é a do arquivo da logo, e que
nada ficou meio rebrandeado: interface trocada e gráfico não, PDF trocado e
capa não.
"""
from __future__ import annotations

import colorsys
import re

import pytest

from aurum import marca


# ----------------------------------------------------------------------------
# As cores vêm do arquivo, não da memória
# ----------------------------------------------------------------------------
def test_as_cores_sao_as_da_logo():
    """
    O âmbar e o preto foram medidos em `assets/pace-logo.png`.

    Um `#F8C112` que virasse `#FFC107` por descuido é o tipo de erro que
    ninguém reporta e todo mundo percebe. O teste amarra o valor ao pixel.
    """
    caminho = marca.logo("clara")
    if caminho is None:
        pytest.skip("logo não está no repositório")

    from collections import Counter

    from PIL import Image

    imagem = Image.open(caminho).convert("RGB")
    contagem = Counter(imagem.getdata())

    amarelos = Counter()
    escuros = Counter()
    for (r, g, b), n in contagem.items():
        h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
        if 0.09 < h < 0.17 and s > 0.6 and v > 0.6:
            amarelos[(r, g, b)] += n
        if max(r, g, b) < 60:
            escuros[(r, g, b)] += n

    assert amarelos, "a logo precisa ter o âmbar da marca"
    dominante = amarelos.most_common(1)[0][0]
    declarado = tuple(int(marca.AMARELO.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    # Tolerância de 6 níveis por canal: a logo é JPEG-comprimida na origem.
    assert all(abs(a - b) <= 6 for a, b in zip(dominante, declarado)), (
        f"o âmbar declarado {marca.AMARELO} não bate com o da logo {dominante}"
    )


def test_toda_cor_e_hexadecimal_valida():
    todas = [marca.AMARELO, marca.PRETO, marca.GRAFITE, marca.CINZA,
             marca.CINZA_CLARO, marca.BRANCO, *marca.APOIO.values()]
    for cor in todas:
        assert re.fullmatch(r"#[0-9A-Fa-f]{6}", cor), cor


def test_o_preto_da_marca_nao_e_preto_puro():
    """É levemente quente, e isso aparece lado a lado com a logo."""
    assert marca.PRETO != "#000000"
    r, g, b = (int(marca.PRETO.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    assert r >= b, "o preto da marca puxa para o quente"


# ----------------------------------------------------------------------------
# Uma fonte só
# ----------------------------------------------------------------------------
def test_os_graficos_usam_a_paleta_da_marca():
    from aurum.bateria import relatorio

    assert relatorio._CORES is marca.CORES_GRAFICO
    assert relatorio._SEM is marca.SEMANTICA


def test_a_geracao_solar_e_ambar():
    """
    O sol é âmbar, que por sorte é a cor da marca.

    Antes ela era `_CORES[2]` — um índice, que sobrevive até alguém reordenar
    a paleta e o sol ficar roxo.
    """
    assert marca.SEMANTICA["geracao"] == marca.AMARELO
    assert marca.SEMANTICA["carga"] == marca.PRETO


def test_o_preambulo_latex_define_as_cores_da_marca():
    from aurum.proposal.sections import PREAMBULO

    for nome in ("pretopace", "amarelopace", "verdepace"):
        assert f"\\definecolor{{{nome}}}" in PREAMBULO, nome
    # O RGB precisa ser o mesmo do módulo da marca, não uma cópia envelhecida.
    for nome, hexa in marca.CORES_LATEX.items():
        if f"\\definecolor{{{nome}}}" in PREAMBULO:
            assert marca.cor_latex(nome, hexa) in PREAMBULO, nome


def test_os_nomes_antigos_ainda_resolvem():
    """
    `azulaurum` continua válido e aponta para a cor da marca.

    Há `.tex` já gerado e teste que ainda citam o nome antigo; quebrá-los não
    repinta nada e derruba a compilação de documento antigo.
    """
    from aurum.proposal.sections import PREAMBULO

    assert "\\colorlet{azulaurum}{pretopace}" in PREAMBULO
    assert "\\colorlet{verdeaurum}{verdepace}" in PREAMBULO


def test_nenhuma_cor_do_tema_antigo_sobrou():
    """O azul-marinho do Aurum não pode reaparecer em canto nenhum."""
    import pathlib

    raiz = pathlib.Path(__file__).resolve().parent.parent
    antigas = ("#1f4e79", "#0f2546", "#7ec8e3", "#061127", "#0b1f3a", "#2c5fa5")
    achados = []
    for arquivo in [*(raiz / "aurum").rglob("*.py"), raiz / "app.py",
                    raiz / ".streamlit" / "config.toml"]:
        if "_legacy" in str(arquivo):
            continue
        texto = arquivo.read_text(encoding="utf-8", errors="ignore").lower()
        achados += [f"{arquivo.name}: {c}" for c in antigas if c in texto]
    assert not achados, f"cores do tema antigo ainda no código: {achados}"


# ----------------------------------------------------------------------------
# Logo
# ----------------------------------------------------------------------------
def test_as_tres_variantes_da_logo_existem():
    for variante in ("clara", "transparente", "escura"):
        assert marca.logo(variante) is not None, variante


def test_logo_ausente_devolve_none_em_vez_de_estourar():
    """Logo faltando não pode derrubar a interface nem a geração do PDF."""
    assert marca.logo("variante-que-nao-existe") is None
