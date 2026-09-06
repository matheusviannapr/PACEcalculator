"""
A identidade da PACE, num lugar só.

Cor de marca espalhada pelo código é o jeito garantido de o software ficar
meio rebrandeado: a interface troca, o gráfico não; o PDF troca, a capa não.
Aqui ficam o nome, as cores e o caminho das logos, e todo o resto importa
daqui — interface, matplotlib e LaTeX.

**As cores foram medidas no arquivo da logo**, não escolhidas de memória. O
amarelo é o do triângulo dentro do "A" e da barra do "E"; o preto é a média
ponderada dos pixels escuros do letreiro. Um `#F8C112` que virasse `#FFC107`
por descuido é o tipo de erro que ninguém reporta e todo mundo percebe.

A marca é preto e âmbar, e isso tem consequência prática: com apenas duas
cores não dá para desenhar um gráfico de seis séries. As cores de apoio
existem para isso e são declaradas como apoio — nunca como marca.
"""
from __future__ import annotations

from pathlib import Path

__all__ = [
    "AMARELO",
    "APOIO",
    "CORES_GRAFICO",
    "SEMANTICA",
    "NOME",
    "PRETO",
    "TAGLINE",
    "cor_latex",
    "logo",
]

#: Como a empresa se chama, e como assina.
NOME = "PACE"
NOME_COMPLETO = "PACE Inteligência Energética"
TAGLINE = "Inteligência Energética"

# ----------------------------------------------------------------------------
# Cores da marca — medidas em assets/pace-logo.png
# ----------------------------------------------------------------------------
#: O âmbar do triângulo do "A". É a cor de destaque, e só de destaque:
#: usada em superfície grande ela cansa e some o contraste do texto.
AMARELO = "#F8C112"
#: O preto do letreiro, levemente quente — não é #000000.
PRETO = "#161615"

#: Tons derivados, para superfície e texto. Saem do preto da marca clareado,
#: e não de um cinza neutro qualquer: o calor do preto precisa continuar.
GRAFITE = "#242422"
CARVAO = "#1C1C1A"
CINZA = "#8A8A85"
CINZA_CLARO = "#E8E8E4"
BRANCO = "#FAFAF8"

#: Cores de apoio. Existem porque um gráfico de seis séries não cabe em duas
#: cores, e porque semáforo (bom/ruim) precisa de verde e vermelho que não
#: são da marca. Declaradas separadas para ninguém confundir com identidade.
APOIO = {
    "verde": "#2E7D52",
    "vermelho": "#B4451F",
    "azul": "#3B6EA5",
    "roxo": "#6B5B95",
    "petroleo": "#2A6570",
}

#: A ordem das séries num gráfico. Começa no preto da marca (a curva
#: principal), passa pelo âmbar (o destaque) e só então usa as de apoio.
CORES_GRAFICO = [
    PRETO,
    AMARELO,
    APOIO["azul"],
    APOIO["verde"],
    APOIO["vermelho"],
    APOIO["roxo"],
    APOIO["petroleo"],
]

#: Cores com significado fixo nos gráficos. Um índice de lista não diz nada:
#: ``_CORES[2]`` para a geração solar sobrevive até alguém reordenar a paleta
#: e o sol ficar roxo. Aqui o significado está no nome.
#:
#: A geração é âmbar de propósito — é a cor da marca e é a cor do sol, e as
#: duas coisas serem a mesma é sorte que vale aproveitar.
SEMANTICA = {
    "carga": PRETO,
    "geracao": AMARELO,
    "backup": APOIO["vermelho"],
    "bateria": APOIO["azul"],
    "gerador": APOIO["roxo"],
    "bom": APOIO["verde"],
    "alerta": APOIO["vermelho"],
    "neutro": CINZA,
}


# ----------------------------------------------------------------------------
# Logos
# ----------------------------------------------------------------------------
_PASTA = Path(__file__).resolve().parent.parent / "assets"

#: As três versões, e quando usar cada uma.
LOGOS = {
    #: Fundo claro — é a original, com fundo branco sólido.
    "clara": _PASTA / "pace-logo.png",
    #: Fundo de qualquer cor — o branco virou transparente.
    "transparente": _PASTA / "pace-logo-transparente.png",
    #: Fundo escuro — o letreiro preto virou quase branco.
    "escura": _PASTA / "pace-logo-escuro.png",
}


def logo(variante: str = "transparente") -> Path | None:
    """
    O caminho da logo pedida, ou ``None`` se o arquivo não estiver lá.

    Devolve ``None`` em vez de estourar porque logo faltando não pode derrubar
    a interface nem a geração do PDF: o documento sai sem a imagem e com o
    nome escrito, que é feio mas entregável.
    """
    caminho = LOGOS.get(variante)
    return caminho if (caminho and caminho.exists()) else None


# ----------------------------------------------------------------------------
# Ponte com o LaTeX
# ----------------------------------------------------------------------------
def _rgb(hexa: str) -> tuple[int, int, int]:
    h = hexa.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def cor_latex(nome: str, hexa: str) -> str:
    """Uma linha ``\\definecolor`` no formato RGB, que é o que o preâmbulo usa."""
    r, g, b = _rgb(hexa)
    return f"\\definecolor{{{nome}}}{{RGB}}{{{r},{g},{b}}}"


#: As cores que o documento LaTeX conhece. Os nomes são os que as seções
#: chamam; trocar um valor aqui repinta o dossiê inteiro.
CORES_LATEX = {
    "pretopace": PRETO,
    "amarelopace": AMARELO,
    "grafitepace": GRAFITE,
    "verdepace": APOIO["verde"],
    "vermelhoaviso": APOIO["vermelho"],
    "cinzaclaro": CINZA_CLARO,
}


def definicoes_latex() -> str:
    """O bloco de ``\\definecolor`` para o preâmbulo."""
    return "\n".join(cor_latex(nome, hexa) for nome, hexa in CORES_LATEX.items())
