"""
A imagem do pré-relatório: um PNG 1080×1350 para mandar pelo WhatsApp.

O cliente acabou de responder o questionário e recebe, de volta, uma imagem
só. Ela tem um trabalho duplo e um pouco contraditório: despertar interesse
(senão a conversa morre ali) e ser honesta (são estimativas, e um número
bonito que o estudo completo desmente queima a venda e a marca). Por isso:

* **o gancho vem primeiro** — quanto tempo o que faz falta fica ligado quando
  a luz cai. É a dor que o cliente acabou de contar, e é o cartão mais forte;
* **toda fração leva "~"** e o rodapé diz de onde saiu o consumo. Um "32%"
  seco parece medição; "~32%" é o que realmente é;
* **o que não veio, não aparece.** Se o estudo não recomendou bateria, não há
  autonomia para mostrar e o cartão diz que isso fica para a conversa; se o
  operador não liberou o investimento, a linha some. O desenho nunca completa
  um buraco com um número de enfeite.

O desenho **não calcula**: recebe um ``ResumoPreRelatorio`` pronto e só
formata. Qualquer conta nova aqui seria um segundo caminho para o mesmo
número — exatamente o que ``resumo.py`` existe para impedir.

O tamanho 1080×1350 (4:5) é o retrato que o WhatsApp mostra inteiro na
conversa, sem cortar. Toda fonte fica em 11 pt ou mais nessa figura, que é o
mínimo que se lê num celular sem dar zoom.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sem janela: isto roda em servidor, nunca numa tela

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle, Wedge  # noqa: E402

from aurum import marca  # noqa: E402
from aurum.pre_relatorio.resumo import Cobertura, ResumoPreRelatorio  # noqa: E402

__all__ = ["desenhar"]

# ----------------------------------------------------------------------------
# Tela
# ----------------------------------------------------------------------------
#: 7,2 × 9 polegadas a 150 dpi dão exatamente 1080 × 1350 px. Todo o desenho é
#: feito em pixels (o eixo de fundo vai de 0 a 1080 e de 0 a 1350, com o y
#: crescendo para baixo), porque é assim que se pensa um layout de tela.
LARGURA, ALTURA, DPI = 1080, 1350, 150
#: Quantos pixels tem um ponto tipográfico nesta figura.
PX = DPI / 72

MARGEM = 40          # da borda da imagem até o cartão
PAD = 28             # do cartão até o texto
VAO = 12             # entre um cartão e o seguinte
RAIO = 26            # canto arredondado dos cartões
ALTURA_PILULA = 72   # as pílulas de cobertura das faltas

#: O fundo é um claro quente, e não o branco do cartão: é a diferença entre
#: os dois que faz o cartão existir sem precisar de borda nem sombra.
FUNDO = "#F3F2EE"
CARTAO = marca.BRANCO
#: Texto secundário sobre o cartão preto. O CINZA da marca some no preto.
CINZA_NO_PRETO = "#B9B8B2"
TRILHO_NO_PRETO = "#3A3A36"
#: "Demais cargas" costuma ser a maior barra e não é ninguém: um cinza
#: mais claro que o da marca, para não roubar o olho do maior consumidor.
DEMAIS = "#C4C3BC"

#: A IBM Plex Sans é a fonte do site; se não estiver instalada, a DejaVu (que
#: vem com o matplotlib) desenha tudo igual, só menos elegante.
#: Espaço que não quebra linha — ver ``_Tela.quebrar``.
NBSP = "\u00a0"

FONTE = marca.registrar_fonte()


# ----------------------------------------------------------------------------
# Formatação pt-BR
# ----------------------------------------------------------------------------
def _num(v: float, casas: int = 0) -> str:
    """``1250.5`` → ``"1.250,5"``: vírgula decimal e ponto de milhar."""
    s = f"{v:,.{casas}f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


def _num_curto(v: float) -> str:
    """Inteiro quando é inteiro (``10 kWh``), uma casa quando não é (``13,5 kWh``)."""
    return _num(v, 0) if abs(v - round(v)) < 0.05 else _num(v, 1)


def _reais(v: float) -> str:
    """Sem centavos: numa estimativa, centavo é precisão fingida."""
    return f"R${NBSP}{_num(round(v), 0)}"


def _pct(f: float) -> str:
    return f"{round(f * 100):.0f}%"


def _lista(itens: list[str]) -> str:
    """``["a", "b", "c"]`` → ``"a, b e c"``."""
    itens = [i for i in itens if i]
    if not itens:
        return ""
    if len(itens) == 1:
        return itens[0]
    return ", ".join(itens[:-1]) + " e " + itens[-1]


def _maiuscula(s: str) -> str:
    """Só a primeira letra: ``capitalize()`` derrubaria a caixa de "Wi-Fi"."""
    return s[:1].upper() + s[1:]


def _duracao(h: float) -> tuple[str, str]:
    """Para o número grande e para a frase: ``(5, "horas")``, ``(40, "minutos")``."""
    if h < 1:
        minutos = max(5, int(round(h * 60 / 5.0)) * 5)
        return str(minutos), "minutos"
    n = int(round(h))
    return str(n), "hora" if n == 1 else "horas"


# Para trocar "a sua casa" pelo nome certo do lugar. Só os setores em que o
# substantivo é óbvio; o resto vira "o negócio", que nunca soa errado.
_LUGARES = {
    "escola": ("a escola", "da escola"),
    "padaria": ("a padaria", "da padaria"),
    "clínica": ("a clínica", "da clínica"),
    "clinica": ("a clínica", "da clínica"),
    "restaurante": ("o restaurante", "do restaurante"),
    "mercado": ("o mercado", "do mercado"),
    "farmácia": ("a farmácia", "da farmácia"),
    "academia": ("a academia", "da academia"),
    "loja": ("a loja", "da loja"),
    "escritório": ("o escritório", "do escritório"),
    "hotel": ("o hotel", "do hotel"),
    "pousada": ("a pousada", "da pousada"),
}


def _lugar(r: ResumoPreRelatorio) -> tuple[str, str]:
    """``("o negócio", "do negócio")`` — ou o nome do setor, quando ele é claro."""
    if not r.negocio:
        return "a sua casa", "da sua casa"
    primeira = (r.setor or "").strip().split(" ")[0].lower()
    return _LUGARES.get(primeira, ("o negócio", "do negócio"))


def _cobertura_da_falta(r: ResumoPreRelatorio) -> Cobertura | None:
    """
    A cobertura que responde pela falta que o cliente descreveu: a de menor
    duração que ainda é maior ou igual à dele. Arredondar para cima é o lado
    conservador — uma falta de 2 h é julgada pela chance de segurar 3 h.
    """
    if r.falta_declarada_h is None or not r.cobertura:
        return None
    candidatas = [c for c in r.cobertura if c.duracao_h >= r.falta_declarada_h - 1e-9]
    return min(candidatas, key=lambda c: c.duracao_h) if candidatas else None


# ----------------------------------------------------------------------------
# Texto que cabe
# ----------------------------------------------------------------------------
@dataclass
class _Bloco:
    """Um texto já quebrado em linhas e com o tamanho que coube."""

    linhas: list[str]
    tamanho: float
    peso: str = "normal"
    entrelinha: float = 1.28

    @property
    def altura(self) -> float:
        if not self.linhas:
            return 0.0
        return self.tamanho * PX * (1 + self.entrelinha * (len(self.linhas) - 1))


class _Tela:
    """
    O eixo de fundo, em pixels, com régua de texto.

    Nome de empresa comprido e quatro cargas críticas são o caso normal, não
    a exceção — então nenhum texto vai para a tela sem antes ser medido. A
    medida é a do próprio renderizador, e não uma estimativa por caracteres:
    "Mmmm" e "iiii" têm larguras muito diferentes.
    """

    def __init__(self) -> None:
        self.fig = plt.figure(figsize=(LARGURA / DPI, ALTURA / DPI), dpi=DPI)
        self.fig.patch.set_facecolor(FUNDO)
        self.ax = self.fig.add_axes((0, 0, 1, 1))
        self.ax.set_xlim(0, LARGURA)
        self.ax.set_ylim(ALTURA, 0)
        self.ax.axis("off")
        self.renderer = self.fig.canvas.get_renderer()
        # Folgas que apertam quando o conteúdo não cabe (ver ``desenhar``).
        self.pad = PAD
        self.alt_linha = 35

    # -- medir ---------------------------------------------------------------
    def largura(self, s: str, tamanho: float, peso: str = "normal") -> float:
        t = self.ax.text(0, 0, s, fontsize=tamanho, fontweight=peso, family=FONTE,
                         parse_math=False)
        w = t.get_window_extent(self.renderer).width
        t.remove()
        return w

    def quebrar(self, s: str, largura: float, tamanho: float, peso: str = "normal") -> list[str]:
        """Quebra gulosa por palavra, medindo cada linha candidata."""
        linhas: list[str] = []
        atual = ""
        # Divide só no espaço comum: o espaço inseparável (NBSP) cola "5 horas"
        # e "R$ 320", que quebrados ao meio perdem o sentido.
        for palavra in (p for p in s.split(" ") if p):
            tentativa = f"{atual} {palavra}".strip()
            if not atual or self.largura(tentativa, tamanho, peso) <= largura:
                atual = tentativa
            else:
                linhas.append(atual)
                atual = palavra
        if atual:
            linhas.append(atual)
        return linhas

    def caber(self, s: str, largura: float, tamanho: float, minimo: float,
              max_linhas: int = 1, peso: str = "normal", entrelinha: float = 1.28) -> _Bloco:
        """
        O maior tamanho, de ``tamanho`` até ``minimo``, em que ``s`` cabe em
        ``max_linhas`` linhas de ``largura`` px. Se nem no mínimo couber, corta
        a última linha com reticências — feio, mas nunca vaza do cartão.
        """
        if not s:
            return _Bloco([], tamanho, peso, entrelinha)
        t = tamanho
        while True:
            linhas = self.quebrar(s, largura, t, peso)
            larga = any(self.largura(l, t, peso) > largura for l in linhas)
            if (len(linhas) <= max_linhas and not larga) or t <= minimo:
                break
            t = max(minimo, t - 0.5)
        if len(linhas) > max_linhas or any(self.largura(l, t, peso) > largura for l in linhas):
            linhas = linhas[:max_linhas]
            ultima = linhas[-1]
            while ultima and self.largura(ultima + "…", t, peso) > largura:
                ultima = ultima[:-1].rstrip()
            linhas[-1] = ultima + "…"
        return _Bloco(linhas, t, peso, entrelinha)

    # -- escrever ------------------------------------------------------------
    def escrever(self, x: float, y: float, bloco: _Bloco, cor: str, ha: str = "left") -> float:
        """
        Escreve com o topo das maiúsculas em ``y`` e devolve a altura ocupada.

        Posiciona pela linha de base, e não por ``va="top"``: o topo do texto
        muda com acento e com letra alta, e linhas vizinhas ficariam tortas.
        """
        tam_px = bloco.tamanho * PX
        for i, linha in enumerate(bloco.linhas):
            base = y + tam_px * 0.76 + i * tam_px * bloco.entrelinha
            self.ax.text(x, base, linha, fontsize=bloco.tamanho, fontweight=bloco.peso,
                         family=FONTE, color=cor, ha=ha, va="baseline", zorder=5,
                         parse_math=False)
        return bloco.altura

    def cartao(self, x: float, y: float, w: float, h: float, cor: str = CARTAO, borda: str | None = None) -> None:
        self.ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={RAIO}",
            facecolor=cor, edgecolor=borda or "none", linewidth=1.2 if borda else 0, zorder=1))

    def pilula(self, x: float, y: float, w: float, h: float, cor: str, raio: float | None = None,
               borda: str | None = None, z: int = 2) -> None:
        r = min(h / 2, w / 2) if raio is None else raio
        self.ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
            facecolor=cor, edgecolor=borda or "none", linewidth=2.0 if borda else 0, zorder=z))

    def eixo(self, x: float, y: float, w: float, h: float):
        """Um eixo do matplotlib posicionado em pixels da tela."""
        return self.fig.add_axes((x / LARGURA, 1 - (y + h) / ALTURA, w / LARGURA, h / ALTURA))


# ----------------------------------------------------------------------------
# Cabeçalho
# ----------------------------------------------------------------------------
def _logo_no_fundo():
    """
    A logo clara recortada e tingida para o fundo da página.

    A versão "clara" tem fundo branco sólido, e um retângulo branco sobre o
    fundo quente da página fica à vista. Multiplicar pela cor do fundo leva o
    branco exatamente para ela e quase não mexe no preto nem no âmbar — o
    mesmo efeito do modo "multiplicar" de qualquer editor de imagem.
    """
    caminho = marca.logo("clara")
    if caminho is None:
        return None
    try:
        from PIL import Image
        img = np.asarray(Image.open(caminho).convert("RGB")).astype(float) / 255.0
    except Exception:  # logo ilegível não pode derrubar o envio
        return None
    escuro = img.min(axis=2) < 0.85
    linhas, colunas = np.where(escuro)
    if linhas.size:
        img = img[linhas.min():linhas.max() + 1, colunas.min():colunas.max() + 1]
    fundo = np.array([int(FUNDO[i:i + 2], 16) / 255.0 for i in (1, 3, 5)])
    return np.clip(img * fundo, 0, 1)


#: O anel do índice de oportunidade, no canto do cabeçalho.
DIAMETRO_INDICE = 128
ESPESSURA_INDICE = 11


def _anel_indice(tela: _Tela, r: ResumoPreRelatorio, y: float) -> float:
    """
    O índice de 0 a 100 num anel, no canto direito do cabeçalho. Devolve a
    largura que ocupou (0 sem índice).

    Fica no cabeçalho, e não num cartão próprio, porque o espaço da página é
    todo do gráfico — e porque é a primeira coisa que se lê: o número resume a
    página antes de ela ser lida. O arco nasce embaixo (6 h) e cresce no
    sentido horário. O eixo tem o y invertido, então o ângulo de dado é o
    espelho do visual: as 6 h ficam em 90° de dado, e "horário" é ângulo
    crescente.
    """
    if r.indice is None:
        return 0.0
    d = DIAMETRO_INDICE
    cx, cy = LARGURA - MARGEM - d / 2, y + d / 2 - 6
    tela.ax.add_patch(Wedge((cx, cy), d / 2, 0, 360, width=ESPESSURA_INDICE,
                            facecolor=marca.CINZA_CLARO, edgecolor="none", zorder=3))
    fracao = max(0.0, min(1.0, r.indice / 100.0))
    if fracao > 0:
        tela.ax.add_patch(Wedge((cx, cy), d / 2, 90, 90 + 360 * fracao, width=ESPESSURA_INDICE,
                                facecolor=marca.AMARELO, edgecolor="none", zorder=4))
    # Número e "de 100" empilhados no miolo (diâmetro interno ~106 px).
    tela.ax.text(cx, cy + 9, str(int(r.indice)), fontsize=22, fontweight="bold", family=FONTE,
                 color=marca.PRETO, ha="center", va="baseline", zorder=5, parse_math=False)
    tela.ax.text(cx, cy + 32, "de 100", fontsize=11, family=FONTE, color=marca.CINZA,
                 ha="center", va="baseline", zorder=5, parse_math=False)
    return d + 22


def _cabecalho(tela: _Tela, r: ResumoPreRelatorio, y: float) -> float:
    altura_logo = 78
    logo = _logo_no_fundo()
    if logo is not None:
        w = altura_logo * logo.shape[1] / logo.shape[0]
        tela.ax.imshow(logo, extent=(MARGEM, MARGEM + w, y + altura_logo, y),
                       aspect="auto", zorder=3, interpolation="lanczos")
    else:
        w = 220
        tela.escrever(MARGEM, y + 18, _Bloco([marca.NOME], 36, "bold"), marca.PRETO)
        tela.escrever(MARGEM, y + 72, _Bloco([marca.TAGLINE], 11), marca.GRAFITE)

    # À direita, para quem é. Negócio aparece pelo nome da empresa (é o que o
    # dono mostra para o sócio); casa, pelo nome da pessoa.
    anel = _anel_indice(tela, r, y)
    direita = LARGURA - MARGEM - anel
    largura = direita - (MARGEM + w + 40)
    if r.negocio and r.empresa:
        nome, sub = r.empresa, r.setor
    else:
        nome, sub = r.cliente, ""
    # Três linhas, para caber na altura da logo: o protocolo vai junto do
    # rótulo, que é onde o cliente procura quando liga perguntando.
    cy = y + 2
    rotulo = " · ".join(p for p in ("PRÉ-RELATÓRIO", r.protocolo) if p)
    rot = tela.caber(rotulo, largura, 11, 11, max_linhas=1, peso="bold")
    cy += tela.escrever(direita, cy, rot, marca.CINZA, ha="right") + 8
    # Nome comprido de escola ou condomínio quebra em duas linhas antes de
    # encolher demais; só no limite é cortado com reticências.
    b = tela.caber(nome, largura, 16, 12, max_linhas=2, peso="bold", entrelinha=1.2)
    cy += tela.escrever(direita, cy, b, marca.PRETO, ha="right") + 8
    linha = " · ".join(p for p in (sub, r.cidade_uf) if p)
    if linha:
        b = tela.caber(linha, largura, 12, 11, max_linhas=1)
        cy += tela.escrever(direita, cy, b, marca.GRAFITE, ha="right")
    if r.indice is not None and r.indice_faixa:
        # A leitura do anel ao lado: sem ela o número não diz do que é.
        cy += 8
        b = tela.caber(f"Oportunidade {r.indice_faixa}", largura, 13, 11, max_linhas=1, peso="bold")
        cy += tela.escrever(direita, cy, b, marca.PRETO, ha="right")
    return max(altura_logo, DIAMETRO_INDICE - 4 if anel else 0, cy - y)


# ----------------------------------------------------------------------------
# Bloco 1 — quando a luz cai
# ----------------------------------------------------------------------------
def _bloco_luz(tela: _Tela, r: ResumoPreRelatorio, y: float, desenhar: bool) -> float:
    """
    O gancho: cartão preto, número âmbar. É a única superfície escura da
    imagem, de propósito — é para onde o olho vai primeiro.
    """
    x, w = MARGEM, LARGURA - 2 * MARGEM
    dentro = w - 2 * tela.pad
    cy = y + tela.pad
    elementos: list[tuple] = []   # (x, y, bloco, cor, ha) — só desenhados no fim

    rot = _Bloco(["QUANDO A LUZ CAI"], 12, "bold")
    criticos = _lista(r.criticos)
    voce = f"{_lugar(r)[0]} tem" if r.negocio else "você tem"

    if r.autonomia_h is None:
        # Sem banco recomendado não existe autonomia — e inventar uma seria
        # pior que não mostrar. O cartão vira um convite para a conversa.
        elementos.append((x + tela.pad, cy, rot, marca.AMARELO, "left"))
        cy += rot.altura + 18
        t = tela.caber("Vamos dimensionar a bateria na conversa", dentro, 22, 16, 2, "bold")
        elementos.append((x + tela.pad, cy, t, marca.AMARELO, "left"))
        cy += t.altura + 16
        if criticos:
            verbo = "faz" if len(r.criticos) == 1 else "fazem"
            frase = (f"Você contou que {criticos} {verbo} falta quando a luz cai. "
                     "O estudo completo diz qual bateria segura isso, e por quanto tempo.")
            t = tela.caber(frase, dentro, 14, 11, 3)
            elementos.append((x + tela.pad, cy, t, marca.BRANCO, "left"))
            cy += t.altura
        cy += tela.pad
    else:
        valor, unidade = _duracao(r.autonomia_h)
        # O número grande à esquerda; o rótulo e a frase à direita, na mesma
        # faixa. Lado a lado gastam metade da altura de um empilhado, e altura
        # é o que mais falta numa imagem de celular.
        grande = _Bloco([valor], 64, "bold")
        w_num = tela.largura(valor, 64, "bold")
        und = _Bloco([unidade], 17, "bold")
        w_col = max(w_num, tela.largura(unidade, 17, "bold"))
        # Para um algarismo só, a altura que conta é a da maiúscula (~0,74 do
        # corpo), não a da linha inteira: senão sobra um vão sob o número.
        cap = 64 * PX * 0.74
        elementos.append((x + tela.pad - 4, cy, grande, marca.AMARELO, "left"))
        elementos.append((x + tela.pad, cy + cap + 14, und, marca.AMARELO, "left"))
        alt_num = cap + 14 + und.altura

        xf = x + tela.pad + w_col + 34
        singular = len(r.criticos) <= 1
        quem = _maiuscula(criticos) if criticos else ("O essencial do negócio" if r.negocio else "O essencial da casa")
        frase = (f"{quem} {'fica ligado' if singular else 'ficam ligados'} "
                 f"por cerca de {valor}{NBSP}{unidade}")
        if r.bateria_kwh:
            frase += f" com uma bateria de {_num_curto(r.bateria_kwh)}{NBSP}kWh."
        else:
            frase += "."
        t = tela.caber(frase, x + w - tela.pad - xf, 16, 12, 4, "bold", entrelinha=1.3)
        elementos.append((xf, cy, rot, marca.AMARELO, "left"))
        elementos.append((xf, cy + rot.altura + 14, t, marca.BRANCO, "left"))
        cy += max(alt_num, rot.altura + 14 + t.altura) + 24

        cob = _cobertura_da_falta(r)
        if cob is not None and r.falta_declarada:
            # "Todas" só quando é todas mesmo; abaixo disso, a porcentagem.
            quanto = "todas elas" if cob.prob >= 0.995 else f"~{_pct(cob.prob)} delas"
            frase = (f"Nas faltas de {r.falta_declarada}, que são as que {voce}, "
                     f"a bateria segura {quanto}.")
            t = tela.caber(frase, dentro, 13, 11, 3)
            elementos.append((x + tela.pad, cy, t, CINZA_NO_PRETO, "left"))
            cy += t.altura + 18

        if r.cobertura:
            cy_pilulas = cy
            cy += ALTURA_PILULA
        cy += tela.pad

    h = cy - y
    if not desenhar:
        return h

    tela.cartao(x, y, w, h, cor=marca.PRETO)
    for ex, ey, bloco, cor, ha in elementos:
        tela.escrever(ex, ey, bloco, cor, ha=ha)

    if r.autonomia_h is not None and r.cobertura:
        _pilulas(tela, r, x + tela.pad, cy_pilulas, dentro)
    return h


def _pilulas(tela: _Tela, r: ResumoPreRelatorio, x: float, y: float, w: float) -> None:
    """Uma pílula por duração de falta, com a barra da chance de segurar."""
    cobs = r.cobertura[:5]
    vao = 12
    pw = (w - vao * (len(cobs) - 1)) / len(cobs)
    ph = ALTURA_PILULA
    destaque = _cobertura_da_falta(r)
    for i, c in enumerate(cobs):
        px = x + i * (pw + vao)
        # A falta que o cliente descreveu ganha contorno âmbar: é a pílula
        # que responde a pergunta dele; as outras dão o contexto.
        marcada = destaque is not None and c.duracao_h == destaque.duracao_h
        tela.pilula(px, y, pw, ph, marca.GRAFITE, raio=14, borda=marca.AMARELO if marcada else None)
        rot = tela.caber(f"falta de {c.rotulo}", pw - 32, 11, 11, 1)
        tela.escrever(px + 16, y + 12, rot, CINZA_NO_PRETO)
        pct = _Bloco([_pct(c.prob)], 16, "bold")
        tela.escrever(px + 16, y + 40, pct, marca.AMARELO)
        # A barra ao lado do número, na mesma linha, em vez de embaixo dele.
        bx = px + 16 + tela.largura("100%", 16, "bold") + 16
        bw, bh = px + pw - 16 - bx, 8
        by = y + 40 + 16 * PX * 0.74 / 2 - bh / 2
        if bw > 20:
            tela.pilula(bx, by, bw, bh, TRILHO_NO_PRETO, z=3)
            if c.prob > 0:
                tela.pilula(bx, by, max(bh, bw * min(1.0, c.prob)), bh, marca.AMARELO, z=4)


# ----------------------------------------------------------------------------
# Bloco 2 — o dia em energia
# ----------------------------------------------------------------------------
def _tem_curva(v: list[float]) -> bool:
    return len(v) == 24 and any(abs(x) > 1e-9 for x in v)


def _tem_curva_alguma(r: ResumoPreRelatorio) -> bool:
    return _tem_curva(r.curva_consumo_kw) or _tem_curva(r.curva_solar_kw)


def _bloco_dia(tela: _Tela, r: ResumoPreRelatorio, y: float, h_grafico: float, desenhar: bool,
               com_grafico: bool = True) -> float:
    x, w = MARGEM, LARGURA - 2 * MARGEM
    dentro = w - 2 * tela.pad
    cy = y + tela.pad
    titulo = _Bloco([f"O dia {_lugar(r)[1]} em energia" if r.negocio else "O seu dia em energia"], 15, "bold")
    elementos = [(x + tela.pad, cy, titulo, marca.PRETO)]
    cy += titulo.altura + 12

    if r.fracao_sem_sol is not None:
        _, de = _lugar(r)
        quem = f"do consumo {de}" if r.negocio else "do seu consumo"
        frase = f"~{_pct(r.fracao_sem_sol)} {quem} acontece quando não tem sol."
        t = tela.caber(frase, dentro - 16, 14, 11, 2)
        elementos.append((x + tela.pad + 16, cy, t, marca.GRAFITE))
        marcador = (x + tela.pad, cy - 2, t.altura + 4)
        cy += t.altura + 10
    else:
        marcador = None

    tem_grafico = com_grafico and _tem_curva_alguma(r)
    if not tem_grafico and marcador is None:
        return 0.0   # sem curva e sem fração, o cartão seria só um título
    if tem_grafico:
        cy += 6
        gy = cy
        cy += h_grafico
    cy += tela.pad - 6
    h = cy - y
    if not desenhar:
        return h

    tela.cartao(x, y, w, h)
    for ex, ey, bloco, cor in elementos:
        tela.escrever(ex, ey, bloco, cor)
    if marcador is not None:
        # Uma barra âmbar ao lado da frase: âmbar em texto sobre claro não tem
        # contraste para ser lido, mas como marca ao lado funciona.
        mx, my, mh = marcador
        tela.pilula(mx, my, 6, mh, marca.AMARELO, raio=3)
    if tem_grafico:
        # A legenda vai na linha do título, à direita: uma linha a menos de
        # altura, e o título curto deixa espaço de sobra ao lado.
        _legenda(tela, r, x + w - tela.pad, y + tela.pad + 3)
        _grafico(tela, r, x + tela.pad - 6, gy, dentro + 6, h_grafico)
    return h


def _legenda(tela: _Tela, r: ResumoPreRelatorio, direita: float, y: float) -> None:
    itens = []
    if _tem_curva(r.curva_consumo_kw):
        itens.append(("Consumo", marca.SEMANTICA["carga"], "area"))
    if _tem_curva(r.curva_solar_kw):
        itens.append(("Sol", marca.SEMANTICA["geracao"], "area"))
    if _tem_curva(r.curva_critica_kw):
        itens.append(("O essencial", marca.SEMANTICA["backup"], "linha"))
    passo = [30 + tela.largura(nome, 11) for nome, _, _ in itens]
    cx = direita - sum(passo) - 22 * (len(itens) - 1)
    for (nome, cor, tipo), dx in zip(itens, passo):
        if tipo == "area":
            tela.pilula(cx, y + 2, 22, 14, cor, raio=4)
        else:
            for k in range(3):
                tela.ax.add_patch(Rectangle((cx + k * 8, y + 7), 5, 3, facecolor=cor, zorder=3))
        tela.escrever(cx + 30, y + 1, _Bloco([nome], 11), marca.GRAFITE)
        cx += dx + 22


def _grafico(tela: _Tela, r: ResumoPreRelatorio, x: float, y: float, w: float, h: float) -> None:
    ax = tela.eixo(x + 70, y, w - 86, h - 34)   # espaço para os rótulos dos eixos
    ax.set_facecolor(CARTAO)
    # Cada valor é a média da hora: vai no meio dela, e as pontas repetem o
    # primeiro e o último para a área encostar em 0 h e 24 h.
    xs = np.concatenate([[0.0], np.arange(24) + 0.5, [24.0]])

    def serie(v: list[float]) -> np.ndarray:
        a = np.asarray(v, dtype=float)
        return np.concatenate([[a[0]], a, [a[-1]]])

    topo = 0.0
    if _tem_curva(r.curva_solar_kw):
        sol = serie(r.curva_solar_kw)
        topo = max(topo, sol.max())
        # A noite em cinza claro: é o que dá corpo à frase "quando não tem sol".
        noite = [hh for hh in range(24) if r.curva_solar_kw[hh] <= 1e-6]
        for hh in noite:
            ax.axvspan(hh, hh + 1, color=marca.CINZA_CLARO, alpha=0.55, lw=0, zorder=0)
        ax.fill_between(xs, 0, sol, color=marca.AMARELO, alpha=0.6, lw=0, zorder=1)
        ax.plot(xs, sol, color=marca.AMARELO, lw=1.6, zorder=2)
    if _tem_curva(r.curva_consumo_kw):
        con = serie(r.curva_consumo_kw)
        topo = max(topo, con.max())
        ax.fill_between(xs, 0, con, color=marca.PRETO, alpha=0.10, lw=0, zorder=2)
        ax.plot(xs, con, color=marca.PRETO, lw=2.4, zorder=4, solid_capstyle="round")
    if _tem_curva(r.curva_critica_kw):
        cri = serie(r.curva_critica_kw)
        topo = max(topo, cri.max())
        ax.plot(xs, cri, color=marca.SEMANTICA["backup"], lw=1.4, ls=(0, (4, 3)), zorder=3)

    ax.set_xlim(0, 24)
    ax.set_ylim(0, topo * 1.12 if topo > 0 else 1)
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_xticklabels(["0h", "6h", "12h", "18h", "24h"])
    # O eixo y é discreto de propósito: o que importa é a forma do dia, e não
    # o valor exato de cada hora. Dois rótulos bastam para dar a escala.
    ax.set_yticks([0, _escala(topo)])
    ax.set_yticklabels(["0", f"{_num_curto(_escala(topo))} kW"])
    ax.tick_params(labelsize=11, colors=marca.CINZA, length=0, pad=6)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_family(FONTE)
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color(marca.CINZA_CLARO)
    ax.grid(axis="y", color=marca.CINZA_CLARO, lw=0.8)
    ax.set_axisbelow(True)


def _escala(topo: float) -> float:
    """Um valor redondo abaixo do pico, para o único rótulo do eixo y."""
    if topo <= 0:
        return 1.0
    for passo in (0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000):
        if topo / passo <= 6:
            return max(passo, np.floor(topo * 0.95 / passo) * passo)
    return float(np.floor(topo))


# ----------------------------------------------------------------------------
# Bloco 3 — para onde vai a conta
# ----------------------------------------------------------------------------
def _bloco_conta_vai(tela: _Tela, r: ResumoPreRelatorio, y: float, desenhar: bool) -> float:
    if not r.consumidores:
        return 0.0
    x, w = MARGEM, LARGURA - 2 * MARGEM
    cy = y + tela.pad
    titulo = _Bloco(["O que mais pesa na conta" if r.negocio else "Para onde vai a sua conta"], 15, "bold")
    ty = cy
    cy += titulo.altura + 16
    linhas = r.consumidores[:6]
    alt_linha = tela.alt_linha
    cy_linhas = cy
    cy += alt_linha * len(linhas) - 12 + tela.pad - 4
    h = cy - y
    if not desenhar:
        return h

    tela.cartao(x, y, w, h)
    tela.escrever(x + tela.pad, ty, titulo, marca.PRETO)
    col_nome = 290
    col_pct = 80
    bx = x + tela.pad + col_nome
    bw = w - 2 * tela.pad - col_nome - col_pct - 12
    maior = max((c.fracao for c in linhas), default=0) or 1.0
    for i, c in enumerate(linhas):
        ly = cy_linhas + i * alt_linha
        demais = c.nome.strip().lower().startswith("demais")
        nome = tela.caber(c.nome, col_nome - 16, 12, 11, 1)
        tela.escrever(x + tela.pad, ly + 2, nome, marca.CINZA if demais else marca.PRETO)
        tela.pilula(bx, ly + 1, bw, 18, marca.CINZA_CLARO, raio=9)
        # O maior consumidor em âmbar: é a primeira coisa que o solar ataca, e
        # é onde a conversa começa. "Demais cargas" fica cinza — não é ninguém.
        cor = DEMAIS if demais else (marca.AMARELO if i == 0 else marca.PRETO)
        tela.pilula(bx, ly + 1, max(18, bw * c.fracao / maior), 18, cor, raio=9, z=3)
        tela.escrever(x + w - tela.pad, ly + 2, _Bloco([f"~{_pct(c.fracao)}"], 12, "bold"),
                      marca.CINZA if demais else marca.PRETO, ha="right")
    return h


# ----------------------------------------------------------------------------
# Bloco 4 — a conta
# ----------------------------------------------------------------------------
def _bloco_conta(tela: _Tela, r: ResumoPreRelatorio, y: float, desenhar: bool) -> float:
    tem_solar = r.conta_solar_min_brl is not None and r.conta_solar_max_brl is not None
    if r.conta_atual_brl is None and not tem_solar:
        return 0.0
    x, w = MARGEM, LARGURA - 2 * MARGEM
    dentro = w - 2 * tela.pad
    cy = y + tela.pad

    colunas = []
    if r.conta_atual_brl is not None:
        colunas.append(("Conta hoje, por mês", _reais(r.conta_atual_brl), False))
    if tem_solar:
        lo, hi = sorted((r.conta_solar_min_brl, r.conta_solar_max_brl))
        faixa = _reais(lo) if round(lo) == round(hi) else f"{_reais(lo)} a {_reais(hi)}"
        colunas.append(("Com solar, por mês", faixa, True))

    # A conta de hoje é curta; a faixa com solar, não. A coluna da direita
    # fica com a maior parte da largura e a seta no meio.
    seta = 64
    if len(colunas) == 2:
        larguras = [dentro * 0.36, dentro * 0.64 - seta]
    else:
        larguras = [dentro]
    valores = [tela.caber(v, lw, 26, 17, 1, "bold") for (_, v, _), lw in zip(colunas, larguras)]
    # O mesmo tamanho nas duas colunas: tamanhos diferentes fariam a conta
    # de hoje parecer maior ou menor só pela tipografia.
    tam = min(v.tamanho for v in valores)
    valores = [_Bloco(v.linhas, tam, "bold") for v in valores]
    rot_h = _Bloco(["x"], 11).altura
    vy = cy + rot_h + 12
    cy = vy + valores[0].altura + 18

    detalhes = []
    if r.potencia_fv_kwp:
        d = f"Sistema solar de ~{_num_curto(r.potencia_fv_kwp)} kWp"
        if r.consumo_mensal_kwh:
            d += f" · consumo de ~{_num(r.consumo_mensal_kwh, 0)} kWh/mês"
        detalhes.append((tela.caber(d, dentro, 12, 11, 2), marca.GRAFITE))
    partes = []
    if r.investimento_brl is not None:
        partes.append(f"Investimento a partir de {_reais(r.investimento_brl)}")
    if r.payback_anos is not None:
        partes.append(f"retorno em ~{_num_curto(r.payback_anos)} anos")
    if partes:
        txt = " · ".join(partes)
        detalhes.append((tela.caber(_maiuscula(txt), dentro, 13, 11, 2, "bold"), marca.PRETO))
    if detalhes:
        cy += 10
    det_y = []
    for b, _ in detalhes:
        det_y.append(cy)
        cy += b.altura + 8
    cy += tela.pad - 8
    h = cy - y
    if not desenhar:
        return h

    tela.cartao(x, y, w, h)
    cx = x + tela.pad
    for i, ((rotulo, _, solar), v, lw) in enumerate(zip(colunas, valores, larguras)):
        tela.escrever(cx, y + tela.pad, _Bloco([rotulo.upper()], 11, "bold"),
                      marca.GRAFITE if solar else marca.CINZA)
        tela.escrever(cx, vy, v, marca.PRETO)
        wv = tela.largura(v.linhas[0], v.tamanho, "bold")
        if solar:
            # O sublinhado âmbar marca o número que interessa sem pintar o
            # texto de âmbar, que no claro não se lê.
            tela.pilula(cx, vy + v.altura + 4, wv, 7, marca.AMARELO, raio=3)
        if i == 0 and len(colunas) == 2:
            tela.ax.annotate("", xy=(cx + lw + seta - 18, vy + v.altura / 2),
                             xytext=(cx + lw + 6, vy + v.altura / 2),
                             arrowprops=dict(arrowstyle="-|>", color=marca.CINZA, lw=2,
                                             mutation_scale=18), zorder=5)
        cx += lw + seta
    for (b, cor), dy in zip(detalhes, det_y):
        tela.escrever(x + tela.pad, dy, b, cor)
    return h


# ----------------------------------------------------------------------------
# Rodapé
# ----------------------------------------------------------------------------
_ORIGEM = {
    "valor": "(consumo estimado pelo valor da conta)",
    "respostas": "(consumo estimado pelas respostas)",
}


def _rodape(tela: _Tela, r: ResumoPreRelatorio, desenhar: bool) -> float:
    texto = ("Estimativa feita com as suas respostas e a sua conta de luz. "
             "Os números finais saem do estudo completo.")
    if r.origem_consumo != "conta":
        texto += " " + _ORIGEM.get(r.origem_consumo, "(consumo estimado)")
    largura_data = tela.largura(r.data, 11) + 30 if r.data else 0
    b = tela.caber(texto, LARGURA - 2 * MARGEM - largura_data, 11, 11, 3)
    h = b.altura
    if desenhar:
        y = ALTURA - MARGEM + 10 - h
        tela.escrever(MARGEM, y, b, marca.CINZA)
        if r.data:
            tela.escrever(LARGURA - MARGEM, y + h - _Bloco(["x"], 11).altura, _Bloco([r.data], 11),
                          marca.CINZA, ha="right")
    return h


# ----------------------------------------------------------------------------
# Montagem
# ----------------------------------------------------------------------------
#: O gráfico é o único bloco elástico: os outros têm a altura do texto que
#: carregam, e o que sobra (ou falta) da página entra ou sai do gráfico.
GRAFICO_MAX = 400
#: (folga do cartão, altura da linha de consumidor, gráfico mínimo), por nível.
APERTOS = [(28, 35, 130), (22, 32, 110), (18, 30, 90)]


def desenhar(resumo: ResumoPreRelatorio, destino: str | Path | None = None) -> bytes:
    """Desenha o PNG 1080×1350 e devolve os bytes; se `destino`, grava também."""
    tela = _Tela()
    try:
        r = resumo
        y = MARGEM - 8
        y += _cabecalho(tela, r, y) + 14

        rodape = _rodape(tela, r, desenhar=False)
        fim = ALTURA - MARGEM + 10 - rodape - 22   # onde o último cartão pode acabar

        # Do mais folgado ao mais apertado: o primeiro nível em que tudo cabe
        # com um gráfico legível é o que vale. Se nem no aperto máximo couber,
        # o gráfico sai (a frase do "sem sol" fica) e a busca recomeça do
        # nível folgado — um gráfico de 50 px não diz nada, e texto vazando do
        # cartão diz pior. Só o caso extremo (nome enorme, seis consumidores,
        # investimento) chega aí.
        def medir(com_grafico: bool) -> tuple[float, float, float, float]:
            h1 = _bloco_luz(tela, r, 0, desenhar=False)
            h2 = _bloco_dia(tela, r, 0, 0, desenhar=False, com_grafico=com_grafico)
            h3 = _bloco_conta_vai(tela, r, 0, desenhar=False)
            h4 = _bloco_conta(tela, r, 0, desenhar=False)
            alturas = [hh for hh in (h1, h2, h3, h4) if hh > 0]
            return h2, h3, h4, fim - y - sum(alturas) - VAO * (len(alturas) - 1)

        tentativas = [(g, nivel) for g in (True, False) for nivel in APERTOS]
        if not _tem_curva_alguma(r):
            tentativas = [(False, nivel) for nivel in APERTOS]
        for com_grafico, (pad, alt_linha, g_min) in tentativas:
            tela.pad, tela.alt_linha = pad, alt_linha
            h2, h3, h4, sobra = medir(com_grafico)
            com_grafico = com_grafico and bool(h2)
            if sobra >= (g_min if com_grafico else 0):
                break
        if com_grafico:
            h_graf = float(np.clip(sobra, g_min, GRAFICO_MAX))
            sobra -= h_graf
        else:
            h_graf = 0.0
        # Com pouco conteúdo (sem curva, sem consumidores), a pilha de cartões
        # desce para o centro ótico da página — um terço da sobra acima, dois
        # abaixo — em vez de ficar grudada no topo.
        y += max(0.0, sobra) / 3

        y += _bloco_luz(tela, r, y, desenhar=True) + VAO
        if h2:
            y += _bloco_dia(tela, r, y, h_graf, desenhar=True, com_grafico=com_grafico) + VAO
        if h3:
            y += _bloco_conta_vai(tela, r, y, desenhar=True) + VAO
        if h4:
            y += _bloco_conta(tela, r, y, desenhar=True) + VAO
        _rodape(tela, r, desenhar=True)

        buf = io.BytesIO()
        tela.fig.savefig(buf, format="png", dpi=DPI, facecolor=FUNDO)
        dados = buf.getvalue()
    finally:
        plt.close(tela.fig)

    if destino is not None:
        Path(destino).write_bytes(dados)
    return dados


if __name__ == "__main__":  # pragma: no cover — atalho para olhar a imagem
    import sys

    from aurum.pre_relatorio.resumo import exemplo

    for tipo in ("casa", "negocio"):
        saida = Path(sys.argv[1] if len(sys.argv) > 1 else ".") / f"pre_relatorio_{tipo}.png"
        desenhar(exemplo(tipo), saida)
        print(saida)
