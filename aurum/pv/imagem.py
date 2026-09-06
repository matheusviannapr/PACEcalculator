"""
Foto aérea do telhado com os módulos desenhados por cima.

O croqui abstrato mostra que o arranjo fecha; a foto mostra que ele fecha
**naquele telhado**. São coisas diferentes para quem lê a proposta: o cliente
reconhece o próprio prédio, vê a caixa d'água que o desenho não tinha, e a
conversa passa a ser sobre a instalação real em vez de sobre um retângulo.

A imagem vem do serviço público World Imagery da Esri, pelo endpoint de
exportação, que devolve um recorte por *bounding box* em vez de um mosaico de
tiles — bem mais simples de compor. Sem rede, ou com o serviço fora do ar, a
função devolve ``None`` e quem chama cai para o croqui, que não depende de
nada externo.

**A imagem é uma referência visual, não um levantamento.** A data do voo não é
publicada por recorte, a resolução varia por região e o georreferenciamento
tem erro de alguns metros em área urbana densa. Serve para o cliente
reconhecer o telhado e para conferir obstáculos grandes; não serve para medir.
Quem mede é o polígono desenhado, em UTM.
"""
from __future__ import annotations

import io
import logging
import math
from dataclasses import dataclass
from typing import Any, Sequence

import requests

from ..config import Settings, get_settings

LOGGER = logging.getLogger(__name__)

__all__ = ["ImagemAerea", "baixar_imagem_aerea"]

URL_EXPORT = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export"
)
ATRIBUICAO = "Imagem: Esri World Imagery"

#: Folga em volta do telhado, como fração do maior lado. Enquadrar colado no
#: contorno tira a referência do entorno, que é justamente o que faz o cliente
#: reconhecer o prédio.
FOLGA = 0.35

#: Resolução mais fina que o serviço aceita, em metros por pixel. Pedir abaixo
#: disso não devolve uma imagem borrada: devolve **HTTP 500**, porque o pedido
#: passa do nível de detalhe máximo publicado. Medido contra o serviço: 0,25
#: m/px passa, 0,21 não. Ficamos no limite observado e, se ele variar por
#: região, a tentativa seguinte reduz o detalhe pela metade.
RESOLUCAO_MIN_M_PX = 0.25

#: Largura mínima do recorte no terreno. Um telhado de 15 m enquadrado justo
#: viraria uma imagem de 60 pixels — tecnicamente válida e visualmente inútil.
LARGURA_MINIMA_M = 90.0


@dataclass(frozen=True)
class ImagemAerea:
    """Um recorte georreferenciado: os bytes PNG e o retângulo que ele cobre."""

    conteudo: bytes
    bbox: tuple[float, float, float, float]  # (min_lon, min_lat, max_lon, max_lat)
    largura_px: int
    altura_px: int
    fonte: str = ATRIBUICAO

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """Na ordem que o ``imshow`` do matplotlib espera: (esq, dir, baixo, cima)."""
        min_lon, min_lat, max_lon, max_lat = self.bbox
        return (min_lon, max_lon, min_lat, max_lat)

    def abrir(self):
        """Devolve o array de pixels, para desenhar por cima."""
        import matplotlib.image as mpimg

        return mpimg.imread(io.BytesIO(self.conteudo), format="png")


def _bbox_com_folga(
    bounds: Sequence[float], folga: float, proporcao_alvo: float
) -> tuple[float, float, float, float]:
    """
    Expande o retângulo do telhado até a proporção da imagem pedida.

    Sem igualar a proporção, o serviço devolve a imagem esticada e os módulos
    desenhados por cima saem fora de lugar — o erro passa despercebido num
    telhado quadrado e é gritante num galpão comprido.
    """
    min_lon, min_lat, max_lon, max_lat = bounds
    centro_lon = (min_lon + max_lon) / 2.0
    centro_lat = (min_lat + max_lat) / 2.0
    # Um grau de longitude encolhe com o cosseno da latitude; ignorar isso
    # entorta a proporção em qualquer lugar fora do equador.
    fator = max(0.05, math.cos(math.radians(centro_lat)))

    largura_g = (max_lon - min_lon) * fator * (1.0 + 2 * folga)
    altura_g = (max_lat - min_lat) * (1.0 + 2 * folga)

    if largura_g / max(1e-12, altura_g) < proporcao_alvo:
        largura_g = altura_g * proporcao_alvo
    else:
        altura_g = largura_g / proporcao_alvo

    # Piso de enquadramento: telhado pequeno ganha entorno em vez de virar uma
    # imagem de poucas dezenas de pixels.
    minimo_g = LARGURA_MINIMA_M / 111_320.0
    if largura_g < minimo_g:
        escala = minimo_g / largura_g
        largura_g *= escala
        altura_g *= escala

    meia_lon = largura_g / fator / 2.0
    meia_lat = altura_g / 2.0
    return (
        centro_lon - meia_lon, centro_lat - meia_lat,
        centro_lon + meia_lon, centro_lat + meia_lat,
    )


def _largura_metros(bbox: Sequence[float]) -> float:
    min_lon, min_lat, max_lon, max_lat = bbox
    fator = max(0.05, math.cos(math.radians((min_lat + max_lat) / 2.0)))
    return (max_lon - min_lon) * fator * 111_320.0


def baixar_imagem_aerea(
    bounds: Sequence[float],
    largura_px: int = 1400,
    altura_px: int = 1000,
    folga: float = FOLGA,
    settings: Settings | None = None,
    timeout_s: float = 30.0,
) -> ImagemAerea | None:
    """
    Recorte de satélite cobrindo ``bounds`` (min_lon, min_lat, max_lon, max_lat).

    Devolve ``None`` em qualquer falha — sem rede, serviço fora, resposta que
    não é imagem. Uma foto ausente não pode derrubar a geração do relatório:
    o croqui já responde a pergunta técnica, e a foto é o complemento.
    """
    cfg = settings or get_settings()
    proporcao = largura_px / max(1, altura_px)
    bbox = _bbox_com_folga(bounds, folga, proporcao)

    # O tamanho em pixels sai da resolução, não do pedido: um recorte de 130 m
    # a 0,25 m/px cabe em 520 pixels, e pedir 1.400 só faz o serviço recusar.
    teto = int(_largura_metros(bbox) / RESOLUCAO_MIN_M_PX)
    largura = max(320, min(int(largura_px), teto))
    altura = max(240, int(round(largura / proporcao)))

    for tentativa in range(3):
        parametros = {
            "bbox": ",".join(f"{v:.8f}" for v in bbox),
            "bboxSR": 4326,
            "imageSR": 4326,
            "size": f"{largura},{altura}",
            "format": "png32",
            "transparent": "false",
            "f": "image",
        }
        try:
            resposta = requests.get(
                URL_EXPORT, params=parametros, timeout=timeout_s,
                headers={"User-Agent": cfg.user_agent},
            )
        except requests.RequestException as exc:
            LOGGER.info("Imagem aérea indisponível: %s", exc)
            return None

        if resposta.ok and resposta.headers.get("Content-Type", "").startswith("image"):
            return ImagemAerea(
                conteudo=resposta.content, bbox=bbox,
                largura_px=largura, altura_px=altura,
            )

        # O serviço recusa com HTTP 500 quando o pedido passa do nível de
        # detalhe disponível na região — e o limite varia com a cobertura
        # local. Em vez de desistir, tenta de novo com metade do detalhe.
        LOGGER.info(
            "Imagem aérea recusada (HTTP %s) em %d×%d; reduzindo o detalhe",
            resposta.status_code, largura, altura,
        )
        if tentativa == 2 or largura <= 320:
            return None
        largura = max(320, largura // 2)
        altura = max(240, int(round(largura / proporcao)))
    return None
