"""
Do telhado desenhado no mapa ao sistema dimensionado.

O caminho da prospecção parte de um polígono que veio do OpenStreetMap. Aqui o
polígono vem do usuário, desenhado por cima da imagem de satélite — o mesmo
destino, outra origem. O que este módulo faz é a ponte: recebe o desenho em
latitude/longitude, projeta em UTM (sem o que nenhuma medida de área ou ângulo
faz sentido), mede, sugere a orientação, empacota os módulos com
:func:`aurum.pv.layout.calcular_layout` e devolve o sistema.

**A orientação não sai do contorno sozinha, e é honesto dizer isso.** Um
retângulo desenhado no mapa dá a direção da cumeeira — o lado mais longo —, e a
água do telhado é perpendicular a ela. Mas *para qual dos dois lados* a água
cai, o contorno não conta: as duas opções são geometricamente idênticas. A
regra aqui é escolher a mais próxima do ótimo da latitude e **declarar a
escolha**, para o usuário corrigir quando souber que é a outra. Em laje plana
com estrutura inclinada o problema não existe: a estrutura aponta para onde se
quiser, e o ótimo da latitude vale direto.

Área é outra coisa que engana. A área do contorno é a **projeção horizontal**;
num telhado de 20° de inclinação a superfície real é 6% maior. Para caber
módulo coplanar o que conta é a superfície real, e é ela que
:attr:`Telhado.area_inclinada_m2` dá.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from shapely.geometry import Polygon, shape
from shapely.geometry.base import BaseGeometry

from ..geo.geometry import (
    azimuth_of_longest_side,
    largest_part,
    normalize,
    polygon_area_m2,
    project_to_utm,
    project_to_wgs84,
    to_geojson_geometry,
    utm_epsg_for,
)
from .equipment import BaseEquipamentos, Modulo
from .layout import LayoutModulos, TipoMontagem, calcular_layout, melhor_layout
from .solar import azimute_otimo, inclinacao_otima, nome_orientacao

LOGGER = logging.getLogger(__name__)

__all__ = [
    "Telhado",
    "croqui_geojson",
    "dimensionar_no_telhado",
    "telhado_de_geojson",
]

#: Inclinação típica de telhado brasileiro com telha cerâmica ou fibrocimento.
INCLINACAO_TELHADO_PADRAO = 20.0


@dataclass
class Telhado:
    """Uma água de telhado marcada no mapa, já medida e projetada."""

    nome: str
    poligono_utm: BaseGeometry
    utm_epsg: int
    latitude: float
    longitude: float
    montagem: TipoMontagem = "coplanar"
    #: Inclinação do plano do telhado (coplanar) ou da estrutura (inclinado).
    inclinacao_deg: float = INCLINACAO_TELHADO_PADRAO
    #: Azimute da face, na convenção interna do pacote: 0 = Norte.
    azimute_deg: float = 0.0
    #: Quanto da área se perde com o que a imagem não mostra: caixa d'água,
    #: claraboia, chaminé, ar-condicionado, sombra de platibanda.
    fator_obstaculos: float = 0.90
    avisos: list[str] = field(default_factory=list)

    # -- medidas -----------------------------------------------------------
    @property
    def area_m2(self) -> float:
        """Área da projeção horizontal — o que o mapa mede."""
        return float(self.poligono_utm.area)

    @property
    def area_inclinada_m2(self) -> float:
        """
        Área da superfície real do telhado.

        A projeção vista de cima encolhe pelo cosseno da inclinação. Num
        telhado de 20°, a superfície é 6,4% maior que o contorno desenhado — e
        é nela que os módulos coplanares assentam.
        """
        if self.montagem != "coplanar":
            return self.area_m2
        return self.area_m2 / max(0.2, math.cos(math.radians(self.inclinacao_deg)))

    @property
    def perimetro_m(self) -> float:
        return float(self.poligono_utm.length)

    @property
    def azimute_cumeeira_deg(self) -> float | None:
        """Direção do lado mais longo do retângulo envolvente mínimo."""
        try:
            return azimuth_of_longest_side(self.poligono_utm.minimum_rotated_rectangle)
        except Exception:  # noqa: BLE001 — geometria degenerada não deve derrubar a tela
            return None

    @property
    def orientacao(self) -> str:
        return nome_orientacao(self.azimute_deg)

    @property
    def poligono_wgs84(self) -> BaseGeometry:
        return project_to_wgs84(self.poligono_utm, self.utm_epsg)

    def as_dict(self) -> dict[str, Any]:
        return {
            "nome": self.nome,
            "area_m2": round(self.area_m2, 1),
            "area_inclinada_m2": round(self.area_inclinada_m2, 1),
            "perimetro_m": round(self.perimetro_m, 1),
            "montagem": self.montagem,
            "inclinacao_deg": round(self.inclinacao_deg, 1),
            "azimute_deg": round(self.azimute_deg, 1),
            "orientacao": self.orientacao,
            "azimute_cumeeira_deg": (
                round(self.azimute_cumeeira_deg, 1) if self.azimute_cumeeira_deg is not None else None
            ),
            "fator_obstaculos": round(self.fator_obstaculos, 2),
            "centroide": [round(self.latitude, 6), round(self.longitude, 6)],
            "geojson": to_geojson_geometry(self.poligono_wgs84),
            "avisos": list(self.avisos),
        }

    # -- orientação --------------------------------------------------------
    def azimutes_possiveis(self) -> list[float]:
        """
        As orientações que o contorno admite.

        Em laje plana, qualquer uma — devolve só o ótimo da latitude. Em
        telhado coplanar, as duas perpendiculares à cumeeira: o desenho não
        distingue para qual lado a água cai.
        """
        if self.montagem != "coplanar":
            return [azimute_otimo(self.latitude)]
        cumeeira = self.azimute_cumeeira_deg
        if cumeeira is None:
            return [azimute_otimo(self.latitude)]
        return [(cumeeira + 90.0) % 360.0, (cumeeira - 90.0) % 360.0]

    def sugerir_azimute(self) -> tuple[float, str]:
        """Devolve (azimute, explicação de por que esse)."""
        opcoes = self.azimutes_possiveis()
        otimo = azimute_otimo(self.latitude)

        def distancia(az: float) -> float:
            return min(abs(az - otimo), 360.0 - abs(az - otimo))

        escolhido = min(opcoes, key=distancia)
        if self.montagem != "coplanar":
            return escolhido, (
                f"Estrutura inclinada em laje: a orientação é livre, então vale o ótimo "
                f"da latitude — {nome_orientacao(escolhido)}."
            )
        if len(opcoes) < 2:
            return escolhido, "Contorno sem direção dominante clara; usando o ótimo da latitude."
        outra = [a for a in opcoes if a != escolhido][0]
        return escolhido, (
            f"A cumeeira corre a {self.azimute_cumeeira_deg:.0f}°, então a água aponta para "
            f"{nome_orientacao(escolhido)} ({escolhido:.0f}°) **ou** para "
            f"{nome_orientacao(outra)} ({outra:.0f}°). O desenho não distingue as duas; "
            f"escolhemos a mais próxima do Norte, que é o ótimo aqui. Se a água cai para o "
            f"outro lado, troque."
        )


# ----------------------------------------------------------------------------
def telhado_de_geojson(
    geometria: dict[str, Any],
    nome: str = "telhado",
    montagem: TipoMontagem = "coplanar",
    inclinacao_deg: float | None = None,
    fator_obstaculos: float = 0.90,
) -> Telhado:
    """
    Converte o polígono desenhado no mapa num :class:`Telhado` medido.

    Aceita tanto uma geometria GeoJSON quanto um Feature. Recusa desenho que
    não fecha uma área — uma linha ou um ponto não têm telhado dentro.
    """
    bruto = geometria.get("geometry", geometria) if isinstance(geometria, dict) else geometria
    try:
        geom = shape(bruto)
    except Exception as exc:  # noqa: BLE001 — desenho do usuário
        raise ValueError(f"não consegui ler o desenho: {exc}") from exc

    normalizado = normalize(geom)
    if normalizado is None or normalizado.is_empty:
        raise ValueError("o desenho não fecha uma área — use a ferramenta de polígono ou retângulo")
    plano = largest_part(normalizado)

    centroide = plano.centroid
    longitude, latitude = float(centroide.x), float(centroide.y)
    epsg = utm_epsg_for(longitude, latitude)
    utm, epsg = project_to_utm(plano, epsg)

    avisos: list[str] = []
    if isinstance(normalizado, Polygon) is False:
        avisos.append("O desenho tinha mais de uma parte; usamos a maior.")
    area = float(utm.area)
    if area < 10:
        avisos.append(f"Área de apenas {area:.0f} m² — confira se o desenho está na escala certa.")

    telhado = Telhado(
        nome=nome,
        poligono_utm=utm,
        utm_epsg=epsg,
        latitude=latitude,
        longitude=longitude,
        montagem=montagem,
        inclinacao_deg=(
            inclinacao_deg
            if inclinacao_deg is not None
            else (INCLINACAO_TELHADO_PADRAO if montagem == "coplanar" else inclinacao_otima(latitude))
        ),
        fator_obstaculos=fator_obstaculos,
        avisos=avisos,
    )
    telhado.azimute_deg = telhado.sugerir_azimute()[0]
    return telhado


def dimensionar_no_telhado(
    telhado: Telhado,
    base: BaseEquipamentos,
    modulo: Modulo | None = None,
) -> LayoutModulos | None:
    """
    Empacota módulos no telhado marcado e devolve o layout.

    Sem módulo escolhido, testa o catálogo inteiro e fica com o que rende mais
    kWp na mesma área — que nem sempre é o de maior potência unitária, porque
    um painel maior pode desperdiçar mais borda.
    """
    comum = dict(
        montagem=telhado.montagem,
        tilt_deg=telhado.inclinacao_deg,
        azimute_fileiras_deg=telhado.azimute_deg,
        utm_epsg=telhado.utm_epsg,
        fator_obstaculos=telhado.fator_obstaculos,
    )
    if modulo is not None:
        return calcular_layout(telhado.poligono_utm, modulo, **comum)
    if not base.modulos:
        return None
    return melhor_layout(telhado.poligono_utm, base.modulos, **comum)


def croqui_geojson(layout: LayoutModulos) -> dict[str, Any]:
    """
    Os módulos empacotados, de volta em latitude/longitude para desenhar no mapa.

    É o croqui que fecha o ciclo: o usuário desenhou o telhado, e o programa
    devolve onde cada painel cabe. Ver o desenho é o que denuncia uma escolha
    de orientação errada mais rápido que qualquer tabela.
    """
    if not layout.modulos_geom or layout.utm_epsg is None:
        return {"type": "FeatureCollection", "features": []}
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"indice": i},
                "geometry": to_geojson_geometry(project_to_wgs84(poligono, layout.utm_epsg)),
            }
            for i, poligono in enumerate(layout.modulos_geom)
        ],
    }


def area_de_coordenadas(coords_latlon: Sequence[tuple[float, float]]) -> float:
    """Área em m² de um anel de coordenadas (lat, lon), sem passar por UTM."""
    return polygon_area_m2(coords_latlon)
