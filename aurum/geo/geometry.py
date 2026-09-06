"""
Geometria de telhados: projeção métrica, métricas de forma e utilidades.

Toda medida em metros é calculada sobre a geometria **projetada** em UTM,
nunca sobre graus. As funções aqui não dependem de geopandas nem de rtree.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

from pyproj import CRS, Transformer
from shapely.geometry import MultiPolygon, Polygon, mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform
from shapely.validation import make_valid

WGS84 = CRS.from_epsg(4326)

# Transformers do pyproj são caros de construir e seguros para reuso.
_TRANSFORMER_CACHE: dict[int, Transformer] = {}


def utm_epsg_for(lon: float, lat: float) -> int:
    """EPSG da zona UTM que contém o ponto (326xx norte, 327xx sul)."""
    zone = int((lon + 180.0) / 6.0) + 1
    zone = min(max(zone, 1), 60)
    return (32600 if lat >= 0 else 32700) + zone


def _transformer(epsg: int, inverse: bool = False) -> Transformer:
    cache_key = -epsg if inverse else epsg
    transformer = _TRANSFORMER_CACHE.get(cache_key)
    if transformer is None:
        target = CRS.from_epsg(epsg)
        transformer = (
            Transformer.from_crs(target, WGS84, always_xy=True)
            if inverse
            else Transformer.from_crs(WGS84, target, always_xy=True)
        )
        _TRANSFORMER_CACHE[cache_key] = transformer
    return transformer


def project_to_utm(geom: BaseGeometry, epsg: int | None = None) -> tuple[BaseGeometry, int]:
    """
    Projeta de WGS84 para a zona UTM local. Devolve (geometria, epsg) para que
    o chamador possa reprojetar de volta com o mesmo referencial.
    """
    if epsg is None:
        centroid = geom.centroid
        epsg = utm_epsg_for(centroid.x, centroid.y)
    return shapely_transform(_transformer(epsg).transform, geom), epsg


def project_to_wgs84(geom: BaseGeometry, epsg: int) -> BaseGeometry:
    """Inverso de project_to_utm."""
    return shapely_transform(_transformer(epsg, inverse=True).transform, geom)


def normalize(geom: BaseGeometry | None) -> Polygon | MultiPolygon | None:
    """
    Repara auto-interseções e devolve apenas conteúdo poligonal.
    Geometrias degeneradas viram None.
    """
    if geom is None or geom.is_empty:
        return None
    if not geom.is_valid:
        try:
            geom = make_valid(geom)
        except Exception:
            geom = geom.buffer(0)
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "GeometryCollection":
        polys = [g for g in geom.geoms if g.geom_type in {"Polygon", "MultiPolygon"}]
        if not polys:
            return None
        geom = max(polys, key=lambda g: g.area)
    if geom.geom_type not in {"Polygon", "MultiPolygon"}:
        return None
    return geom


def largest_part(geom: BaseGeometry) -> Polygon:
    """Maior polígono de um MultiPolygon; identidade para Polygon."""
    if geom.geom_type == "MultiPolygon":
        return max(geom.geoms, key=lambda g: g.area)
    return geom


def rectangle_sides(rect: BaseGeometry) -> tuple[float, float]:
    """
    Comprimento dos lados maior e menor de um retângulo mínimo rotacionado.

    Retorna (0, 0) para geometrias degeneradas: minimum_rotated_rectangle
    devolve LineString ou Point quando o polígono colapsa, e nesses casos
    o acesso a .exterior levantaria AttributeError.
    """
    if rect.geom_type != "Polygon" or rect.is_empty:
        return 0.0, 0.0
    coords = list(rect.exterior.coords)
    if len(coords) < 4:
        return 0.0, 0.0
    sides = [math.dist(coords[i], coords[i + 1]) for i in range(len(coords) - 1)]
    return max(sides), min(sides)


def azimuth_of_longest_side(rect: BaseGeometry) -> float | None:
    """
    Azimute (graus a partir do Norte, sentido horário) do lado mais longo do
    retângulo mínimo. Em telhado de duas águas isso é a direção da cumeeira.

    A geometria precisa estar **projetada em UTM**: o cálculo assume que +y
    aponta para o norte e +x para o leste.
    """
    if rect.geom_type != "Polygon" or rect.is_empty:
        return None
    coords = list(rect.exterior.coords)
    if len(coords) < 4:
        return None
    best_len = -1.0
    best_vec = (0.0, 0.0)
    for i in range(len(coords) - 1):
        dx = coords[i + 1][0] - coords[i][0]
        dy = coords[i + 1][1] - coords[i][1]
        length = math.hypot(dx, dy)
        if length > best_len:
            best_len, best_vec = length, (dx, dy)
    if best_len <= 0:
        return None
    # atan2(dx, dy) devolve azimute geográfico medido do Norte;
    # atan2(dy, dx) devolveria o ângulo matemático medido do Leste.
    return math.degrees(math.atan2(best_vec[0], best_vec[1])) % 360.0


@dataclass(frozen=True)
class ShapeMetrics:
    """Métricas de forma de um telhado, todas em unidades métricas."""

    area_m2: float
    perimeter_m: float
    compactness: float       # 4*pi*A/P^2 -- 1.0 e o circulo
    convexity: float         # A / A(fecho convexo)
    rectangularity: float    # A / A(retangulo minimo)
    aspect_ratio: float      # lado maior / lado menor do retangulo minimo
    vertex_count: int
    hole_area_ratio: float   # area de vazios / area bruta
    min_width_m: float       # menor lado do retangulo minimo
    max_length_m: float
    ridge_azimuth_deg: float | None

    def as_dict(self) -> dict:
        return asdict(self)


def compute_metrics(geom_utm: BaseGeometry) -> ShapeMetrics | None:
    """Calcula as métricas de forma sobre uma geometria já projetada em UTM."""
    poly = largest_part(geom_utm)
    if poly.is_empty or poly.geom_type != "Polygon":
        return None

    area = float(poly.area)
    perimeter = float(poly.length)
    if area <= 0 or perimeter <= 0:
        return None

    hull_area = float(poly.convex_hull.area)
    rect = poly.minimum_rotated_rectangle
    rect_area = float(rect.area) if rect.geom_type == "Polygon" else 0.0
    long_side, short_side = rectangle_sides(rect)

    hole_area = sum(Polygon(ring).area for ring in poly.interiors)

    return ShapeMetrics(
        area_m2=area,
        perimeter_m=perimeter,
        compactness=(4.0 * math.pi * area) / (perimeter * perimeter),
        convexity=area / hull_area if hull_area > 0 else 0.0,
        rectangularity=area / rect_area if rect_area > 0 else 0.0,
        aspect_ratio=long_side / short_side if short_side > 1e-6 else 0.0,
        vertex_count=len(poly.exterior.coords) - 1,
        hole_area_ratio=hole_area / area if area > 0 else 0.0,
        min_width_m=short_side,
        max_length_m=long_side,
        ridge_azimuth_deg=azimuth_of_longest_side(rect),
    )


def intersection_over_union(a: BaseGeometry, b: BaseGeometry) -> float:
    """IoU entre duas geometrias; 0.0 quando não se tocam."""
    if not a.intersects(b):
        return 0.0
    try:
        inter = a.intersection(b).area
        union = a.union(b).area
    except Exception:
        return 0.0
    return float(inter / union) if union > 0 else 0.0


def polygon_from_geojson(payload: dict) -> BaseGeometry | None:
    """Aceita Feature, FeatureCollection ou geometry crua."""
    try:
        kind = payload.get("type")
        if kind == "FeatureCollection":
            geoms = [shape(f["geometry"]) for f in payload.get("features", []) if f.get("geometry")]
            if not geoms:
                return None
            merged = geoms[0]
            for g in geoms[1:]:
                merged = merged.union(g)
            return normalize(merged)
        if kind == "Feature":
            return normalize(shape(payload["geometry"]))
        return normalize(shape(payload))
    except Exception:
        return None


def to_geojson_geometry(geom: BaseGeometry, precision: int = 7) -> dict:
    """Serializa arredondando coordenadas -- 7 casas equivalem a ~1 cm."""

    def _round(value):
        if isinstance(value, (list, tuple)):
            return [_round(v) for v in value]
        return round(float(value), precision)

    payload = mapping(geom)
    return {"type": payload["type"], "coordinates": _round(payload["coordinates"])}


def bbox_area_km2(bbox: Sequence[float]) -> float:
    """Área aproximada de um bbox (min_lon, min_lat, max_lon, max_lat) em km2."""
    min_lon, min_lat, max_lon, max_lat = bbox
    mid_lat = math.radians((min_lat + max_lat) / 2.0)
    width_km = (max_lon - min_lon) * 111.320 * math.cos(mid_lat)
    height_km = (max_lat - min_lat) * 110.574
    return abs(width_km * height_km)


def tile_bbox(bbox: Sequence[float], tile_size_deg: float) -> list[tuple[float, float, float, float]]:
    """
    Divide um bbox em tiles de no máximo tile_size_deg de lado.

    Percorre em graus, não em metros: o Overpass recebe bbox em graus e a
    distorção longitudinal em latitudes brasileiras é irrelevante para o
    propósito de limitar o tamanho de cada resposta.
    """
    if tile_size_deg <= 0:
        raise ValueError("tile_size_deg deve ser > 0")
    min_lon, min_lat, max_lon, max_lat = bbox
    tiles: list[tuple[float, float, float, float]] = []
    lon = min_lon
    while lon < max_lon - 1e-12:
        next_lon = min(lon + tile_size_deg, max_lon)
        lat = min_lat
        while lat < max_lat - 1e-12:
            next_lat = min(lat + tile_size_deg, max_lat)
            tiles.append((lon, lat, next_lon, next_lat))
            lat = next_lat
        lon = next_lon
    return tiles


def azimuth_between(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Azimute inicial do trajeto (lat1, lon1) -> (lat2, lon2), em graus."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def polygon_area_m2(coords_latlon: Iterable[tuple[float, float]]) -> float:
    """Área em m2 de um anel dado como sequência de (lat, lon)."""
    points = list(coords_latlon)
    if len(points) < 3:
        return 0.0
    poly = Polygon([(lon, lat) for lat, lon in points])
    projected, _ = project_to_utm(poly)
    return float(abs(projected.area))
