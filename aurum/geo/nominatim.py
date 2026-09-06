"""
Geocodificação via Nominatim: nome de região -> bbox/polígono, e
reverse geocoding de coordenadas -> endereço.

A política de uso do Nominatim exige User-Agent identificável e no máximo
1 requisição por segundo. Ambos são respeitados aqui, e o cache em disco
faz com que a mesma região consultada de novo não gere tráfego nenhum.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

import requests

from ..config import NominatimSettings, Settings, get_settings
from ..util.cache import DiskCache

LOGGER = logging.getLogger(__name__)


class GeocodingError(RuntimeError):
    """Região ou endereço não localizável."""


@dataclass(frozen=True)
class Place:
    """Resultado de geocodificação de uma região."""

    display_name: str
    lat: float
    lon: float
    #: (min_lon, min_lat, max_lon, max_lat) -- mesma ordem usada em todo o pacote.
    bbox: tuple[float, float, float, float]
    osm_type: str | None = None
    osm_id: int | None = None
    #: GeoJSON do contorno administrativo, quando disponível.
    geojson: dict[str, Any] | None = None

    @property
    def bbox_str(self) -> str:
        return ",".join(f"{v:.6f}" for v in self.bbox)


@dataclass(frozen=True)
class Address:
    """Endereço estruturado obtido por reverse geocoding."""

    label: str
    road: str | None = None
    house_number: str | None = None
    suburb: str | None = None
    city: str | None = None
    state: str | None = None
    postcode: str | None = None
    country: str | None = None

    def linha_curta(self) -> str:
        """Rua, número - bairro, cidade/UF. Omite partes ausentes."""
        rua = " ".join(p for p in (self.road, self.house_number) if p)
        partes = [p for p in (rua, self.suburb) if p]
        local = "/".join(p for p in (self.city, self.state) if p)
        if local:
            partes.append(local)
        return " - ".join(partes) if partes else self.label


class NominatimClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings: Settings = settings or get_settings()
        self.config: NominatimSettings = self.settings.nominatim
        self.cache = DiskCache(self.settings.cache_dir / "nominatim", ttl_s=self.config.cache_ttl_s)
        self._lock = threading.Lock()
        self._last_call = 0.0
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": self.settings.user_agent})

    def _throttle(self) -> None:
        with self._lock:
            elapsed = time.monotonic() - self._last_call
            remaining = self.config.min_interval_s - elapsed
            if remaining > 0:
                time.sleep(remaining)
            self._last_call = time.monotonic()

    def _get(self, path: str, params: dict[str, Any]) -> Any:
        cache_key = {"path": path, "params": params}
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        url = f"{self.config.endpoint.rstrip('/')}/{path}"
        last_error = "sem tentativa"
        for attempt in range(self.config.max_retries):
            self._throttle()
            try:
                response = self._session.get(url, params=params, timeout=self.config.timeout_s)
            except requests.RequestException as exc:
                last_error = str(exc)
                time.sleep(min(2.0 ** attempt, 15.0))
                continue
            if response.status_code in {429, 503}:
                last_error = f"HTTP {response.status_code}"
                time.sleep(min(2.0 ** attempt, 15.0) + 1.0)
                continue
            if not response.ok:
                last_error = f"HTTP {response.status_code}: {response.text[:150]}"
                break
            try:
                payload = response.json()
            except ValueError:
                last_error = "resposta não é JSON"
                break
            self.cache.set(cache_key, payload)
            return payload

        raise GeocodingError(f"Nominatim indisponível ({last_error})")

    # ------------------------------------------------------------------
    def search(self, query: str, limit: int = 5, with_geometry: bool = True) -> list[Place]:
        """Busca regiões por nome. O primeiro resultado é o mais relevante."""
        if not query or not query.strip():
            raise GeocodingError("Informe o nome da região.")
        params: dict[str, Any] = {
            "q": query.strip(),
            "format": "jsonv2",
            "limit": max(1, limit),
            "addressdetails": 1,
            "accept-language": "pt-BR",
        }
        if with_geometry:
            params["polygon_geojson"] = 1

        payload = self._get("search", params)
        if not isinstance(payload, list) or not payload:
            raise GeocodingError(f"Região não encontrada: {query}")

        places: list[Place] = []
        for item in payload:
            raw_bbox = item.get("boundingbox")
            if not raw_bbox or len(raw_bbox) != 4:
                continue
            # Nominatim devolve [min_lat, max_lat, min_lon, max_lon];
            # o pacote inteiro trabalha com (min_lon, min_lat, max_lon, max_lat).
            south, north, west, east = (float(v) for v in raw_bbox)
            places.append(
                Place(
                    display_name=str(item.get("display_name", query)),
                    lat=float(item["lat"]),
                    lon=float(item["lon"]),
                    bbox=(west, south, east, north),
                    osm_type=item.get("osm_type"),
                    osm_id=int(item["osm_id"]) if item.get("osm_id") is not None else None,
                    geojson=item.get("geojson"),
                )
            )
        if not places:
            raise GeocodingError(f"Região sem bbox utilizável: {query}")
        return places

    def geocode(self, query: str) -> Place:
        """Melhor resultado de :meth:`search`."""
        return self.search(query, limit=1)[0]

    def reverse(self, lat: float, lon: float, zoom: int = 18) -> Address | None:
        """Endereço aproximado de um ponto. Devolve None se indisponível."""
        try:
            payload = self._get(
                "reverse",
                {
                    "lat": f"{lat:.7f}",
                    "lon": f"{lon:.7f}",
                    "format": "jsonv2",
                    "zoom": zoom,
                    "addressdetails": 1,
                    "accept-language": "pt-BR",
                },
            )
        except GeocodingError as exc:
            LOGGER.info("Reverse geocoding falhou em %.5f,%.5f: %s", lat, lon, exc)
            return None

        if not isinstance(payload, dict) or "address" not in payload:
            return None
        addr = payload.get("address", {})
        return Address(
            label=str(payload.get("display_name", "")),
            road=addr.get("road") or addr.get("pedestrian"),
            house_number=addr.get("house_number"),
            suburb=addr.get("suburb") or addr.get("neighbourhood") or addr.get("city_district"),
            city=addr.get("city") or addr.get("town") or addr.get("village") or addr.get("municipality"),
            state=addr.get("state_code") or addr.get("state"),
            postcode=addr.get("postcode"),
            country=addr.get("country"),
        )


def parse_bbox(text: str) -> tuple[float, float, float, float]:
    """
    Interpreta 'min_lon,min_lat,max_lon,max_lat' e normaliza a ordem dos
    cantos, para que um bbox invertido não devolva zero resultados em silêncio.
    """
    parts = [p.strip() for p in str(text).split(",")]
    if len(parts) != 4:
        raise ValueError("bbox deve ter 4 números: min_lon,min_lat,max_lon,max_lat")
    try:
        values = [float(p) for p in parts]
    except ValueError as exc:
        raise ValueError(f"bbox com valor não numérico: {text}") from exc

    min_lon, min_lat, max_lon, max_lat = values
    min_lon, max_lon = min(min_lon, max_lon), max(min_lon, max_lon)
    min_lat, max_lat = min(min_lat, max_lat), max(min_lat, max_lat)

    if not (-180 <= min_lon <= 180 and -180 <= max_lon <= 180):
        raise ValueError("longitude fora do intervalo [-180, 180]")
    if not (-90 <= min_lat <= 90 and -90 <= max_lat <= 90):
        raise ValueError("latitude fora do intervalo [-90, 90]")
    if min_lon == max_lon or min_lat == max_lat:
        raise ValueError("bbox degenerado (área nula)")
    return (min_lon, min_lat, max_lon, max_lat)
