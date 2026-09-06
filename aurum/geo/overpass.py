"""
Cliente Overpass com cache em disco, rate limiting global e failover entre
espelhos.

Substitui as três implementações divergentes que existiam antes (main.py,
src/overpass.py e aurum_lead_mapper_app.py). Correções relevantes em relação
a elas:

* o corpo da requisição vai como formulário ``data={"data": query}``, que é o
  formato que a API documenta -- antes o texto cru era enviado com
  Content-Type de formulário, o que funciona por acidente;
* HTTP 429/504 fazem backoff **e** avançam para o próximo espelho, em vez de
  repetir indefinidamente contra um servidor sobrecarregado;
* respostas com "remark" de timeout/memória do Overpass são detectadas, já
  que vêm com HTTP 200 e um payload parcial silencioso.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import requests

from ..config import OverpassSettings, Settings, get_settings
from ..util.cache import DiskCache

LOGGER = logging.getLogger(__name__)


class OverpassError(RuntimeError):
    """Falha definitiva ao consultar o Overpass."""


@dataclass
class _RateLimiter:
    """Serializa requisições de todas as threads a um intervalo mínimo."""

    min_interval_s: float
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _last: float = 0.0

    def wait(self) -> None:
        with self._lock:
            elapsed = time.monotonic() - self._last
            remaining = self.min_interval_s - elapsed
            if remaining > 0:
                time.sleep(remaining)
            self._last = time.monotonic()


#: Tags que identificam telhado já ocupado por geração solar. Prospectar esses
#: prédios seria desperdício de tempo comercial.
SOLAR_TAGS = {
    "generator:source": {"solar"},
    "generator:method": {"photovoltaic"},
    "solar_panel": {"yes"},
    "solar_panels": {"yes"},
    "rooftop:solar": None,   # None = a simples presença da chave já basta
    "roof:solar": None,
}


#: Um carimbo de base sadio é ISO-8601, como "2026-08-24T01:29:50Z".
_PADRAO_TIMESTAMP_OSM = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z?$")


def _instancia_sincronizada(payload: dict[str, Any]) -> bool:
    """
    Verifica se a instância que respondeu tem a base em dia.

    Um espelho com o banco quebrado responde HTTP 200, com envelope bem
    formado e nenhum elemento -- resultado indistinguível de uma área
    realmente vazia. O carimbo ``osm3s.timestamp_osm_base`` denuncia:
    numa instância sadia é ISO-8601; na degradada observada vinha como
    um inteiro solto ("116619").

    Respostas **com** elementos são aceitas mesmo sem carimbo reconhecível:
    dado é dado, e nem toda instância preenche o campo do mesmo jeito.
    """
    if payload.get("elements"):
        return True
    carimbo = str(payload.get("osm3s", {}).get("timestamp_osm_base", ""))
    return bool(_PADRAO_TIMESTAMP_OSM.match(carimbo))


class OverpassClient:
    """Consulta edificações e POIs, com cache persistente por consulta."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings: Settings = settings or get_settings()
        self.config: OverpassSettings = self.settings.overpass
        self.cache = DiskCache(self.settings.cache_dir / "overpass", ttl_s=self.config.cache_ttl_s)
        self._limiter = _RateLimiter(self.config.min_interval_s)
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": self.settings.user_agent})
        #: Momento até o qual cada espelho fica de castigo, por índice. Sem
        #: isso, uma varredura de dezenas de tiles pagava o timeout de conexão
        #: de um espelho fora do ar uma vez por tile -- e o primeiro da lista
        #: pode estar inacessível a partir de uma rede específica.
        self._castigo: dict[int, float] = {}
        self._lock_endpoint = threading.Lock()

    # ------------------------------------------------------------------
    # Saúde dos espelhos
    # ------------------------------------------------------------------
    def _escolher_endpoint(self, tentativa: int, total: int) -> int:
        """
        Escolhe o espelho da vez, pulando os que falharam há pouco.

        Na última tentativa o castigo é ignorado: é melhor insistir num
        espelho suspeito do que desistir sem tentar nenhum.
        """
        agora = time.monotonic()
        with self._lock_endpoint:
            disponiveis = [i for i in range(total) if self._castigo.get(i, 0.0) <= agora]
        if not disponiveis or tentativa >= self.config.max_retries - 1:
            disponiveis = list(range(total))
        return disponiveis[tentativa % len(disponiveis)]

    def _punir_endpoint(self, indice: int, segundos: float = 120.0) -> None:
        with self._lock_endpoint:
            self._castigo[indice] = time.monotonic() + segundos

    def _absolver_endpoint(self, indice: int) -> None:
        with self._lock_endpoint:
            self._castigo.pop(indice, None)

    # ------------------------------------------------------------------
    # Transporte
    # ------------------------------------------------------------------
    def run_query(self, query: str, use_cache: bool = True) -> dict[str, Any]:
        """Executa uma consulta Overpass QL e devolve o JSON decodificado."""
        if use_cache:
            cached = self.cache.get(query)
            if cached is not None:
                return cached

        endpoints = list(self.config.endpoints)
        last_error: str = "nenhuma tentativa realizada"

        for attempt in range(self.config.max_retries):
            indice = self._escolher_endpoint(attempt, len(endpoints))
            endpoint = endpoints[indice]
            self._limiter.wait()
            try:
                response = self._session.post(
                    endpoint,
                    data={"data": query},
                    # (conectar, ler): falha rápido em host fora do ar, espera
                    # o quanto for preciso por uma consulta que está rodando.
                    timeout=(self.config.connect_timeout_s, self.config.timeout_s),
                )
            except requests.RequestException as exc:
                last_error = f"{endpoint}: {exc}"
                LOGGER.warning("Overpass indisponível (%s). Tentativa %s.", last_error, attempt + 1)
                self._punir_endpoint(indice, 300.0)
                time.sleep(min(2.0 ** attempt, 30.0))
                continue

            if response.status_code in {429, 502, 503, 504}:
                last_error = f"{endpoint}: HTTP {response.status_code}"
                backoff = min(2.0 ** attempt, 30.0)
                LOGGER.warning("Overpass sobrecarregado (%s). Aguardando %.0fs.", last_error, backoff)
                self._punir_endpoint(indice, 60.0)
                time.sleep(backoff)
                continue

            if not response.ok:
                last_error = f"{endpoint}: HTTP {response.status_code} {response.text[:200]}"
                LOGGER.warning("Overpass recusou a consulta (%s).", last_error)
                self._punir_endpoint(indice, 60.0)
                time.sleep(min(2.0 ** attempt, 30.0))
                continue

            try:
                payload = response.json()
            except ValueError:
                last_error = f"{endpoint}: resposta não é JSON"
                continue

            # O Overpass sinaliza estouro de recursos em "remark", com HTTP 200
            # e um conjunto de elementos truncado. Sem esta checagem, o
            # resultado parcial passaria como se fosse completo.
            remark = str(payload.get("remark", ""))
            if "error" in remark.lower() or "timed out" in remark.lower():
                last_error = f"{endpoint}: remark={remark[:200]}"
                LOGGER.warning("Overpass devolveu resultado parcial (%s).", last_error)
                time.sleep(min(2.0 ** attempt, 30.0))
                continue

            if not _instancia_sincronizada(payload):
                # Espelho com a base dessincronizada devolve HTTP 200, envelope
                # bem formado e **zero elementos**, o que é indistinguível de
                # uma região realmente vazia. O sinal está no carimbo de tempo
                # da base: numa instância sadia é ISO-8601, e nesta vinha como
                # um número solto. Sem esta checagem o resultado vazio entrava
                # no cache e a região ficava "sem telhados" por 30 dias.
                last_error = (
                    f"{endpoint}: base dessincronizada "
                    f"(timestamp_osm_base={payload.get('osm3s', {}).get('timestamp_osm_base')!r})"
                )
                LOGGER.warning("Overpass com base inconsistente (%s).", last_error)
                self._punir_endpoint(indice, 900.0)
                time.sleep(min(2.0 ** attempt, 15.0))
                continue

            # Este espelho respondeu: sai de qualquer castigo anterior.
            self._absolver_endpoint(indice)

            if use_cache:
                self.cache.set(query, payload)
            return payload

        raise OverpassError(f"Overpass falhou após {self.config.max_retries} tentativas. Último erro: {last_error}")

    # ------------------------------------------------------------------
    # Consultas
    # ------------------------------------------------------------------
    def buildings_query(self, bbox: Sequence[float], min_area_hint_m2: float = 0.0) -> str:
        """
        Monta a consulta de edificações para um bbox
        (min_lon, min_lat, max_lon, max_lat).

        Usa ``out tags geom`` para receber a geometria completa junto das tags
        numa única viagem. Relações entram porque galpões grandes e shoppings
        são frequentemente mapeados como multipolígonos.
        """
        min_lon, min_lat, max_lon, max_lat = bbox
        area = f"{min_lat:.7f},{min_lon:.7f},{max_lat:.7f},{max_lon:.7f}"
        return (
            f"[out:json][timeout:{self.config.timeout_s}];"
            "("
            f'way["building"]({area});'
            f'relation["building"]["type"="multipolygon"]({area});'
            f'way["building:part"]({area});'
            ");"
            "out tags geom;"
        )

    def fetch_buildings(
        self,
        bbox: Sequence[float],
        progress: Callable[[int, int], None] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """
        Baixa todas as edificações de um bbox, dividindo em tiles e
        paralelizando as consultas.

        Devolve (elementos, estatísticas). Tiles que falham são registrados nas
        estatísticas em vez de abortar a varredura inteira -- numa região
        grande, perder um tile é preferível a perder a prospecção toda.
        """
        from .geometry import tile_bbox  # import local evita ciclo na importação

        tiles = tile_bbox(bbox, self.config.tile_size_deg)
        stats: dict[str, Any] = {
            "tiles_total": len(tiles),
            "tiles_ok": 0,
            "tiles_falhos": 0,
            "tiles_com_erro": [],
            "elementos_brutos": 0,
        }
        elements: list[dict[str, Any]] = []
        done = 0

        with ThreadPoolExecutor(max_workers=max(1, self.config.workers)) as pool:
            futures = {pool.submit(self.run_query, self.buildings_query(tile)): tile for tile in tiles}
            for future in as_completed(futures):
                tile = futures[future]
                done += 1
                try:
                    payload = future.result()
                except Exception as exc:
                    stats["tiles_falhos"] += 1
                    stats["tiles_com_erro"].append({"bbox": list(tile), "erro": str(exc)[:300]})
                    LOGGER.warning("Tile %s falhou: %s", tile, exc)
                else:
                    stats["tiles_ok"] += 1
                    elements.extend(payload.get("elements", []))
                if progress is not None:
                    progress(done, len(tiles))

        # Se nenhum tile respondeu, o problema é o serviço, não a região. Sem
        # esta distinção o usuário receberia "nenhum telhado encontrado" e iria
        # procurar o erro nos filtros.
        if tiles and stats["tiles_ok"] == 0:
            primeiro_erro = (stats["tiles_com_erro"] or [{}])[0].get("erro", "causa não registrada")
            raise OverpassError(
                f"Nenhum dos {len(tiles)} tiles foi baixado: o Overpass está indisponível ou "
                f"sobrecarregado. Tente novamente em alguns minutos. Último erro: {primeiro_erro}"
            )

        # Um mesmo elemento pode aparecer em tiles vizinhos quando cruza a borda.
        seen: set[tuple[str, int]] = set()
        unique: list[dict[str, Any]] = []
        for element in elements:
            key = (element.get("type", ""), int(element.get("id", 0)))
            if key in seen:
                continue
            seen.add(key)
            unique.append(element)

        stats["elementos_brutos"] = len(elements)
        stats["elementos_unicos"] = len(unique)
        return unique, stats

    def fetch_uso_do_solo(self, bbox: Sequence[float]) -> list[dict[str, Any]]:
        """
        Baixa as áreas de uso do solo da região (landuse, leisure, tourism).

        É o contexto que permite classificar as edificações sem tag: no
        mapeamento brasileiro a maioria é ``building=yes``, e o que diz se
        aquilo é uma granja ou um centro de distribuição costuma ser a área
        que a contém, não o prédio.

        Uma consulta por região, não por telhado.
        """
        min_lon, min_lat, max_lon, max_lat = bbox
        area = f"{min_lat:.7f},{min_lon:.7f},{max_lat:.7f},{max_lon:.7f}"
        query = (
            f"[out:json][timeout:{self.config.timeout_s}];"
            "("
            f'way["landuse"]({area});'
            f'relation["landuse"]["type"="multipolygon"]({area});'
            f'way["amenity"~"^(school|university|college|hospital|clinic|marketplace)$"]({area});'
            f'way["leisure"~"^(sports_centre|stadium|golf_course|water_park)$"]({area});'
            f'way["tourism"~"^(hotel|resort|camp_site)$"]({area});'
            ");"
            "out tags geom;"
        )
        try:
            payload = self.run_query(query)
        except OverpassError as exc:
            LOGGER.warning("Uso do solo indisponível para a região: %s", exc)
            return []
        return payload.get("elements", [])

    def fetch_pois_near(
        self,
        lat: float,
        lon: float,
        radius_m: float = 40.0,
    ) -> list[dict[str, Any]]:
        """
        Busca POIs (nós e ways com nome/contato) num raio ao redor de um ponto.

        É o truque que faz o enriquecimento funcionar na prática: no OSM
        brasileiro, a edificação costuma vir sem tags comerciais, enquanto um
        nó de loja/empresa dentro dela carrega nome, telefone e site.
        """
        query = (
            f"[out:json][timeout:{min(self.config.timeout_s, 60)}];"
            "("
            f'node(around:{radius_m:.0f},{lat:.7f},{lon:.7f})["name"];'
            f'way(around:{radius_m:.0f},{lat:.7f},{lon:.7f})["name"]["building"!~"."];'
            ");"
            "out tags center;"
        )
        try:
            payload = self.run_query(query)
        except OverpassError as exc:
            LOGGER.info("Enriquecimento por POI indisponível em %.5f,%.5f: %s", lat, lon, exc)
            return []
        return payload.get("elements", [])

    def fetch_pois_in_bbox(self, bbox: Sequence[float]) -> list[dict[str, Any]]:
        """
        Busca de uma vez todos os POIs nomeados de um bbox.

        Prospecção em lote usa esta rota em vez de :meth:`fetch_pois_near` por
        telhado: uma consulta por região em vez de N consultas por edificação.
        """
        min_lon, min_lat, max_lon, max_lat = bbox
        area = f"{min_lat:.7f},{min_lon:.7f},{max_lat:.7f},{max_lon:.7f}"
        query = (
            f"[out:json][timeout:{self.config.timeout_s}];"
            "("
            f'node["name"]({area});'
            f'node["office"]({area});'
            f'node["shop"]({area});'
            f'node["amenity"]({area});'
            ");"
            "out tags center;"
        )
        try:
            payload = self.run_query(query)
        except OverpassError as exc:
            LOGGER.warning("Falha ao buscar POIs da região: %s", exc)
            return []
        return payload.get("elements", [])


def has_explicit_solar(tags: dict[str, Any]) -> bool:
    """True quando as tags declaram geração solar já instalada."""
    for key, accepted in SOLAR_TAGS.items():
        if key not in tags:
            continue
        if accepted is None:
            return True
        if str(tags[key]).strip().lower() in accepted:
            return True
    if tags.get("power") == "generator" and str(tags.get("generator:source", "")).lower() == "solar":
        return True
    return False
