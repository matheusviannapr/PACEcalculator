"""
Recurso solar: quanto um sistema de 1 kWp gera numa localidade.

Três fontes, tentadas em ordem:

1. **PVGIS** (Comissão Europeia / JRC) -- base SARAH3 derivada de satélite,
   cobertura global, sem chave de API e sem limite de requisições. É a fonte
   primária justamente por isso: prospecção em lote faz muitas consultas.
2. **PVWatts v8** (NREL) -- exige chave; a DEMO_KEY tolera poucas dezenas de
   chamadas por hora, o que não sustenta prospecção em lote.
3. **Estimativa offline** -- correlação com a latitude, quando não há rede.
   Fica marcada como tal para que a proposta declare a origem.

Convenção interna de azimute (a mesma do PVWatts): **0 = Norte**, 90 = Leste,
180 = Sul, 270 = Oeste. O PVGIS usa 0 = Sul e 180 = Norte, e a conversão é
feita na fronteira do cliente.

O erro corrigido aqui: o código anterior usava azimute 180 como padrão no
Brasil. Medido em Curitiba, um telhado voltado ao Sul gera 967 kWh/kWp/ano
contra 1.304 voltado ao Norte -- 26% a menos, o que inflava na mesma
proporção o sistema dimensionado e o investimento proposto.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import requests

from ..config import Settings, get_settings
from ..util.cache import DiskCache

LOGGER = logging.getLogger(__name__)

MESES = ("Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez")

PVGIS_URL = "https://re.jrc.ec.europa.eu/api/v5_3/PVcalc"

#: Distribuição mensal do fallback offline. Sazonalidade moderada; soma 1,0.
_PESOS_MENSAIS = (0.092, 0.090, 0.088, 0.084, 0.080, 0.076, 0.074, 0.076, 0.081, 0.086, 0.089, 0.084)


def azimute_otimo(latitude: float) -> float:
    """
    Azimute de maior geração anual, na convenção interna (0 = Norte).

    Hemisfério sul aponta ao Norte; hemisfério norte, ao Sul.
    """
    return 0.0 if float(latitude) < 0 else 180.0


def inclinacao_otima(latitude: float) -> float:
    """
    Inclinação recomendada: módulo da latitude, limitado a [10, 40] graus.

    O piso de 10 graus não é ótica -- é manutenção: abaixo disso a água da
    chuva não escorre e a perda por sujidade acumulada supera o ganho de
    captação. O teto de 40 evita esforço de vento desnecessário.
    """
    return max(10.0, min(40.0, abs(float(latitude))))


def rendimento_estimado_kwh_kwp_ano(latitude: float) -> float:
    """
    Produtividade anual estimada, usada só quando não há rede.

    Faixa brasileira típica: ~1.150 (extremo sul, nublado) a ~1.800
    kWh/kWp/ano (semiárido nordestino).
    """
    try:
        lat_abs = abs(float(latitude))
    except (TypeError, ValueError):
        lat_abs = 15.0
    return float(max(1150.0, min(1800.0, 1800.0 - lat_abs * 18.0)))


def azimute_para_pvgis(azimute_interno: float) -> float:
    """
    Converte da convenção interna (0 = Norte) para a do PVGIS (0 = Sul),
    normalizando para o intervalo [-180, 180] que a API aceita.
    """
    aspecto = (float(azimute_interno) + 180.0) % 360.0
    return aspecto - 360.0 if aspecto > 180.0 else aspecto


def nome_orientacao(azimute: float) -> str:
    """Rótulo cardinal legível para o azimute interno."""
    pontos = [
        (0, "Norte"), (45, "Nordeste"), (90, "Leste"), (135, "Sudeste"),
        (180, "Sul"), (225, "Sudoeste"), (270, "Oeste"), (315, "Noroeste"),
    ]
    az = float(azimute) % 360.0
    melhor = min(pontos, key=lambda p: min(abs(az - p[0]), 360 - abs(az - p[0])))
    return melhor[1]


@dataclass
class PerfilGeracao:
    """
    Geração de um sistema de **1 kWp** numa localidade e configuração.

    Como a saída de qualquer simulador é linear na potência instalada para a
    mesma configuração, este perfil normalizado é tudo que precisa ser
    guardado: qualquer potência se obtém multiplicando.
    """

    latitude: float
    longitude: float
    azimuth_deg: float
    tilt_deg: float
    losses_percent: float
    anual_kwh_por_kwp: float
    mensal_kwh_por_kwp: tuple[float, ...]
    fonte: str = "pvgis"
    base_dados: str | None = None
    irradiacao_kwh_m2_ano: float | None = None
    elevacao_m: float | None = None
    aviso: str | None = None

    @property
    def confiavel(self) -> bool:
        """False quando o valor veio da estimativa offline."""
        return self.fonte in {"pvgis", "pvwatts"}

    @property
    def orientacao(self) -> str:
        return nome_orientacao(self.azimuth_deg)

    def anual(self, kwp: float) -> float:
        return float(kwp) * self.anual_kwh_por_kwp

    def mensal(self, kwp: float) -> list[float]:
        return [float(kwp) * v for v in self.mensal_kwh_por_kwp]

    def fator_capacidade(self) -> float:
        """Fator de capacidade em fração (0..1)."""
        return self.anual_kwh_por_kwp / 8760.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "azimuth_deg": self.azimuth_deg,
            "orientacao": self.orientacao,
            "tilt_deg": self.tilt_deg,
            "losses_percent": self.losses_percent,
            "anual_kwh_por_kwp": self.anual_kwh_por_kwp,
            "mensal_kwh_por_kwp": list(self.mensal_kwh_por_kwp),
            "fonte": self.fonte,
            "base_dados": self.base_dados,
            "irradiacao_kwh_m2_ano": self.irradiacao_kwh_m2_ano,
            "elevacao_m": self.elevacao_m,
            "fator_capacidade": self.fator_capacidade(),
            "aviso": self.aviso,
        }


class SolarResourceClient:
    """Obtém perfis de geração com cache por localidade arredondada."""

    def __init__(
        self,
        settings: Settings | None = None,
        provedores: Sequence[str] = ("pvgis", "pvwatts"),
    ) -> None:
        self.settings: Settings = settings or get_settings()
        self.provedores = tuple(provedores)
        self.cache = DiskCache(self.settings.cache_dir / "solar", ttl_s=self.settings.pvwatts.cache_ttl_s)
        #: Cache curto para falhas, para que um lote de 300 telhados não
        #: repita 300 vezes a mesma consulta de rede que já falhou.
        self.cache_falhas = DiskCache(self.settings.cache_dir / "solar_falhas", ttl_s=900)
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": self.settings.user_agent})
        self.ultimo_erro: str | None = None

    # ------------------------------------------------------------------
    def _chave(self, lat: float, lon: float, azimuth: float, tilt: float, losses: float) -> dict[str, Any]:
        casas = self.settings.pvwatts.coord_precision
        return {
            "lat": round(float(lat), casas),
            "lon": round(float(lon), casas),
            "az": round(float(azimuth), 1),
            "tilt": round(float(tilt), 1),
            "loss": round(float(losses), 1),
        }

    # ------------------------------------------------------------------
    def _consultar_pvgis(
        self, lat: float, lon: float, azimuth: float, tilt: float, losses: float
    ) -> tuple[dict[str, Any] | None, str | None]:
        params = {
            "lat": round(float(lat), 5),
            "lon": round(float(lon), 5),
            "peakpower": 1.0,
            "loss": round(float(losses), 2),
            "angle": round(float(tilt), 2),
            "aspect": round(azimute_para_pvgis(azimuth), 2),
            "mountingplace": "building",
            "outputformat": "json",
        }
        try:
            response = self._session.get(PVGIS_URL, params=params, timeout=60)
        except requests.RequestException as exc:
            return None, f"PVGIS: falha de rede ({exc})"
        if not response.ok:
            return None, f"PVGIS: HTTP {response.status_code} {response.text[:160]}"
        try:
            payload = response.json()
        except ValueError:
            return None, "PVGIS: resposta não é JSON"

        try:
            saidas = payload["outputs"]
            totais = saidas["totals"]["fixed"]
            mensal_raw = saidas["monthly"]["fixed"]
            anual = float(totais["E_y"])
            mensal = [float(m["E_m"]) for m in sorted(mensal_raw, key=lambda m: int(m["month"]))]
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            return None, f"PVGIS: resposta inesperada ({exc})"

        if len(mensal) != 12 or anual <= 0:
            return None, "PVGIS: série mensal incompleta"

        entradas = payload.get("inputs", {})
        return (
            {
                "anual": anual,
                "mensal": mensal,
                "fonte": "pvgis",
                "base_dados": entradas.get("meteo_data", {}).get("radiation_db"),
                "irradiacao": float(totais.get("H(i)_y")) if totais.get("H(i)_y") is not None else None,
                "elevacao": entradas.get("location", {}).get("elevation"),
            },
            None,
        )

    def _consultar_pvwatts(
        self, lat: float, lon: float, azimuth: float, tilt: float, losses: float
    ) -> tuple[dict[str, Any] | None, str | None]:
        cfg = self.settings.pvwatts
        base = {
            "system_capacity": 1.0,
            "module_type": self.settings.solar.module_type,
            "losses": round(float(losses), 2),
            "array_type": self.settings.solar.array_type,
            "tilt": round(float(tilt), 2),
            "azimuth": round(float(azimuth) % 360.0, 2),
            "lat": round(float(lat), 4),
            "lon": round(float(lon), 4),
            "timeframe": "monthly",
            "api_key": cfg.api_key,
        }
        # A cobertura NSRDB não alcança boa parte do Brasil; a base "intl" sim.
        for dataset in (None, "intl"):
            params = dict(base) if dataset is None else dict(base, dataset=dataset)
            try:
                response = self._session.get(cfg.url, params=params, timeout=cfg.timeout_s)
            except requests.RequestException as exc:
                return None, f"PVWatts: falha de rede ({exc})"
            try:
                payload = response.json()
            except ValueError:
                payload = None
            if response.status_code == 429:
                return None, "PVWatts: cota da chave esgotada (HTTP 429)"
            if not response.ok or not isinstance(payload, dict):
                continue
            if payload.get("errors"):
                continue
            saidas = payload.get("outputs", {})
            anual = float(saidas.get("ac_annual") or 0.0)
            mensal = saidas.get("ac_monthly")
            if anual > 0 and isinstance(mensal, list) and len(mensal) == 12:
                return (
                    {
                        "anual": anual,
                        "mensal": [float(v) for v in mensal],
                        "fonte": "pvwatts",
                        "base_dados": (payload.get("station_info") or {}).get("solar_resource_file"),
                        "irradiacao": saidas.get("solrad_annual"),
                        "elevacao": (payload.get("station_info") or {}).get("elev"),
                    },
                    None,
                )
        return None, "PVWatts: nenhuma base climática cobre a localidade"

    # ------------------------------------------------------------------
    def perfil(
        self,
        latitude: float,
        longitude: float,
        azimuth_deg: float | None = None,
        tilt_deg: float | None = None,
        losses_percent: float | None = None,
        permitir_fallback: bool = True,
    ) -> PerfilGeracao:
        """
        Perfil de geração de 1 kWp. Sem azimute/inclinação explícitos, usa os
        ótimos da latitude -- no Brasil, azimute 0 (Norte).
        """
        azimuth = azimute_otimo(latitude) if azimuth_deg is None else float(azimuth_deg) % 360.0
        tilt = inclinacao_otima(latitude) if tilt_deg is None else float(tilt_deg)
        losses = self.settings.solar.losses_percent if losses_percent is None else float(losses_percent)

        chave = self._chave(latitude, longitude, azimuth, tilt, losses)

        def _montar(dados: dict[str, Any], aviso: str | None = None) -> PerfilGeracao:
            return PerfilGeracao(
                latitude=float(latitude),
                longitude=float(longitude),
                azimuth_deg=azimuth,
                tilt_deg=tilt,
                losses_percent=losses,
                anual_kwh_por_kwp=float(dados["anual"]),
                mensal_kwh_por_kwp=tuple(float(v) for v in dados["mensal"]),
                fonte=str(dados.get("fonte", "pvgis")),
                base_dados=dados.get("base_dados"),
                irradiacao_kwh_m2_ano=dados.get("irradiacao"),
                elevacao_m=dados.get("elevacao"),
                aviso=aviso or dados.get("aviso"),
            )

        guardado = self.cache.get(chave)
        if guardado is not None:
            return _montar(guardado)

        erros: list[str] = []
        # Se a mesma consulta falhou há pouco, não repete a rede: em lote,
        # isso transformaria uma indisponibilidade em minutos de espera.
        falha_recente = self.cache_falhas.get(chave)
        if falha_recente is None:
            consultas = {"pvgis": self._consultar_pvgis, "pvwatts": self._consultar_pvwatts}
            for nome in self.provedores:
                consulta = consultas.get(nome)
                if consulta is None:
                    continue
                dados, erro = consulta(latitude, longitude, azimuth, tilt, losses)
                if dados is not None:
                    self.cache.set(chave, dados)
                    self.ultimo_erro = None
                    return _montar(dados)
                if erro:
                    erros.append(erro)
                    LOGGER.info("Fonte %s indisponível: %s", nome, erro)
            self.cache_falhas.set(chave, {"erros": erros, "quando": time.time()})
        else:
            erros = list(falha_recente.get("erros") or ["consulta anterior falhou há menos de 15 min"])

        self.ultimo_erro = "; ".join(erros) if erros else "nenhum provedor disponível"

        if not permitir_fallback:
            raise RuntimeError(f"Recurso solar indisponível: {self.ultimo_erro}")

        anual = rendimento_estimado_kwh_kwp_ano(latitude)
        aviso = (
            "Geração estimada por correlação com a latitude, sem consulta a base de "
            f"irradiação ({self.ultimo_erro}). Validar antes de emitir proposta comercial."
        )
        return _montar(
            {
                "anual": anual,
                "mensal": [anual * p for p in _PESOS_MENSAIS],
                "fonte": "estimativa_offline",
                "base_dados": None,
                "irradiacao": None,
                "elevacao": None,
            },
            aviso=aviso,
        )

    def perfil_para_lead(self, lead, tilt_deg: float | None = None) -> PerfilGeracao:
        """Atalho: perfil no centroide de um :class:`~aurum.geo.roofs.RoofLead`."""
        return self.perfil(lead.centroid_lat, lead.centroid_lon, tilt_deg=tilt_deg)
