"""
Configuração central do Aurum.

Tudo que é ajustável — caminhos, chaves de API, limites de rede e premissas
técnicas padrão — fica aqui. Variáveis de ambiente sobrepõem os defaults, e a
`Settings` pode ser sobrescrita em memória pela interface Streamlit.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Raiz do projeto = pasta que contém o pacote `aurum`
ROOT_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = Path(os.environ.get("AURUM_CACHE_DIR", ROOT_DIR / ".cache"))
OUTPUT_DIR = Path(os.environ.get("AURUM_OUTPUT_DIR", ROOT_DIR / "outputs"))
DATA_DIR = Path(os.environ.get("AURUM_DATA_DIR", ROOT_DIR / "data"))
EQUIPMENT_XLSX = Path(os.environ.get("AURUM_EQUIPMENT_XLSX", ROOT_DIR / "BDFotovoltaica.xlsx"))

#: Enviado em toda requisição a serviços da OSMF. A política de uso do
#: Nominatim exige identificação real da aplicação.
USER_AGENT = os.environ.get(
    "AURUM_USER_AGENT",
    "aurum-solar-prospector/3.0 (https://github.com/matheusviannapr/Aurum)",
)


@dataclass
class OverpassSettings:
    """Parâmetros de acesso à API Overpass."""

    #: Espelhos tentados em ordem; se um falha, o próximo assume.
    #: Espelhos públicos do Overpass, tentados em ordem com desvio automático
    #: para os que estão respondendo. A lista é longa de propósito: em teste,
    #: três dos oito espelhos conhecidos estavam simultaneamente fora do ar,
    #: com HTTP 500 ou com a base dessincronizada. Com um só endereço, a
    #: prospecção simplesmente não roda nesses momentos.
    endpoints: tuple[str, ...] = (
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://overpass.private.coffee/api/interpreter",
        "https://overpass.osm.ch/api/interpreter",
        "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    )
    #: Tempo máximo de execução da consulta, informado ao próprio Overpass e
    #: usado como timeout de leitura. Consultas legítimas demoram minutos.
    timeout_s: int = 180
    #: Timeout apenas para estabelecer a conexão. Curto de propósito: um
    #: espelho fora do ar deve ceder a vez ao próximo em segundos, não em
    #: três minutos. Sem esta separação, uma varredura contra um servidor
    #: sobrecarregado ficava parada sem dar sinal de vida.
    connect_timeout_s: int = 10
    #: Intervalo mínimo entre requisições, global ao processo.
    min_interval_s: float = 1.0
    #: Precisa cobrir a lista de espelhos com folga: com menos tentativas que
    #: endereços, um espelho saudável no fim da fila nunca chega a ser testado.
    max_retries: int = 6
    workers: int = 4
    #: Lado do tile em graus. 0,02° ≈ 2,2 km — equilibra número de requisições
    #: e risco de estourar o limite de memória do Overpass.
    tile_size_deg: float = 0.02
    cache_ttl_s: float | None = 30 * 24 * 3600  # 30 dias


@dataclass
class NominatimSettings:
    """Parâmetros de geocodificação."""

    endpoint: str = "https://nominatim.openstreetmap.org"
    timeout_s: int = 30
    min_interval_s: float = 1.1  # a política do Nominatim exige >= 1 req/s
    max_retries: int = 3
    cache_ttl_s: float | None = 90 * 24 * 3600  # 90 dias


@dataclass
class PVWattsSettings:
    """Parâmetros da API PVWatts v8 (NREL)."""

    api_key: str = field(default_factory=lambda: os.environ.get("PVWATTS_API_KEY", "DEMO_KEY"))
    url: str = os.environ.get("PVWATTS_URL", "https://developer.nrel.gov/api/pvwatts/v8.json")
    timeout_s: int = 30
    max_retries: int = 3
    #: Casas decimais usadas para arredondar lat/lon na chave de cache.
    #: 2 casas ≈ 1,1 km — telhados da mesma região compartilham uma consulta,
    #: o que é essencial para prospecção em lote com a DEMO_KEY.
    coord_precision: int = 2
    cache_ttl_s: float | None = None  # dados climáticos típicos não expiram

    @property
    def is_demo_key(self) -> bool:
        return self.api_key.strip().upper() == "DEMO_KEY"


@dataclass
class SolarDefaults:
    """Premissas técnicas padrão do dimensionamento."""

    #: Perdas totais do sistema em % (sujeira, cabos, mismatch, temperatura).
    losses_percent: float = 14.0
    #: 0 = standard, 1 = premium, 2 = filme fino (convenção PVWatts).
    module_type: int = 0
    #: 0 = fixo aberto, 1 = fixo em telhado, 2 = 1 eixo, 3 = 1 eixo backtracking, 4 = 2 eixos.
    array_type: int = 1
    #: Razão CC/CA de projeto (oversizing do arranjo em relação ao inversor).
    dc_ac_ratio: float = 1.20
    inverter_efficiency_percent: float = 96.0
    #: Recuo livre a partir da borda do telhado, em metros (circulação/NBR).
    edge_setback_m: float = 0.80
    #: Razão de ocupação do solo para estruturas inclinadas em laje plana.
    #: 0,45 evita sombreamento entre fileiras em latitudes brasileiras.
    ground_coverage_ratio: float = 0.45
    #: Folga entre módulos coplanares, em metros.
    module_gap_m: float = 0.02
    degradation_per_year: float = 0.006
    tariff_escalation_per_year: float = 0.05
    discount_rate_per_year: float = 0.10
    analysis_years: int = 25
    #: OPEX anual como fração do CAPEX (limpeza, seguro, monitoramento).
    opex_percent_of_capex: float = 0.01
    #: Custo de referência do sistema instalado, em R$/kWp. Decresce com a
    #: escala; ver `aurum.pv.financials.estimar_capex`.
    capex_reference_brl_per_kwp: float = 3200.0
    #: Preços indicativos de equipamento, usados para comparar alternativas de
    #: arranjo e para compor o CAPEX. Ajuste conforme a tabela do fornecedor.
    preco_modulo_brl_por_wp: float = 0.90
    preco_inversor_brl_por_w: float = 0.45


@dataclass
class Settings:
    """Agregador de configuração passado explicitamente aos clientes."""

    overpass: OverpassSettings = field(default_factory=OverpassSettings)
    nominatim: NominatimSettings = field(default_factory=NominatimSettings)
    pvwatts: PVWattsSettings = field(default_factory=PVWattsSettings)
    solar: SolarDefaults = field(default_factory=SolarDefaults)
    user_agent: str = USER_AGENT
    cache_dir: Path = CACHE_DIR
    output_dir: Path = OUTPUT_DIR
    equipment_xlsx: Path = EQUIPMENT_XLSX


#: Instância padrão usada quando o chamador não fornece uma.
SETTINGS = Settings()


def get_settings() -> Settings:
    return SETTINGS
