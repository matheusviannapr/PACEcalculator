"""
Geração fotovoltaica hora a hora — a série que o estudo de baterias exige.

O :mod:`aurum.pv.solar` resolve o dimensionamento com energia **mensal**, e
para isso está certo: para saber quantos módulos cabem no telhado e quanto o
sistema economiza no ano, o total do mês basta. Para bateria, não basta em
nenhum grau. Um apagão às 19 h de um dia nublado de junho e um apagão às 11 h
de um dia limpo de janeiro têm o mesmo total mensal por trás e resultados
opostos: no primeiro a bateria sai sozinha, no segundo o sol recarrega o banco
enquanto ele atende a carga.

Duas fontes:

1. **PVGIS ``seriescalc``** (padrão) — série horária real de anos reais, base
   SARAH3. Vem daí a propriedade que nenhum "dia típico" reproduz: a
   **persistência do tempo**. Três dias nublados seguidos existem na série
   porque existiram no céu, e um apagão de 36 h que cai dentro deles é
   exatamente o caso que dimensiona o banco.
2. **Série sintética** — quando não há rede, ou para teste. Reconstrói o
   formato do dia por geometria solar e distribui a energia mensal conhecida
   (a que o :mod:`aurum.pv.solar` já sabe obter), com um índice de limpidez
   diário correlacionado ao do dia anterior para não perder a persistência.
   Fica marcada como sintética; quem emitir proposta em cima dela é avisado.

Unidade em toda a parte: **W por kWp instalado**. Multiplicar pela potência do
sistema dá watts. É a mesma normalização de :class:`aurum.pv.solar.PerfilGeracao`.
"""
from __future__ import annotations

import calendar
import logging
import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Sequence

import numpy as np
import requests

from ..config import Settings, get_settings
from ..pv.solar import PerfilGeracao, azimute_otimo, azimute_para_pvgis, inclinacao_otima
from ..util.cache import DiskCache

LOGGER = logging.getLogger(__name__)

PVGIS_SERIES_URL = "https://re.jrc.ec.europa.eu/api/v5_3/seriescalc"

#: Nomes das estações na grafia do D² — precisam bater exatamente, porque o
#: ensemble de carga é indexado por eles.
ESTACOES = ("verão", "outono", "inverno", "primavera")

#: Janela padrão de anos consultada no PVGIS. Cinco anos dão variedade de
#: clima sem transformar a consulta num download de dezenas de MB.
ANOS_PADRAO = (2016, 2020)

#: Fronteiras aproximadas das estações no hemisfério sul, como (mês, dia) de
#: início. Solstícios e equinócios variam um dia entre anos; a diferença não
#: muda nenhuma conclusão do estudo.
_INICIO_ESTACOES_SUL = (
    ((12, 21), "verão"),
    ((9, 23), "primavera"),
    ((6, 21), "inverno"),
    ((3, 21), "outono"),
)

__all__ = [
    "combinar_series",
    "SerieGeracao",
    "estacao_da_data",
    "obter_serie_horaria",
    "serie_sintetica",
]


def estacao_da_data(dia: date, hemisferio_sul: bool = True) -> str:
    """Estação astronômica da data, na grafia usada pelo ensemble de carga."""
    chave = (dia.month, dia.day)
    nome = "verão"  # antes de 21/3 o ano ainda está no verão que começou em dezembro
    for inicio, estacao in _INICIO_ESTACOES_SUL:
        if chave >= inicio:
            nome = estacao
            break
    if hemisferio_sul:
        return nome
    oposta = {"verão": "inverno", "inverno": "verão", "outono": "primavera", "primavera": "outono"}
    return oposta[nome]


# ----------------------------------------------------------------------------
# A série
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class SerieGeracao:
    """
    Geração horária de um sistema de **1 kWp**, organizada por dia.

    ``potencia_w_por_kwp`` tem forma ``(n_dias, 24)``. Guardar em dias e não
    numa fila corrida é o que permite sortear "um dia de inverno" e seguir dali
    para a frente pelos dias reais seguintes.
    """

    datas: tuple[date, ...]
    potencia_w_por_kwp: np.ndarray
    latitude: float
    longitude: float
    azimute_deg: float
    inclinacao_deg: float
    perdas_percent: float
    fonte: str = "pvgis_seriescalc"
    aviso: str | None = None
    metadados: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        matriz = np.asarray(self.potencia_w_por_kwp, dtype=float)
        if matriz.ndim != 2 or matriz.shape[1] != 24:
            raise ValueError("potencia_w_por_kwp deve ter forma (n_dias, 24)")
        if matriz.shape[0] != len(self.datas):
            raise ValueError("número de dias não bate com o número de datas")
        object.__setattr__(self, "potencia_w_por_kwp", matriz)
        # Classificar 1.800 datas é barato uma vez e caro a cada consulta:
        # `dias_da_estacao` é chamado dentro do laço de cenários.
        sul = self.latitude < 0
        object.__setattr__(
            self, "_estacoes", np.array([estacao_da_data(d, sul) for d in self.datas])
        )

    # -- propriedades ------------------------------------------------------
    @property
    def n_dias(self) -> int:
        return int(self.potencia_w_por_kwp.shape[0])

    @property
    def confiavel(self) -> bool:
        """False para a série sintética — que não deve sustentar proposta."""
        return self.fonte.startswith("pvgis")

    @property
    def estacoes(self) -> np.ndarray:
        """Vetor de estações, uma por dia."""
        return getattr(self, "_estacoes")

    def energia_diaria_kwh_por_kwp(self) -> np.ndarray:
        return self.potencia_w_por_kwp.sum(axis=1) / 1000.0

    def anual_kwh_por_kwp(self) -> float:
        """Produtividade anual média — comparável ao ``E_y`` do PVGIS."""
        return float(self.energia_diaria_kwh_por_kwp().sum() / self.n_dias * 365.25)

    def dias_da_estacao(self, estacao: str) -> np.ndarray:
        return np.flatnonzero(self.estacoes == estacao)

    # -- amostragem --------------------------------------------------------
    def amostrar_janelas(
        self,
        estacao: str | None,
        hora_inicio: int,
        n_passos: int,
        passo_min: int,
        quantidade: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        ``quantidade`` trechos de ``n_passos`` começando às ``hora_inicio``.

        Cada trecho parte de um dia sorteado da estação e **continua pelos dias
        reais seguintes** da série. É essa continuidade que carrega a
        persistência do tempo para dentro dos apagões longos: um trecho de 36 h
        que começa num dia encoberto tem alta chance de seguir encoberto, como
        no céu de verdade.

        Devolve ``(quantidade, n_passos)`` em W/kWp, com o degrau de uma hora
        preservado — a potência dentro da hora é constante, que é o que o dado
        do PVGIS de fato afirma (média horária), sem suavização inventada.
        """
        # Horas necessárias a partir do início, arredondando para cima.
        horas_necessarias = int(math.ceil((hora_inicio * 60 + n_passos * passo_min) / 60.0))
        dias_necessarios = int(math.ceil(horas_necessarias / 24.0))

        candidatos = np.arange(self.n_dias) if estacao is None else self.dias_da_estacao(estacao)
        candidatos = candidatos[candidatos <= self.n_dias - dias_necessarios]
        if candidatos.size == 0:
            raise ValueError(
                f"série curta demais: {n_passos * passo_min / 60:.0f} h a partir das "
                f"{hora_inicio}h não cabem em nenhum dia de {estacao or 'qualquer estação'}"
            )

        dias = rng.choice(candidatos, size=int(quantidade), replace=True)
        plano = self.potencia_w_por_kwp.reshape(-1)  # fila corrida de horas
        base = dias * 24 + int(hora_inicio)
        # Índice da hora de cada passo: degrau, não interpolação.
        deslocamento = (np.arange(n_passos) * passo_min) // 60
        return plano[base[:, None] + deslocamento[None, :]]

    def janela_media_por_hora(self, estacao: str | None = None) -> np.ndarray:
        """Perfil médio de 24 h da estação — para o gráfico, não para o despacho."""
        indices = np.arange(self.n_dias) if estacao is None else self.dias_da_estacao(estacao)
        return self.potencia_w_por_kwp[indices].mean(axis=0)

    def resumo(self) -> dict[str, Any]:
        return {
            "fonte": self.fonte,
            "confiavel": self.confiavel,
            "dias": self.n_dias,
            "periodo": f"{self.datas[0].isoformat()} a {self.datas[-1].isoformat()}",
            "anual_kwh_por_kwp": self.anual_kwh_por_kwp(),
            "azimute_deg": self.azimute_deg,
            "inclinacao_deg": self.inclinacao_deg,
            "perdas_percent": self.perdas_percent,
            "energia_diaria_media_kwh_por_kwp": {
                estacao: float(self.energia_diaria_kwh_por_kwp()[self.dias_da_estacao(estacao)].mean())
                for estacao in ESTACOES
                if self.dias_da_estacao(estacao).size
            },
            "aviso": self.aviso,
        }


# ----------------------------------------------------------------------------
# PVGIS
# ----------------------------------------------------------------------------
def deslocamento_utc_horas(longitude: float) -> int:
    """
    Fuso padrão da localidade, em horas inteiras a partir da longitude.

    O PVGIS carimba a série em UTC mesmo quando ``localtime=1`` é enviado --
    verificado em Curitiba, onde o pico de geração caía às 15 h. Três horas de
    erro não são detalhe: deslocam toda a sobreposição entre geração e carga,
    e com ela o autoconsumo, o estado de carga no início do apagão e a
    economia calculada.

    A longitude dividida por 15 dá o fuso de todas as capitais brasileiras.
    Ela erra em faixas onde a fronteira política do fuso não segue o meridiano
    -- o Acre é o caso conhecido -- e por isso o resultado é conferido contra
    o próprio dado em :func:`_alinhar_ao_relogio_local`.
    """
    return int(round(float(longitude) / 15.0))


def _alinhar_ao_relogio_local(
    datas: tuple[date, ...], matriz: np.ndarray, longitude: float
) -> tuple[tuple[date, ...], np.ndarray, str | None]:
    """
    Reindexa a matriz [dia, hora] de UTC para a hora do relógio local.

    Desloca a série corrida e descarta o primeiro e o último dia, que ficam
    incompletos: com fuso negativo, as horas finais do último dia viriam do
    começo do ano por circularidade.

    Devolve também um aviso quando o resultado não passa no teste de sanidade
    -- o centro de massa da geração tem de cair perto do meio-dia.
    """
    horas = deslocamento_utc_horas(longitude)
    if horas == 0:
        return datas, matriz, None

    corrida = matriz.reshape(-1)
    deslocada = np.roll(corrida, horas).reshape(matriz.shape)
    # A borda contaminada pela circularidade é o primeiro dia (fuso positivo)
    # ou o último (fuso negativo); descartar os dois custa dois dias em 365 e
    # dispensa o caso especial.
    datas_ok = datas[1:-1]
    matriz_ok = deslocada[1:-1]

    media = matriz_ok.mean(axis=0)
    if media.sum() > 0:
        centro = float((np.arange(24) + 0.5) @ media / media.sum())
    else:
        centro = float("nan")
    aviso = None
    if not (10.5 <= centro <= 13.5):
        aviso = (
            f"O centro de massa da geração caiu às {centro:.1f} h locais depois de "
            f"corrigir o fuso ({horas:+d} h). Conferir a hora da série antes de usar "
            "os números de autoconsumo."
        )
    return datas_ok, matriz_ok, aviso


def _montar_serie_pvgis(payload: dict, **kwargs: Any) -> SerieGeracao:
    horarios = payload["outputs"]["hourly"]
    por_dia: dict[date, list[float]] = {}
    for registro in horarios:
        carimbo = str(registro["time"])  # "20160101:0010"
        dia = date(int(carimbo[0:4]), int(carimbo[4:6]), int(carimbo[6:8]))
        por_dia.setdefault(dia, []).append(float(registro["P"]))

    # Dias incompletos aparecem nas bordas por causa do fuso; descartá-los é
    # mais honesto que completar com zero, que fabricaria noite onde havia sol.
    datas = tuple(sorted(d for d, horas in por_dia.items() if len(horas) == 24))
    if len(datas) < 60:
        raise ValueError(f"PVGIS devolveu apenas {len(datas)} dias completos")
    matriz = np.array([por_dia[d] for d in datas], dtype=float)
    datas, matriz, aviso = _alinhar_ao_relogio_local(datas, matriz, kwargs["longitude"])
    if aviso:
        LOGGER.warning("%s", aviso)
        kwargs.setdefault("aviso", aviso)
    return SerieGeracao(datas=datas, potencia_w_por_kwp=matriz, **kwargs)


def combinar_series(
    series: "Sequence[SerieGeracao]",
    pesos_kwp: "Sequence[float]",
) -> SerieGeracao:
    """
    A série do sistema inteiro, a partir de uma série por água.

    Cada água contribui com a sua geração por kWp na proporção da potência que
    carrega. O resultado continua sendo uma série de 1 kWp -- do sistema, e não
    de uma água -- e por isso tudo que consome :class:`SerieGeracao` continua
    funcionando sem saber que existem águas.

    A média ponderada é a operação certa e não uma aproximação: geração é
    aditiva, e a potência de cada água é conhecida. O que não se pode fazer é
    a média das *orientações* e pedir uma série só -- duas águas a leste e a
    oeste dariam uma água ao norte, com pico ao meio-dia que nenhuma das duas
    tem.

    As séries precisam cobrir os mesmos dias; é o que acontece quando saem da
    mesma coordenada e do mesmo intervalo de anos, que é o único uso previsto.
    """
    if not series:
        raise ValueError("combinar_series precisa de pelo menos uma série")
    if len(series) != len(pesos_kwp):
        raise ValueError("uma potência por série")

    pesos = np.asarray([max(0.0, float(p)) for p in pesos_kwp], dtype=float)
    total = float(pesos.sum())
    if total <= 0:
        raise ValueError("a potência total das águas precisa ser positiva")
    if len(series) == 1:
        return series[0]

    dias = {s.n_dias for s in series}
    if len(dias) != 1:
        raise ValueError(f"as séries cobrem números de dias diferentes: {sorted(dias)}")

    matriz = np.zeros_like(np.asarray(series[0].potencia_w_por_kwp, dtype=float))
    for serie, peso in zip(series, pesos):
        matriz += np.asarray(serie.potencia_w_por_kwp, dtype=float) * peso
    matriz /= total

    # A orientação declarada passa a ser a da água de maior potência: é a que
    # descreve melhor o conjunto, e guardar a média dos ângulos seria guardar
    # uma orientação que não existe no telhado.
    principal = series[int(np.argmax(pesos))]
    fontes = {s.fonte for s in series}
    avisos = [s.aviso for s in series if s.aviso]

    return SerieGeracao(
        datas=principal.datas,
        potencia_w_por_kwp=matriz,
        latitude=principal.latitude,
        longitude=principal.longitude,
        azimute_deg=principal.azimute_deg,
        inclinacao_deg=principal.inclinacao_deg,
        perdas_percent=principal.perdas_percent,
        fonte=principal.fonte if len(fontes) == 1 else "misto",
        aviso=" ".join(dict.fromkeys(avisos)) or None,
        metadados={
            "aguas": [
                {
                    "azimute_deg": s.azimute_deg,
                    "inclinacao_deg": s.inclinacao_deg,
                    "potencia_kwp": float(peso),
                    "produtividade_kwh_kwp_ano": s.anual_kwh_por_kwp(),
                    "fonte": s.fonte,
                }
                for s, peso in zip(series, pesos)
            ],
            "potencia_total_kwp": total,
        },
    )


def obter_serie_horaria(
    latitude: float,
    longitude: float,
    azimute_deg: float | None = None,
    inclinacao_deg: float | None = None,
    perdas_percent: float | None = None,
    anos: tuple[int, int] = ANOS_PADRAO,
    settings: Settings | None = None,
    permitir_fallback: bool = True,
    perfil_mensal: PerfilGeracao | None = None,
) -> SerieGeracao:
    """
    Série horária de 1 kWp na localidade, via PVGIS, com cache em disco.

    Sem rede — ou com o PVGIS fora do ar — cai para a série sintética, desde
    que ``permitir_fallback``. O ``perfil_mensal``, se dado, ancora a energia
    da série sintética nos totais mensais que o dimensionamento já usou; sem
    ele, a âncora é a correlação com a latitude, bem mais grosseira.
    """
    cfg = settings or get_settings()
    azimute = azimute_otimo(latitude) if azimute_deg is None else float(azimute_deg) % 360.0
    inclinacao = inclinacao_otima(latitude) if inclinacao_deg is None else float(inclinacao_deg)
    perdas = cfg.solar.losses_percent if perdas_percent is None else float(perdas_percent)

    comum = {
        "latitude": float(latitude),
        "longitude": float(longitude),
        "azimute_deg": azimute,
        "inclinacao_deg": inclinacao,
        "perdas_percent": perdas,
    }
    # A versão entra na chave porque as séries gravadas antes da correção de
    # fuso estão em UTC e não podem ser reaproveitadas.
    chave = dict(comum, anos=list(anos), versao=2)
    cache = DiskCache(cfg.cache_dir / "solar_horario", ttl_s=cfg.pvwatts.cache_ttl_s)

    guardado = cache.get(chave)
    if guardado is not None:
        return SerieGeracao(
            datas=tuple(date.fromisoformat(d) for d in guardado["datas"]),
            potencia_w_por_kwp=np.array(guardado["matriz"], dtype=float),
            fonte=guardado.get("fonte", "pvgis_seriescalc"),
            metadados=guardado.get("metadados", {}),
            **comum,
        )

    params = {
        "lat": round(float(latitude), 5),
        "lon": round(float(longitude), 5),
        "startyear": int(anos[0]),
        "endyear": int(anos[1]),
        "pvcalculation": 1,
        "peakpower": 1.0,
        "loss": round(perdas, 2),
        "angle": round(inclinacao, 2),
        "aspect": round(azimute_para_pvgis(azimute), 2),
        "mountingplace": "building",
        "pvtechchoice": "crystSi",
        "components": 0,
        "localtime": 1,
        "outputformat": "json",
    }
    erro: str | None = None
    try:
        resposta = requests.get(
            PVGIS_SERIES_URL, params=params, timeout=180,
            headers={"User-Agent": cfg.user_agent},
        )
        if resposta.ok:
            serie = _montar_serie_pvgis(
                resposta.json(),
                fonte="pvgis_seriescalc",
                metadados={"anos": list(anos)},
                **comum,
            )
            cache.set(chave, {
                "datas": [d.isoformat() for d in serie.datas],
                "matriz": serie.potencia_w_por_kwp.tolist(),
                "fonte": serie.fonte,
                "metadados": serie.metadados,
            })
            return serie
        erro = f"PVGIS seriescalc: HTTP {resposta.status_code} {resposta.text[:160]}"
    except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
        erro = f"PVGIS seriescalc: {exc}"

    LOGGER.info("Série horária indisponível: %s", erro)
    if not permitir_fallback:
        raise RuntimeError(f"Série horária de geração indisponível: {erro}")
    return serie_sintetica(
        perfil_mensal=perfil_mensal,
        anos=anos,
        aviso=(
            "Série horária reconstruída por geometria solar, sem consulta a base de "
            f"irradiação ({erro}). Serve para estudo interno; validar antes de proposta."
        ),
        **comum,
    )


# ----------------------------------------------------------------------------
# Série sintética
# ----------------------------------------------------------------------------
def _declinacao_solar(dia_do_ano: int) -> float:
    """Declinação em radianos pela equação de Cooper."""
    return math.radians(23.45) * math.sin(2.0 * math.pi * (284 + dia_do_ano) / 365.0)


def _perfil_horario_ceu_limpo(latitude_rad: float, declinacao: float) -> np.ndarray:
    """
    Forma do dia: irradiância relativa hora a hora, sem nuvem.

    Proporcional ao cosseno do ângulo zenital, que é a projeção geométrica do
    feixe sobre o plano — o que faz o dia ser mais longo e mais intenso no
    verão sem que nada disso precise ser tabelado.
    """
    horas = np.arange(24) + 0.5
    angulo_horario = np.radians(15.0 * (horas - 12.0))
    cos_zenital = (
        math.sin(latitude_rad) * math.sin(declinacao)
        + math.cos(latitude_rad) * math.cos(declinacao) * np.cos(angulo_horario)
    )
    return np.clip(cos_zenital, 0.0, None)


def serie_sintetica(
    latitude: float,
    longitude: float,
    azimute_deg: float,
    inclinacao_deg: float,
    perdas_percent: float,
    perfil_mensal: PerfilGeracao | None = None,
    anos: tuple[int, int] = ANOS_PADRAO,
    semente: int = 7,
    aviso: str | None = None,
) -> SerieGeracao:
    """
    Reconstrói uma série horária plausível a partir da energia mensal.

    Três camadas, nesta ordem:

    1. **Forma do dia** por geometria solar — dias longos no verão, curtos no
       inverno, pico ao meio-dia solar.
    2. **Limpidez diária** como processo AR(1): o dia de hoje se parece com o
       de ontem. Sem isso, três dias nublados seguidos seriam raríssimos no
       modelo e comuns na realidade, e o apagão longo sairia fácil demais.
    3. **Renormalização mensal** para que a energia do mês bata exatamente com
       o total conhecido — do ``perfil_mensal``, quando há, ou da correlação
       com a latitude.
    """
    from ..pv.solar import rendimento_estimado_kwh_kwp_ano

    if perfil_mensal is not None:
        mensal_kwh = np.array(perfil_mensal.mensal_kwh_por_kwp, dtype=float)
    else:
        anual = rendimento_estimado_kwh_kwp_ano(latitude)
        pesos = np.array([0.092, 0.090, 0.088, 0.084, 0.080, 0.076, 0.074, 0.076, 0.081, 0.086, 0.089, 0.084])
        mensal_kwh = anual * pesos

    rng = np.random.default_rng(semente)
    lat_rad = math.radians(float(latitude))

    datas: list[date] = []
    for ano in range(int(anos[0]), int(anos[1]) + 1):
        for mes in range(1, 13):
            for dia in range(1, calendar.monthrange(ano, mes)[1] + 1):
                datas.append(date(ano, mes, dia))

    formas = np.array(
        [_perfil_horario_ceu_limpo(lat_rad, _declinacao_solar(d.timetuple().tm_yday)) for d in datas]
    )

    # Limpidez: AR(1) em [0,1] com média ~0.72 e persistência de ~2 dias.
    rho, media, desvio = 0.62, 0.72, 0.20
    ruido = rng.normal(0.0, desvio * math.sqrt(1 - rho**2), size=len(datas))
    limpidez = np.empty(len(datas))
    limpidez[0] = media
    for i in range(1, len(datas)):
        limpidez[i] = media + rho * (limpidez[i - 1] - media) + ruido[i]
    limpidez = np.clip(limpidez, 0.12, 1.0)

    bruto = formas * limpidez[:, None]

    # Renormaliza mês a mês para casar com a energia mensal conhecida.
    matriz = np.zeros_like(bruto)
    meses = np.array([d.month for d in datas])
    anos_arr = np.array([d.year for d in datas])
    for ano in range(int(anos[0]), int(anos[1]) + 1):
        for mes in range(1, 13):
            sel = (meses == mes) & (anos_arr == ano)
            soma = bruto[sel].sum()
            if soma <= 0:
                continue
            # bruto está em unidade arbitrária; escala para kWh do mês -> W por hora
            matriz[sel] = bruto[sel] * (mensal_kwh[mes - 1] * 1000.0 / soma)

    return SerieGeracao(
        datas=tuple(datas),
        potencia_w_por_kwp=matriz,
        latitude=float(latitude),
        longitude=float(longitude),
        azimute_deg=float(azimute_deg),
        inclinacao_deg=float(inclinacao_deg),
        perdas_percent=float(perdas_percent),
        fonte="sintetica",
        aviso=aviso or "Série horária sintética: forma por geometria solar, energia ancorada no total mensal.",
        metadados={"semente": semente, "anos": list(anos)},
    )
