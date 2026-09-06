"""
Orquestração: da região ao pacote de propostas.

Este é o módulo que une os dois programas originais. O fluxo completo:

    região (nome ou bbox)
        -> Overpass: todas as edificações
        -> filtro e pontuação: telhados aproveitáveis, do maior para o menor
        -> enriquecimento: quem ocupa e como falar com eles
        -> [seleção do usuário]
        -> por telhado: recurso solar, layout de módulos, arranjo elétrico,
           consumo estimado, análise econômica
        -> proposta em LaTeX + JSON + índice consolidado

As duas etapas são separadas de propósito: a prospecção é cara em rede e
rápida em CPU; o dimensionamento é o contrário. Separá-las permite prospectar
uma cidade inteira, olhar a lista e só então gastar cálculo nos telhados que
interessam.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .config import Settings, get_settings
from .geo.enrich import enriquecer_leads
from .geo.geometry import bbox_area_km2, polygon_from_geojson
from .geo.nominatim import NominatimClient, Place, parse_bbox
from .geo.overpass import OverpassClient
from .geo.roofs import ContextoUsoSolo, RoofFilterConfig, RoofLead, build_leads
from .proposal.context import ContextoProposta, DadosEmissor
from .pv import consumption, financials
from .pv.equipment import BaseEquipamentos, carregar_base
from .pv.layout import calcular_layout, fator_obstaculos_para
from .pv.sizing import melhor_sistema
from .pv.solar import SolarResourceClient, inclinacao_otima

LOGGER = logging.getLogger(__name__)

#: Acima desta área a varredura já é lenta, mas ainda vale a pena esperar.
AREA_ALERTA_KM2 = 150.0

#: Acima desta área a varredura vira um problema: milhares de tiles, dezenas
#: de minutos e uma carga desproporcional nos servidores públicos do Overpass.
#: Um município inteiro passa fácil de 1.000 km², e quase sempre é engano --
#: o que se quer é o distrito industrial, não a área rural toda junto.
#: Passar deste limite exige dizer que é intencional.
AREA_MAXIMA_KM2 = 400.0


class AreaGrandeDemais(ValueError):
    """
    A região pedida é grande demais para uma varredura sem confirmação.

    O diagnóstico é o mesmo em toda parte, mas a forma de liberar muda com a
    interface. Por isso a exceção separa as duas coisas: ``diagnostico`` é
    fixo e ``remedio`` cada chamador ajusta ao que o usuário dele pode fazer.
    """

    REMEDIO_PADRAO = (
        "Para varrer assim mesmo, passe confirmar_area_grande=True "
        "(ou --area-grande na linha de comando)."
    )

    def __init__(self, area_km2: float, tiles_estimados: int, remedio: str | None = None) -> None:
        self.area_km2 = area_km2
        self.tiles_estimados = tiles_estimados
        self.remedio = remedio or self.REMEDIO_PADRAO
        from .util.numfmt import br_int

        self.diagnostico = (
            f"A região tem {br_int(area_km2)} km², o que exigiria cerca de "
            f"{br_int(tiles_estimados)} consultas ao OpenStreetMap e muitos minutos de espera.\n"
            "Recorte em bairros ou distritos (o distrito industrial costuma ser o que "
            "interessa), ou informe um bbox menor."
        )
        super().__init__(f"{self.diagnostico}\n{self.remedio}")

    def com_remedio(self, remedio: str) -> "AreaGrandeDemais":
        """Mesma falha, com a instrução adequada à interface que a exibe."""
        return AreaGrandeDemais(self.area_km2, self.tiles_estimados, remedio)


@dataclass
class ResultadoProspeccao:
    """Saída da etapa de prospecção."""

    leads: list[RoofLead]
    bbox: tuple[float, float, float, float]
    place: Place | None
    estatisticas: dict[str, Any] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return len(self.leads)

    def top(self, n: int = 50) -> list[RoofLead]:
        return self.leads[:n]

    def por_area(self, n: int | None = None) -> list[RoofLead]:
        """Os maiores telhados, ignorando a pontuação de qualidade."""
        ordenados = sorted(self.leads, key=lambda l: l.metrics.area_m2, reverse=True)
        return ordenados[:n] if n else ordenados

    def to_geojson(self) -> dict[str, Any]:
        return {
            "type": "FeatureCollection",
            "features": [lead.to_geojson_feature() for lead in self.leads],
        }

    def to_records(self) -> list[dict[str, Any]]:
        return [lead.to_record() for lead in self.leads]


def resolver_area(
    regiao: str | None = None,
    bbox: str | Sequence[float] | None = None,
    settings: Settings | None = None,
    nominatim: NominatimClient | None = None,
) -> tuple[tuple[float, float, float, float], Place | None]:
    """
    Resolve a área de interesse a partir de um nome ou de um bbox explícito.

    Devolve (bbox, place). O ``place`` traz o contorno administrativo quando o
    Nominatim o fornece, o que permite descartar edificações que caem no bbox
    mas fora do município.
    """
    if bbox is not None:
        caixa = parse_bbox(bbox) if isinstance(bbox, str) else tuple(float(v) for v in bbox)
        return caixa, None  # type: ignore[return-value]
    if not regiao:
        raise ValueError("Informe uma região (nome) ou um bbox.")

    cliente = nominatim or NominatimClient(settings or get_settings())
    place = cliente.geocode(regiao)
    return place.bbox, place


def estimar_tiles(bbox: Sequence[float], settings: Settings | None = None) -> int:
    """Quantas consultas ao Overpass uma região exigiria."""
    from .geo.geometry import tile_bbox

    settings = settings or get_settings()
    return len(tile_bbox(bbox, settings.overpass.tile_size_deg))


def resumo_da_area(
    regiao: str | None = None,
    bbox: str | Sequence[float] | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """
    Descreve a região antes de varrer: nome, área, tiles e tempo estimado.

    Serve para a interface avisar que o usuário pediu um município inteiro
    *antes* de gastar meia hora descobrindo isso.
    """
    settings = settings or get_settings()
    caixa, place = resolver_area(regiao, bbox, settings)
    area_km2 = bbox_area_km2(caixa)
    tiles = estimar_tiles(caixa, settings)
    # Medido em campo: ~1,2 s por tile com o cache frio e 4 trabalhadores.
    segundos = tiles * 1.2
    return {
        "nome": place.display_name if place else "bbox informado",
        "bbox": caixa,
        "area_km2": area_km2,
        "tiles": tiles,
        "minutos_estimados": segundos / 60.0,
        "grande_demais": area_km2 > AREA_MAXIMA_KM2,
        "lenta": area_km2 > AREA_ALERTA_KM2,
    }


def prospectar(
    regiao: str | None = None,
    bbox: str | Sequence[float] | None = None,
    min_area_m2: float = 500.0,
    alvo: str | Sequence[str] = "todos",
    rigoroso: bool = False,
    limite: int | None = None,
    usar_uso_do_solo: bool = True,
    confirmar_area_grande: bool = False,
    enriquecer: bool = True,
    limite_endereco: int = 40,
    restringir_ao_contorno: bool = True,
    settings: Settings | None = None,
    progresso: Callable[[str, float], None] | None = None,
) -> ResultadoProspeccao:
    """
    Varre uma região e devolve os telhados aproveitáveis, do melhor para o pior.

    ``alvo`` aceita um segmento, vários separados por vírgula ou uma lista --
    ver :mod:`aurum.geo.segments` para as chaves disponíveis.

    ``usar_uso_do_solo`` baixa também as áreas de uso do solo da região, para
    classificar as edificações sem tag. Custa uma consulta a mais e é o que
    faz o filtro por segmento funcionar fora dos grandes centros, onde quase
    tudo é ``building=yes``.

    ``progresso`` recebe (mensagem, fração de 0 a 1) e permite que a interface
    mostre andamento -- uma varredura de bairro leva minutos.
    """
    settings = settings or get_settings()

    def avisar(mensagem: str, fracao: float) -> None:
        LOGGER.info("[%3.0f%%] %s", fracao * 100, mensagem)
        if progresso:
            progresso(mensagem, fracao)

    avisar("Resolvendo a região", 0.02)
    caixa, place = resolver_area(regiao, bbox, settings)
    area_km2 = bbox_area_km2(caixa)

    if area_km2 > AREA_MAXIMA_KM2 and not confirmar_area_grande:
        raise AreaGrandeDemais(area_km2, estimar_tiles(caixa, settings))
    if area_km2 > AREA_ALERTA_KM2:
        LOGGER.warning(
            "Área de %.0f km² é grande; a varredura pode levar muitos minutos. "
            "Considere recortar em bairros.", area_km2
        )

    avisar(f"Consultando o OpenStreetMap ({area_km2:.1f} km²)", 0.08)
    overpass = OverpassClient(settings)
    elementos, stats_rede = overpass.fetch_buildings(
        caixa,
        progress=lambda feito, total: avisar(
            f"Baixando edificações: tile {feito} de {total}", 0.08 + 0.52 * (feito / max(total, 1))
        ),
    )

    aoi = None
    if restringir_ao_contorno and place is not None and place.geojson:
        aoi = polygon_from_geojson(place.geojson)
        if aoi is not None and aoi.geom_type not in {"Polygon", "MultiPolygon"}:
            # Nominatim devolve Point para muitos lugares; nesse caso não há
            # contorno para restringir e o bbox continua valendo.
            aoi = None

    contexto = None
    if usar_uso_do_solo:
        avisar("Mapeando o uso do solo da região", 0.62)
        contexto = ContextoUsoSolo(overpass.fetch_uso_do_solo(caixa))
        LOGGER.info("Áreas de uso do solo indexadas: %s", len(contexto))

    avisar(f"Qualificando {len(elementos)} edificações", 0.64)
    config = RoofFilterConfig(min_area_m2=min_area_m2)
    if rigoroso:
        config = RoofFilterConfig.rigoroso(config)
    leads, stats_filtro = build_leads(
        elementos, config, target=alvo, aoi=aoi, contexto_uso_solo=contexto
    )

    if limite:
        leads = leads[:limite]

    stats: dict[str, Any] = {
        "regiao": place.display_name if place else None,
        "bbox": list(caixa),
        "area_km2": round(area_km2, 2),
        "restrito_ao_contorno": aoi is not None,
        "areas_uso_do_solo": len(contexto) if contexto else 0,
        **stats_rede,
        **stats_filtro,
        "leads_retornados": len(leads),
    }

    if enriquecer and leads:
        avisar(f"Identificando as empresas de {len(leads)} telhados", 0.78)
        stats["enriquecimento"] = enriquecer_leads(
            leads,
            settings=settings,
            overpass=overpass,
            bbox=caixa,
            limite_endereco=limite_endereco,
        )

    avisar(f"Prospecção concluída: {len(leads)} telhados", 1.0)
    return ResultadoProspeccao(leads=leads, bbox=caixa, place=place, estatisticas=stats)


# ----------------------------------------------------------------------
# Dimensionamento e proposta
# ----------------------------------------------------------------------
@dataclass
class OpcoesDimensionamento:
    """Parâmetros que o usuário pode ajustar antes de gerar as propostas."""

    #: "coplanar" para telhado inclinado, "inclinado" para laje plana.
    montagem: str = "coplanar"
    #: Inclinação dos módulos. None usa o ótimo da latitude (laje) ou 8° (telhado).
    tilt_deg: float | None = None
    #: Alinha as fileiras com a cumeeira detectada na geometria.
    seguir_cumeeira: bool = True
    #: None usa o fator típico do segmento da edificação.
    fator_obstaculos: float | None = None
    #: Limita o sistema ao consumo estimado em vez de encher o telhado.
    limitar_ao_consumo: bool = False
    tarifa_brl_kwh: float | None = None
    capex_brl_por_kwp: float | None = None
    taxa_desconto: float = 0.10
    escalada_tarifa: float = 0.05
    anos: int = 25
    ano_conexao: int = 2026
    aplicar_lei_14300: bool = True
    eui_personalizado: float | None = None


def dimensionar_lead(
    lead: RoofLead,
    base: BaseEquipamentos,
    solar: SolarResourceClient,
    opcoes: OpcoesDimensionamento | None = None,
    emissor: DadosEmissor | None = None,
    settings: Settings | None = None,
) -> ContextoProposta | None:
    """
    Calcula o sistema completo de um telhado e monta o contexto da proposta.

    Devolve None quando não há sistema viável -- telhado pequeno demais ou sem
    combinação módulo/inversor compatível no catálogo.
    """
    settings = settings or get_settings()
    opcoes = opcoes or OpcoesDimensionamento()

    # 1) Recurso solar na localidade, com a orientação ótima para a latitude.
    if opcoes.tilt_deg is not None:
        tilt = opcoes.tilt_deg
    elif opcoes.montagem == "coplanar":
        # Telhado industrial brasileiro é quase plano; a inclinação real é a
        # da própria cobertura, não a ótima da latitude.
        tilt = 8.0
    else:
        tilt = inclinacao_otima(lead.centroid_lat)

    perfil = solar.perfil(lead.centroid_lat, lead.centroid_lon, tilt_deg=tilt)

    # 2) Layout de módulos sobre o polígono real.
    azimute_fileiras = 90.0
    if opcoes.seguir_cumeeira and lead.metrics.ridge_azimuth_deg is not None:
        azimute_fileiras = lead.metrics.ridge_azimuth_deg
    fator = (
        opcoes.fator_obstaculos
        if opcoes.fator_obstaculos is not None
        else fator_obstaculos_para(lead.tipo_para_calculo)
    )

    quantidades: dict[str, int] = {}
    layouts = {}
    for modulo in base.modulos:
        layout = calcular_layout(
            lead.geometry_utm,
            modulo,
            montagem=opcoes.montagem,  # type: ignore[arg-type]
            tilt_deg=tilt,
            azimute_fileiras_deg=azimute_fileiras,
            utm_epsg=lead.utm_epsg,
            defaults=settings.solar,
            fator_obstaculos=fator,
        )
        quantidades[modulo.modelo] = layout.quantidade
        layouts[modulo.modelo] = layout

    # 3) Consumo estimado, que também define a tarifa de referência.
    estimativa = consumption.estimar_de_lead(lead, eui_personalizado=opcoes.eui_personalizado)

    # 4) Arranjo elétrico.
    if opcoes.limitar_ao_consumo and estimativa.consumo_anual_kwh > 0:
        # Não faz sentido cobrir o telhado inteiro se o cliente não consome tanto.
        kwp_alvo = estimativa.consumo_anual_kwh / max(perfil.anual_kwh_por_kwp, 1.0)
        for modelo, quantidade in list(quantidades.items()):
            modulo = base.modulo_por_modelo(modelo)
            if modulo is None:
                continue
            teto = int(kwp_alvo * 1000.0 / modulo.potencia_wp) + 1
            quantidades[modelo] = min(quantidade, teto)

    sistema = melhor_sistema(quantidades, base, defaults=settings.solar)
    if sistema is None or sistema.modulos_totais == 0:
        LOGGER.info("Telhado %s sem sistema viável no catálogo atual.", lead.lead_id)
        return None

    layout = layouts[sistema.modulo.modelo]
    sistema.geracao_anual_kwh = perfil.anual(sistema.potencia_cc_kwp)
    sistema.geracao_mensal_kwh = perfil.mensal(sistema.potencia_cc_kwp)

    # 5) Economia.
    tarifa = opcoes.tarifa_brl_kwh
    if tarifa is None:
        tarifa, _grupo = consumption.tarifa_referencia(
            lead.tipo_para_calculo, estimativa.consumo_mensal_kwh
        )

    capex = (
        sistema.potencia_cc_kwp * opcoes.capex_brl_por_kwp
        if opcoes.capex_brl_por_kwp
        else financials.estimar_capex(sistema.potencia_cc_kwp, settings.solar)
    )

    premissas = financials.PremissasEconomicas(
        tarifa_brl_kwh=tarifa,
        capex_brl=capex,
        fracao_autoconsumo=estimativa.fracao_autoconsumo,
        degradacao_anual=settings.solar.degradation_per_year,
        escalada_tarifa=opcoes.escalada_tarifa,
        taxa_desconto=opcoes.taxa_desconto,
        anos=opcoes.anos,
        ano_conexao=opcoes.ano_conexao,
        aplicar_lei_14300=opcoes.aplicar_lei_14300,
        consumo_anual_kwh=estimativa.consumo_anual_kwh,
    )
    economia = financials.calcular(sistema.geracao_anual_kwh, premissas, settings.solar)

    return ContextoProposta(
        lead=lead,
        perfil_solar=perfil,
        layout=layout,
        sistema=sistema,
        consumo=estimativa,
        economia=economia,
        emissor=emissor or DadosEmissor(),
    )


def dimensionar_lote(
    leads: Sequence[RoofLead],
    opcoes: OpcoesDimensionamento | None = None,
    emissor: DadosEmissor | None = None,
    settings: Settings | None = None,
    base: BaseEquipamentos | None = None,
    progresso: Callable[[str, float], None] | None = None,
) -> tuple[list[ContextoProposta], list[dict[str, str]]]:
    """
    Dimensiona vários telhados. Devolve (contextos, falhas).

    Uma falha num telhado não derruba o lote: em prospecção, alguns polígonos
    sempre serão pequenos ou estranhos demais, e o restante da lista continua
    valendo.
    """
    settings = settings or get_settings()
    base = base or carregar_base(settings=settings)
    solar = SolarResourceClient(settings)

    contextos: list[ContextoProposta] = []
    falhas: list[dict[str, str]] = []
    total = max(len(leads), 1)

    for indice, lead in enumerate(leads, start=1):
        if progresso:
            progresso(f"Dimensionando {lead.name[:40]} ({indice}/{len(leads)})", indice / total)
        try:
            contexto = dimensionar_lead(lead, base, solar, opcoes, emissor, settings)
        except Exception as exc:  # noqa: BLE001 - um telhado ruim não pode derrubar o lote
            LOGGER.exception("Falha ao dimensionar %s", lead.lead_id)
            falhas.append({"lead_id": lead.lead_id, "nome": lead.name, "erro": str(exc)[:300]})
            continue
        if contexto is None:
            falhas.append({
                "lead_id": lead.lead_id,
                "nome": lead.name,
                "erro": "nenhum arranjo viável com o catálogo de equipamentos atual",
            })
            continue
        contextos.append(contexto)

    return contextos, falhas


# ----------------------------------------------------------------------
# Saídas
# ----------------------------------------------------------------------
def escrever_indice(
    contextos: Sequence[ContextoProposta],
    diretorio: Path | str,
    falhas: Sequence[dict[str, str]] = (),
) -> dict[str, Path]:
    """Grava o índice consolidado do lote em CSV e Markdown."""
    import csv

    destino = Path(diretorio)
    destino.mkdir(parents=True, exist_ok=True)
    registros = [ctx.resumo() for ctx in contextos]

    caminho_csv = destino / "indice.csv"
    if registros:
        with caminho_csv.open("w", newline="", encoding="utf-8-sig") as arquivo:
            escritor = csv.DictWriter(arquivo, fieldnames=list(registros[0].keys()))
            escritor.writeheader()
            escritor.writerows(registros)

    linhas = [
        "# Propostas geradas",
        "",
        f"Total de propostas: **{len(registros)}**",
        "",
        "| Referência | Cliente | Segmento | Área (m²) | Potência (kWp) | Geração (kWh/ano) "
        "| Investimento (R$) | Payback (anos) | Telefone | Site |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for r in registros:
        linhas.append(
            f"| {r['referencia']} | {r['cliente']} | {r['tipo_edificacao']} "
            f"| {r['area_telhado_m2']:,.0f} | {r['potencia_kwp']:,.1f} | {r['geracao_anual_kwh']:,.0f} "
            f"| {r['capex_brl']:,.2f} | {r['payback_anos'] if r['payback_anos'] else '—'} "
            f"| {r['telefone'] or '—'} | {r['site'] or '—'} |"
        )

    if falhas:
        linhas += ["", "## Telhados sem proposta", ""]
        linhas += [f"- **{f['nome']}** ({f['lead_id']}): {f['erro']}" for f in falhas]

    caminho_md = destino / "indice.md"
    caminho_md.write_text("\n".join(linhas) + "\n", encoding="utf-8")

    return {"csv": caminho_csv, "md": caminho_md}


# ----------------------------------------------------------------------
# Seleção: separar o que foi prospectado do que vai virar estudo
# ----------------------------------------------------------------------
#: Valores aceitos na coluna de seleção da planilha de leads.
MARCAS_DE_SELECAO = {"x", "s", "sim", "1", "true", "v", "ok", "y", "yes"}


def carregar_leads(caminho: Path | str) -> list[RoofLead]:
    """
    Recarrega os leads de um ``leads.geojson`` gravado pela prospecção.

    Permite olhar a lista, decidir com calma e só então calcular -- sem
    consultar o OpenStreetMap outra vez.
    """
    arquivo = Path(caminho)
    if arquivo.is_dir():
        arquivo = arquivo / "leads.geojson"
    if not arquivo.is_file():
        raise FileNotFoundError(f"Lista de telhados não encontrada: {arquivo}")

    dados = json.loads(arquivo.read_text(encoding="utf-8"))
    leads: list[RoofLead] = []
    for feature in dados.get("features", []):
        lead = RoofLead.from_geojson_feature(feature)
        if lead is not None:
            leads.append(lead)
    return leads


def ler_selecao_csv(caminho: Path | str, coluna: str = "selecionar") -> set[str]:
    """
    Lê quais telhados foram marcados na planilha de leads.

    O usuário abre o CSV, escreve ``x`` na coluna de seleção das linhas que
    interessam, salva, e só esses viram estudo. Devolve os ``lead_id``
    marcados; conjunto vazio significa que ninguém foi marcado.
    """
    import csv

    arquivo = Path(caminho)
    if arquivo.is_dir():
        arquivo = arquivo / "leads.csv"
    if not arquivo.is_file():
        raise FileNotFoundError(f"Planilha de leads não encontrada: {arquivo}")

    marcados: set[str] = set()
    # utf-8-sig porque o Excel grava BOM ao salvar o arquivo de volta.
    with arquivo.open("r", encoding="utf-8-sig", newline="") as handle:
        leitor = csv.DictReader(handle)
        if leitor.fieldnames is None or coluna not in leitor.fieldnames:
            raise ValueError(
                f"A planilha {arquivo.name} não tem a coluna '{coluna}'. "
                "Gere a lista com 'aurum prospectar --saida <pasta>'."
            )
        for linha in leitor:
            marca = str(linha.get(coluna) or "").strip().lower()
            if marca in MARCAS_DE_SELECAO:
                identificador = str(linha.get("lead_id") or "").strip()
                if identificador:
                    marcados.add(identificador)
    return marcados


def filtrar_selecionados(leads: Sequence[RoofLead], identificadores: Iterable[str]) -> list[RoofLead]:
    """Mantém apenas os leads cujos identificadores foram escolhidos."""
    escolhidos = {str(i).strip() for i in identificadores if str(i).strip()}
    return [lead for lead in leads if lead.lead_id in escolhidos]


def escrever_leads_csv(leads: Sequence[RoofLead], caminho: Path | str) -> Path:
    """
    Grava a planilha de leads com a coluna de seleção em branco na frente.

    A coluna vazia é o convite: o arquivo sai pronto para ser aberto,
    marcado e devolvido ao comando de proposta.
    """
    import csv

    destino = Path(caminho)
    destino.parent.mkdir(parents=True, exist_ok=True)
    registros = [{"selecionar": "", **lead.to_record()} for lead in leads]
    if not registros:
        destino.write_text("selecionar,lead_id\n", encoding="utf-8-sig")
        return destino

    with destino.open("w", newline="", encoding="utf-8-sig") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=list(registros[0].keys()))
        escritor.writeheader()
        escritor.writerows(registros)
    return destino


def escrever_geojson(leads: Sequence[RoofLead], caminho: Path | str) -> Path:
    """Grava as geometrias dos telhados em GeoJSON (QGIS, geojson.io)."""
    destino = Path(caminho)
    destino.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "type": "FeatureCollection",
        "features": [lead.to_geojson_feature() for lead in leads],
    }
    destino.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return destino


def executar(
    regiao: str | None = None,
    bbox: str | Sequence[float] | None = None,
    min_area_m2: float = 500.0,
    alvo: str | Sequence[str] = "todos",
    top: int | None = None,
    ordenar_por: str = "score",
    diretorio_saida: Path | str | None = None,
    opcoes: OpcoesDimensionamento | None = None,
    emissor: DadosEmissor | None = None,
    compilar_pdf: bool = True,
    settings: Settings | None = None,
    progresso: Callable[[str, float], None] | None = None,
    selecao: Iterable[str] | None = None,
) -> dict[str, Any]:
    """
    Fluxo completo, não interativo: região -> propostas em disco.

    A seleção precisa ser declarada, de um jeito ou de outro: ``selecao`` com
    os identificadores desejados, ou ``top`` aceitando explicitamente os N
    melhores. Sem nenhum dos dois a função não calcula nada -- dimensionar
    todos os telhados de uma região porque ninguém disse quantos queria é
    trabalho jogado fora, e em região grande é muito trabalho.

    ``ordenar_por`` aceita "score" (qualidade geral do lead) ou "area" (os
    maiores telhados, que é o critério pedido quando o objetivo é volume).
    """
    if selecao is None and top is None:
        raise ValueError(
            "Informe quais telhados calcular: 'selecao' com os identificadores "
            "escolhidos, ou 'top=N' para aceitar os N melhores automaticamente. "
            "Para só listar a região sem calcular nada, use prospectar()."
        )
    from .proposal.render import escrever_proposta, montar_zip

    settings = settings or get_settings()
    destino = Path(diretorio_saida) if diretorio_saida else settings.output_dir
    destino.mkdir(parents=True, exist_ok=True)

    resultado = prospectar(
        regiao=regiao,
        bbox=bbox,
        min_area_m2=min_area_m2,
        alvo=alvo,
        settings=settings,
        progresso=progresso,
    )

    if selecao is not None:
        selecionados = filtrar_selecionados(resultado.leads, selecao)
        if not selecionados:
            raise ValueError("Nenhum dos identificadores selecionados foi encontrado na região.")
    else:
        selecionados = (
            resultado.por_area(top) if ordenar_por == "area" else resultado.top(top)
        )

    contextos, falhas = dimensionar_lote(
        selecionados, opcoes=opcoes, emissor=emissor, settings=settings, progresso=progresso
    )

    arquivos: list[dict[str, Any]] = []
    for ctx in contextos:
        pasta = destino / f"{ctx.referencia}"
        gerados = escrever_proposta(ctx, pasta, compilar=compilar_pdf)
        arquivos.append({k: (str(v) if v else None) for k, v in gerados.items()})

    indice = escrever_indice(contextos, destino, falhas)
    geojson = escrever_geojson(resultado.leads, destino / "leads.geojson")
    (destino / "prospeccao.json").write_text(
        json.dumps(resultado.estatisticas, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    zip_final = montar_zip(destino)

    return {
        "diretorio": str(destino),
        "leads_encontrados": resultado.total,
        "propostas_geradas": len(contextos),
        "falhas": falhas,
        "estatisticas": resultado.estatisticas,
        "indice_csv": str(indice["csv"]),
        "indice_md": str(indice["md"]),
        "geojson": str(geojson),
        "zip": str(zip_final),
        "arquivos": arquivos,
    }
