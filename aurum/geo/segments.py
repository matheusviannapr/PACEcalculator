"""
Segmentos de prospecção: que tipo de lugar é aquele telhado.

Substitui os quatro grupos grossos anteriores (industrial, comercial,
institucional, misto) por uma taxonomia que corresponde a como a prospecção
comercial de energia solar realmente é organizada no Brasil. Agro é um
segmento próprio, e não um apêndice da indústria; escola não é a mesma coisa
que hospital; condomínio residencial tem lógica de decisão inteiramente
diferente de galpão logístico.

Cada segmento declara as tags do OpenStreetMap que o identificam. A
classificação percorre os segmentos em ordem de especificidade: um
``building=warehouse`` num ``landuse=farmyard`` é agro, não logística, porque
agro é avaliado antes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

#: Marca que aceita qualquer valor para a chave, bastando ela existir.
QUALQUER = "*"


@dataclass(frozen=True)
class Segmento:
    """Um segmento de prospecção e as tags do OSM que o identificam."""

    chave: str
    rotulo: str
    descricao: str
    #: chave do OSM -> valores aceitos (ou {QUALQUER} para qualquer valor).
    tags: dict[str, set[str]] = field(default_factory=dict)
    #: Peso de 0 a 20 na pontuação do lead, refletindo o quanto o segmento
    #: costuma ser bom cliente de geração solar: consumo alto, diurno e com
    #: um decisor único.
    peso_comercial: float = 10.0
    #: Valores de ``landuse`` que reforçam o segmento quando a edificação em
    #: si não tem tag útil (o famoso ``building=yes``).
    landuse_contexto: set[str] = field(default_factory=set)
    #: Tipo de edificação representativo, usado para escolher faixa de
    #: consumo, fração de autoconsumo e fator de obstáculos quando o prédio
    #: não declara o próprio tipo. Sem isso, um galpão de fazenda reconhecido
    #: como agro pelo uso do solo seria dimensionado com os parâmetros
    #: genéricos de um prédio urbano qualquer.
    tipo_referencia: str = ""

    def combina(self, tags: dict[str, Any]) -> bool:
        """True se as tags da edificação identificam este segmento."""
        for chave, aceitos in self.tags.items():
            valor = tags.get(chave)
            if valor is None:
                continue
            if QUALQUER in aceitos:
                return True
            if str(valor).strip().lower() in aceitos:
                return True
        return False


#: Ordem importa: do mais específico para o mais genérico.
SEGMENTOS: tuple[Segmento, ...] = (
    Segmento(
        chave="agro",
        tipo_referencia="farm",
        rotulo="Agro e agroindústria",
        descricao="Granjas, aviários, silos, armazéns rurais, laticínios e frigoríficos",
        tags={
            "building": {
                "barn", "cowshed", "farm", "farm_auxiliary", "greenhouse", "silo",
                "stable", "sty", "slurry_tank", "livestock", "chicken_coop", "grain_silo",
            },
            "man_made": {"silo", "grain_silo", "storage_tank"},
            "landuse": {"farmyard", "greenhouse_horticulture", "orchard", "vineyard", "aquaculture"},
            "industrial": {"slaughterhouse", "dairy", "grain", "sawmill", "mill", "feed"},
            "craft": {"agricultural_engines", "winery", "distillery"},
            "product": {"milk", "grain", "feed", "poultry"},
        },
        # Consumo alto e diurno (irrigação, ordenha, ventilação de aviário,
        # câmara fria) e decisor único. É o melhor cliente da lista.
        peso_comercial=20.0,
        landuse_contexto={"farmyard", "greenhouse_horticulture", "orchard", "vineyard", "aquaculture"},
    ),
    Segmento(
        chave="frigorifico",
        tipo_referencia="cold_storage",
        rotulo="Frigorífico e câmara fria",
        descricao="Frigoríficos, câmaras frias e centrais de resfriamento",
        tags={
            "building": {"cold_storage", "refrigeration"},
            "industrial": {"cold_storage", "refrigeration", "meat", "abattoir"},
        },
        # Refrigeração roda 24 h; o autoconsumo diurno é quase total.
        peso_comercial=20.0,
    ),
    Segmento(
        chave="supermercado",
        tipo_referencia="supermarket",
        rotulo="Supermercado e atacado",
        descricao="Supermercados, atacarejos e centrais de abastecimento",
        tags={
            "building": {"supermarket"},
            "shop": {"supermarket", "wholesale", "department_store", "hypermarket"},
            "amenity": {"marketplace"},
        },
        peso_comercial=19.0,
    ),
    Segmento(
        chave="saude",
        tipo_referencia="hospital",
        rotulo="Saúde",
        descricao="Hospitais, clínicas, laboratórios e centros de diagnóstico",
        tags={
            "building": {"hospital", "clinic", "medical", "health_post"},
            "amenity": {"hospital", "clinic", "doctors", "veterinary", "laboratory"},
            "healthcare": {QUALQUER},
        },
        peso_comercial=18.0,
    ),
    Segmento(
        chave="educacao",
        tipo_referencia="school",
        rotulo="Escolas e universidades",
        descricao="Escolas, creches, faculdades, universidades e centros de formação",
        tags={
            "building": {
                "school", "university", "college", "kindergarten", "educational",
                "dormitory", "schoolhouse",
            },
            "amenity": {
                "school", "university", "college", "kindergarten", "childcare",
                "language_school", "driving_school", "music_school", "library",
                "research_institute",
            },
            "landuse": {"education"},
        },
        # Consumo diurno alinhado à geração, mas a decisão passa por conselho
        # ou licitação, o que alonga o ciclo comercial.
        peso_comercial=16.0,
        landuse_contexto={"education"},
    ),
    Segmento(
        chave="industrial",
        tipo_referencia="industrial",
        rotulo="Indústria",
        descricao="Fábricas, metalúrgicas, plantas de manufatura e processo",
        tags={
            "building": {"industrial", "factory", "manufacture", "works", "hangar"},
            "landuse": {"industrial"},
            "industrial": {QUALQUER},
            "man_made": {"works", "wastewater_plant", "water_works", "pumping_station"},
            "craft": {QUALQUER},
        },
        peso_comercial=20.0,
        landuse_contexto={"industrial"},
    ),
    Segmento(
        chave="logistica",
        tipo_referencia="warehouse",
        rotulo="Logística e armazenagem",
        descricao="Galpões, centros de distribuição e transportadoras",
        tags={
            "building": {"warehouse", "storage", "distribution", "depot", "transportation"},
            "landuse": {"logistics", "depot"},
            "amenity": {"parcel_locker"},
        },
        # Telhado enorme, mas consumo por metro quadrado baixo: cabe muito
        # sistema, e boa parte da energia vai para a rede.
        peso_comercial=16.0,
    ),
    Segmento(
        chave="comercio",
        tipo_referencia="retail",
        rotulo="Comércio e varejo",
        descricao="Lojas, shoppings, concessionárias e centros comerciais",
        tags={
            "building": {"commercial", "retail", "mall", "kiosk", "shop"},
            "shop": {QUALQUER},
            "landuse": {"retail", "commercial"},
            "amenity": {"fuel", "restaurant", "fast_food", "cafe", "bar", "pharmacy", "bank"},
        },
        peso_comercial=17.0,
        landuse_contexto={"retail", "commercial"},
    ),
    Segmento(
        chave="escritorio",
        tipo_referencia="office",
        rotulo="Escritórios e corporativo",
        descricao="Edifícios de escritórios, sedes administrativas e coworkings",
        tags={
            "building": {"office", "corporate", "commercial_office"},
            "office": {QUALQUER},
        },
        # Prédio alto consome muito mais do que o telhado consegue gerar.
        peso_comercial=15.0,
    ),
    Segmento(
        chave="hotelaria",
        tipo_referencia="hotel",
        rotulo="Hotelaria",
        descricao="Hotéis, pousadas, resorts e motéis",
        tags={
            "building": {"hotel", "motel", "hostel", "resort", "guest_house"},
            "tourism": {"hotel", "motel", "hostel", "resort", "guest_house", "apartment", "chalet"},
        },
        # Boa parte do consumo é noturno, o que reduz o autoconsumo.
        peso_comercial=13.0,
    ),
    Segmento(
        chave="condominio",
        tipo_referencia="apartments",
        rotulo="Condomínios residenciais",
        descricao="Prédios de apartamentos e condomínios com área comum",
        tags={
            "building": {"apartments", "residential", "condominium", "flats"},
            "residential": {QUALQUER},
            "landuse": {"residential"},
        },
        # A geração normalmente atende só a área comum, e a decisão passa por
        # assembleia de condôminos. Ciclo longo, ticket menor.
        peso_comercial=11.0,
    ),
    Segmento(
        chave="publico",
        tipo_referencia="public",
        rotulo="Público e institucional",
        descricao="Prefeituras, órgãos públicos, quartéis e equipamentos urbanos",
        tags={
            "building": {
                "civic", "public", "government", "government_office", "fire_station",
                "police", "prison", "courthouse", "townhall",
            },
            "amenity": {
                "townhall", "courthouse", "police", "fire_station", "prison",
                "post_office", "community_centre", "social_facility",
            },
            "office": {"government"},
        },
        # Licitação: ciclo longo, mas contrato grande e pagamento seguro.
        peso_comercial=14.0,
    ),
    Segmento(
        chave="esporte_lazer",
        tipo_referencia="sports_centre",
        rotulo="Esporte e lazer",
        descricao="Ginásios, clubes, estádios, academias e parques aquáticos",
        tags={
            "building": {"sports_hall", "sports_centre", "stadium", "grandstand", "pavilion"},
            "leisure": {
                "sports_centre", "sports_hall", "stadium", "fitness_centre",
                "swimming_pool", "water_park", "golf_course", "pitch",
            },
        },
        peso_comercial=12.0,
    ),
    Segmento(
        chave="religioso",
        tipo_referencia="church",
        rotulo="Templos e igrejas",
        descricao="Igrejas, templos, salões paroquiais e centros religiosos",
        tags={
            "building": {
                "church", "cathedral", "chapel", "mosque", "synagogue", "temple",
                "religious", "monastery", "shrine",
            },
            "amenity": {"place_of_worship", "monastery"},
            "religion": {QUALQUER},
        },
        # Uso concentrado à noite e no fim de semana: autoconsumo baixo.
        peso_comercial=8.0,
    ),
)

#: Acesso por chave.
POR_CHAVE: dict[str, Segmento] = {s.chave: s for s in SEGMENTOS}

#: Segmento atribuído quando nenhuma tag identifica a edificação. No OSM
#: brasileiro é o caso mais comum: muito ``building=yes`` sem mais nada.
SEGMENTO_DESCONHECIDO = Segmento(
    chave="desconhecido",
    rotulo="Não identificado",
    descricao="Edificação sem tipo declarado no OpenStreetMap",
    peso_comercial=9.0,
    tipo_referencia="",
)

#: Aliases dos nomes antigos, para não quebrar scripts e comandos existentes.
ALIASES: dict[str, tuple[str, ...]] = {
    "todos": (),
    "misto": (),
    "comercial": ("comercio", "supermercado", "escritorio", "hotelaria"),
    "institucional": ("educacao", "saude", "publico", "religioso", "esporte_lazer"),
}


def rotulos() -> dict[str, str]:
    """Mapa chave -> rótulo legível, para montar seletores na interface."""
    return {s.chave: s.rotulo for s in SEGMENTOS}


def descricoes() -> dict[str, str]:
    return {s.chave: s.descricao for s in SEGMENTOS}


def classificar(tags: dict[str, Any], landuse_contexto: str | None = None) -> Segmento:
    """
    Determina o segmento de uma edificação a partir das suas tags.

    ``landuse_contexto`` é o uso do solo da área que contém a edificação,
    quando conhecido. Serve justamente para os ``building=yes``: um galpão sem
    tag dentro de um ``landuse=farmyard`` é agro, e sem esse contexto ficaria
    para sempre como "não identificado".
    """
    for segmento in SEGMENTOS:
        if segmento.combina(tags):
            return segmento

    if landuse_contexto:
        alvo = str(landuse_contexto).strip().lower()
        for segmento in SEGMENTOS:
            if alvo in segmento.landuse_contexto:
                return segmento

    return SEGMENTO_DESCONHECIDO


def normalizar_alvos(alvo: str | Sequence[str] | None) -> set[str]:
    """
    Interpreta o parâmetro de filtro e devolve o conjunto de chaves de segmento.

    Aceita uma string, uma lista, os nomes antigos de grupo e o valor
    ``"todos"``. Conjunto vazio significa "sem filtro".
    """
    if alvo is None:
        return set()
    if isinstance(alvo, str):
        entradas = [p.strip() for p in alvo.split(",") if p.strip()]
    else:
        entradas = [str(p).strip() for p in alvo if str(p).strip()]

    chaves: set[str] = set()
    for entrada in entradas:
        minusculo = entrada.lower()
        if minusculo in {"todos", "all", "misto", ""}:
            return set()
        if minusculo in ALIASES:
            chaves.update(ALIASES[minusculo])
        elif minusculo in POR_CHAVE:
            chaves.add(minusculo)
        elif minusculo == "desconhecido":
            chaves.add("desconhecido")
    return chaves


def pertence(
    tags: dict[str, Any],
    alvos: set[str],
    landuse_contexto: str | None = None,
) -> bool:
    """True se a edificação pertence a algum dos segmentos pedidos."""
    if not alvos:
        return True
    return classificar(tags, landuse_contexto).chave in alvos


def peso_comercial(chave: str) -> float:
    """Peso de 0 a 20 do segmento na pontuação do lead."""
    if chave == SEGMENTO_DESCONHECIDO.chave:
        return SEGMENTO_DESCONHECIDO.peso_comercial
    segmento = POR_CHAVE.get(chave)
    return segmento.peso_comercial if segmento else SEGMENTO_DESCONHECIDO.peso_comercial


def chaves_validas() -> list[str]:
    """Chaves aceitas pelo filtro, incluindo os aliases antigos."""
    return ["todos", *POR_CHAVE.keys(), "desconhecido", *ALIASES.keys()]


#: Tipos de edificação genéricos demais para escolher parâmetros de cálculo.
TIPOS_GENERICOS = {"yes", "true", "1", "desconhecido", "", "building", "roof"}


def tipo_para_calculo(building_type: str | None, chave_segmento: str | None) -> str:
    """
    Tipo a usar nas tabelas de consumo, autoconsumo e obstáculos.

    Prefere o tipo declarado no próprio prédio, que é mais específico. Quando
    ele é genérico (``building=yes``, a maioria no Brasil), recorre ao tipo
    representativo do segmento -- é o que faz um galpão reconhecido como agro
    pelo uso do solo ser dimensionado como galpão rural, e não como prédio
    urbano qualquer.
    """
    tipo = (building_type or "").strip().lower()
    if tipo and tipo not in TIPOS_GENERICOS:
        return tipo
    segmento = POR_CHAVE.get(str(chave_segmento or "").strip().lower())
    return (segmento.tipo_referencia if segmento else "") or tipo or "desconhecido"
