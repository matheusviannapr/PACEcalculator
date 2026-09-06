"""
Layout de módulos sobre o polígono real do telhado.

Substitui a heurística anterior -- ``area * fator_ocupacao / area_modulo`` --
por um empacotamento em grade sobre a geometria de verdade. A diferença
importa: um telhado em L de 2.000 m2 aceita bem menos módulos que um retângulo
de 2.000 m2, e a heurística antiga dava o mesmo número para os dois.

O procedimento:

1. Recuar a borda do telhado (afastamento de manutenção / NBR 5419).
2. Rotacionar o polígono de modo que as fileiras fiquem alinhadas ao eixo
   escolhido (cumeeira, para telhado inclinado; leste-oeste, para laje plana).
3. Varrer uma grade de posições de módulo e manter as que cabem inteiras.
4. Rotacionar as posições de volta e devolver o layout.

Dois tipos de montagem:

* **coplanar** -- telhado inclinado, módulos rentes à cobertura. Não há
  sombreamento entre fileiras, então o passo é a própria dimensão do módulo
  mais uma folga.
* **inclinado** -- laje plana com estrutura triangular. As fileiras precisam
  de espaçamento para não se sombrearem no inverno, o que é parametrizado
  pela razão de ocupação do solo (GCR).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import shapely
from shapely.affinity import rotate
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

from ..config import SolarDefaults, get_settings
from .equipment import Modulo

TipoMontagem = Literal["coplanar", "inclinado"]

#: Acima deste número de posições candidatas, a varredura exata fica cara e o
#: resultado deixa de ser mais informativo que a estimativa por área. Telhados
#: assim (>~50 mil módulos) são usinas, não geração distribuída em telhado.
MAX_POSICOES_VARREDURA = 400_000

#: Fração do telhado que permanece utilizável depois de descontar o que o OSM
#: não mapeia: casas de máquinas, climatização, exaustores, claraboias,
#: corredores de manutenção e juntas de dilatação. O polígono do OSM é a
#: projeção do prédio, não uma planta de cobertura -- sem este desconto o
#: empacotamento geométrico superestima a capacidade real. Só é confirmável
#: em visita técnica ou imagem de satélite de alta resolução.
FATOR_OBSTACULOS_PADRAO: dict[str, float] = {
    "industrial": 0.85,
    "warehouse": 0.88,
    "commercial": 0.82,
    "retail": 0.82,
    "supermarket": 0.80,
    "office": 0.75,
    "school": 0.85,
    "hospital": 0.70,
    "hotel": 0.75,
    # Telhado rural é limpo: pouca casa de máquinas e nenhuma climatização
    # central. Estufa é o oposto -- a cobertura precisa passar luz.
    "barn": 0.90,
    "cowshed": 0.88,
    "farm": 0.88,
    "farm_auxiliary": 0.90,
    "silo": 0.75,
    "stable": 0.88,
    "sty": 0.88,
    "chicken_coop": 0.86,
    "greenhouse": 0.35,
    "slaughterhouse": 0.78,
    "cold_storage": 0.80,
    "factory": 0.83,
    "mall": 0.78,
    "apartments": 0.60,
    "residential": 0.60,
    "church": 0.85,
    # Prédio público e institucional: laje com caixa d'água, casa de máquinas
    # e antenas, mas sem a densidade de climatização de um shopping.
    "public": 0.78,
    "civic": 0.78,
    "government": 0.75,
    "sports_centre": 0.80,
    "sports_hall": 0.82,
    "stadium": 0.70,
    "clinic": 0.72,
    "university": 0.80,
    "college": 0.82,
    "kindergarten": 0.85,
    "dormitory": 0.70,
    "logistics": 0.88,
    "commercial_office": 0.75,
    "_padrao": 0.82,
}


def fator_obstaculos_para(tipo_edificacao: str | None) -> float:
    """Fator de obstáculos de cobertura típico para um tipo de edificação."""
    if not tipo_edificacao:
        return FATOR_OBSTACULOS_PADRAO["_padrao"]
    return FATOR_OBSTACULOS_PADRAO.get(str(tipo_edificacao).lower(), FATOR_OBSTACULOS_PADRAO["_padrao"])


@dataclass
class LayoutModulos:
    """Resultado do empacotamento de módulos num telhado."""

    modulo: Modulo
    quantidade: int
    orientacao_modulo: str            # "retrato" ou "paisagem"
    montagem: TipoMontagem
    azimute_fileiras_deg: float
    #: Área do telhado depois do recuo de borda.
    area_util_m2: float
    area_bruta_m2: float
    #: Área efetivamente coberta por módulos.
    area_ocupada_m2: float
    passo_fileira_m: float
    #: Polígonos dos módulos em UTM, para desenhar o croqui.
    modulos_geom: list[Polygon] = field(default_factory=list, repr=False)
    utm_epsg: int | None = None
    metodo: str = "grade"
    avisos: list[str] = field(default_factory=list)
    #: Quantidade puramente geométrica, antes do desconto de obstáculos.
    quantidade_geometrica: int = 0
    #: Fator aplicado para descontar obstáculos não mapeados no OSM.
    fator_obstaculos: float = 1.0

    @property
    def potencia_kwp(self) -> float:
        return self.quantidade * self.modulo.potencia_wp / 1000.0

    @property
    def taxa_ocupacao(self) -> float:
        """Fração da área bruta do telhado ocupada por módulos."""
        return self.area_ocupada_m2 / self.area_bruta_m2 if self.area_bruta_m2 > 0 else 0.0

    @property
    def densidade_wp_m2(self) -> float:
        """Watt-pico instalado por m2 de telhado bruto."""
        return (self.quantidade * self.modulo.potencia_wp / self.area_bruta_m2) if self.area_bruta_m2 > 0 else 0.0

    def as_dict(self) -> dict:
        return {
            "modulo": str(self.modulo),
            "modulo_wp": self.modulo.potencia_wp,
            "modulo_dimensoes_m": [self.modulo.comprimento_m, self.modulo.largura_m],
            "quantidade": self.quantidade,
            "potencia_kwp": round(self.potencia_kwp, 2),
            "orientacao_modulo": self.orientacao_modulo,
            "montagem": self.montagem,
            "azimute_fileiras_deg": round(self.azimute_fileiras_deg, 1),
            "area_bruta_m2": round(self.area_bruta_m2, 1),
            "area_util_m2": round(self.area_util_m2, 1),
            "area_ocupada_m2": round(self.area_ocupada_m2, 1),
            "taxa_ocupacao": round(self.taxa_ocupacao, 3),
            "densidade_wp_m2": round(self.densidade_wp_m2, 1),
            "passo_fileira_m": round(self.passo_fileira_m, 3),
            "quantidade_geometrica": self.quantidade_geometrica,
            "fator_obstaculos": round(self.fator_obstaculos, 3),
            "metodo": self.metodo,
            "avisos": list(self.avisos),
        }


def passo_entre_fileiras(
    modulo: Modulo,
    montagem: TipoMontagem,
    tilt_deg: float,
    gcr: float,
    gap_m: float,
    orientacao: str,
) -> tuple[float, float]:
    """
    Passo (centro a centro) nas direções transversal e longitudinal à fileira.

    Devolve (passo_profundidade, passo_lateral), em metros. "Profundidade" é a
    direção em que as fileiras se sucedem -- a que sofre sombreamento mútuo.
    """
    if orientacao == "retrato":
        profundidade_modulo = modulo.comprimento_m
        largura_modulo = modulo.largura_m
    else:
        profundidade_modulo = modulo.largura_m
        largura_modulo = modulo.comprimento_m

    if montagem == "coplanar":
        # Rente ao telhado: sem sombreamento entre fileiras.
        return profundidade_modulo + gap_m, largura_modulo + gap_m

    # Estrutura inclinada em laje: a projeção horizontal do módulo encolhe com
    # o cosseno da inclinação, e o passo sai da razão de ocupação do solo.
    projecao = profundidade_modulo * math.cos(math.radians(max(0.0, tilt_deg)))
    gcr_seguro = min(max(gcr, 0.15), 0.95)
    return projecao / gcr_seguro, largura_modulo + gap_m


def _recuar(poligono: BaseGeometry, recuo_m: float) -> BaseGeometry | None:
    """Aplica o afastamento de borda; None se o telhado sumir."""
    if recuo_m <= 0:
        return poligono
    reduzido = poligono.buffer(-abs(recuo_m))
    if reduzido.is_empty:
        return None
    return reduzido


def _partes_poligonais(geom: BaseGeometry) -> list[Polygon]:
    if geom.geom_type == "Polygon":
        return [geom]
    if geom.geom_type == "MultiPolygon":
        return list(geom.geoms)
    if geom.geom_type == "GeometryCollection":
        return [g for g in geom.geoms if g.geom_type == "Polygon"]
    return []


def _aplicar_fator_obstaculos(layout: LayoutModulos, fator: float) -> None:
    """
    Reduz a quantidade de módulos para refletir obstáculos de cobertura,
    in-place.

    Os módulos descartados saem da grade de forma espalhada (passo uniforme
    ao longo da lista), e não de um canto só: obstáculos reais aparecem
    distribuídos pelo telhado, e remover um bloco contíguo daria um croqui
    enganoso.
    """
    fator = min(max(float(fator), 0.1), 1.0)
    layout.fator_obstaculos = fator
    if fator >= 1.0 or layout.quantidade == 0:
        return

    alvo = max(1, int(layout.quantidade * fator))
    if alvo >= layout.quantidade:
        return

    if layout.modulos_geom:
        total = len(layout.modulos_geom)
        passo = total / alvo
        mantidos = [layout.modulos_geom[min(total - 1, int(i * passo))] for i in range(alvo)]
        layout.modulos_geom = mantidos

    layout.quantidade = alvo
    layout.area_ocupada_m2 = alvo * layout.modulo.area_m2
    layout.avisos.append(
        f"Aplicado fator de obstáculos de cobertura de {fator:.0%} "
        f"({layout.quantidade_geometrica} posições geométricas -> {alvo} módulos). "
        "Confirmar com imagem de satélite ou visita técnica."
    )


def _empacotar_em_parte(
    parte: Polygon,
    largura_modulo: float,
    profundidade_modulo: float,
    passo_lateral: float,
    passo_profundidade: float,
    angulo_deg: float,
    centro_rotacao,
) -> list[Polygon]:
    """
    Empacota módulos numa única parte do telhado.

    A parte é rotacionada para o eixo das fileiras ficar horizontal, a grade é
    varrida em coordenadas alinhadas, e os módulos aprovados voltam à
    orientação original.
    """
    alinhada = rotate(parte, -angulo_deg, origin=centro_rotacao, use_radians=False)
    min_x, min_y, max_x, max_y = alinhada.bounds

    largura_disponivel = max_x - min_x
    altura_disponivel = max_y - min_y
    if largura_disponivel < largura_modulo or altura_disponivel < profundidade_modulo:
        return []

    n_colunas = int((largura_disponivel - largura_modulo) / passo_lateral) + 1
    n_fileiras = int((altura_disponivel - profundidade_modulo) / passo_profundidade) + 1
    if n_colunas <= 0 or n_fileiras <= 0:
        return []
    if n_colunas * n_fileiras > MAX_POSICOES_VARREDURA:
        return []

    # Centraliza a grade na parte: sobra dividida igualmente nos dois lados.
    folga_x = (largura_disponivel - ((n_colunas - 1) * passo_lateral + largura_modulo)) / 2.0
    folga_y = (altura_disponivel - ((n_fileiras - 1) * passo_profundidade + profundidade_modulo)) / 2.0

    # Monta todas as posições candidatas de uma vez e testa a contenção em
    # lote. O laço posição a posição com `prepared.contains` custava segundos
    # num galpão grande, e a prospecção roda isso para cada modelo de módulo
    # de cada telhado do lote.
    xs = min_x + folga_x + np.arange(n_colunas) * passo_lateral
    ys = min_y + folga_y + np.arange(n_fileiras) * passo_profundidade
    grade_x, grade_y = np.meshgrid(xs, ys)
    candidatos = shapely.box(
        grade_x.ravel(),
        grade_y.ravel(),
        grade_x.ravel() + largura_modulo,
        grade_y.ravel() + profundidade_modulo,
    )

    # `contains` (e não `intersects`): o módulo precisa caber inteiro, senão o
    # croqui mostraria painéis pendurados para fora do telhado.
    dentro = shapely.contains(alinhada, candidatos)
    aceitos = candidatos[dentro]
    if aceitos.size == 0:
        return []

    return list(shapely.transform(
        aceitos,
        lambda coords: _rotacionar_coords(coords, angulo_deg, centro_rotacao),
    ))


def _rotacionar_coords(coords, angulo_deg: float, centro) -> "np.ndarray":
    """Rotaciona um bloco de coordenadas em torno de um ponto."""
    angulo = math.radians(angulo_deg)
    cos_a, sen_a = math.cos(angulo), math.sin(angulo)
    dx = coords[:, 0] - centro.x
    dy = coords[:, 1] - centro.y
    return np.column_stack((
        centro.x + dx * cos_a - dy * sen_a,
        centro.y + dx * sen_a + dy * cos_a,
    ))


def calcular_layout(
    geometria_utm: BaseGeometry,
    modulo: Modulo,
    montagem: TipoMontagem = "coplanar",
    tilt_deg: float = 20.0,
    azimute_fileiras_deg: float | None = None,
    recuo_m: float | None = None,
    gcr: float | None = None,
    gap_m: float | None = None,
    utm_epsg: int | None = None,
    defaults: SolarDefaults | None = None,
    testar_ambas_orientacoes: bool = True,
    fator_obstaculos: float = 1.0,
) -> LayoutModulos:
    """
    Empacota módulos no telhado e devolve o layout de maior potência.

    ``geometria_utm`` precisa estar projetada em metros. ``azimute_fileiras_deg``
    define a direção das fileiras: para telhado de duas águas use a cumeeira;
    para laje plana, o padrão leste-oeste (90 graus) mantém as faces ao Norte.

    ``fator_obstaculos`` desconta o que o polígono do OSM não mostra
    (climatização, claraboias, circulação). Ver :data:`FATOR_OBSTACULOS_PADRAO`.
    """
    defaults = defaults or get_settings().solar
    recuo = defaults.edge_setback_m if recuo_m is None else float(recuo_m)
    razao_ocupacao = defaults.ground_coverage_ratio if gcr is None else float(gcr)
    folga = defaults.module_gap_m if gap_m is None else float(gap_m)
    azimute = 90.0 if azimute_fileiras_deg is None else float(azimute_fileiras_deg)

    area_bruta = float(geometria_utm.area)
    avisos: list[str] = []

    util = _recuar(geometria_utm, recuo)
    if util is None or util.is_empty:
        return LayoutModulos(
            modulo=modulo,
            quantidade=0,
            orientacao_modulo="retrato",
            montagem=montagem,
            azimute_fileiras_deg=azimute,
            area_util_m2=0.0,
            area_bruta_m2=area_bruta,
            area_ocupada_m2=0.0,
            passo_fileira_m=0.0,
            utm_epsg=utm_epsg,
            metodo="grade",
            avisos=[f"O recuo de borda de {recuo:.2f} m consome todo o telhado."],
        )

    partes = _partes_poligonais(util)
    if not partes:
        avisos.append("Telhado sem parte poligonal utilizável após o recuo.")
        partes = []

    area_util = sum(p.area for p in partes)
    centro = geometria_utm.centroid

    # O azimute é medido do Norte no sentido horário; a rotação do shapely é
    # medida do eixo x no sentido anti-horário. A conversão é 90 - azimute.
    angulo_grade = 90.0 - azimute

    orientacoes = ("retrato", "paisagem") if testar_ambas_orientacoes else ("retrato",)
    melhor: LayoutModulos | None = None

    for orientacao in orientacoes:
        passo_prof, passo_lat = passo_entre_fileiras(
            modulo, montagem, tilt_deg, razao_ocupacao, folga, orientacao
        )
        if orientacao == "retrato":
            profundidade_modulo, largura_modulo = modulo.comprimento_m, modulo.largura_m
        else:
            profundidade_modulo, largura_modulo = modulo.largura_m, modulo.comprimento_m

        geoms: list[Polygon] = []
        for parte in partes:
            geoms.extend(
                _empacotar_em_parte(
                    parte,
                    largura_modulo,
                    profundidade_modulo,
                    passo_lat,
                    passo_prof,
                    angulo_grade,
                    centro,
                )
            )

        candidato = LayoutModulos(
            modulo=modulo,
            quantidade=len(geoms),
            orientacao_modulo=orientacao,
            montagem=montagem,
            azimute_fileiras_deg=azimute,
            area_util_m2=area_util,
            area_bruta_m2=area_bruta,
            area_ocupada_m2=len(geoms) * modulo.area_m2,
            passo_fileira_m=passo_prof,
            modulos_geom=geoms,
            utm_epsg=utm_epsg,
            metodo="grade",
            quantidade_geometrica=len(geoms),
        )
        if melhor is None or candidato.quantidade > melhor.quantidade:
            melhor = candidato

    assert melhor is not None
    _aplicar_fator_obstaculos(melhor, fator_obstaculos)

    if melhor.quantidade == 0 and area_util > 0:
        # A varredura exata pode falhar num telhado estreito ou muito recortado;
        # a estimativa por área evita devolver "zero módulos" para um telhado
        # que na prática comporta alguns.
        melhor = estimar_por_area(
            geometria_utm,
            modulo,
            montagem=montagem,
            tilt_deg=tilt_deg,
            recuo_m=recuo,
            gcr=razao_ocupacao,
            gap_m=folga,
            utm_epsg=utm_epsg,
            defaults=defaults,
        )
        melhor.avisos.append(
            "Empacotamento em grade não encontrou posição válida; quantidade obtida por "
            "estimativa de área. Conferir o croqui antes de propor."
        )

    melhor.avisos.extend(avisos)
    if modulo.dimensoes_derivadas:
        melhor.avisos.append(
            f"Dimensões do módulo {modulo.modelo} derivadas da eficiência declarada "
            f"({modulo.eficiencia_percent:.1f}%), não do datasheet."
        )
    return melhor


def estimar_por_area(
    geometria_utm: BaseGeometry,
    modulo: Modulo,
    montagem: TipoMontagem = "coplanar",
    tilt_deg: float = 20.0,
    recuo_m: float | None = None,
    gcr: float | None = None,
    gap_m: float | None = None,
    utm_epsg: int | None = None,
    defaults: SolarDefaults | None = None,
    fator_obstaculos: float = 1.0,
) -> LayoutModulos:
    """
    Estimativa rápida por área, sem varredura em grade.

    Usada em triagem de muitos telhados, onde o custo do empacotamento exato
    não se justifica, e como rede de segurança quando a grade falha.
    """
    defaults = defaults or get_settings().solar
    recuo = defaults.edge_setback_m if recuo_m is None else float(recuo_m)
    razao_ocupacao = defaults.ground_coverage_ratio if gcr is None else float(gcr)
    folga = defaults.module_gap_m if gap_m is None else float(gap_m)

    area_bruta = float(geometria_utm.area)
    util = _recuar(geometria_utm, recuo)
    area_util = float(util.area) if util is not None and not util.is_empty else 0.0

    passo_prof, passo_lat = passo_entre_fileiras(modulo, montagem, tilt_deg, razao_ocupacao, folga, "retrato")
    area_por_posicao = passo_prof * passo_lat
    # 0,85 desconta corredores, obstáculos de cobertura e bordas irregulares
    # que a divisão pura pela área da célula não enxerga.
    quantidade = int((area_util * 0.85) // area_por_posicao) if area_por_posicao > 0 else 0

    layout = LayoutModulos(
        modulo=modulo,
        quantidade=max(0, quantidade),
        orientacao_modulo="retrato",
        montagem=montagem,
        azimute_fileiras_deg=90.0,
        area_util_m2=area_util,
        area_bruta_m2=area_bruta,
        area_ocupada_m2=max(0, quantidade) * modulo.area_m2,
        passo_fileira_m=passo_prof,
        modulos_geom=[],
        utm_epsg=utm_epsg,
        metodo="estimativa_area",
        avisos=["Quantidade estimada pela área útil, sem verificação de encaixe geométrico."],
        quantidade_geometrica=max(0, quantidade),
    )
    _aplicar_fator_obstaculos(layout, fator_obstaculos)
    return layout


def melhor_layout(
    geometria_utm: BaseGeometry,
    modulos: list[Modulo],
    **kwargs,
) -> LayoutModulos | None:
    """
    Testa vários modelos de módulo e devolve o de maior potência instalada.

    Não é o de maior contagem: um módulo grande costuma render mais kWp mesmo
    entrando em menor número.
    """
    melhor: LayoutModulos | None = None
    for modulo in modulos:
        layout = calcular_layout(geometria_utm, modulo, **kwargs)
        if melhor is None or layout.potencia_kwp > melhor.potencia_kwp:
            melhor = layout
    return melhor
