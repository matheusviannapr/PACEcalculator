"""
Edição do arranjo: mexer nos módulos que o empacotamento automático propôs.

O empacotamento em grade é um bom ponto de partida e um péssimo ponto final.
Ele não sabe onde fica a caixa d'água, não vê a sombra da casa de máquinas do
elevador, não sabe que aquele canto do telhado é o único acesso para
manutenção, e não sabe que o cliente quer os módulos alinhados com a fachada
por razões que não são de engenharia. Quem sabe tudo isso é quem está olhando
a foto — e precisa de um jeito de dizer.

Este módulo dá esse jeito, com quatro operações que cobrem o que aparece na
prática:

* **Desligar um módulo** — clicar em cima e tirar. É a operação mais usada:
  abrir espaço para um obstáculo que a imagem não mostrou.
* **Acrescentar um módulo** — clicar num vazio e pôr. Serve para a beirada que
  a grade descartou por milímetros, e para o pedaço de telhado que o recuo
  padrão tratou com conservadorismo excessivo.
* **Deslocar a grade inteira** — em metros, para norte/sul e leste/oeste. É o
  que alinha as fileiras com a cumeeira real em vez do centróide do desenho.
* **Girar a grade** — mudar o azimute das fileiras e reempacotar.

As três primeiras operam sobre o resultado existente e são reversíveis; a
quarta refaz o empacotamento, e por isso descarta as edições manuais. A
:class:`EdicaoLayout` guarda a intenção do usuário, e não o resultado: assim
trocar o módulo do catálogo, ou mudar o recuo, não apaga o trabalho de tirar
os seis painéis de cima da claraboia.

Duas invariantes que o módulo garante, porque são as que separam um croqui de
um desenho bonito:

1. **Nenhum módulo fora do telhado.** Nem depois de deslocar a grade, nem ao
   acrescentar à mão. O contorno que vale é o do telhado com o recuo de borda
   aplicado -- é onde dá para parafusar.
2. **Nenhum módulo em cima de outro.** Um croqui com sobreposição some do
   papel e reaparece na obra, quando o instalador descobre que faltam trilhos.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any, Sequence

from shapely.affinity import translate
from shapely.geometry import Point, Polygon
from shapely.geometry.base import BaseGeometry

from ..config import get_settings
from ..geo.geometry import project_to_utm, project_to_wgs84, to_geojson_geometry
from .layout import LayoutModulos

LOGGER = logging.getLogger(__name__)

__all__ = [
    "EdicaoLayout",
    "aplicar_edicao",
    "croqui_editavel",
    "geometria_deslocada",
    "indice_no_ponto",
    "ponto_para_utm",
]

#: Sobreposição tolerada entre dois módulos, como fração da área de um deles.
#: Não é zero porque o polígono vem de rotação em ponto flutuante, e dois
#: módulos vizinhos que se tocam pela aresta acusam interseção de área nula
#: com ruído numérico.
_TOLERANCIA_SOBREPOSICAO = 0.02

#: Folga aceita para considerar um módulo "dentro" do telhado. Um milímetro:
#: o suficiente para o erro de projeção, insuficiente para esconder um painel
#: pendurado na calha.
_TOLERANCIA_CONTIDO_M = 0.001


@dataclass
class EdicaoLayout:
    """
    O que o usuário mudou sobre o arranjo automático.

    Guarda intenção, não geometria: os índices desligados valem sobre a ordem
    estável do empacotamento, e os módulos acrescentados são pontos de centro
    em UTM. Assim a edição sobrevive a uma troca de módulo do catálogo, que é
    exatamente quando não se quer refazer o trabalho manual.
    """

    #: Índices do arranjo automático que o usuário desligou.
    desativados: set[int] = field(default_factory=set)
    #: Centros, em coordenadas UTM, dos módulos postos à mão.
    extras_utm: list[tuple[float, float]] = field(default_factory=list)
    #: Deslocamento da grade inteira, em metros. Leste e Norte positivos.
    deslocamento_leste_m: float = 0.0
    deslocamento_norte_m: float = 0.0

    @property
    def vazia(self) -> bool:
        return (
            not self.desativados
            and not self.extras_utm
            and self.deslocamento_leste_m == 0.0
            and self.deslocamento_norte_m == 0.0
        )

    def limpar(self) -> "EdicaoLayout":
        return EdicaoLayout()

    def alternar(self, indice: int) -> None:
        """Liga ou desliga um módulo do arranjo automático."""
        if indice in self.desativados:
            self.desativados.discard(indice)
        else:
            self.desativados.add(indice)

    def resumo(self) -> str:
        partes = []
        if self.desativados:
            partes.append(f"{len(self.desativados)} removido(s)")
        if self.extras_utm:
            partes.append(f"{len(self.extras_utm)} acrescentado(s)")
        if self.deslocamento_leste_m or self.deslocamento_norte_m:
            partes.append(
                f"grade deslocada {self.deslocamento_leste_m:+.1f} m L / "
                f"{self.deslocamento_norte_m:+.1f} m N"
            )
        return ", ".join(partes) if partes else "sem alterações manuais"

    def as_dict(self) -> dict[str, Any]:
        return {
            "desativados": sorted(self.desativados),
            "extras": len(self.extras_utm),
            "deslocamento_leste_m": self.deslocamento_leste_m,
            "deslocamento_norte_m": self.deslocamento_norte_m,
        }


# ----------------------------------------------------------------------------
# Aplicação
# ----------------------------------------------------------------------------
def _area_permitida(telhado, recuo_m: float | None = None) -> BaseGeometry:
    """
    Onde dá para parafusar: o telhado menos o recuo de borda.

    É o mesmo recuo que o empacotamento automático usa. Aplicar um contorno
    mais generoso na edição manual permitiria ao usuário pôr um módulo onde a
    grade não pôs — e o motivo de a grade não ter posto é justamente que ali
    não cabe.
    """
    recuo = get_settings().solar.edge_setback_m if recuo_m is None else float(recuo_m)
    util = telhado.poligono_utm.buffer(-abs(recuo))
    return util if (util is not None and not util.is_empty) else telhado.poligono_utm


def _molde_do_modulo(layout: LayoutModulos) -> Polygon | None:
    """
    O retângulo de um módulo do arranjo, na posição e no giro corretos.

    Serve de carimbo para os módulos acrescentados à mão: copiar o primeiro
    módulo existente garante orientação, tamanho e giro idênticos aos da
    grade, sem recalcular nada. É também o motivo de não se poder acrescentar
    módulo num telhado onde a grade não pôs nenhum: sem molde, o carimbo teria
    de ser inventado.
    """
    return layout.modulos_geom[0] if layout.modulos_geom else None


def _cabe(candidato: Polygon, permitido: BaseGeometry, existentes: Sequence[Polygon]) -> bool:
    """Dentro do telhado e sem pisar em ninguém."""
    if not permitido.buffer(_TOLERANCIA_CONTIDO_M).contains(candidato):
        return False
    limite = candidato.area * _TOLERANCIA_SOBREPOSICAO
    return not any(
        candidato.intersection(outro).area > limite
        for outro in existentes
        if candidato.intersects(outro)
    )


def geometria_deslocada(
    layout: LayoutModulos, edicao: "EdicaoLayout | None"
) -> list[Polygon]:
    """
    Os módulos do arranjo automático na posição deslocada, sem descartar nada.

    É a única lista cuja numeração é estável, e por isso é a que o croqui
    desenha e a que o clique consulta. Descartar aqui os que saíram do telhado
    renumeraria tudo, e o índice 12 que o usuário desligou passaria a ser outro
    módulo a cada arrasto da grade.
    """
    if edicao is None or (not edicao.deslocamento_leste_m and not edicao.deslocamento_norte_m):
        return list(layout.modulos_geom)
    return [
        translate(g, xoff=edicao.deslocamento_leste_m, yoff=edicao.deslocamento_norte_m)
        for g in layout.modulos_geom
    ]


def aplicar_edicao(
    layout: LayoutModulos,
    telhado,
    edicao: EdicaoLayout | None,
    recuo_m: float | None = None,
) -> LayoutModulos:
    """
    Devolve o arranjo com as edições do usuário aplicadas.

    A ordem importa e é esta: primeiro desloca a grade inteira, depois derruba
    o que saiu do telhado, depois tira o que o usuário desligou, e só então
    acrescenta o que ele pôs à mão. Deslocar por último deixaria os módulos
    manuais andarem junto com a grade, o que não é o que se pede ao arrastar o
    arranjo para longe da caixa d'água.

    O ``quantidade_geometrica`` do arranjo automático é preservado: é ele que
    diz quantos módulos caberiam sem intervenção, e a diferença para a
    quantidade final é a medida do que a edição custou em potência.
    """
    if edicao is None or edicao.vazia:
        return layout

    permitido = _area_permitida(telhado, recuo_m)
    avisos = list(layout.avisos)

    # 1. desloca a grade inteira
    base = geometria_deslocada(layout, edicao)

    # 2. derruba o que saiu do telhado, guardando os índices originais
    dentro: list[tuple[int, Polygon]] = [
        (i, g)
        for i, g in enumerate(base)
        if permitido.buffer(_TOLERANCIA_CONTIDO_M).contains(g)
    ]
    perdidos = len(base) - len(dentro)
    if perdidos:
        avisos.append(
            f"O deslocamento da grade jogou {perdidos} módulo(s) para fora do telhado; "
            "eles foram retirados do arranjo."
        )

    # 3. tira o que o usuário desligou
    geoms = [g for i, g in dentro if i not in edicao.desativados]

    # 4. acrescenta os postos à mão, um a um, conferindo cada um
    molde = _molde_do_modulo(layout)
    recusados = 0
    if edicao.extras_utm and molde is not None:
        centro_molde = molde.centroid
        for leste, norte in edicao.extras_utm:
            candidato = translate(
                molde, xoff=leste - centro_molde.x, yoff=norte - centro_molde.y
            )
            if _cabe(candidato, permitido, geoms):
                geoms.append(candidato)
            else:
                recusados += 1
    if recusados:
        avisos.append(
            f"{recusados} módulo(s) acrescentado(s) à mão não couberam — ficariam fora do "
            "telhado ou sobre outro módulo — e não entraram no arranjo."
        )

    return replace(
        layout,
        quantidade=len(geoms),
        area_ocupada_m2=len(geoms) * layout.modulo.area_m2,
        modulos_geom=geoms,
        metodo=f"{layout.metodo}+edicao",
        avisos=avisos,
    )


# ----------------------------------------------------------------------------
# Ponte com o mapa
# ----------------------------------------------------------------------------
def indice_no_ponto(
    layout: LayoutModulos,
    latitude: float,
    longitude: float,
    edicao: "EdicaoLayout | None" = None,
) -> int | None:
    """
    Qual módulo do arranjo automático está sob esse ponto do mapa.

    É o que transforma um clique no mapa numa operação de edição. O ponto vem
    em latitude/longitude do folium e é projetado para o UTM do arranjo, que é
    o único sistema em que "dentro do retângulo" quer dizer alguma coisa.

    O índice devolvido é o do arranjo **automático**, na numeração estável de
    :func:`geometria_deslocada` — é sobre ela que a edição guarda o que está
    desligado. Devolve ``None`` quando o clique caiu em telhado vazio.
    """
    if not layout.modulos_geom or layout.utm_epsg is None:
        return None
    ponto, _ = project_to_utm(Point(float(longitude), float(latitude)), layout.utm_epsg)
    for i, poligono in enumerate(geometria_deslocada(layout, edicao)):
        if poligono.contains(ponto):
            return i
    return None


def ponto_para_utm(layout: LayoutModulos, latitude: float, longitude: float) -> tuple[float, float]:
    """O clique do mapa em coordenadas UTM, para virar centro de um módulo novo."""
    if layout.utm_epsg is None:
        raise ValueError("o arranjo não tem projeção UTM associada")
    ponto, _ = project_to_utm(Point(float(longitude), float(latitude)), layout.utm_epsg)
    return float(ponto.x), float(ponto.y)


def croqui_editavel(
    layout: LayoutModulos,
    telhado=None,
    edicao: "EdicaoLayout | None" = None,
    recuo_m: float | None = None,
) -> dict[str, Any]:
    """
    O croqui com cada módulo marcado como ativo ou desligado.

    Diferente de :func:`aurum.pv.telhado.croqui_geojson`, que desenha só o que
    entra no sistema, este desenha também os desligados — em outra cor — para
    que o usuário veja o que tirou e possa pôr de volta clicando. Um módulo
    que some do mapa ao ser desligado não tem como voltar.

    Um módulo aparece inativo por dois motivos diferentes e igualmente úteis
    de ver: porque o usuário o desligou, ou porque o deslocamento da grade o
    empurrou para fora do telhado. O segundo caso é o que mostra, enquanto se
    arrasta, quanta potência o arrasto está custando.
    """
    if not layout.modulos_geom or layout.utm_epsg is None:
        return {"type": "FeatureCollection", "features": []}

    desativados = set() if edicao is None else set(edicao.desativados)
    permitido = _area_permitida(telhado, recuo_m) if telhado is not None else None
    geoms = geometria_deslocada(layout, edicao)

    def esta_ativo(indice: int, geom: Polygon) -> bool:
        if indice in desativados:
            return False
        if permitido is None:
            return True
        return bool(permitido.buffer(_TOLERANCIA_CONTIDO_M).contains(geom))

    feicoes = [
        {
            "type": "Feature",
            "properties": {"indice": i, "ativo": esta_ativo(i, geom), "origem": "grade"},
            "geometry": to_geojson_geometry(project_to_wgs84(geom, layout.utm_epsg)),
        }
        for i, geom in enumerate(geoms)
    ]

    # Os módulos postos à mão vêm depois, com índice negativo: eles não fazem
    # parte da numeração da grade, e misturá-los nela quebraria os desligados.
    molde = _molde_do_modulo(layout)
    if edicao is not None and edicao.extras_utm and molde is not None:
        centro = molde.centroid
        aceitos = [g for i, g in enumerate(geoms) if esta_ativo(i, g)]
        for j, (leste, norte) in enumerate(edicao.extras_utm):
            candidato = translate(molde, xoff=leste - centro.x, yoff=norte - centro.y)
            valido = permitido is None or _cabe(candidato, permitido, aceitos)
            if valido:
                aceitos.append(candidato)
            feicoes.append({
                "type": "Feature",
                "properties": {"indice": -(j + 1), "ativo": valido, "origem": "manual"},
                "geometry": to_geojson_geometry(project_to_wgs84(candidato, layout.utm_epsg)),
            })

    return {"type": "FeatureCollection", "features": feicoes}
