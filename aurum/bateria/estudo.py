"""
Orquestração: da planilha de cargas ao conjunto recomendado.

Junta as peças na ordem em que uma pergunta depende da anterior:

1. **Carga** — o Monte Carlo do D², por estação (:mod:`aurum.demanda`).
2. **Geração** — série horária de anos reais no local (:mod:`.geracao`), com a
   potência do sistema vindo de fora ou dimensionada pelo consumo anual que o
   próprio ensemble acabou de produzir.
3. **Triagem por potência** — quais inversores do catálogo sequer aguentam a
   carga (:mod:`.excedencia`). Roda antes da simulação de apagão porque é
   barata e elimina a maioria dos candidatos: não há razão para simular 40 mil
   apagões com um inversor que a curva de excedência já reprovou.
4. **Resiliência** — a malha duração × hora × estação para os sobreviventes
   (:mod:`.apagao`).
5. **Degradação** — a mesma malha reavaliada com o banco envelhecido
   (:mod:`.degradacao`), porque a promessa é de 10 anos, não do dia da entrega.
6. **Economia** — o fluxo de caixa (:mod:`.economia`).
7. **Cenários de fontes** — solar, bateria e gerador ligados e desligados, cada
   arranjo medido contra a conta que o cliente paga hoje (:mod:`.fontes`).
   Vem por último de propósito: só depois de saber qual conjunto é o
   recomendado faz sentido perguntar se vale a pena comprá-lo.

A recomendação sai de um critério declarado, não de um índice composto
inventado: **o conjunto mais barato que cumpre a meta de autonomia com a
confiabilidade pedida, no pior par (estação, hora)**. Quando nenhum cumpre, o
estudo diz isso e mostra o que faltou, em vez de eleger o menos ruim em
silêncio.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..demanda.ensemble import EnsembleCarga, cenario_exemplo, simular_ensemble
from ..demanda.nucleo import Comodo
from .apagao import MalhaApagao, ResultadoResiliencia, avaliar_conjunto
from .catalogo import BaseBaterias, ConjuntoArmazenamento, carregar_catalogo
from .degradacao import ModeloDegradacao, trajetoria_de_vida
from .economia import PremissasBateria, ResultadoEconomicoBateria, avaliar_economia
from .excedencia import avaliar_inversores, potencia_para_excedencia, tabela_por_faixa_horaria
from .fontes import ComparacaoFontes, Composicao, Fatura, Gerador, comparar_fontes
from .geracao import (
    SerieGeracao,
    combinar_series,
    obter_serie_horaria,
    serie_sintetica,
)

LOGGER = logging.getLogger(__name__)

__all__ = ["ConfiguracaoEstudo", "ResultadoEstudo", "executar_estudo"]

#: Anos em que a resiliência é reavaliada com o banco degradado.
ANOS_REAVALIACAO = (0, 5, 10)


@dataclass
class ConfiguracaoEstudo:
    """Tudo que o estudo precisa saber antes de rodar."""

    # -- local -------------------------------------------------------------
    latitude: float
    longitude: float
    nome: str = "instalação"
    #: Tensão de linha da rede no local, em volts. 220 para 127/220 -- a rede
    #: de boa parte do país -- e 380 para 220/380.
    #:
    #: Não é detalhe de instalação: a maior parte dos híbridos trifásicos do
    #: mercado é de 380/400 V e simplesmente não liga numa rede de 220. Sem
    #: este campo o estudo recomendava, num prédio de 127/220, equipamento que
    #: o eletricista devolve na entrega.
    tensao_rede_v: float = 380.0
    inclinacao_deg: float | None = None
    azimute_deg: float | None = None

    # -- carga -------------------------------------------------------------
    comodos: Sequence[Comodo] | None = None
    instancias_por_comodo: dict[str, int] | None = None
    planilha_cargas: str | Path | None = None
    cenario_demo: str = "hotel"
    ajustes_sazonais: dict[str, dict] | None = None
    simulacoes: int = 300
    #: Cômodos ligados ao quadro de backup. ``None`` significa a instalação
    #: inteira — quase nunca é o caso real, e a diferença é grande: um hotel
    #: que põe só elevador, bombas e circulação no backup pede um terço do
    #: inversor que a carga total pediria.
    comodos_essenciais: Sequence[str] | None = None
    #: O quadro de backup montado equipamento a equipamento, quando a origem é
    #: uma vistoria técnica com criticidade por item.
    #:
    #: Recortar por cômodo leva junto tudo que estava no ambiente: numa
    #: cozinha, a geladeira crítica arrasta o forno elétrico de 4 kW. Na
    #: residência do primeiro caso real, o recorte por equipamento deixou
    #: 1,9 kW no backup contra 31,5 kW instalados — 6%. Por cômodo teriam
    #: sobrado quase todos os 31,5.
    comodos_backup: Sequence[Comodo] | None = None
    instancias_backup: dict[str, int] | None = None
    #: A classificação de criticidade que produziu o quadro de backup.
    #:
    #: Espera ``{"tabela": DataFrame, "corte": ("MC", "C"), "descricoes": {...}}``,
    #: com a tabela longa de :meth:`Cenario.por_criticidade` — uma linha por
    #: equipamento, com ambiente, nível e potência. O dossiê usa isso para
    #: listar nominalmente o que a bateria segura: dizer que o backup tem
    #: 1,9 kW não responde à pergunta que o cliente faz, que é se a geladeira
    #: fica de pé.
    criticidade: dict[str, Any] | None = None
    #: Comparar o quadro essencial (MC+C) com o ampliado (MC+C+P).
    #:
    #: A vistoria classifica em quatro níveis e o estudo usava só a linha de
    #: corte. O nível ``P`` — preferível — não é crítico, mas o cliente sente
    #: falta, e decidir se vale pagar por ele exige medir o que ele consome e
    #: quanto banco ele exige **a mais**. Ver :mod:`aurum.bateria.escopos`.
    comparar_escopos: bool = False
    #: Comparar os três cenários de uso da residência, cada um com o seu
    #: sistema. Ver :mod:`aurum.bateria.uso`.
    comparar_uso: bool = False
    #: As tabelas do cenário, por cômodo, como o usuário as editou. O núcleo
    #: do D² converte intervalo e duração em minutos e descarta o resto; o
    #: anexo precisa do original para dizer por quanto tempo cada equipamento
    #: fica ligado, e não só em que janela ele pode ligar.
    tabelas_cenario: dict[str, Any] | None = None
    #: Comparação entre perfis de ocupação (:mod:`aurum.demanda.ocupacao`) e
    #: qual deles dimensionou. Numa residência a mesma casa tem duas curvas —
    #: dia de semana com a casa vazia e fim de semana com a casa cheia — e
    #: dimensionar pela primeira é entregar um sistema que desarma no sábado.
    ocupacao: Any = None
    #: Consumo anual medido ou ponderado, em kWh. Quando informado, é ele que
    #: dimensiona o sistema fotovoltaico — e não a energia do ensemble.
    #:
    #: A distinção existe porque as duas grandezas vêm de perfis diferentes:
    #: o pico e o quadro de backup saem do pior caso (numa residência, o fim
    #: de semana com a casa cheia), e a energia do ano sai da média ponderada
    #: dos dias. Dimensionar o solar pelo pior dia o deixa 40% maior que o
    #: consumo real, e um sistema que gera o que ninguém consome não se paga.
    consumo_anual_kwh: float | None = None

    # -- geração -----------------------------------------------------------
    potencia_fv_kwp: float | None = None
    anos_serie: tuple[int, int] = (2016, 2020)
    permitir_serie_sintetica: bool = True
    #: Telhado marcado no mapa e o empacotamento de módulos que coube nele
    #: (:mod:`aurum.pv.telhado`). Quando existem, a potência do sistema vem
    #: daqui — e o relatório passa a poder afirmar que o sistema cabe, em vez
    #: de só dizer quantos kWp compensariam a conta.
    telhado: Any = None
    layout: Any = None
    #: As águas do telhado, quando há mais de uma.
    #:
    #: Cada entrada é ``{"nome", "telhado", "layout"}``. Um telhado real quase
    #: nunca é um plano só, e duas águas opostas não somam a uma água média: a
    #: do nascente enche de manhã, a do poente à tarde, e o conjunto é mais
    #: plano que qualquer uma. O estudo busca uma série por orientação e as
    #: combina ponderadas pela potência — ver
    #: :func:`aurum.bateria.geracao.combinar_series`.
    #:
    #: Com uma água só, ``telhado``/``layout`` bastam e este campo fica vazio.
    aguas: Sequence[dict[str, Any]] | None = None
    #: Memória de cálculo do arranjo escolhido (:mod:`aurum.pv.memoria`).
    #: É o que permite ao relatório mostrar a conta em vez de pedir fé.
    memoria: Any = None
    #: Análise completa da demanda (:mod:`aurum.demanda.analise`). Quando vem
    #: pronta, o estudo a reaproveita em vez de recalcular.
    analise: Any = None

    # -- equipamentos ------------------------------------------------------
    catalogo: str | Path | None = None
    candidatos: Sequence[ConjuntoArmazenamento] | None = None
    modulos_candidatos: tuple[int, ...] = (1, 2, 3, 4, 6, 8)
    energia_util_min_kwh: float = 0.0
    energia_util_max_kwh: float = float("inf")
    max_candidatos: int = 12

    # -- fontes e conta de luz ---------------------------------------------
    #: A fatura do cliente. Sem ela o estudo ainda roda, mas a comparação de
    #: cenários passa a usar a tarifa padrão, e a economia deixa de ser a
    #: economia dele.
    fatura: Fatura | None = None
    #: Grupo gerador a considerar. ``None`` remove da comparação todos os
    #: cenários com gerador, em vez de inventar um. Com ``potencia_kw=0``, o
    #: estudo dimensiona o grupo para suprir o pico da carga essencial -- que
    #: é o que se espera de um gerador, e tira a potência da discussão para
    #: deixar só o custo de rodar.
    gerador: Gerador | None = None
    #: Folga sobre o pico exigido no dimensionamento automático do grupo.
    folga_gerador: float = 1.15
    #: Quais fontes o cliente aceita considerar. Serve para tirar do quadro o
    #: que ele já descartou -- quatro linhas verdadeiras valem mais que oito
    #: com metade irrelevante.
    #:
    #: ``considerar_solar=False`` não é só uma linha a menos no quadro: é a
    #: chave geral do sol. A potência fotovoltaica vai a zero, a série horária
    #: deixa de ser consultada no PVGIS, e o dossiê perde as seções de
    #: telhado, recurso solar e sistema fotovoltaico inteiras. Um estudo de
    #: cliente que não vai instalar painel não pode exibir curva de geração:
    #: quem lê entende que ela faz parte da proposta.
    considerar_solar: bool = True
    considerar_bateria: bool = True
    considerar_gerador: bool = True
    #: CAPEX do sistema fotovoltaico. ``None`` usa a curva de referência de
    #: :func:`aurum.pv.financials.estimar_capex`. Cotação real, quando existe,
    #: é sempre melhor que qualquer curva.
    capex_fv_brl: float | None = None
    #: Padrão construtivo da obra, entre os
    #: :data:`aurum.pv.financials.PADROES_CAPEX`. O mesmo kWp custa coisas
    #: muito diferentes num galpão e num prédio ocupado de alto padrão: o
    #: nível "alto" foi calibrado numa obra real de 312 kWp fechada a
    #: R$ 4.104/kWp, contra os R$ 2.792/kWp que a curva média daria.
    padrao_capex: str = "padrao"
    #: Topologia do kit fotovoltaico, quando o sistema cabe na tabela de preço
    #: de varejo (:mod:`aurum.pv.kits`, até 40 kWp).
    #:
    #: Abaixo de 40 kWp o preço não sai de curva de escala: sai de tabela de
    #: distribuidor, e a diferença entre um kit monofásico e um split-phase de
    #: mesma potência passa de 50%. Vazio, ou acima de 40 kWp, vale a curva de
    #: ``padrao_capex``.
    topologia_kit: str | None = None
    #: Mão de obra de instalação, em R$/kWp, somada ao preço do kit. Vazio usa
    #: :data:`aurum.pv.kits.MAO_DE_OBRA_BRL_KWP`.
    mao_de_obra_brl_kwp: float | None = None
    #: Material do lado CA, em R$/kWp, também somado. Vazio usa
    #: :data:`aurum.pv.kits.MATERIAL_CA_BRL_KWP`.
    material_ca_brl_kwp: float | None = None

    # -- critério ----------------------------------------------------------
    autonomia_alvo_h: float = 12.0
    confiabilidade_alvo: float = 0.95
    exigir_pior_caso: bool = True

    # -- execução ----------------------------------------------------------
    malha: MalhaApagao = field(default_factory=MalhaApagao)
    premissas: PremissasBateria = field(default_factory=PremissasBateria)
    degradacao: ModeloDegradacao | None = None
    semente: int = 20260902


@dataclass
class ResultadoEstudo:
    """O estudo inteiro, pronto para virar relatório."""

    configuracao: ConfiguracaoEstudo
    ensemble_total: EnsembleCarga
    ensemble_backup: EnsembleCarga
    serie: SerieGeracao
    potencia_fv_kwp: float
    diagnostico_inversores: pd.DataFrame
    tabela_excedencia: pd.DataFrame
    tabela_horaria: pd.DataFrame
    resiliencia: list[ResultadoResiliencia]
    economia: dict[str, ResultadoEconomicoBateria]
    degradacao: dict[str, pd.DataFrame]
    resiliencia_degradada: dict[str, dict[int, float]]
    ranking: pd.DataFrame
    recomendado: ResultadoResiliencia | None
    #: Solar, bateria e gerador ligados e desligados. ``None`` quando não há
    #: conjunto recomendado nem gerador -- não há o que comparar.
    cenarios: ComparacaoFontes | None = None
    #: Quadro essencial contra quadro ampliado, quando a vistoria classificou
    #: equipamentos como preferíveis. ``None`` quando não há o que comparar.
    escopos: Any = None
    #: Os três cenários de uso, cada um com solar e banco próprios.
    uso: Any = None
    avisos: list[str] = field(default_factory=list)

    @property
    def atende_meta(self) -> bool:
        return self.recomendado is not None

    @property
    def com_solar(self) -> bool:
        """
        Se o estudo tem energia solar — a chave que o relatório consulta.

        Vem da potência, e não só da configuração: um estudo marcado com sol
        mas dimensionado em zero kWp também não tem o que mostrar sobre
        geração.
        """
        return bool(self.configuracao.considerar_solar and self.potencia_fv_kwp > 0)

    def resumo(self) -> dict[str, Any]:
        cfg = self.configuracao
        return {
            "local": {
                "nome": cfg.nome,
                "latitude": cfg.latitude,
                "longitude": cfg.longitude,
                "tensao_rede_v": cfg.tensao_rede_v,
                "potencia_fv_kwp": self.potencia_fv_kwp,
                "fonte_geracao": self.serie.fonte,
                "produtividade_kwh_kwp_ano": self.serie.anual_kwh_por_kwp(),
            },
            "telhado": cfg.telhado.as_dict() if cfg.telhado is not None else None,
            "aguas": [
                {
                    "nome": a.get("nome"),
                    "azimute_deg": getattr(a.get("telhado"), "azimute_deg", None),
                    "inclinacao_deg": getattr(a.get("telhado"), "inclinacao_deg", None),
                    "area_m2": getattr(a.get("telhado"), "area_m2", None),
                    "modulos": getattr(a.get("layout"), "quantidade", None),
                    "potencia_kwp": getattr(a.get("layout"), "potencia_kwp", None),
                }
                for a in (cfg.aguas or [])
            ] or None,
            "layout": cfg.layout.as_dict() if cfg.layout is not None else None,
            "memoria_de_calculo": cfg.memoria.as_dict() if cfg.memoria is not None else None,
            "analise_da_demanda": cfg.analise.resumo() if cfg.analise is not None else None,
            "carga": {
                "total": self.ensemble_total.resumo(),
                "backup": self.ensemble_backup.resumo(),
                "essenciais": list(cfg.comodos_essenciais or []),
            },
            "criterio": {
                "autonomia_alvo_h": cfg.autonomia_alvo_h,
                "confiabilidade_alvo": cfg.confiabilidade_alvo,
                "pior_caso": cfg.exigir_pior_caso,
                "atendido": self.atende_meta,
            },
            "recomendado": self.recomendado.resumo() if self.recomendado else None,
            "malha": cfg.malha.descricao(),
            "cenarios_de_fontes": self.cenarios.resumo() if self.cenarios else None,
            "avisos": self.avisos,
        }


# ----------------------------------------------------------------------------
def _carregar_cargas(cfg: ConfiguracaoEstudo) -> tuple[list[Comodo], dict[str, int]]:
    if cfg.comodos:
        instancias = cfg.instancias_por_comodo or {c.nome: 1 for c in cfg.comodos}
        return list(cfg.comodos), dict(instancias)
    if cfg.planilha_cargas:
        from ..demanda.ensemble import carregar_cenario_excel

        comodos = carregar_cenario_excel(str(cfg.planilha_cargas))
        instancias = cfg.instancias_por_comodo or {c.nome: 1 for c in comodos}
        return comodos, dict(instancias)
    return cenario_exemplo(cfg.cenario_demo)


def _serie_do_telhado(cfg: ConfiguracaoEstudo) -> SerieGeracao:
    """
    A série de 1 kWp do sistema: uma água, ou a combinação de várias.

    Águas de mesma orientação e mesma inclinação compartilham a consulta — o
    PVGIS não tem por que ser chamado duas vezes para o mesmo par de ângulos, e
    o cache em disco é por par.
    """
    if not cfg.aguas:
        return obter_serie_horaria(
            cfg.latitude, cfg.longitude,
            azimute_deg=cfg.azimute_deg,
            inclinacao_deg=cfg.inclinacao_deg,
            anos=cfg.anos_serie,
            permitir_fallback=cfg.permitir_serie_sintetica,
        )

    potencia_por_angulo: dict[tuple[float, float], float] = {}
    for agua in cfg.aguas:
        telhado = agua.get("telhado")
        layout = agua.get("layout")
        kwp = float(getattr(layout, "potencia_kwp", 0.0) or 0.0)
        if kwp <= 0:
            continue
        chave = (
            round(float(getattr(telhado, "azimute_deg", cfg.azimute_deg or 0.0)), 1),
            round(float(getattr(telhado, "inclinacao_deg", cfg.inclinacao_deg or 0.0)), 1),
        )
        potencia_por_angulo[chave] = potencia_por_angulo.get(chave, 0.0) + kwp

    if not potencia_por_angulo:
        return obter_serie_horaria(
            cfg.latitude, cfg.longitude,
            azimute_deg=cfg.azimute_deg, inclinacao_deg=cfg.inclinacao_deg,
            anos=cfg.anos_serie, permitir_fallback=cfg.permitir_serie_sintetica,
        )

    series, pesos = [], []
    for (azimute, inclinacao), kwp in potencia_por_angulo.items():
        series.append(obter_serie_horaria(
            cfg.latitude, cfg.longitude,
            azimute_deg=azimute, inclinacao_deg=inclinacao,
            anos=cfg.anos_serie, permitir_fallback=cfg.permitir_serie_sintetica,
        ))
        pesos.append(kwp)
    return combinar_series(series, pesos)


def _dimensionar_fv(cfg: ConfiguracaoEstudo, ensemble: EnsembleCarga, serie: SerieGeracao) -> tuple[float, list[str]]:
    """Potência do sistema: a que cabe no telhado, a informada, ou a que compensa."""
    if cfg.aguas:
        kwp = sum(
            float(getattr(a.get("layout"), "potencia_kwp", 0.0) or 0.0)
            for a in cfg.aguas
        )
        modulos = sum(int(getattr(a.get("layout"), "quantidade", 0) or 0) for a in cfg.aguas)
        area = sum(
            float(getattr(a.get("telhado"), "area_m2", 0.0) or 0.0) for a in cfg.aguas
        )
        if kwp > 0:
            geracao = kwp * serie.anual_kwh_por_kwp()
            # O separador de milhar é trocado número a número: aplicar o
            # `replace` na frase inteira comeria a vírgula do texto junto.
            area_txt = f"{area:,.0f}".replace(",", ".")
            geracao_txt = f"{geracao:,.0f}".replace(",", ".")
            return kwp, [
                f"Potência definida pelas {len(cfg.aguas)} águas marcadas: {modulos} "
                f"módulos em {area_txt} m² de projeção, somando {kwp:.1f} kWp e "
                f"{geracao_txt} kWh/ano."
            ]

    if cfg.layout is not None and getattr(cfg.layout, "potencia_kwp", 0) > 0:
        kwp = float(cfg.layout.potencia_kwp)
        area = getattr(cfg.telhado, "area_m2", 0.0) if cfg.telhado is not None else 0.0
        consumo_anual = float(np.mean(ensemble.energia_diaria_kwh())) * 365.0
        geracao = kwp * serie.anual_kwh_por_kwp()
        # O separador de milhar vira ponto só no número; aplicar o replace na
        # frase inteira comia as vírgulas do texto.
        area_texto = f"{area:,.0f}".replace(",", ".")
        aviso = (
            f"Potência definida pelo telhado marcado: {cfg.layout.quantidade} módulos de "
            f"{cfg.layout.modulo.potencia_wp:.0f} Wp cabem em {area_texto} m², somando "
            f"{kwp:.1f} kWp. Isso cobre {geracao / consumo_anual:.0%} do consumo anual."
        )
        return kwp, [aviso]
    if cfg.potencia_fv_kwp is not None:
        return float(cfg.potencia_fv_kwp), []
    # O consumo informado vence o do ensemble: o ensemble é do perfil que
    # dimensiona o equipamento, e não do ano médio.
    do_ensemble = float(np.mean(ensemble.energia_diaria_kwh())) * 365.0
    consumo_anual = float(cfg.consumo_anual_kwh or do_ensemble)
    rendimento = serie.anual_kwh_por_kwp()
    kwp = consumo_anual / rendimento if rendimento > 0 else 0.0
    consumo_texto = f"{consumo_anual:,.0f}".replace(",", ".")
    origem = (
        "consumo anual ponderado entre os perfis de ocupação"
        if cfg.consumo_anual_kwh
        else "consumo anual estimado"
    )
    aviso = (
        f"Potência FV não informada: dimensionada em {kwp:.1f} kWp para compensar o "
        f"{origem} de {consumo_texto} kWh a {rendimento:.0f} kWh/kWp. "
        "O estudo não verifica se esse sistema cabe na cobertura — para isso, marque "
        "o telhado no mapa."
    )
    return kwp, [aviso]


def _selecionar_candidatos(
    cfg: ConfiguracaoEstudo,
    base: BaseBaterias,
    diagnostico: pd.DataFrame,
    energia_alvo_kwh: float = 0.0,
) -> tuple[list[ConjuntoArmazenamento], list[str]]:
    """Combinações do catálogo, filtradas pela triagem de potência."""
    if cfg.candidatos:
        return list(cfg.candidatos), []

    aprovados = set(diagnostico.loc[diagnostico["aprovado"], "modelo"].str.lower())
    avisos: list[str] = []
    if not aprovados:
        aprovados = set(diagnostico["modelo"].str.lower())
        avisos.append(
            "Nenhum inversor do catálogo passa na triagem de potência para a carga de backup: "
            "todos desarmariam a sobrecarga em mais de 1% dos dias. A varredura seguiu com o "
            "catálogo inteiro, mas o resultado exige corte seletivo de carga ou um inversor "
            "maior que os cadastrados."
        )

    combinacoes = [
        conjunto
        for conjunto in base.combinacoes(
            modulos=cfg.modulos_candidatos,
            energia_util_min_kwh=cfg.energia_util_min_kwh,
            energia_util_max_kwh=cfg.energia_util_max_kwh,
        )
        if conjunto.inversor.modelo.lower() in aprovados
    ]
    if not combinacoes:
        raise ValueError("nenhuma combinação de inversor e bateria atende aos filtros dados")

    combinacoes.sort(key=lambda c: (c.energia_util_kwh, c.potencia_descarga_kw))
    if len(combinacoes) <= cfg.max_candidatos:
        return combinacoes, avisos

    if energia_alvo_kwh > 0:
        # Amostrar por `linspace` sobre a lista inteira parece isento e não é:
        # a lista vai de 2 kWh a 80 kWh, e a resposta quase sempre mora numa
        # faixa estreita em torno da energia que a meta exige. Espalhar os
        # candidatos uniformemente gasta metade deles em bancos absurdamente
        # grandes e pode pular o mais barato que atende — que é justamente o
        # que o estudo existe para achar. Ancorar as buscas em múltiplos da
        # energia alvo põe candidatos dos dois lados da fronteira.
        fatores = (0.4, 0.7, 1.0, 1.3, 1.8, 2.5, 0.25, 3.5)
        alvos = [energia_alvo_kwh * f for f in fatores[: cfg.max_candidatos]]
        escolhidos: dict[str, ConjuntoArmazenamento] = {}
        for alvo in alvos:
            perto = min(combinacoes, key=lambda c: abs(c.energia_util_kwh - alvo))
            escolhidos.setdefault(perto.descricao(), perto)
        # Completa com os extremos, para a fronteira aparecer no gráfico.
        for extremo in (combinacoes[0], combinacoes[-1]):
            if len(escolhidos) < cfg.max_candidatos:
                escolhidos.setdefault(extremo.descricao(), extremo)
        return sorted(escolhidos.values(), key=lambda c: c.energia_util_kwh), avisos

    indices = np.linspace(0, len(combinacoes) - 1, cfg.max_candidatos).round().astype(int)
    return [combinacoes[i] for i in dict.fromkeys(indices.tolist())], avisos


def _ensemble_de_backup(
    cfg: ConfiguracaoEstudo,
    comodos: Sequence[Comodo],
    instancias: dict[str, int],
) -> tuple[EnsembleCarga, list[str]]:
    # Backup montado item a item vence o recorte por cômodo: é mais fino, e
    # quem o montou sabia de coisa que o nome do ambiente não carrega.
    if cfg.comodos_backup:
        instancias_backup = cfg.instancias_backup or {
            c.nome: instancias.get(c.nome, 1) for c in cfg.comodos_backup
        }
        return simular_ensemble(
            list(cfg.comodos_backup), instancias_backup, cfg.simulacoes,
            cfg.ajustes_sazonais, semente=cfg.semente,
        ), []
    if not cfg.comodos_essenciais:
        return simular_ensemble(
            comodos, instancias, cfg.simulacoes, cfg.ajustes_sazonais, semente=cfg.semente
        ), []
    nomes = {n.strip().lower() for n in cfg.comodos_essenciais}
    escolhidos = [c for c in comodos if c.nome.strip().lower() in nomes]
    if not escolhidos:
        disponiveis = ", ".join(c.nome for c in comodos)
        raise ValueError(f"nenhum cômodo essencial encontrado; disponíveis: {disponiveis}")
    faltando = nomes - {c.nome.strip().lower() for c in escolhidos}
    avisos = [f"Cômodos essenciais não encontrados e ignorados: {', '.join(sorted(faltando))}"] if faltando else []
    return (
        simular_ensemble(
            escolhidos,
            {c.nome: instancias.get(c.nome, 1) for c in escolhidos},
            cfg.simulacoes,
            cfg.ajustes_sazonais,
            semente=cfg.semente,
        ),
        avisos,
    )


# ----------------------------------------------------------------------------
def executar_estudo(cfg: ConfiguracaoEstudo, progresso=None) -> ResultadoEstudo:
    """Roda o estudo completo e devolve tudo que o relatório precisa."""

    def avisar(mensagem: str, fracao: float) -> None:
        LOGGER.info(mensagem)
        if progresso is not None:
            progresso(mensagem, fracao)

    avisos: list[str] = []

    # 1. carga -------------------------------------------------------------
    avisar("Simulando a demanda (Monte Carlo por estação)", 0.05)
    comodos, instancias = _carregar_cargas(cfg)
    ensemble_total = simular_ensemble(
        comodos, instancias, cfg.simulacoes, cfg.ajustes_sazonais, semente=cfg.semente
    )
    ensemble_backup, avisos_backup = _ensemble_de_backup(cfg, comodos, instancias)
    avisos.extend(avisos_backup)

    # 2. geração -----------------------------------------------------------
    if not cfg.considerar_solar:
        # Sem sol no estudo, a série existe só para atravessar as assinaturas
        # do despacho, multiplicada por zero kWp. Consultar o PVGIS aqui seria
        # gastar rede para buscar um vetor que não entra em conta nenhuma.
        avisar("Estudo sem energia solar: pulando a série de geração", 0.25)
        serie = serie_sintetica(
            cfg.latitude, cfg.longitude, 0.0, 20.0, 14.0, anos=cfg.anos_serie,
            aviso=None,
        )
        kwp = 0.0
    else:
        avisar("Obtendo a série horária de geração", 0.25)
        serie = _serie_do_telhado(cfg)
        if serie.aviso:
            avisos.append(serie.aviso)
        kwp, avisos_fv = _dimensionar_fv(cfg, ensemble_total, serie)
        avisos.extend(avisos_fv)

    # 3. triagem por potência ---------------------------------------------
    avisar("Cruzando a curva de excedência de pico com o catálogo de inversores", 0.35)
    base = carregar_catalogo(cfg.catalogo)
    base, avisos_rede = _filtrar_por_rede(base, cfg.tensao_rede_v)
    avisos.extend(avisos_rede)
    ensemble_backup_passo = ensemble_backup.reamostrar(cfg.malha.passo_min)
    diagnostico = avaliar_inversores(base, ensemble_backup_passo)
    potencias = sorted({round(i.potencia_ca_nominal_kw, 1) for i in base.inversores})
    tabela_horaria = tabela_por_faixa_horaria(ensemble_backup_passo, potencias)
    tabela_exc = potencia_para_excedencia(ensemble_backup_passo)

    # Energia que a meta exige sem nenhum sol: o consumo médio de backup pela
    # duração alvo. É a âncora da busca de candidatos — com sol o banco pode
    # ser menor, sem sol precisa ser maior, e a resposta mora em volta disso.
    consumo_medio_kw = float(np.mean(ensemble_backup_passo.perfis())) / 1000.0
    energia_alvo = consumo_medio_kw * cfg.autonomia_alvo_h

    # O grupo gerador, quando pedido sem potência, é dimensionado aqui: a
    # curva de excedência acabou de dizer qual é o pico que a carga essencial
    # exige em 99% dos dias, e é ele que define a placa do grupo.
    gerador = cfg.gerador
    if gerador is not None and gerador.potencia_kw <= 0:
        gerador, aviso_grupo = _dimensionar_gerador(cfg, tabela_exc, gerador)
        cfg = replace(cfg, gerador=gerador)
        avisos.append(aviso_grupo)

    candidatos, avisos_cand = _selecionar_candidatos(cfg, base, diagnostico, energia_alvo)
    avisos.extend(avisos_cand)
    candidatos = [
        ConjuntoArmazenamento(c.inversor, c.bateria, c.modulos, potencia_fv_kwp=kwp)
        for c in candidatos
    ]

    # 4. resiliência -------------------------------------------------------
    resiliencias: list[ResultadoResiliencia] = []
    for i, conjunto in enumerate(candidatos):
        avisar(f"Apagões — {conjunto.descricao()}", 0.35 + 0.40 * (i + 1) / len(candidatos))
        resiliencias.append(
            avaliar_conjunto(conjunto, ensemble_backup_passo, serie, cfg.malha, cfg.semente)
        )

    # 5. economia e degradação --------------------------------------------
    avisar("Economia, degradação e fluxo de caixa", 0.80)
    economia: dict[str, ResultadoEconomicoBateria] = {}
    degradacoes: dict[str, pd.DataFrame] = {}
    resiliencia_degradada: dict[str, dict[int, float]] = {}

    for resultado in resiliencias:
        chave = resultado.conjunto.descricao()
        modelo = cfg.degradacao or ModeloDegradacao.do_conjunto(resultado.conjunto)
        economico = avaliar_economia(
            resultado.conjunto, ensemble_backup_passo, serie,
            cfg.premissas, resultado, modelo, semente=cfg.semente,
        )
        economia[chave] = economico
        degradacoes[chave] = trajetoria_de_vida(
            modelo,
            economico.operacao.ciclos_equivalentes,
            resultado.conjunto.energia_util_kwh,
            cfg.premissas.anos_analise,
        )
        # A promessa de autonomia reavaliada com o banco de 5 e 10 anos.
        por_ano: dict[int, float] = {}
        for ano in ANOS_REAVALIACAO:
            retencao = modelo.retencao(ano, economico.operacao.ciclos_equivalentes * ano)
            envelhecido = (
                resultado
                if ano == 0
                else avaliar_conjunto(
                    resultado.conjunto, ensemble_backup_passo, serie, cfg.malha,
                    cfg.semente, fator_capacidade=retencao,
                )
            )
            por_ano[ano] = envelhecido.autonomia_garantida_h(
                cfg.confiabilidade_alvo, cfg.exigir_pior_caso
            )
        resiliencia_degradada[chave] = por_ano

    # 6. ranking e recomendação -------------------------------------------
    ranking = _montar_ranking(cfg, resiliencias, economia, resiliencia_degradada)
    aptos = ranking[ranking["atende_meta"]]
    recomendado = None
    if not aptos.empty:
        escolhido = aptos.sort_values(["capex_brl", "energia_util_kwh"]).iloc[0]["conjunto"]
        recomendado = next(r for r in resiliencias if r.conjunto.descricao() == escolhido)
    else:
        melhor = ranking.sort_values("autonomia_garantida_h", ascending=False).iloc[0]
        avisos.append(
            f"Nenhum candidato cumpre {cfg.autonomia_alvo_h:g} h com "
            f"{cfg.confiabilidade_alvo:.0%} de confiabilidade"
            f"{' no pior par estação/hora' if cfg.exigir_pior_caso else ''}. "
            f"O melhor chegou a {melhor['autonomia_garantida_h']:g} h "
            f"({melhor['conjunto']}). Caminhos: reduzir a carga do quadro de backup, "
            f"aceitar confiabilidade menor, ou ampliar banco e inversor além do catálogo."
        )

    # 7. cenários de fontes ------------------------------------------------
    avisar("Comparando cenários com e sem solar, bateria e gerador", 0.92)
    # Sem conjunto recomendado, o quadro usa o de maior autonomia entre os
    # avaliados: a pergunta "vale a pena ter bateria?" continua valendo mesmo
    # quando nenhuma bateria do catálogo cumpre a meta, e suprimir as quatro
    # linhas com bateria responderia "não" por omissão.
    conjunto_dos_cenarios = recomendado.conjunto if recomendado is not None else None
    if conjunto_dos_cenarios is None and resiliencias:
        melhor = max(
            resiliencias,
            key=lambda r: r.autonomia_garantida_h(cfg.confiabilidade_alvo, cfg.exigir_pior_caso),
        )
        conjunto_dos_cenarios = melhor.conjunto
        avisos.append(
            f"Os cenários com bateria usam {melhor.conjunto.descricao()}, o de maior "
            "autonomia entre os avaliados, já que nenhum cumpre a meta. Serve para "
            "medir o que uma bateria acrescenta; não é recomendação de compra."
        )
    cenarios = _comparar_fontes(
        cfg, ensemble_total, ensemble_backup_passo, serie, kwp, conjunto_dos_cenarios,
    )
    if cenarios is not None:
        avisos.extend(cenarios.avisos)

    # O quadro em dois níveis. Roda por último e nunca derruba o estudo: é uma
    # leitura a mais sobre o mesmo dado, e não uma etapa de que o resto depende.
    escopos = None
    if cfg.comparar_escopos and cfg.tabelas_cenario:
        avisar("Comparando o quadro essencial com o ampliado", 0.95)
        try:
            from .escopos import comparar_escopos as _comparar_escopos

            escopos = _comparar_escopos(
                cfg.tabelas_cenario,
                cfg.instancias_backup or cfg.instancias_por_comodo or {},
                candidatos, serie, cfg.malha,
                autonomia_alvo_h=cfg.autonomia_alvo_h,
                confiabilidade=cfg.confiabilidade_alvo,
                simulacoes=cfg.simulacoes,
                premissas=cfg.premissas,
                semente=cfg.semente,
                ajustes_sazonais=cfg.ajustes_sazonais,
                catalogo=base,
            )
            avisos.extend(escopos.avisos)
        except Exception as exc:  # noqa: BLE001 — leitura extra não derruba estudo
            LOGGER.warning("comparação de escopos falhou: %s", exc)
            avisos.append(f"A comparação entre quadro essencial e ampliado falhou: {exc}")

    uso = None
    if cfg.comparar_uso and cfg.tabelas_cenario:
        avisar("Comparando os cenários de uso da residência", 0.97)
        try:
            from ..demanda.cenario import Cenario
            from .uso import comparar_cenarios_de_uso

            base_uso = Cenario(
                nome=cfg.nome, segmento="residencia",
                comodos=dict(cfg.tabelas_cenario),
                instancias=dict(cfg.instancias_por_comodo or {}),
            )
            base_uso.criticidades_essenciais = tuple(
                (cfg.criticidade or {}).get("corte", ()) or ())
            uso = comparar_cenarios_de_uso(
                base_uso, serie, base, malha=cfg.malha,
                autonomia_alvo_h=cfg.autonomia_alvo_h,
                confiabilidade=cfg.confiabilidade_alvo,
                simulacoes=max(60, cfg.simulacoes // 2),
                premissas=cfg.premissas,
                topologia_kit=cfg.topologia_kit or "splitphase",
                semente=cfg.semente,
            )
            avisos.extend(uso.avisos)
        except Exception as exc:  # noqa: BLE001 — leitura extra não derruba estudo
            LOGGER.warning("comparação de cenários de uso falhou: %s", exc)
            avisos.append(f"A comparação entre cenários de uso falhou: {exc}")

    avisar("Estudo concluído", 1.0)
    return ResultadoEstudo(
        configuracao=cfg,
        ensemble_total=ensemble_total,
        ensemble_backup=ensemble_backup_passo,
        serie=serie,
        potencia_fv_kwp=kwp,
        diagnostico_inversores=diagnostico,
        tabela_excedencia=tabela_exc,
        tabela_horaria=tabela_horaria,
        resiliencia=resiliencias,
        economia=economia,
        degradacao=degradacoes,
        resiliencia_degradada=resiliencia_degradada,
        ranking=ranking,
        recomendado=recomendado,
        cenarios=cenarios,
        escopos=escopos,
        uso=uso,
        avisos=avisos,
    )


def _filtrar_por_rede(base: BaseBaterias, tensao_linha_v: float) -> tuple[BaseBaterias, list[str]]:
    """
    Tira do catálogo o que não liga na rede do local.

    Antes de qualquer simulação, porque é o filtro mais barato e o mais
    definitivo: um inversor de 380 V não vira 220 V com corte de carga nem com
    banco maior. Recomendá-lo é um erro que só aparece na entrega.

    Quando o filtro esvazia o catálogo, ele **não** é aplicado -- o estudo
    segue com tudo e diz por quê. Bloquear aqui deixaria o usuário sem
    resposta nenhuma; avisar deixa a decisão com quem sabe se o projeto prevê
    transformador.
    """
    from dataclasses import replace as _replace

    compativeis = [i for i in base.inversores if i.atende_rede(tensao_linha_v)]
    if not compativeis:
        return base, [
            f"Nenhum inversor do catálogo atende a rede de {tensao_linha_v:.0f} V. "
            "O estudo seguiu com o catálogo inteiro, mas o equipamento recomendado "
            "exige transformador ou substituição por modelo da tensão certa."
        ]
    descartados = len(base.inversores) - len(compativeis)
    avisos = []
    if descartados:
        avisos.append(
            f"{descartados} inversor(es) do catálogo não atendem a rede de "
            f"{tensao_linha_v:.0f} V e ficaram de fora da triagem."
        )
    return _replace(base, inversores=compativeis), avisos


def _dimensionar_gerador(
    cfg: ConfiguracaoEstudo, tabela_excedencia: pd.DataFrame, pedido: Gerador
) -> tuple[Gerador, str]:
    """
    Escolhe a placa do grupo a partir do pico que a carga essencial exige.

    O critério é o pico diário superado em 1% dos dias, e não o máximo
    absoluto da simulação: dimensionar pelo máximo de dezenas de milhares de
    dias sorteados compra um grupo para um dia que talvez nunca aconteça. Um
    por cento dos dias é o mesmo critério com que os inversores são triados,
    e usar dois critérios diferentes para a mesma carga produziria um gerador
    e um inversor que não conversam.
    """
    linha = tabela_excedencia[np.isclose(tabela_excedencia["prob_excedencia"], 0.01)]
    exigido = (
        float(linha["pico_diario_kw"].iloc[0])
        if not linha.empty
        else float(tabela_excedencia["pico_diario_kw"].max())
    )
    grupo = Gerador.para_carga(
        exigido,
        folga=cfg.folga_gerador,
        custo_energia_brl_kwh=pedido.custo_energia_brl_kwh,
        capex_brl=pedido.capex_brl,
        opex_fixo_brl_ano=pedido.opex_fixo_brl_ano,
        atraso_partida_min=pedido.atraso_partida_min,
        modelo=pedido.modelo,
        recarrega_bateria=pedido.recarrega_bateria,
    )
    aviso = (
        f"Grupo gerador dimensionado em {grupo.potencia_kw:g} kW: a carga de backup "
        f"supera {exigido:.1f} kW em 1% dos dias, e o critério aplicou {cfg.folga_gerador:.0%} "
        "de folga sobre esse pico. Com essa placa o grupo supre a carga, e o que resta "
        "para decidir é o custo de rodar."
    )
    return grupo, aviso


def _comparar_fontes(
    cfg: ConfiguracaoEstudo,
    ensemble_total: EnsembleCarga,
    ensemble_backup: EnsembleCarga,
    serie: SerieGeracao,
    potencia_fv_kwp: float,
    conjunto: ConjuntoArmazenamento | None,
) -> ComparacaoFontes | None:
    """
    Monta o quadro de cenários, ou explica por que ele não existe.

    O conjunto que entra nos cenários é o recomendado, e não o mais barato do
    catálogo: comparar "com bateria" contra "sem bateria" usando uma bateria
    que não cumpre a meta responderia a pergunta errada.
    """
    if conjunto is None and cfg.gerador is None:
        return None

    # A malha dos cenários é mais grossa que a do estudo de propósito: aqui a
    # pergunta é qual arranjo escolher, e ela se responde com a ordem de
    # grandeza da autonomia. A malha fina fica para o conjunto recomendado,
    # que é o único que vai para a proposta.
    malha = replace(
        cfg.malha,
        horas_inicio=tuple(range(0, 24, 3)),
        amostras=max(30, cfg.malha.amostras // 4),
    )
    fontes = combinacoes_pedidas(cfg, potencia_fv_kwp, conjunto)
    return comparar_fontes(
        ensemble_total=ensemble_total,
        ensemble_backup=ensemble_backup,
        serie=serie,
        potencia_fv_kwp=potencia_fv_kwp,
        fatura=cfg.fatura or Fatura(),
        conjunto=conjunto,
        gerador=cfg.gerador,
        premissas=cfg.premissas,
        malha=malha,
        capex_fv_brl=cfg.capex_fv_brl,
        padrao_capex=cfg.padrao_capex,
        topologia_kit=cfg.topologia_kit,
        mao_de_obra_brl_kwp=cfg.mao_de_obra_brl_kwp,
        material_ca_brl_kwp=cfg.material_ca_brl_kwp,
        composicoes=fontes,
        semente=cfg.semente,
    )


def combinacoes_pedidas(
    cfg: ConfiguracaoEstudo,
    potencia_fv_kwp: float,
    conjunto: ConjuntoArmazenamento | None,
) -> list[Composicao]:
    """As combinações que fazem sentido: pedidas pelo cliente e possíveis aqui."""
    from .fontes import combinacoes

    return combinacoes(
        solar=cfg.considerar_solar and potencia_fv_kwp > 0,
        bateria=cfg.considerar_bateria and conjunto is not None,
        gerador=cfg.considerar_gerador and cfg.gerador is not None,
    )


def _montar_ranking(
    cfg: ConfiguracaoEstudo,
    resiliencias: Sequence[ResultadoResiliencia],
    economia: dict[str, ResultadoEconomicoBateria],
    degradada: dict[str, dict[int, float]],
) -> pd.DataFrame:
    linhas: list[dict[str, Any]] = []
    for resultado in resiliencias:
        chave = resultado.conjunto.descricao()
        economico = economia[chave]
        autonomia = resultado.autonomia_garantida_h(cfg.confiabilidade_alvo, cfg.exigir_pior_caso)
        alvo = cfg.autonomia_alvo_h
        recorte = resultado.tabela[np.isclose(resultado.tabela["duracao_h"], alvo)]
        prob_no_alvo = (
            float(recorte["prob_atendimento"].min() if cfg.exigir_pior_caso else recorte["prob_atendimento"].mean())
            if not recorte.empty
            else float("nan")
        )
        linhas.append({
            "conjunto": chave,
            "inversor": str(resultado.conjunto.inversor),
            "bateria": str(resultado.conjunto.bateria),
            "modulos": resultado.conjunto.modulos,
            "energia_util_kwh": resultado.conjunto.energia_util_kwh,
            "potencia_kw": resultado.conjunto.potencia_descarga_kw,
            "potencia_pico_kw": resultado.conjunto.potencia_pico_kw,
            "autonomia_garantida_h": autonomia,
            f"prob_em_{alvo:g}h": prob_no_alvo,
            "atende_meta": bool(autonomia >= alvo),
            "autonomia_ano5_h": degradada[chave].get(5, float("nan")),
            "autonomia_ano10_h": degradada[chave].get(10, float("nan")),
            "capex_brl": economico.capex_brl,
            "vpl_brl": economico.vpl_brl,
            "tir": economico.tir,
            "payback_anos": economico.payback_anos,
            "lcos_brl_kwh": economico.custo_nivelado_brl_kwh,
            "ciclos_ano": economico.operacao.ciclos_equivalentes,
            "vida_util_anos": economico.vida_util_anos,
            "brl_por_hora_garantida": (
                economico.capex_brl / autonomia if autonomia > 0 else float("inf")
            ),
        })
    return pd.DataFrame(linhas).sort_values("energia_util_kwh").reset_index(drop=True)
