"""
Estimativa de consumo de energia a partir do tipo e do porte da edificação.

Na prospecção não existe conta de luz: o telhado veio de um mapa. Para que a
proposta tenha um número de referência, o consumo é estimado por **intensidade
de uso de energia** (EUI) -- kWh por m2 de área construída por mês, por
segmento.

Isto é uma estimativa de triagem, e o módulo trata assim: cada resultado
carrega a faixa plausível e o aviso de que a conta real precisa ser
confirmada. Nenhuma proposta deve ser fechada com base neste número.

Referências das faixas: PROCEL Edifica, pesquisas de posse e hábitos da
EPE/Eletrobras e benchmarks de eficiência energética do setor comercial
brasileiro. Ordens de grandeza, não valores normativos.
"""
from __future__ import annotations

from dataclasses import dataclass, field

#: kWh por m2 de área construída por mês: (mínimo, típico, máximo).
#: A variação dentro do mesmo segmento é grande -- daí a faixa.
EUI_KWH_M2_MES: dict[str, tuple[float, float, float]] = {
    "industrial":    (6.0, 14.0, 35.0),
    "warehouse":     (1.5,  4.0, 10.0),
    "factory":       (8.0, 18.0, 40.0),
    "logistics":     (2.0,  5.0, 12.0),
    "commercial":    (8.0, 14.0, 25.0),
    "retail":        (8.0, 15.0, 28.0),
    "supermarket":   (18.0, 30.0, 50.0),
    "mall":          (12.0, 20.0, 32.0),
    "office":        (6.0, 11.0, 20.0),
    "hotel":         (10.0, 18.0, 30.0),
    "school":        (2.5,  5.0, 10.0),
    "university":    (4.0,  8.0, 15.0),
    "college":       (3.0,  6.0, 12.0),
    "hospital":      (14.0, 25.0, 45.0),
    "clinic":        (8.0, 15.0, 28.0),
    "civic":         (4.0,  8.0, 15.0),
    "public":        (4.0,  8.0, 15.0),
    "government":    (5.0,  9.0, 16.0),
    "church":        (1.0,  2.5,  6.0),
    "sports_centre": (4.0,  9.0, 18.0),
    "sports_hall":   (4.0,  9.0, 18.0),
    "stadium":       (3.0,  7.0, 15.0),
    "residential":   (2.0,  4.0,  8.0),
    "apartments":    (2.5,  5.0,  9.0),
    "dormitory":     (3.0,  6.0, 12.0),
    "kindergarten":  (2.0,  4.5,  9.0),
    "chapel":        (0.8,  2.0,  5.0),
    "cathedral":     (1.5,  3.5,  8.0),
    "fire_station":  (3.0,  6.0, 12.0),
    "prison":        (5.0, 10.0, 18.0),
    # Agro. A faixa é larga porque um barracão de máquinas quase não consome
    # e um aviário climatizado consome como uma pequena indústria.
    "barn":           (1.0,  4.0, 15.0),
    "cowshed":        (3.0,  9.0, 20.0),
    "farm":           (1.5,  5.0, 14.0),
    "farm_auxiliary": (1.0,  4.0, 12.0),
    "greenhouse":     (2.0,  8.0, 25.0),
    "silo":           (2.0,  7.0, 18.0),
    "stable":         (1.5,  5.0, 12.0),
    "sty":            (3.0, 10.0, 22.0),
    "chicken_coop":   (5.0, 14.0, 30.0),
    # Refrigeração é o maior consumidor por metro quadrado da lista.
    "slaughterhouse": (20.0, 35.0, 60.0),
    "cold_storage":   (25.0, 45.0, 80.0),
}
#: Usado quando o tipo é desconhecido ou apenas "building=yes".
EUI_PADRAO = (4.0, 9.0, 18.0)

#: Fração da área construída que fica no telhado projetado. Prédios de vários
#: pavimentos têm muito mais área construída do que telhado -- e por isso
#: consomem muito mais do que o telhado consegue gerar.
FATOR_PAVIMENTOS_MAX = 12


@dataclass
class EstimativaConsumo:
    """Consumo estimado de uma edificação, com a faixa de incerteza."""

    area_construida_m2: float
    pavimentos: int
    tipo_edificacao: str
    eui_kwh_m2_mes: float
    consumo_mensal_kwh: float
    consumo_anual_kwh: float
    faixa_mensal_kwh: tuple[float, float]
    #: Fração da geração tipicamente consumida no horário de produção.
    fracao_autoconsumo: float
    confianca: str = "baixa"
    avisos: list[str] = field(default_factory=list)

    @property
    def demanda_media_kw(self) -> float:
        """Demanda média no horário comercial (22 dias, 10 h/dia)."""
        return self.consumo_mensal_kwh / (22.0 * 10.0)

    def as_dict(self) -> dict:
        return {
            "tipo_edificacao": self.tipo_edificacao,
            "area_construida_m2": round(self.area_construida_m2, 1),
            "pavimentos": self.pavimentos,
            "eui_kwh_m2_mes": self.eui_kwh_m2_mes,
            "consumo_mensal_kwh": round(self.consumo_mensal_kwh, 0),
            "consumo_anual_kwh": round(self.consumo_anual_kwh, 0),
            "faixa_mensal_kwh": [round(self.faixa_mensal_kwh[0], 0), round(self.faixa_mensal_kwh[1], 0)],
            "fracao_autoconsumo": self.fracao_autoconsumo,
            "demanda_media_kw": round(self.demanda_media_kw, 1),
            "confianca": self.confianca,
            "avisos": list(self.avisos),
        }


#: Fração da geração consumida no próprio horário de produção, por segmento.
#: Operação diurna e contínua aproveita quase tudo; operação noturna, pouco.
FRACAO_AUTOCONSUMO: dict[str, float] = {
    "industrial": 0.80, "factory": 0.82, "warehouse": 0.60, "logistics": 0.65,
    "commercial": 0.70, "retail": 0.72, "supermarket": 0.78, "mall": 0.75,
    "office": 0.75, "hotel": 0.50, "school": 0.70, "university": 0.72,
    "college": 0.70, "hospital": 0.65, "clinic": 0.72, "civic": 0.70,
    "public": 0.70, "government": 0.72, "church": 0.25,
    "sports_centre": 0.45, "sports_hall": 0.45, "stadium": 0.35,
    "residential": 0.30, "apartments": 0.35,
    # Agro: irrigação, ventilação e secagem acompanham o sol; refrigeração
    # roda dia e noite, e a parte diurna é toda aproveitada.
    "barn": 0.55, "cowshed": 0.60, "farm": 0.65, "farm_auxiliary": 0.60,
    "greenhouse": 0.70, "silo": 0.70, "stable": 0.55, "sty": 0.65,
    "chicken_coop": 0.72, "slaughterhouse": 0.75, "cold_storage": 0.80,
    "kindergarten": 0.72, "chapel": 0.25, "cathedral": 0.25,
    "dormitory": 0.35, "fire_station": 0.55, "prison": 0.55,
}
FRACAO_AUTOCONSUMO_PADRAO = 0.60


def faixa_eui(tipo_edificacao: str | None) -> tuple[float, float, float]:
    """Faixa (mínimo, típico, máximo) de EUI para o segmento."""
    if not tipo_edificacao:
        return EUI_PADRAO
    return EUI_KWH_M2_MES.get(str(tipo_edificacao).strip().lower(), EUI_PADRAO)


def fracao_autoconsumo(tipo_edificacao: str | None) -> float:
    """Fração da geração tipicamente consumida no próprio horário."""
    if not tipo_edificacao:
        return FRACAO_AUTOCONSUMO_PADRAO
    return FRACAO_AUTOCONSUMO.get(str(tipo_edificacao).strip().lower(), FRACAO_AUTOCONSUMO_PADRAO)


def estimar(
    area_telhado_m2: float,
    tipo_edificacao: str | None = None,
    pavimentos: int = 1,
    eui_personalizado: float | None = None,
) -> EstimativaConsumo:
    """
    Estima o consumo a partir da área de telhado e do número de pavimentos.

    A área construída é o telhado multiplicado pelos pavimentos: um edifício de
    escritórios de 10 andares consome dez vezes o que a projeção do telhado
    sugeriria, e o telhado não dá conta -- o que é justamente a informação que
    a triagem precisa dar.
    """
    tipo = (tipo_edificacao or "desconhecido").strip().lower()
    pavimentos = max(1, min(int(pavimentos or 1), FATOR_PAVIMENTOS_MAX))
    area_construida = max(0.0, float(area_telhado_m2)) * pavimentos

    minimo, tipico, maximo = faixa_eui(tipo)
    eui = float(eui_personalizado) if eui_personalizado else tipico

    consumo_mensal = area_construida * eui
    avisos = [
        "Consumo estimado por intensidade de uso de energia (kWh/m2/mês) do segmento, "
        "não por conta de luz. Confirmar com fatura antes de fechar proposta."
    ]

    confianca = "baixa"
    if tipo in EUI_KWH_M2_MES and pavimentos == 1:
        confianca = "média"
    if tipo in {"desconhecido", "yes", ""}:
        confianca = "muito baixa"
        avisos.append(
            "O tipo de edificação não está mapeado no OpenStreetMap; foi usada a faixa genérica. "
            "A incerteza aqui é grande."
        )
    if pavimentos > 1:
        avisos.append(
            f"Área construída estimada em {pavimentos} pavimentos ({area_construida:,.0f} m2). "
            "Se o prédio tiver outro número de andares, o consumo muda proporcionalmente."
        )

    return EstimativaConsumo(
        area_construida_m2=area_construida,
        pavimentos=pavimentos,
        tipo_edificacao=tipo,
        eui_kwh_m2_mes=eui,
        consumo_mensal_kwh=consumo_mensal,
        consumo_anual_kwh=consumo_mensal * 12.0,
        faixa_mensal_kwh=(area_construida * minimo, area_construida * maximo),
        fracao_autoconsumo=fracao_autoconsumo(tipo),
        confianca=confianca,
        avisos=avisos,
    )


def estimar_de_lead(lead, eui_personalizado: float | None = None) -> EstimativaConsumo:
    """
    Estima o consumo de um :class:`~aurum.geo.roofs.RoofLead`.

    Usa ``tipo_para_calculo`` e não ``building_type``: um galpão de fazenda
    quase sempre é ``building=yes`` e só vira agro pelo uso do solo. Pelo tipo
    cru, receberia a faixa genérica de prédio urbano.
    """
    return estimar(
        area_telhado_m2=lead.metrics.area_m2,
        tipo_edificacao=getattr(lead, "tipo_para_calculo", None) or lead.building_type,
        pavimentos=lead.levels,
        eui_personalizado=eui_personalizado,
    )


#: Tarifa média de referência por classe, em R$/kWh, com tributos.
#: Ordens de grandeza para 2026; a tarifa real vem da fatura do cliente.
TARIFA_REFERENCIA_BRL_KWH = {
    "b3_comercial": 0.92,
    "b1_residencial": 0.85,
    "a4_verde": 0.62,
    "a4_azul": 0.58,
}


def tarifa_referencia(tipo_edificacao: str | None, consumo_mensal_kwh: float) -> tuple[float, str]:
    """
    Tarifa de referência e o grupo tarifário provável.

    A separação por demanda é a regra prática: acima de ~150 kW médios o
    cliente está no Grupo A (média tensão), com tarifa de energia menor porém
    com cobrança de demanda contratada -- que o modelo simplificado não trata.
    """
    demanda_media_kw = consumo_mensal_kwh / (22.0 * 10.0)
    if demanda_media_kw >= 150.0:
        return TARIFA_REFERENCIA_BRL_KWH["a4_verde"], "A4 (média tensão, provável)"
    if str(tipo_edificacao or "").lower() in {"residential", "apartments", "house"}:
        return TARIFA_REFERENCIA_BRL_KWH["b1_residencial"], "B1 residencial"
    return TARIFA_REFERENCIA_BRL_KWH["b3_comercial"], "B3 comercial"
