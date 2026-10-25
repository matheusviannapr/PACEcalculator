"""
O que a imagem do pré-relatório mostra — e nada além disso.

Este é o contrato entre o cálculo (``aurum.pre_relatorio.estudo``) e o desenho
(``aurum.pre_relatorio.imagem``). O desenho não calcula: recebe números
prontos, já arredondados para o que se diz a um cliente, e o texto de cada
bloco já escolhido para casa ou negócio.

**De onde vem cada número**, para ninguém inventar um segundo caminho:

* autonomia e cobertura das faltas — a resiliência do conjunto recomendado
  pelo estudo de bateria (``ResultadoEstudo.recomendado``);
* o dia em energia — a curva híbrida (``aurum.pre_relatorio.curva``) e a série
  solar do estudo;
* para onde vai a conta — a tabela de cargas da curva híbrida, escalada pela
  conta de luz;
* conta hoje e com solar — ``FinanceiroProposta`` e ``Fatura`` do estudo, que
  são a mesma fonte do dossiê completo.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Cobertura:
    """Numa falta de luz de ``duracao_h`` horas, a chance de a bateria segurar tudo."""

    duracao_h: float
    prob: float          # 0..1

    @property
    def rotulo(self) -> str:
        if self.duracao_h < 1:
            return f"{self.duracao_h * 60:.0f} min"
        return f"{self.duracao_h:.0f} h"


@dataclass(frozen=True)
class Consumidor:
    """Uma fatia da conta: ``nome`` como o cliente reconhece, ``fracao`` de 0 a 1."""

    nome: str
    fracao: float
    estimado: bool = True   # potência típica, não medida


@dataclass
class ResumoPreRelatorio:
    # -- quem ---------------------------------------------------------------
    cliente: str
    protocolo: str
    cidade_uf: str
    tipo: str = "casa"                      # "casa" | "negocio"
    empresa: str = ""
    setor: str = ""                         # "Padaria ou confeitaria"…
    data: str = ""                          # "27/09/2026"

    # -- bloco 1: quando a luz cai --------------------------------------------
    #: O que o cliente disse que fez falta, com artigo, na ordem de criticidade.
    criticos: list[str] = field(default_factory=list)
    bateria_kwh: float | None = None
    #: Horas que a bateria segura as cargas críticas com 95% de confiança,
    #: no pior horário de início. ``None`` quando o estudo não recomendou banco.
    autonomia_h: float | None = None
    #: Chance de atender por duração de falta, das mais curtas às mais longas.
    cobertura: list[Cobertura] = field(default_factory=list)
    #: A falta que o cliente descreveu ("uma ou duas horas") e a sua duração.
    falta_declarada: str = ""
    falta_declarada_h: float | None = None

    # -- bloco 2: o dia em energia -------------------------------------------
    curva_consumo_kw: list[float] = field(default_factory=list)   # 24 valores
    curva_solar_kw: list[float] = field(default_factory=list)     # 24 valores
    curva_critica_kw: list[float] = field(default_factory=list)   # 24 valores
    #: Fração do consumo do dia que acontece sem sol (0..1).
    fracao_sem_sol: float | None = None

    # -- bloco 3: para onde vai a conta --------------------------------------
    #: Os maiores consumidores e, por último, "Demais cargas".
    consumidores: list[Consumidor] = field(default_factory=list)

    # -- bloco 4: a conta ----------------------------------------------------
    consumo_mensal_kwh: float | None = None
    conta_atual_brl: float | None = None
    conta_solar_min_brl: float | None = None
    conta_solar_max_brl: float | None = None
    potencia_fv_kwp: float | None = None
    #: Só aparece quando o operador pede (decisão comercial).
    investimento_brl: float | None = None
    payback_anos: float | None = None

    # -- índice de oportunidade (aurum.pre_relatorio.indice) ------------------
    indice: int | None = None                 # 0..100
    indice_faixa: str = ""                    # "baixa" | "moderada" | "alta" | "muito alta"
    #: As partes do índice: {"nome", "pontos", "teto", "detalhe"}.
    indice_partes: list[dict] = field(default_factory=list)

    # -- honestidade ---------------------------------------------------------
    #: De onde saiu o consumo: "conta" (kWh lido), "valor" (R$ ÷ tarifa) ou "respostas".
    origem_consumo: str = "conta"
    avisos: list[str] = field(default_factory=list)

    @property
    def negocio(self) -> bool:
        return self.tipo == "negocio"

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["negocio"] = self.negocio
        return d


def exemplo(tipo: str = "casa") -> ResumoPreRelatorio:
    """Um resumo plausível, para desenhar e testar a imagem sem rodar o estudo."""
    import math

    consumo = [0.35, 0.3, 0.28, 0.28, 0.3, 0.45, 0.9, 1.1, 0.7, 0.55, 0.55, 0.7,
               0.8, 0.65, 0.55, 0.55, 0.65, 0.9, 1.4, 1.8, 1.9, 1.6, 1.1, 0.6]
    solar = [0.0] * 24
    for h in range(6, 18):
        solar[h] = round(3.2 * math.sin(math.pi * (h - 5.5) / 12.5), 2)
    critica = [0.18] * 24
    base = dict(
        cliente="Maria Souza", protocolo="PACE-X9KYZI", cidade_uf="Niterói/RJ", data="27/09/2026",
        criticos=["a geladeira", "a internet", "o ar-condicionado"], bateria_kwh=10.0, autonomia_h=5.0,
        cobertura=[Cobertura(1, 1.0), Cobertura(3, 0.97), Cobertura(6, 0.81), Cobertura(12, 0.42)],
        falta_declarada="uma ou duas horas", falta_declarada_h=2.0,
        curva_consumo_kw=consumo, curva_solar_kw=solar, curva_critica_kw=critica, fracao_sem_sol=0.62,
        consumidores=[Consumidor("Chuveiro", 0.32), Consumidor("Ar-condicionado", 0.24),
                      Consumidor("Geladeira", 0.09), Consumidor("Demais cargas", 0.35)],
        consumo_mensal_kwh=620, conta_atual_brl=1250.0, conta_solar_min_brl=180.0, conta_solar_max_brl=320.0,
        potencia_fv_kwp=5.5, origem_consumo="conta",
        indice=78, indice_faixa="alta", indice_partes=[
            {"nome": "Economia", "pontos": 27.0, "teto": 35.0, "detalhe": "solar tira ~80% da conta"},
            {"nome": "Falta de luz", "pontos": 21.0, "teto": 30.0, "detalhe": "algo não pode ficar sem luz"},
            {"nome": "Consumo sem sol", "pontos": 10.0, "teto": 15.0, "detalhe": "62% do consumo sem sol"},
            {"nome": "Encaixe da solução", "pontos": 20.0, "teto": 20.0, "detalhe": "segura todas as faltas declaradas"}],
    )
    if tipo == "negocio":
        base.update(tipo="negocio", empresa="Padaria Pão Quente", setor="Padaria ou confeitaria",
                    cliente="Carla Mendes", criticos=["o balcão refrigerado", "a maquininha", "o forno de pães"],
                    consumidores=[Consumidor("Forno de pães", 0.38), Consumidor("Balcão refrigerado", 0.16),
                                  Consumidor("Ar-condicionado", 0.12), Consumidor("Demais cargas", 0.34)],
                    consumo_mensal_kwh=2100, conta_atual_brl=2400.0, conta_solar_min_brl=420.0,
                    conta_solar_max_brl=700.0, potencia_fv_kwp=16.0)
    return ResumoPreRelatorio(**base)
