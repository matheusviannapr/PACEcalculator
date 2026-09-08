"""
Os três cenários de uso de uma residência, cada um com o seu sistema.

Uma casa não tem uma curva de carga, e o estudo até aqui escolhia uma delas e
seguia. Escolher é justamente o que não se pode fazer antes de mostrar as
alternativas: o sistema que atende um dia de semana de casa vazia é metade do
que atende uma casa cheia o ano inteiro, e a diferença entre os dois é dinheiro
do cliente — para mais ou para menos.

Os três cenários, e o que cada um responde:

* **Como a vistoria levantou.** O dado de campo sem nenhuma interpretação.
  Serve de piso e de controle: se um cenário transformado sai muito acima
  deste, a diferença tem de ser explicável pela transformação, e não por um
  erro dela.
* **Casa cheia o ano inteiro.** Todo mundo em casa todos os dias — férias,
  home office da família toda, casa de veraneio na temporada. É o teto.
* **Semana quase vazia, fim de semana cheio.** A rotina de quem trabalha fora.
  Fica entre os dois, e é o que a maioria das famílias vive.

**Cada cenário recebe o seu próprio sistema**, e é aí que a comparação passa a
valer alguma coisa. O gerador fotovoltaico é dimensionado pela energia do ano
daquele cenário; o banco, pelo pior dia dele. Dimensionar um sistema só e
mostrá-lo sob três consumos diferentes responderia outra pergunta — quanto
sobraria ou faltaria —, que é útil depois e não antes.

A carga de backup segue a mesma regra dos demais estudos: recorte por
criticidade, equipamento a equipamento, e não por ambiente.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..demanda import ocupacao, simular_ensemble
from ..demanda.cenario import Cenario
from ..demanda.ensemble import EnsembleCarga
from ..pv.kits import (
    BATERIA_BLOCO_BRL,
    BATERIA_BLOCO_KWH,
    MAO_DE_OBRA_BRL_KWP,
    MATERIAL_CA_BRL_KWP,
    capex_de_kit,
)
from .apagao import MalhaApagao
from .catalogo import ConjuntoArmazenamento
from .economia import PremissasBateria
from .escopos import _candidatos_do_escopo, _mais_barato_que_cumpre
from .geracao import SerieGeracao

__all__ = ["CenarioDeUso", "ComparacaoDeUso", "comparar_cenarios_de_uso"]


@dataclass
class CenarioDeUso:
    """Um cenário de uso, com o sistema que ele exige."""

    chave: str
    nome: str
    para_que: str
    #: Quantos dias por semana de cada perfil.
    composicao: dict[str, int]
    #: O perfil de maior pico — é ele que dimensiona o equipamento.
    dimensionante: str

    energia_diaria_kwh: float
    consumo_anual_kwh: float
    pico_p95_kw: float
    curva_w: np.ndarray
    passo_min: int

    potencia_fv_kwp: float = 0.0
    geracao_anual_kwh: float = 0.0
    banco_kwh: float = 0.0
    banco_kw: float = 0.0
    autonomia_h: float = 0.0
    capex_sem_bateria_brl: float = 0.0
    capex_com_bateria_brl: float = 0.0
    #: Economia anual estimada e retorno, à tarifa informada.
    #:
    #: Estimativa direta: a geração substitui compra, ao preço da tarifa, até
    #: o limite do que a casa consome. A conta rigorosa — balanço horário,
    #: autoconsumo, injeção, custo de disponibilidade — fica na seção de
    #: economia, para o cenário sobre o qual o documento foi calculado. Duas
    #: contas convivem desde que se saiba qual é qual.
    economia_anual_brl: float = 0.0
    payback_anos: float | None = None
    conjunto: ConjuntoArmazenamento | None = None
    avisos: list[str] = field(default_factory=list)

    @property
    def consumo_mensal_kwh(self) -> float:
        return self.energia_diaria_kwh * 30.0

    @property
    def cobertura(self) -> float:
        """Quanto da energia do ano o gerador cobre. 1,0 é compensação plena."""
        return self.geracao_anual_kwh / self.consumo_anual_kwh if self.consumo_anual_kwh else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "cenario": self.nome,
            "consumo_kwh_dia": round(self.energia_diaria_kwh, 1),
            "consumo_kwh_mes": round(self.consumo_mensal_kwh, 0),
            "pico_p95_kw": round(self.pico_p95_kw, 2),
            "solar_kwp": round(self.potencia_fv_kwp, 1),
            "geracao_kwh_ano": round(self.geracao_anual_kwh, 0),
            "banco_kwh": round(self.banco_kwh, 1),
            "autonomia_h": round(self.autonomia_h, 1),
            "capex_solar_brl": round(self.capex_sem_bateria_brl, 0),
            "capex_com_bateria_brl": round(self.capex_com_bateria_brl, 0),
            "economia_ano_brl": round(self.economia_anual_brl, 0),
            "payback_anos": round(self.payback_anos, 1) if self.payback_anos else None,
        }


@dataclass
class ComparacaoDeUso:
    """Os três cenários lado a lado."""

    cenarios: list[CenarioDeUso]
    avisos: list[str] = field(default_factory=list)

    def de(self, chave: str) -> CenarioDeUso | None:
        return next((c for c in self.cenarios if c.chave == chave), None)

    @property
    def piso(self) -> CenarioDeUso | None:
        """O menor consumo — quase sempre o levantado em campo."""
        return min(self.cenarios, key=lambda c: c.consumo_anual_kwh, default=None)

    @property
    def teto(self) -> CenarioDeUso | None:
        return max(self.cenarios, key=lambda c: c.consumo_anual_kwh, default=None)

    @property
    def intermediario(self) -> CenarioDeUso | None:
        """
        O do meio, que costuma ser a recomendação.

        Numa residência a resposta quase nunca é o piso nem o teto: o piso
        descreve um dia que a casa não vive sempre, e o teto paga por uma
        ocupação que ela não tem o ano inteiro. Nomear o meio evita que a
        comparação termine sem recomendação nenhuma.
        """
        if len(self.cenarios) < 3:
            return None
        return sorted(self.cenarios, key=lambda c: c.consumo_anual_kwh)[1]

    def tabela(self) -> pd.DataFrame:
        return pd.DataFrame([c.as_dict() for c in self.cenarios])

    def amplitude(self) -> dict[str, float]:
        """
        Quanto os extremos se afastam — a medida de quanto a escolha importa.

        Quando a razão entre teto e piso é pequena, o cenário deixa de ser uma
        decisão e vira detalhe; quando é grande, escolher errado custa caro nos
        dois sentidos, e o cliente precisa ver os três antes de assinar.
        """
        piso, teto = self.piso, self.teto
        if piso is None or teto is None or piso.consumo_anual_kwh <= 0:
            return {}
        return {
            "consumo": teto.consumo_anual_kwh / piso.consumo_anual_kwh,
            "solar_kwp": (teto.potencia_fv_kwp / piso.potencia_fv_kwp
                          if piso.potencia_fv_kwp else float("nan")),
            "capex": (teto.capex_com_bateria_brl / piso.capex_com_bateria_brl
                      if piso.capex_com_bateria_brl else float("nan")),
        }


# ----------------------------------------------------------------------------
def _medir_perfil(
    cenario: Cenario, identificador: str, simulacoes: int, semente: int,
) -> dict[str, Any]:
    """Simula um perfil uma vez e guarda o que os três cenários vão reusar."""
    perfil = ocupacao.PERFIS[identificador]
    ajustado = ocupacao.aplicar(cenario, perfil)
    ajustado.criticidades_essenciais = cenario.criticidades_essenciais
    ensemble = simular_ensemble(
        ajustado.para_comodos(), ajustado.instancias_de(), simulacoes, semente=semente)
    suave = ensemble.reamostrar(15) if ensemble.passo_min == 1 else ensemble
    geral = ensemble.resumo()["geral"]

    # A carga de backup do mesmo perfil, quando há criticidade classificada.
    backup: EnsembleCarga | None = None
    if ajustado.tem_criticidade:
        comodos = ajustado.para_comodos(True)
        if comodos:
            backup = simular_ensemble(
                comodos, ajustado.instancias_de(True), simulacoes, semente=semente)
    return {
        "perfil": perfil,
        "ensemble": ensemble,
        "backup": backup,
        "curva": suave.perfil_medio_w(),
        "passo_min": suave.passo_min,
        "pico_p95_kw": float(geral["pico_p95_kw"]),
        "energia_diaria_kwh": float(geral["energia_diaria_media_kwh"]),
    }


def comparar_cenarios_de_uso(
    cenario: Cenario,
    serie: SerieGeracao,
    catalogo: Any,
    malha: MalhaApagao | None = None,
    autonomia_alvo_h: float = 6.0,
    confiabilidade: float = 0.95,
    simulacoes: int = 150,
    premissas: PremissasBateria | None = None,
    topologia_kit: str | None = "splitphase",
    mao_de_obra_brl_kwp: float = MAO_DE_OBRA_BRL_KWP,
    material_ca_brl_kwp: float = MATERIAL_CA_BRL_KWP,
    bloco_kwh: float = BATERIA_BLOCO_KWH,
    bloco_brl: float = BATERIA_BLOCO_BRL,
    semente: int = 20260902,
    estudos: Sequence[str] | None = None,
    tarifa_brl_kwh: float = 0.0,
) -> ComparacaoDeUso:
    """
    Mede os três cenários e dimensiona solar e bateria para cada um.

    Cada perfil é simulado **uma vez**, e os cenários se servem dessa medida:
    o de fim de semana e o de casa cheia compartilham o perfil ``casa_cheia``,
    e refazer a simulação para os dois custaria o dobro sem mudar nada.
    """
    premissas = premissas or PremissasBateria()
    malha = malha or MalhaApagao()
    escolhidos = list(estudos or ocupacao.ESTUDOS)
    produtividade = serie.anual_kwh_por_kwp()

    necessarios = {
        identificador
        for chave in escolhidos
        for identificador in ocupacao.ESTUDOS[chave]["perfis"]
    }
    medidos = {
        identificador: _medir_perfil(cenario, identificador, simulacoes, semente)
        for identificador in necessarios
    }

    resultados: list[CenarioDeUso] = []
    avisos: list[str] = []

    for chave in escolhidos:
        definicao = ocupacao.ESTUDOS[chave]
        pesos: dict[str, int] = definicao["perfis"]
        dias = sum(pesos.values())

        diaria = sum(medidos[p]["energia_diaria_kwh"] * n for p, n in pesos.items()) / dias
        # O equipamento é do pior dia; a energia do ano é da média ponderada.
        # Trocar um pelo outro é o erro clássico: solar dimensionado pelo fim
        # de semana fica grande demais, banco dimensionado pela média não
        # atravessa o sábado.
        dimensionante = max(pesos, key=lambda p: medidos[p]["pico_p95_kw"])
        medida = medidos[dimensionante]

        consumo_anual = diaria * 365.0
        kwp = consumo_anual / produtividade if produtividade > 0 else 0.0

        # A curva do cenário é a **média ponderada** dos perfis que o compõem,
        # e não a do perfil que dimensiona. As duas coisas são diferentes e a
        # confusão salta aos olhos no gráfico: com a curva do dimensionante, o
        # cenário de fim de semana saía idêntico ao de casa cheia, porque os
        # dois são dimensionados pelo mesmo dia. A energia da curva tem de
        # bater com a energia declarada do cenário, e só a ponderada bate.
        curvas = [np.asarray(medidos[p]["curva"], dtype=float) for p in pesos]
        tamanho = min(len(c) for c in curvas)
        curva = sum(
            c[:tamanho] * n for c, n in zip(curvas, pesos.values())
        ) / dias

        banco = CenarioDeUso(
            chave=chave, nome=definicao["nome"], para_que=definicao["para_que"],
            composicao=dict(pesos), dimensionante=medida["perfil"].nome,
            energia_diaria_kwh=diaria, consumo_anual_kwh=consumo_anual,
            pico_p95_kw=medida["pico_p95_kw"], curva_w=curva,
            passo_min=medida["passo_min"],
            potencia_fv_kwp=kwp, geracao_anual_kwh=kwp * produtividade,
        )

        # O banco, dimensionado sobre a carga de backup do pior dia.
        ensemble_backup = medida["backup"]
        if ensemble_backup is not None:
            energia_alvo = float(
                np.mean(ensemble_backup.perfis())) / 1000.0 * autonomia_alvo_h
            from .catalogo import candidatos_em_blocos

            candidatos = (
                candidatos_em_blocos(catalogo)
                or _candidatos_do_escopo(catalogo, energia_alvo)
            )
            if candidatos:
                conjunto, _, autonomia, _ = _mais_barato_que_cumpre(
                    candidatos, ensemble_backup, serie, malha,
                    autonomia_alvo_h, confiabilidade, semente, premissas)
                if conjunto is not None:
                    banco.conjunto = conjunto
                    banco.banco_kwh = float(conjunto.energia_util_kwh)
                    banco.banco_kw = float(conjunto.potencia_descarga_kw)
                    banco.autonomia_h = autonomia
        else:
            banco.avisos.append(
                "Sem criticidade classificada, este cenário não dimensiona banco: "
                "não há como saber o que o quadro de backup deveria atender."
            )

        sem_bateria = capex_de_kit(
            kwp, topologia_kit or "splitphase",
            mao_de_obra_brl_kwp, material_ca_brl_kwp)
        com_bateria = capex_de_kit(
            kwp, topologia_kit or "splitphase",
            mao_de_obra_brl_kwp, material_ca_brl_kwp,
            bateria_kwh=banco.banco_kwh, bloco_kwh=bloco_kwh, bloco_brl=bloco_brl)
        if sem_bateria is None or com_bateria is None:
            banco.avisos.append(
                f"A tabela de kit não cobre {kwp:.1f} kWp nessa topologia; o "
                "investimento deste cenário ficou de fora da comparação."
            )
        banco.capex_sem_bateria_brl = float(sem_bateria or 0.0)
        banco.capex_com_bateria_brl = float(com_bateria or 0.0)

        # O dinheiro de cada cenário. Sem ele a tabela responde "quanto custa"
        # e não responde "vale a pena", que é a pergunta que se faz.
        if tarifa_brl_kwh > 0:
            aproveitada = min(banco.geracao_anual_kwh, banco.consumo_anual_kwh)
            banco.economia_anual_brl = aproveitada * float(tarifa_brl_kwh)
            if banco.economia_anual_brl > 0:
                banco.payback_anos = banco.capex_com_bateria_brl / banco.economia_anual_brl

        avisos.extend(banco.avisos)
        resultados.append(banco)

    return ComparacaoDeUso(cenarios=resultados, avisos=list(dict.fromkeys(avisos)))
