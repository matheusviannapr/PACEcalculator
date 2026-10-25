"""
Comparação de fontes: solar, bateria e gerador, ligados e desligados.

A pergunta que este módulo responde não é "qual bateria comprar", e sim a que
vem antes dela: **vale a pena comprar alguma coisa, e o quê**. O cliente tem
uma conta de luz e um problema de falta de energia; entre não fazer nada e
comprar tudo existem seis arranjos intermediários, e cada um resolve uma parte
diferente do problema por um preço diferente.

As três fontes têm naturezas distintas, e é isso que faz a comparação valer:

* **Solar** reduz a conta e não dá backup nenhum. Inversor conectado à rede
  desliga quando a rede cai -- é exigência de anti-ilhamento da ABNT NBR
  16149, não limitação de marca. Vender solar como segurança energética é o
  engano mais comum do setor, e o quadro de cenários o desfaz sozinho: a
  coluna de autonomia do cenário "só solar" é zero.
* **Bateria** dá backup e, com tarifa simples, quase nada de economia -- não
  há diferença de tarifa para arbitrar. O que ela economiza é o pedaço que o
  Fio B come da energia injetada, guardando o excedente do meio-dia para a
  noite. É pouco. Quem paga a bateria, na tarifa simples, é a resiliência.
* **Gerador** é o inverso da bateria: energia praticamente ilimitada, potência
  limitada, e um custo por kWh que só aparece quando ele roda. Atravessa 36 h
  sem esforço e não ajuda em nada na conta de luz.

Todo cenário é medido contra o mesmo referencial -- **só a rede**, com a conta
que o cliente paga hoje -- e as três respostas saem lado a lado: quanto custa,
quanto economiza por ano, e o que acontece quando a luz cai.

Tarifa simples apenas, de propósito. Tarifa branca e Grupo A mudam a conta da
bateria por completo (a arbitragem passa a existir) e exigem a modelagem de
posto tarifário e demanda contratada, que :mod:`aurum.bateria.economia` já
tem. Aqui o objetivo é a decisão de arranjo, não a otimização do despacho.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from datetime import date
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ..demanda.ensemble import EnsembleCarga
from ..pv.financials import FRACAO_FIO_B_NA_TARIFA, _tir, estimar_capex, fator_fio_b
from .apagao import MalhaApagao, ResultadoResiliencia, avaliar_limites
from .catalogo import ConjuntoArmazenamento
from .degradacao import ModeloDegradacao
from .despacho import Gerador, LimitesDespacho
from .economia import PremissasBateria
from .geracao import SerieGeracao

LOGGER = logging.getLogger(__name__)

#: Confiabilidade exigida para declarar autonomia no quadro de cenários.
#: A mesma do estudo principal, e pelo mesmo motivo: prometer autonomia
#: média é prometer o que não se cumpre na noite em que a luz cai.
CONFIABILIDADE_AUTONOMIA = 0.95

__all__ = [
    "Composicao",
    "ComparacaoFontes",
    "Fatura",
    "Gerador",
    "ResultadoCenario",
    "comparar_fontes",
    "estudo_do_gerador",
]


# ----------------------------------------------------------------------------
# Entradas
# ----------------------------------------------------------------------------
def _ano_corrente() -> int:
    return date.today().year


@dataclass(frozen=True)
class Fatura:
    """
    A conta de luz do cliente, na tarifa simples (convencional, Grupo B).

    ``consumo_mensal_kwh`` é o que a fatura diz. Quando informado, ele manda:
    a simulação de Monte Carlo é reescalada para bater com ele, porque o
    levantamento de cargas erra e a fatura não. Quando ausente, vale o consumo
    que a própria simulação produziu -- e o estudo diz que foi assim.

    **Como a energia injetada vira desconto** (Lei 14.300/2022, art. 27, para
    quem se conecta a partir de 2023):

    * o que o sol entrega na hora do consumo não passa pelo medidor e evita a
      tarifa cheia;
    * o que sobra é injetado e vira crédito em kWh, que abate o consumo
      comprado da rede — inclusive a bandeira, que incide sobre o faturado
      depois da compensação;
    * cada kWh **compensado** paga uma fração crescente do Fio B: 60% em 2026,
      75% em 2027, 90% em 2028 e, por hipótese conservadora, 100% daí em
      diante (:func:`aurum.pv.financials.fator_fio_b`);
    * a conta nunca cai abaixo do custo de disponibilidade, e o crédito que
      sobraria abaixo dele não tem onde ser usado.

    A compensação é anual: o crédito do verão abate o inverno, como o sistema
    de compensação permite (os créditos valem por 60 meses). Isto descreve o
    ano típico em regime, não o primeiro mês de operação.
    """

    tarifa_brl_kwh: float = 0.98
    consumo_mensal_kwh: float | None = None
    #: Custo de disponibilidade em kWh: 30 monofásico, 50 bifásico, 100
    #: trifásico. É o piso da conta -- nenhum sistema leva a fatura a zero.
    custo_disponibilidade_kwh: float = 100.0
    #: Adicional de bandeira tarifária, em R$/kWh, somado à tarifa em todo kWh
    #: faturado. Zero quando a tarifa informada já é a média da fatura, com a
    #: bandeira dentro — o caso comum, e o que a tela recomenda.
    bandeira_brl_kwh: float = 0.0
    #: Participação do Fio B (TUSD de distribuição) na tarifa, sem bandeira.
    #: Varia por distribuidora; 28% é a ordem de grandeza da baixa tensão.
    participacao_fio_b: float = FRACAO_FIO_B_NA_TARIFA
    #: Ano em que o sistema é ligado. Decide em que degrau do cronograma do
    #: Fio B o primeiro ano do fluxo de caixa começa.
    ano_conexao: int = field(default_factory=_ano_corrente)
    escalada_tarifaria_ano: float = 0.05

    @property
    def tarifa_efetiva_brl_kwh(self) -> float:
        """O que 1 kWh comprado da rede custa: tarifa mais bandeira."""
        return float(self.tarifa_brl_kwh) + max(0.0, float(self.bandeira_brl_kwh))

    def fio_b_brl_kwh(self, ano_calendario: int | None = None) -> float:
        """O que cada kWh compensado paga de Fio B no ano pedido (preços do ano 1)."""
        ano = self.ano_conexao if ano_calendario is None else int(ano_calendario)
        participacao = float(np.clip(self.participacao_fio_b, 0.0, 1.0))
        return float(self.tarifa_brl_kwh) * participacao * fator_fio_b(ano)

    def valor_injecao_no_ano(self, ano_calendario: int | None = None) -> float:
        """Quanto vale 1 kWh injetado e compensado, já descontado o Fio B."""
        return self.tarifa_efetiva_brl_kwh - self.fio_b_brl_kwh(ano_calendario)

    @property
    def valor_injecao_brl_kwh(self) -> float:
        """O valor do kWh compensado no ano de conexão."""
        return self.valor_injecao_no_ano(None)

    @property
    def consumo_anual_kwh(self) -> float | None:
        return None if self.consumo_mensal_kwh is None else self.consumo_mensal_kwh * 12.0

    def detalhar_conta(
        self,
        compra_kwh_ano: float,
        injecao_kwh_ano: float = 0.0,
        ano_calendario: int | None = None,
    ) -> dict[str, float]:
        """
        A conta do ano, parcela a parcela — é o que a memória de cálculo imprime.

        ``compra`` é o que o medidor registra entrando; ``injecao``, saindo. O
        compensado é o menor entre o crédito e o que há acima do piso; o que
        passa disso é crédito sem uso.
        """
        piso = max(0.0, float(self.custo_disponibilidade_kwh)) * 12.0
        compra = max(0.0, float(compra_kwh_ano))
        credito = max(0.0, float(injecao_kwh_ano))
        compensado = min(credito, max(0.0, compra - piso))
        faturado = max(piso, compra - compensado)
        tarifa = self.tarifa_efetiva_brl_kwh
        fio_b = self.fio_b_brl_kwh(ano_calendario)
        energia = faturado * tarifa
        custo_fio_b = compensado * fio_b
        return {
            "compra_kwh": compra,
            "injecao_kwh": credito,
            "piso_kwh": piso,
            "compensado_kwh": compensado,
            "credito_sem_uso_kwh": credito - compensado,
            "faturado_kwh": faturado,
            "tarifa_brl_kwh": tarifa,
            "fio_b_brl_kwh": fio_b,
            "energia_brl": energia,
            "fio_b_brl": custo_fio_b,
            "conta_brl": energia + custo_fio_b,
        }

    def conta_anual_brl(
        self,
        compra_kwh_ano: float,
        injecao_kwh_ano: float = 0.0,
        ano_calendario: int | None = None,
    ) -> float:
        """
        A conta do ano, com o piso do custo de disponibilidade.

        O piso vale sobre o ano (doze mínimos): com a compensação anual, o
        crédito de um mês sobrando abate o mês faltando, e o que a conta não
        pode é ficar abaixo de doze mínimos. ``ano_calendario`` escolhe o
        degrau do Fio B; vazio, é o ano de conexão.
        """
        return self.detalhar_conta(compra_kwh_ano, injecao_kwh_ano, ano_calendario)["conta_brl"]

    def as_dict(self) -> dict[str, Any]:
        return {
            "tarifa_brl_kwh": self.tarifa_brl_kwh,
            "bandeira_brl_kwh": self.bandeira_brl_kwh,
            "tarifa_efetiva_brl_kwh": self.tarifa_efetiva_brl_kwh,
            "consumo_mensal_kwh": self.consumo_mensal_kwh,
            "custo_disponibilidade_kwh": self.custo_disponibilidade_kwh,
            "participacao_fio_b": self.participacao_fio_b,
            "ano_conexao": self.ano_conexao,
            "fio_b_brl_kwh": self.fio_b_brl_kwh(),
            "valor_injecao_brl_kwh": self.valor_injecao_brl_kwh,
            "escalada_tarifaria_ano": self.escalada_tarifaria_ano,
        }


@dataclass(frozen=True)
class Composicao:
    """Quais fontes existem num cenário."""

    solar: bool = False
    bateria: bool = False
    gerador: bool = False

    @property
    def chave(self) -> str:
        partes = [n for n, on in (("solar", self.solar), ("bateria", self.bateria), ("gerador", self.gerador)) if on]
        return "+".join(partes) if partes else "rede"

    @property
    def nome(self) -> str:
        rotulos = [n for n, on in (("solar", self.solar), ("bateria", self.bateria), ("gerador", self.gerador)) if on]
        if not rotulos:
            return "Somente a rede"
        return "Rede + " + " + ".join(rotulos)

    @property
    def n_fontes(self) -> int:
        return int(self.solar) + int(self.bateria) + int(self.gerador)

    def as_dict(self) -> dict[str, Any]:
        return {"solar": self.solar, "bateria": self.bateria, "gerador": self.gerador}


def combinacoes(
    solar: bool = True, bateria: bool = True, gerador: bool = True
) -> list[Composicao]:
    """
    Todos os arranjos possíveis com as fontes que o cliente aceita considerar.

    O cenário vazio -- só a rede -- entra sempre, porque é o referencial contra
    o qual todos os outros são medidos. Sem ele não existe economia: existe
    apenas um custo.
    """
    disponiveis = []
    for s in (False, True) if solar else (False,):
        for b in (False, True) if bateria else (False,):
            for g in (False, True) if gerador else (False,):
                disponiveis.append(Composicao(s, b, g))
    return sorted(disponiveis, key=lambda c: (c.n_fontes, c.chave))


# ----------------------------------------------------------------------------
# Saída
# ----------------------------------------------------------------------------
@dataclass
class ResultadoCenario:
    """Um arranjo de fontes, medido por dinheiro e por resiliência."""

    composicao: Composicao
    capex_brl: float

    # -- energia do ano típico --------------------------------------------
    consumo_kwh_ano: float
    geracao_fv_kwh_ano: float
    autoconsumo_kwh_ano: float
    injecao_kwh_ano: float
    compra_da_rede_kwh_ano: float
    gerador_em_paralelo_kwh_ano: float

    # -- dinheiro ----------------------------------------------------------
    conta_anual_brl: float
    custo_operacional_brl_ano: float
    economia_anual_brl: float
    valor_resiliencia_brl_ano: float

    # -- resiliência -------------------------------------------------------
    autonomia_garantida_h: float
    prob_por_duracao: dict[float, float]
    ens_por_evento_kwh: float
    ens_evitada_kwh_ano: float
    combustivel_em_apagao_brl_ano: float

    # -- finanças ----------------------------------------------------------
    fluxo: pd.DataFrame
    vpl_brl: float
    tir: float | None
    payback_anos: float | None
    resiliencia: ResultadoResiliencia | None = None
    avisos: list[str] = field(default_factory=list)
    #: Valor presente somando o valor da resiliência ao caixa. Serve para a
    #: decisão de arranjo; a proposta e o payback usam o caixa (`vpl_brl`).
    vpl_com_resiliencia_brl: float = 0.0
    #: O balanço energético bruto do ano típico — é dele que a memória de
    #: cálculo refaz a conta de luz.
    balanco: dict[str, float] = field(default_factory=dict)
    #: A economia de cada ano a preços do ano 1, com o Fio B e a degradação
    #: do ano (ver :func:`economias_por_ano`).
    economias_por_ano: list[dict[str, float]] = field(default_factory=list)
    #: O que o fluxo precisa para ser refeito com outro investimento — o de
    #: cada topologia de kit que aponta para este cenário.
    ciclos_ano: float = 0.0
    capex_baterias_brl: float = 0.0
    custo_grupo_paralelo_brl_ano: float = 0.0
    opex_fixo_brl_ano: float = 0.0

    @property
    def nome(self) -> str:
        return self.composicao.nome

    @property
    def payback_simples_anos(self) -> float | None:
        """Investimento dividido pela economia do primeiro ano — sem mais nada."""
        if self.capex_brl <= 0 or self.economia_anual_brl <= 0:
            return None
        return self.capex_brl / self.economia_anual_brl

    @property
    def autoconsumo_frac(self) -> float:
        """Fração da geração consumida no próprio imóvel (direto ou via banco)."""
        return self.autoconsumo_kwh_ano / self.geracao_fv_kwh_ano if self.geracao_fv_kwh_ano > 0 else 0.0

    @property
    def beneficio_anual_brl(self) -> float:
        """Economia de conta mais valor da resiliência, menos o que custa operar."""
        return (
            self.economia_anual_brl
            + self.valor_resiliencia_brl_ano
            - self.custo_operacional_brl_ano
            - self.combustivel_em_apagao_brl_ano
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "cenario": self.nome,
            "chave": self.composicao.chave,
            **self.composicao.as_dict(),
            "capex_brl": self.capex_brl,
            "consumo_kwh_ano": self.consumo_kwh_ano,
            "geracao_fv_kwh_ano": self.geracao_fv_kwh_ano,
            "autoconsumo_kwh_ano": self.autoconsumo_kwh_ano,
            "injecao_kwh_ano": self.injecao_kwh_ano,
            "compra_da_rede_kwh_ano": self.compra_da_rede_kwh_ano,
            "conta_anual_brl": self.conta_anual_brl,
            "economia_anual_brl": self.economia_anual_brl,
            "custo_operacional_brl_ano": self.custo_operacional_brl_ano,
            "valor_resiliencia_brl_ano": self.valor_resiliencia_brl_ano,
            "beneficio_anual_brl": self.beneficio_anual_brl,
            "autonomia_garantida_h": self.autonomia_garantida_h,
            "ens_por_evento_kwh": self.ens_por_evento_kwh,
            "vpl_brl": self.vpl_brl,
            "vpl_com_resiliencia_brl": self.vpl_com_resiliencia_brl,
            "tir": self.tir,
            "payback_simples_anos": self.payback_simples_anos,
            "payback_anos": self.payback_anos,
        }


@dataclass
class ComparacaoFontes:
    """Os cenários lado a lado, com o referencial destacado."""

    cenarios: list[ResultadoCenario]
    base: ResultadoCenario
    fatura: Fatura
    premissas: PremissasBateria
    gerador: Gerador | None
    conjunto: ConjuntoArmazenamento | None
    potencia_fv_kwp: float
    avisos: list[str] = field(default_factory=list)
    #: O consumo anual que a curva simulada produziu, antes de ser escalada, e
    #: o fator que a levou ao consumo de referência da conta. Guardados para a
    #: memória de cálculo citar o mesmo número que o aviso cita.
    consumo_simulado_kwh: float = 0.0
    fator_consumo: float = 1.0

    def tabela(self) -> pd.DataFrame:
        return pd.DataFrame([c.as_dict() for c in self.cenarios])

    def por_chave(self, chave: str) -> ResultadoCenario | None:
        return next((c for c in self.cenarios if c.composicao.chave == chave), None)

    @property
    def melhor_vpl(self) -> ResultadoCenario | None:
        """
        O arranjo de maior valor presente entre os que investem alguma coisa.

        O referencial fica de fora porque o VPL dele é zero por construção, e
        um empate técnico com o "não faça nada" precisa ser lido na tabela, não
        resolvido por arredondamento.
        """
        investem = [c for c in self.cenarios if c.capex_brl > 0]
        return max(investem, key=lambda c: c.vpl_brl) if investem else None

    @property
    def melhor_resiliencia(self) -> ResultadoCenario | None:
        investem = [c for c in self.cenarios if c.capex_brl > 0]
        if not investem:
            return None
        # A autonomia garantida empata em zero sempre que nenhum arranjo
        # atravessa a falta inteira em 95% dos casos — comum em carga
        # essencial pesada. O desempate vai para a energia que efetivamente
        # deixa de faltar, que é o estrago real, e só depois para o preço.
        return max(
            investem,
            key=lambda c: (c.autonomia_garantida_h, -c.ens_por_evento_kwh, -c.capex_brl),
        )

    def resumo(self) -> dict[str, Any]:
        return {
            "fatura": self.fatura.as_dict(),
            "potencia_fv_kwp": self.potencia_fv_kwp,
            "gerador": self.gerador.as_dict() if self.gerador else None,
            "conjunto": self.conjunto.descricao() if self.conjunto else None,
            "conta_atual_brl_ano": self.base.conta_anual_brl,
            "cenarios": [c.as_dict() for c in self.cenarios],
            "melhor_vpl": self.melhor_vpl.nome if self.melhor_vpl else None,
            "melhor_resiliencia": self.melhor_resiliencia.nome if self.melhor_resiliencia else None,
            "avisos": self.avisos,
        }


# ----------------------------------------------------------------------------
# Balanço anual conectado à rede
# ----------------------------------------------------------------------------
def _balanco_anual(
    ensemble: EnsembleCarga,
    serie: SerieGeracao,
    potencia_fv_kwp: float,
    limites: LimitesDespacho | None,
    gerador: Gerador | None,
    fatura: Fatura,
    reserva_backup_frac: float,
    dias_por_estacao: int | None,
    semente: int,
    fator_carga: float = 1.0,
    fv_no_hibrido: bool = True,
) -> dict[str, float]:
    """
    Um ano típico com a rede disponível, dia a dia, para um arranjo de fontes.

    **A geração percorre a série inteira**, e não uma amostra dela: cada dia
    real da série solar entra uma vez, pareado com um dia de carga do
    ensemble da mesma estação. Sorteando 45 dias por estação, a geração anual
    do balanço saía alguns por cento diferente de ``kWp × produtividade`` — e
    o slide imprimia um número, a seção de compensação outro, do mesmo
    sistema. ``dias_por_estacao`` inteiro volta ao sorteio antigo (é mais
    rápido, e os testes de forma o usam); ``None`` é o balanço exato.

    O que o balanço devolve fecha duas contas, e a memória de cálculo confere
    as duas: ``consumo = autoconsumo + compra + gerador`` e
    ``geracao = autoconsumo_direto + recarga + injecao``.

    ``fv_no_hibrido`` diz por onde o sol entra. Num sistema residencial o
    híbrido é o único inversor, e o que passa dele é perdido. Num sistema com
    inversor de rede próprio o acoplamento é em corrente alternada: o gerador
    entrega no quadro, o híbrido só carrega o banco com o excedente, e cortar
    a geração pela entrada CC dele é descrever outra usina — 106 kWp
    aparecendo como 22 kW porque o híbrido de backup aceita 5 kW de painel.

    ``fator_carga`` escala a carga simulada para a que a fatura mede. Ele entra
    **aqui**, antes do despacho, e não sobre o resultado: escalar o resultado
    multiplicaria junto a geração, o autoconsumo e a injeção, que vêm da
    potência instalada e da série solar e nada têm a ver com o erro do
    levantamento — uma usina de 2,8 kWp saía do estudo gerando 871 kWh/ano no
    lugar de 3.600. Escalada a carga, o despacho refaz sozinho o que sobra e o
    que falta.

    A ordem de despacho é a única que faz sentido economicamente com tarifa
    simples: o sol atende a carga, o excedente carrega o banco, o que sobra é
    injetado, e a bateria devolve à noite o que guardou. Não há carga pela
    rede, porque sem diferença de tarifa entre horários a arbitragem só perde
    dinheiro nas duas conversões.

    O gerador só entra em paralelo à rede se o kWh dele for mais barato que o
    da distribuidora -- o que quase nunca acontece com diesel a preço de 2026,
    e é exatamente por isso que a regra é explícita em vez de suposta.
    """
    rng = np.random.default_rng(semente)
    passos_dia = ensemble.passos_por_dia
    dt_h = ensemble.passo_min / 60.0

    tem_banco = limites is not None and limites.energia_util_dc_kwh > 0
    capacidade = limites.energia_util_dc_kwh if tem_banco else 0.0
    piso = capacidade * float(np.clip(reserva_backup_frac, 0.0, 0.95))
    ger_paralelo_kw = (
        float(gerador.potencia_kw)
        if gerador is not None and gerador.custo_energia_brl_kwh < fatura.tarifa_brl_kwh
        else 0.0
    )

    totais = dict.fromkeys(
        ["consumo", "geracao", "geracao_bruta", "autoconsumo", "autoconsumo_direto",
         "recarga", "injecao", "compra", "gerador", "descarga"],
        0.0,
    )
    exato = dias_por_estacao is None
    dias_somados = 0
    # Índice da hora de cada passo do dia: degrau, como o PVGIS afirma.
    hora_do_passo = (np.arange(passos_dia) * ensemble.passo_min) // 60

    for estacao in ensemble.estacoes:
        if exato:
            dias_serie = serie.dias_da_estacao(estacao)
            if dias_serie.size == 0:
                continue
            quantidade = int(dias_serie.size)
            geracao = serie.potencia_w_por_kwp[dias_serie][:, hora_do_passo]
            # Cada simulação de carga entra o mesmo número de vezes (±1): o
            # consumo do balanço fica igual à média do ensemble, sem ruído
            # de sorteio.
            total_sim = ensemble.perfis_w[estacao].shape[0]
            indices = np.resize(rng.permutation(total_sim), quantidade)
            carga = ensemble.perfis_w[estacao][indices]
        else:
            quantidade = int(dias_por_estacao)
            carga, _ = ensemble.amostrar_dias(estacao, quantidade, rng)
            geracao = serie.amostrar_janelas(
                estacao, 0, passos_dia, ensemble.passo_min, quantidade, rng
            )
        dias_somados += quantidade
        carga = carga / 1000.0 * float(fator_carga)
        geracao = geracao * float(potencia_fv_kwp) / 1000.0
        geracao_bruta = geracao
        if limites is not None and limites.potencia_fv_kw > 0 and fv_no_hibrido:
            geracao = np.minimum(geracao, limites.potencia_fv_kw)

        def rodar(soc: np.ndarray, acumular: bool) -> np.ndarray:
            """Roda o dia inteiro; só a segunda passada entra na conta."""
            for t in range(passos_dia):
                demanda = carga[:, t]
                pv = geracao[:, t]

                direto = np.minimum(pv, demanda)
                sobra = pv - direto
                falta = demanda - direto

                descarga = np.zeros_like(falta)
                recarga = np.zeros_like(sobra)
                if tem_banco:
                    disponivel = np.maximum(0.0, soc - piso) * limites.eficiencia_descarga / dt_h
                    descarga = np.minimum(
                        np.minimum(falta, limites.potencia_descarga_kw), disponivel
                    )
                    soc = soc - descarga * dt_h / limites.eficiencia_descarga
                    falta = falta - descarga

                    espaco = np.maximum(0.0, capacidade - soc) / dt_h / limites.eficiencia_carga
                    recarga = np.minimum(np.minimum(sobra, limites.potencia_carga_kw), espaco)
                    soc = np.clip(soc + recarga * dt_h * limites.eficiencia_carga, 0.0, capacidade)
                    sobra = sobra - recarga

                do_grupo = np.minimum(falta, ger_paralelo_kw)
                falta = falta - do_grupo

                if acumular:
                    totais["consumo"] += demanda.sum() * dt_h
                    totais["geracao"] += pv.sum() * dt_h
                    totais["geracao_bruta"] += geracao_bruta[:, t].sum() * dt_h
                    totais["autoconsumo"] += (direto + descarga).sum() * dt_h
                    totais["autoconsumo_direto"] += direto.sum() * dt_h
                    totais["recarga"] += recarga.sum() * dt_h
                    totais["injecao"] += sobra.sum() * dt_h
                    totais["compra"] += falta.sum() * dt_h
                    totais["gerador"] += do_grupo.sum() * dt_h
                    totais["descarga"] += descarga.sum() * dt_h
            return soc

        # Uma passada de aquecimento antes de contar. O banco não pode começar
        # o dia com energia que ninguém pagou: sem sol e sem carga pela rede,
        # um estado inicial arbitrário vira economia fantasma, e o cenário
        # "bateria sem solar" apareceria economizando o que não economiza.
        # A segunda passada começa onde a primeira terminou, que é o regime
        # permanente de um banco em uso diário.
        soc_inicial = np.full(quantidade, piso)
        rodar(rodar(soc_inicial, acumular=False), acumular=True)

    # Dias somados viram ano: 365,25 dias por ano, a mesma convenção de
    # `SerieGeracao.anual_kwh_por_kwp` — é o que faz a geração do balanço
    # bater com kWp × produtividade até a última casa.
    escala = 365.25 / dias_somados if dias_somados else 0.0
    return {chave: valor * escala for chave, valor in totais.items()}


def _fator_de_fatura(consumo_simulado_kwh_ano: float, fatura: Fatura) -> tuple[float, str | None]:
    """
    Quanto a simulação precisa ser esticada para bater com a conta de luz.

    O levantamento de cargas é uma estimativa; a fatura é medida. Quando as
    duas discordam em mais de 10%, o estudo escala a simulação para a fatura e
    diz que fez isso -- calar seria apresentar como consumo do cliente um
    número que a conta dele desmente.
    """
    alvo = fatura.consumo_anual_kwh
    if alvo is None or consumo_simulado_kwh_ano <= 0:
        return 1.0, None
    fator = alvo / consumo_simulado_kwh_ano
    if abs(fator - 1.0) < 0.10:
        return fator, None
    # O separador de milhar vira ponto só no número: aplicar o replace na
    # frase inteira comeria as vírgulas do texto.
    simulado = f"{consumo_simulado_kwh_ano:,.0f}".replace(",", ".")
    medido = f"{alvo:,.0f}".replace(",", ".")
    return fator, (
        f"A curva simulada dá {simulado} kWh/ano e o consumo de referência da conta "
        f"(a fatura, ou o consumo anual que dimensionou o solar), {medido} kWh/ano — "
        f"diferença de {abs(fator - 1) * 100:.0f}%. Os valores de conta e economia "
        "foram escalados para o consumo de referência, mantendo a forma da curva; a "
        "análise de potência e de resiliência continua na curva simulada, que é quem "
        "descreve o pico."
    )


# ----------------------------------------------------------------------------
# Resiliência do arranjo
# ----------------------------------------------------------------------------
def _resiliencia_do_cenario(
    composicao: Composicao,
    conjunto: ConjuntoArmazenamento | None,
    gerador: Gerador | None,
    ensemble_backup: EnsembleCarga,
    serie: SerieGeracao,
    malha: MalhaApagao,
    semente: int,
) -> ResultadoResiliencia | None:
    """
    Varre a malha de apagões para o arranjo, ou devolve ``None`` quando é óbvio.

    Sem bateria e sem gerador não há o que simular: a carga fica sem energia do
    primeiro minuto ao último, seja qual for a potência instalada em painéis.
    """
    if not composicao.bateria and not composicao.gerador:
        return None

    if composicao.bateria and conjunto is not None:
        limites = LimitesDespacho.do_conjunto(conjunto)
        fv_kw = conjunto.potencia_fv_aproveitavel_kw if composicao.solar else 0.0
        limites = replace(limites, potencia_fv_kw=fv_kw)
    else:
        # Sem banco, o FV não ilha: o inversor conectado à rede desliga junto
        # com ela. O gerador é a única fonte, e ele entra pelo parâmetro.
        limites = LimitesDespacho.sem_bateria(0.0)

    return avaliar_limites(
        limites,
        limites.potencia_fv_kw,
        ensemble_backup,
        serie,
        malha,
        semente,
        gerador=gerador if composicao.gerador else None,
        conjunto=conjunto if composicao.bateria else None,
        rotulo=composicao.nome,
    )


def _interpolar_por_duracao(
    resiliencia: ResultadoResiliencia | None, coluna: str, duracao_h: float
) -> float:
    """Lê uma coluna da tabela por duração na duração média de interrupção."""
    if resiliencia is None or resiliencia.tabela.empty:
        return float("nan")
    resumo = resiliencia.por_duracao().sort_values("duracao_h")
    return float(
        np.interp(
            float(duracao_h),
            resumo["duracao_h"].to_numpy(dtype=float),
            resumo[coluna].to_numpy(dtype=float),
        )
    )


# ----------------------------------------------------------------------------
# Fluxo de caixa do cenário
# ----------------------------------------------------------------------------
def economias_por_ano(
    balanco: Mapping[str, float],
    base_balanco: Mapping[str, float],
    fatura: Fatura,
    anos: int,
    degradacao_fv_ano: float = 0.0,
    tem_solar: bool = True,
) -> list[dict[str, float]]:
    """
    A economia de conta de cada ano, **a preços do ano 1**.

    Duas coisas mudam de um ano para o outro sem que a tarifa mude, e as duas
    ficavam de fora quando a economia do primeiro ano era só inflacionada:

    * **o Fio B sobe** — 60% em 2026, 90% em 2028, 100% daí em diante. Num
      sistema residencial, que injeta metade do que gera, é o que mais pesa;
    * **o módulo perde rendimento**, ~0,5% ao ano. Menos geração é menos
      autoconsumo (volta a ser compra) e menos crédito.

    A escalada tarifária entra depois, em :func:`_fluxo_do_cenario`, sobre
    tudo. Cada linha traz as parcelas para a memória de cálculo.
    """
    conta_base = fatura.conta_anual_brl(base_balanco["compra"], 0.0)
    linhas = []
    for ano in range(1, int(anos) + 1):
        calendario = int(fatura.ano_conexao) + ano - 1
        fator = (1.0 - float(degradacao_fv_ano)) ** (ano - 1) if tem_solar else 1.0
        # O autoconsumo que o módulo degradado deixa de fazer volta a ser
        # comprado; a injeção encolhe na mesma proporção. É uma aproximação
        # de primeira ordem, exata para perdas pequenas, e evita refazer o
        # balanço horário quinze vezes.
        compra = balanco["compra"] + balanco["autoconsumo"] * (1.0 - fator)
        injecao = balanco["injecao"] * fator
        detalhe = fatura.detalhar_conta(compra, injecao, calendario)
        linhas.append({
            "ano": ano,
            "ano_calendario": calendario,
            "fator_geracao": fator,
            "fio_b_brl_kwh": detalhe["fio_b_brl_kwh"],
            "compensado_kwh": detalhe["compensado_kwh"],
            "conta_base_brl": conta_base,
            "conta_brl": detalhe["conta_brl"],
            "economia_brl": conta_base - detalhe["conta_brl"],
        })
    return linhas


def _payback(caixa: Sequence[float]) -> float | None:
    """O ano (fracionário) em que o acumulado do caixa cruza o zero."""
    acumulado = np.cumsum(caixa)
    for i in range(1, len(acumulado)):
        if acumulado[i] >= 0 and acumulado[i - 1] < 0:
            fatia = caixa[i]
            return float(i - 1 + (-acumulado[i - 1] / fatia)) if fatia else float(i)
    return None


def _fluxo_do_cenario(
    capex_brl: float,
    economias_reais_brl: Sequence[float],
    valor_resiliencia_brl_ano: float,
    custo_operacional_brl_ano: float,
    combustivel_brl_ano: float,
    premissas: PremissasBateria,
    fatura: Fatura,
    degradacao: ModeloDegradacao | None,
    ciclos_ano: float,
    capex_baterias_brl: float,
) -> tuple[pd.DataFrame, float, float | None, float | None, float]:
    """
    Fluxo de caixa incremental do arranjo contra o cenário "só a rede".

    ``economias_reais_brl`` é a economia de conta de cada ano a preços do ano
    1 (ver :func:`economias_por_ano`). Tudo que acompanha a tarifa --
    economia de conta e valor da resiliência -- sobe com a escalada
    tarifária. O que é custo de operação sobe junto: um fluxo que inflaciona
    só o benefício fabrica retorno.

    **O caixa é só o que passa pelo bolso.** Valor presente, TIR e payback
    saem de ``fluxo_brl`` — economia de conta menos operação e troca do
    banco. O valor da resiliência (energia que deixa de faltar × custo da
    interrupção) não é dinheiro que o cliente vê na fatura: fica na coluna
    própria e num valor presente à parte, ``vpl_com_resiliencia``. Somá-lo ao
    caixa fazia a proposta mostrar "economia de R$ X por ano" ao lado de um
    payback que não era ``investimento ÷ X`` — o defeito do cartão do
    Ed. Mourisco, que misturava duas economias num número só.

    A troca do banco entra como desembolso no ano em que a retenção cruza o
    fim de vida útil, com o preço do kWh de bateria corrigido pela queda
    histórica. Sem ela, um horizonte de 15 anos faz qualquer bateria parecer
    eterna.
    """
    linhas: list[dict[str, Any]] = []
    idade = 0.0
    ciclos = 0.0
    economias = list(economias_reais_brl)
    for ano in range(1, premissas.anos_analise + 1):
        inflacao = (1.0 + fatura.escalada_tarifaria_ano) ** (ano - 1)
        retencao = 1.0
        troca = 0.0
        if degradacao is not None:
            idade += 1.0
            ciclos += ciclos_ano
            retencao = degradacao.retencao(idade, ciclos)
            if retencao <= degradacao.retencao_fim_vida:
                troca = capex_baterias_brl * (1.0 - premissas.queda_preco_bateria_ano) ** (ano - 1)
                idade, ciclos = 0.0, 0.0

        # A economia de conta que vem do sol não depende do banco; a que vem
        # do banco, sim. Aplicar a retenção ao total seria pessimista demais,
        # e ignorá-la, otimista. O meio-termo honesto é aplicá-la ao valor da
        # resiliência, que é inteiramente da bateria.
        real = economias[ano - 1] if ano - 1 < len(economias) else (economias[-1] if economias else 0.0)
        beneficio = real * inflacao
        resiliencia = valor_resiliencia_brl_ano * retencao * inflacao
        custos = (custo_operacional_brl_ano + combustivel_brl_ano) * inflacao
        caixa_do_ano = beneficio - custos - troca
        linhas.append({
            "ano": ano,
            "retencao": retencao,
            "economia_conta_brl": beneficio,
            "valor_resiliencia_brl": resiliencia,
            "custo_operacional_brl": custos,
            "troca_brl": troca,
            "fluxo_brl": caixa_do_ano,
            "fluxo_com_resiliencia_brl": caixa_do_ano + resiliencia,
        })

    fluxo = pd.DataFrame(linhas)
    taxa = premissas.taxa_desconto_ano
    caixa = [-capex_brl, *fluxo["fluxo_brl"].tolist()]
    vpl = float(sum(v / (1.0 + taxa) ** i for i, v in enumerate(caixa)))
    com_resiliencia = [-capex_brl, *fluxo["fluxo_com_resiliencia_brl"].tolist()]
    vpl_resiliencia = float(sum(v / (1.0 + taxa) ** i for i, v in enumerate(com_resiliencia)))

    return (
        fluxo, vpl, (_tir(caixa) if capex_brl > 0 else None), _payback(caixa),
        vpl_resiliencia,
    )


# ----------------------------------------------------------------------------
# O financeiro de uma proposta: um investimento, um fluxo, um conjunto de números
# ----------------------------------------------------------------------------
@dataclass
class FinanceiroProposta:
    """
    Os números de dinheiro de **uma** proposta, todos do mesmo fluxo.

    É a única fonte do que a tela, o dossiê e os slides dizem sobre retorno.
    Antes, cada um calculava o seu: a tela dividia investimento por economia,
    o slide principal lia o payback do cenário (com a resiliência dentro e o
    investimento da topologia mais barata), os slides de valor mostravam o
    preço da própria topologia ao lado da TIR de outra. O mesmo sistema saía
    com três paybacks.

    * ``payback_simples_anos`` = ``capex / economia_ano1`` — a conta que o
      cliente faz de cabeça, e a que o rótulo "payback simples" promete;
    * ``payback_anos`` = quando o caixa acumulado cruza o zero, com reajuste
      tarifário, degradação do módulo, Fio B crescente, operação e troca do
      banco;
    * ``tir`` e ``vpl_brl`` saem desse mesmo caixa, com **este** capex.
    """

    nome: str
    cenario: str
    capex_brl: float
    economia_ano1_brl: float
    opex_ano1_brl: float
    payback_simples_anos: float | None
    payback_anos: float | None
    tir: float | None
    vpl_brl: float
    vpl_com_resiliencia_brl: float
    valor_resiliencia_ano1_brl: float
    autonomia_h: float
    horizonte_anos: int
    fluxo: pd.DataFrame
    topologia: str | None = None

    @property
    def economia_mensal_brl(self) -> float:
        return self.economia_ano1_brl / 12.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "nome": self.nome,
            "topologia": self.topologia,
            "cenario": self.cenario,
            "capex_brl": self.capex_brl,
            "economia_ano1_brl": self.economia_ano1_brl,
            "opex_ano1_brl": self.opex_ano1_brl,
            "payback_simples_anos": self.payback_simples_anos,
            "payback_anos": self.payback_anos,
            "tir": self.tir,
            "vpl_brl": self.vpl_brl,
            "vpl_com_resiliencia_brl": self.vpl_com_resiliencia_brl,
            "valor_resiliencia_ano1_brl": self.valor_resiliencia_ano1_brl,
            "autonomia_h": self.autonomia_h,
            "horizonte_anos": self.horizonte_anos,
        }


def financeiro_da_proposta(
    comparacao: "ComparacaoFontes",
    cenario: ResultadoCenario,
    capex_brl: float | None = None,
    nome: str | None = None,
    topologia: str | None = None,
) -> FinanceiroProposta:
    """
    O fluxo de caixa de um cenário refeito com o investimento de uma proposta.

    Duas propostas podem apontar para o mesmo cenário — microinversor e
    split-phase sem banco produzem a mesma energia —, e o que muda entre elas
    é só o preço. A economia é a do cenário; a operação (percentual do
    investimento), o payback, a TIR e o valor presente são refeitos com o
    ``capex_brl`` desta proposta. Sem ``capex_brl``, vale o do cenário.
    """
    premissas = comparacao.premissas
    capex = float(cenario.capex_brl if capex_brl is None else capex_brl)
    opex = capex * premissas.opex_percent_do_capex_ano + cenario.opex_fixo_brl_ano
    degradacao = (
        ModeloDegradacao.do_conjunto(comparacao.conjunto)
        if cenario.composicao.bateria and comparacao.conjunto is not None else None
    )
    economias = [a["economia_brl"] for a in cenario.economias_por_ano] or [cenario.economia_anual_brl]
    fluxo, vpl, tir, payback, vpl_resiliencia = _fluxo_do_cenario(
        capex, economias, cenario.valor_resiliencia_brl_ano,
        opex + cenario.custo_grupo_paralelo_brl_ano,
        cenario.combustivel_em_apagao_brl_ano,
        premissas, comparacao.fatura, degradacao, cenario.ciclos_ano,
        cenario.capex_baterias_brl,
    )
    economia = float(cenario.economia_anual_brl)
    return FinanceiroProposta(
        nome=nome or cenario.nome,
        topologia=topologia,
        cenario=cenario.composicao.chave,
        capex_brl=capex,
        economia_ano1_brl=economia,
        opex_ano1_brl=opex,
        payback_simples_anos=(capex / economia) if capex > 0 and economia > 0 else None,
        payback_anos=payback,
        tir=tir,
        vpl_brl=vpl,
        vpl_com_resiliencia_brl=vpl_resiliencia,
        valor_resiliencia_ano1_brl=float(cenario.valor_resiliencia_brl_ano),
        autonomia_h=float(cenario.autonomia_garantida_h),
        horizonte_anos=int(premissas.anos_analise),
        fluxo=fluxo,
    )


# ----------------------------------------------------------------------------
# O custo de rodar o gerador
# ----------------------------------------------------------------------------
def estudo_do_gerador(comparacao: "ComparacaoFontes") -> pd.DataFrame:
    """
    Quanto combustível o grupo queima, e quanto isso custa, por duração.

    É a pergunta que sobra depois de dimensionar o grupo para suprir a carga:
    a potência deixa de ser incerteza e o custo passa a ser o assunto. A
    tabela sai por duração de falta porque é assim que a conta muda -- 1 h de
    gerador é troco, 36 h é uma decisão de orçamento.

    A comparação entre "só gerador" e "gerador com bateria" aparece de graça:
    com banco, o grupo entra depois do sol e da bateria, e queima menos. É o
    argumento econômico da dupla, e ele não existe sem esta tabela.
    """
    grupo = comparacao.gerador
    if grupo is None:
        return pd.DataFrame()

    premissas = comparacao.premissas
    linhas: list[dict[str, Any]] = []
    for cenario in comparacao.cenarios:
        if not cenario.composicao.gerador or cenario.resiliencia is None:
            continue
        for linha in cenario.resiliencia.por_duracao().itertuples():
            energia = float(getattr(linha, "energia_gerador_kwh", 0.0))
            horas = energia / grupo.potencia_kw if grupo.potencia_kw > 0 else float("nan")
            linhas.append({
                "cenario": cenario.nome,
                "duracao_h": float(linha.duracao_h),
                "energia_gerada_kwh": energia,
                "horas_equivalentes_a_plena_carga": horas,
                "custo_do_evento_brl": grupo.custo_da_energia_brl(energia),
                "prob_atendimento": float(linha.prob_atendimento),
                "energia_nao_suprida_kwh": float(linha.ens_medio_kwh),
            })

    tabela = pd.DataFrame(linhas)
    if tabela.empty:
        return tabela
    # O custo anual usa a frequência de interrupções declarada: é o número que
    # entra no orçamento, e ele só existe quando alguém diz quantas faltas por
    # ano a região tem.
    tabela["custo_anual_brl"] = (
        tabela["custo_do_evento_brl"] * premissas.interrupcoes_por_ano
    )
    return tabela


# ----------------------------------------------------------------------------
# Orquestração
# ----------------------------------------------------------------------------
def comparar_fontes(
    ensemble_total: EnsembleCarga,
    ensemble_backup: EnsembleCarga,
    serie: SerieGeracao,
    potencia_fv_kwp: float,
    fatura: Fatura,
    conjunto: ConjuntoArmazenamento | None = None,
    gerador: Gerador | None = None,
    premissas: PremissasBateria | None = None,
    malha: MalhaApagao | None = None,
    capex_fv_brl: float | None = None,
    padrao_capex: str | None = None,
    topologia_kit: str | None = None,
    mao_de_obra_brl_kwp: float | None = None,
    material_ca_brl_kwp: float | None = None,
    composicoes: Sequence[Composicao] | None = None,
    dias_por_estacao: int | None = None,
    semente: int = 20260902,
    fv_no_hibrido: bool = True,
    capex_fv_por_cenario: Mapping[str, float] | None = None,
    progresso=None,
) -> ComparacaoFontes:
    """
    Avalia cada arranjo de fontes e devolve os cenários lado a lado.

    ``capex_fv_por_cenario`` dá a cada arranjo o preço da **sua** topologia:
    o gerador sem bateria é um kit de microinversor e o com bateria é um kit
    split-phase, e eles não custam o mesmo por kWp. Um preço único para os
    dois faria a bateria parecer de graça num e caríssima no outro.

    ``padrao_capex`` escolhe a referência de R$/kWp do sistema fotovoltaico
    entre os :data:`aurum.pv.financials.PADROES_CAPEX`. O mesmo kWp custa
    coisas muito diferentes num galpão e num prédio ocupado de alto padrão, e
    tratar os dois pela mesma curva erra o payback em quase 50%.

    ``conjunto`` é o armazenamento a considerar nos cenários com bateria --
    tipicamente o recomendado pelo estudo. ``gerador`` é o grupo a considerar
    nos cenários com gerador. Quando um deles é ``None``, os cenários que o
    exigiriam simplesmente não são gerados: é melhor um quadro com quatro
    linhas verdadeiras que um com oito e metade inventada.
    """
    premissas = premissas or PremissasBateria()
    malha = malha or MalhaApagao()
    avisos: list[str] = []

    lista = list(
        composicoes
        if composicoes is not None
        else combinacoes(
            solar=potencia_fv_kwp > 0, bateria=conjunto is not None, gerador=gerador is not None
        )
    )
    if Composicao() not in lista:
        lista.insert(0, Composicao())

    # -- referencial: consumo simulado × fatura ----------------------------
    base_balanco = _balanco_anual(
        ensemble_total, serie, 0.0, None, None, fatura,
        premissas.reserva_backup_frac, dias_por_estacao, semente,
    )
    consumo_simulado = float(base_balanco["consumo"])
    fator, aviso_fatura = _fator_de_fatura(consumo_simulado, fatura)
    if aviso_fatura:
        avisos.append(aviso_fatura)
    if fator != 1.0:
        # O referencial "só a rede" é todo carga: escalar as suas grandezas é
        # a mesma coisa que refazer o balanço com a carga escalada, e sai de
        # graça. Os demais cenários recebem o fator dentro do despacho.
        base_balanco = {k: v * fator for k, v in base_balanco.items()}

    limites_conjunto = LimitesDespacho.do_conjunto(conjunto) if conjunto is not None else None
    # Cotação de verdade passa na frente de qualquer curva; sem ela, o padrão
    # construtivo escolhe a referência de R$/kWp.
    capex_fv = (
        float(capex_fv_brl)
        if capex_fv_brl is not None
        else estimar_capex(
            potencia_fv_kwp, padrao=padrao_capex,
            topologia=topologia_kit,
            mao_de_obra_brl_kwp=mao_de_obra_brl_kwp,
            material_ca_brl_kwp=material_ca_brl_kwp,
        )
    )
    # Duas contas para o mesmo banco, porque o inversor pertence a uma ou a
    # outra conforme o arranjo. Num sistema com solar, o híbrido veio no kit e
    # o banco custa só os blocos de expansão; num sistema **só** de bateria,
    # não há kit nenhum, e o inversor é compra à parte. Uma conta só erraria
    # um dos dois cenários, e os dois aparecem lado a lado no quadro final.
    capex_bat_com_kit, capex_baterias = _capex_do_conjunto(
        conjunto, premissas, topologia_kit)
    capex_bat_sozinho, _ = _capex_do_conjunto(conjunto, premissas, None)
    modelo_degradacao = (
        ModeloDegradacao.do_conjunto(conjunto) if conjunto is not None else None
    )

    resultados: list[ResultadoCenario] = []
    base: ResultadoCenario | None = None

    for i, composicao in enumerate(lista):
        if progresso is not None:
            progresso(f"Cenário: {composicao.nome}", (i + 1) / len(lista))

        balanco = (
            base_balanco
            if composicao.n_fontes == 0
            else _balanco_anual(
                ensemble_total,
                serie,
                potencia_fv_kwp if composicao.solar else 0.0,
                limites_conjunto if composicao.bateria else None,
                gerador if composicao.gerador else None,
                fatura,
                premissas.reserva_backup_frac,
                dias_por_estacao,
                semente,
                fator_carga=fator,
                fv_no_hibrido=fv_no_hibrido,
            )
        )
        conta = fatura.conta_anual_brl(balanco["compra"], balanco["injecao"])

        capex_bat = capex_bat_com_kit if composicao.solar else capex_bat_sozinho
        capex_fv_do_cenario = float(
            (capex_fv_por_cenario or {}).get(composicao.chave, capex_fv)
        )
        capex = (
            (capex_fv_do_cenario if composicao.solar else 0.0)
            + (capex_bat if composicao.bateria else 0.0)
            + (float(gerador.capex_brl) if composicao.gerador and gerador else 0.0)
        )
        opex = capex * premissas.opex_percent_do_capex_ano + (
            float(gerador.opex_fixo_brl_ano) if composicao.gerador and gerador else 0.0
        )
        custo_grupo_paralelo = (
            balanco["gerador"] * float(gerador.custo_energia_brl_kwh)
            if composicao.gerador and gerador
            else 0.0
        )

        resiliencia = _resiliencia_do_cenario(
            composicao, conjunto, gerador, ensemble_backup, serie, malha, semente
        )
        autonomia = (
            resiliencia.autonomia_garantida_h(CONFIABILIDADE_AUTONOMIA, pior_caso=True)
            if resiliencia is not None
            else 0.0
        )
        duracao = premissas.duracao_media_interrupcao_h
        ens_evento = _interpolar_por_duracao(resiliencia, "ens_medio_kwh", duracao)
        ger_evento = _interpolar_por_duracao(resiliencia, "energia_gerador_kwh", duracao)

        # Sem nenhuma fonte de backup, tudo que a carga essencial pediria no
        # apagão é energia não suprida: é o referencial da resiliência.
        carga_backup_media_kw = float(np.mean(ensemble_backup.perfis())) / 1000.0
        ens_sem_nada = carga_backup_media_kw * duracao
        if resiliencia is None:
            ens_evento = ens_sem_nada
            ger_evento = 0.0
        evitada_ano = max(0.0, ens_sem_nada - ens_evento) * premissas.interrupcoes_por_ano
        # Por evento quando informado, por kWh quando não — a mesma regra da
        # análise econômica. Sem esta linha, o quadro de arranjos continuava
        # valorando por energia enquanto a seção econômica valorava por evento,
        # e o mesmo banco aparecia com R$ 18 numa página e R$ 4.730 na outra.
        if premissas.custo_interrupcao_brl_evento > 0:
            # A fração do evento que o backup cobre é a fração de energia que
            # ele supre: atravessar metade do apagão evita metade do
            # transtorno, e não o transtorno inteiro.
            fracao = (
                max(0.0, ens_sem_nada - ens_evento) / ens_sem_nada
                if ens_sem_nada > 0 else 0.0
            )
            valor_resiliencia = (
                fracao * premissas.interrupcoes_por_ano
                * premissas.custo_interrupcao_brl_evento
            )
        else:
            valor_resiliencia = evitada_ano * premissas.custo_interrupcao_brl_kwh
        combustivel_apagao = (
            float(np.nan_to_num(ger_evento))
            * premissas.interrupcoes_por_ano
            * float(gerador.custo_energia_brl_kwh)
            if composicao.gerador and gerador
            else 0.0
        )

        economia = (base.conta_anual_brl - conta) if base is not None else 0.0
        ciclos_ano = _ciclos_estimados(balanco, limites_conjunto) if composicao.bateria else 0.0
        anuais = (
            economias_por_ano(
                balanco, base.balanco, fatura, premissas.anos_analise,
                degradacao_fv_ano=premissas.degradacao_fv_ano,
                tem_solar=composicao.solar,
            )
            if base is not None else []
        )
        fluxo, vpl, tir, payback, vpl_resiliencia = _fluxo_do_cenario(
            capex,
            [a["economia_brl"] for a in anuais] or [0.0],
            valor_resiliencia,
            opex + custo_grupo_paralelo,
            combustivel_apagao,
            premissas,
            fatura,
            modelo_degradacao if composicao.bateria else None,
            ciclos_ano,
            capex_baterias,
        )

        avisos_cenario: list[str] = []
        if composicao.solar and not composicao.bateria:
            avisos_cenario.append(
                "Solar sem bateria não fornece backup: o inversor conectado à rede "
                "desliga quando a rede cai, por exigência de anti-ilhamento."
            )

        cenario = ResultadoCenario(
            composicao=composicao,
            capex_brl=capex,
            consumo_kwh_ano=balanco["consumo"],
            geracao_fv_kwh_ano=balanco["geracao"],
            autoconsumo_kwh_ano=balanco["autoconsumo"],
            injecao_kwh_ano=balanco["injecao"],
            compra_da_rede_kwh_ano=balanco["compra"],
            gerador_em_paralelo_kwh_ano=balanco["gerador"],
            conta_anual_brl=conta,
            custo_operacional_brl_ano=opex + custo_grupo_paralelo,
            economia_anual_brl=economia,
            valor_resiliencia_brl_ano=valor_resiliencia,
            autonomia_garantida_h=autonomia,
            prob_por_duracao=(
                {
                    float(l.duracao_h): float(l.prob_atendimento)
                    for l in resiliencia.por_duracao().itertuples()
                }
                if resiliencia is not None
                else {float(d): 0.0 for d in malha.duracoes_h}
            ),
            ens_por_evento_kwh=float(np.nan_to_num(ens_evento)),
            ens_evitada_kwh_ano=evitada_ano,
            combustivel_em_apagao_brl_ano=combustivel_apagao,
            fluxo=fluxo,
            vpl_brl=vpl,
            tir=tir,
            payback_anos=payback,
            resiliencia=resiliencia,
            avisos=avisos_cenario,
            vpl_com_resiliencia_brl=vpl_resiliencia,
            balanco=dict(balanco),
            economias_por_ano=anuais,
            ciclos_ano=ciclos_ano,
            capex_baterias_brl=capex_baterias if composicao.bateria else 0.0,
            custo_grupo_paralelo_brl_ano=custo_grupo_paralelo,
            opex_fixo_brl_ano=(
                float(gerador.opex_fixo_brl_ano) if composicao.gerador and gerador else 0.0
            ),
        )
        resultados.append(cenario)
        if composicao.n_fontes == 0:
            base = cenario

    assert base is not None, "o cenário de referência é sempre montado primeiro"
    # As duas unidades contam. O custo por evento existe porque o por kWh serve
    # mal em residência — quanto melhor o recorte de criticidade, menor o
    # quadro, menos energia falta, e menor o valor que o modelo dá à bateria —,
    # e olhar só para o segundo fazia o aviso disparar exatamente em quem usa o
    # modelo bom, dizendo que a resiliência não entra em payback nenhum
    # enquanto o payback ao lado tinha sido calculado com ela dentro.
    if (premissas.custo_interrupcao_brl_kwh <= 0
            and premissas.custo_interrupcao_brl_evento <= 0):
        avisos.append(
            "O custo da interrupção foi deixado em zero, nas duas unidades — por kWh "
            "não suprido e por evento evitado —, então o valor da resiliência não "
            "entra em nenhum payback. Com tarifa simples é ele que paga a bateria: sem "
            "informá-lo, o quadro compara apenas economia de conta, e a bateria aparece "
            "pior do que é para quem não pode ficar sem energia."
        )

    return ComparacaoFontes(
        cenarios=resultados,
        base=base,
        fatura=fatura,
        premissas=premissas,
        gerador=gerador,
        conjunto=conjunto,
        potencia_fv_kwp=potencia_fv_kwp,
        avisos=avisos,
        consumo_simulado_kwh=consumo_simulado,
        fator_consumo=float(fator),
    )


def _capex_do_conjunto(
    conjunto: ConjuntoArmazenamento | None,
    premissas: PremissasBateria,
    topologia_kit: str | None = None,
) -> tuple[float, float]:
    """
    (CAPEX instalado, custo só das baterias) — o segundo é o que se troca.

    Com o kit split-phase escolhido, o banco é contado em **blocos de
    expansão** e o inversor **não** entra: ele já veio no kit fotovoltaico, e
    cobrá-lo outra vez dobrava o preço do armazenamento num sistema que tem as
    duas coisas. A própria tabela de preço confirma o modelo — a diferença
    entre a coluna split-phase e a coluna com 5 kWh é o bloco, e nada mais.

    Fora desse caminho vale a conta de sempre: bateria por R$/kWh mais
    inversor, porque aí o inversor é mesmo uma compra à parte.
    """
    if conjunto is None:
        return 0.0, 0.0
    from ..pv.kits import TOPOLOGIA_COM_BATERIA, preco_da_bateria
    from .economia import _capex

    if topologia_kit == TOPOLOGIA_COM_BATERIA:
        # Pela capacidade **nominal**, e não pela útil: o bloco é vendido pela
        # placa, e um banco de 5 kWh nominais entrega 4,6 kWh úteis. Contar
        # pelos úteis pedia um segundo bloco onde um basta.
        # O bloco das premissas, e não o padrão do módulo de kits: a premissa
        # é a única fonte do preço do banco, e com um bloco que não é de 5 kWh
        # o padrão devolvia outro número — 12,8 kWh viravam três blocos de 5.
        if premissas.bloco_bateria_kwh > 0 and premissas.bloco_bateria_brl > 0:
            baterias = preco_da_bateria(
                conjunto.capacidade_nominal_kwh,
                premissas.bloco_bateria_kwh, premissas.bloco_bateria_brl)
        else:
            baterias = preco_da_bateria(conjunto.capacidade_nominal_kwh)
        if baterias > 0:
            # Sem os 35% de instalação: o bloco já é preço de módulo pronto,
            # com caixa, BMS e a instalação do módulo dentro.
            return baterias, baterias

    # Sem topologia, o inversor é compra à parte — e dizer isso aqui é
    # necessário. `premissas.inversor_no_kit_fv` é global do estudo e fica
    # ligada quando há solar; sem esta linha, o arranjo "Rede + bateria", que
    # não tem kit fotovoltaico nenhum, recebia o híbrido de graça e aparecia
    # por R$ 12.000 num quadro em que ele custa o banco mais o inversor.
    if topologia_kit is None and premissas.inversor_no_kit_fv:
        from dataclasses import replace as _replace

        premissas = _replace(premissas, inversor_no_kit_fv=False)
    return _capex(conjunto, premissas)


def _ciclos_estimados(balanco: dict[str, float], limites: LimitesDespacho | None) -> float:
    """Ciclos equivalentes por ano, a partir do que a bateria deslocou."""
    if limites is None or limites.energia_util_ca_kwh <= 0:
        return 0.0
    return max(0.0, balanco["descarga"]) / limites.energia_util_ca_kwh
