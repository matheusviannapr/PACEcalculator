"""
Análise econômica do sistema fotovoltaico.

Diferenças relevantes em relação à versão anterior:

* **Lei 14.300/2022.** O modelo antigo valorava 100% da energia gerada pela
  tarifa cheia. A lei instituiu a cobrança gradual do Fio B sobre a energia
  *injetada* na rede, e mantém o custo de disponibilidade (30/50/100 kWh, ou a
  demanda contratada no Grupo A) independentemente da geração. Ignorar os dois
  superestima a economia -- em sistema com muita injeção, na casa de 15%.

* **Autoconsumo explícito.** A economia da energia consumida na hora em que é
  gerada vale a tarifa cheia; a injetada vale a tarifa menos o Fio B. A
  fração de autoconsumo depende do perfil de carga e é parâmetro, não
  suposição escondida.

* **TIR sobre o fluxo real.** A bissecção antiga recalculava um fluxo próprio,
  que podia divergir do fluxo apresentado na proposta. Aqui a TIR é calculada
  sobre exatamente o mesmo vetor de caixa que a tabela mostra.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from ..config import SolarDefaults, get_settings

#: Fator médio de emissão do Sistema Interligado Nacional, em tCO2eq/MWh.
#: Fonte: inventário do MCTI para o SIN. A rede brasileira é predominantemente
#: hidrelétrica, por isso o valor é baixo para padrões internacionais.
FATOR_EMISSAO_SIN_TCO2_MWH = 0.0839

#: Absorção anual média de CO2 por árvore nativa em crescimento, em kg.
#: Serve só para a equivalência ilustrativa da proposta.
ABSORCAO_ARVORE_KG_CO2_ANO = 100.0

#: Cronograma de cobrança do Fio B sobre a energia injetada, conforme o
#: art. 27 da Lei 14.300/2022, para quem se conectou a partir de 2023.
CRONOGRAMA_FIO_B: dict[int, float] = {
    2023: 0.15, 2024: 0.30, 2025: 0.45, 2026: 0.60, 2027: 0.75, 2028: 0.90,
}
#: A partir de 2029 a regra passa a ser definida por regulamentação
#: específica; adota-se 100% como hipótese conservadora.
FIO_B_APOS_TRANSICAO = 1.00

#: Participação típica do Fio B (TUSD distribuição) na tarifa cheia de baixa
#: tensão no Brasil. Varia por distribuidora; é parâmetro ajustável.
FRACAO_FIO_B_NA_TARIFA = 0.28

#: Consumo mínimo faturado (custo de disponibilidade) por tipo de ligação,
#: em kWh/mês -- art. 98 da REN 1.000/2021.
CUSTO_DISPONIBILIDADE_KWH = {"monofasico": 30, "bifasico": 50, "trifasico": 100}


def fator_fio_b(ano_calendario: int) -> float:
    """Fração do Fio B cobrada sobre a energia injetada, no ano informado."""
    if ano_calendario in CRONOGRAMA_FIO_B:
        return CRONOGRAMA_FIO_B[ano_calendario]
    if ano_calendario < min(CRONOGRAMA_FIO_B):
        return 0.0
    return FIO_B_APOS_TRANSICAO


@dataclass
class PremissasEconomicas:
    """Parâmetros do modelo financeiro."""

    tarifa_brl_kwh: float
    capex_brl: float
    #: Fração da geração consumida no próprio horário de produção.
    #: Indústria/comércio diurno fica entre 0,55 e 0,85; residencial, 0,20-0,35.
    fracao_autoconsumo: float = 0.65
    opex_anual_brl: float | None = None
    degradacao_anual: float = 0.006
    escalada_tarifa: float = 0.05
    taxa_desconto: float = 0.10
    anos: int = 25
    ano_conexao: int = 2026
    tipo_ligacao: str = "trifasico"
    aplicar_lei_14300: bool = True
    fracao_fio_b_na_tarifa: float = FRACAO_FIO_B_NA_TARIFA
    #: Consumo anual do cliente, quando conhecido. Limita a economia: não se
    #: economiza mais do que se gasta.
    consumo_anual_kwh: float | None = None

    def opex(self, defaults: SolarDefaults | None = None) -> float:
        if self.opex_anual_brl is not None:
            return float(self.opex_anual_brl)
        defaults = defaults or get_settings().solar
        return self.capex_brl * defaults.opex_percent_of_capex


@dataclass
class LinhaFluxo:
    """Um ano do fluxo de caixa."""

    ano: int
    ano_calendario: int
    geracao_kwh: float
    tarifa_brl_kwh: float
    economia_autoconsumo_brl: float
    economia_injecao_brl: float
    custo_fio_b_brl: float
    opex_brl: float
    fluxo_liquido_brl: float
    fluxo_descontado_brl: float
    vpl_acumulado_brl: float

    @property
    def economia_bruta_brl(self) -> float:
        return self.economia_autoconsumo_brl + self.economia_injecao_brl


@dataclass
class ResultadoEconomico:
    """Indicadores e fluxo de caixa completo."""

    premissas: PremissasEconomicas
    fluxo: list[LinhaFluxo]
    economia_ano1_brl: float
    payback_simples_anos: float | None
    payback_descontado_anos: float | None
    vpl_brl: float
    tir_anual: float | None
    roi_percent: float | None
    geracao_total_kwh: float
    economia_total_brl: float
    co2_evitado_t: float
    arvores_equivalentes: int
    avisos: list[str] = field(default_factory=list)

    @property
    def lcoe_brl_kwh(self) -> float | None:
        """
        Custo nivelado da energia: CAPEX e OPEX descontados divididos pela
        geração descontada. É o número que se compara com a tarifa.
        """
        taxa = self.premissas.taxa_desconto
        custos = self.premissas.capex_brl
        energia = 0.0
        for linha in self.fluxo:
            fator = (1.0 + taxa) ** linha.ano
            custos += linha.opex_brl / fator
            energia += linha.geracao_kwh / fator
        return custos / energia if energia > 0 else None

    def as_dict(self) -> dict:
        return {
            "economia_ano1_brl": round(self.economia_ano1_brl, 2),
            "payback_simples_anos": (
                round(self.payback_simples_anos, 2) if self.payback_simples_anos else None
            ),
            "payback_descontado_anos": (
                round(self.payback_descontado_anos, 2) if self.payback_descontado_anos else None
            ),
            "vpl_brl": round(self.vpl_brl, 2),
            "tir_anual": round(self.tir_anual, 4) if self.tir_anual is not None else None,
            "roi_percent": round(self.roi_percent, 1) if self.roi_percent is not None else None,
            "lcoe_brl_kwh": round(self.lcoe_brl_kwh, 4) if self.lcoe_brl_kwh else None,
            "geracao_total_kwh": round(self.geracao_total_kwh, 0),
            "economia_total_brl": round(self.economia_total_brl, 2),
            "co2_evitado_t": round(self.co2_evitado_t, 1),
            "arvores_equivalentes": self.arvores_equivalentes,
            "capex_brl": round(self.premissas.capex_brl, 2),
            "tarifa_brl_kwh": self.premissas.tarifa_brl_kwh,
            "fracao_autoconsumo": self.premissas.fracao_autoconsumo,
            "aplicar_lei_14300": self.premissas.aplicar_lei_14300,
            "avisos": list(self.avisos),
        }


#: Referência de R$/kWp em 100 kWp, por padrão construtivo. A curva de escala
#: parte daqui; ver :func:`estimar_capex`.
#:
#: Os três níveis existem porque o mesmo kWp custa coisas muito diferentes
#: conforme o cliente. O que muda não é o módulo -- é estrutura, acabamento,
#: prazo de obra em prédio ocupado, ART, seguro, e a exigência de projeto
#: executivo que um condomínio de alto padrão faz e um galpão não faz.
PADROES_CAPEX: dict[str, float] = {
    #: Obra simples: telhado metálico, acesso livre, estrutura padrão.
    "basico": 2_700.0,
    #: O caso médio do mercado de geração distribuída.
    "padrao": 3_200.0,
    #: Alto padrão: prédio ocupado, estrutura e acabamento acima da média,
    #: exigência de projeto executivo e prazo curto. Calibrado no Ed. Mourisco
    #: (Rio, 2026): 312 kWp fechados a R$ 4.104/kWp, que a curva de escala
    #: devolve a partir desta referência.
    "alto": 4_705.0,
}

#: Padrão adotado quando ninguém diz qual é.
PADRAO_CAPEX_DEFAULT = "padrao"


def referencia_capex(padrao: str | None = None, defaults: SolarDefaults | None = None) -> float:
    """
    O R$/kWp de referência do padrão construtivo pedido.

    Sem padrão, vale o da configuração -- que é o que os chamadores antigos
    esperam e o que a linha de comando continua usando.
    """
    if padrao is None:
        defaults = defaults or get_settings().solar
        return float(defaults.capex_reference_brl_per_kwp)
    chave = str(padrao).strip().lower()
    if chave not in PADROES_CAPEX:
        conhecidos = ", ".join(sorted(PADROES_CAPEX))
        raise ValueError(f"padrão de CAPEX desconhecido: {padrao!r}. Conhecidos: {conhecidos}")
    return PADROES_CAPEX[chave]


def estimar_capex(
    potencia_kwp: float,
    defaults: SolarDefaults | None = None,
    referencia_brl_kwp: float | None = None,
    padrao: str | None = None,
    topologia: str | None = None,
    mao_de_obra_brl_kwp: float | None = None,
    material_ca_brl_kwp: float | None = None,
) -> float:
    """
    CAPEX indicativo: tabela de kit no varejo, curva de escala acima dela.

    Até 40 kWp o preço não vem de curva nenhuma -- vem da tabela de kit do
    distribuidor, com degraus por potência e colunas por topologia de inversor
    (:mod:`aurum.pv.kits`). A diferença entre um kit monofásico e um
    split-phase de mesma potência passa de 50%, e isso é topologia, não
    tamanho: nenhuma curva de escala a captura. Basta informar ``topologia``
    para a tabela ser consultada.

    Acima de 40 kWp, e para topologia que a tabela não cobre naquela potência,
    vale a curva de sempre: o R$/kWp cai com o tamanho porque engenharia,
    mobilização e projeto se diluem. Expoente -0,12 sobre a razão de potência,
    calibrada em torno de 100 kWp -- um sistema de 1 MWp sai cerca de 25% mais
    barato por kWp que um de 10 kWp.

    A ordem de precedência é a da confiança na origem do número:
    ``referencia_brl_kwp`` explícito (cotação) vence a tabela de kit, que vence
    ``padrao`` (:data:`PADROES_CAPEX`). Cotação de verdade sempre passa na
    frente de qualquer curva.

    Nos dois caminhos o resultado é **obra entregue**, e não material posto: ao
    preço de kit são somadas a mão de obra e o material CA, ambos em R$/kWp.
    Somadas, e não multiplicadas -- a equipe leva o mesmo tempo para instalar
    um kit mono e um split-phase de mesma potência.
    """
    if potencia_kwp <= 0:
        return 0.0

    if topologia and not referencia_brl_kwp:
        from .kits import MAO_DE_OBRA_BRL_KWP, MATERIAL_CA_BRL_KWP, capex_de_kit

        do_kit = capex_de_kit(
            potencia_kwp, topologia,
            MAO_DE_OBRA_BRL_KWP if mao_de_obra_brl_kwp is None else mao_de_obra_brl_kwp,
            MATERIAL_CA_BRL_KWP if material_ca_brl_kwp is None else material_ca_brl_kwp,
        )
        if do_kit is not None:
            return do_kit

    referencia = (
        float(referencia_brl_kwp)
        if referencia_brl_kwp
        else referencia_capex(padrao, defaults)
    )
    escala = (max(potencia_kwp, 1.0) / 100.0) ** -0.12
    # Trava a curva para não extrapolar em faixas onde não foi calibrada.
    escala = min(max(escala, 0.70), 1.45)
    return potencia_kwp * referencia * escala


def composicao_capex(capex_total: float) -> dict[str, float]:
    """
    Decomposição indicativa do investimento, para a proposta.

    Percentuais típicos do mercado brasileiro de geração distribuída
    comercial. A soma fecha exatamente no total, sem resíduo de arredondamento.
    """
    pesos = {
        "Módulos fotovoltaicos": 0.38,
        "Inversores": 0.17,
        "Estrutura de fixação": 0.11,
        "Cabeamento e proteções CC/CA": 0.09,
        "Instalação e mão de obra": 0.13,
        "Projeto, ART e homologação": 0.07,
        "Frete, seguro e comissionamento": 0.05,
    }
    itens = {nome: round(capex_total * peso, 2) for nome, peso in pesos.items()}
    # Joga a diferença de arredondamento no maior item.
    diferenca = round(capex_total - sum(itens.values()), 2)
    if diferenca:
        itens["Módulos fotovoltaicos"] = round(itens["Módulos fotovoltaicos"] + diferenca, 2)
    return itens


def _tir(fluxos: Sequence[float], palpite_min: float = -0.95, palpite_max: float = 3.0) -> float | None:
    """
    TIR por bissecção sobre o vetor de caixa (índice 0 = investimento).

    Devolve None quando não há troca de sinal -- projeto que nunca se paga não
    tem TIR real, e devolver um número nesse caso seria enganoso.
    """
    if not fluxos or fluxos[0] >= 0:
        return None
    if sum(fluxos) <= 0:
        return None

    def vpl(taxa: float) -> float:
        return sum(f / ((1.0 + taxa) ** i) for i, f in enumerate(fluxos))

    baixo, alto = palpite_min, palpite_max
    if vpl(baixo) * vpl(alto) > 0:
        return None
    for _ in range(200):
        meio = (baixo + alto) / 2.0
        valor = vpl(meio)
        if abs(valor) < 1e-6:
            return meio
        if valor > 0:
            baixo = meio
        else:
            alto = meio
    return (baixo + alto) / 2.0


def calcular(
    geracao_ano1_kwh: float,
    premissas: PremissasEconomicas,
    defaults: SolarDefaults | None = None,
) -> ResultadoEconomico:
    """Monta o fluxo de caixa e os indicadores econômicos do projeto."""
    defaults = defaults or get_settings().solar
    avisos: list[str] = []
    opex_base = premissas.opex(defaults)

    fracao_auto = min(max(premissas.fracao_autoconsumo, 0.0), 1.0)
    if premissas.consumo_anual_kwh and premissas.consumo_anual_kwh > 0:
        if geracao_ano1_kwh > premissas.consumo_anual_kwh * 1.05:
            avisos.append(
                f"A geração estimada ({geracao_ano1_kwh:,.0f} kWh/ano) supera o consumo informado "
                f"({premissas.consumo_anual_kwh:,.0f} kWh/ano). O excedente não compensado não gera "
                "economia e foi desconsiderado no fluxo de caixa."
            )

    fluxo: list[LinhaFluxo] = []
    vetor_caixa: list[float] = [-premissas.capex_brl]
    vpl_acumulado = -premissas.capex_brl

    for ano in range(1, premissas.anos + 1):
        ano_calendario = premissas.ano_conexao + ano - 1
        geracao = geracao_ano1_kwh * ((1.0 - premissas.degradacao_anual) ** (ano - 1))

        # A energia só vira economia até o limite do que o cliente consome.
        aproveitavel = geracao
        if premissas.consumo_anual_kwh and premissas.consumo_anual_kwh > 0:
            aproveitavel = min(geracao, premissas.consumo_anual_kwh)

        tarifa = premissas.tarifa_brl_kwh * ((1.0 + premissas.escalada_tarifa) ** (ano - 1))
        energia_autoconsumo = aproveitavel * fracao_auto
        energia_injetada = aproveitavel - energia_autoconsumo

        # Autoconsumo instantâneo evita a tarifa cheia, sem passar pela rede.
        economia_auto = energia_autoconsumo * tarifa
        # A energia injetada é compensada, mas paga o Fio B a partir de 2023.
        economia_injecao = energia_injetada * tarifa
        if premissas.aplicar_lei_14300:
            fio_b_unitario = tarifa * premissas.fracao_fio_b_na_tarifa * fator_fio_b(ano_calendario)
            custo_fio_b = energia_injetada * fio_b_unitario
        else:
            custo_fio_b = 0.0

        opex = opex_base * ((1.0 + premissas.escalada_tarifa) ** (ano - 1))
        liquido = economia_auto + economia_injecao - custo_fio_b - opex
        descontado = liquido / ((1.0 + premissas.taxa_desconto) ** ano)
        vpl_acumulado += descontado
        vetor_caixa.append(liquido)

        fluxo.append(
            LinhaFluxo(
                ano=ano,
                ano_calendario=ano_calendario,
                geracao_kwh=geracao,
                tarifa_brl_kwh=tarifa,
                economia_autoconsumo_brl=economia_auto,
                economia_injecao_brl=economia_injecao,
                custo_fio_b_brl=custo_fio_b,
                opex_brl=opex,
                fluxo_liquido_brl=liquido,
                fluxo_descontado_brl=descontado,
                vpl_acumulado_brl=vpl_acumulado,
            )
        )

    economia_ano1 = fluxo[0].fluxo_liquido_brl if fluxo else 0.0
    payback_simples = premissas.capex_brl / economia_ano1 if economia_ano1 > 0 else None

    payback_descontado: float | None = None
    anterior = -premissas.capex_brl
    for linha in fluxo:
        if linha.vpl_acumulado_brl >= 0 and anterior < 0:
            intervalo = linha.vpl_acumulado_brl - anterior
            fracao = (-anterior / intervalo) if intervalo > 0 else 0.0
            payback_descontado = (linha.ano - 1) + fracao
            break
        anterior = linha.vpl_acumulado_brl

    geracao_total = sum(l.geracao_kwh for l in fluxo)
    economia_total = sum(l.fluxo_liquido_brl for l in fluxo)
    vpl = fluxo[-1].vpl_acumulado_brl if fluxo else -premissas.capex_brl
    tir = _tir(vetor_caixa)
    roi = ((economia_total - premissas.capex_brl) / premissas.capex_brl * 100.0) if premissas.capex_brl > 0 else None

    co2_t = geracao_total / 1000.0 * FATOR_EMISSAO_SIN_TCO2_MWH
    # Equivalência: quantas árvores absorveriam esse CO2 ao longo do mesmo
    # horizonte de análise (não em um único ano).
    arvores = int(round(co2_t * 1000.0 / (ABSORCAO_ARVORE_KG_CO2_ANO * premissas.anos)))

    if payback_descontado is None:
        avisos.append(
            f"O VPL não se torna positivo em {premissas.anos} anos com as premissas atuais "
            f"(taxa de desconto de {premissas.taxa_desconto:.1%})."
        )
    if premissas.aplicar_lei_14300:
        avisos.append(
            "Fluxo já considera a cobrança gradual do Fio B sobre a energia injetada "
            "(Lei 14.300/2022) e o custo de disponibilidade permanece a cargo do cliente."
        )

    return ResultadoEconomico(
        premissas=premissas,
        fluxo=fluxo,
        economia_ano1_brl=economia_ano1,
        payback_simples_anos=payback_simples,
        payback_descontado_anos=payback_descontado,
        vpl_brl=vpl,
        tir_anual=tir,
        roi_percent=roi,
        geracao_total_kwh=geracao_total,
        economia_total_brl=economia_total,
        co2_evitado_t=co2_t,
        arvores_equivalentes=arvores,
        avisos=avisos,
    )


def sensibilidade_tarifa(
    geracao_ano1_kwh: float,
    premissas: PremissasEconomicas,
    variacoes: Sequence[float] = (-0.20, -0.10, 0.0, 0.10, 0.20),
) -> dict[str, dict[str, float | None]]:
    """
    Recalcula os indicadores para variações da tarifa base.

    Diferente da versão anterior, que apenas multiplicava a economia do ano 1
    pelo fator -- o que não mostrava efeito nenhum sobre payback, VPL ou TIR,
    que são exatamente o que muda.
    """
    resultado: dict[str, dict[str, float | None]] = {}
    for variacao in variacoes:
        ajustadas = PremissasEconomicas(
            **{**premissas.__dict__, "tarifa_brl_kwh": premissas.tarifa_brl_kwh * (1.0 + variacao)}
        )
        calculo = calcular(geracao_ano1_kwh, ajustadas)
        rotulo = "base" if abs(variacao) < 1e-9 else f"{variacao:+.0%}"
        resultado[rotulo] = {
            "tarifa": ajustadas.tarifa_brl_kwh,
            "economia_ano1": calculo.economia_ano1_brl,
            "payback": calculo.payback_simples_anos,
            "vpl": calculo.vpl_brl,
            "tir": calculo.tir_anual,
        }
    return resultado
