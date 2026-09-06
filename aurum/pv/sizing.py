"""
Dimensionamento elétrico: escolha de inversores e arranjo de strings.

Correções de engenharia em relação à versão anterior:

* **Correção de temperatura na tensão.** A janela de MPPT era verificada com
  a Voc de catálogo (25 graus C). A Voc cresce quando esfria; numa manhã de
  10 graus C em Curitiba, uma string dimensionada no limite ultrapassa a
  tensão máxima do inversor e o queima. Aqui a checagem usa a Voc corrigida
  para a mínima temperatura de projeto, e o limite inferior usa a Vmp
  corrigida para a máxima temperatura de célula -- que é quando a tensão
  despenca e o inversor sai do MPPT.

* **A geração passa a corresponder ao sistema realmente montado.** Antes, a
  energia anual era calculada para a potência *necessária*, não para a
  potência *instalada* de cada alternativa, de modo que as linhas do
  comparativo mostravam todas o mesmo número.

* **Sistemas grandes.** A busca cobre múltiplas unidades do mesmo inversor,
  partindo dos maiores, em vez de assumir uma faixa fixa em torno de um
  palpite inicial.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from ..config import SolarDefaults, get_settings
from .equipment import BaseEquipamentos, Inversor, Modulo

#: Coeficiente térmico típico da tensão de circuito aberto, por grau Celsius.
#: Módulos de silício cristalino ficam entre -0,0025 e -0,0030 /K.
COEF_TEMP_VOC = -0.0028
#: Coeficiente térmico típico da tensão de máxima potência.
COEF_TEMP_VMP = -0.0035

#: Temperatura mínima de projeto da célula (graus C). Vale para o Sul do
#: Brasil; regiões mais quentes podem relaxar, mas o conservadorismo aqui
#: custa pouco e evita perda de equipamento.
TEMP_MIN_PROJETO_C = 0.0
#: Temperatura máxima de célula em operação (graus C). Módulo em telhado
#: escuro com pouca ventilação chega a isso com folga.
TEMP_MAX_CELULA_C = 70.0

#: Razão CC/CA máxima aceitável. Acima disso o corte por potência (clipping)
#: no inversor passa a descartar energia em volume relevante.
DC_AC_MAX = 1.35
DC_AC_MIN = 0.90


def voc_corrigida(modulo: Modulo, temperatura_c: float = TEMP_MIN_PROJETO_C) -> float:
    """Voc do módulo na temperatura informada (maior que a de catálogo no frio)."""
    return modulo.voc * (1.0 + COEF_TEMP_VOC * (temperatura_c - 25.0))


def vmp_corrigida(modulo: Modulo, temperatura_c: float = TEMP_MAX_CELULA_C) -> float:
    """Vmp do módulo na temperatura informada (menor que a de catálogo no calor)."""
    base = modulo.vmp if modulo.vmp > 0 else modulo.voc * 0.82
    return base * (1.0 + COEF_TEMP_VMP * (temperatura_c - 25.0))


def modulos_por_string(
    modulo: Modulo,
    inversor: Inversor,
    temp_min_c: float = TEMP_MIN_PROJETO_C,
    temp_max_c: float = TEMP_MAX_CELULA_C,
) -> tuple[int, int]:
    """
    Faixa (mínimo, máximo) de módulos em série que respeita o inversor.

    O máximo vem da Voc no frio contra a tensão máxima CC; o mínimo, da Vmp no
    calor contra a tensão de partida. Devolve (0, 0) quando não existe faixa
    válida para o par módulo/inversor.
    """
    voc_fria = voc_corrigida(modulo, temp_min_c)
    vmp_quente = vmp_corrigida(modulo, temp_max_c)
    if voc_fria <= 0 or vmp_quente <= 0:
        return 0, 0

    maximo = int(inversor.tensao_max_cc // voc_fria)
    # Uma margem de 10% acima da tensão de partida evita que o inversor fique
    # entrando e saindo de operação em dia nublado.
    minimo = max(1, math.ceil((inversor.tensao_start * 1.10) / vmp_quente))

    if maximo < minimo:
        return 0, 0
    return minimo, maximo


def strings_por_mppt(modulo: Modulo, inversor: Inversor) -> int:
    """
    Quantas strings em paralelo cabem num MPPT sem estourar a corrente.

    Aplica os dois limites que o datasheet do inversor declara separadamente,
    e que significam coisas diferentes:

    * ``corrente_max_mppt`` -- corrente máxima de **operação** por MPPT, a ser
      comparada com a Imp do módulo;
    * ``corrente_curto_max_mppt`` -- corrente máxima de **curto-circuito**
      suportada, a ser comparada com a Isc.

    Não se acrescenta aqui o fator 1,25 da NBR 16690: esse fator dimensiona
    condutores e dispositivos de proteção, e os limites do inversor já
    embutem a própria margem. Aplicá-lo em cima seria contar a margem duas
    vezes -- foi o que, no teste, deixou todos os inversores pequenos sem
    nenhum módulo compatível.
    """
    imp = modulo.imp if modulo.imp > 0 else modulo.isc * 0.93
    if imp <= 0 or modulo.isc <= 0:
        return 0
    por_operacao = int(inversor.corrente_max_mppt / imp)
    por_curto = int(inversor.corrente_curto_max_mppt / modulo.isc)
    return max(0, min(por_operacao, por_curto))


@dataclass
class ArranjoInversor:
    """Configuração de um modelo de inversor dentro do sistema."""

    inversor: Inversor
    quantidade: int
    modulos_por_string: int
    strings_por_mppt: int
    mppts_usados: int

    @property
    def strings_totais(self) -> int:
        return self.strings_por_mppt * self.mppts_usados * self.quantidade

    @property
    def modulos_totais(self) -> int:
        return self.modulos_por_string * self.strings_totais

    @property
    def potencia_ca_kw(self) -> float:
        return self.inversor.potencia_ca_kw * self.quantidade


@dataclass
class SistemaFV:
    """Um sistema fotovoltaico completo e verificado."""

    modulo: Modulo
    arranjos: list[ArranjoInversor]
    #: Quantos módulos o telhado comportava (limite físico da busca).
    modulos_disponiveis: int
    avisos: list[str] = field(default_factory=list)
    #: Preenchidos pelo chamador após consultar o recurso solar.
    geracao_anual_kwh: float | None = None
    geracao_mensal_kwh: list[float] = field(default_factory=list)

    @property
    def modulos_totais(self) -> int:
        return sum(a.modulos_totais for a in self.arranjos)

    @property
    def potencia_cc_kwp(self) -> float:
        return self.modulos_totais * self.modulo.potencia_wp / 1000.0

    @property
    def potencia_ca_kw(self) -> float:
        return sum(a.potencia_ca_kw for a in self.arranjos)

    @property
    def razao_dc_ac(self) -> float:
        return self.potencia_cc_kwp / self.potencia_ca_kw if self.potencia_ca_kw > 0 else 0.0

    @property
    def area_modulos_m2(self) -> float:
        return self.modulos_totais * self.modulo.area_m2

    @property
    def aproveitamento_do_telhado(self) -> float:
        """Fração dos módulos possíveis que o arranjo elétrico conseguiu usar."""
        if self.modulos_disponiveis <= 0:
            return 0.0
        return self.modulos_totais / self.modulos_disponiveis

    @property
    def rendimento_especifico(self) -> float | None:
        """kWh por kWp ao ano do sistema montado."""
        if self.geracao_anual_kwh is None or self.potencia_cc_kwp <= 0:
            return None
        return self.geracao_anual_kwh / self.potencia_cc_kwp

    def descricao_inversores(self) -> str:
        return " + ".join(f"{a.quantidade}x {a.inversor}" for a in self.arranjos)

    def as_dict(self) -> dict:
        return {
            "modulo": str(self.modulo),
            "modulo_modelo": self.modulo.modelo,
            "modulo_fabricante": self.modulo.fabricante,
            "modulo_wp": self.modulo.potencia_wp,
            "modulos_totais": self.modulos_totais,
            "modulos_disponiveis": self.modulos_disponiveis,
            "aproveitamento_telhado": round(self.aproveitamento_do_telhado, 3),
            "potencia_cc_kwp": round(self.potencia_cc_kwp, 2),
            "potencia_ca_kw": round(self.potencia_ca_kw, 2),
            "razao_dc_ac": round(self.razao_dc_ac, 3),
            "area_modulos_m2": round(self.area_modulos_m2, 1),
            "inversores": [
                {
                    "modelo": a.inversor.modelo,
                    "fabricante": a.inversor.fabricante,
                    "potencia_ca_kw": a.inversor.potencia_ca_kw,
                    "quantidade": a.quantidade,
                    "mppts_usados": a.mppts_usados,
                    "modulos_por_string": a.modulos_por_string,
                    "strings_por_mppt": a.strings_por_mppt,
                    "strings_totais": a.strings_totais,
                    "modulos_totais": a.modulos_totais,
                }
                for a in self.arranjos
            ],
            "descricao_inversores": self.descricao_inversores(),
            "geracao_anual_kwh": self.geracao_anual_kwh,
            "rendimento_especifico_kwh_kwp": (
                round(self.rendimento_especifico, 1) if self.rendimento_especifico else None
            ),
            "avisos": list(self.avisos),
        }


def _melhor_config_para(
    modulo: Modulo,
    inversor: Inversor,
    modulos_alvo: int,
) -> ArranjoInversor | None:
    """
    Melhor configuração de uma unidade deste inversor, sem exceder o alvo.

    Maximiza módulos aproveitados respeitando janela de tensão, corrente por
    MPPT, potência FV máxima e a razão CC/CA de projeto.
    """
    min_serie, max_serie = modulos_por_string(modulo, inversor)
    if max_serie == 0:
        return None
    max_paralelo = strings_por_mppt(modulo, inversor)
    if max_paralelo == 0:
        return None

    limite_potencia_w = min(
        inversor.potencia_fv_max_w,
        inversor.potencia_ca_w * DC_AC_MAX,
    )

    # A razão CC/CA mínima é preferência, não veto: para alguns pares
    # módulo/inversor nenhuma combinação a alcança, e rejeitar tudo deixaria
    # o modelo de módulo sem sistema nenhum em vez de com um levemente
    # subaproveitado.
    melhor_preferido: ArranjoInversor | None = None
    melhor_qualquer: ArranjoInversor | None = None

    def _supera(candidato: ArranjoInversor, atual: ArranjoInversor | None) -> bool:
        if atual is None:
            return True
        # Preferir mais módulos; em empate, menos strings (menos cabeamento,
        # menos proteções, obra mais barata).
        if candidato.modulos_totais != atual.modulos_totais:
            return candidato.modulos_totais > atual.modulos_totais
        return candidato.strings_totais < atual.strings_totais

    for n_serie in range(max_serie, min_serie - 1, -1):
        if n_serie * modulo.potencia_wp <= 0:
            continue
        for mppts in range(inversor.num_mppt, 0, -1):
            for n_paralelo in range(max_paralelo, 0, -1):
                modulos = n_serie * n_paralelo * mppts
                potencia = modulos * modulo.potencia_wp
                if potencia > limite_potencia_w or modulos > modulos_alvo:
                    continue
                candidato = ArranjoInversor(
                    inversor=inversor,
                    quantidade=1,
                    modulos_por_string=n_serie,
                    strings_por_mppt=n_paralelo,
                    mppts_usados=mppts,
                )
                if _supera(candidato, melhor_qualquer):
                    melhor_qualquer = candidato
                if potencia >= inversor.potencia_ca_w * DC_AC_MIN and _supera(candidato, melhor_preferido):
                    melhor_preferido = candidato

    return melhor_preferido or melhor_qualquer


def dimensionar(
    modulos_disponiveis: int,
    modulo: Modulo,
    base: BaseEquipamentos,
    inversores_permitidos: Sequence[Inversor] | None = None,
) -> SistemaFV | None:
    """
    Monta o sistema que melhor aproveita os módulos que cabem no telhado.

    Estratégia: começa pelos inversores maiores e vai preenchendo o saldo de
    módulos com unidades sucessivas, usando modelos menores para o resto.
    Isso reflete a prática de projeto -- um telhado de 800 kWp não é feito com
    266 inversores de 3 kW.
    """
    if modulos_disponiveis <= 0:
        return None

    candidatos = list(inversores_permitidos or base.inversores_ordenados())
    if not candidatos:
        return None

    # Descarta inversores grandes demais para o telhado: sem isso, um telhado
    # de 10 kWp recebia um inversor de 75 kW, que atende eletricamente mas é
    # um disparate comercial. Mantém sempre o menor do catálogo como saída,
    # para não devolver "nenhum sistema" quando o telhado é muito pequeno.
    potencia_cc_disponivel_w = modulos_disponiveis * modulo.potencia_wp
    teto_ca_w = potencia_cc_disponivel_w / DC_AC_MIN
    viaveis = [i for i in candidatos if i.potencia_ca_w <= teto_ca_w]
    if not viaveis:
        viaveis = [min(candidatos, key=lambda i: i.potencia_ca_w)]
    candidatos = sorted(viaveis, key=lambda i: i.potencia_ca_w, reverse=True)

    arranjos: list[ArranjoInversor] = []
    avisos: list[str] = []
    restantes = modulos_disponiveis

    for inversor in candidatos:
        config = _melhor_config_para(modulo, inversor, restantes)
        if config is None or config.modulos_totais <= 0:
            continue
        quantidade = restantes // config.modulos_totais
        if quantidade <= 0:
            continue
        arranjos.append(
            ArranjoInversor(
                inversor=inversor,
                quantidade=quantidade,
                modulos_por_string=config.modulos_por_string,
                strings_por_mppt=config.strings_por_mppt,
                mppts_usados=config.mppts_usados,
            )
        )
        restantes -= quantidade * config.modulos_totais
        if restantes <= 0:
            break

    if not arranjos:
        return None

    # Sobra pequena é normal: o arranjo elétrico é quantizado por string.
    if restantes > 0:
        sobra = restantes / modulos_disponiveis
        if sobra > 0.10:
            avisos.append(
                f"{restantes} de {modulos_disponiveis} posições ({sobra:.0%}) ficaram sem inversor "
                "compatível. Ampliar o catálogo de inversores melhora o aproveitamento do telhado."
            )

    sistema = SistemaFV(modulo=modulo, arranjos=arranjos, modulos_disponiveis=modulos_disponiveis, avisos=avisos)

    if sistema.razao_dc_ac > DC_AC_MAX:
        sistema.avisos.append(
            f"Razão CC/CA de {sistema.razao_dc_ac:.2f} acima do recomendado ({DC_AC_MAX:.2f}); "
            "haverá corte de potência (clipping) nos horários de pico."
        )
    elif sistema.razao_dc_ac < 1.0:
        sistema.avisos.append(
            f"Razão CC/CA de {sistema.razao_dc_ac:.2f} indica inversor superdimensionado; "
            "há espaço para reduzir custo de equipamento."
        )

    # Registro explícito da verificação de tensão, para a memória de cálculo.
    for arranjo in sistema.arranjos:
        voc_total = arranjo.modulos_por_string * voc_corrigida(modulo, TEMP_MIN_PROJETO_C)
        if voc_total > arranjo.inversor.tensao_max_cc:
            sistema.avisos.append(
                f"String de {arranjo.modulos_por_string} módulos atinge {voc_total:.0f} V a "
                f"{TEMP_MIN_PROJETO_C:.0f} graus C, acima dos {arranjo.inversor.tensao_max_cc:.0f} V "
                f"do {arranjo.inversor.modelo}."
            )

    return sistema


def dimensionar_por_consumo(
    consumo_anual_kwh: float,
    rendimento_kwh_kwp_ano: float,
    modulo: Modulo,
    base: BaseEquipamentos,
    modulos_maximos: int | None = None,
) -> SistemaFV | None:
    """
    Dimensiona para compensar um consumo anual conhecido.

    Quando o telhado não comporta o sistema necessário, ``modulos_maximos``
    trunca a busca e o sistema resultante compensa apenas parte do consumo --
    o que a proposta deve declarar em vez de esconder.
    """
    if consumo_anual_kwh <= 0 or rendimento_kwh_kwp_ano <= 0:
        return None
    kwp_necessario = consumo_anual_kwh / rendimento_kwh_kwp_ano
    modulos_necessarios = math.ceil(kwp_necessario * 1000.0 / modulo.potencia_wp)
    alvo = modulos_necessarios if modulos_maximos is None else min(modulos_necessarios, modulos_maximos)

    sistema = dimensionar(alvo, modulo, base)
    if sistema is None:
        return None

    if modulos_maximos is not None and modulos_necessarios > modulos_maximos:
        cobertura = sistema.potencia_cc_kwp * rendimento_kwh_kwp_ano / consumo_anual_kwh
        sistema.avisos.append(
            f"A área disponível limita o sistema a {sistema.potencia_cc_kwp:.1f} kWp, contra "
            f"{kwp_necessario:.1f} kWp necessários para compensação integral. "
            f"Cobertura estimada do consumo: {cobertura:.0%}."
        )
    return sistema


def custo_equipamento_brl(sistema: SistemaFV, defaults: SolarDefaults | None = None) -> float:
    """
    Custo indicativo de módulos + inversores, em reais.

    Serve para comparar alternativas de arranjo entre si, não como orçamento:
    não inclui estrutura, cabeamento, mão de obra, projeto nem homologação.
    """
    defaults = defaults or get_settings().solar
    custo_modulos = sistema.modulos_totais * sistema.modulo.potencia_wp * defaults.preco_modulo_brl_por_wp
    custo_inversores = sum(
        a.quantidade * a.inversor.potencia_ca_w * defaults.preco_inversor_brl_por_w for a in sistema.arranjos
    )
    return custo_modulos + custo_inversores


def melhor_sistema(
    modulos_por_modelo: dict[str, int],
    base: BaseEquipamentos,
    defaults: SolarDefaults | None = None,
) -> SistemaFV | None:
    """
    Escolhe o melhor sistema entre modelos de módulo.

    ``modulos_por_modelo`` mapeia o modelo do módulo para quantos cabem no
    telhado -- valor que vem de :mod:`aurum.pv.layout`, já que módulos de
    tamanhos diferentes cabem em quantidades diferentes.

    O critério é **custo de equipamento por kWp instalado**, não potência CC
    pura. A diferença é material: num galpão de 5.000 m2 testado, o maior
    sistema em kWp exigia 975 kW de inversor para 866 kWp (razão 0,89),
    enquanto a alternativa de 776 kWp precisava de apenas 675 kW (razão 1,15)
    -- 11% menos geração por 25% menos investimento em equipamento.
    """
    defaults = defaults or get_settings().solar
    sistemas: list[SistemaFV] = []
    for nome_modelo, quantidade in modulos_por_modelo.items():
        modulo = base.modulo_por_modelo(nome_modelo)
        if modulo is None or quantidade <= 0:
            continue
        sistema = dimensionar(quantidade, modulo, base)
        if sistema is not None and sistema.modulos_totais > 0:
            sistemas.append(sistema)

    if not sistemas:
        return None

    def custo_por_kwp(sistema: SistemaFV) -> float:
        if sistema.potencia_cc_kwp <= 0:
            return float("inf")
        return custo_equipamento_brl(sistema, defaults) / sistema.potencia_cc_kwp

    return min(sistemas, key=custo_por_kwp)
