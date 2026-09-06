"""
Memória de cálculo do dimensionamento, para o equipamento escolhido.

Um relatório que diz "cabem 288 módulos e o inversor é o GW50KN-MT" pede que se
acredite nele. Um que mostra a conta permite conferir — e é o que um
responsável técnico assina.

Cada passo aqui devolve três coisas: a **fórmula**, os **números que entraram**
e o **resultado com a unidade**. A ordem é a ordem em que a decisão acontece de
verdade:

1. **Tensão no frio.** A Voc do módulo sobe quando esfria, e é ela contra a
   tensão máxima CC do inversor que fixa quantos módulos cabem em série. Errar
   aqui não é perder rendimento: é queimar o inversor no primeiro amanhecer
   frio do ano.
2. **Tensão no calor.** A Vmp cai quando esquenta, e é ela contra a tensão de
   partida que fixa o mínimo da string. Abaixo disso o inversor fica entrando
   e saindo de operação em dia nublado.
3. **Corrente por MPPT.** Dois limites diferentes no datasheet — corrente de
   operação e corrente de curto — comparados com Imp e Isc.
4. **Fechamento do arranjo.** Quantas strings, quantos módulos, quanta
   potência, e a razão CC/CA resultante.

As temperaturas de projeto são explícitas e ficam registradas: 0 °C de célula
mínima e 70 °C de célula máxima é a hipótese do Sul do Brasil. Quem dimensiona
em Petrolina relaxa a primeira; quem dimensiona telha metálica escura sem
ventilação deveria endurecer a segunda.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from .equipment import Inversor, Modulo
from .sizing import (
    COEF_TEMP_VMP,
    COEF_TEMP_VOC,
    DC_AC_MAX,
    DC_AC_MIN,
    TEMP_MAX_CELULA_C,
    TEMP_MIN_PROJETO_C,
    modulos_por_string,
    strings_por_mppt,
    vmp_corrigida,
    voc_corrigida,
)

__all__ = ["MemoriaCalculo", "PassoCalculo", "memoria_do_arranjo"]


@dataclass(frozen=True)
class PassoCalculo:
    """Um passo conferível: o que se calcula, com que fórmula, e o que deu."""

    titulo: str
    formula: str
    substituicao: str
    resultado: str
    comentario: str = ""
    critico: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "titulo": self.titulo,
            "formula": self.formula,
            "substituicao": self.substituicao,
            "resultado": self.resultado,
            "comentario": self.comentario,
            "critico": self.critico,
        }


@dataclass
class MemoriaCalculo:
    """A conta inteira do arranjo, passo a passo, mais o veredito."""

    modulo: Modulo
    inversor: Inversor
    modulos_por_string: int
    strings: int
    #: Quantos inversores deste modelo o telhado pede. A memória descreve um.
    inversores: int = 1
    #: Strings do sistema inteiro. Não é `inversores × strings`: o último
    #: inversor costuma receber menos strings que os demais, porque o telhado
    #: acaba antes de completá-lo.
    strings_do_sistema: int = 0
    passos: list[PassoCalculo] = field(default_factory=list)
    avisos: list[str] = field(default_factory=list)
    temp_min_c: float = TEMP_MIN_PROJETO_C
    temp_max_c: float = TEMP_MAX_CELULA_C

    @property
    def modulos_totais(self) -> int:
        """Módulos ligados a **um** inversor."""
        return self.modulos_por_string * self.strings

    @property
    def modulos_do_sistema(self) -> int:
        strings = self.strings_do_sistema or (self.strings * self.inversores)
        return strings * self.modulos_por_string

    @property
    def potencia_do_sistema_kwp(self) -> float:
        return self.modulos_do_sistema * self.modulo.potencia_wp / 1000.0

    @property
    def potencia_cc_kwp(self) -> float:
        return self.modulos_totais * self.modulo.potencia_wp / 1000.0

    @property
    def razao_cc_ca(self) -> float:
        return self.potencia_cc_kwp / self.inversor.potencia_ca_kw if self.inversor.potencia_ca_kw else 0.0

    @property
    def viavel(self) -> bool:
        return self.modulos_totais > 0 and not any(p.critico for p in self.passos if p.critico)

    def as_dict(self) -> dict[str, Any]:
        return {
            "modulo": str(self.modulo),
            "inversor": str(self.inversor),
            "modulos_por_string": self.modulos_por_string,
            "strings": self.strings,
            "modulos_por_inversor": self.modulos_totais,
            "inversores": self.inversores,
            "strings_do_sistema": self.strings_do_sistema or self.strings,
            "modulos_do_sistema": self.modulos_do_sistema,
            "potencia_cc_kwp": self.potencia_cc_kwp,
            "potencia_do_sistema_kwp": self.potencia_do_sistema_kwp,
            "potencia_ca_kw": self.inversor.potencia_ca_kw,
            "razao_cc_ca": self.razao_cc_ca,
            "temperatura_minima_projeto_c": self.temp_min_c,
            "temperatura_maxima_celula_c": self.temp_max_c,
            "passos": [p.as_dict() for p in self.passos],
            "avisos": list(self.avisos),
            "viavel": self.viavel,
        }


def memoria_do_arranjo(
    modulo: Modulo,
    inversor: Inversor,
    modulos_disponiveis: int | None = None,
    temp_min_c: float = TEMP_MIN_PROJETO_C,
    temp_max_c: float = TEMP_MAX_CELULA_C,
) -> MemoriaCalculo:
    """
    Monta a memória de cálculo do par módulo × inversor.

    ``modulos_disponiveis`` é o teto físico — quantos módulos o telhado
    comporta. Sem ele, a conta devolve o arranjo cheio que o inversor aceita,
    que é o número de catálogo e não o número do projeto.
    """
    passos: list[PassoCalculo] = []
    avisos: list[str] = []

    # -- 1. Tensão no frio -------------------------------------------------
    voc_fria = voc_corrigida(modulo, temp_min_c)
    maximo_serie = int(inversor.tensao_max_cc // voc_fria) if voc_fria > 0 else 0
    passos.append(PassoCalculo(
        titulo="Tensão de circuito aberto corrigida para a temperatura mínima",
        formula="Voc(T) = Voc_STC × [1 + β_Voc × (T − 25 °C)]",
        substituicao=(
            f"Voc({temp_min_c:.0f} °C) = {modulo.voc:.2f} V × "
            f"[1 + ({COEF_TEMP_VOC:+.4f}/K) × ({temp_min_c:.0f} − 25)]"
        ),
        resultado=f"{voc_fria:.2f} V por módulo",
        comentario=(
            "A tensão de circuito aberto sobe no frio. É este valor, e não o de "
            "catálogo, que não pode ultrapassar a tensão máxima do inversor."
        ),
    ))
    passos.append(PassoCalculo(
        titulo="Máximo de módulos em série",
        formula="N_máx = ⌊ V_máx,CC do inversor ÷ Voc(T_mín) ⌋",
        substituicao=f"N_máx = ⌊ {inversor.tensao_max_cc:.0f} V ÷ {voc_fria:.2f} V ⌋",
        resultado=f"{maximo_serie} módulos",
        comentario="Ultrapassar este limite danifica o inversor no amanhecer mais frio do ano.",
        critico=maximo_serie < 1,
    ))

    # -- 2. Tensão no calor ------------------------------------------------
    vmp_quente = vmp_corrigida(modulo, temp_max_c)
    tensao_partida_com_margem = inversor.tensao_start * 1.10
    minimo_serie = max(1, math.ceil(tensao_partida_com_margem / vmp_quente)) if vmp_quente > 0 else 0
    passos.append(PassoCalculo(
        titulo="Tensão de máxima potência corrigida para a temperatura máxima",
        formula="Vmp(T) = Vmp_STC × [1 + β_Vmp × (T − 25 °C)]",
        substituicao=(
            f"Vmp({temp_max_c:.0f} °C) = {modulo.vmp:.2f} V × "
            f"[1 + ({COEF_TEMP_VMP:+.4f}/K) × ({temp_max_c:.0f} − 25)]"
        ),
        resultado=f"{vmp_quente:.2f} V por módulo",
        comentario="A tensão de operação cai no calor, e é a condição de verão que fixa o mínimo.",
    ))
    passos.append(PassoCalculo(
        titulo="Mínimo de módulos em série",
        formula="N_mín = ⌈ 1,10 × V_partida ÷ Vmp(T_máx) ⌉",
        substituicao=(
            f"N_mín = ⌈ 1,10 × {inversor.tensao_start:.0f} V ÷ {vmp_quente:.2f} V ⌉ "
            f"= ⌈ {tensao_partida_com_margem:.0f} ÷ {vmp_quente:.2f} ⌉"
        ),
        resultado=f"{minimo_serie} módulos",
        comentario=(
            "A margem de 10% sobre a tensão de partida evita que o inversor fique "
            "ligando e desligando em dia encoberto."
        ),
    ))

    faixa_min, faixa_max = modulos_por_string(modulo, inversor, temp_min_c, temp_max_c)
    if faixa_max < faixa_min or faixa_max == 0:
        avisos.append(
            f"Não existe número de módulos em série que satisfaça os dois limites "
            f"simultaneamente: o máximo pela tensão de frio ({maximo_serie}) é menor que o "
            f"mínimo pela tensão de partida ({minimo_serie}). Este par módulo/inversor é "
            "incompatível."
        )
        return MemoriaCalculo(
        modulo, inversor, 0, 0, 1, 0, passos, avisos, temp_min_c, temp_max_c)

    # Adota o máximo da faixa: string mais longa significa menos strings, menos
    # cabo e menos perda, e a tensão mais alta melhora o rendimento do inversor.
    n_serie = faixa_max
    passos.append(PassoCalculo(
        titulo="Módulos em série adotados",
        formula="N_mín ≤ N_série ≤ N_máx, adotando o máximo",
        substituicao=f"{faixa_min} ≤ N_série ≤ {faixa_max}",
        resultado=f"{n_serie} módulos por string",
        comentario=(
            "Adota-se o máximo da faixa: string mais longa usa menos cabo, tem menos "
            "perda e trabalha numa tensão em que o inversor rende mais."
        ),
    ))

    # -- 3. Corrente por MPPT ----------------------------------------------
    imp = modulo.imp if modulo.imp > 0 else modulo.isc * 0.93
    por_operacao = int(inversor.corrente_max_mppt / imp) if imp > 0 else 0
    por_curto = int(inversor.corrente_curto_max_mppt / modulo.isc) if modulo.isc > 0 else 0
    strings_mppt = strings_por_mppt(modulo, inversor)
    passos.append(PassoCalculo(
        titulo="Strings em paralelo por MPPT",
        formula="N_par = mín( ⌊I_máx,MPPT ÷ Imp⌋ , ⌊I_curto,MPPT ÷ Isc⌋ )",
        substituicao=(
            f"N_par = mín( ⌊{inversor.corrente_max_mppt:.1f} A ÷ {imp:.2f} A⌋ , "
            f"⌊{inversor.corrente_curto_max_mppt:.1f} A ÷ {modulo.isc:.2f} A⌋ ) "
            f"= mín({por_operacao}, {por_curto})"
        ),
        resultado=f"{strings_mppt} strings por MPPT",
        comentario=(
            "O datasheet declara dois limites diferentes, e eles significam coisas "
            "diferentes: corrente máxima de operação, comparada com a Imp, e corrente "
            "máxima de curto-circuito, comparada com a Isc. O menor dos dois manda."
        ),
        critico=strings_mppt < 1,
    ))

    strings_por_corrente = strings_mppt * inversor.num_mppt
    passos.append(PassoCalculo(
        titulo="Strings aceitas pela corrente dos MPPTs",
        formula="N_corrente = N_par × número de MPPTs",
        substituicao=f"N_corrente = {strings_mppt} × {inversor.num_mppt}",
        resultado=f"{strings_por_corrente} strings",
    ))

    # -- 4. Limite pela potência FV do inversor ---------------------------
    # A corrente por MPPT não é o único teto: o inversor declara a potência
    # fotovoltaica máxima que aceita, e ela costuma ser o limite que morde
    # primeiro em módulo de alta potência. Sem esta verificação o arranjo
    # fecha no papel com uma razão CC/CA absurda.
    potencia_string_kwp = n_serie * modulo.potencia_wp / 1000.0
    fv_max_kwp = inversor.potencia_fv_max_w / 1000.0
    strings_por_potencia = int(fv_max_kwp // potencia_string_kwp) if potencia_string_kwp else 0
    passos.append(PassoCalculo(
        titulo="Strings aceitas pela potência fotovoltaica máxima",
        formula="N_potência = ⌊ P_FV,máx do inversor ÷ (N_série × P_módulo) ⌋",
        substituicao=(
            f"N_potência = ⌊ {fv_max_kwp:.1f} kWp ÷ ({n_serie} × "
            f"{modulo.potencia_wp:.0f} Wp) ⌋ = ⌊ {fv_max_kwp:.1f} ÷ "
            f"{potencia_string_kwp:.2f} ⌋"
        ),
        resultado=f"{strings_por_potencia} strings",
        comentario=(
            "Com módulo de alta potência este costuma ser o limite que morde antes da "
            "corrente. Ignorá-lo fecha o arranjo no papel com uma razão CC/CA que o "
            "inversor não sustenta."
        ),
        critico=strings_por_potencia < 1,
    ))

    # -- 5. Limite pela razão CC/CA de projeto ----------------------------
    # O fabricante permite mais do que o projeto quer. A série MT anuncia 150%
    # de sobredimensionamento CC; a regra de projeto aqui é 1,35, porque acima
    # disso o corte por potência nos dias claros passa a descartar energia em
    # volume que o kWp extra não paga. Os dois limites existem e são
    # diferentes: o do fabricante protege o equipamento, o de projeto protege
    # o rendimento do investimento.
    teto_projeto_kwp = DC_AC_MAX * inversor.potencia_ca_kw
    strings_por_projeto = int(teto_projeto_kwp // potencia_string_kwp) if potencia_string_kwp else 0
    passos.append(PassoCalculo(
        titulo="Strings aceitas pela razão CC/CA de projeto",
        formula="N_projeto = ⌊ (FDI_máx × P_CA) ÷ (N_série × P_módulo) ⌋",
        substituicao=(
            f"N_projeto = ⌊ ({DC_AC_MAX:.2f} × {inversor.potencia_ca_kw:.1f} kW) ÷ "
            f"{potencia_string_kwp:.2f} kWp ⌋ = ⌊ {teto_projeto_kwp:.2f} ÷ "
            f"{potencia_string_kwp:.2f} ⌋"
        ),
        resultado=f"{strings_por_projeto} strings",
        comentario=(
            f"O fabricante admite até {fv_max_kwp / inversor.potencia_ca_kw:.2f} de razão "
            f"CC/CA; a regra de projeto adotada é {DC_AC_MAX:.2f}. O limite do fabricante "
            "protege o equipamento; o de projeto protege o rendimento do investimento."
        ),
    ))

    # -- 6. Fechamento do arranjo -----------------------------------------
    limites = {
        "corrente dos MPPTs": strings_por_corrente,
        "potência FV do inversor": strings_por_potencia,
        "razão CC/CA de projeto": strings_por_projeto,
    }
    strings_maximas = min(limites.values())
    governa = min(limites, key=lambda k: limites[k])
    passos.append(PassoCalculo(
        titulo="Strings por inversor",
        formula="N_strings = mín( N_corrente , N_potência , N_projeto )",
        substituicao=(
            f"N_strings = mín( {strings_por_corrente} , {strings_por_potencia} , "
            f"{strings_por_projeto} )"
        ),
        resultado=f"{strings_maximas} strings por inversor",
        comentario=f"O limite que governa este arranjo é a **{governa}**.",
    ))

    strings = strings_maximas
    inversores = 1
    strings_do_sistema = strings_maximas
    if modulos_disponiveis is not None:
        cabem = modulos_disponiveis // n_serie if n_serie else 0
        strings = min(strings_maximas, cabem)
        strings_do_sistema = strings
        passos.append(PassoCalculo(
            titulo="Strings limitadas pela área do telhado",
            formula="N_strings = mín( aceitas pelo inversor , ⌊módulos que cabem ÷ N_série⌋ )",
            substituicao=(
                f"N_strings = mín( {strings_maximas} , ⌊{modulos_disponiveis} ÷ {n_serie}⌋ ) "
                f"= mín({strings_maximas}, {cabem})"
            ),
            resultado=f"{strings} strings",
        ))
        if strings_maximas > 0 and cabem > strings_maximas:
            inversores = math.ceil(cabem / strings_maximas)
            strings_do_sistema = cabem
            resto = cabem - (inversores - 1) * strings_maximas
            passos.append(PassoCalculo(
                titulo="Quantidade de inversores",
                formula="N_inversores = ⌈ strings que cabem no telhado ÷ strings por inversor ⌉",
                substituicao=f"N_inversores = ⌈ {cabem} ÷ {strings_maximas} ⌉",
                resultado=(
                    f"{inversores} inversores {inversor.modelo}, somando {cabem} strings"
                    + (f" (o último recebe {resto})" if resto != strings_maximas else "")
                ),
                comentario=(
                    "Um inversor só não absorve o telhado inteiro. O último recebe as "
                    "strings que sobraram, e por isso o total do sistema não é o produto "
                    "simples de inversores por strings."
                ),
            ))

    modulos_totais = n_serie * strings
    modulos_do_sistema = n_serie * strings_do_sistema
    potencia_cc = modulos_totais * modulo.potencia_wp / 1000.0
    razao = potencia_cc / inversor.potencia_ca_kw if inversor.potencia_ca_kw else 0.0

    passos.append(PassoCalculo(
        titulo="Potência do gerador fotovoltaico",
        formula="P_CC = N_série × N_strings × P_módulo",
        substituicao=(
            f"P_CC = {n_serie} × {strings_do_sistema} × {modulo.potencia_wp:.0f} Wp"
        ),
        resultado=(
            f"{modulos_do_sistema * modulo.potencia_wp / 1000.0:.2f} kWp "
            f"({modulos_do_sistema} módulos no sistema, {modulos_totais} por inversor)"
        ),
    ))
    passos.append(PassoCalculo(
        titulo="Razão entre potência CC e potência CA",
        formula="FDI = P_CC ÷ P_CA,nominal",
        substituicao=f"FDI = {potencia_cc:.2f} kWp ÷ {inversor.potencia_ca_kw:.1f} kW",
        resultado=f"{razao:.2f}",
        comentario=(
            f"A faixa de projeto adotada é de {DC_AC_MIN:.2f} a {DC_AC_MAX:.2f}. Abaixo "
            "dela o inversor fica ocioso; acima, o corte por potência nos dias claros "
            "passa a descartar energia em volume relevante."
        ),
    ))
    passos.append(PassoCalculo(
        titulo="Tensão máxima do arranjo, na condição crítica",
        formula="V_string(T_mín) = N_série × Voc(T_mín)",
        substituicao=f"V_string = {n_serie} × {voc_fria:.2f} V",
        resultado=(
            f"{n_serie * voc_fria:.0f} V contra {inversor.tensao_max_cc:.0f} V admissíveis "
            f"({n_serie * voc_fria / inversor.tensao_max_cc:.0%} do limite)"
        ),
        comentario="É esta a verificação que a norma exige e que o instalador confere em campo.",
        critico=n_serie * voc_fria > inversor.tensao_max_cc,
    ))

    if razao > DC_AC_MAX:
        avisos.append(
            f"A razão CC/CA de {razao:.2f} passa do teto de projeto ({DC_AC_MAX:.2f}): "
            "haverá corte por potência nos dias de céu limpo. Considere um inversor maior "
            "ou menos módulos por inversor."
        )
    elif razao < DC_AC_MIN and modulos_totais > 0:
        avisos.append(
            f"A razão CC/CA de {razao:.2f} fica abaixo do piso de projeto ({DC_AC_MIN:.2f}): "
            "o inversor opera bem abaixo do nominal na maior parte do tempo, o que encarece "
            "o quilowatt instalado."
        )
    if modulos_disponiveis is not None:
        sobra = modulos_disponiveis - modulos_do_sistema
        if sobra >= n_serie:
            avisos.append(
                f"Mesmo com {inversores} inversores sobram {sobra} módulos que o telhado "
                "comportaria. O resto não fecha uma string inteira ou exige um inversor "
                "de outro porte."
            )

    return MemoriaCalculo(
        modulo=modulo, inversor=inversor, modulos_por_string=n_serie, strings=strings,
        inversores=inversores, strings_do_sistema=strings_do_sistema,
        passos=passos, avisos=avisos,
        temp_min_c=temp_min_c, temp_max_c=temp_max_c,
    )
