"""
Base de equipamentos (módulos e inversores) lida da planilha Excel.

Além de carregar, este módulo **deriva a geometria física do módulo** quando a
planilha não traz largura e comprimento. A versão anterior assumia 2,279 x
1,134 m fixos para qualquer modelo, o que fazia a restrição de área mentir
para painéis de potência diferente.

Derivação: nas condições padrão de ensaio a irradiância é 1.000 W/m2, logo
    area_m2 = Pmax_W / (eficiencia_percentual / 100 * 1000)
A proporção é fixada em 2:1, típica de módulos de 60/72 células.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import Settings, get_settings

LOGGER = logging.getLogger(__name__)

ABA_PAINEIS = "paineis_solares"
ABA_INVERSORES = "Inversores"

#: Razão comprimento/largura assumida quando as dimensões não são informadas.
PROPORCAO_MODULO = 2.0
#: Dimensões de referência para o caso extremo em que nem potência nem
#: eficiência estão disponíveis (módulo de 550 W, 21% -- padrão de mercado).
MODULO_PADRAO_M = (2.279, 1.134)


class EquipmentError(RuntimeError):
    """Base de equipamentos ausente, vazia ou sem colunas obrigatórias."""


def _num(valor: Any) -> float | None:
    """Converte para float aceitando vírgula decimal; None quando não numérico."""
    if valor is None:
        return None
    if isinstance(valor, (int, float)):
        return None if (isinstance(valor, float) and math.isnan(valor)) else float(valor)
    texto = str(valor).strip().replace(",", ".")
    if not texto:
        return None
    try:
        return float(texto)
    except ValueError:
        return None


@dataclass(frozen=True)
class Modulo:
    """Um modelo de módulo fotovoltaico."""

    modelo: str
    fabricante: str
    potencia_wp: float
    vmp: float
    imp: float
    voc: float
    isc: float
    eficiencia_percent: float
    comprimento_m: float
    largura_m: float

    @property
    def area_m2(self) -> float:
        return self.comprimento_m * self.largura_m

    @property
    def dimensoes_derivadas(self) -> bool:
        """True quando as dimensões vieram da eficiência, não da planilha."""
        return bool(getattr(self, "_derivado", False))

    def __str__(self) -> str:
        return f"{self.fabricante} {self.modelo} ({self.potencia_wp:.0f} Wp)"


@dataclass(frozen=True)
class Inversor:
    """Um modelo de inversor."""

    modelo: str
    fabricante: str
    potencia_ca_w: float
    potencia_fv_max_w: float
    tensao_max_cc: float
    tensao_start: float
    num_mppt: int
    corrente_max_mppt: float
    corrente_curto_max_mppt: float
    fases: int

    @property
    def potencia_ca_kw(self) -> float:
        return self.potencia_ca_w / 1000.0

    def __str__(self) -> str:
        return f"{self.fabricante} {self.modelo} ({self.potencia_ca_kw:.1f} kW)"


def _dimensoes_do_modulo(
    potencia_wp: float | None,
    eficiencia_percent: float | None,
    comprimento: float | None,
    largura: float | None,
) -> tuple[float, float, bool]:
    """
    Resolve (comprimento, largura, foi_derivado) para um módulo.

    Prioriza dimensões explícitas da planilha; cai para derivação física a
    partir de potência e eficiência; por último usa o módulo de referência.
    """
    if comprimento and largura and comprimento > 0 and largura > 0:
        return float(comprimento), float(largura), False

    if potencia_wp and eficiencia_percent and potencia_wp > 0 and eficiencia_percent > 0:
        area = potencia_wp / (eficiencia_percent / 100.0 * 1000.0)
        # area = c * l, com c = PROPORCAO * l  =>  l = sqrt(area / PROPORCAO)
        larg = math.sqrt(area / PROPORCAO_MODULO)
        return round(larg * PROPORCAO_MODULO, 4), round(larg, 4), True

    return MODULO_PADRAO_M[0], MODULO_PADRAO_M[1], True


def carregar_modulos(df: pd.DataFrame) -> list[Modulo]:
    """Converte a aba de painéis em objetos, ignorando linhas inválidas."""
    modulos: list[Modulo] = []
    for _, linha in df.iterrows():
        potencia = _num(linha.get("potencia_maxima_nominal_pmax"))
        voc = _num(linha.get("tensao_circuito_aberto_voc"))
        isc = _num(linha.get("corrente_curto_circuito_isc"))
        if not (potencia and potencia > 0 and voc and voc > 0 and isc and isc > 0):
            LOGGER.debug("Módulo ignorado por dados incompletos: %s", linha.get("modelo"))
            continue

        eficiencia = _num(linha.get("eficiencia_modulo")) or 0.0
        comprimento, largura, derivado = _dimensoes_do_modulo(
            potencia,
            eficiencia,
            _num(linha.get("comprimento_m")) or _num(linha.get("comprimento")),
            _num(linha.get("largura_m")) or _num(linha.get("largura")),
        )
        modulo = Modulo(
            modelo=str(linha.get("modelo") or "sem modelo").strip(),
            fabricante=str(linha.get("fabricante") or "sem fabricante").strip(),
            potencia_wp=potencia,
            vmp=_num(linha.get("tensao_operacao_otima_vmp")) or 0.0,
            imp=_num(linha.get("corrente_operacao_otima_imp")) or 0.0,
            voc=voc,
            isc=isc,
            eficiencia_percent=eficiencia,
            comprimento_m=comprimento,
            largura_m=largura,
        )
        object.__setattr__(modulo, "_derivado", derivado)
        modulos.append(modulo)
    return modulos


def carregar_inversores(df: pd.DataFrame) -> list[Inversor]:
    """Converte a aba de inversores em objetos, ignorando linhas inválidas."""
    inversores: list[Inversor] = []
    for _, linha in df.iterrows():
        potencia_ca = _num(linha.get("maxima_potencia_nominal_ca"))
        tensao_max = _num(linha.get("tensao_maxima_cc"))
        num_mppt = _num(linha.get("numero_mpp_trackers"))
        corrente_mppt = _num(linha.get("corrente_maxima_entrada_por_mpp_tracker"))
        if not (potencia_ca and potencia_ca > 0 and tensao_max and tensao_max > 0):
            LOGGER.debug("Inversor ignorado por dados incompletos: %s", linha.get("modelo"))
            continue
        if not (num_mppt and num_mppt >= 1) or not (corrente_mppt and corrente_mppt > 0):
            LOGGER.debug("Inversor sem dados de MPPT: %s", linha.get("modelo"))
            continue

        # Sem o limite de potência FV declarado, assume a razão CC/CA de projeto.
        potencia_fv = _num(linha.get("potencia_maxima_fv_maxima")) or potencia_ca * 1.3
        inversores.append(
            Inversor(
                modelo=str(linha.get("modelo") or "sem modelo").strip(),
                fabricante=str(linha.get("fabricante") or "sem fabricante").strip(),
                potencia_ca_w=potencia_ca,
                potencia_fv_max_w=potencia_fv,
                tensao_max_cc=tensao_max,
                # Sem tensão de partida, 20% da tensão máxima é uma hipótese
                # conservadora que mantém a janela de MPPT válida.
                tensao_start=_num(linha.get("tensao_start")) or tensao_max * 0.2,
                num_mppt=int(num_mppt),
                corrente_max_mppt=corrente_mppt,
                corrente_curto_max_mppt=(
                    _num(linha.get("corrente_maxima_curto_circuito_por_mpp_tracker")) or corrente_mppt * 1.25
                ),
                fases=int(_num(linha.get("quantidade_fases_ca")) or 1),
            )
        )
    return inversores


@dataclass
class BaseEquipamentos:
    """Catálogo carregado, com acesso por modelo e ordenação por potência."""

    modulos: list[Modulo]
    inversores: list[Inversor]
    caminho: Path | None = None

    @property
    def vazia(self) -> bool:
        return not self.modulos or not self.inversores

    def modulo_por_modelo(self, modelo: str) -> Modulo | None:
        alvo = str(modelo).strip().lower()
        return next((m for m in self.modulos if m.modelo.lower() == alvo), None)

    def inversor_por_modelo(self, modelo: str) -> Inversor | None:
        alvo = str(modelo).strip().lower()
        return next((i for i in self.inversores if i.modelo.lower() == alvo), None)

    def modulo_mais_potente(self) -> Modulo | None:
        return max(self.modulos, key=lambda m: m.potencia_wp, default=None)

    def inversores_ordenados(self) -> list[Inversor]:
        """Do maior para o menor: sistemas grandes devem começar pelos grandes."""
        return sorted(self.inversores, key=lambda i: i.potencia_ca_w, reverse=True)

    def resumo(self) -> dict[str, Any]:
        return {
            "modulos": len(self.modulos),
            "inversores": len(self.inversores),
            "potencia_modulo_min_wp": min((m.potencia_wp for m in self.modulos), default=0),
            "potencia_modulo_max_wp": max((m.potencia_wp for m in self.modulos), default=0),
            "potencia_inversor_min_kw": min((i.potencia_ca_kw for i in self.inversores), default=0),
            "potencia_inversor_max_kw": max((i.potencia_ca_kw for i in self.inversores), default=0),
            "dimensoes_derivadas": sum(1 for m in self.modulos if m.dimensoes_derivadas),
        }


def carregar_base(caminho: str | Path | None = None, settings: Settings | None = None) -> BaseEquipamentos:
    """
    Lê a planilha de equipamentos.

    Levanta :class:`EquipmentError` em vez de devolver DataFrames vazios em
    silêncio -- na versão anterior, um arquivo ausente virava "nenhuma
    combinação encontrada" muitos passos adiante, sem indicar a causa real.
    """
    settings = settings or get_settings()
    caminho = Path(caminho) if caminho else settings.equipment_xlsx

    if not caminho.exists():
        raise EquipmentError(f"Base de equipamentos não encontrada: {caminho}")

    try:
        planilha = pd.ExcelFile(caminho)
        abas = set(planilha.sheet_names)
        aba_p = ABA_PAINEIS if ABA_PAINEIS in abas else next((a for a in abas if "pain" in a.lower()), None)
        aba_i = ABA_INVERSORES if ABA_INVERSORES in abas else next((a for a in abas if "invers" in a.lower()), None)
        if aba_p is None or aba_i is None:
            raise EquipmentError(
                f"Planilha {caminho.name} precisa de uma aba de painéis e uma de inversores. "
                f"Abas encontradas: {sorted(abas)}"
            )
        df_paineis = pd.read_excel(planilha, sheet_name=aba_p)
        df_inversores = pd.read_excel(planilha, sheet_name=aba_i)
    except EquipmentError:
        raise
    except Exception as exc:
        raise EquipmentError(f"Falha ao ler {caminho}: {exc}") from exc

    base = BaseEquipamentos(
        modulos=carregar_modulos(df_paineis),
        inversores=carregar_inversores(df_inversores),
        caminho=caminho,
    )
    if base.vazia:
        raise EquipmentError(
            f"Base {caminho.name} não tem módulos e inversores válidos "
            f"(módulos: {len(base.modulos)}, inversores: {len(base.inversores)})."
        )
    return base


def salvar_base(base: BaseEquipamentos, caminho: str | Path | None = None) -> Path:
    """Grava o catálogo de volta na planilha, preservando as duas abas."""
    destino = Path(caminho) if caminho else (base.caminho or get_settings().equipment_xlsx)
    df_modulos = pd.DataFrame(
        [
            {
                "modelo": m.modelo,
                "fabricante": m.fabricante,
                "potencia_maxima_nominal_pmax": m.potencia_wp,
                "tensao_operacao_otima_vmp": m.vmp,
                "corrente_operacao_otima_imp": m.imp,
                "tensao_circuito_aberto_voc": m.voc,
                "corrente_curto_circuito_isc": m.isc,
                "eficiencia_modulo": m.eficiencia_percent,
                "comprimento_m": m.comprimento_m,
                "largura_m": m.largura_m,
            }
            for m in base.modulos
        ]
    )
    df_inversores = pd.DataFrame(
        [
            {
                "modelo": i.modelo,
                "fabricante": i.fabricante,
                "potencia_maxima_fv_maxima": i.potencia_fv_max_w,
                "tensao_maxima_cc": i.tensao_max_cc,
                "tensao_start": i.tensao_start,
                "numero_mpp_trackers": i.num_mppt,
                "corrente_maxima_entrada_por_mpp_tracker": i.corrente_max_mppt,
                "corrente_maxima_curto_circuito_por_mpp_tracker": i.corrente_curto_max_mppt,
                "maxima_potencia_nominal_ca": i.potencia_ca_w,
                "quantidade_fases_ca": i.fases,
            }
            for i in base.inversores
        ]
    )
    destino.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(destino, engine="openpyxl") as writer:
        df_modulos.to_excel(writer, sheet_name=ABA_PAINEIS, index=False)
        df_inversores.to_excel(writer, sheet_name=ABA_INVERSORES, index=False)
    return destino
