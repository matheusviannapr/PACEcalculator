"""
Semente do catálogo fotovoltaico: módulos e inversores de rede.

A planilha ``BDFotovoltaica.xlsx`` continua sendo a fonte da verdade — quem
manda é o que está lá. Este módulo existe para duas coisas que a planilha
sozinha não dá:

* **Regenerar** o catálogo do zero, ou **acrescentar** modelos a um catálogo
  existente sem sobrescrever o que o usuário já ajustou
  (:func:`semear_planilha`).
* **Registrar a procedência de cada linha.** A coluna ``fonte_dado`` diz de
  onde veio cada conjunto de números: o URL do datasheet do fabricante, ou a
  marca de que o valor foi *derivado* e não lido. A diferença importa — um
  Voc lido do datasheet e um Voc estimado pela contagem de células levam à
  mesma string de texto e a graus de confiança muito diferentes.

Os valores marcados ``derivado`` seguem a mesma física em todos os casos:
módulo TOPCon de N células em série tem Voc ≈ 0,73 V por célula e
Vmp ≈ 0,61 V por célula a STC. Serve para o arranjo de strings fechar; não
serve para memorial de cálculo assinado. Substitua pelo datasheet do lote que
você comprou.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import get_settings

LOGGER = logging.getLogger(__name__)

__all__ = ["SEMENTE_INVERSORES", "SEMENTE_MODULOS", "semear_planilha"]

ABA_PAINEIS = "paineis_solares"
ABA_INVERSORES = "Inversores"

#: Datasheets consultados. Guardados como constantes para que a planilha
#: aponte o documento, e não uma descrição vaga da origem.
FONTES = {
    "longi_himo7": "LONGi Hi-MO 7 LR7-72HGD (datasheet do fabricante)",
    "dah_72fs": "DAH Solar DHN-72X16/FS(BW) (datasheet do fabricante)",
    "dah_78dg": "DAH Solar DHN-78X16/DG(BW) 620-640W (datasheet do fabricante)",
    "goodwe_mt": "GoodWe MT 50-80kW — en.goodwe.com/Ftp/EN/Downloads/Datasheet/GW_MT_Datasheet-EN.pdf",
    "goodwe_smt": "GoodWe SMT 50-60kW — en.goodwe.com/Ftp/EN/Downloads/Datasheet/GW_SMT 50-60kW_Datasheet-EN.pdf",
}

#: Sufixo aplicado a campo que foi calculado, não lido.
DERIVADO = " | elétricos derivados (0,73 V Voc e 0,61 V Vmp por célula TOPCon)"


def _modulo(
    modelo: str,
    fabricante: str,
    pmax: float,
    vmp: float,
    imp: float,
    voc: float,
    isc: float,
    eficiencia: float,
    comprimento_m: float,
    largura_m: float,
    fonte: str,
) -> dict[str, Any]:
    return {
        "modelo": modelo,
        "fabricante": fabricante,
        "potencia_maxima_nominal_pmax": pmax,
        "tensao_operacao_otima_vmp": vmp,
        "corrente_operacao_otima_imp": imp,
        "tensao_circuito_aberto_voc": voc,
        "corrente_curto_circuito_isc": isc,
        "eficiencia_modulo": eficiencia,
        "comprimento_m": comprimento_m,
        "largura_m": largura_m,
        "fonte_dado": fonte,
    }


#: Módulos das marcas que a PACE especifica hoje. As dimensões importam tanto
#: quanto a potência: é delas que sai quantos módulos cabem no telhado, e o
#: código antes assumia 2,279 × 1,134 m para qualquer modelo — o que faz a
#: restrição de área mentir para um painel de 2,465 m de comprimento.
SEMENTE_MODULOS: list[dict[str, Any]] = [
    _modulo("LR7-72HGD-620M", "LONGi", 620, 44.33, 13.99, 52.77, 14.85, 23.0,
            2.382, 1.134, FONTES["longi_himo7"]),
    _modulo("LR7-72HGD-610M", "LONGi", 610, 43.97, 13.87, 52.51, 14.72, 22.6,
            2.382, 1.134, FONTES["longi_himo7"] + DERIVADO),
    _modulo("LR7-72HGD-585M", "LONGi", 585, 43.30, 13.51, 52.10, 14.50, 21.7,
            2.382, 1.134, FONTES["longi_himo7"] + DERIVADO),
    _modulo("DHN-78X16/DG(BW)-620W", "DAH Solar", 620, 47.60, 13.03, 56.90, 13.64, 22.18,
            2.465, 1.134, FONTES["dah_78dg"] + DERIVADO),
    _modulo("DHN-78X16/DG(BW)-640W", "DAH Solar", 640, 48.10, 13.31, 57.30, 13.93, 22.90,
            2.465, 1.134, FONTES["dah_78dg"] + DERIVADO),
    _modulo("DHN-72X16/FS(BW)-585W", "DAH Solar", 585, 43.80, 13.36, 51.60, 14.20, 22.64,
            2.279, 1.134, FONTES["dah_72fs"]),
]


def _inversor(
    modelo: str,
    fabricante: str,
    fv_max_w: float,
    v_max_cc: float,
    v_start: float,
    v_nominal: float,
    faixa_mpp: str,
    n_mppt: int,
    i_max_mppt: float,
    isc_max_mppt: float,
    p_ca_w: float,
    s_ca_va: float,
    v_ca: str,
    i_saida_max: float,
    fases: int,
    fonte: str,
) -> dict[str, Any]:
    return {
        "modelo": modelo,
        "fabricante": fabricante,
        "potencia_maxima_fv_maxima": fv_max_w,
        "tensao_maxima_cc": v_max_cc,
        "tensao_start": v_start,
        "tensao_nominal": v_nominal,
        "faixa_tensao_mpp": faixa_mpp,
        "numero_mpp_trackers": n_mppt,
        "corrente_maxima_entrada_por_mpp_tracker": i_max_mppt,
        "corrente_maxima_curto_circuito_por_mpp_tracker": isc_max_mppt,
        "maxima_potencia_nominal_ca": p_ca_w,
        "potencia_maxima_aparente_ca": s_ca_va,
        "tensao_nominal_ca": v_ca,
        "frequencia_rede_ca": "50/60Hz",
        "corrente_saida_maxima": i_saida_max,
        "fator_potencia_ajustavel": "0.8i-0.8c",
        "quantidade_fases_ca": fases,
        "fonte_dado": fonte,
    }


#: Inversores de rede GoodWe. A potência FV máxima não vem tabelada nos
#: datasheets das séries MT e SMT: vem da regra de sobredimensionamento CC que
#: o próprio documento anuncia — 150% para a MT, 130% para a SMT.
_MT = FONTES["goodwe_mt"]
_SMT = FONTES["goodwe_smt"]
_FV = " | potência FV máxima = 150% do nominal CA (sobredimensionamento declarado na série)"
_FV_SMT = " | potência FV máxima = 130% do nominal CA"

SEMENTE_INVERSORES: list[dict[str, Any]] = [
    # -- MT 50-80 kW, trifásico, 4 MPPT ------------------------------------
    _inversor("GW50KN-MT", "GoodWe", 75000, 1100, 200, 620, "200V-1000V", 4,
              33.0, 41.5, 50000, 55000, "400", 80.0, 3, _MT + _FV
              + " | correntes por MPPT 33/33/22/22 A; a menor governa o arranjo"),
    _inversor("GW60KN-MT", "GoodWe", 90000, 1100, 200, 620, "200V-1000V", 4,
              33.0, 41.5, 60000, 66000, "400", 96.0, 3, _MT + _FV),
    _inversor("GW50KBF-MT", "GoodWe", 75000, 1100, 200, 620, "200V-1000V", 4,
              30.0, 37.5, 50000, 55000, "400", 80.0, 3, _MT + _FV),
    _inversor("GW60KBF-MT", "GoodWe", 90000, 1100, 200, 620, "200V-1000V", 4,
              44.0, 55.0, 60000, 66000, "400", 96.0, 3, _MT + _FV),
    _inversor("GW75KBF-MT", "GoodWe", 112500, 1100, 200, 750, "200V-1000V", 4,
              44.0, 55.0, 75000, 82500, "500", 96.0, 3, _MT + _FV),
    _inversor("GW80KBF-MT", "GoodWe", 120000, 1100, 200, 800, "200V-1000V", 4,
              39.0, 54.8, 80000, 88000, "540", 95.3, 3, _MT + _FV),
    _inversor("GW70KHV-MT", "GoodWe", 105000, 1100, 200, 750, "200V-1000V", 4,
              33.0, 41.5, 70000, 77000, "500", 94.1, 3, _MT + _FV),
    _inversor("GW80KHV-MT", "GoodWe", 120000, 1100, 200, 800, "200V-1000V", 4,
              44.0, 55.0, 80000, 88000, "540", 89.0, 3, _MT + _FV),
    _inversor("GW75K-MT", "GoodWe", 112500, 1100, 200, 600, "200V-1000V", 4,
              44.0, 55.0, 75000, 75000, "400", 133.0, 3, _MT + _FV),
    _inversor("GW80K-MT", "GoodWe", 120000, 1100, 200, 620, "200V-1000V", 4,
              44.0, 55.0, 80000, 88000, "400", 133.0, 3, _MT + _FV),
    # -- SMT 50-60 kW, trifásico, 5 e 6 MPPT -------------------------------
    _inversor("GW50KS-MT", "GoodWe", 65000, 1100, 180, 600, "200V-950V", 5,
              30.0, 37.5, 50000, 50000, "220/380 (Brasil)", 80.0, 3, _SMT + _FV_SMT
              + " | no Brasil a saída nominal é 220/380 V e a potência ativa máxima, 50 kW"),
    _inversor("GW60KS-MT", "GoodWe", 78000, 1100, 180, 600, "200V-950V", 6,
              30.0, 37.5, 60000, 60000, "220/380 (Brasil)", 96.0, 3, _SMT + _FV_SMT
              + " | no Brasil a potência ativa máxima é 60 kW"),
]


def semear_planilha(
    caminho: str | Path | None = None,
    substituir: bool = False,
) -> tuple[Path, dict[str, int]]:
    """
    Acrescenta os modelos da semente à ``BDFotovoltaica.xlsx``.

    Por padrão **acrescenta**: linhas cujo modelo já existe na planilha são
    deixadas em paz, porque o que está lá pode ter sido corrigido à mão com o
    datasheet do lote comprado — e sobrescrever isso apagaria trabalho. Com
    ``substituir=True``, a semente vence.

    Devolve o caminho e quantas linhas foram acrescentadas em cada aba.
    """
    destino = Path(caminho) if caminho else Path(get_settings().equipment_xlsx)
    existente: dict[str, pd.DataFrame] = {}
    if destino.exists():
        existente = pd.read_excel(destino, sheet_name=None)

    resumo: dict[str, int] = {}
    saida: dict[str, pd.DataFrame] = dict(existente)

    for aba, semente in ((ABA_PAINEIS, SEMENTE_MODULOS), (ABA_INVERSORES, SEMENTE_INVERSORES)):
        atual = existente.get(aba, pd.DataFrame())
        novos = pd.DataFrame(semente)
        if atual.empty:
            saida[aba] = novos
            resumo[aba] = len(novos)
            continue

        conhecidos = {str(m).strip().lower() for m in atual.get("modelo", [])}
        if substituir:
            filtrados = atual[~atual["modelo"].astype(str).str.strip().str.lower().isin(
                {str(m).strip().lower() for m in novos["modelo"]})]
            saida[aba] = pd.concat([filtrados, novos], ignore_index=True)
            resumo[aba] = len(novos)
        else:
            faltantes = novos[~novos["modelo"].astype(str).str.strip().str.lower().isin(conhecidos)]
            saida[aba] = pd.concat([atual, faltantes], ignore_index=True)
            resumo[aba] = len(faltantes)

        # Linhas antigas sem procedência declarada ficam marcadas como tal, em
        # vez de passarem por conferidas só porque estavam na planilha antes.
        if "fonte_dado" in saida[aba].columns:
            saida[aba]["fonte_dado"] = saida[aba]["fonte_dado"].fillna("a conferir")

    destino.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(destino, engine="openpyxl") as escritor:
        for aba, tabela in saida.items():
            tabela.to_excel(escritor, sheet_name=str(aba)[:31], index=False)
    return destino, resumo
