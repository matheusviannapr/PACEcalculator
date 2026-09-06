"""Formatação numérica no padrão pt-BR (1.234,56)."""
from __future__ import annotations

import math
from typing import Any

DASH = "—"


def _finite(value: Any) -> float | None:
    """Converte para float, devolvendo None para nulos/NaN/infinitos."""
    if value is None or isinstance(value, bool):
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(num) or math.isinf(num):
        return None
    return num


def br_float(value: Any, casas: int = 2, default: str = DASH) -> str:
    """1234.5 -> '1.234,50'."""
    num = _finite(value)
    if num is None:
        return default
    texto = f"{num:,.{casas}f}"
    # troca separadores en-US por pt-BR em uma passagem
    return texto.replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def br_int(value: Any, default: str = DASH) -> str:
    """1234.7 -> '1.235'."""
    return br_float(value, casas=0, default=default)


def br_money(value: Any, casas: int = 2, default: str = DASH, simbolo: str = "R$ ") -> str:
    """1234.5 -> 'R$ 1.234,50'."""
    num = _finite(value)
    if num is None:
        return default
    return f"{simbolo}{br_float(num, casas)}"


def br_percent(value: Any, casas: int = 1, default: str = DASH, fracao: bool = False) -> str:
    """
    Formata percentual. Com ``fracao=True`` a entrada é 0..1 e é multiplicada
    por 100; caso contrário a entrada já está em pontos percentuais.
    """
    num = _finite(value)
    if num is None:
        return default
    if fracao:
        num *= 100.0
    return f"{br_float(num, casas)}%"
