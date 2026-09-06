"""
Correções aplicadas ao núcleo do D² sem editar a extração literal.

:mod:`aurum.demanda.nucleo` é uma cópia fiel do simulador do D², e o valor
dessa fidelidade é poder comparar os dois lado a lado. Correções de
comportamento não podem, portanto, ser escritas lá dentro. Ficam aqui, uma por
função, ligadas explicitamente por um gerenciador de contexto.

Cada correção tem um teste em ``tests/test_bateria.py`` que falha se o
comportamento original voltar.

---

**Janela que atravessa a meia-noite (`intervalos_circulares`)**

``Equipamento.simula_carga`` faz ``carga[inicio:fim] += potência``. Quando a
janela cruza a meia-noite — "18:00 as 06:00" vira ``(1080, 360)`` — a fatia
``carga[1080:360]`` é **vazia** em NumPy, e o equipamento desaparece do dia
inteiro sem erro, sem aviso e sem deixar rastro. O mesmo acontece por outro
caminho com o gerador dinâmico, que pode produzir ``fim > 1440`` e ter o
excedente truncado na meia-noite.

O efeito não é sutil e não é acadêmico:

* iluminação de área comum, bombas noturnas, câmaras frias e sistemas de
  segurança — as cargas que passam a noite — somam **zero** ao perfil;
* como são justamente as cargas que um sistema de backup existe para atender,
  um estudo de bateria feito sobre o perfil não corrigido dimensionaria o banco
  para a metade errada do dia.

A correção trata o dia como cíclico, que é o que um perfil diário é: a janela
``(1080, 360)`` acende das 18 h à meia-noite e da meia-noite às 6 h. A energia
total passa a ser conservada, e nenhum caso que já funcionava muda de valor —
janelas dentro do dia seguem a mesma aritmética de antes.

Vale a pena corrigir isso também no repositório do D²: qualquer perfil já
gerado lá com equipamento noturno está subestimado.

---

**Janela de operação noturna com duração sorteada (`gerar_intervalo_uso`)**

O mesmo problema pela outra porta. Equipamento de duração variável dentro de
uma janela — um degelo de câmara fria que roda 2 h em algum momento entre
22 h e 2 h — passa por ``gerar_intervalo_uso``, que **levanta ValueError**
quando ``janela_fim <= janela_inicio``. Não some em silêncio como o caso
anterior: quebra a simulação inteira, e o usuário fica sem saber que o
problema é a janela de um equipamento entre dezenas.

A correção estende a janela em um dia quando ela vira a meia-noite. O sorteio
do início passa a valer no intervalo estendido, e o índice resultante pode
passar de 1440 — o que ``simula_carga_circular`` já sabe tratar, porque as
duas correções são a mesma ideia aplicada em dois pontos.
"""
from __future__ import annotations

import random
from contextlib import ExitStack, contextmanager
from typing import Iterator, Optional, Tuple

import numpy as np

from . import nucleo
from .nucleo import Equipamento, _duracao_horas_para_passos, parse_time

__all__ = [
    "gerar_intervalo_uso_circular",
    "intervalos_circulares",
    "janelas_circulares",
    "simula_carga_circular",
]


def simula_carga_circular(self: Equipamento, tempo_total: int = 1440) -> np.ndarray:
    """``Equipamento.simula_carga`` com o dia tratado como circular."""
    carga = np.zeros(tempo_total)
    intervalos = self.intervalos() if callable(self.intervalos) else self.intervalos

    if self.probabilisticado_no_intervalo or np.random.rand() < self.probabilidade:
        potencia_efetiva = self.potencia * self.fator_demanda * self.quantidade
        for bruto_inicio, bruto_fim in intervalos:
            inicio = int(bruto_inicio) % tempo_total
            fim = int(bruto_fim)
            if fim == int(bruto_inicio):
                continue
            # Comprimento medido no tempo original, antes de dobrar o dia: é
            # ele que distingue "das 18 h às 6 h" (12 h) de um erro de digitação.
            duracao = fim - int(bruto_inicio)
            if duracao < 0:
                duracao += tempo_total
            duracao = min(duracao, tempo_total)
            indices = (np.arange(inicio, inicio + duracao) % tempo_total)
            carga[indices] += potencia_efetiva
    return carga


def gerar_intervalo_uso_circular(
    janela_inicio: str,
    janela_fim: str,
    duracao_min_h: float,
    duracao_max_h: float,
    dt_min: int = 1,
    probabilidade: float = 1.0,
    seed: Optional[int] = None,
    on_overflow: str = "clamp",
) -> Optional[Tuple[int, int]]:
    """``gerar_intervalo_uso`` aceitando janela que vira a meia-noite."""
    if dt_min <= 0:
        raise ValueError("dt_min deve ser > 0")
    rng = random.Random(seed) if seed is not None else random
    if rng.random() > probabilidade:
        return None
    if duracao_min_h > duracao_max_h:
        raise ValueError("duracao_min_h não pode ser maior que duracao_max_h")

    inicio_min = parse_time(janela_inicio)
    fim_min = parse_time(janela_fim)
    if fim_min <= inicio_min:
        # A janela vira o dia: estende em 24 h em vez de recusar. O índice
        # resultante pode passar de 1440, e quem consome sabe dobrar o dia.
        fim_min += 1440

    duracao_h = rng.uniform(duracao_min_h, duracao_max_h)
    n_passos = _duracao_horas_para_passos(duracao_h, dt_min)

    inicio_idx = inicio_min // dt_min
    fim_idx = fim_min // dt_min
    tamanho = fim_idx - inicio_idx
    if n_passos > tamanho:
        if on_overflow == "clamp":
            n_passos = tamanho
        else:
            return None

    sorteado = rng.randint(inicio_idx, fim_idx - n_passos)
    return sorteado, sorteado + n_passos


@contextmanager
def janelas_circulares(ativo: bool = True) -> Iterator[None]:
    """Liga só a correção da janela de duração sorteada."""
    if not ativo:
        yield
        return
    original = nucleo.gerar_intervalo_uso
    nucleo.gerar_intervalo_uso = gerar_intervalo_uso_circular  # type: ignore[assignment]
    try:
        yield
    finally:
        nucleo.gerar_intervalo_uso = original  # type: ignore[assignment]


@contextmanager
def intervalos_circulares(ativo: bool = True) -> Iterator[None]:
    """
    Liga as duas correções de meia-noite pelo tempo do bloco.

    Gerenciador de contexto, e não patch permanente no import, para que quem
    quiser reproduzir o número original do D² consiga — é a diferença entre
    corrigir um modelo e apagar a versão anterior dele.
    """
    if not ativo:
        yield
        return
    original = Equipamento.simula_carga
    Equipamento.simula_carga = simula_carga_circular  # type: ignore[method-assign]
    try:
        with ExitStack() as pilha:
            pilha.enter_context(janelas_circulares())
            yield
    finally:
        Equipamento.simula_carga = original  # type: ignore[method-assign]
