"""
Testes da tensão da rede — o filtro que impede recomendar o que não liga.

A maior parte dos híbridos trifásicos do mercado é de 220/380 V. Num prédio de
127/220 — que é a rede de boa parte do país — recomendar um deles é um erro que
não aparece em simulação nenhuma: aparece na entrega, quando o eletricista
devolve o equipamento. É o tipo de erro que um estudo tem obrigação de não
cometer, porque é barato de evitar e caro de descobrir.
"""
from __future__ import annotations

import pytest

from aurum.bateria.catalogo import InversorHibrido, carregar_catalogo


def _inversor(tensao: str) -> InversorHibrido:
    return InversorHibrido(
        modelo="teste", fabricante="teste",
        potencia_ca_nominal_kw=10.0, potencia_ca_pico_kw=20.0, duracao_pico_s=10.0,
        potencia_fv_max_kw=20.0, potencia_carga_bateria_kw=10.0,
        tensao_bateria_min_v=40.0, tensao_bateria_max_v=60.0,
        tensao_ca_v=tensao,
    )


# ----------------------------------------------------------------------------
# Leitura do que o datasheet escreve
# ----------------------------------------------------------------------------
def test_a_tensao_de_linha_e_o_maior_do_par():
    """
    "127/220" quer dizer 127 V de fase e 220 V de linha.

    Quem dimensiona pensa em linha — é o número da conta de luz e do projeto
    elétrico. Ler o 127 como se fosse a rede faria um inversor de 127/220
    parecer incompatível com uma instalação de 220.
    """
    assert _inversor("127/220").tensoes_de_linha_v == (220.0,)
    assert _inversor("220/380").tensoes_de_linha_v == (380.0,)


def test_modelo_multitensao_lista_todas():
    """A ES LD atende quatro redes; o catálogo precisa saber das quatro."""
    inversor = _inversor("120/208, 127/220, 120/240, 127/254")
    assert inversor.tensoes_de_linha_v == (208.0, 220.0, 240.0, 254.0)
    assert inversor.atende_rede(220)
    assert inversor.atende_rede(208)


def test_tolerancia_junta_380_400_e_415_mas_nao_220():
    """
    380, 400 e 415 V são a mesma rede na prática; 220 não é nenhuma delas.

    A tolerância existe para não reprovar um inversor de 400 V numa rede de
    380 — e não pode ser tão larga que deixe um de 380 passar por 220.
    """
    trifasico = _inversor("220/380, 230/400, 240/415")
    assert trifasico.atende_rede(380)
    assert trifasico.atende_rede(400)
    assert trifasico.atende_rede(415)
    assert not trifasico.atende_rede(220)

    baixa = _inversor("127/220")
    assert baixa.atende_rede(220)
    assert not baixa.atende_rede(380)


def test_planilha_antiga_sem_a_coluna_assume_380():
    """Compatibilidade: o padrão é o que a maioria do catálogo já era."""
    assert InversorHibrido(
        modelo="x", fabricante="y", potencia_ca_nominal_kw=5.0,
        potencia_ca_pico_kw=10.0, duracao_pico_s=10.0, potencia_fv_max_kw=7.5,
        potencia_carga_bateria_kw=5.0, tensao_bateria_min_v=40.0,
        tensao_bateria_max_v=60.0,
    ).atende_rede(380)


# ----------------------------------------------------------------------------
# O catálogo de verdade
# ----------------------------------------------------------------------------
def test_o_catalogo_cobre_as_duas_redes():
    """
    Não adianta o filtro existir se só uma das redes tem equipamento.

    Antes das linhas ES LD, ET LV e ETR, o catálogo inteiro era de 380/400 V:
    o filtro teria zerado a lista em toda instalação de 127/220.
    """
    base = carregar_catalogo()
    for rede in (220.0, 380.0):
        atendem = [i for i in base.inversores if i.atende_rede(rede)]
        assert atendem, f"nenhum inversor atende {rede:.0f} V"
        faixa = [i.potencia_ca_nominal_kw for i in atendem]
        assert min(faixa) <= 10.0, f"falta opção pequena em {rede:.0f} V"
        assert max(faixa) >= 30.0, f"falta opção grande em {rede:.0f} V"


def test_nenhum_inversor_atende_as_duas_redes_por_engano():
    """
    Um modelo que aparecesse nas duas listas seria sinal de tensão mal
    cadastrada — 220 e 380 exigem enrolamentos diferentes.
    """
    base = carregar_catalogo()
    ambiguos = [
        i.modelo for i in base.inversores
        if i.atende_rede(220) and i.atende_rede(380)
    ]
    assert not ambiguos, f"cadastrados nas duas redes: {ambiguos}"


# ----------------------------------------------------------------------------
# O filtro no estudo
# ----------------------------------------------------------------------------
def test_o_filtro_tira_o_que_nao_liga():
    from aurum.bateria.estudo import _filtrar_por_rede

    base = carregar_catalogo()
    filtrada, avisos = _filtrar_por_rede(base, 220.0)
    assert len(filtrada.inversores) < len(base.inversores)
    assert all(i.atende_rede(220) for i in filtrada.inversores)
    assert avisos and "220" in avisos[0]


def test_catalogo_vazio_avisa_em_vez_de_bloquear():
    """
    Zerar o catálogo deixaria o usuário sem resposta nenhuma.

    Avisar deixa a decisão com quem sabe se o projeto prevê transformador —
    que é uma solução real, e não cabe ao software descartá-la.
    """
    from dataclasses import replace

    from aurum.bateria.estudo import _filtrar_por_rede

    base = carregar_catalogo()
    so_380 = replace(base, inversores=[i for i in base.inversores if i.atende_rede(380)])
    filtrada, avisos = _filtrar_por_rede(so_380, 220.0)
    assert filtrada.inversores == so_380.inversores, "não pode esvaziar o catálogo"
    assert avisos and "transformador" in avisos[0]
