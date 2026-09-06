"""
Testes da tabela de preços de referência.

Preço não é datasheet: tem data, varia entre distribuidores e envelhece. O que
se testa aqui não é o valor — esse muda — mas as propriedades que impedem a
tabela de virar folclore de planilha: que cada faixa seja coerente, que diga de
onde veio, e que os padrões do estudo saiam daqui em vez de estarem soltos em
três arquivos diferentes.
"""
from __future__ import annotations

from datetime import date

import pytest

from aurum.bateria.despacho import Gerador
from aurum.bateria.economia import PremissasBateria
from aurum.precos import (
    PRECOS,
    FaixaPreco,
    preco_bateria_brl,
    preco_inversor_hibrido_brl,
    preco_modulo_brl,
    procedencia,
)


def test_toda_faixa_e_coerente_e_datada():
    for chave, faixa in PRECOS.items():
        assert faixa.minimo <= faixa.tipico <= faixa.maximo, chave
        assert faixa.minimo > 0, chave
        assert faixa.unidade, chave
        # A origem é o que separa um número de um chute lembrado de cor.
        assert len(faixa.fonte) > 40, f"{chave}: a origem precisa dizer de onde veio"
        assert isinstance(faixa.apurado_em, date), chave


def test_faixa_invertida_e_recusada_na_construcao():
    """Erro de digitação em faixa é silencioso; melhor estourar na carga."""
    with pytest.raises(ValueError, match="inconsistente"):
        FaixaPreco(10.0, 5.0, 20.0, "R$/kWh", "x" * 50, date(2026, 9, 1))


def test_dispersao_mede_a_incerteza():
    faixa = FaixaPreco(1.0, 1.5, 2.0, "R$/W", "x" * 50, date(2026, 9, 1))
    assert faixa.dispersao == pytest.approx(1.0)


def test_descricao_nao_come_a_virgula_da_frase():
    """
    O separador de milhar vira ponto só no número.

    Aplicar `.replace(",", ".")` na frase inteira é o atalho óbvio, e já mordeu
    este repositório duas vezes: transformava "3.200,00 R$/kWh, apurado em"
    em "3.200.00 R$/kWh. apurado em".
    """
    texto = PRECOS["bateria_lfp_brl_kwh"].descricao()
    assert "3.200,00" in texto
    assert ", apurado em" in texto


def test_precos_por_unidade_escalam_com_a_grandeza():
    assert preco_modulo_brl(620) == pytest.approx(620 * PRECOS["modulo_brl_wp"].tipico)
    assert preco_inversor_hibrido_brl(10) == pytest.approx(
        10_000 * PRECOS["inversor_hibrido_brl_w"].tipico
    )
    assert preco_bateria_brl(10) == pytest.approx(10 * PRECOS["bateria_lfp_brl_kwh"].tipico)
    # Grandeza negativa não vira desconto.
    assert preco_modulo_brl(-100) == 0.0


def test_nivel_pessimista_e_otimista_cercam_o_tipico():
    baixo = preco_bateria_brl(10, "minimo")
    alto = preco_bateria_brl(10, "maximo")
    assert baixo < preco_bateria_brl(10) < alto


def test_o_estudo_usa_a_tabela_em_vez_de_numeros_soltos():
    """
    Uma fonte só. Antes, o preço do kWh de bateria estava escrito em
    `economia.py`, o do gerador em `despacho.py`, e nenhum dos dois dizia de
    quando era — então atualizar o mercado exigia caçar constantes.
    """
    premissas = PremissasBateria()
    assert premissas.capex_bateria_brl_kwh == PRECOS["bateria_lfp_brl_kwh"].tipico
    assert premissas.capex_inversor_brl_kw == PRECOS["inversor_hibrido_brl_w"].tipico * 1000
    assert Gerador(10.0).custo_energia_brl_kwh == PRECOS["gerador_diesel_brl_kwh"].tipico


def test_procedencia_diz_a_data_e_a_ressalva():
    texto = procedencia()
    assert "2026" in texto
    assert "cotação" in texto, "o texto precisa dizer que orçamento substitui a faixa"
