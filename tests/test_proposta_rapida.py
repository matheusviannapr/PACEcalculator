"""
Dimensionamento rápido do CRM: a conta em Python (aurum.pre_relatorio.proposta_rapida)
e a do navegador (vistoria/cliente/dimensionamento.js) dão o mesmo número, e o
preço e a parcela são os da Calculator.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from aurum.bateria.apresentacao import OpcoesComerciais, parcela_price
from aurum.pre_relatorio import proposta_rapida as pr
from aurum.pv import kits

JS = Path(__file__).resolve().parents[1] / "vistoria" / "cliente" / "dimensionamento.js"
NODE = shutil.which("node")


@pytest.fixture(scope="module")
def P():
    return pr.parametros_para_crm()


CASOS = [
    {"consumo_kwh": 200, "tarifa": 1.15},                                   # Uilson, Light
    {"consumo_kwh": 348, "tarifa": 1.15, "ligacao": "Monofásico"},
    {"consumo_kwh": 522, "tarifa": 1.15, "bateria_kwh": 12},                # Maria, Unamar
    {"consumo_kwh": 136, "tarifa": 1.47, "ligacao": "Bifásico"},            # Enel R$ 200
    {"consumo_kwh": 40, "tarifa": 1.15},                                    # abaixo da disponibilidade
    {"consumo_kwh": 1800, "tarifa": 1.0, "ligacao": "Trifásico", "latitude": -15.8, "cobertura": 0.8},
    {"consumo_kwh": 6500, "tarifa": 0.95, "ligacao": "Trifásico"},          # acima de 40 kWp: só trifásico
    {"consumo_kwh": 450, "tarifa": 1.15, "mao_de_obra_kwp": 500,
     "curva": [0.2] * 7 + [1.0] * 11 + [0.6] * 6},                          # quem consome de dia
]


def test_preco_e_parcela_sao_os_da_calculator(P):
    for kwp in (1.2, 5, 6.25, 13.6, 20, 40, 41, 90):
        for col in ("microinversor", "splitphase", "trifasico"):
            a, b = pr.preco_kit(P, kwp, col), kits.preco_kit(kwp, col)
            assert (a is None) == (b is None) and (a is None or abs(a - b) < 1e-6), (kwp, col)
    o = OpcoesComerciais()
    assert pr.parcela(P, 42653) == pytest.approx(parcela_price(
        42653, o.juros_financiamento_am, o.prazo_financiamento_meses, o.carencia_financiamento_meses))


def test_dimensiona_como_a_regra(P):
    r = pr.dimensionar(P, {"consumo_kwh": 200, "tarifa": 1.15, "ano": 2026})
    assert r["disponibilidade_kwh"] == 50 and r["modulos"] == 3 and r["kwp"] == pytest.approx(1.875)
    assert r["conta_hoje_brl"] == pytest.approx(230) and 50 * 1.15 <= r["conta_solar_brl"] < 100
    micro, split, bat = r["opcoes"]
    # 1,875 kWp: abaixo do kit de 5 kWp, metade fixa e metade proporcional (1,875/5 = 37,5%).
    fator = 0.5 + 0.5 * 1.875 / 5
    assert micro["investimento_brl"] == pytest.approx(20_550 * fator) and split["investimento_brl"] == pytest.approx(30_653 * fator)
    assert bat["investimento_brl"] == pytest.approx(30_653 * fator + 12_000) and bat["bateria_kwh"] == 5
    assert micro["payback_anos"] < split["payback_anos"] < bat["payback_anos"]
    nada = pr.dimensionar(P, {"consumo_kwh": 40, "tarifa": 1.15})
    assert nada["kwp"] == 0 and all(o["investimento_brl"] is None for o in nada["opcoes"])
    grande = pr.dimensionar(P, {"consumo_kwh": 6500, "tarifa": 0.95, "ligacao": "Trifásico"})
    assert grande["kwp"] > 40 and all(o["investimento_brl"] is None for o in grande["opcoes"])   # sem kit de tabela


def test_bateria_pelas_cargas_criticas(P):
    assert pr.bateria_sugerida(P, None, "horas") == 5
    assert pr.bateria_sugerida(P, 2.4, "horas") == 5            # 0,1 kW × 3 h ÷ 0,9 → 1 bloco
    assert pr.bateria_sugerida(P, 48, "dia") == 30              # 2 kW × 12 h ÷ 0,9 = 26,7 → 6 blocos


@pytest.mark.skipif(NODE is None or not JS.exists(), reason="Node.js ou o CRM (vistoria/) ausente")
def test_navegador_da_o_mesmo_numero(P):
    script = """
const d = require(process.argv[1]); const P = JSON.parse(process.argv[2]); const casos = JSON.parse(process.argv[3]);
process.stdout.write(JSON.stringify(casos.map(c => d.dimensionar(c, P))));
"""
    casos = [dict(c, ano=2026) for c in CASOS]
    saida = subprocess.run([NODE, "-e", script, str(JS), json.dumps(P), json.dumps(casos)],
                           capture_output=True, text=True, encoding="utf-8", check=True).stdout
    for caso, js in zip(casos, json.loads(saida)):
        py = pr.dimensionar(P, caso)
        for k in ("kwp", "modulos", "geracao_kwh", "autoconsumo_kwh", "conta_hoje_brl", "conta_solar_brl", "economia_mes_brl"):
            assert js[k] == pytest.approx(py[k], rel=1e-9, abs=1e-9), (caso, k)
        assert js["economia_por_ano"] == pytest.approx(py["economia_por_ano"], rel=1e-9, abs=1e-9), caso
        for a, b in zip(js["opcoes"], py["opcoes"]):
            for k in ("investimento_brl", "parcela_brl", "payback_anos", "bateria_kwh"):
                assert (a[k] is None) == (b[k] is None) and (a[k] is None or a[k] == pytest.approx(b[k], rel=1e-9)), (caso, a["id"], k)


LOCAIS = [
    ("Rio de Janeiro / RJ", "Light", False), ("Belford Roxo", "Light", False), ("sao joao de meriti", "Light", False),
    ("Duque de Caxias / RJ", "Light", True), ("Três Rios - RJ", "Light", True), ("Volta Redonda, RJ", "Light", False),
    ("Petrópolis / RJ", "Enel", False), ("Niteroi", "Enel", False), ("São Gonçalo", "Enel", False),
    ("Cabo Frio / RJ", "Enel", False), ("Magé", "Enel", False), ("Resende", "Enel", False),
    ("Caxias", "Light", True), ("Rio", "Light", False),
    ("Santa Cruz", None, False), ("Unamar", None, False), ("São Paulo / SP", None, False), ("", None, False),
]


@pytest.mark.parametrize("texto,esperada,parcial", LOCAIS)
def test_distribuidora_pela_cidade(P, texto, esperada, parcial):
    r = pr.distribuidora_do_local(P, texto)
    assert r["distribuidora"] == esperada and r["parcial"] == parcial, r


def test_area_da_light_e_os_municipios_do_rj(P):
    a = P["areaLight"]
    assert len(a["municipios"]) == 31 and len(a["municipiosRJ"]) == 92
    assert set(a["parciais"]) <= set(a["municipios"]) <= set(a["municipiosRJ"])
    assert P["tarifas"]["Light"] == 1.15 and P["tarifas"]["Enel"] == 1.47


@pytest.mark.skipif(NODE is None or not JS.exists(), reason="Node.js ou o CRM (vistoria/) ausente")
def test_navegador_acha_a_mesma_distribuidora(P):
    script = """
const d = require(process.argv[1]); const P = JSON.parse(process.argv[2]); const textos = JSON.parse(process.argv[3]);
process.stdout.write(JSON.stringify(textos.map(t => d.distribuidoraDoLocal(t, P))));
"""
    textos = [t for t, _, _ in LOCAIS]
    js = json.loads(subprocess.run([NODE, "-e", script, str(JS), json.dumps(P), json.dumps(textos)],
                                   capture_output=True, text=True, encoding="utf-8", check=True).stdout)
    for t, r in zip(textos, js):
        assert r == pr.distribuidora_do_local(P, t), t
