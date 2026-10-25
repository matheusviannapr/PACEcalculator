"""
Pré-relatório: do JSON do questionário à curva híbrida, ao estudo e ao resumo.

Os pacotes vêm do JavaScript de verdade (``vistoria/cliente``), rodado no
Node, como no teste do questionário: se a tradução do app mudar, este teste
pega antes do operador. A série solar é sintética para o teste não depender
da rede (o estudo de verdade consulta o PVGIS).
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from aurum.pre_relatorio.curva import CRITICAS, DEMAIS, montar_curva
from aurum.pre_relatorio.entrada import montar_entrada

RAIZ = Path(__file__).resolve().parents[1] / "vistoria"
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="Node.js ausente")

PNG = ("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
       "z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")
BASE = {
    "conta": {"nome": "c.png", "tipo": "image/png", "dataUrl": PNG}, "aceite": True,
    "local": {"latitude": -22.9, "longitude": -43.1, "origem": "t", "confirmadoEm": "2026-09-27T00:00:00Z"},
    "dados": {"nome": "Maria Souza", "whatsapp": "21999990000", "endereco": "Rua A", "cidade": "Niterói",
              "uf": "RJ", "empresa": "Padaria Pão Quente"},
    "apagao": {"frequencia": "mensal", "duracao": "horas"},
}
CASA = dict(BASE, id="CLI-TESTE-CASA", falta={"geladeira": 1, "internet": 1, "ar": 2},
            tolerancia={"geladeira": "C", "internet": "MC", "ar": "C"},
            casa={"tv": 1, "chuveiro": 2, "microondas": 1, "lavar": 1, "forno": 1},
            rotina={"pessoas": "4", "diaSemana": "uma", "almoco": "as_vezes", "fimDeSemana": "ficam",
                    "acorda": "normal", "janta": "normal", "dorme": "normal", "arNoite": "noite_toda",
                    "madrugada": "nunca"}, contaValor="850")
PADARIA = dict(BASE, id="CLI-TESTE-PADA", tipo="negocio", setor="padaria",
               falta={"n_forno_paes": 1, "n_expositor": 2, "n_pdv": 1},
               tolerancia={"n_forno_paes": "C", "n_expositor": "MC", "n_pdv": "MC"},
               casa={"n_internet": 1, "n_luz": 1, "n_masseira": 1, "n_ar": 2, "n_letreiro": 1, "n_refrigerador": 1},
               negocio={"dias": "todos", "abre": "madrugada", "fecha": "noite", "pico": "manha", "producao": "muito",
                        "tamanho": "2", "impacto": {"mercadoria": True}}, contaValor="2.400")

SCRIPT = """
global.window = globalThis;
require(process.argv[1] + '/dados.js');
require(process.argv[1] + '/cliente/negocio.js');
require(process.argv[1] + '/cliente/cliente.js');
const M = globalThis.PQ.modelo;
const e = Object.assign(M.novoEstado(), JSON.parse(process.argv[2]));
process.stdout.write(JSON.stringify(M.montarPacote(e, '2026-09-27T12:00:00.000Z')));
"""


def _pacote(respostas: dict) -> dict:
    saida = subprocess.run([NODE, "-e", SCRIPT, str(RAIZ), json.dumps(respostas)],
                           capture_output=True, text=True, encoding="utf-8", check=True)
    return json.loads(saida.stdout)


@pytest.fixture(scope="module")
def casa() -> dict:
    return _pacote(CASA)


@pytest.fixture(scope="module")
def padaria() -> dict:
    return _pacote(PADARIA)


# ----------------------------------------------------------------------------
# Entrada
# ----------------------------------------------------------------------------
def test_refrigeracao_vira_carga_continua(casa):
    e = montar_entrada(casa, consumo_mensal_kwh=450, tarifa_brl_kwh=1.05)
    geladeira = [i for i in e.pacote["itens"] if i["equipamento"] == "Geladeira duplex"][0]
    assert (geladeira["tipoInt"], geladeira["intervalo"], geladeira["fd"]) == ("fixo", "00:00 as 23:59", 0.35)
    curva = montar_curva(e.dados.cenario, e.conta, e.perfil_id, simulacoes=60)
    kwh_mes = float(curva.cargas.loc[curva.cargas["equipamento"] == "Geladeira duplex", "energia_kwh_mes"].sum())
    assert 20 <= kwh_mes <= 60     # antes da correção saía ~1 kWh/mês


def test_origem_do_consumo(casa):
    assert montar_entrada(casa, consumo_mensal_kwh=450, tarifa_brl_kwh=1.0).origem_consumo == "conta"
    pelo_valor = montar_entrada(casa, tarifa_brl_kwh=1.0)
    assert pelo_valor.origem_consumo == "valor"
    assert pelo_valor.conta.consumo_mensal_kwh == pytest.approx(850)    # R$ 850 ÷ R$ 1,00
    assert any("estimado pelo valor" in a for a in pelo_valor.avisos)


def test_negocio_leva_o_horario_para_a_conta(padaria):
    e = montar_entrada(padaria, consumo_mensal_kwh=2000, tarifa_brl_kwh=1.1)
    assert (e.conta.abertura_h, e.conta.fechamento_h, e.conta.dias_operacao_semana) == (5, 22, 7)
    assert e.perfil_id == "padaria"
    assert e.rotulos["Expositor refrigerado"] == "Balcão refrigerado"


# ----------------------------------------------------------------------------
# Curva híbrida
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("nome,kwh", [("casa", 450), ("padaria", 2000)])
def test_curva_fecha_com_a_conta_e_contem_as_criticas(nome, kwh, request):
    e = montar_entrada(request.getfixturevalue(nome), consumo_mensal_kwh=kwh, tarifa_brl_kwh=1.0)
    c = montar_curva(e.dados.cenario, e.conta, e.perfil_id, simulacoes=60, rotulos=e.rotulos)
    # A energia do mês é a da conta.
    assert c.ajuste.erros_pct["energia_mensal_kwh"] == pytest.approx(0.0, abs=0.5)
    # Em nenhuma hora o total fica abaixo das críticas.
    assert np.all(c.curva_total_kw + 1e-9 >= c.curva_critica_kw)
    # O ensemble total tem a energia do dia de operação.
    assert float(c.ensemble_total.energia_diaria_kwh().mean()) == pytest.approx(c.curva_total_kw.sum(), rel=0.08)
    # A tabela de cargas soma a conta, e as críticas estão nela com o nível declarado.
    assert float(c.cargas["energia_kwh_mes"].sum()) == pytest.approx(kwh, rel=0.01)
    assert c.cargas["criticidade"].isin(CRITICAS).any()
    assert DEMAIS in set(c.cargas["equipamento"])
    assert 0.25 <= c.peso_respostas <= 0.75


def test_padaria_produz_de_madrugada(padaria):
    e = montar_entrada(padaria, consumo_mensal_kwh=2000, tarifa_brl_kwh=1.1)
    c = montar_curva(e.dados.cenario, e.conta, e.perfil_id, simulacoes=60, rotulos=e.rotulos)
    # Forno, masseira e fermentação antes de abrir: a madrugada pesa mais que a noite fechada.
    assert c.curva_total_kw[3:6].mean() > c.curva_total_kw[22:24].mean()
    top = c.principais_consumidores(3)
    assert "Balcão refrigerado" in set(top["rotulo"]) or "Forno de pães" in set(top["rotulo"])
    assert top["fracao"].sum() == pytest.approx(1.0, abs=0.01)


# ----------------------------------------------------------------------------
# Estudo e resumo
# ----------------------------------------------------------------------------
@pytest.fixture
def sem_rede(monkeypatch):
    """A série solar sintética no lugar do PVGIS."""
    from aurum.bateria import estudo as modulo
    from aurum.bateria.geracao import serie_sintetica

    def falsa(lat, lon, azimute_deg=None, inclinacao_deg=None, anos=None, permitir_fallback=True):
        return serie_sintetica(lat, lon, azimute_deg or 0.0, inclinacao_deg or 20.0, 14.0, anos=(2020, 2020))
    monkeypatch.setattr(modulo, "obter_serie_horaria", falsa)


def test_resumo_da_casa(casa, sem_rede):
    from aurum.pre_relatorio.estudo import OpcoesPreRelatorio, gerar

    e = montar_entrada(casa, consumo_mensal_kwh=450, tarifa_brl_kwh=1.05)
    pr = gerar(e, OpcoesPreRelatorio(simulacoes=60, amostras_apagao=20, candidatos=4))
    r = pr.resumo
    assert r.criticos[:2] == ["a internet", "a geladeira"]      # como o cliente disse, MC primeiro
    assert r.bateria_kwh and r.bateria_kwh > 0
    assert r.autonomia_h is not None
    assert [c.duracao_h for c in r.cobertura] == [1.0, 3.0, 6.0, 12.0]
    assert all(0 <= c.prob <= 1 for c in r.cobertura)
    assert len(r.curva_consumo_kw) == len(r.curva_solar_kw) == 24
    assert 0 < r.fracao_sem_sol < 1
    assert r.consumidores and r.consumidores[-1].nome == "Demais cargas"
    assert sum(c.fracao for c in r.consumidores) == pytest.approx(1.0, abs=0.01)
    assert r.conta_atual_brl == pytest.approx(450 * 1.05, abs=10)
    assert r.conta_solar_min_brl < r.conta_solar_max_brl < r.conta_atual_brl
    assert r.investimento_brl is None           # só com mostrar_investimento
    assert r.origem_consumo == "conta"
    assert 0 <= r.indice <= 100 and r.indice_faixa
    assert [p["nome"] for p in r.indice_partes] == ["Economia", "Falta de luz", "Consumo sem sol", "Encaixe da solução"]
    assert sum(p["pontos"] for p in r.indice_partes) == pytest.approx(r.indice, abs=0.6)
