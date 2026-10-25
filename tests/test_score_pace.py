"""
Score PACE: os cinco critérios da planilha, a tabela de eficiência e a medição
de hoje a partir do questionário de verdade.
"""
from __future__ import annotations

import numpy as np
import pytest

from aurum.pre_relatorio import score as S
from aurum.pre_relatorio.score import EntradaScore, calcular, com

# ----------------------------------------------------------------------------
# A casa de exemplo da planilha "Score PACE"
# ----------------------------------------------------------------------------
#: Hoje: 100 kWh por pessoa (faixa 4) e tarifa 10% acima da mínima; nada segura
#: a casa sem rede; metade do consumo sobraria com aparelhos eficientes; o
#: consumo encaixa 45% no sol (9 de 15) e chegaria a 60% (12); sem solar, com
#: telhado de sobra.
PLANILHA_HOJE = EntradaScore(
    consumo_mensal_kwh=300, pessoas=3, tarifa_real_brl_kwh=1.0, tarifa_minima_brl_kwh=0.9,
    horas_sem_rede=0, consumo_possivel_mensal_kwh=150,
    encaixe_sol=0.45, encaixe_sol_maximo=0.6, tem_geracao=False,
    producao_anual_kwh=0, producao_possivel_anual_kwh=5000,
)


def test_planilha_hoje():
    s = calcular(PLANILHA_HOJE)
    pontos = {c.id: round(c.pontos, 2) for c in s.criterios}
    assert pontos == {"economico": 3.6, "autonomia": 0, "eficiencia": 15, "uso_no_sol": 4.5, "area": 0}
    assert s.criterio("uso_no_sol").crit == pytest.approx(9)
    assert s.criterio("uso_no_sol").correcao == 0.5
    assert s.valor == pytest.approx(23.1)
    assert s.maximo_local == pytest.approx(97)       # o uso no sol só chega a 12 aqui


def test_planilha_sugestao():
    """Bateria de 3 h que segura tudo, trocas que levam a 28 de 30, horário mudado e solar."""
    sug = com(PLANILHA_HOJE, horas_sem_rede=3, curva_t=1.0, consumo_possivel_mensal_kwh=280,
              encaixe_sol=0.6, tem_geracao=True, producao_anual_kwh=3600)
    assert calcular(sug).valor == pytest.approx(83.6)


def test_so_solar_liga_a_correcao_do_sol():
    s = calcular(com(PLANILHA_HOJE, tem_geracao=True, producao_anual_kwh=3600))
    assert s.criterio("uso_no_sol").pontos == pytest.approx(9)
    assert s.criterio("area").pontos == pytest.approx(30)


def test_faixas():
    assert S.economico(com(PLANILHA_HOJE, pessoas=1)).crit == 1           # 300 kWh/pessoa
    assert S.economico(com(PLANILHA_HOJE, pessoas=None)).crit == S.CRIT_ECONOMICO_NEGOCIO
    assert [S.autonomia(com(PLANILHA_HOJE, horas_sem_rede=h)).crit for h in (0, 1, 3, 6, 12)] == [0, 5, 10, 15, 20]
    # A curva T corta a autonomia: 6 h que seguram 80% das faltas.
    assert S.autonomia(com(PLANILHA_HOJE, horas_sem_rede=6, curva_t=0.8)).pontos == pytest.approx(12)


def test_area_sem_telhado_baixa_o_maximo_do_local():
    apto = com(PLANILHA_HOJE, producao_possivel_anual_kwh=0.0)
    assert calcular(apto).maximo_local == pytest.approx(67)
    desconhecido = com(PLANILHA_HOJE, producao_possivel_anual_kwh=None)
    assert calcular(desconhecido).criterio("area").teto_local == 30


# ----------------------------------------------------------------------------
# Curvas
# ----------------------------------------------------------------------------
SOL = np.clip(np.sin((np.arange(24) + 0.5 - 6) / 12 * np.pi), 0, None)


def test_encaixe():
    assert S.encaixe_com_sol(SOL * 3, SOL) == pytest.approx(1.0)
    noite = np.where(SOL > 0, 0.0, 1.0)
    assert S.encaixe_com_sol(noite, SOL) == pytest.approx(0.0)
    plano = np.ones(24)
    depois = S.deslocar_para_o_sol(plano, SOL, 6.0)
    assert depois.sum() == pytest.approx(24.0)
    assert S.encaixe_com_sol(depois, SOL) > S.encaixe_com_sol(plano, SOL)


def test_tarifa_branca():
    # Consumo concentrado na ponta: a branca sai mais cara (razão > 1).
    ponta = np.zeros(24)
    ponta[18:21] = 1
    assert S.razao_tarifa_branca(ponta) > 1
    # Consumo de madrugada: a branca é mais barata.
    madrugada = np.zeros(24)
    madrugada[0:6] = 1
    assert S.razao_tarifa_branca(madrugada) == pytest.approx(S.BRANCA_FORA_PONTA)


def test_telhado():
    assert S.kwp_do_telhado({"acesso": "predio"}) == 0
    assert S.kwp_do_telhado({"acesso": "proprio"}) is None
    assert S.kwp_do_telhado({"acesso": "proprio", "tamanho": "m", "sombra": "nao"}) == pytest.approx(60 * 0.6 * 0.2)
    assert S.kwp_existente({"solar": True}, {"placas": "12"}) == pytest.approx(6.6)
    assert S.kwp_existente({"solar": True}, {}) is None
    assert S.kwp_existente({}, {}) == 0


# ----------------------------------------------------------------------------
# Do questionário de verdade (JavaScript no Node) à medição de hoje
# ----------------------------------------------------------------------------
NODE_OK = pytest.importorskip("shutil").which("node") is not None


@pytest.mark.skipif(not NODE_OK, reason="Node.js ausente")
def test_hoje_da_casa():
    from test_pre_relatorio import CASA, _pacote

    from aurum.pre_relatorio.curva import montar_curva
    from aurum.pre_relatorio.eficiencia import avaliar
    from aurum.pre_relatorio.entrada import montar_entrada

    respostas = dict(CASA, aparelhos={"ar": "nenhum", "refrigeracao": "velho", "luz": "poucas", "lavar": "daria"},
                     imovel={"acesso": "proprio", "tamanho": "m", "sombra": "nao", "tarifa": "convencional"})
    pacote = _pacote(respostas)
    q = pacote["questionario"]
    assert q["aparelhos"]["ar"] == "nenhum" and q["imovel"] == {"acesso": "proprio", "tamanho": "m", "sombra": "nao"}
    assert q["tarifaModalidade"] == "convencional"
    assert any(i.get("cartaoId") == "ar" for i in pacote["itens"])

    e = montar_entrada(pacote, consumo_mensal_kwh=450, tarifa_brl_kwh=1.05)
    curva = montar_curva(e.dados.cenario, e.conta, e.perfil_id, simulacoes=60, rotulos=e.rotulos)
    ef = avaliar(curva.cargas, e.pacote)
    acoes = {t.classe: t for t in ef.trocas}
    assert {"ar", "refrigeracao", "luz", "chuveiro"} <= set(acoes)
    assert acoes["ar"].unidades == 2                      # os dois, nenhum é inverter
    assert 0.4 < ef.razao < 1

    m = S.medir_hoje(e, curva)
    s = calcular(m.hoje)
    assert m.hoje.pessoas == 4 and not m.hoje.tem_geracao
    assert s.criterio("autonomia").pontos == 0
    assert s.criterio("area").pontos == 0 and s.criterio("area").teto_local > 0
    assert s.criterio("uso_no_sol").teto_local > s.criterio("uso_no_sol").crit    # a máquina de lavar vai para o sol
    assert 0 < s.valor < s.maximo_local <= 100


@pytest.mark.skipif(not NODE_OK, reason="Node.js ausente")
def test_hoje_da_padaria():
    from test_pre_relatorio import PADARIA, _pacote

    from aurum.pre_relatorio.curva import montar_curva
    from aurum.pre_relatorio.entrada import montar_entrada

    respostas = dict(PADARIA, aparelhos={"expositor": "aberto", "luz": "maioria"},
                     imovel={"acesso": "predio"})
    e = montar_entrada(_pacote(respostas), consumo_mensal_kwh=2000, tarifa_brl_kwh=1.1)
    curva = montar_curva(e.dados.cenario, e.conta, e.perfil_id, simulacoes=60, rotulos=e.rotulos)
    m = S.medir_hoje(e, curva)
    assert m.hoje.pessoas is None                         # negócio: faixa neutra
    assert any(t.classe == "expositor" and t.unidades == 2 for t in m.eficiencia.trocas)
    s = calcular(m.hoje)
    assert s.criterio("area").teto_local == 0             # em prédio, sem telhado
    assert s.maximo_local <= 70


# ----------------------------------------------------------------------------
# O score do app (vistoria/cliente/score.js), com os mesmos parâmetros
# ----------------------------------------------------------------------------
SCRIPT_APP = """
global.window = globalThis;
window.PACE_SCORE_PARAM = JSON.parse(require('fs').readFileSync(process.argv[2], 'utf8'));
require(process.argv[1] + '/dados.js');
require(process.argv[1] + '/cliente/negocio.js');
require(process.argv[1] + '/cliente/cliente.js');
require(process.argv[1] + '/cliente/score.js');
const M = globalThis.PQ.modelo;
const e = Object.assign(M.novoEstado(), JSON.parse(process.argv[3]));
const r = PQS.calcular(e);
process.stdout.write(JSON.stringify({r: {hoje: r.hoje, sugestao: r.sugestao, cenarios: r.cenarios, sugestoes: r.sugestoes,
  consumo: r.consumoMensalKwh}, svg: PQS.cardSvg(r, {mostrarValores: true}), svgCliente: PQS.cardSvg(r)}));
"""


def _score_app(respostas: dict) -> dict:
    import json
    import shutil
    import subprocess
    from pathlib import Path

    raiz = Path(__file__).resolve().parents[1]
    saida = subprocess.run([shutil.which("node"), "-e", SCRIPT_APP, str(raiz / "vistoria"),
                            str(raiz / "aurum" / "pre_relatorio" / "score_parametros.json"), json.dumps(respostas)],
                           capture_output=True, text=True, encoding="utf-8", check=True)
    return json.loads(saida.stdout)


@pytest.mark.skipif(not NODE_OK, reason="Node.js ausente")
def test_score_do_app():
    import xml.etree.ElementTree as ET

    from test_pre_relatorio import CASA

    respostas = dict(CASA, contaValor="480",
                     aparelhos={"ar": "nenhum", "refrigeracao": "velho", "luz": "poucas", "lavar": "daria"},
                     imovel={"acesso": "proprio", "tamanho": "m", "sombra": "nao"})
    o = _score_app(respostas)
    r = o["r"]
    assert r["consumo"] == pytest.approx(480)                 # R$ 480 ÷ tarifa padrão R$ 1,00
    ids = [c["id"] for c in r["cenarios"]]
    assert ids == ["hoje", "solar", "sugestao", "maximo"]
    v = {c["id"]: c["valor"] for c in r["cenarios"]}
    assert 0 < v["hoje"] < v["solar"] < v["sugestao"] <= v["maximo"] <= 100
    assert r["sugestoes"][0]["id"] == "solar"                # sem solar, é o que mais soma
    assert all(s["pontos"] > 0 for s in r["sugestoes"])
    # "Como subir sua pontuação" soma exatamente a sugestão PACE
    assert sum(s["pontos"] for s in r["sugestoes"]) == round(v["sugestao"]) - round(v["hoje"])
    assert len(r["sugestoes"]) <= 4
    assert "COMO SUBIR SUA PONTUA" in o["svg"]
    # o card é SVG válido, com e sem preço
    for svg in (o["svg"], o["svgCliente"]):
        assert ET.fromstring(svg).tag.endswith("svg")
    assert "R$" in o["svg"] and "R$" not in o["svgCliente"]   # o cliente não vê preço


@pytest.mark.skipif(not NODE_OK, reason="Node.js ausente")
def test_app_e_python_concordam_nos_criterios():
    """Mesmas medidas, mesma nota: os dois leem score_parametros.json."""
    import json
    import shutil
    import subprocess
    from pathlib import Path

    raiz = Path(__file__).resolve().parents[1]
    js = """
global.window = globalThis;
window.PACE_SCORE_PARAM = JSON.parse(require('fs').readFileSync(process.argv[1], 'utf8'));
require(process.argv[2] + '/cliente/score.js');
const x = JSON.parse(process.argv[3]);
process.stdout.write(JSON.stringify(PQS.interno.criterios(x)));
"""
    x = {"consumo": 300, "pessoas": 3, "tarifaReal": 1.0, "tarifaMinima": 0.9, "horas": 0, "curvaT": None,
         "possivel": 150, "encaixe": 0.45, "encaixeMax": 0.6, "temGeracao": False, "producao": 0, "producaoPossivel": 5000}
    saida = subprocess.run([shutil.which("node"), "-e", js, str(raiz / "aurum" / "pre_relatorio" / "score_parametros.json"),
                            str(raiz / "vistoria"), json.dumps(x)], capture_output=True, text=True, encoding="utf-8", check=True)
    app = json.loads(saida.stdout)
    py = calcular(PLANILHA_HOJE)
    assert app["valor"] == pytest.approx(py.valor)
    assert app["maximoLocal"] == pytest.approx(py.maximo_local)


@pytest.mark.skipif(not NODE_OK, reason="Node.js ausente")
def test_apartamento_nao_tem_so_solar():
    from test_pre_relatorio import CASA

    o = _score_app(dict(CASA, contaValor="480", imovel={"acesso": "predio"}))
    ids = [c["id"] for c in o["r"]["cenarios"]]
    assert "solar" not in ids and ids == ["hoje", "sugestao", "maximo"]
    assert "Só solar" not in o["svgCliente"]


@pytest.mark.skipif(not NODE_OK, reason="Node.js ausente")
def test_quem_ja_tem_solar_amplia():
    from test_pre_relatorio import CASA

    o = _score_app(dict(CASA, contaValor="480", fontes={"solar": True},
                        imovel={"acesso": "proprio", "tamanho": "m", "sombra": "nao", "placas": "6"}))
    nomes = {c["id"]: c["nome"] for c in o["r"]["cenarios"]}
    assert nomes.get("solar") == "Ampliar o solar"
    assert "sem telhado" not in o["svgCliente"]
