"""
Dimensionamento rápido: o sistema e as três propostas que o CRM mostra na
ficha do lead, em milissegundos e sem o motor do estudo.

Não é o estudo. O estudo (``aurum.bateria.estudo``) roda Monte Carlo de
demanda, a série solar hora a hora e o despacho da bateria; aqui a conta é a
de balcão, explícita, para o SDR responder o cliente na hora. O que ela tem
em comum com o estudo é o que mais importa no número final, e vem do mesmo
código:

* o **preço** é o da tabela de kits (:func:`aurum.pv.kits.preco_kit`), com a
  mesma interpolação e o mesmo piso de 5 kWp, e a bateria em blocos;
* a **parcela** é a :func:`aurum.bateria.apresentacao.parcela_price`, com os
  juros, o prazo e a carência padrão da apresentação;
* a **geração** por kWp é a série sintética que a tela da Calculator usa para
  pré-dimensionar (:func:`aurum.bateria.geracao.serie_sintetica`);
* o **Fio B** segue o cronograma da Lei 14.300 (:mod:`aurum.pv.financials`).

As três propostas são as da apresentação: microinversor, split-phase sem
banco e split-phase com banco (:data:`aurum.pv.kits.TOPOLOGIAS_DE_PROPOSTA`).

O CRM roda a mesma conta em JavaScript (``vistoria/cliente/dimensionamento.js``)
com os parâmetros que :func:`parametros_para_crm` exporta no build, e um teste
compara as duas linha a linha.
"""
from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path
from typing import Any

PARAMETROS_JSON = Path(__file__).with_name("proposta_parametros.json")
SCORE_JSON = Path(__file__).with_name("score_parametros.json")

#: As três propostas, na ordem da apresentação: id -> (coluna da tabela, com bateria).
OPCOES = (
    ("microinversor", "microinversor", False),
    ("splitphase", "splitphase", False),
    ("splitphase_bateria", "splitphase", True),
)


def parametros_para_crm() -> dict[str, Any]:
    """Tudo o que o dimensionamento usa, num dicionário que vira JSON no navegador."""
    from ..bateria.apresentacao import OpcoesComerciais
    from ..bateria.geracao import serie_sintetica
    from ..pv import financials, kits

    import numpy as np

    p = json.loads(PARAMETROS_JSON.read_text(encoding="utf-8"))
    score = json.loads(SCORE_JSON.read_text(encoding="utf-8"))

    produtividade = []
    for lat in range(-1, -35, -2):
        s = serie_sintetica(float(lat), -43.0, 0.0, 20.0, 14.0, anos=(2019, 2020))
        produtividade.append([float(lat), round(float(s.anual_kwh_por_kwp()), 1)])
    s = serie_sintetica(float(p["latitudePadrao"]), -43.2, 0.0, 20.0, 14.0, anos=(2019, 2020))
    media = np.asarray(s.potencia_w_por_kwp, dtype=float).mean(axis=0)
    forma_solar = [round(float(v), 5) for v in media / media.sum()]

    com = OpcoesComerciais()
    tabela = [
        {"kwp": kwp, "modulos": mods, **{k: v for k, v in valores.items()}}
        for kwp, (mods, valores) in sorted(kits.TABELA_KIT.items())
    ]
    est = score["estimativa"]
    return {
        **{k: v for k, v in p.items() if not k.startswith("_")},
        "capa": {k: v for k, v in p["capa"].items() if not k.startswith("_")},
        "tarifas": {k: v for k, v in p["tarifas"].items() if not k.startswith("_")},
        "ligacoes": {k: v for k, v in p["ligacoes"].items() if not k.startswith("_")},
        "kits": tabela,
        "kitsApuradoEm": kits.APURADO_EM.isoformat(),
        "potenciaMinimaKwp": kits.POTENCIA_MINIMA_KWP,
        "fracaoFixaAbaixoDaTabela": kits.FRACAO_FIXA_ABAIXO_DA_TABELA,
        "bateriaBlocoKwh": kits.BATERIA_BLOCO_KWH,
        "bateriaBlocoBrl": kits.BATERIA_BLOCO_BRL,
        "bateriaEmTabelaAte": kits.BATERIA_EM_TABELA_ATE,
        "maoDeObraKwp": kits.MAO_DE_OBRA_BRL_KWP,
        "materialCaKwp": kits.MATERIAL_CA_BRL_KWP,
        "fioB": {"fracao": financials.FRACAO_FIO_B_NA_TARIFA,
                 "cronograma": {str(a): f for a, f in financials.CRONOGRAMA_FIO_B.items()},
                 "depois": financials.FIO_B_APOS_TRANSICAO},
        "financiamento": {"jurosAm": com.juros_financiamento_am, "prazoMeses": com.prazo_financiamento_meses,
                          "carenciaMeses": com.carencia_financiamento_meses},
        "comercial": {"seguroMes": com.seguro_brl_mes, "gerenciamentoMes": com.gerenciamento_brl_mes,
                      "validadeDias": com.validade_dias, "entrada": com.entrada_financiamento_brl,
                      "estrutura": com.estrutura, "garantiaInversorAnos": com.garantia_inversor_anos,
                      "garantiaModuloAnos": com.garantia_modulo_anos, "garantiaBateriaAnos": com.garantia_bateria_anos,
                      "garantiaEstruturaAnos": com.garantia_estrutura_anos, "clientesGeridos": com.clientes_geridos,
                      "potenciaGerida": com.potencia_gerida, "anosFluxo": list(com.anos_fluxo)},
        "produtividade": produtividade,
        "formaSolar": forma_solar,
        "formaCasa": est.get("formaCasa") or score.get("formaCasa"),
        "bateria": {"profundidade": est["profundidadeBateria"], "horasDaFalta": est["horasDaFalta"]},
    }


# ----------------------------------------------------------------------------
# A distribuidora pela cidade (espelhada em dimensionamento.js)
# ----------------------------------------------------------------------------
#: Como o cliente costuma escrever a cidade, sem acento e em minúsculas.
APELIDOS_CIDADE = {"rio": "Rio de Janeiro", "rj": "Rio de Janeiro", "caxias": "Duque de Caxias",
                   "meriti": "São João de Meriti", "sjm": "São João de Meriti", "sao joao": "São João de Meriti",
                   "nova iguacu": "Nova Iguaçu", "sao goncalo": "São Gonçalo", "niteroi": "Niterói"}


def _normal(texto: str) -> str:
    import unicodedata
    t = unicodedata.normalize("NFKD", str(texto or "")).encode("ascii", "ignore").decode().lower()
    return " ".join(t.replace("-", " ").split())


def separar_cidade_uf(texto: str) -> tuple[str, str]:
    """'Petrópolis / RJ', 'Niterói-RJ', 'Rio de Janeiro, RJ' -> ('Petrópolis', 'RJ')."""
    import re
    t = str(texto or "").strip()
    m = re.match(r"^(.*?)[\s]*[/,\-][\s]*([A-Za-z]{2})\s*$", t)
    return (m.group(1).strip(), m.group(2).upper()) if m else (t, "")


def distribuidora_do_local(P: dict, texto: str) -> dict[str, Any]:
    """
    No estado do Rio: município da área da Light é Light; qualquer outro município
    do estado é Enel. Bairro, cidade de fora ou nome que não é município: ``None``
    (o SDR escolhe), porque chutar Enel para "Santa Cruz" erraria a tarifa.
    """
    a = P["areaLight"]
    cidade, uf = separar_cidade_uf(texto)
    if not cidade:
        return {"distribuidora": None, "cidade": "", "parcial": False, "motivo": "sem cidade"}
    if uf and uf != a["uf"]:
        return {"distribuidora": None, "cidade": cidade, "parcial": False, "motivo": "fora do RJ"}
    n = _normal(cidade)
    todos = {_normal(m): m for m in a["municipiosRJ"]}
    nome = todos.get(n) or APELIDOS_CIDADE.get(n)
    if not nome:
        return {"distribuidora": None, "cidade": cidade, "parcial": False, "motivo": "não é um município do RJ"}
    light = {_normal(m) for m in a["municipios"]}
    if _normal(nome) in light:
        return {"distribuidora": "Light", "cidade": nome, "parcial": nome in a["parciais"], "motivo": "área da Light"}
    return {"distribuidora": a["resto"], "cidade": nome, "parcial": False, "motivo": "fora da área da Light"}


# ----------------------------------------------------------------------------
# A conta (espelhada em dimensionamento.js)
# ----------------------------------------------------------------------------
def _interp(pontos: list[list[float]], x: float) -> float:
    pts = sorted(pontos)
    if x <= pts[0][0]:
        return pts[0][1]
    if x >= pts[-1][0]:
        return pts[-1][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x <= x1:
            return y0 + (x - x0) / (x1 - x0) * (y1 - y0)
    return pts[-1][1]


def preco_kit(P: dict, kwp: float, coluna: str) -> float | None:
    """Mesmo que :func:`aurum.pv.kits.preco_kit`, lido da tabela exportada."""
    pontos = [(k["kwp"], k[coluna]) for k in P["kits"] if k.get(coluna) is not None]
    if not pontos or kwp <= 0:
        return 0.0 if kwp <= 0 else None
    if kwp > pontos[-1][0]:
        return None
    if kwp <= pontos[0][0]:
        menor = pontos[0][0]
        if menor <= P["potenciaMinimaKwp"] and kwp < menor:     # metade fixa, metade proporcional
            f = P["fracaoFixaAbaixoDaTabela"]
            return pontos[0][1] * (f + (1 - f) * kwp / menor)
        return pontos[0][1]
    for (x0, y0), (x1, y1) in zip(pontos, pontos[1:]):
        if kwp <= x1:
            return y0 + (kwp - x0) / (x1 - x0) * (y1 - y0)
    return pontos[-1][1]


def parcela(P: dict, principal: float) -> float:
    """Tabela Price com juros da carência capitalizados (``parcela_price``)."""
    f = P["financiamento"]
    j, n, c = f["jurosAm"], f["prazoMeses"], f["carenciaMeses"]
    if principal <= 0 or n <= 0:
        return 0.0
    saldo = principal * (1 + j) ** c
    return saldo / n if j <= 0 else saldo * j / (1 - (1 + j) ** (-n))


def fator_fio_b(P: dict, ano: int) -> float:
    cron = {int(a): f for a, f in P["fioB"]["cronograma"].items()}
    if ano in cron:
        return cron[ano]
    return 0.0 if ano < min(cron) else P["fioB"]["depois"]


def bateria_sugerida(P: dict, energia_critica_dia_kwh: float | None, duracao_falta: str | None) -> float:
    """O banco que segura as cargas críticas pela falta declarada, em blocos."""
    passo = P["bateriaBlocoKwh"]
    if not energia_critica_dia_kwh:
        return float(P["bateriaPadraoKwh"])
    horas = P["bateria"]["horasDaFalta"].get(duracao_falta or "", 3)
    kwh = energia_critica_dia_kwh / 24.0 * horas / P["bateria"]["profundidade"]
    return max(passo, math.ceil(kwh / passo - 1e-9) * passo)


def dimensionar(P: dict, e: dict) -> dict[str, Any]:
    """
    ``e``: consumo_kwh (mês), tarifa, ligacao, latitude, cobertura (0–1),
    curva (24 valores, a forma do dia; opcional), bateria_kwh, ano,
    mao_de_obra_kwp (opcional; padrão o da tabela).
    """
    consumo = max(0.0, float(e.get("consumo_kwh") or 0))
    tarifa = float(e.get("tarifa") or P["tarifas"]["padrao"])
    ligacao = e.get("ligacao") if e.get("ligacao") in P["ligacoes"] else P["ligacaoPadrao"]
    disp = float(P["ligacoes"][ligacao])
    lat = float(e.get("latitude") if e.get("latitude") is not None else P["latitudePadrao"])
    cobertura = float(e.get("cobertura") if e.get("cobertura") is not None else P["coberturaPadrao"])
    ano = int(e.get("ano") or date.today().year)
    obra_kwp = float(e.get("mao_de_obra_kwp") if e.get("mao_de_obra_kwp") is not None else P["maoDeObraKwp"]) \
        + float(P["materialCaKwp"])

    prod = _interp(P["produtividade"], lat)
    alvo = max(0.0, consumo - disp) * cobertura
    kwp_bruto = alvo * 12 / prod if prod > 0 else 0.0
    modulos = math.ceil(kwp_bruto * 1000 / P["moduloWp"] - 1e-9) if kwp_bruto > 0 else 0
    kwp = modulos * P["moduloWp"] / 1000.0
    geracao = kwp * prod / 12.0

    forma = [max(0.0, float(x)) for x in (e.get("curva") or P["formaCasa"])]
    sf = sum(forma) or 1.0
    cons_h = [x / sf * consumo / 30.0 for x in forma]
    ger_h = [x * geracao / 30.0 for x in P["formaSolar"]]
    autoconsumo = sum(min(c, g) for c, g in zip(cons_h, ger_h)) * 30.0
    injetado = max(0.0, geracao - autoconsumo)

    def conta_com_solar(tar: float, a: int) -> float:
        compra = consumo - autoconsumo
        compensado = min(injetado, max(0.0, compra - disp))
        faturado = max(compra - compensado, disp)
        return faturado * tar + compensado * tar * P["fioB"]["fracao"] * fator_fio_b(P, a)

    conta_hoje = max(consumo, disp) * tarifa
    conta_solar = conta_com_solar(tarifa, ano) if kwp > 0 else conta_hoje
    economia_mes = conta_hoje - conta_solar

    def economia_no_ano(i: int) -> float:
        tar = tarifa * (1 + P["escaladaTarifa"]) ** i
        return 12 * (max(consumo, disp) * tar - conta_com_solar(tar, ano + i)) if kwp > 0 else 0.0

    #: Economia de cada ano (1º, 2º, ...), com o reajuste da tarifa e o Fio B subindo.
    economia_por_ano = [economia_no_ano(i) for i in range(P["anosPayback"])]

    def payback(capex: float) -> float | None:
        acumulado = 0.0
        for i in range(P["anosPayback"]):
            tar = tarifa * (1 + P["escaladaTarifa"]) ** i
            eco = 12 * (max(consumo, disp) * tar - conta_com_solar(tar, ano + i))
            if eco > 0 and acumulado + eco >= capex:
                return i + (capex - acumulado) / eco
            acumulado += eco
        return None

    bateria = float(e.get("bateria_kwh") or P["bateriaPadraoKwh"])
    blocos = math.ceil(bateria / P["bateriaBlocoKwh"] - 1e-9)
    opcoes = []
    for oid, coluna, com_bateria in OPCOES:
        kit = preco_kit(P, kwp, coluna) if kwp > 0 else None
        fora = kit is None or (com_bateria and kwp > P["bateriaEmTabelaAte"])
        capex = None if fora else kit + obra_kwp * kwp + (blocos * P["bateriaBlocoBrl"] if com_bateria else 0.0)
        opcoes.append({
            "id": oid, "bateria_kwh": blocos * P["bateriaBlocoKwh"] if com_bateria else 0.0,
            "kit_brl": kit, "investimento_brl": capex,
            "parcela_brl": parcela(P, capex) if capex else None,
            "payback_anos": payback(capex) if capex else None,
        })
    return {
        "consumo_kwh": consumo, "tarifa": tarifa, "ligacao": ligacao, "disponibilidade_kwh": disp,
        "produtividade": prod, "kwp": kwp, "modulos": modulos, "geracao_kwh": geracao,
        "autoconsumo_kwh": autoconsumo, "injetado_kwh": injetado,
        "conta_hoje_brl": conta_hoje, "conta_solar_brl": conta_solar, "economia_mes_brl": economia_mes,
        "opcoes": opcoes, "economia_por_ano": economia_por_ano,
    }
