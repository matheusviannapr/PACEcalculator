"""
Testes da apresentação comercial.

O risco aqui não é o LaTeX quebrar — é o número certo ir para o slide errado.
Por isso os testes conferem o dicionário de :func:`dados_da_apresentacao`
contra o cenário do estudo antes de olhar para o ``dados.tex``.
"""
from __future__ import annotations

import io
import zipfile
from datetime import date

import pytest
from pathlib import Path

from aurum.bateria.apresentacao import (
    OpcoesComerciais,
    dados_da_apresentacao,
    dados_tex,
    escrever_apresentacao,
    parcela_price,
)
from aurum.bateria.documento import DadosCapa, escrever_dossie, zip_do_dossie
from aurum.proposal.render import encontrar_compilador

#: A pasta do esqueleto, para os testes que conferem o que o LaTeX desenha.
RAIZ_APRES = Path(__file__).resolve().parent.parent / "apresentacao"

from test_documento import estudo_com_telhado  # noqa: F401 — fixture compartilhada


# ----------------------------------------------------------------------------
# A prestação
# ----------------------------------------------------------------------------
def test_parcela_sem_juros_e_o_principal_dividido():
    assert parcela_price(12_000.0, 0.0, 12) == pytest.approx(1_000.0)


def test_parcela_price_fecha_com_a_formula():
    # 100 mil a 1% a.m. em 60 meses: prestação de R$ 2.224,44 (tabela Price).
    assert parcela_price(100_000.0, 0.01, 60) == pytest.approx(2_224.44, abs=0.01)


def test_carencia_capitaliza_os_juros():
    sem = parcela_price(100_000.0, 0.01, 60)
    com = parcela_price(100_000.0, 0.01, 60, carencia_meses=3)
    assert com == pytest.approx(sem * 1.01 ** 3)


# ----------------------------------------------------------------------------
# Os dados vêm do cenário certo, e o fluxo fecha
# ----------------------------------------------------------------------------
def test_dados_saem_do_cenario_que_o_estudo_recomenda(estudo_com_telhado):
    estudo = estudo_com_telhado
    dados = dados_da_apresentacao(estudo, opcoes=OpcoesComerciais(data=date(2026, 9, 11)))
    chave = "solar+bateria" if estudo.recomendado is not None else "solar"
    cenario = estudo.cenarios.por_chave(chave)
    assert dados["cenario"] == chave
    assert dados["economia"]["capex_brl"] == pytest.approx(cenario.capex_brl)
    assert dados["economia"]["economia_anual_brl"] == pytest.approx(cenario.economia_anual_brl)
    assert dados["sistema"]["geracao_anual_kwh"] == pytest.approx(cenario.geracao_fv_kwh_ano)
    assert dados["sistema"]["modulos"] == estudo.configuracao.memoria.modulos_do_sistema
    # O módulo é o da memória, não o do empacotamento: quantidade, potência e
    # Wp têm que fechar a mesma conta na tabela do slide.
    memoria = estudo.configuracao.memoria
    assert dados["sistema"]["modulo_wp"] == memoria.modulo.potencia_wp
    assert dados["sistema"]["potencia_kwp"] == pytest.approx(
        memoria.modulos_do_sistema * memoria.modulo.potencia_wp / 1000.0)
    assert dados["sistema"]["area_ocupada_m2"] == pytest.approx(
        estudo.configuracao.layout.area_ocupada_m2)


def test_fluxo_acumulado_e_o_do_estudo_menos_o_que_a_proposta_acrescenta(estudo_com_telhado):
    estudo = estudo_com_telhado
    opcoes = OpcoesComerciais(seguro_brl_mes=40.0, gerenciamento_brl_mes=40.0,
                              carencia_financiamento_meses=3, prazo_financiamento_meses=60)
    dados = dados_da_apresentacao(estudo, opcoes=opcoes)
    cenario = estudo.cenarios.por_chave(dados["cenario"])
    fluxo = cenario.fluxo["fluxo_brl"].tolist()
    ano1 = dados["fluxo_acumulado"][0]
    assert ano1["ano"] == 1
    # À vista: o investimento sai no ano zero e o primeiro ano soma o fluxo
    # do estudo menos os R$ 80/mês de serviços.
    assert ano1["a_vista"] == pytest.approx(-cenario.capex_brl + fluxo[0] - 960.0)
    # Financiado: sem entrada, nove prestações no primeiro ano (três de carência).
    parcela = dados["economia"]["parcela_brl_mes"]
    assert ano1["financiado"] == pytest.approx(fluxo[0] - 960.0 - 9 * parcela)
    # Leasing: a mensalidade é a fração da economia, sem serviços à parte.
    mensalidade = dados["economia"]["mensalidade_leasing_brl_mes"]
    assert mensalidade == pytest.approx(cenario.economia_anual_brl / 12 * 0.86)
    assert ano1["leasing"] == pytest.approx(fluxo[0] - 9 * mensalidade)
    # O último ano mostrado é o horizonte do estudo, nunca além dele.
    assert dados["fluxo_acumulado"][-1]["ano"] == estudo.cenarios.premissas.anos_analise


def test_cliente_da_capa_vem_das_opcoes_e_cai_no_nome_do_estudo(estudo_com_telhado):
    estudo = estudo_com_telhado
    assert dados_da_apresentacao(estudo)["cliente"] == estudo.configuracao.nome
    dados = dados_da_apresentacao(estudo, opcoes=OpcoesComerciais(cliente="Condomínio Sol"))
    assert dados["cliente"] == "Condomínio Sol"
    tex = dados_tex(dados)
    assert "\\newcommand{\\cliente}{Condomínio Sol}" in tex
    # O esqueleto põe o cliente em destaque na capa, não só no "preparado para".
    capa = (RAIZ_APRES / "apresentacao.tex").read_text(encoding="utf-8")
    inicio = capa.index("1. Capa"); fim = capa.index("2. Quem somos")
    assert capa[inicio:fim].count(r"\cliente") >= 2


def test_cenario_inexistente_e_erro_claro(estudo_com_telhado):
    with pytest.raises(ValueError, match="não existe"):
        dados_da_apresentacao(estudo_com_telhado, opcoes=OpcoesComerciais(cenario="eolica"))


# ----------------------------------------------------------------------------
# dados.tex
# ----------------------------------------------------------------------------
def test_dados_tex_define_todos_os_comandos_do_esqueleto(estudo_com_telhado):
    tex = dados_tex(dados_da_apresentacao(
        estudo_com_telhado,
        capa=DadosCapa(responsavel="Fulano & Cia", crea="CREA 1", credenciais="Eng. 100%"),
    ))
    for comando in (
        "cliente", "dataproposta", "responsavel", "potenciaprojeto", "qtdmodulos",
        "potenciainversor", "areaocupada", "producaoanual", "bess", "investimento",
        "economiaanual", "payback", "tir", "mensalidadefinanciada", "mensalidadeleasing",
        "linhasdesembolso", "linhasfluxo", "fluxoanos", "fluxoavista", "fluxofinanciado",
        "fluxoservico", "fluxozero", "notafluxo", "linhasequipamentos", "fotosequipamentos",
        "invsembateria", "invcombateria", "economiaanualsembateria",
        "economiaanualcombateria", "paybacksembateria", "paybackcombateria",
        "autonomiacombateria", "custodabateria", "notasembateria", "notacombateria",
        "mensalidadesembateria", "mensalidadecombateria", "prazofinanciamentocurto",
        "tirsembateria",
    ):
        assert f"\\newcommand{{\\{comando}}}" in tex, comando
    # Texto vindo do operador passa pelo escape — o & e o % derrubariam o LaTeX.
    assert r"Fulano \& Cia" in tex
    assert r"Eng. 100\%" in tex
    # As logos passam a vir da pasta copiada, não do repositório.
    assert r"\renewcommand{\pastalogos}{estaticas/}" in tex


def test_os_dois_sistemas_saem_em_slides_distintos(estudo_com_telhado):
    """
    Sem bateria e com bateria são duas compras, e cada uma tem o seu slide.

    O que se fixa aqui é que os números de cada slide são os do seu cenário —
    e não os do outro, nem a soma dos dois.
    """
    estudo = estudo_com_telhado
    dados = dados_da_apresentacao(estudo)
    c = dados["comparativo"]
    assert c is not None, "o estudo precificou as duas topologias"
    micro, seco, split = c["sem_bateria"], c["split_sem_bateria"], c["com_bateria"]
    precos = estudo.precos_por_topologia
    # Cada slide traz o preço da sua coluna da tabela de kit.
    for bloco, chave in ((micro, "microinversor"), (seco, "splitphase"),
                         (split, "splitphase_bateria")):
        assert bloco["capex_brl"] == pytest.approx(precos[chave]["capex_brl"])
    # O banco custa dinheiro e compra autonomia — e a diferença é medida
    # contra o mesmo kit sem banco, não contra o microinversor.
    assert c["capex_da_bateria_brl"] == pytest.approx(
        split["capex_brl"] - seco["capex_brl"])
    assert split["autonomia_h"] > seco["autonomia_h"]
    assert micro["com_banco"] is False and seco["com_banco"] is False
    assert split["com_banco"] is True

    tex = dados_tex(dados)
    for marca in ("microtrue", "splitsecotrue", "splittrue"):
        assert f"\\{marca}" in tex, marca
    esqueleto = (RAIZ_APRES / "apresentacao.tex").read_text(encoding="utf-8")
    for marca in ("ifmicro", "ifsplitseco", "ifsplit"):
        assert f"\\{marca}" in esqueleto, marca


def test_o_comercial_escolhe_quais_topologias_a_proposta_leva(estudo_com_telhado):
    """Só o microinversor, só o split-phase, ou os dois — a proposta obedece."""
    estudo = estudo_com_telhado
    so_micro = dados_da_apresentacao(
        estudo, opcoes=OpcoesComerciais(topologias=["microinversor"]))
    assert set(so_micro["comparativo"]["topologias"]) == {"microinversor"}
    tex = dados_tex(so_micro)
    assert "\\microtrue" in tex and "\\splitfalse" in tex

    so_split = dados_da_apresentacao(
        estudo, opcoes=OpcoesComerciais(topologias=["splitphase_bateria"]))
    assert set(so_split["comparativo"]["topologias"]) == {"splitphase_bateria"}
    tex = dados_tex(so_split)
    assert "\\splittrue" in tex and "\\microfalse" in tex
    assert "\\splitsecofalse" in tex


def test_sem_topologia_precificada_os_slides_de_valor_nao_entram(estudo_com_telhado):
    """Nenhuma topologia pedida: os slides ficam de fora, em vez de sair vazios."""
    dados = dados_da_apresentacao(
        estudo_com_telhado, opcoes=OpcoesComerciais(topologias=[]))
    assert dados["comparativo"] is None
    assert "\\semslidesdevalor" in dados_tex(dados)


def test_apresentacao_sai_autocontida(estudo_com_telhado, tmp_path):
    escritos = escrever_apresentacao(estudo_com_telhado, tmp_path / "apresentacao", compilar=False)
    pasta = tmp_path / "apresentacao"
    assert escritos["apresentacao_dados"].exists()
    assert (pasta / "apresentacao.tex").exists()
    assert (pasta / "pace-apresentacao.sty").exists()
    assert (pasta / "estaticas" / "pace-logo-escuro.png").exists()
    assert (pasta / "estaticas" / "capa-foto.png").exists()


@pytest.mark.skipif(encontrar_compilador() is None, reason="sem LaTeX na máquina")
def test_apresentacao_compila(estudo_com_telhado, tmp_path):
    escritos = escrever_apresentacao(estudo_com_telhado, tmp_path / "apresentacao", compilar=True)
    assert "apresentacao_pdf" in escritos and escritos["apresentacao_pdf"].stat().st_size > 10_000


# ----------------------------------------------------------------------------
# Integração com o dossiê
# ----------------------------------------------------------------------------
def test_dossie_leva_a_apresentacao_junto(estudo_com_telhado, tmp_path):
    escritos = escrever_dossie(
        estudo_com_telhado, tmp_path / "estudo", compilar=False, comercial=OpcoesComerciais())
    assert escritos["apresentacao_dados"] == tmp_path / "estudo" / "apresentacao" / "dados.tex"
    # A foto do telhado do dossiê é a mesma da apresentação.
    assert (tmp_path / "estudo" / "apresentacao" / "figuras" / "foto_telhado.png").exists()


def test_zip_inclui_a_apresentacao(estudo_com_telhado):
    dados = zip_do_dossie(estudo_com_telhado, compilar=False, comercial=OpcoesComerciais())
    nomes = zipfile.ZipFile(io.BytesIO(dados)).namelist()
    assert "estudo/apresentacao/dados.tex" in nomes
    assert "estudo/apresentacao/apresentacao.tex" in nomes
    assert any(n.startswith("estudo/apresentacao/estaticas/") for n in nomes)


def test_sem_comercial_nada_muda(estudo_com_telhado, tmp_path):
    escritos = escrever_dossie(estudo_com_telhado, tmp_path / "estudo", compilar=False)
    assert "apresentacao_dados" not in escritos
    assert not (tmp_path / "estudo" / "apresentacao").exists()
