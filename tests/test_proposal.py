r"""
Camada de proposta: escape LaTeX, montagem do documento e escrita dos arquivos.

Vários testes aqui existem para travar bugs específicos da versão anterior,
que produzia arquivos .tex que não compilavam.
"""
from __future__ import annotations

import json
import re

import pytest

from aurum.proposal.context import ContextoProposta, DadosEmissor
from aurum.proposal.latex import (
    Raw,
    area,
    aviso,
    dinheiro,
    energia,
    esc,
    grafico_barras,
    grafico_linha,
    inteiro,
    link,
    lista,
    nota,
    numero,
    percentual,
    potencia,
    tabela,
    tabela_longa,
)
from aurum.proposal.render import escrever_proposta, slug, zip_em_memoria
from aurum.proposal.sections import montar_documento
from aurum.pv import consumption, financials
from aurum.pv.layout import calcular_layout
from aurum.pv.sizing import melhor_sistema


# ----------------------------------------------------------------------
# Escape
# ----------------------------------------------------------------------
@pytest.mark.parametrize("entrada,esperado", [
    ("&", r"\&"),
    ("%", r"\%"),
    ("$", r"\$"),
    ("#", r"\#"),
    ("_", r"\_"),
    ("{", r"\{"),
    ("}", r"\}"),
    ("~", r"\textasciitilde{}"),
    ("^", r"\textasciicircum{}"),
    ("\\", r"\textbackslash{}"),
])
def test_escape_usa_barra_simples(entrada, esperado):
    """
    A versão anterior declarava as substituições como r"\\&", que em Python é
    a string de dois caracteres \\ seguida de &  --  uma quebra de linha do
    LaTeX colada num E comercial. Todos os dez caracteres estavam assim.
    """
    resultado = esc(entrada)
    assert resultado == esperado
    assert "\\\\" not in resultado


def test_escape_de_barra_nao_reescapa_as_proprias_chaves():
    """
    Substituir a barra produz \textbackslash{}, que contém chaves. Uma segunda
    passada as escaparia de novo. A varredura por regex visita cada caractere
    do original uma única vez.
    """
    assert esc("C:\\temp") == r"C:\textbackslash{}temp"
    assert r"\{\}" not in esc("a\\b")


def test_escape_de_texto_comercial_completo():
    resultado = esc("Padaria Pão & Cia 100% #1 R$_50 {top}")
    assert resultado == r"Padaria Pão \& Cia 100\% \#1 R\$\_50 \{top\}"


def test_raw_atravessa_intacto():
    assert esc(Raw(r"\textbf{ok}")) == r"\textbf{ok}"


def test_nulos_e_booleanos():
    assert esc(None) == "---"
    assert esc(True) == "sim"
    assert esc(False) == "não"


def test_inteiros_nao_ganham_separador_de_milhar():
    """Anos e identificadores do OSM não levam ponto: 2026, não 2.026."""
    assert esc(2026) == "2026"
    assert esc(447998569) == "447998569"
    assert str(inteiro(2304)) == "2.304"


def test_float_sai_no_padrao_brasileiro():
    assert esc(1234.5) == "1.234,50"


# ----------------------------------------------------------------------
# Formatadores — o bug do escape duplo
# ----------------------------------------------------------------------
@pytest.mark.parametrize("formatador,valor", [
    (dinheiro, 1234567.89), (numero, 3.14159), (percentual, 12.34),
    (area, 5432.1), (potencia, 776.25), (energia, 1002160.0), (inteiro, 2304),
])
def test_formatador_e_idempotente_dentro_de_tabela(formatador, valor):
    """
    Antes, células eram montadas como f"R\\$ {v:,.2f}" e depois passavam pelo
    escapador, que virava a barra em \textbackslash{}. Toda tabela de valores
    saía corrompida. O tipo Raw impede isso.
    """
    formatado = formatador(valor)
    assert esc(formatado) == str(formatado)
    assert r"\textbackslash" not in esc(formatado)


def test_moeda_no_padrao_brasileiro():
    assert str(dinheiro(1234.5)) == r"R\$ 1.234,50"
    assert str(dinheiro(None)) == "---"


def test_percentual_escapa_o_simbolo():
    assert str(percentual(12.34)) == r"12,3\%"
    assert str(percentual(0.1234, fracao=True)) == r"12,3\%"


def test_link_nao_escapa_a_url():
    r"""\url e \href lidam com os caracteres especiais; escapá-los quebra o endereço."""
    resultado = str(link("https://a.com/x?y=1&z=2", "site"))
    assert resultado == r"\href{https://a.com/x?y=1&z=2}{site}"
    assert r"\&" not in resultado


# ----------------------------------------------------------------------
# Estruturas
# ----------------------------------------------------------------------
def test_tabela_mantem_valores_formatados_e_escapa_o_texto():
    saida = tabela(
        ["Item", "Valor"],
        [["Empresa & Cia", dinheiro(1994661.32)], ["Meta", percentual(45.3)]],
    )
    assert r"R\$ 1.994.661,32" in saida
    assert r"Empresa \& Cia" in saida
    assert r"\textbackslash" not in saida
    assert saida.count(r"\\") >= 3


def test_tabela_completa_celulas_faltantes():
    """Linha curta geraria 'Extra alignment tab' e abortaria a compilação."""
    saida = tabela(["A", "B", "C"], [["um"], ["um", "dois", "tres"]])
    for linha in saida.splitlines():
        if linha.endswith(r"\\") and "toprule" not in linha:
            assert linha.count("&") == 2


def test_longtable_repete_o_cabecalho():
    saida = tabela_longa(["Ano", "Valor"], [[2026, dinheiro(100.0)]])
    assert r"\endfirsthead" in saida and r"\endhead" in saida
    assert r"\begin{longtable}" in saida and r"\end{longtable}" in saida


def test_grafico_de_linha_nao_tem_fstring_esquecida():
    """
    A versão anterior escrevia a linha do zero numa string comum que parecia
    f-string, e o texto {len(sym_coords)} ia literalmente para o .tex,
    impedindo a compilação.
    """
    saida = grafico_linha(
        [str(a) for a in range(2026, 2031)],
        [-1_230_949, -519_790, 142_195, 762_185, 1_350_353],
        "VPL acumulado", r"R\$",
    )
    assert "{len(" not in saida
    assert "sym_coords" not in saida
    assert "(2026,0) (2030,0)" in saida


def test_grafico_omite_a_linha_do_zero_quando_nao_ha_troca_de_sinal():
    saida = grafico_linha(["a", "b"], [10.0, 20.0], "t", "y")
    assert "dashed" not in saida


def test_grafico_de_barras_sanitiza_rotulos():
    """Vírgula num rótulo quebra symbolic x coords."""
    saida = grafico_barras(["Jan/24, pico", "Fev"], [10.0, 20.0], "t", "kWh")
    simbolicos = re.search(r"symbolic x coords=\{([^}]*)\}", saida).group(1)
    assert simbolicos.count(",") == 1


def test_grafico_com_dados_incoerentes_devolve_vazio():
    assert grafico_barras(["a", "b"], [1.0], "t", "y") == ""
    assert grafico_linha([], [], "t", "y") == ""


def test_caixas_e_listas_escapam_o_conteudo():
    assert r"5\%" in aviso("Erro de 5% & risco")
    assert r"\&" in nota("A & B")
    assert lista([]) == ""
    assert r"\item" in lista(["um", "dois"])


# ----------------------------------------------------------------------
# Documento completo
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def contexto(base_equipamentos_modulo, perfil_curitiba_modulo) -> ContextoProposta:
    """
    Contexto completo de proposta, montado sem tocar a rede.

    Escopo de módulo porque o empacotamento de módulos custa segundos e
    nenhum teste desta suíte modifica o contexto.
    """
    from tests.conftest import construir_contexto

    return construir_contexto(base_equipamentos_modulo, perfil_curitiba_modulo)


def test_documento_tem_estrutura_valida(contexto):
    documento = montar_documento(contexto)
    assert documento.startswith(r"\documentclass")
    assert documento.rstrip().endswith(r"\end{document}")
    assert documento.count(r"\begin{document}") == 1
    assert documento.count(r"\end{document}") == 1


def test_ambientes_do_documento_estao_balanceados(contexto):
    documento = montar_documento(contexto)
    for ambiente in ("table", "tabular", "longtable", "figure", "itemize",
                     "enumerate", "titlepage", "axis", "tikzpicture", "center"):
        abre = len(re.findall(rf"\\begin\{{{ambiente}\}}", documento))
        fecha = len(re.findall(rf"\\end\{{{ambiente}\}}", documento))
        assert abre == fecha, f"{ambiente}: {abre} aberturas para {fecha} fechamentos"


def test_documento_nao_carrega_unicode_que_o_pdflatex_recusa(contexto):
    """
    ₂, ² e ° não passam pelo inputenc utf8 do pdflatex e abortam a compilação
    com 'Unicode character not set up for use with LaTeX'.
    """
    documento = montar_documento(contexto)
    for simbolo in ("\u2082", "\u00b2", "\u00b0"):
        assert simbolo not in documento, f"símbolo {simbolo!r} precisa virar macro"


def test_documento_nao_tem_resto_de_escape_quebrado(contexto):
    documento = montar_documento(contexto)
    assert r"\textbackslash{}$" not in documento
    assert "{len(" not in documento


def test_dados_do_cliente_aparecem_escapados(contexto):
    documento = montar_documento(contexto)
    assert r"Indústria Ômega \& Filhos Ltda" in documento
    assert r"Rua 100\% Nova" in documento
    # a URL fica dentro de \href e não é escapada
    assert "https://omega.ind.br?ref=a&b=1" in documento


def test_documento_cobre_todas_as_secoes(contexto):
    documento = montar_documento(contexto)
    for titulo in ("Resumo executivo", "Identificação do prospect",
                   "Recurso solar", "Perfil de consumo", "Sistema fotovoltaico proposto",
                   "Análise econômico", "Impacto ambiental", "Limitações e ressalvas",
                   "Próximos passos"):
        assert titulo in documento, f"seção ausente: {titulo}"


def test_ressalvas_declaram_a_origem_dos_dados(contexto):
    avisos = contexto.avisos_consolidados()
    texto = " ".join(avisos)
    assert "OpenStreetMap" in texto
    assert "estimado" in texto or "estimativa" in texto
    assert len(avisos) == len(set(avisos)), "avisos não devem repetir"


def test_resumo_traz_os_campos_comerciais(contexto):
    resumo = contexto.resumo()
    for campo in ("referencia", "cliente", "telefone", "site", "potencia_kwp",
                  "capex_brl", "payback_anos", "maps_url"):
        assert campo in resumo
    assert resumo["cliente"] == "Indústria Ômega & Filhos Ltda"


def test_contexto_serializa_para_json(contexto):
    texto = json.dumps(contexto.as_dict(), ensure_ascii=False, default=str)
    recuperado = json.loads(texto)
    assert recuperado["sistema"]["potencia_cc_kwp"] > 0
    assert len(recuperado["fluxo_caixa"]) == contexto.economia.premissas.anos


# ----------------------------------------------------------------------
# Escrita em disco
# ----------------------------------------------------------------------
@pytest.mark.parametrize("entrada,esperado", [
    ("Indústria Ômega & Cia", "industria-omega-cia"),
    ("AUR-W447998569", "aur-w447998569"),
    ("///", "sem-nome"),
])
def test_slug_gera_nome_de_arquivo_seguro(entrada, esperado):
    assert slug(entrada) == esperado


def test_escrever_proposta_produz_tex_e_json(contexto, tmp_path):
    gerados = escrever_proposta(contexto, tmp_path, compilar=False)
    assert gerados["tex"].exists() and gerados["json"].exists()
    assert gerados["tex"].read_text(encoding="utf-8").startswith(r"\documentclass")
    dados = json.loads(gerados["json"].read_text(encoding="utf-8"))
    assert dados["referencia"] == contexto.referencia
    # sem compilar, não deve sobrar auxiliar do LaTeX
    assert not list(tmp_path.glob("*.aux"))


def test_zip_em_memoria_traz_tudo(contexto):
    import io
    import zipfile

    conteudo = zip_em_memoria([contexto])
    with zipfile.ZipFile(io.BytesIO(conteudo)) as pacote:
        nomes = pacote.namelist()
        assert any(n.endswith(".tex") for n in nomes)
        assert any(n.endswith(".json") for n in nomes)
        assert "LEIA-ME.txt" in nomes


# ----------------------------------------------------------------------------
# Caracteres que o pdflatex não desenha
# ----------------------------------------------------------------------------
def test_letra_grega_e_transliterada_em_vez_de_derrubar_o_pdf():
    """
    Um `η` perdido numa observação de catálogo já derrubou a compilação.

    O `pdflatex` com `inputenc utf8` estoura com "Unicode character not set up
    for use with LaTeX" — sem PDF nenhum, e a mensagem não diz de onde veio.
    Aconteceu duas vezes: primeiro com `β` na memória de cálculo, depois com o
    `ηRT` da ressalva das baterias Lynx. A tradução vive no `esc` porque é a
    fronteira por onde todo dado externo passa.
    """
    from aurum.proposal.latex import esc

    assert esc("ciclos e ηRT a conferir") == "ciclos e etaRT a conferir"
    assert esc("β = -0,29 %/°C") == r"beta = -0,29 \%/°C"
    assert esc("5 × 3 ≤ 20") == "5 x 3 <= 20"


def test_acento_portugues_atravessa_intacto():
    """
    A transliteração não pode comer o português.

    Acentos latinos o `inputenc utf8` desenha sem reclamar; trocá-los por
    versões sem acento estragaria todo texto do documento para resolver um
    problema que eles não têm.
    """
    from aurum.proposal.latex import esc

    frase = "Acentuação: coração, ação, ampère — travessão, º e ª"
    assert esc(frase) == frase


def test_transliteracao_acontece_antes_do_escape():
    """
    Ordem importa: traduzir depois de escapar produziria `\\textbackslash`
    dentro do símbolo traduzido, que é o defeito de escape duplo já conhecido.
    """
    from aurum.proposal.latex import esc

    assert "textbackslash" not in esc("η & β")
    assert esc("η & β") == r"eta \& beta"
