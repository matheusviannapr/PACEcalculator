"""
Testes do dossiê em LaTeX.

O documento é o produto que sai da empresa, e ele falha de dois jeitos que
nenhum teste de cálculo pega:

* **Escape duplo.** Todo lugar que monta LaTeX passa o conteúdo por ``esc``.
  Um formatador que já devolve ``\\%`` faz o escape acontecer duas vezes e
  imprime ``\\textbackslash\\%`` no PDF — sem erro de compilação, só um
  documento feio entregue ao cliente.
* **Seção que some.** Quando o estudo não acha solução, ou quando não há
  telhado marcado, as seções correspondentes mudam de conteúdo. É fácil um
  ``None`` derrubar a geração inteira justamente no caso ruim, que é quando o
  relatório mais importa.
"""
from __future__ import annotations

import io
import math
import zipfile

import pytest

from aurum.bateria.apagao import MalhaApagao
from aurum.bateria.documento import DadosCapa, escrever_dossie, montar_documento, zip_do_dossie
from aurum.bateria.economia import PremissasBateria
from aurum.bateria.estudo import ConfiguracaoEstudo, executar_estudo
from aurum.bateria.fontes import Fatura, Gerador
from aurum.demanda.cenario import Cenario
from aurum.pv.equipment import carregar_base
from aurum.pv.telhado import dimensionar_no_telhado, telhado_de_geojson

LAT, LON = -25.4284, -49.2733


def _retangulo(largura_m: float, altura_m: float) -> dict:
    dlon = largura_m / (111_320 * abs(math.cos(math.radians(LAT))))
    dlat = altura_m / 110_540
    return {
        "type": "Polygon",
        "coordinates": [[
            [LON, LAT], [LON + dlon, LAT], [LON + dlon, LAT + dlat],
            [LON, LAT + dlat], [LON, LAT],
        ]],
    }


def _estudo(com_telhado: bool = True, autonomia: float = 3.0, com_analise: bool = True):
    """
    Monta um estudo pequeno, mas **completo**.

    A análise da demanda e a memória de cálculo entram de propósito: sem elas
    as seções correspondentes do documento não são geradas, e um defeito nelas
    passa despercebido. Foi assim que um `%` sem escape num parágrafo da
    análise comentou meia frase no PDF entregue.
    """
    from aurum.demanda import simular_ensemble
    from aurum.demanda.analise import analisar
    from aurum.pv.memoria import memoria_do_arranjo

    cenario = Cenario.de_segmento("escritorio", "Escritório Teste")
    base = carregar_base()
    telhado = layout = memoria = None
    if com_telhado:
        telhado = telhado_de_geojson(_retangulo(30, 20), "Escritório Teste")
        layout = dimensionar_no_telhado(telhado, base)
        memoria = memoria_do_arranjo(
            base.modulo_por_modelo("LR7-72HGD-620M"),
            base.inversor_por_modelo("GW50KN-MT"), layout.quantidade)

    analise = None
    if com_analise:
        ensemble = simular_ensemble(
            cenario.para_comodos(), cenario.instancias_de(), num_simulacoes=30)
        analise = analisar(
            ensemble, cenario.para_comodos(), cenario.instancias_de(),
            potencia_instalada_w=cenario.potencia_instalada_w(),
            num_simulacoes_detalhe=30)

    cfg = ConfiguracaoEstudo(
        latitude=LAT, longitude=LON, nome="Escritório Teste",
        comodos=cenario.para_comodos(), instancias_por_comodo=cenario.instancias_de(),
        comodos_essenciais=cenario.essenciais, simulacoes=30,
        telhado=telhado, layout=layout, memoria=memoria, analise=analise,
        potencia_fv_kwp=None if com_telhado else 25.0,
        max_candidatos=2, autonomia_alvo_h=autonomia,
        malha=MalhaApagao(duracoes_h=(1.0, autonomia), horas_inicio=(0, 12, 18), amostras=20),
        premissas=PremissasBateria(custo_interrupcao_brl_kwh=30.0),
        fatura=Fatura(tarifa_brl_kwh=0.96, custo_disponibilidade_kwh=100.0),
        gerador=Gerador(potencia_kw=30.0, custo_energia_brl_kwh=2.3, capex_brl=45_000.0),
    )
    return executar_estudo(cfg)


@pytest.fixture(scope="module")
def estudo_com_telhado():
    return _estudo(com_telhado=True)


@pytest.fixture(scope="module")
def estudo_sem_telhado():
    return _estudo(com_telhado=False)


# ----------------------------------------------------------------------------
# Estrutura do documento
# ----------------------------------------------------------------------------
def test_documento_e_latex_completo(estudo_com_telhado):
    tex = montar_documento(estudo_com_telhado)
    assert tex.lstrip().startswith("\\documentclass")
    assert tex.rstrip().endswith("\\end{document}")
    assert tex.count("\\begin{document}") == 1
    assert "\\tableofcontents" in tex


def test_documento_tem_todas_as_secoes(estudo_com_telhado):
    tex = montar_documento(estudo_com_telhado)
    for titulo in (
        "O que se conclui", "O telhado", "O recurso solar", "A demanda elétrica",
        "O sistema fotovoltaico", "O armazenamento",
        "Cenários: com e sem cada fonte",
        "Premissas, limitações e procedência dos dados",
    ):
        assert f"\\section{{{titulo}}}" in tex, f"faltou a seção '{titulo}'"
    # As subseções da análise completa e a memória de cálculo: sem elas o
    # documento compila e deixa de entregar o que foi pedido.
    for titulo in (
        "Distribuição dos picos", "O que causa o pico",
        "Fatores de carga, demanda e coincidência", "Curva de duração de carga",
        "Comportamento sazonal", "Memória de cálculo do arranjo",
    ):
        assert f"\\subsection{{{titulo}}}" in tex, f"faltou a subseção '{titulo}'"


def test_secao_de_cenarios_mostra_todos_os_arranjos(estudo_com_telhado):
    """
    O quadro precisa trazer o referencial e cada combinação de fontes.

    Sem a linha "só a rede" não existe economia — existe apenas um custo — e
    sem as combinações intermediárias o documento pressupõe a resposta que
    deveria demonstrar.
    """
    tex = montar_documento(estudo_com_telhado)
    inicio = tex.index("\\section{Cenários: com e sem cada fonte}")
    fim = tex.index("\\section{Premissas, limitações e procedência dos dados}")
    secao = tex[inicio:fim]
    for arranjo in ("Somente a rede", "Solar", "Bateria", "Gerador",
                    "Solar + bateria + gerador"):
        assert arranjo in secao, f"faltou o arranjo '{arranjo}' no quadro"
    assert "anti-ilhamento" in secao, "a seção precisa dizer que solar não faz backup"


def test_cenarios_nao_quebram_o_latex(estudo_com_telhado):
    """Cifrão de tarifa em prosa é o jeito mais fácil de abrir modo matemático."""
    tex = montar_documento(estudo_com_telhado)
    inicio = tex.index("\\section{Cenários: com e sem cada fonte}")
    fim = tex.index("\\section{Premissas, limitações e procedência dos dados}")
    secao = tex[inicio:fim]
    # Todo R$ do texto tem que estar escapado; um `$` solto abre matemática.
    assert "R$ " not in secao.replace("R\\$ ", "")
    assert secao.count("$") == secao.count("\\$")


def test_tarifa_por_kwh_nao_e_arredondada_para_inteiro(estudo_com_telhado):
    """R$ 0,96/kWh impresso como "R$ 1/kWh" é número errado com cara de tabela."""
    tex = montar_documento(estudo_com_telhado)
    assert "R\\$ 0,96/kWh" in tex


def test_nada_de_escape_duplo(estudo_com_telhado):
    """`\\textbackslash` no documento é sempre sinal de conteúdo escapado duas vezes."""
    tex = montar_documento(estudo_com_telhado)
    assert "textbackslash" not in tex


def test_porcentagem_sai_escapada_uma_vez_so(estudo_com_telhado):
    tex = montar_documento(estudo_com_telhado)
    assert "\\%" in tex, "os percentuais precisam do escape do LaTeX"

    # Um `%` sem barra invertida comenta o resto da linha. Duas formas são
    # legítimas e precisam passar: a linha que começa com `%` (comentário
    # inteiro) e o `%` no fim da linha, que o LaTeX usa para emendar linhas sem
    # inserir espaço — é assim que `fcolorbox` monta as caixas de aviso.
    corpo = tex.split(r"\begin{document}", 1)[1]
    for linha in corpo.splitlines():
        if linha.lstrip().startswith("%"):
            continue
        for i, caractere in enumerate(linha):
            if caractere != "%":
                continue
            escapado = i > 0 and linha[i - 1] == "\\"
            fim_de_linha = i == len(linha) - 1
            assert escapado or fim_de_linha, f"percentual solto em: {linha}"


def test_capa_traz_os_dados_do_emissor(estudo_com_telhado):
    capa = DadosCapa(empresa="PACE Inteligência Energética", responsavel="Fulano & Cia",
                     crea="PR-1234/D", referencia="Ref 1")
    tex = montar_documento(estudo_com_telhado, capa=capa)
    assert "PACE Inteligência Energética" in tex
    assert "Fulano \\& Cia" in tex, "o nome do responsável tem que ser escapado"
    assert "PR-1234/D" in tex


# ----------------------------------------------------------------------------
# Conteúdo condicional
# ----------------------------------------------------------------------------
def test_secao_do_telhado_traz_area_e_modulos(estudo_com_telhado):
    tex = montar_documento(estudo_com_telhado)
    layout = estudo_com_telhado.configuracao.layout
    assert "Área em planta" in tex
    assert "Superfície real do telhado" in tex
    assert str(layout.quantidade) in tex


def test_sem_telhado_o_documento_diz_que_nao_verificou(estudo_sem_telhado):
    """
    A ressalva que impede o estudo de prometer o que não verificou.

    Sem telhado marcado, "O telhado" e "O sistema fotovoltaico" viraram uma
    seção só: as duas diziam variações de "não se aplica", em oito e quatro
    linhas, e duas entradas no sumário para isso interrompem a leitura entre a
    demanda e o armazenamento. O que não pode sumir é a ressalva — e é ela que
    este teste guarda, e não o título.
    """
    tex = montar_documento(estudo_sem_telhado)
    assert "\\section{O telhado e o sistema fotovoltaico}" in tex
    assert "\\section{O telhado}" not in tex, "sem telhado, as duas viram uma"
    assert "não afirma que o sistema cabe" in tex


def test_secao_de_procedencia_separa_datasheet_de_estimativa(estudo_com_telhado):
    tex = montar_documento(estudo_com_telhado)
    assert "Procedência dos dados" in tex or "procedência" in tex.lower()
    assert "O que este estudo não afirma" in tex
    assert "projeto elétrico executivo" in tex


def test_estudo_sem_solucao_ainda_gera_documento():
    """O caso ruim é quando o relatório mais importa — não pode ser o que quebra."""
    estudo = _estudo(com_telhado=True, autonomia=36.0)
    tex = montar_documento(estudo)
    assert "\\end{document}" in tex
    if estudo.recomendado is None:
        assert "Nenhuma combinação" in tex


# ----------------------------------------------------------------------------
# Pacote
# ----------------------------------------------------------------------------
def test_dossie_grava_a_arvore_esperada(estudo_com_telhado, tmp_path):
    escritos = escrever_dossie(estudo_com_telhado, tmp_path / "estudo", compilar=False)
    pasta = tmp_path / "estudo"
    assert (pasta / "estudo.tex").exists()
    assert (pasta / "resumo.md").exists()
    assert list((pasta / "figuras").glob("*.png")), "nenhuma figura gravada"
    assert list((pasta / "tabelas").glob("*.csv")), "nenhuma tabela gravada"
    # O .tex referencia as figuras pelo caminho relativo que a pasta usa.
    tex = (pasta / "estudo.tex").read_text(encoding="utf-8")
    for figura in (pasta / "figuras").glob("*.png"):
        if figura.name in tex:
            assert f"figuras/{figura.name}" in tex
    assert escritos["tex"].exists()


def test_zip_traz_tudo_e_o_leia_me(estudo_com_telhado):
    dados = zip_do_dossie(estudo_com_telhado, compilar=False)
    assert dados[:2] == b"PK"
    with zipfile.ZipFile(io.BytesIO(dados)) as zf:
        nomes = zf.namelist()
        leia_me = zf.read("LEIA-ME.txt").decode("utf-8")
    assert any(n.endswith("estudo/estudo.tex") for n in nomes)
    assert any("figuras/" in n for n in nomes)
    assert any("tabelas/" in n for n in nomes)
    assert "Overleaf" in leia_me
    # Auxiliares de compilação não têm por que viajar no pacote.
    assert not any(n.endswith((".aux", ".log", ".out", ".toc")) for n in nomes)


@pytest.mark.rede
def test_documento_compila_em_pdf(estudo_com_telhado, tmp_path):
    """Garante que o `.tex` entregue de fato compila, e não só parece LaTeX."""
    from aurum.proposal.render import encontrar_compilador

    if not encontrar_compilador():
        pytest.skip("nenhuma distribuição LaTeX nesta máquina")
    escritos = escrever_dossie(estudo_com_telhado, tmp_path / "estudo", compilar=True)
    assert "pdf" in escritos, "o documento não compilou"
    assert escritos["pdf"].stat().st_size > 50_000


# ----------------------------------------------------------------------------
# Classificação da procedência
# ----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "fonte, esperado",
    [
        ("a conferir", "Ordem de grandeza de mercado"),
        ("", "Ordem de grandeza de mercado"),
        ("GoodWe ET 15-30kW — en.goodwe.com/.../GW_ET.pdf", "Datasheet do fabricante"),
        ("DAH Solar DHN-72X16/FS(BW) (datasheet do fabricante)", "Datasheet do fabricante"),
        # A ressalva das baterias GoodWe contém a palavra "conferir" e a linha
        # continua sendo datasheet: por isso o terceiro estado existe.
        ("GoodWe Lynx F G2 — .../GW_Lynx-F-G2_Datasheet-EN.pdf | energia utilizável a "
         "100% DOD; ciclos e ηRT a conferir", "Datasheet, com ressalva"),
    ],
)
def test_natureza_do_dado_tem_tres_estados(fonte, esperado):
    from aurum.bateria.documento import _natureza

    assert _natureza(fonte) == esperado


def test_baterias_goodwe_nao_viram_ordem_de_grandeza(estudo_com_telhado):
    """Regressão: a ressalva do datasheet não pode rebaixar a linha inteira."""
    from aurum.bateria.catalogo import carregar_catalogo
    from aurum.bateria.documento import _natureza

    catalogo = carregar_catalogo()
    lynx = [b for b in catalogo.baterias if b.fabricante == "GoodWe"]
    assert lynx
    assert all(_natureza(b.fonte_dado).startswith("Datasheet") for b in lynx)


# ----------------------------------------------------------------------------
# A análise completa e a memória, dentro do documento
# ----------------------------------------------------------------------------
def test_analise_completa_entra_no_documento(estudo_com_telhado):
    tex = montar_documento(estudo_com_telhado)
    # Estatística, fatores e composição do pico.
    assert "Coeficiente de variação" in tex
    assert "Incerteza do P95 pelo Monte Carlo" in tex
    assert "Fator de coincidência" in tex
    assert "Fatia do pico" in tex
    # E nada de markdown vazando: `**` é sintaxe de outro formato.
    corpo = tex.split(r"\begin{document}", 1)[1]
    assert "**" not in corpo, "marcação Markdown vazou para o LaTeX"


def test_memoria_de_calculo_mostra_a_conta(estudo_com_telhado):
    tex = montar_documento(estudo_com_telhado)
    memoria = estudo_com_telhado.configuracao.memoria
    assert "Memória de cálculo do arranjo" in tex
    # A fórmula, a substituição e o resultado de cada passo.
    assert "Voc(T)" in tex or "Voc(T" in tex
    assert "Tensão máxima CC" in tex
    assert str(memoria.modulos_por_string) in tex
    assert "Razão CC/CA por inversor" in tex
    # Símbolos que o pdflatex não desenha em modo texto não podem sobreviver.
    for simbolo in ("β", "⌊", "⌋", "⌈", "⌉", "≤", "≥"):
        assert simbolo not in tex, f"'{simbolo}' quebra a compilação do PDF"


def test_sem_analise_o_documento_ainda_sai():
    """A análise é opcional; a ausência dela não pode derrubar o documento."""
    estudo = _estudo(com_telhado=False, com_analise=False)
    tex = montar_documento(estudo)
    assert tex.rstrip().endswith("\\end{document}")
    assert "Distribuição dos picos" not in tex
