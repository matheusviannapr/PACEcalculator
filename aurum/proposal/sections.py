r"""
Seções do documento da proposta.

Cada função devolve um trecho de LaTeX pronto. O preâmbulo é único e
compartilhado, para que a proposta individual e o dossiê em lote compilem com
o mesmo conjunto de pacotes.
"""
from __future__ import annotations

import logging
import os
from datetime import date, timedelta
from pathlib import Path

from ..pv.consumption import faixa_eui
from ..pv.solar import MESES
from .context import ContextoProposta
from .latex import (
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
    secao,
    tabela,
    tabela_longa,
)

LOGGER = logging.getLogger(__name__)

#: Símbolos que o pdflatex não aceita como Unicode cru e precisam de macro.
#: Ficam como constantes para não aparecerem como chaves literais dentro de
#: f-strings, onde `{}` seria lido como campo de substituição.
GRAU = r"\textdegree{}"
METRO_QUADRADO = r"m\textsuperscript{2}"
CO2 = r"CO\textsubscript{2}"

PREAMBULO = r"""\documentclass[11pt,a4paper]{article}

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[brazil]{babel}
\usepackage{lmodern}
\usepackage{textcomp}
\usepackage[margin=2.2cm,headheight=15pt]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{float}
\usepackage{xcolor}
\usepackage{fancyhdr}
\usepackage{titlesec}
\usepackage{caption}
\usepackage{pgfplots}
\usepackage{tikz}
\usepackage[hidelinks]{hyperref}
% URLs de datasheet passam de 60 caracteres e não quebram sozinhas:
% sem xurl elas atravessam a margem direita da página de procedência.
\usepackage{xurl}

\pgfplotsset{compat=1.18}

% Cores da PACE, medidas no arquivo da logo. Trocar aqui é trocar no
% documento inteiro; ver `aurum.marca`.
\definecolor{pretopace}{RGB}{22,22,21}
\definecolor{amarelopace}{RGB}{248,193,18}
\definecolor{grafitepace}{RGB}{36,36,34}
\definecolor{verdepace}{RGB}{46,125,82}
\definecolor{vermelhoaviso}{RGB}{180,69,31}
\definecolor{cinzaclaro}{RGB}{232,232,228}
% Os nomes antigos continuam válidos e apontam para as cores da marca: há
% documento e teste que ainda os citam, e quebrá-los não repinta nada.
\colorlet{azulaurum}{pretopace}
\colorlet{verdeaurum}{verdepace}

\hypersetup{colorlinks=true,linkcolor=pretopace,urlcolor=pretopace,citecolor=pretopace}

\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\footnotesize\textbf{\cabecalhoempresa}}
\fancyhead[R]{\footnotesize Proposta \cabecalhoreferencia}
\fancyfoot[C]{\footnotesize\thepage}
\renewcommand{\headrulewidth}{0.4pt}

\titleformat{\section}{\large\bfseries\color{pretopace}}{\thesection}{0.7em}{}
\titleformat{\subsection}{\normalsize\bfseries\color{pretopace!80!black}}{\thesubsection}{0.7em}{}

\captionsetup{font=small,labelfont={bf,color=pretopace}}
\setlength{\parskip}{0.4em}
\setlength{\parindent}{0pt}
"""


def carregar_preambulo() -> str:
    r"""
    Preâmbulo do documento, com identidade visual própria quando houver.

    Aponte ``AURUM_LATEX_PREAMBLE`` para um arquivo .tex e ele substitui o
    preâmbulo padrão inteiro -- é o ponto de entrada para uma identidade
    visual própria sem mexer no código.

    O arquivo precisa ir de ``\documentclass`` até logo antes de
    ``\begin{document}``, e definir as quatro cores usadas pelas seções
    (``pretopace``, ``amarelopace``, ``verdepace``, ``vermelhoaviso``, ``cinzaclaro``) além
    de carregar ``booktabs``, ``longtable``, ``float``, ``pgfplots``,
    ``hyperref``, ``xurl`` e ``textcomp``. Os comandos ``\cabecalhoempresa`` e
    ``\cabecalhoreferencia`` são injetados depois e podem ser usados no
    cabeçalho ou rodapé.
    """
    caminho = os.environ.get("AURUM_LATEX_PREAMBLE")
    if not caminho:
        return PREAMBULO
    arquivo = Path(caminho)
    if not arquivo.is_file():
        LOGGER.warning(
            "AURUM_LATEX_PREAMBLE aponta para %s, que não existe. Usando o preâmbulo padrão.",
            arquivo,
        )
        return PREAMBULO
    return arquivo.read_text(encoding="utf-8")


def preambulo(ctx: ContextoProposta) -> str:
    """Preâmbulo com os dados do cabeçalho já definidos como comandos."""
    return (
        carregar_preambulo()
        + f"\n\\newcommand{{\\cabecalhoempresa}}{{{esc(ctx.emissor.empresa)}}}\n"
        + f"\\newcommand{{\\cabecalhoreferencia}}{{{esc(ctx.referencia)}}}\n"
        + "\n\\begin{document}\n"
    )


def capa(ctx: ContextoProposta) -> str:
    """Página de rosto."""
    validade = ctx.data_emissao + timedelta(days=ctx.validade_dias)
    kwp = ctx.sistema.potencia_cc_kwp
    geracao = ctx.sistema.geracao_anual_kwh or 0.0

    linhas_emissor = [rf"\textbf{{{esc(ctx.emissor.empresa)}}}"]
    for rotulo, valor in (
        ("Responsável técnico", ctx.emissor.responsavel_tecnico),
        ("CREA", ctx.emissor.crea),
        ("Telefone", ctx.emissor.telefone),
        ("E-mail", ctx.emissor.email),
    ):
        if valor:
            linhas_emissor.append(rf"{esc(rotulo)}: {esc(valor)}")

    return "\n".join([
        r"\begin{titlepage}",
        r"\centering",
        r"\vspace*{1.5cm}",
        r"{\Huge\bfseries\color{pretopace} Proposta de Geração Solar\par}",
        r"\vspace{0.3cm}",
        r"{\Large Sistema Fotovoltaico Conectado à Rede\par}",
        r"\vspace{1.2cm}",
        r"\begin{tikzpicture}",
        r"\node[draw=pretopace,line width=1pt,fill=cinzaclaro,rounded corners=6pt,"
        r"inner sep=14pt,text width=0.78\linewidth,align=center] {",
        rf"  {{\Large\bfseries {esc(ctx.titulo_cliente)}}} \\[6pt]",
        rf"  {esc(ctx.endereco)} \\[10pt]",
        rf"  {{\LARGE\bfseries\color{{pretopace}} {potencia(kwp)}}} \\[4pt]",
        rf"  {energia(geracao)} por ano",
        r"};",
        r"\end{tikzpicture}",
        r"\vspace{1.2cm}",
        r"\begin{tabular}{@{}ll@{}}",
        rf"\textbf{{Referência:}} & {esc(ctx.referencia)} \\",
        rf"\textbf{{Data de emissão:}} & {esc(ctx.data_emissao.strftime('%d/%m/%Y'))} \\",
        rf"\textbf{{Validade:}} & {esc(validade.strftime('%d/%m/%Y'))} \\",
        rf"\textbf{{Coordenadas:}} & {numero(ctx.lead.centroid_lat, 5)}, {numero(ctx.lead.centroid_lon, 5)} \\",
        r"\end{tabular}",
        r"\vfill",
        r"\begin{minipage}{0.9\linewidth}\centering\small",
        r" \\ ".join(linhas_emissor),
        r"\end{minipage}",
        r"\vspace{0.8cm}",
        r"{\footnotesize Documento gerado automaticamente. Estimativa de pré-viabilidade "
        r"sujeita a confirmação em visita técnica.\par}",
        r"\end{titlepage}",
    ])


def sumario() -> str:
    return "\\tableofcontents\n\\newpage"


def resumo_executivo(ctx: ContextoProposta) -> str:
    """Os números que decidem, na primeira página de conteúdo."""
    eco = ctx.economia
    kwp = ctx.sistema.potencia_cc_kwp
    geracao = ctx.sistema.geracao_anual_kwh or 0.0
    cobertura = ctx.cobertura_consumo

    texto = (
        f"Este documento apresenta o estudo de pré-viabilidade para instalação de um sistema "
        f"fotovoltaico conectado à rede na edificação identificada como "
        f"\\textbf{{{esc(ctx.titulo_cliente)}}}, com área de cobertura de "
        f"{area(ctx.lead.metrics.area_m2)}.\n\n"
        f"O sistema proposto tem \\textbf{{{potencia(kwp)}}} de potência instalada, com "
        f"{inteiro(ctx.sistema.modulos_totais)} módulos, e geração estimada de "
        f"\\textbf{{{energia(geracao)}}} por ano."
    )
    if cobertura is not None:
        texto += (
            f" Isso corresponde a aproximadamente \\textbf{{{percentual(cobertura, 0, fracao=True)}}} "
            f"do consumo estimado da edificação."
        )

    linhas = [
        ["Potência instalada", potencia(kwp)],
        ["Módulos fotovoltaicos", inteiro(ctx.sistema.modulos_totais)],
        ["Geração estimada", Raw(f"{energia(geracao)} / ano")],
        ["Investimento estimado", dinheiro(eco.premissas.capex_brl)],
        ["Economia no primeiro ano", dinheiro(eco.economia_ano1_brl)],
        [
            "Retorno do investimento",
            Raw(f"{numero(eco.payback_simples_anos, 1)} anos") if eco.payback_simples_anos else "não atingido",
        ],
        ["Valor presente líquido (25 anos)", dinheiro(eco.vpl_brl)],
        [
            "Taxa interna de retorno",
            percentual(eco.tir_anual, 1, fracao=True) if eco.tir_anual is not None else "não aplicável",
        ],
        [
            "Custo nivelado da energia",
            Raw(f"{dinheiro(eco.lcoe_brl_kwh, 4)} / kWh") if eco.lcoe_brl_kwh else "---",
        ],
        ["Tarifa de referência adotada", Raw(f"{dinheiro(eco.premissas.tarifa_brl_kwh, 4)} / kWh")],
    ]

    return "\n\n".join([
        secao("Resumo executivo"),
        texto,
        tabela(["Indicador", "Valor"], linhas, largura_primeira_coluna="8.5cm"),
    ])


def identificacao(ctx: ContextoProposta) -> str:
    """Quem é o prospect e como falar com ele."""
    empresa = ctx.lead.company
    lead = ctx.lead

    linhas_cliente = [
        ["Razão social / nome", ctx.titulo_cliente],
        ["Segmento (OpenStreetMap)", lead.building_type],
        ["Endereço", ctx.endereco],
        ["Telefone", empresa.get("telefone") or "não identificado"],
        ["E-mail", empresa.get("email") or "não identificado"],
        ["Site", link(empresa.get("site")) if empresa.get("site") else "não identificado"],
        ["Horário de funcionamento", empresa.get("horario") or "não informado"],
        ["Origem dos dados cadastrais", empresa.get("fonte") or "não enriquecido"],
    ]
    if empresa.get("operador"):
        linhas_cliente.insert(1, ["Operador / rede", empresa["operador"]])
    if empresa.get("cnpj"):
        linhas_cliente.insert(1, ["CNPJ", empresa["cnpj"]])

    linhas_local = [
        ["Identificador OpenStreetMap", lead.lead_id],
        ["Latitude", numero(lead.centroid_lat, 6)],
        ["Longitude", numero(lead.centroid_lon, 6)],
        ["Área de cobertura (projeção)", area(lead.metrics.area_m2)],
        ["Pavimentos estimados", inteiro(lead.levels)],
        ["Dimensão maior", Raw(f"{numero(lead.metrics.max_length_m, 1)} m")],
        ["Dimensão menor", Raw(f"{numero(lead.metrics.min_width_m, 1)} m")],
        ["Ver no mapa", link(lead.maps_url, "abrir no Google Maps")],
        ["Ver no OpenStreetMap", link(lead.osm_url, "abrir no OSM")],
    ]

    return "\n\n".join([
        secao("Identificação do prospect"),
        secao("Dados cadastrais", 2),
        tabela(["Campo", "Valor"], linhas_cliente, largura_primeira_coluna="6cm", alinhamento="p{6cm}p{8cm}"),
        secao("Localização e cobertura", 2),
        tabela(["Campo", "Valor"], linhas_local, largura_primeira_coluna="6cm", alinhamento="p{6cm}p{8cm}"),
    ])


def recurso_solar(ctx: ContextoProposta) -> str:
    """Irradiação e produtividade da localidade."""
    perfil = ctx.perfil_solar
    kwp = ctx.sistema.potencia_cc_kwp
    mensal = perfil.mensal(kwp)

    linhas = [
        ["Fonte de dados de irradiação", perfil.fonte.upper()],
        ["Base climática", perfil.base_dados or "não informada"],
        ["Orientação dos módulos (azimute)", Raw(f"{numero(perfil.azimuth_deg, 0)}{GRAU} ({esc(perfil.orientacao)})")],
        ["Inclinação dos módulos", Raw(f"{numero(perfil.tilt_deg, 1)}{GRAU}")],
        ["Perdas totais do sistema", percentual(perfil.losses_percent, 1)],
        ["Produtividade específica", Raw(f"{numero(perfil.anual_kwh_por_kwp, 0)} kWh/kWp por ano")],
        ["Fator de capacidade", percentual(perfil.fator_capacidade(), 1, fracao=True)],
    ]
    if perfil.irradiacao_kwh_m2_ano:
        linhas.append(
            ["Irradiação no plano dos módulos", Raw(f"{numero(perfil.irradiacao_kwh_m2_ano, 0)} kWh/{METRO_QUADRADO} por ano")]
        )
    if perfil.elevacao_m is not None:
        linhas.append(["Altitude do local", Raw(f"{numero(perfil.elevacao_m, 0)} m")])

    partes = [
        secao("Recurso solar e geração estimada"),
        (
            "A orientação adotada é a de maior produção anual para a latitude do local. "
            "No hemisfério sul isso significa módulos voltados para o Norte geográfico."
        ),
        tabela(["Parâmetro", "Valor"], linhas, largura_primeira_coluna="8cm"),
        secao("Geração mês a mês", 2),
        grafico_barras(list(MESES), mensal, f"Geração mensal estimada do sistema de {kwp:.0f} kWp", "kWh"),
        tabela(
            ["Mês", "Geração (kWh)", "Média diária (kWh)"],
            [
                [mes, energia(valor), energia(valor / 30.0, 1)]
                for mes, valor in zip(MESES, mensal)
            ]
            + [[Raw(r"\textbf{Total anual}"), Raw(rf"\textbf{{{energia(sum(mensal))}}}"), ""]],
            legenda="Geração estimada mês a mês",
        ),
    ]
    if not perfil.confiavel:
        partes.append(aviso(perfil.aviso or "Dados de irradiação estimados, não medidos."))
    return "\n\n".join(p for p in partes if p)


def consumo_estimado(ctx: ContextoProposta) -> str:
    """Consumo de referência e a incerteza associada."""
    est = ctx.consumo
    minimo, tipico, maximo = faixa_eui(est.tipo_edificacao)

    linhas = [
        ["Origem do dado", "fatura de energia" if ctx.origem_consumo == "fatura" else "estimativa por segmento"],
        ["Segmento considerado", est.tipo_edificacao],
        ["Área construída estimada", area(est.area_construida_m2)],
        ["Pavimentos considerados", inteiro(est.pavimentos)],
        ["Intensidade adotada", Raw(f"{numero(est.eui_kwh_m2_mes, 1)} kWh/{METRO_QUADRADO} por mês")],
        ["Faixa do segmento", Raw(f"{numero(minimo, 1)} a {numero(maximo, 1)} kWh/{METRO_QUADRADO} por mês")],
        ["Consumo mensal estimado", energia(est.consumo_mensal_kwh)],
        [
            "Faixa plausível mensal",
            Raw(f"{energia(est.faixa_mensal_kwh[0])} a {energia(est.faixa_mensal_kwh[1])}"),
        ],
        ["Consumo anual estimado", energia(est.consumo_anual_kwh)],
        ["Demanda média em horário comercial", Raw(f"{numero(est.demanda_media_kw, 1)} kW")],
        ["Autoconsumo esperado", percentual(est.fracao_autoconsumo, 0, fracao=True)],
        ["Confiança da estimativa", est.confianca],
    ]

    partes = [
        secao("Perfil de consumo de referência"),
        (
            "Na fase de prospecção não há fatura disponível. O consumo abaixo foi estimado pela "
            "intensidade de uso de energia típica do segmento, multiplicada pela área construída. "
            "Serve para dimensionar a conversa comercial, não para fechar contrato."
        ),
        tabela(["Parâmetro", "Valor"], linhas, largura_primeira_coluna="8cm"),
    ]
    if ctx.origem_consumo == "estimado":
        partes.append(
            nota(
                "A faixa do segmento é larga porque duas edificações do mesmo tipo e tamanho podem "
                "consumir muito diferente conforme processo produtivo, climatização e horário de "
                "operação. O primeiro pedido na visita comercial deve ser a fatura dos últimos 12 meses."
            )
        )
    return "\n\n".join(partes)


def sistema_proposto(ctx: ContextoProposta) -> str:
    """Configuração técnica: módulos, inversores, strings e ocupação."""
    sistema = ctx.sistema
    layout = ctx.layout

    linhas_geral = [
        ["Potência do gerador (CC)", potencia(sistema.potencia_cc_kwp)],
        ["Potência dos inversores (CA)", potencia(sistema.potencia_ca_kw, unidade="kW")],
        ["Razão CC/CA", numero(sistema.razao_dc_ac, 2)],
        ["Quantidade de módulos", inteiro(sistema.modulos_totais)],
        ["Modelo do módulo", str(sistema.modulo)],
        ["Dimensões do módulo", Raw(
            f"{numero(sistema.modulo.comprimento_m, 3)} x {numero(sistema.modulo.largura_m, 3)} m"
        )],
        ["Eficiência do módulo", percentual(sistema.modulo.eficiencia_percent, 1)],
        ["Área ocupada por módulos", area(sistema.area_modulos_m2)],
        ["Geração anual estimada", energia(sistema.geracao_anual_kwh or 0)],
        [
            "Rendimento específico",
            Raw(f"{numero(sistema.rendimento_especifico, 0)} kWh/kWp") if sistema.rendimento_especifico else "---",
        ],
    ]

    linhas_inv = [
        [
            f"{a.quantidade}x {a.inversor.fabricante} {a.inversor.modelo}",
            potencia(a.inversor.potencia_ca_kw, 1, "kW"),
            inteiro(a.mppts_usados),
            inteiro(a.modulos_por_string),
            inteiro(a.strings_por_mppt),
            inteiro(a.strings_totais),
            inteiro(a.modulos_totais),
        ]
        for a in sistema.arranjos
    ]

    linhas_layout = [
        ["Método de cálculo", "empacotamento geométrico" if layout.metodo == "grade" else "estimativa por área"],
        ["Tipo de montagem", "coplanar ao telhado" if layout.montagem == "coplanar" else "estrutura inclinada"],
        ["Orientação do módulo", layout.orientacao_modulo],
        ["Direção das fileiras (azimute)", Raw(f"{numero(layout.azimute_fileiras_deg, 0)}{GRAU}")],
        ["Passo entre fileiras", Raw(f"{numero(layout.passo_fileira_m, 2)} m")],
        ["Área bruta do telhado", area(layout.area_bruta_m2)],
        ["Área útil após recuo de borda", area(layout.area_util_m2)],
        ["Posições geométricas encontradas", inteiro(layout.quantidade_geometrica)],
        ["Fator de obstáculos aplicado", percentual(layout.fator_obstaculos, 0, fracao=True)],
        ["Módulos considerados no layout", inteiro(layout.quantidade)],
        ["Densidade de potência", Raw(f"{numero(layout.densidade_wp_m2, 0)} Wp/{METRO_QUADRADO}")],
        ["Ocupação do telhado", percentual(layout.taxa_ocupacao, 1, fracao=True)],
    ]

    return "\n\n".join([
        secao("Sistema fotovoltaico proposto"),
        secao("Configuração geral", 2),
        tabela(["Parâmetro", "Valor"], linhas_geral, largura_primeira_coluna="8cm"),
        secao("Inversores e arranjo de strings", 2),
        tabela(
            ["Inversor", "Potência", "MPPTs", "Mód./string", "Strings/MPPT", "Strings", "Módulos"],
            linhas_inv,
            alinhamento="lrrrrrr",
            legenda="Configuração elétrica por modelo de inversor",
            tamanho_fonte="small",
        ),
        nota(
            "O número de módulos por string respeita a tensão de circuito aberto corrigida para "
            "0 graus Celsius (condição de tensão máxima) e a tensão de máxima potência corrigida para 70 graus Celsius "
            "(condição de tensão mínima), conforme exigido para não exceder a tensão máxima de "
            "entrada do inversor nem sair da janela de rastreamento."
        ),
        secao("Ocupação da cobertura", 2),
        tabela(["Parâmetro", "Valor"], linhas_layout, largura_primeira_coluna="8cm"),
    ])


def analise_economica(ctx: ContextoProposta) -> str:
    """Premissas, indicadores, composição do investimento e fluxo de caixa."""
    from ..pv.financials import composicao_capex

    eco = ctx.economia
    prem = eco.premissas

    linhas_premissas = [
        ["Investimento total (CAPEX)", dinheiro(prem.capex_brl)],
        ["Custo por kWp instalado", Raw(
            f"{dinheiro(prem.capex_brl / ctx.sistema.potencia_cc_kwp)} / kWp"
        ) if ctx.sistema.potencia_cc_kwp > 0 else "---"],
        ["Tarifa de energia adotada", Raw(f"{dinheiro(prem.tarifa_brl_kwh, 4)} / kWh")],
        ["Fração de autoconsumo", percentual(prem.fracao_autoconsumo, 0, fracao=True)],
        ["Operação e manutenção (ano 1)", dinheiro(prem.opex())],
        ["Degradação anual dos módulos", percentual(prem.degradacao_anual, 2, fracao=True)],
        ["Escalada tarifária anual", percentual(prem.escalada_tarifa, 1, fracao=True)],
        ["Taxa de desconto", percentual(prem.taxa_desconto, 1, fracao=True)],
        ["Horizonte de análise", Raw(f"{esc(prem.anos)} anos")],
        ["Ano de conexão considerado", prem.ano_conexao],
        ["Lei 14.300/2022 aplicada", "sim" if prem.aplicar_lei_14300 else "não"],
    ]

    linhas_indicadores = [
        ["Economia líquida no ano 1", dinheiro(eco.economia_ano1_brl)],
        [
            "Retorno simples",
            Raw(f"{numero(eco.payback_simples_anos, 1)} anos") if eco.payback_simples_anos else "não atingido",
        ],
        [
            "Retorno descontado",
            Raw(f"{numero(eco.payback_descontado_anos, 1)} anos")
            if eco.payback_descontado_anos
            else f"não atingido em {prem.anos} anos",
        ],
        ["Valor presente líquido", dinheiro(eco.vpl_brl)],
        [
            "Taxa interna de retorno",
            percentual(eco.tir_anual, 2, fracao=True) if eco.tir_anual is not None else "não aplicável",
        ],
        ["Custo nivelado da energia", Raw(f"{dinheiro(eco.lcoe_brl_kwh, 4)} / kWh") if eco.lcoe_brl_kwh else "---"],
        ["Economia acumulada no período", dinheiro(eco.economia_total_brl)],
        ["Energia gerada no período", energia(eco.geracao_total_kwh)],
    ]

    capex_itens = composicao_capex(prem.capex_brl)
    linhas_capex = [
        [nome, dinheiro(valor), percentual(valor / prem.capex_brl, 1, fracao=True)]
        for nome, valor in capex_itens.items()
    ]
    linhas_capex.append([
        Raw(r"\textbf{Total}"),
        Raw(rf"\textbf{{{dinheiro(prem.capex_brl)}}}"),
        Raw(r"\textbf{100,0\%}"),
    ])

    anos = [str(l.ano_calendario) for l in eco.fluxo]
    vpls = [l.vpl_acumulado_brl for l in eco.fluxo]

    linhas_fluxo = [
        [
            l.ano_calendario,
            energia(l.geracao_kwh),
            dinheiro(l.tarifa_brl_kwh, 4),
            dinheiro(l.economia_bruta_brl),
            dinheiro(-l.custo_fio_b_brl),
            dinheiro(-l.opex_brl),
            dinheiro(l.fluxo_liquido_brl),
            dinheiro(l.vpl_acumulado_brl),
        ]
        for l in eco.fluxo
    ]

    return "\n\n".join([
        secao("Análise econômico-financeira"),
        secao("Premissas adotadas", 2),
        tabela(["Premissa", "Valor"], linhas_premissas, largura_primeira_coluna="8cm"),
        secao("Indicadores de retorno", 2),
        tabela(["Indicador", "Valor"], linhas_indicadores, largura_primeira_coluna="8cm"),
        grafico_linha(anos, vpls, "Evolução do valor presente líquido acumulado", r"R\$"),
        secao("Composição do investimento", 2),
        tabela(
            ["Item", "Valor", "Participação"],
            linhas_capex,
            alinhamento="p{7.5cm}rr",
            legenda="Composição indicativa do CAPEX",
        ),
        nota(
            "A composição do investimento é indicativa, baseada em percentuais típicos do mercado "
            "brasileiro de geração distribuída. O orçamento definitivo depende de cotação de "
            "equipamentos, condições de acesso à cobertura e do projeto executivo."
        ),
        secao("Fluxo de caixa projetado", 2),
        tabela_longa(
            ["Ano", "Geração", "Tarifa", "Economia bruta", "Fio B", "O&M", "Fluxo líquido", "VPL acumulado"],
            linhas_fluxo,
            alinhamento="lrrrrrrr",
            legenda=f"Fluxo de caixa projetado para {prem.anos} anos",
        ),
    ])


def impacto_ambiental(ctx: ContextoProposta) -> str:
    eco = ctx.economia
    linhas = [
        ["Energia limpa gerada no período", energia(eco.geracao_total_kwh)],
        [Raw(f"Emissões de {CO2} evitadas"), Raw(f"{numero(eco.co2_evitado_t, 1)} t")],
        [
            "Equivalência em árvores",
            Raw(f"{inteiro(eco.arvores_equivalentes)} árvores mantidas por {esc(eco.premissas.anos)} anos"),
        ],
        ["Fator de emissão do SIN", Raw(r"0,0839 tCO\textsubscript{2}eq/MWh")],
    ]
    return "\n\n".join([
        secao("Impacto ambiental"),
        tabela(["Indicador", "Valor"], linhas, largura_primeira_coluna="8cm"),
        (
            "O fator de emissão utilizado é o médio do Sistema Interligado Nacional, publicado pelo "
            "Ministério da Ciência, Tecnologia e Inovação. A matriz elétrica brasileira é "
            "majoritariamente renovável, o que torna esse fator baixo em comparação internacional. "
            "A equivalência em árvores considera absorção de 100 kg de CO2 por árvore por ano."
        ),
    ])


def ressalvas(ctx: ContextoProposta) -> str:
    """Limitações do estudo. Seção obrigatória, nunca omitida."""
    avisos = ctx.avisos_consolidados()
    return "\n\n".join([
        secao("Limitações e ressalvas técnicas"),
        (
            "Este estudo é uma triagem automatizada de pré-viabilidade, construída a partir de dados "
            "públicos de mapeamento colaborativo e de bases de irradiação por satélite. Os pontos "
            "abaixo precisam ser verificados antes de qualquer compromisso comercial:"
        ),
        lista(avisos),
    ])


def proximos_passos(ctx: ContextoProposta) -> str:
    passos = [
        "Confirmar o ocupante da edificação e o responsável pela decisão de investimento.",
        "Obter as faturas de energia dos últimos 12 meses e a classificação tarifária real.",
        "Realizar visita técnica para levantar a estrutura da cobertura, obstáculos e sombreamento.",
        "Verificar a capacidade do padrão de entrada e a disponibilidade de rede junto à distribuidora.",
        "Elaborar projeto executivo com laudo estrutural e ART do responsável técnico.",
        "Protocolar a solicitação de acesso à distribuidora conforme a REN 1.000/2021.",
        "Emitir proposta comercial definitiva com equipamentos cotados e prazo de execução.",
    ]
    contato = []
    if ctx.emissor.telefone:
        contato.append(f"Telefone: {esc(ctx.emissor.telefone)}")
    if ctx.emissor.email:
        contato.append(f"E-mail: {esc(ctx.emissor.email)}")
    if ctx.emissor.site:
        contato.append(f"Site: {link(ctx.emissor.site)}")

    partes = [secao("Próximos passos"), lista(passos, numerada=True)]
    if contato:
        partes.append(
            nota("Para avançar com o estudo detalhado, entre em contato: " + " | ".join(contato))
        )
    return "\n\n".join(partes)


def fim() -> str:
    return r"\end{document}"


def montar_documento(ctx: ContextoProposta) -> str:
    """Concatena todas as seções numa proposta completa."""
    blocos = [
        preambulo(ctx),
        capa(ctx),
        sumario(),
        resumo_executivo(ctx),
        identificacao(ctx),
        recurso_solar(ctx),
        consumo_estimado(ctx),
        sistema_proposto(ctx),
        analise_economica(ctx),
        impacto_ambiental(ctx),
        ressalvas(ctx),
        proximos_passos(ctx),
        fim(),
    ]
    return "\n\n".join(b for b in blocos if b) + "\n"
