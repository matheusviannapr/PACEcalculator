"""
O dossiê: relatório completo do estudo em LaTeX, com PDF e ZIP.

O ``estudo.md`` do :mod:`aurum.bateria.relatorio` é a entrega interna — rápida
de ler, fácil de colar num e-mail. Este módulo produz a outra coisa: o
documento que vai junto da proposta, com capa, sumário, todas as análises e
as ressalvas onde elas pertencem.

A estrutura segue a ordem em que uma pergunta depende da anterior, que é
também a ordem em que um engenheiro defende o dimensionamento:

1. **O que se conclui** — antes de qualquer tabela. Quem lê quer saber o que
   comprar; o resto é a justificativa.
2. **O telhado** — a foto aérea com os módulos, o croqui cotado, a área.
3. **O recurso solar** — de onde vem a geração e quanto ela varia no ano
   (omitido quando o estudo não tem energia solar).
4. **A demanda** — o método de Monte Carlo, o levantamento de cargas, as
   curvas e a excedência de pico.
5. **O sistema fotovoltaico** — módulo, inversor, arranjo.
6. **O armazenamento** — triagem por potência, resiliência a apagões,
   degradação, economia.
7. **Premissas, limitações e procedência** — o que foi lido de datasheet, o
   que foi arbitrado, e o que o estudo não afirma.

A última seção não é formalidade. Um estudo que mistura número de datasheet
com ordem de grandeza de mercado e não diz qual é qual transfere ao leitor um
risco que ele não tem como avaliar; a coluna ``fonte_dado`` dos catálogos
existe para isso, e aqui ela vira tabela.
"""
from __future__ import annotations

import io
import logging
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .. import marca
from ..proposal.latex import (
    Raw,
    aviso as caixa_aviso,
    caixa,
    esc,
    lista,
    nota,
    secao,
    tabela,
    tabela_longa,
)
from ..proposal.render import compilar_pdf, encontrar_compilador
from ..proposal.sections import PREAMBULO
from .estudo import ResultadoEstudo

LOGGER = logging.getLogger(__name__)

__all__ = ["DadosCapa", "escrever_dossie", "montar_documento", "zip_do_dossie"]


@dataclass
class DadosCapa:
    """Identificação de quem emite o estudo."""

    empresa: str = marca.NOME_COMPLETO
    responsavel: str = ""
    crea: str = ""
    telefone: str = ""
    email: str = ""
    site: str = ""
    referencia: str = "Estudo de energia"


# ----------------------------------------------------------------------------
# Formatação
# ----------------------------------------------------------------------------
# Estes três devolvem **texto puro**, não LaTeX. Todo lugar que os consome —
# células de tabela, `caixa`, `nota`, `lista` — passa o conteúdo por `esc`, que
# é quem transforma `%` em `\%` e `$` em `\$`. Devolver já escapado daqui faria
# o escape acontecer duas vezes e imprimir uma barra invertida literal.
def _n(valor: Any, casas: int = 1, unidade: str = "") -> str:
    """Número no padrão brasileiro: ponto de milhar, vírgula decimal."""
    try:
        numero = float(valor)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(numero):
        return "—"
    texto = f"{numero:,.{casas}f}".replace(",", "\x00").replace(".", ",").replace("\x00", ".")
    return f"{texto} {unidade}".strip()


def _pct(valor: Any, casas: int = 0) -> str:
    try:
        return _n(float(valor) * 100.0, casas, "%")
    except (TypeError, ValueError):
        return "—"


def _brl(valor: Any, casas: int = 0) -> str:
    """
    Reais. Sem casas por padrão, porque investimento em centavos é ruído.

    Preço por kWh é a exceção que obriga o parâmetro: arredondado para
    inteiro, uma tarifa de R$ 0,96/kWh vira "R$ 1/kWh" -- número errado
    impresso com toda a autoridade de uma tabela.
    """
    return f"R$ {_n(valor, casas)}"


def _figura(caminho: Path | None, legenda: str, largura: str = "0.92") -> str:
    if caminho is None:
        return ""
    return "\n".join([
        r"\begin{figure}[H]\centering",
        rf"\includegraphics[width={largura}\linewidth]{{figuras/{caminho.name}}}",
        rf"\caption{{{esc(legenda)}}}",
        r"\end{figure}",
    ])


# ----------------------------------------------------------------------------
# Seções
# ----------------------------------------------------------------------------
def _capa(estudo: ResultadoEstudo, capa: DadosCapa) -> str:
    cfg = estudo.configuracao
    contato = " · ".join(x for x in (capa.telefone, capa.email, capa.site) if x)
    assinatura = " — ".join(x for x in (capa.responsavel, capa.crea) if x)
    from .. import marca

    logo = marca.logo("clara")
    return "\n".join([
        r"\begin{titlepage}\centering\vspace*{2cm}",
        # A logo entra por caminho absoluto: o .tex é compilado numa pasta de
        # saída que não é a do projeto, e um caminho relativo sairia quebrado
        # no Overleaf.
        (rf"\includegraphics[width=0.42\linewidth]{{{logo.as_posix()}}}\par\vspace{{1.4cm}}"
         if logo else ""),
        r"{\Huge\bfseries\color{pretopace} Estudo de energia\par}",
        r"\vspace{0.15cm}",
        r"{\color{amarelopace}\rule{4cm}{2pt}\par}",
        r"\vspace{0.4cm}",
        (r"{\Large Geração solar e armazenamento\par}" if estudo.com_solar
         else r"{\Large Armazenamento e continuidade de energia\par}"),
        r"\vspace{1.6cm}",
        rf"{{\LARGE\bfseries {esc(cfg.nome)}\par}}",
        r"\vspace{0.3cm}",
        rf"{{\large {esc(f'{cfg.latitude:.5f}, {cfg.longitude:.5f}')}\par}}",
        r"\vspace{2.2cm}",
        rf"{{\large {esc(capa.empresa)}\par}}",
        (rf"{{\normalsize {esc(assinatura)}\par}}" if assinatura else ""),
        (rf"\vspace{{0.2cm}}{{\small {esc(contato)}\par}}" if contato else ""),
        r"\vfill",
        r"{\footnotesize Documento gerado automaticamente pela plataforma da PACE. "
        r"As premissas, limitações e a procedência de cada dado estão declaradas "
        r"na última seção.\par}",
        r"\end{titlepage}",
        r"\tableofcontents",
        r"\newpage",
    ])


def _sumario_executivo(estudo: ResultadoEstudo) -> str:
    cfg = estudo.configuracao
    partes = [secao("O que se conclui")]

    if estudo.recomendado is not None:
        conjunto = estudo.recomendado.conjunto
        economico = estudo.economia[conjunto.descricao()]
        pior = estudo.recomendado.pior_janela(cfg.autonomia_alvo_h)
        partes.append(caixa(
            "Conjunto recomendado",
            f"{conjunto.descricao()}. Atravessa {cfg.autonomia_alvo_h:g} h de falta de "
            f"energia em pelo menos {cfg.confiabilidade_alvo:.0%} dos casos, inclusive no "
            f"pior horário simulado ({int(pior['hora_inicio'])} h, {pior['estacao']}), onde "
            f"ainda entrega {pior['prob_atendimento']:.0%}. É o conjunto de menor "
            f"investimento, entre os avaliados, que cumpre esse critério.",
            cor="verdepace",
        ))
        linhas = [
            *([("Sistema fotovoltaico", _n(estudo.potencia_fv_kwp, 1, "kWp"))]
              if estudo.com_solar else []),
            ("Banco de baterias", _n(conjunto.energia_util_kwh, 1, "kWh úteis")),
            ("Inversor híbrido", str(conjunto.inversor)),
            ("Potência contínua / de pico",
             f"{_n(conjunto.potencia_descarga_kw, 1, 'kW')} / "
             f"{_n(conjunto.potencia_pico_kw, 1, 'kW')} por "
             f"{_n(conjunto.inversor.duracao_pico_s, 0, 's')}"),
            ("Investimento estimado", _brl(economico.capex_brl)),
            ("Vida útil estimada do banco", _n(economico.vida_util_anos, 1, "anos")),
        ]
        if economico.payback_anos:
            linhas.append(("Retorno do investimento", _n(economico.payback_anos, 1, "anos")))
        partes.append(tabela(["Item", "Valor"], linhas, alinhamento="lr",
                             legenda="Resumo da solução recomendada"))
    else:
        melhor = estudo.ranking.sort_values("autonomia_garantida_h", ascending=False).iloc[0]
        partes.append(caixa_aviso(
            f"Nenhuma combinação do catálogo atravessa {cfg.autonomia_alvo_h:g} h com "
            f"{cfg.confiabilidade_alvo:.0%} de garantia. O conjunto que mais se aproxima é "
            f"{melhor['conjunto']}, com {melhor['autonomia_garantida_h']:g} h. A seção de "
            "armazenamento mostra se o gargalo é energia ou potência — e as duas pedem "
            "soluções opostas."
        ))
    return "\n\n".join(p for p in partes if p)


def _secao_telhado(estudo: ResultadoEstudo, figuras: dict[str, Path]) -> str:
    telhado = estudo.configuracao.telhado
    layout = estudo.configuracao.layout
    if telhado is None or layout is None:
        return "\n\n".join([
            secao("O telhado"),
            "Não foi marcada uma área de cobertura para este estudo. A potência "
            "fotovoltaica foi definida pelo consumo, e o documento não afirma que "
            "o sistema cabe fisicamente na edificação.",
            nota("Marcar o telhado no mapa permite medir a área real, derivar a "
                 "orientação e contar quantos módulos cabem — passando de uma "
                 "estimativa de compensação para um projeto verificável."),
        ])

    partes = [
        secao("O telhado"),
        f"A área de cobertura foi marcada sobre imagem de satélite e medida em "
        f"projeção métrica. O contorno tem {_n(telhado.area_m2, 0, 'm²')} em planta; "
        f"considerando a inclinação de {_n(telhado.inclinacao_deg, 0, 'graus')}, a "
        f"superfície real do telhado é de {_n(telhado.area_inclinada_m2, 0, 'm²')}.",
    ]
    if "foto_telhado" in figuras:
        partes.append(_figura(
            figuras["foto_telhado"],
            "Imagem aérea da edificação com o arranjo de módulos sobreposto.",
        ))
    partes.append(_figura(
        figuras.get("telhado"),
        "Croqui cotado do arranjo: contorno do telhado, módulos e a direção da face.",
        largura="0.80",
    ))
    partes.append(tabela(
        ["Grandeza", "Valor"],
        [
            ("Área em planta", _n(telhado.area_m2, 0, "m²")),
            ("Superfície real do telhado", _n(telhado.area_inclinada_m2, 0, "m²")),
            ("Perímetro", _n(telhado.perimetro_m, 0, "m")),
            ("Tipo de montagem",
             "coplanar (módulo acompanha a água)" if telhado.montagem == "coplanar"
             else "estrutura inclinada sobre laje"),
            ("Inclinação", _n(telhado.inclinacao_deg, 0, "graus")),
            ("Orientação da face",
             f"{telhado.orientacao} ({_n(telhado.azimute_deg, 0, 'graus')})"),
            ("Área aproveitável considerada", _pct(telhado.fator_obstaculos)),
            ("Módulo adotado", str(layout.modulo)),
            ("Orientação do módulo", layout.orientacao_modulo),
            ("Quantidade de módulos", _n(layout.quantidade, 0)),
            ("Potência instalada", _n(layout.potencia_kwp, 1, "kWp")),
            ("Ocupação do telhado", _pct(layout.taxa_ocupacao)),
            ("Densidade", _n(layout.densidade_wp_m2, 0, "Wp/m²")),
            ("Passo entre fileiras", _n(layout.passo_fileira_m, 2, "m")),
        ],
        alinhamento="lr", largura_primeira_coluna="7cm",
        legenda="Medidas do telhado e do arranjo",
    ))
    if telhado.montagem == "coplanar":
        partes.append(nota(
            "A orientação da face foi deduzida do contorno: a água é perpendicular à "
            "cumeeira, e o desenho em planta não distingue para qual dos dois lados ela "
            "cai. Foi adotada a mais próxima do ótimo da latitude. Confirmar em campo "
            "antes da emissão da proposta comercial."
        ))
    if "foto_telhado" in figuras:
        partes.append(nota(
            "A imagem aérea é referência visual, não levantamento topográfico: a data do "
            "voo varia por região e o georreferenciamento tem erro de alguns metros. As "
            "medidas do estudo vêm do polígono marcado, projetado em coordenadas métricas."
        ))
    return "\n\n".join(p for p in partes if p)


def _secao_solar(estudo: ResultadoEstudo, figuras: dict[str, Path]) -> str:
    serie = estudo.serie
    diaria = serie.energia_diaria_kwh_por_kwp()
    por_estacao = [
        (estacao, _n(diaria[serie.dias_da_estacao(estacao)].mean(), 2, "kWh/kWp"))
        for estacao in ("verão", "outono", "inverno", "primavera")
        if serie.dias_da_estacao(estacao).size
    ]
    partes = [
        secao("O recurso solar"),
        f"A geração foi obtida como série horária de {serie.n_dias} dias "
        f"({esc(serie.datas[0].isoformat())} a {esc(serie.datas[-1].isoformat())}), "
        f"normalizada por kWp instalado. A produtividade anual do local é de "
        f"{_n(serie.anual_kwh_por_kwp(), 0, 'kWh/kWp')}, e o sistema de "
        f"{_n(estudo.potencia_fv_kwp, 1, 'kWp')} gera cerca de "
        f"{_n(estudo.potencia_fv_kwp * serie.anual_kwh_por_kwp(), 0, 'kWh')} por ano.",
        "Trabalhar com a série horária de anos reais, e não com a média mensal, é o que "
        "carrega a persistência do tempo para dentro do estudo: sequências de dias "
        "encobertos existem na série porque existiram no céu, e um apagão longo que cai "
        "dentro de uma delas é justamente o caso que dimensiona o banco.",
        tabela(
            ["Grandeza", "Valor"],
            [
                # Sem `esc` aqui: `tabela` já escapa cada célula, e escapar duas
                # vezes imprime a barra invertida em vez de aplicá-la.
                ("Fonte da série", serie.fonte),
                ("Período", f"{serie.datas[0].isoformat()} a {serie.datas[-1].isoformat()}"),
                ("Inclinação adotada", _n(serie.inclinacao_deg, 0, "graus")),
                ("Azimute adotado", _n(serie.azimute_deg, 0, "graus (0 = Norte)")),
                ("Perdas do sistema", _n(serie.perdas_percent, 1, "%")),
                ("Produtividade anual", _n(serie.anual_kwh_por_kwp(), 0, "kWh/kWp")),
                *[(f"Geração diária média — {e}", v) for e, v in por_estacao],
            ],
            alinhamento="lr", largura_primeira_coluna="7cm",
            legenda="Recurso solar no local",
        ),
        _figura(figuras.get("dia_medio"),
                "Carga de backup e geração fotovoltaica ao longo do dia médio, por estação."),
    ]
    if not serie.confiavel:
        partes.append(caixa_aviso(
            "A série horária é sintética: a forma do dia vem de geometria solar e a "
            "energia foi ancorada em totais mensais estimados. Serve para estudo interno; "
            "substituir por consulta ao PVGIS antes de emitir proposta comercial."
        ))
    return "\n\n".join(p for p in partes if p)


def _secao_demanda(estudo: ResultadoEstudo, figuras: dict[str, Path]) -> str:
    cfg = estudo.configuracao
    total = estudo.ensemble_total.resumo()
    backup = estudo.ensemble_backup.resumo()
    partes = [
        secao("A demanda elétrica"),
        secao("Método", nivel=2),
        "A carga não foi representada por uma curva única, e sim por uma distribuição "
        "de curvas. Cada equipamento entra com potência, quantidade, probabilidade de "
        "uso no dia, fator de demanda e janela de operação; a simulação de Monte Carlo "
        f"sorteia {cfg.simulacoes} dias por estação do ano e devolve o conjunto de "
        "curvas diárias possíveis.",
        "A diferença importa para o dimensionamento: o inversor não é decidido pela "
        "média, e sim pela cauda. Uma curva média esconde exatamente o pico raro que "
        "faz o equipamento desarmar.",
    ]

    linhas = [
        ("Pico diário médio", _n(total["geral"]["pico_medio_kw"], 1, "kW"),
         _n(backup["geral"]["pico_medio_kw"], 1, "kW")),
        ("Pico diário P95", _n(total["geral"]["pico_p95_kw"], 1, "kW"),
         _n(backup["geral"]["pico_p95_kw"], 1, "kW")),
        ("Consumo diário médio", _n(total["geral"]["energia_diaria_media_kwh"], 0, "kWh"),
         _n(backup["geral"]["energia_diaria_media_kwh"], 0, "kWh")),
        ("Consumo anual estimado",
         _n(total["geral"]["energia_diaria_media_kwh"] * 365, 0, "kWh"),
         _n(backup["geral"]["energia_diaria_media_kwh"] * 365, 0, "kWh")),
    ]
    partes.append(tabela(
        ["Grandeza", "Instalação inteira", "Quadro de backup"], linhas,
        alinhamento="lrr", largura_primeira_coluna="6cm",
        legenda="Demanda simulada, com e sem o recorte de circuitos essenciais",
    ))
    if cfg.comodos_essenciais:
        partes.append(
            "Foram considerados essenciais, e portanto ligados ao inversor durante a "
            "falta de energia, os seguintes ambientes: "
            + esc(", ".join(cfg.comodos_essenciais)) + "."
        )
        partes.append(nota(
            "O recorte de circuitos essenciais é a decisão que mais influencia o custo "
            "do sistema. Alimentar a instalação inteira em modo ilhado exige inversor "
            "várias vezes maior, e o inversor domina o investimento."
        ))

    analise = cfg.analise
    if analise is not None:
        partes.extend(_analise_completa(analise, figuras))

    partes += [
        secao("Excedência de pico", nivel=2),
        "A tabela abaixo lê a distribuição de picos ao contrário: para cada "
        "probabilidade, qual potência é excedida. A coluna de pico diário responde com "
        "que frequência o limite é ultrapassado em algum instante do dia; a de carga "
        "instantânea, que fração do tempo a carga permanece acima dele.",
        tabela(
            ["Probabilidade de excedência", "Pico diário", "Carga instantânea"],
            [
                (_pct(linha.prob_excedencia, 1), _n(linha.pico_diario_kw, 1, "kW"),
                 _n(linha.carga_instantanea_kw, 1, "kW"))
                for linha in estudo.tabela_excedencia.itertuples()
            ],
            alinhamento="lrr", legenda="Potência exigida por probabilidade de excedência",
        ),
        _figura(figuras.get("excedencia"),
                "Curva de excedência do pico diário, com o limite nominal de cada inversor."),
    ]
    return "\n\n".join(p for p in partes if p)



def _analise_completa(analise, figuras: dict[str, Path]) -> list[str]:
    """
    A análise estatística do consumo — o miolo do que o D² entrega.

    Vem antes da excedência de pico de propósito: a excedência responde "que
    inversor aguenta", e essa é uma pergunta de equipamento. Antes dela cabe a
    pergunta de engenharia, que é como aquele lugar consome energia.
    """
    estatisticas = analise.estatisticas
    indicadores = analise.indicadores
    partes = [
        secao("Distribuição dos picos", nivel=2),
        "O pico de demanda não é um número, é uma distribuição. Cada dia simulado tem o "
        "seu, e o que dimensiona é um percentil dessa distribuição — não a média, que "
        "seria excedida em metade dos dias, nem o máximo, que pagaria por um evento "
        "único.",
        tabela(
            ["Estatística", "Valor"],
            [
                ("Simulações realizadas", _n(estatisticas.n, 0)),
                ("Pico médio", _n(estatisticas.media / 1000, 1, "kW")),
                ("Pico mediano", _n(estatisticas.mediana / 1000, 1, "kW")),
                ("Menor pico simulado", _n(estatisticas.minimo / 1000, 1, "kW")),
                ("Maior pico simulado", _n(estatisticas.maximo / 1000, 1, "kW")),
                ("Desvio padrão", _n(estatisticas.desvio / 1000, 1, "kW")),
                ("Coeficiente de variação", _pct(estatisticas.coeficiente_variacao, 1)),
                *[
                    (f"Percentil {q}", _n(valor / 1000, 1, "kW"))
                    for q, valor in estatisticas.percentis.items()
                ],
                ("Intervalo de 95% dos dias",
                 f"{_n(estatisticas.ic_inferior / 1000, 1)} a "
                 f"{_n(estatisticas.ic_superior / 1000, 1)} kW"),
                ("Incerteza do P95 pelo Monte Carlo",
                 f"± {_n(estatisticas.erro_padrao_p95 / 1000, 2, 'kW')}"),
            ],
            alinhamento="lr", largura_primeira_coluna="7.5cm",
            legenda="Distribuição do pico diário de demanda",
        ),
        nota(
            f"{estatisticas.interpretacao.capitalize()}. A última linha é a incerteza do "
            "próprio método: com o número de simulações realizadas, o P95 tem essa margem. "
            "Reportá-lo sem ela sugeriria uma precisão que o número não tem."
        ),
        _figura(figuras.get("histograma_picos"),
                "Distribuição dos picos diários, com a média e o percentil 95."),
    ]

    if analise.composicao is not None and not analise.composicao.por_equipamento.empty:
        composicao = analise.composicao
        principais = composicao.por_equipamento.head(8)
        partes += [
            secao("O que causa o pico", nivel=2),
            f"O instante de pico acontece com mais frequência às "
            rf"\textbf{{{composicao.hora_mais_provavel} h}}. A tabela abaixo decompõe esse "
            "instante: "
            "para cada equipamento, quanto ele contribui em média e em que fração das "
            "simulações ele estava ligado quando o pico ocorreu.",
            tabela(
                ["Equipamento", "Ambiente", "Contribuição", "Presença", "Fatia do pico"],
                [
                    (
                        linha.equipamento, linha.comodo,
                        _n(linha.carga_media_kw, 2, "kW"),
                        _pct(linha.presenca), _pct(linha.participacao, 1),
                    )
                    for linha in principais.itertuples()
                ],
                alinhamento="p{5.1cm}p{3cm}rrr", tamanho_fonte="small",
                legenda="Composição do instante de pico, por equipamento",
            ),
            nota(
                "Esta é a tabela que transforma um diagnóstico em uma decisão. Saber que o "
                "pico é alto não sugere nada; saber que ele é dominado por um único tipo "
                "de equipamento, numa hora conhecida, abre a conversa sobre deslocamento "
                "de carga ou substituição tecnológica — quase sempre mais barata que "
                "aumentar o sistema."
            ),
            _figura(figuras.get("composicao_pico"),
                    "Equipamentos que mais contribuem no instante de pico."),
        ]

    partes += [
        secao("Fatores de carga, demanda e coincidência", nivel=2),
        "Os três fatores descrevem a mesma instalação de ângulos diferentes, e é a "
        "distância entre eles que revela o tamanho real do sistema necessário.",
        tabela(
            ["Grandeza", "Valor", "O que significa"],
            [
                ("Potência instalada", _n(indicadores.potencia_instalada_kw, 1, "kW"),
                 "soma das potências de placa"),
                ("Soma dos picos por ambiente",
                 _n(indicadores.soma_dos_picos_individuais_kw, 1, "kW"),
                 "o que cada ambiente pediria isolado dos demais"),
                ("Demanda máxima do conjunto", _n(indicadores.demanda_maxima_kw, 1, "kW"),
                 "o que a instalação pede junta, no percentil de dimensionamento"),
                ("Demanda média", _n(indicadores.demanda_media_kw, 1, "kW"),
                 "consumo diário dividido por 24 h"),
                ("Fator de demanda", _n(indicadores.fator_de_demanda, 2),
                 "demanda máxima sobre potência instalada"),
                ("Fator de coincidência", _n(indicadores.fator_de_coincidencia, 2),
                 "quanto se economiza por os equipamentos não ligarem juntos"),
                ("Fator de carga", _n(indicadores.fator_de_carga, 2),
                 indicadores.interpretacao_fator_carga),
            ],
            alinhamento="p{5.5cm}rp{5.5cm}",
            legenda="Fatores característicos da instalação",
        ),
        _figura(figuras.get("fator_carga_hora"),
                "Fator de carga hora a hora: onde a demanda é firme e onde depende de coincidência."),
        secao("Curva de duração de carga", nivel=2),
        "A curva de duração ordena todas as potências instantâneas simuladas da maior "
        r"para a menor. Lê-se assim: a altura em 10\% é a potência que a instalação supera "
        "em um décimo do tempo. A parte alta e estreita da curva é a carga que existe em "
        "poucas horas do ano — exatamente a que um banco de baterias corta sem que "
        "ninguém perceba, e a que encarece o transformador se for atendida pela rede.",
        _figura(figuras.get("duracao_carga"),
                "Curva de duração de carga, por estação e no agregado."),
    ]

    if figuras.get("por_ambiente"):
        partes += [
            secao("Composição por ambiente", nivel=2),
            "A área empilhada mostra de onde vem a demanda em cada hora. É o gráfico que "
            "identifica qual ambiente domina o pico e qual sustenta a carga de base.",
            _figura(figuras["por_ambiente"],
                    "Demanda por ambiente ao longo do dia médio."),
        ]

    partes += [
        secao("Comportamento sazonal", nivel=2),
        tabela(
            ["Estação", "Pico médio", "Pico P95", "Maior pico", "Consumo diário", "Fator de carga"],
            [
                (
                    linha.estacao, _n(linha.pico_medio_kw, 1, "kW"),
                    _n(linha.pico_p95_kw, 1, "kW"), _n(linha.pico_maximo_kw, 1, "kW"),
                    _n(linha.consumo_diario_kwh, 0, "kWh"), _n(linha.fator_de_carga, 2),
                )
                for linha in analise.por_estacao.itertuples()
            ],
            alinhamento="lrrrrr",
            legenda="Demanda por estação do ano",
        ),
        "O mesmo prédio tem picos diferentes em janeiro e em julho, e é o pior deles que "
        "dimensiona. Simular as quatro estações em separado, em vez de uma média anual, é "
        "o que impede um sistema correto no papel de ficar pequeno no verão.",
    ]
    return [x for x in partes if x]


def _secao_fotovoltaico(estudo: ResultadoEstudo) -> str:
    layout = estudo.configuracao.layout
    partes = [secao("O sistema fotovoltaico")]
    if layout is not None:
        modulo = layout.modulo
        partes.append(tabela(
            ["Característica", "Valor"],
            [
                ("Módulo", str(modulo)),
                ("Potência unitária", _n(modulo.potencia_wp, 0, "Wp")),
                ("Tensão de máxima potência (Vmp)", _n(modulo.vmp, 2, "V")),
                ("Corrente de máxima potência (Imp)", _n(modulo.imp, 2, "A")),
                ("Tensão de circuito aberto (Voc)", _n(modulo.voc, 2, "V")),
                ("Corrente de curto-circuito (Isc)", _n(modulo.isc, 2, "A")),
                ("Eficiência", _n(modulo.eficiencia_percent, 1, "%")),
                ("Dimensões",
                 f"{_n(modulo.comprimento_m, 3, 'm')} × {_n(modulo.largura_m, 3, 'm')}"),
                ("Área unitária", _n(modulo.area_m2, 2, "m²")),
                # A contagem do arranjo vem da memória quando ela existe: o
                # telhado comporta uma quantidade, e o inversor aceita outra.
                # Publicar a do telhado ao lado de uma memória que fecha em
                # número menor deixa duas verdades no mesmo documento.
                ("Módulos que o telhado comporta", _n(layout.quantidade, 0)),
                *(
                    [
                        ("Módulos no arranjo dimensionado",
                         _n(estudo.configuracao.memoria.modulos_do_sistema, 0)),
                        ("Potência total do gerador",
                         _n(estudo.configuracao.memoria.potencia_do_sistema_kwp, 1, "kWp")),
                    ]
                    if estudo.configuracao.memoria is not None
                    else [("Potência total do gerador", _n(layout.potencia_kwp, 1, "kWp"))]
                ),
            ],
            alinhamento="lr", largura_primeira_coluna="7cm",
            legenda="Módulo fotovoltaico adotado",
        ))
        if modulo.dimensoes_derivadas:
            partes.append(caixa_aviso(
                "As dimensões deste módulo foram derivadas da eficiência declarada, e não "
                "lidas do datasheet. Como a área do módulo determina quantos cabem no "
                "telhado, conferir a folha de dados do modelo cotado antes de fechar a "
                "quantidade."
            ))
    memoria = estudo.configuracao.memoria
    if memoria is not None:
        partes.extend(_memoria_de_calculo(memoria))
    else:
        partes.append(
            f"O gerador foi dimensionado em {_n(estudo.potencia_fv_kwp, 1, 'kWp')} a partir "
            "do consumo anual estimado, sem verificação de área disponível."
        )
    return "\n\n".join(p for p in partes if p)



#: Símbolos que a memória de cálculo usa e que o pdflatex não sabe desenhar em
#: modo texto com `inputenc utf8`. O β mata a compilação inteira com "Unicode
#: character not set up for use with LaTeX" — sem PDF nenhum, e a mensagem não
#: diz de onde veio. Traduzir na fronteira é mais simples que carregar um pacote
#: de fontes só por causa de três caracteres.
_SIMBOLOS_LATEX = {
    "β": "b", "×": "x", "÷": "/", "≤": "<=", "≥": ">=",
    "⌊": "[", "⌋": "]", "⌈": "[", "⌉": "]", "−": "-", "√": "raiz",
    "Δ": "delta", "η": "eta", "α": "a", "γ": "g",
    # Os comentários da memória são escritos uma vez e lidos em dois lugares:
    # na interface, onde `**` vira negrito, e aqui, onde sairia literal no PDF.
    # Descartar a ênfase é melhor que traduzi-la, porque `\textbf` seria
    # escapado pelo `esc` logo em seguida.
    "**": "",
}


def _sem_simbolos(texto: str) -> str:
    for antes, depois in _SIMBOLOS_LATEX.items():
        texto = texto.replace(antes, depois)
    return texto


def _memoria_de_calculo(memoria) -> list[str]:
    """
    A conta do arranjo, passo a passo, para o equipamento escolhido.

    Um relatório que apresenta a quantidade de módulos sem mostrar de onde ela
    veio pede que se acredite nele. A memória permite conferir — e é ela que um
    responsável técnico assina.
    """
    inversor = memoria.inversor
    partes = [
        secao("Memória de cálculo do arranjo", nivel=2),
        tabela(
            ["Característica do inversor", "Valor"],
            [
                ("Modelo", str(inversor)),
                ("Potência nominal CA", _n(inversor.potencia_ca_kw, 1, "kW")),
                ("Potência FV máxima", _n(inversor.potencia_fv_max_w / 1000, 1, "kWp")),
                ("Tensão máxima CC", _n(inversor.tensao_max_cc, 0, "V")),
                ("Tensão de partida", _n(inversor.tensao_start, 0, "V")),
                ("Rastreadores de máxima potência", _n(inversor.num_mppt, 0)),
                ("Corrente máxima por MPPT", _n(inversor.corrente_max_mppt, 1, "A")),
                ("Corrente de curto máxima por MPPT",
                 _n(inversor.corrente_curto_max_mppt, 1, "A")),
                ("Fases", _n(inversor.fases, 0)),
            ],
            alinhamento="lr", largura_primeira_coluna="7.5cm",
            legenda="Inversor adotado",
        ),
        f"As temperaturas de projeto adotadas são {_n(memoria.temp_min_c, 0)} °C de célula "
        f"mínima e {_n(memoria.temp_max_c, 0)} °C de célula máxima. A primeira é a condição "
        "crítica de tensão, e a segunda, a de corrente. Ambas ficam registradas porque uma "
        "instalação em clima diferente exige refazer esta conta.",
    ]

    for i, passo in enumerate(memoria.passos, start=1):
        corpo = [
            f"\\textbf{{{i}. {esc(_sem_simbolos(passo.titulo))}}}",
            "",
            rf"\begin{{center}}\texttt{{{esc(_sem_simbolos(passo.formula))}}}\end{{center}}",
            rf"\begin{{center}}\texttt{{{esc(_sem_simbolos(passo.substituicao))}}}\end{{center}}",
            rf"\begin{{center}}\textbf{{= {esc(_sem_simbolos(passo.resultado))}}}\end{{center}}",
        ]
        if passo.comentario:
            corpo.append(rf"\small {esc(_sem_simbolos(passo.comentario))}\normalsize")
        partes.append("\n".join(corpo))

    partes.append(tabela(
        ["Resultado do dimensionamento", "Valor"],
        [
            ("Módulos em série por string", _n(memoria.modulos_por_string, 0)),
            ("Strings por inversor", _n(memoria.strings, 0)),
            ("Inversores", _n(memoria.inversores, 0)),
            ("Strings no sistema", _n(memoria.strings_do_sistema or memoria.strings, 0)),
            ("Módulos no sistema", _n(memoria.modulos_do_sistema, 0)),
            ("Potência do gerador", _n(memoria.potencia_do_sistema_kwp, 2, "kWp")),
            ("Potência CA instalada",
             _n(inversor.potencia_ca_kw * memoria.inversores, 1, "kW")),
            ("Razão CC/CA por inversor", _n(memoria.razao_cc_ca, 2)),
        ],
        alinhamento="lr", largura_primeira_coluna="7.5cm",
        legenda="Arranjo fotovoltaico dimensionado",
    ))
    for aviso in memoria.avisos:
        partes.append(caixa_aviso(aviso))
    return [x for x in partes if x]


def _sem_parenteses(descricao: str) -> str:
    """
    Tira o parentético da descrição do conjunto.

    A descrição do catálogo repete kWh úteis e kW, que já são colunas próprias
    da tabela; repetidos, só alargam a coluna até estourar a margem.
    """
    return re.sub(r"\s*\([^)]*\)\s*$", "", str(descricao)).strip()


def _resumo_da_triagem(estudo: ResultadoEstudo) -> str:
    """
    A síntese da triagem, em vez do catálogo inteiro linha a linha.

    O que decide é a exigência de potência da carga e onde cada modelo cai em
    relação a ela — não a lista de quinze inversores que ninguém vai cotar.
    """
    diagnostico = estudo.diagnostico_inversores
    if diagnostico.empty:
        return ""
    aprovados = diagnostico[diagnostico["aprovado"]]
    reprovados = diagnostico[~diagnostico["aprovado"]]

    exigencia = estudo.tabela_excedencia
    linha_1pc = exigencia[np.isclose(exigencia["prob_excedencia"], 0.01)]
    exigido = float(linha_1pc["pico_diario_kw"].iloc[0]) if not linha_1pc.empty else float("nan")

    linhas = [
        ("Pico exigido em 99% dos dias", _n(exigido, 1, "kW")),
        ("Modelos avaliados", _n(len(diagnostico), 0)),
        ("Modelos que atendem", _n(len(aprovados), 0)),
        ("Modelos reprovados por potência", _n(len(reprovados), 0)),
    ]
    if not aprovados.empty:
        menor = aprovados.iloc[0]
        linhas.append((
            "Menor inversor que atende",
            f"{menor['fabricante']} {menor['modelo']} ({_n(menor['nominal_kw'], 1, 'kW')})",
        ))
    return tabela(
        ["Triagem", "Resultado"], linhas, alinhamento="lr",
        largura_primeira_coluna="8cm",
        legenda="Síntese da triagem de inversores por potência",
    )


def _secao_armazenamento(estudo: ResultadoEstudo, figuras: dict[str, Path]) -> str:
    cfg = estudo.configuracao
    partes = [
        secao("O armazenamento"),
        secao("Triagem por potência", nivel=2),
        "Antes de simular autonomia, cada inversor do catálogo é confrontado com a "
        "distribuição de picos da carga de backup. Um inversor que desarma não tem "
        "autonomia nenhuma, por maior que seja o banco ligado a ele.",
        # A tabela linha a linha do catálogo inteiro fica na ferramenta e no CSV
        # anexo: numa proposta, listar quinze modelos que ninguém vai comprar
        # rouba a atenção da conclusão. O documento traz a síntese — o que
        # passou, o que não passou, e a exigência que separa os dois.
        _resumo_da_triagem(estudo),
        nota(
            "Potência nominal e sobrecarga respondem perguntas diferentes. Um inversor de "
            "5 kW com sobrecarga de 10 kW por 10 s atende a partida de um motor que puxa "
            "8 kW por 2 s, e não atende um chuveiro de 5,5 kW ligado por 8 minutos — "
            "potência menor, duração maior. A triagem completa, modelo a modelo, está na "
            "planilha 'diagnostico_inversores.csv' que acompanha este documento."
        ),
        secao("Resiliência a faltas de energia", nivel=2),
        f"A varredura simulou {esc(cfg.malha.descricao())}, por conjunto candidato. "
        "Cada cenário parte do banco no estado de carga informado, aplica a curva de "
        "carga sorteada"
        + (" e a geração solar real daquele trecho do ano" if estudo.com_solar else "")
        + ", e acompanha o despacho passo a passo até o fim da falta ou até a "
        "primeira interrupção.",
    ]

    if estudo.recomendado is not None:
        resultado = estudo.recomendado
        # Cabeçalhos curtos e cinco colunas estreitas: a versão anterior tinha
        # títulos de até 27 caracteres numa tabela de largura fixa, e a última
        # coluna saía para fora da margem.
        partes.append(tabela(
            ["Falta", "Atravessa", "Pior caso", "Não suprida", "Por potência"],
            [
                (_n(linha.duracao_h, 0, "h"), _pct(linha.prob_atendimento, 1),
                 _pct(linha.pior_prob, 1), _n(linha.ens_medio_kwh, 1, "kWh"),
                 _pct(linha.falhas_por_potencia, 0))
                for linha in resultado.por_duracao().itertuples()
            ],
            alinhamento="lrrrr", tamanho_fonte="small",
            legenda=(
                f"Desempenho do conjunto recomendado. "
                f"'Atravessa' é a média sobre horas e estações; 'pior caso', o pior par "
                f"delas; 'não suprida', a energia que falta quando não atravessa"
            ),
        ))
        partes.append(_figura(
            figuras.get("mapa_atendimento"),
            "Probabilidade de atravessar a falta, por hora de início e duração, em cada estação.",
            largura="1.0",
        ))
        partes.append(_figura(
            figuras.get("soc"),
            "Estado de carga do banco durante a falta mais longa simulada, com faixa P5–P95.",
        ))

    partes.append(secao("Alternativas avaliadas", nivel=2))
    partes.append(tabela_longa(
        ["Conjunto", "kWh", "kW", "Hoje", "Ano 10", "Investimento", "Meta"],
        [
            (
                _sem_parenteses(linha.conjunto),
                _n(linha.energia_util_kwh, 1), _n(linha.potencia_kw, 1),
                _n(linha.autonomia_garantida_h, 0, "h"),
                _n(linha.autonomia_ano10_h, 0, "h"),
                _brl(linha.capex_brl), "sim" if linha.atende_meta else "não",
            )
            for linha in estudo.ranking.itertuples()
        ],
        alinhamento="p{6cm}rrrrrl",
        legenda=(
            "Conjuntos avaliados. 'kWh' é a energia útil do banco; 'Hoje' e 'Ano 10', "
            "a autonomia garantida agora e depois de dez anos de degradação"
        ),
    ))
    partes.append(_figura(
        figuras.get("fronteira"),
        "Fronteira de dimensionamento: a partir de certo ponto, mais energia deixa de "
        "aumentar a autonomia porque o limite passa a ser a potência do inversor.",
    ))
    return "\n\n".join(p for p in partes if p)


def _secao_vida_util(estudo: ResultadoEstudo, figuras: dict[str, Path]) -> str:
    if estudo.recomendado is None:
        return ""
    chave = estudo.recomendado.conjunto.descricao()
    if chave not in estudo.degradacao:
        return ""
    trajetoria = estudo.degradacao[chave]
    autonomias = estudo.resiliencia_degradada.get(chave, {})
    economico = estudo.economia[chave]
    partes = [
        secao("Vida útil e degradação"),
        "A capacidade do banco diminui por dois mecanismos simultâneos: o calendário, "
        "que age mesmo sem uso e cresce com a raiz do tempo, e a ciclagem, proporcional "
        "à energia processada. O estudo aplica os dois e reavalia a autonomia com o "
        "banco já envelhecido — a promessa de autonomia passa a ter data.",
        tabela(
            ["Ano", "Retenção de capacidade", "Energia útil", "Autonomia garantida"],
            [
                (
                    _n(linha.ano, 0),
                    _pct(linha.retencao, 1),
                    _n(linha.energia_util_kwh, 1, "kWh"),
                    _n(autonomias[int(linha.ano)], 0, "h") if int(linha.ano) in autonomias else "—",
                )
                for linha in trajetoria.itertuples()
                if int(linha.ano) in (0, 1, 2, 5, 8, 10, 12, 15)
            ],
            alinhamento="lrrr",
            legenda="Degradação do banco e efeito sobre a autonomia",
        ),
        f"Com {_n(economico.operacao.ciclos_equivalentes, 0)} ciclos equivalentes por ano, "
        f"a vida útil estimada do banco é de {_n(economico.vida_util_anos, 1, 'anos')}"
        + (f", com substituição prevista no ano {economico.trocas[0]} do horizonte de análise."
           if economico.trocas else ", sem substituição dentro do horizonte de análise."),
        _figura(figuras.get("degradacao"),
                "Perda de capacidade por calendário e por ciclagem, e a autonomia resultante."),
        nota(
            "O modelo de degradação não considera temperatura. A dependência é forte — a "
            "perda de calendário praticamente dobra a cada 10 °C acima de 25 °C — e "
            "estimá-la sem conhecer o local de instalação daria precisão falsa. Banco "
            "instalado em ambiente sem ventilação exige revisão desta seção."
        ),
    ]
    return "\n\n".join(p for p in partes if p)


def _secao_cenarios(estudo: ResultadoEstudo, figuras: dict[str, Path]) -> str:
    """
    Solar, bateria e gerador ligados e desligados, contra a conta de hoje.

    É a seção que responde à pergunta que vem antes de todas as outras --
    comprar o quê -- e a única que compara o investimento com a alternativa de
    não investir. Sem ela o documento inteiro pressupõe a resposta.
    """
    comparacao = estudo.cenarios
    if comparacao is None or len(comparacao.cenarios) < 2:
        return ""

    fatura = comparacao.fatura
    partes: list[str] = [
        secao("Cenários: com e sem cada fonte"),
        "As três fontes resolvem problemas diferentes e por isso não competem "
        "diretamente. O solar reduz a conta e não fornece backup nenhum: inversor "
        "conectado à rede desliga quando a rede cai, por exigência de anti-ilhamento "
        "da ABNT NBR 16149. A bateria fornece backup e, na tarifa simples, economiza "
        "pouco -- sem diferença de tarifa entre horários não há arbitragem, e o que "
        "sobra é recuperar o pedaço que o Fio B retém da energia injetada. O gerador "
        "é o inverso da bateria: energia praticamente ilimitada, potência limitada, e "
        "um custo por kWh que só aparece quando ele roda.",
        caixa(
            "O referencial",
            f"Todos os números desta seção são medidos contra a conta de hoje: "
            f"{_brl(comparacao.base.conta_anual_brl)} por ano, a "
            f"{_brl(fatura.tarifa_brl_kwh, 2)}/kWh, com custo de disponibilidade de "
            f"{_n(fatura.custo_disponibilidade_kwh, 0, 'kWh')} por mês. Tarifa simples: "
            "posto tarifário e demanda contratada não entram nesta comparação.",
        ),
        tabela_longa(
            ["Arranjo", "Investimento", "Economia", "Autonomia", "Sem energia", "VPL", "Payback"],
            [
                (
                    cenario.nome.replace("Rede + ", "").capitalize(),
                    _brl(cenario.capex_brl) if cenario.capex_brl > 0 else "—",
                    _brl(cenario.economia_anual_brl) if cenario.economia_anual_brl > 0 else "—",
                    (
                        _n(cenario.autonomia_garantida_h, 0, "h")
                        if cenario.autonomia_garantida_h > 0
                        else "0 h"
                    ),
                    _n(cenario.ens_por_evento_kwh, 1, "kWh"),
                    _brl(cenario.vpl_brl),
                    _n(cenario.payback_anos, 1, "anos") if cenario.payback_anos else "—",
                )
                for cenario in comparacao.cenarios
            ],
            alinhamento="p{3.9cm}rrrrrr", tamanho_fonte="scriptsize",
            legenda=(
                "Cada arranjo de fontes contra a conta de hoje. 'Economia' é por ano; "
                "'Autonomia', a maior falta atravessada em 95% dos casos no pior par de "
                "estação e hora; 'Sem energia', o que falta à carga essencial numa "
                "interrupção média"
            ),
        ),
    ]

    partes.append(_premissa_de_capex(estudo, comparacao))

    partes.append(nota(
        "Autonomia zero não quer dizer que o arranjo não sirva para nada. Ela exige "
        "atravessar a falta inteira em 95% dos casos, no pior par de estação e hora; "
        "um grupo gerador que cobre 95% da energia mas afunda na partida de um motor "
        "aparece com autonomia zero e com 'sem energia' perto de zero. A primeira "
        "coluna mede a promessa que se pode assinar, a segunda mede o estrago real."
    ))

    if "cenarios" in figuras:
        partes.append(_figura(
            figuras.get("cenarios"),
            "Os arranjos nas duas dimensões que decidem: o que sobra na conta e o que "
            "acontece quando a luz cai.",
            largura="1.0",
        ))

    melhor_vpl = comparacao.melhor_vpl
    melhor_res = comparacao.melhor_resiliencia
    if melhor_vpl is not None:
        frase = (
            "Pelo valor presente, o arranjo de maior retorno é "
            + rf"\textbf{{{esc(melhor_vpl.nome)}}}: "
            + esc(
                f"{_brl(melhor_vpl.capex_brl)} de investimento, "
                f"{_brl(melhor_vpl.economia_anual_brl)} por ano de economia e valor "
                f"presente de {_brl(melhor_vpl.vpl_brl)} em "
                f"{comparacao.premissas.anos_analise} anos."
            )
        )
        if melhor_res is not None and melhor_res is not melhor_vpl:
            frase += (
                " Pela resiliência, é "
                + rf"\textbf{{{esc(melhor_res.nome)}}}"
                + esc(
                    f", com {_n(melhor_res.autonomia_garantida_h, 0, 'h')} de autonomia "
                    "garantida. Os dois critérios apontam para arranjos diferentes, e a "
                    "escolha entre eles não é técnica: depende de quanto custa ao cliente "
                    "uma hora sem energia."
                )
            )
        partes.append(frase)

    if comparacao.premissas.custo_interrupcao_brl_kwh > 0:
        partes.append(nota(
            f"O valor da resiliência entra a "
            f"{_brl(comparacao.premissas.custo_interrupcao_brl_kwh, 2)} por kWh não suprido, "
            f"com {_n(comparacao.premissas.interrupcoes_por_ano, 1)} interrupções por ano "
            f"de {_n(comparacao.premissas.duracao_media_interrupcao_h, 1, 'h')} em média. "
            "É o parâmetro mais sensível de toda a análise e o único que não se estima "
            "de fora: ele vem do cliente."
        ))
    else:
        partes.append(caixa(
            "A resiliência não foi precificada",
            "O custo da interrupção ficou em zero, então nenhum payback desta seção "
            "inclui o valor de não ficar sem energia. Na tarifa simples é justamente "
            "esse valor que paga a bateria: o quadro acima compara apenas economia de "
            "conta, e por isso subestima os arranjos com armazenamento.",
            cor="vermelhoaviso",
        ))

    if comparacao.gerador is not None:
        partes.extend(_subsecao_gerador(comparacao, figuras))

    return "\n\n".join(partes)


def _premissa_de_capex(estudo: ResultadoEstudo, comparacao) -> str:
    """
    De onde veio o R$/kWp do sistema solar — e por que isso não é detalhe.

    É a premissa mais sensível de toda a seção de cenários. Entre o padrão
    médio do mercado e uma obra de alto padrão a diferença chega a 47% no
    investimento, e ela atravessa inteira para o payback e para o valor
    presente. Um documento que apresenta payback sem dizer a que preço o
    calculou pede confiança em vez de dar informação.
    """
    if not estudo.com_solar:
        return ""
    from ..pv.financials import PADROES_CAPEX

    cfg = estudo.configuracao
    kwp = estudo.potencia_fv_kwp
    capex = next(
        (c.capex_brl for c in comparacao.cenarios
         if c.composicao.solar and not c.composicao.bateria and not c.composicao.gerador),
        0.0,
    )
    if capex <= 0 or kwp <= 0:
        return ""

    if cfg.capex_fv_brl is not None:
        origem = (
            "O investimento no sistema fotovoltaico foi informado, e não estimado: "
            f"{_brl(capex)} para {_n(kwp, 1, 'kWp')}, ou {_brl(capex / kwp)}/kWp."
        )
    else:
        rotulos = {
            "basico": "básico (telhado metálico, acesso livre, estrutura padrão)",
            "padrao": "padrão (o caso médio do mercado de geração distribuída)",
            "alto": "alto padrão (prédio ocupado, estrutura e acabamento acima da "
                    "média, projeto executivo e prazo curto)",
        }
        nome = rotulos.get(cfg.padrao_capex, cfg.padrao_capex)
        origem = (
            f"O investimento no sistema fotovoltaico é estimado para obra de "
            f"{esc(nome)}: {_brl(capex)} para {_n(kwp, 1, 'kWp')}, ou "
            f"{_brl(capex / kwp)}/kWp. A curva tem ganho de escala — engenharia, "
            f"mobilização e projeto se diluem —, então o R$/kWp cai com o tamanho do "
            "sistema."
        )
        faixa = sorted(PADROES_CAPEX.values())
        origem += (
            f" Entre o padrão mais simples e o mais exigente do catálogo a diferença "
            f"chega a {faixa[-1] / faixa[0] - 1:.0%}, e ela atravessa inteira para o "
            "payback e para o valor presente."
        )
    return caixa("De onde vem o investimento", origem, cor="pretopace")


def _subsecao_gerador(comparacao, figuras: dict[str, Path]) -> list[str]:
    """
    O grupo gerador: dimensionado para suprir, cobrado por hora rodada.

    Grupo gerador não se dimensiona por autonomia -- combustível se compra. O
    que ele tem de específico, e o que decide a compra, é o custo de rodar: um
    número que não aparece em nenhuma outra fonte porque nenhuma outra queima
    dinheiro enquanto trabalha.
    """
    from .fontes import estudo_do_gerador

    grupo = comparacao.gerador
    fatura = comparacao.fatura
    premissas = comparacao.premissas
    partes: list[str] = [
        secao("O grupo gerador e o custo de rodar", nivel=2),
        "O grupo foi dimensionado para suprir a carga essencial, e não para durar um "
        "tempo determinado: combustível se compra, e um gerador que atende a carga "
        "atravessa qualquer duração de falta. Com a potência resolvida, o que resta "
        "para decidir é quanto custa cada hora ligada.",
        tabela(
            ["Grupo gerador", "Valor"],
            [
                ("Potência", _n(grupo.potencia_kw, 0, "kW")),
                ("Custo da energia gerada", f"{_brl(grupo.custo_energia_brl_kwh, 2)}/kWh"),
                ("Investimento", _brl(grupo.capex_brl)),
                ("Manutenção fixa", f"{_brl(grupo.opex_fixo_brl_ano)}/ano"),
                ("Atraso de partida", _n(grupo.atraso_partida_min * 60, 0, "s")),
                ("Interrupções consideradas",
                 f"{_n(premissas.interrupcoes_por_ano, 1)} por ano"),
            ],
            alinhamento="lr", largura_primeira_coluna="8cm",
            legenda="Premissas do grupo gerador",
        ),
    ]

    custo = estudo_do_gerador(comparacao)
    if not custo.empty:
        partes.append(tabela_longa(
            ["Arranjo", "Falta", "Energia gerada", "Custo do evento", "Custo anual"],
            [
                (
                    linha.cenario.replace("Rede + ", "").capitalize(),
                    _n(linha.duracao_h, 0, "h"),
                    _n(linha.energia_gerada_kwh, 0, "kWh"),
                    _brl(linha.custo_do_evento_brl),
                    _brl(linha.custo_anual_brl),
                )
                for linha in custo.itertuples()
            ],
            alinhamento="p{5cm}rrrr",
            legenda=(
                "Combustível queimado e o que ele custa, por duração de falta. O custo "
                "anual multiplica o custo do evento pela frequência de interrupções "
                "declarada"
            ),
        ))
        if "gerador" in figuras:
            partes.append(_figura(
                figuras.get("gerador"),
                "O custo de rodar o grupo cresce com a duração da falta, porque o que se "
                "queima é energia. É por isso que a escolha entre gerador e bateria muda "
                "de lado conforme a falta é curta ou longa.",
            ))

        # A comparação é entre o arranjo que mais queima e o que menos queima,
        # e não entre "com banco" e "sem banco": sem sol, a bateria faz o grupo
        # queimar um pouco MAIS, porque ele a recarrega. Quem economiza
        # combustível é o sol, e afirmar o contrário seria vender bateria com
        # argumento que a própria tabela desmente.
        maior = custo["duracao_h"].max()
        na_maior = custo[custo["duracao_h"] == maior]
        caro = na_maior.loc[na_maior["custo_do_evento_brl"].idxmax()]
        barato = na_maior.loc[na_maior["custo_do_evento_brl"].idxmin()]
        if float(caro["custo_do_evento_brl"]) > 0 and caro["cenario"] != barato["cenario"]:
            reducao = 1 - float(barato["custo_do_evento_brl"]) / float(caro["custo_do_evento_brl"])
            partes.append(nota(
                f"Numa falta de {maior:g} h, o arranjo que mais queima combustível é "
                f"{esc(caro['cenario'].lower())}, com {_brl(caro['custo_do_evento_brl'])}; "
                f"o que menos queima é {esc(barato['cenario'].lower())}, com "
                f"{_brl(barato['custo_do_evento_brl'])} — {_pct(reducao, 0)} a menos. "
                "O que poupa combustível é a energia que chega de graça: cada kWh que o "
                "sol entrega ao quadro ilhado é um kWh que o grupo não queima. A bateria "
                "sozinha não poupa, e pode até aumentar o consumo, porque o grupo a "
                "recarrega — o que ela compra é a transferência instantânea e o pico que "
                "o grupo não aguenta."
            ))

    partes.append(nota(
        f"O grupo não é acionado em paralelo à rede porque o kWh dele "
        f"({_brl(grupo.custo_energia_brl_kwh, 2)}) sai mais caro que o da distribuidora "
        f"({_brl(fatura.tarifa_brl_kwh, 2)}). Ele existe para o apagão, e é lá que o "
        "combustível é contabilizado."
        if grupo.custo_energia_brl_kwh >= fatura.tarifa_brl_kwh else
        "O kWh do grupo sai mais barato que o da distribuidora, então ele também "
        "trabalha em paralelo à rede -- e o combustível desse regime está no custo "
        "operacional de cada cenário com gerador."
    ))
    return partes


def _secao_economia(estudo: ResultadoEstudo) -> str:
    if estudo.recomendado is None:
        return ""
    chave = estudo.recomendado.conjunto.descricao()
    economico = estudo.economia[chave]
    premissas = economico.premissas
    operacao = economico.operacao
    partes = [
        secao("Análise econômica"),
        (
            "O retorno do armazenamento vem de três fontes que disputam o mesmo estado "
            "de carga: arbitragem tarifária (carregar barato, descarregar na ponta), "
            "aumento do autoconsumo solar, e resiliência — a energia que deixa de "
            "faltar. As duas primeiras aparecem na conta de luz; a terceira, não."
            if estudo.com_solar else
            "Sem geração solar, o retorno do armazenamento vem de duas fontes: "
            "arbitragem tarifária (carregar barato, descarregar na ponta) e "
            "resiliência — a energia que deixa de faltar. A primeira aparece na conta "
            "de luz; a segunda, não, e em tarifa simples ela é a única que existe."
        ),
        tabela(
            ["Grandeza", "Valor"],
            [
                ("Investimento inicial", _brl(economico.capex_brl)),
                ("Energia deslocada da ponta",
                 _n(operacao.deslocamento_ponta_kwh, 0, "kWh/ano")),
                *([("Excedente solar convertido em autoconsumo",
                    _n(operacao.exportacao_convertida_kwh, 0, "kWh/ano"))]
                  if estudo.com_solar else []),
                ("Energia descarregada pelo banco",
                 _n(operacao.descarga_bateria_kwh, 0, "kWh/ano")),
                ("Ciclos equivalentes por ano", _n(operacao.ciclos_equivalentes, 0)),
                ("Economia tarifária no primeiro ano", _brl(economico.economia_ano1_brl)),
                ("Valor da resiliência no primeiro ano",
                 _brl(economico.valor_resiliencia_ano1_brl)),
                ("Valor presente líquido", _brl(economico.vpl_brl)),
                ("Taxa interna de retorno",
                 _pct(economico.tir, 1) if economico.tir is not None else "não converge"),
                ("Retorno do investimento",
                 _n(economico.payback_anos, 1, "anos") if economico.payback_anos
                 else "não se paga no horizonte"),
                ("Custo nivelado do armazenamento",
                 f"{_brl(economico.custo_nivelado_brl_kwh, 2)}/kWh"
                 if economico.custo_nivelado_brl_kwh else "—"),
            ],
            alinhamento="lr", largura_primeira_coluna="8cm",
            legenda="Indicadores econômicos do conjunto recomendado",
        ),
    ]
    if premissas is not None and premissas.custo_interrupcao_brl_kwh == 0:
        partes.append(caixa_aviso(
            "O custo da interrupção foi informado como zero, de modo que a resiliência "
            "não entrou no fluxo de caixa — apenas o ganho tarifário. Em cliente que "
            "perde produção, carga refrigerada ou faturamento durante a falta, esse "
            "costuma ser o maior dos três benefícios, e a análise acima o subestima."
        ))
    if premissas is not None:
        partes.append(tabela(
            ["Premissa", "Valor"],
            [
                ("Tarifa fora de ponta", f"{_brl(premissas.tarifa_fora_ponta_brl_kwh)}/kWh"),
                ("Tarifa de ponta", f"{_brl(premissas.tarifa_ponta_brl_kwh)}/kWh"),
                ("Valor da energia injetada", f"{_brl(premissas.valor_injecao_brl_kwh)}/kWh"),
                ("Demanda contratada",
                 f"{_brl(premissas.tarifa_demanda_brl_kw_mes, 2)}/kW·mês"),
                ("Custo da interrupção",
                 f"{_brl(premissas.custo_interrupcao_brl_kwh, 2)}/kWh"),
                ("Interrupções por ano", _n(premissas.interrupcoes_por_ano, 0)),
                ("Reserva de backup", _pct(premissas.reserva_backup_frac)),
                ("Taxa de desconto", _pct(premissas.taxa_desconto_ano, 1)),
                ("Escalada tarifária", _pct(premissas.escalada_tarifaria_ano, 1)),
                ("Horizonte de análise", _n(premissas.anos_analise, 0, "anos")),
            ],
            alinhamento="lr", largura_primeira_coluna="7cm",
            legenda="Premissas econômicas adotadas",
        ))
        partes.append(tabela_longa(
            ["Ano", "Retenção", "Benefício tarifário", "Resiliência", "OPEX", "Troca", "Fluxo"],
            [
                (
                    _n(linha.ano, 0), _pct(linha.retencao, 1),
                    _brl(linha.beneficio_tarifario_brl), _brl(linha.beneficio_resiliencia_brl),
                    _brl(-linha.opex_brl), _brl(-linha.troca_brl) if linha.troca_brl else "—",
                    _brl(linha.fluxo_brl),
                )
                for linha in economico.fluxo.itertuples()
            ],
            alinhamento="lrrrrrr", legenda="Fluxo de caixa do armazenamento",
        ))
    return "\n\n".join(p for p in partes if p)


def _natureza(fonte: str) -> str:
    """
    Classifica a procedência de uma linha de catálogo em três estados, não dois.

    "Tem a palavra conferir" não serve como teste: as linhas de datasheet da
    GoodWe carregam a ressalva de que ciclos de vida e rendimento de ida e
    volta **não** constam do documento, e essa ressalva contém a palavra. O
    que separa os casos é a presença de uma referência ao documento do
    fabricante; a ressalva, quando existe, vira o terceiro estado.
    """
    texto = fonte.lower()
    tem_documento = any(marca in texto for marca in ("datasheet", ".pdf", "fabricante"))
    tem_ressalva = "conferir" in texto
    if tem_documento and tem_ressalva:
        return "Datasheet, com ressalva"
    if tem_documento:
        return "Datasheet do fabricante"
    return "Ordem de grandeza de mercado"


def _origem(fonte: str) -> Raw:
    """
    A origem do dado com a URL marcada como URL, e a nota fora dela.

    Três armadilhas, todas já encontradas em documento entregue:

    1. Escapada como texto comum, a URL de datasheet da GoodWe -- 60 caracteres
       sem espaço onde quebrar -- atravessa a margem direita. Passada por
       ``\\url``, quebra em qualquer ponto.
    2. O endereço vai da primeira ocorrência de um domínio até o fim do texto,
       e não até o primeiro espaço: o nome do arquivo da GoodWe tem espaço no
       meio ("GW_ET PLUS+_Datasheet-EN.pdf"), e cortar ali entregaria meia URL.
       Mas ele para antes de ``|``, que é o separador das notas que o catálogo
       acrescenta à fonte -- sem isso, " | 3 MPPT" virava parte do endereço.
    3. Espaço vira ``%20``, e ``%`` em LaTeX comenta o resto da linha. Dentro
       de ``\\url`` ele precisa sair escapado como ``\\%``, senão o ``\\\\`` que
       fecha a linha da tabela some junto e o documento quebra sem erro.
    """
    texto = str(fonte).strip()
    # A nota do catálogo vem depois de "|" e não faz parte do endereço.
    endereco_bruto, _, nota = texto.partition("|")
    endereco_bruto, nota = endereco_bruto.strip(), nota.strip()

    achado = re.search(
        r"https?://|(?:[\w-]+\.)+(?:com|net|org|io|br)(?:\.[a-z]{2})?/", endereco_bruto
    )
    if not achado:
        return Raw(esc(texto[:110]))

    endereco = endereco_bruto[achado.start():].strip().replace(" ", "%20")
    # O escape do percentual vem depois da codificação, nunca antes.
    endereco = endereco.replace("%", r"\%")
    descricao = endereco_bruto[: achado.start()].strip(" \u2014-|,;")

    partes = []
    if descricao:
        partes.append(esc(descricao[:70]))
    partes.append(rf"\url{{{endereco}}}")
    if nota:
        partes.append(esc(nota[:70]))
    return Raw(r" \newline ".join(partes))


def _janela_legivel(equipamento) -> str:
    """
    A janela de uso em horas do relógio, não em minutos desde a meia-noite.

    O núcleo do D² guarda os intervalos em minutos porque é assim que ele
    simula. Publicar "1080 a 1380" num anexo obriga o leitor a dividir por 60
    para conferir se o horário do chuveiro faz sentido -- que é exatamente a
    conferência que o anexo existe para permitir.
    """
    intervalos = equipamento.intervalos
    if callable(intervalos):
        return "sorteada a cada dia"
    try:
        trechos = [
            f"{int(inicio) // 60:02d}:{int(inicio) % 60:02d}–"
            f"{int(fim) // 60:02d}:{int(fim) % 60:02d}"
            for inicio, fim in intervalos
        ]
    except (TypeError, ValueError):
        return "—"
    return ", ".join(trechos) if trechos else "—"


def _anexo_cargas(estudo: ResultadoEstudo) -> str:
    """
    Todo equipamento que entrou na simulação, ambiente por ambiente.

    Vai no fim, como anexo, e não no meio da análise: quem lê a proposta quer
    a conclusão, e quem confere o estudo quer a lista. As duas necessidades
    são reais e não cabem na mesma página.

    A lista é o que torna o estudo auditável. Um pico de 60 kW não se discute
    -- ou se aceita, ou não. Uma linha dizendo "chuveiro elétrico, 5.500 W, 40
    unidades, 90% de probabilidade, das 06:00 às 09:00" se discute com o
    cliente, e é assim que o levantamento melhora.
    """
    comodos = estudo.configuracao.comodos
    if not comodos:
        return ""
    instancias = estudo.configuracao.instancias_por_comodo or {}
    essenciais = {n.strip().lower() for n in (estudo.configuracao.comodos_essenciais or [])}

    partes = [
        secao("Anexo — o levantamento de cargas"),
        "Esta é a lista completa do que entrou na simulação. Cada linha é um "
        "equipamento com a potência que ele puxa ligado, quantos existem no ambiente, "
        "com que probabilidade ele é usado num dia qualquer, o fator de demanda "
        "aplicado e a janela em que pode ligar. É o insumo do método de Monte Carlo "
        "descrito na seção da demanda elétrica, e é por onde se confere o estudo.",
    ]

    total_instalada_w = 0.0
    resumo: list[tuple[Any, ...]] = []
    detalhe: list[str] = []

    for comodo in comodos:
        n = int(instancias.get(comodo.nome, 1))
        potencia_comodo = sum(
            float(eq.potencia) * int(eq.quantidade) for eq in comodo.equipamentos
        )
        total_instalada_w += potencia_comodo * n
        resumo.append((
            comodo.nome,
            _n(n, 0),
            _n(len(comodo.equipamentos), 0),
            _n(potencia_comodo / 1000.0, 2, "kW"),
            _n(potencia_comodo * n / 1000.0, 2, "kW"),
            "sim" if comodo.nome.strip().lower() in essenciais else "não",
        ))

        titulo = comodo.nome if n == 1 else f"{comodo.nome} (×{n})"
        detalhe.append(secao(titulo, nivel=2))
        detalhe.append(tabela_longa(
            ["Equipamento", "Potência", "Qtd.", "Prob.", "F. dem.", "Janela de uso"],
            [
                (
                    eq.nome,
                    _n(eq.potencia, 0, "W"),
                    _n(eq.quantidade, 0),
                    _pct(eq.probabilidade, 0),
                    _n(eq.fator_demanda, 2),
                    _janela_legivel(eq),
                )
                for eq in comodo.equipamentos
            ],
            alinhamento="p{5.4cm}rrrrp{3.6cm}",
            legenda=f"Equipamentos considerados em {comodo.nome}",
        ))

    partes.append(tabela_longa(
        ["Ambiente", "Instâncias", "Equip.", "Por instância", "Total", "No backup"],
        resumo,
        alinhamento="p{4.4cm}rrrrl",
        legenda="Resumo do levantamento por ambiente",
    ))
    partes.append(
        "A potência instalada somada é de "
        f"{_n(total_instalada_w / 1000.0, 1, 'kW')}. A demanda máxima simulada fica bem "
        "abaixo disso, e a razão entre as duas é o fator de demanda apresentado na "
        "análise da demanda elétrica: equipamento instalado não é equipamento ligado."
    )
    partes.extend(detalhe)
    partes.append(nota(
        "Probabilidade é a chance de o equipamento ser usado num dia qualquer; fator de "
        "demanda é a fração da potência de placa efetivamente puxada quando ele está "
        "ligado. Os dois multiplicam a potência, e confundi-los é o erro mais comum de "
        "levantamento: um chuveiro de 5.500 W com 90% de probabilidade e fator 1,0 puxa "
        "5.500 W em 9 de cada 10 dias, e não 4.950 W todo dia."
    ))
    return "\n\n".join(partes)


def _secao_procedencia(estudo: ResultadoEstudo) -> str:
    """
    De onde veio cada número — a seção que separa datasheet de estimativa.

    Um estudo que mistura os dois e não avisa transfere ao leitor um risco que
    ele não tem como avaliar.
    """
    from .catalogo import carregar_catalogo

    partes = [secao("Premissas, limitações e procedência dos dados")]

    linhas: list[tuple[str, str, Any]] = []
    try:
        catalogo = carregar_catalogo(estudo.configuracao.catalogo)
    except Exception:  # noqa: BLE001 — a seção é informativa, não pode derrubar o documento
        catalogo = None

    usados: set[str] = set()
    for resultado in estudo.resiliencia:
        usados.add(resultado.conjunto.inversor.modelo)
        usados.add(resultado.conjunto.bateria.modelo)
    if catalogo is not None:
        for item in [*catalogo.inversores, *catalogo.baterias]:
            if item.modelo in usados:
                linhas.append((
                    f"{item.fabricante} {item.modelo}",
                    _natureza(str(item.fonte_dado)),
                    _origem(str(item.fonte_dado)),
                ))

    if estudo.configuracao.layout is not None:
        modulo = estudo.configuracao.layout.modulo
        linhas.append((
            f"{modulo.fabricante} {modulo.modelo}",
            "Datasheet do fabricante" if not modulo.dimensoes_derivadas else "Parcialmente derivado",
            "Catálogo BDFotovoltaica.xlsx",
        ))

    if linhas:
        partes.append(tabela_longa(
            ["Equipamento", "Natureza do dado", "Origem"], linhas,
            alinhamento="p{5cm}p{3.6cm}p{6.5cm}",
            legenda="Procedência dos dados de equipamento usados neste estudo",
        ))

    # Preço tem data; datasheet não. A distinção precisa estar escrita, senão
    # o leitor trata "custa R$ 18.000" com a mesma confiança que "entrega
    # 24 kVA por 10 s".
    from ..precos import PRECOS, procedencia

    partes.append(secao("De quando são os preços", nivel=2))
    partes.append(procedencia())
    partes.append(tabela(
        ["Item", "Faixa de mercado", "Adotado"],
        [
            (
                {"modulo_brl_wp": "Módulo fotovoltaico",
                 "inversor_hibrido_brl_w": "Inversor híbrido",
                 "bateria_lfp_brl_kwh": "Bateria de lítio (LFP)",
                 "gerador_diesel_brl_kwh": "Energia de grupo gerador"}.get(chave, chave),
                f"{faixa.minimo:,.2f} a {faixa.maximo:,.2f} {faixa.unidade}".replace(",", "."),
                f"{faixa.tipico:,.2f} {faixa.unidade}".replace(",", "."),
            )
            for chave, faixa in PRECOS.items()
        ],
        alinhamento="lrr", largura_primeira_coluna="6cm",
        legenda="Faixas de preço de referência e o valor adotado",
    ))
    partes.append(nota(
        "A distância entre o mínimo e o máximo de cada faixa é a informação mais "
        "honesta desta tabela: onde ela é larga, o valor do meio não merece a "
        "confiança que um número único aparenta ter. Equipamento com cotação "
        "cadastrada no catálogo usa o preço cotado, e não estas faixas."
    ))

    partes.append(secao("O que este estudo não afirma", nivel=2))
    partes.append(lista([
        "Não substitui projeto elétrico executivo nem memorial de cálculo assinado.",
        "Não dimensiona proteções, condutores, aterramento ou sistema contra descargas "
        "atmosféricas.",
        "Não avalia a capacidade estrutural da cobertura para receber o peso dos módulos.",
        "Não considera sombreamento por edificações ou vegetação vizinhas — apenas o "
        "sombreamento entre fileiras, no caso de estrutura inclinada.",
        "Não modela o efeito da temperatura sobre a degradação da bateria.",
        "Não inclui receitas de mercado livre, resposta da demanda ou serviços "
        "ancilares, por não estarem acessíveis à geração distribuída no Brasil.",
    ]))

    if estudo.avisos:
        partes.append(secao("Ressalvas registradas durante o cálculo", nivel=2))
        partes.append(lista(estudo.avisos))
    return "\n\n".join(p for p in partes if p)


# ----------------------------------------------------------------------------
# Montagem
# ----------------------------------------------------------------------------
def montar_documento(
    estudo: ResultadoEstudo,
    figuras: dict[str, Path] | None = None,
    capa: DadosCapa | None = None,
) -> str:
    """Devolve o documento LaTeX completo, do ``\\documentclass`` ao ``\\end``."""
    figuras = figuras or {}
    capa = capa or DadosCapa(referencia=estudo.configuracao.nome)
    # Sem sol no estudo, as três seções fotovoltaicas somem inteiras. Deixar a
    # curva de geração num estudo de cliente que não vai instalar painel faz
    # quem lê entender que ela faz parte da proposta.
    corpo = [
        _capa(estudo, capa),
        _sumario_executivo(estudo),
        _secao_telhado(estudo, figuras) if estudo.com_solar else "",
        _secao_solar(estudo, figuras) if estudo.com_solar else "",
        _secao_demanda(estudo, figuras),
        _secao_fotovoltaico(estudo) if estudo.com_solar else "",
        _secao_armazenamento(estudo, figuras),
        _secao_vida_util(estudo, figuras),
        _secao_cenarios(estudo, figuras),
        _secao_economia(estudo),
        _secao_procedencia(estudo),
        _anexo_cargas(estudo),
    ]
    return "\n\n".join([
        PREAMBULO,
        f"\\newcommand{{\\cabecalhoempresa}}{{{esc(capa.empresa)}}}",
        f"\\newcommand{{\\cabecalhoreferencia}}{{{esc(capa.referencia)}}}",
        r"\begin{document}",
        "\n\n".join(p for p in corpo if p),
        r"\end{document}",
        "",
    ])


def escrever_dossie(
    estudo: ResultadoEstudo,
    destino: str | Path,
    capa: DadosCapa | None = None,
    compilar: bool = True,
) -> dict[str, Path]:
    """
    Grava o dossiê completo: ``.tex``, figuras, planilhas e — se houver LaTeX — o PDF.

    As figuras vão para ``figuras/`` e as tabelas para ``tabelas/``, que é a
    organização que o ``.tex`` referencia. Compilar no Overleaf exige só subir
    a pasta inteira.
    """
    from .relatorio import escrever_relatorio

    pasta = Path(destino)
    pasta.mkdir(parents=True, exist_ok=True)
    figuras_dir = pasta / "figuras"
    tabelas_dir = pasta / "tabelas"
    figuras_dir.mkdir(exist_ok=True)
    tabelas_dir.mkdir(exist_ok=True)

    escritos = escrever_relatorio(estudo, figuras_dir, com_graficos=True, com_latex=False)
    figuras = {k.removeprefix("figura_"): v for k, v in escritos.items() if k.startswith("figura_")}
    for chave, caminho in list(escritos.items()):
        if chave.startswith("csv_"):
            caminho.replace(tabelas_dir / caminho.name)
            escritos[chave] = tabelas_dir / caminho.name
    # O estudo.md do relatório rápido fica na raiz, ao lado do documento.
    if "markdown" in escritos:
        escritos["markdown"] = Path(escritos["markdown"]).replace(pasta / "resumo.md")

    caminho_tex = pasta / "estudo.tex"
    caminho_tex.write_text(montar_documento(estudo, figuras, capa), encoding="utf-8")
    resultado: dict[str, Path] = {"tex": caminho_tex, **escritos}

    if compilar and encontrar_compilador():
        pdf = compilar_pdf(caminho_tex)
        if pdf is not None:
            resultado["pdf"] = pdf
    return resultado


def zip_do_dossie(
    estudo: ResultadoEstudo,
    capa: DadosCapa | None = None,
    compilar: bool = True,
) -> bytes:
    """
    O dossiê inteiro num ZIP em memória, pronto para download.

    Compila o PDF quando há LaTeX na máquina; quando não há, o ZIP sai com o
    ``.tex`` e as figuras, que compilam no Overleaf sem ajuste nenhum.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as temporario:
        pasta = Path(temporario) / "estudo"
        escrever_dossie(estudo, pasta, capa, compilar=compilar)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("LEIA-ME.txt", _leia_me(estudo))
            for arquivo in sorted(pasta.rglob("*")):
                if arquivo.is_file() and arquivo.suffix not in {".aux", ".log", ".out", ".toc"}:
                    zf.write(arquivo, arquivo.relative_to(pasta.parent))
        return buffer.getvalue()


def _leia_me(estudo: ResultadoEstudo) -> str:
    cfg = estudo.configuracao
    return "\n".join([
        f"ESTUDO DE ENERGIA — {cfg.nome}",
        "=" * 60,
        "",
        "estudo/estudo.tex     documento completo em LaTeX",
        "estudo/estudo.pdf     o mesmo documento compilado (quando havia LaTeX na máquina)",
        "estudo/resumo.md      resumo em Markdown, para colar em e-mail",
        "estudo/figuras/       todas as figuras em PNG",
        "estudo/tabelas/       todas as tabelas em CSV (separador ';', decimal ',')",
        "",
        "Para compilar sem LaTeX instalado: suba a pasta 'estudo' inteira no",
        "Overleaf (overleaf.com) e compile estudo.tex. Nenhum ajuste é necessário.",
        "",
        "Os dados de equipamento marcados 'a conferir' nos catálogos são ordens de",
        "grandeza de mercado, não datasheets. A seção de procedência do documento",
        "lista quais são quais.",
    ])
