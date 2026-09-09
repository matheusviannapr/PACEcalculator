"""
Saída do estudo: gráficos, planilhas e o relatório em Markdown e LaTeX.

Cinco figuras, escolhidas porque cada uma responde uma pergunta que uma tabela
responde pior:

1. **Curva de excedência de pico**, com os limites de cada inversor marcados —
   a leitura direta de "este inversor não dá conta em X% dos dias".
2. **Mapa de atendimento** (hora de início × duração) — mostra de um golpe que
   o problema não é a bateria, é a hora em que a luz cai.
3. **Trajetória do estado de carga** num apagão longo, com percentis — o
   gráfico que explica *por que* falhou, e quando.
4. **Carga e geração no dia médio**, por estação — a sobreposição que justifica
   a bateria existir.
5. **Degradação e autonomia ao longo dos anos** — a promessa com data.

O relatório em Markdown é a entrega interna; o ``.tex`` é uma seção pronta para
entrar na proposta que o :mod:`aurum.proposal` já gera, com o mesmo escape de
caracteres para não quebrar a compilação.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # sem display: o relatório roda em servidor e em CI
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ..proposal.latex import esc as _escapar_latex  # noqa: E402
from .. import marca  # noqa: E402
from .estudo import ResultadoEstudo  # noqa: E402
from .fontes import estudo_do_gerador  # noqa: E402

LOGGER = logging.getLogger(__name__)

__all__ = ["escrever_relatorio", "gerar_graficos"]

# A ordem das séries vem da marca: preto para a curva principal, âmbar para o
# destaque, e as de apoio depois. Trocar a paleta em `aurum.marca` repinta
# todas as figuras do dossiê de uma vez.
_CORES = marca.CORES_GRAFICO
#: Atalho para as cores que têm significado, e não posição numa paleta.
_SEM = marca.SEMANTICA


def _milhar(valor: float, casas: int = 0) -> str:
    """
    Número com ponto de milhar, sem tocar no resto da frase.

    Aplicar ``.replace(",", ".")`` na frase inteira é o atalho óbvio e come as
    vírgulas do texto: "1.003 m², somando" virava "1.003 m². somando".
    """
    return f"{valor:,.{casas}f}".replace(",", ".")


def _figura(largura: float = 9.0, altura: float = 5.0):
    fig, eixo = plt.subplots(figsize=(largura, altura), dpi=140)
    eixo.grid(alpha=0.25, linewidth=0.6)
    eixo.set_axisbelow(True)
    return fig, eixo


def _salvar(fig, destino: Path, nome: str) -> Path:
    caminho = destino / f"{nome}.png"
    fig.savefig(caminho, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return caminho


# ----------------------------------------------------------------------------
def gerar_graficos(estudo: ResultadoEstudo, destino: Path) -> dict[str, Path]:
    """Grava as figuras em ``destino`` e devolve o mapa nome → caminho."""
    destino.mkdir(parents=True, exist_ok=True)
    figuras: dict[str, Path] = {}

    if (
        estudo.com_solar
        and estudo.configuracao.telhado is not None
        and estudo.configuracao.layout is not None
    ):
        figuras["telhado"] = _grafico_telhado(estudo, destino)
        foto = _foto_do_telhado(estudo, destino)
        if foto is not None:
            figuras["foto_telhado"] = foto
    figuras["excedencia"] = _grafico_excedencia(estudo, destino)
    figuras["dia_medio"] = _grafico_dia_medio(estudo, destino)
    analise = estudo.configuracao.analise
    if analise is not None:
        figuras["histograma_picos"] = _grafico_histograma(analise, destino)
        figuras["duracao_carga"] = _grafico_duracao(analise, destino)
        figuras["fator_carga_hora"] = _grafico_fator_horario(analise, destino)
        if analise.composicao is not None and not analise.composicao.por_equipamento.empty:
            figuras["composicao_pico"] = _grafico_composicao(analise, destino)
        if analise.perfis_por_comodo is not None and not analise.perfis_por_comodo.empty:
            figuras["por_ambiente"] = _grafico_por_ambiente(analise, destino)
    if estudo.resiliencia:
        alvo = estudo.recomendado or estudo.resiliencia[-1]
        figuras["mapa_atendimento"] = _grafico_mapa(alvo, destino)
        figuras["soc"] = _grafico_soc(estudo, alvo, destino)
        figuras["fronteira"] = _grafico_fronteira(estudo, destino)
        chave = alvo.conjunto.descricao()
        if chave in estudo.degradacao:
            figuras["degradacao"] = _grafico_degradacao(estudo, chave, destino)
    ocupacao = getattr(estudo.configuracao, "ocupacao", None) or {}
    if ocupacao.get("curvas"):
        figuras["perfis_ocupacao"] = _grafico_perfis_ocupacao(estudo, destino)
    uso = getattr(estudo, "uso", None)
    if uso is not None and len(getattr(uso, "cenarios", ())) > 1:
        figuras["cenarios_de_uso"] = _grafico_cenarios_de_uso(uso, destino)
    escopos = getattr(estudo, "escopos", None)
    if escopos is not None and escopos.tem_preferiveis:
        figuras["escopos_backup"] = _grafico_escopos(escopos, destino)
    if estudo.cenarios is not None and len(estudo.cenarios.cenarios) > 1:
        figuras["cenarios"] = _grafico_cenarios(estudo.cenarios, destino)
        custo_grupo = estudo_do_gerador(estudo.cenarios)
        if not custo_grupo.empty:
            figuras["gerador"] = _grafico_gerador(estudo.cenarios, custo_grupo, destino)
    return figuras


def _foto_do_telhado(estudo: ResultadoEstudo, destino: Path) -> Path | None:
    """
    A imagem de satélite do telhado com os módulos desenhados por cima.

    É a figura que o cliente entende sem explicação: ele reconhece o próprio
    prédio e vê onde os painéis vão. Depende de rede — sem ela, devolve
    ``None`` e o relatório segue com o croqui, que responde à mesma pergunta
    técnica sem depender de ninguém.
    """
    from ..pv.imagem import baixar_imagem_aerea
    from ..pv.telhado import croqui_geojson

    telhado = estudo.configuracao.telhado
    layout = estudo.configuracao.layout
    contorno = telhado.poligono_wgs84
    imagem = baixar_imagem_aerea(contorno.bounds)
    if imagem is None:
        return None

    fig, eixo = plt.subplots(figsize=(9.0, 6.4), dpi=150)
    try:
        eixo.imshow(imagem.abrir(), extent=imagem.extent, origin="upper", zorder=0)
    except Exception as exc:  # noqa: BLE001 — imagem corrompida não derruba o relatório
        LOGGER.info("Não consegui desenhar a imagem aérea: %s", exc)
        plt.close(fig)
        return None

    x, y = contorno.exterior.xy
    eixo.plot(x, y, color="#ffd166", lw=2.4, zorder=2, label="telhado marcado")
    for feature in croqui_geojson(layout)["features"]:
        anel = feature["geometry"]["coordinates"][0]
        eixo.fill(
            [p[0] for p in anel], [p[1] for p in anel],
            facecolor="#1d4ed8", alpha=0.72, edgecolor="white", linewidth=0.25, zorder=3,
        )

    eixo.set_xlim(imagem.extent[0], imagem.extent[1])
    eixo.set_ylim(imagem.extent[2], imagem.extent[3])
    # Compensa o encolhimento do grau de longitude com a latitude: sem isso, o
    # telhado aparece esticado e os módulos saem fora do contorno.
    eixo.set_aspect(1.0 / max(0.05, np.cos(np.radians(telhado.latitude))))
    eixo.set_xticks([])
    eixo.set_yticks([])
    eixo.set_title(
        f"{telhado.nome} — {layout.quantidade} módulos de "
        f"{layout.modulo.potencia_wp:.0f} Wp ({layout.potencia_kwp:.1f} kWp)"
    )
    eixo.text(
        0.005, -0.03, f"{imagem.fonte}. Referência visual, não levantamento topográfico.",
        transform=eixo.transAxes, fontsize=7, color="#555555", va="top",
    )
    eixo.legend(fontsize=8, loc="upper right", framealpha=0.85)
    return _salvar(fig, destino, "foto_telhado")


def _grafico_telhado(estudo: ResultadoEstudo, destino: Path) -> Path:
    """
    Croqui do telhado marcado com os módulos que couberam.

    O desenho vale mais que a contagem: um leitor confere de relance se a
    orientação faz sentido, se sobrou faixa livre onde não devia, se o recuo
    de borda comeu área demais. Nenhuma dessas três coisas salta de uma tabela.
    """
    telhado = estudo.configuracao.telhado
    layout = estudo.configuracao.layout

    fig, eixo = plt.subplots(figsize=(8.0, 6.0), dpi=140)
    contorno = telhado.poligono_utm
    x, y = contorno.exterior.xy
    # Coordenadas relativas ao canto do contorno: UTM absoluto tem sete dígitos
    # e polui os eixos sem informar nada.
    x0, y0 = min(x), min(y)
    eixo.fill(
        [v - x0 for v in x], [v - y0 for v in y],
        facecolor="#f5f5f5", edgecolor=_CORES[1], linewidth=2.0, zorder=1,
        label=f"telhado — {_milhar(telhado.area_m2)} m²",
    )
    for poligono in layout.modulos_geom:
        px, py = poligono.exterior.xy
        eixo.fill(
            [v - x0 for v in px], [v - y0 for v in py],
            facecolor=_CORES[0], edgecolor="white", linewidth=0.3, zorder=2,
        )

    # Seta do azimute: para onde a face aponta, desenhada no plano do mapa.
    largura = max(x) - x0
    altura = max(y) - y0
    comprimento = 0.18 * max(largura, altura)
    angulo = np.radians(telhado.azimute_deg)
    ponta = (largura * 0.5 + comprimento * np.sin(angulo),
             altura * 0.5 + comprimento * np.cos(angulo))
    eixo.annotate(
        "", xytext=(largura * 0.5, altura * 0.5), xy=ponta,
        arrowprops=dict(arrowstyle="-|>", color=_CORES[4], lw=2.2), zorder=4,
    )
    eixo.text(
        ponta[0], ponta[1], f" {telhado.orientacao}", color=_CORES[4],
        fontsize=8.5, fontweight="bold", va="center", ha="left", zorder=4,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="none"),
    )
    eixo.text(
        0.02, 0.98,
        f"{layout.quantidade} × {layout.modulo.potencia_wp:.0f} Wp = "
        f"{layout.potencia_kwp:.1f} kWp\n"
        f"face {telhado.orientacao} ({telhado.azimute_deg:.0f}°), "
        f"inclinação {telhado.inclinacao_deg:.0f}°\n"
        f"ocupação {layout.taxa_ocupacao:.0%} · {layout.densidade_wp_m2:.0f} Wp/m²",
        transform=eixo.transAxes, va="top", ha="left", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85, edgecolor="#cccccc"),
    )
    eixo.set_aspect("equal")
    eixo.set_xlabel("metros")
    eixo.set_ylabel("metros")
    eixo.set_title(f"Croqui do telhado — {telhado.nome}")
    # A legenda sai do desenho: dentro dela cobre módulos, e num croqui o que
    # está coberto é exatamente o que o leitor precisa conferir.
    eixo.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0.0, -0.08))
    return _salvar(fig, destino, "croqui_telhado")



# ----------------------------------------------------------------------------
# Análise da demanda
# ----------------------------------------------------------------------------
def _grafico_histograma(analise, destino: Path) -> Path:
    """Distribuição dos picos diários, com a média e o P95 marcados."""
    fig, eixo = _figura(9.0, 4.8)
    picos = analise.ensemble.picos_diarios_w() / 1000.0
    eixo.hist(picos, bins=30, color=_CORES[0], alpha=0.80, edgecolor="white", linewidth=0.5)
    media = analise.estatisticas.media / 1000.0
    p95 = analise.estatisticas.percentis[95] / 1000.0
    eixo.axvline(media, color=_CORES[1], ls="--", lw=2.0, label=f"média {media:.1f} kW")
    eixo.axvline(p95, color=_CORES[4], ls="--", lw=2.0, label=f"P95 {p95:.1f} kW")
    eixo.set_xlabel("Pico diário (kW)")
    eixo.set_ylabel("Dias simulados")
    eixo.set_title("Distribuição dos picos de demanda")
    eixo.legend(fontsize=8, frameon=False)
    return _salvar(fig, destino, "histograma_picos")


def _grafico_duracao(analise, destino: Path) -> Path:
    """Curva de duração de carga: a área é energia, o joelho é o pico raro."""
    fig, eixo = _figura(9.0, 4.8)
    for i, estacao in enumerate(analise.ensemble.estacoes):
        fracao, valores = analise.curva_de_duracao(estacao)
        amostra = np.linspace(0, len(valores) - 1, 600).astype(int)
        eixo.plot(fracao[amostra] * 100, valores[amostra] / 1000.0,
                  color=_CORES[i % len(_CORES)], lw=1.2, label=estacao)
    fracao, valores = analise.curva_de_duracao()
    amostra = np.linspace(0, len(valores) - 1, 600).astype(int)
    eixo.plot(fracao[amostra] * 100, valores[amostra] / 1000.0,
              color="black", lw=2.0, label="todas as estações")
    eixo.set_xlabel("Fração do tempo acima da potência (%)")
    eixo.set_ylabel("Potência (kW)")
    eixo.set_title("Curva de duração de carga")
    eixo.legend(fontsize=8, frameon=False)
    return _salvar(fig, destino, "duracao_carga")


def _grafico_fator_horario(analise, destino: Path) -> Path:
    fig, eixo = _figura(9.0, 4.0)
    valores = analise.fator_de_carga_horario()
    cores = [_SEM["bom"] if v > 0.6 else (_SEM["geracao"] if v > 0.35 else _SEM["alerta"])
             for v in valores]
    eixo.bar(np.arange(24), valores, color=cores, alpha=0.85)
    eixo.set_xticks(range(0, 24, 2))
    eixo.set_xlabel("Hora do dia")
    eixo.set_ylabel("Fator de carga")
    eixo.set_ylim(0, 1)
    eixo.set_title("Fator de carga hora a hora — o que é carga firme e o que é coincidência")
    return _salvar(fig, destino, "fator_carga_hora")


def _grafico_composicao(analise, destino: Path) -> Path:
    """Quem estava ligado no instante do pico, do maior para o menor."""
    fig, eixo = _figura(9.0, 5.0)
    tabela = analise.composicao.por_equipamento.head(10).iloc[::-1]
    rotulos = [
        f"{linha.equipamento[:34]}\n({linha.comodo[:22]})" for linha in tabela.itertuples()
    ]
    eixo.barh(rotulos, tabela["carga_media_kw"], color=_CORES[0], alpha=0.85)
    for i, (valor, fatia) in enumerate(zip(tabela["carga_media_kw"], tabela["participacao"])):
        eixo.text(valor, i, f"  {valor:.1f} kW ({fatia:.0%})", va="center", fontsize=7.5)
    eixo.set_xlabel("Contribuição média no instante de pico (kW)")
    eixo.set_title(
        f"Quem causa o pico — mais provável às {analise.composicao.hora_mais_provavel}h")
    eixo.tick_params(axis="y", labelsize=7)
    eixo.set_xlim(0, tabela["carga_media_kw"].max() * 1.35)
    return _salvar(fig, destino, "composicao_pico")


def _grafico_por_ambiente(analise, destino: Path) -> Path:
    fig, eixo = _figura(9.0, 4.8)
    tabela = analise.perfis_por_comodo.iloc[::10]
    eixo.stackplot(
        tabela.index, *[tabela[c].to_numpy() for c in tabela.columns],
        labels=list(tabela.columns),
        colors=[_CORES[i % len(_CORES)] for i in range(len(tabela.columns))], alpha=0.85)
    eixo.set_xlim(0, 24)
    eixo.set_xticks(range(0, 25, 3))
    eixo.set_xlabel("Hora do dia")
    eixo.set_ylabel("Potência (kW)")
    eixo.set_title("Composição da demanda por ambiente, ao longo do dia")
    eixo.legend(fontsize=7.5, loc="upper left", frameon=False)
    return _salvar(fig, destino, "por_ambiente")


def _grafico_excedencia(estudo: ResultadoEstudo, destino: Path) -> Path:
    fig, eixo = _figura()
    ensemble = estudo.ensemble_backup
    for i, estacao in enumerate(ensemble.estacoes):
        x, p = ensemble.curva_excedencia_pico(estacao).pontos()
        eixo.plot(x / 1000.0, p * 100.0, color=_CORES[i % len(_CORES)], lw=1.4, label=estacao)
    curva = ensemble.curva_excedencia_pico()
    x, p = curva.pontos()
    eixo.plot(x / 1000.0, p * 100.0, color="black", lw=2.2, label="todas as estações")

    # A escala tem que caber a curva. Um catálogo com inversor de 12 kW numa
    # carga de 3 kW espremeria a curva inteira no canto esquerdo, que é onde
    # está toda a informação — os inversores fora da faixa viram uma nota.
    pico_max = float(curva.amostras[-1]) / 1000.0
    pico_min = float(curva.amostras[0]) / 1000.0
    limite_x = pico_max * 1.35

    # Caso degenerado: quando o quadro de backup só tem carga fixa de 24 h — um
    # rack de servidores, CFTV, iluminação de emergência — todos os dias têm
    # exatamente o mesmo pico. A curva de excedência vira um degrau vertical, e
    # desenhá-la na escala de sempre produz uma figura vazia. Aqui ela ganha
    # uma escala própria e a explicação do porquê.
    constante = (pico_max - pico_min) < max(0.02, 0.01 * pico_max)
    if constante:
        meia_faixa = max(0.15 * pico_max, 0.2)
        eixo.set_xlim(max(0.0, pico_min - meia_faixa), pico_max + meia_faixa)
        eixo.axvline(pico_max, color="black", lw=2.2)
        eixo.text(
            0.5, 0.55,
            f"A carga de backup é praticamente constante: todos os dias simulados\n"
            f"têm o mesmo pico, de {pico_max:.2f} kW. A curva de excedência degenera\n"
            "num degrau porque não há variabilidade a distribuir — o que é esperado\n"
            "num quadro composto só de cargas fixas de 24 horas.",
            transform=eixo.transAxes, ha="center", va="center", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9,
                      edgecolor="#cccccc"),
        )

    # Uma linha por potência nominal, e **sem nome de equipamento**: o estudo
    # discute a característica nominal, não a marca. Vários modelos do
    # catálogo compartilham a mesma potência, e é a potência que decide se o
    # inversor aguenta — o nome de quem a fabrica não muda a curva. As
    # referências comerciais vão no anexo, no fim do documento.
    por_potencia: dict[float, int] = {}
    for linha in estudo.diagnostico_inversores.itertuples():
        chave = round(float(linha.nominal_kw), 1)
        por_potencia[chave] = por_potencia.get(chave, 0) + 1

    fora_da_faixa: list[str] = []
    for i, (potencia, quantos) in enumerate(sorted(por_potencia.items())):
        if potencia > limite_x:
            fora_da_faixa.append(f"{potencia:g} kW")
            continue
        cor = _CORES[i % len(_CORES)]
        # A contagem vai junto porque diz que há alternativa naquela potência:
        # "3 opções de 5 kW" é informação de compra, e nenhuma marca aparece.
        eixo.axvline(
            potencia, color=cor, ls="--", lw=1.0, alpha=0.85,
            label=f"{potencia:g} kW nominal"
            + (f" — {quantos} opções" if quantos > 1 else ""),
        )

    if fora_da_faixa:
        eixo.text(
            0.99, 0.02,
            "acima da faixa do gráfico: " + ", ".join(dict.fromkeys(fora_da_faixa)),
            transform=eixo.transAxes, ha="right", va="bottom", fontsize=6.5, color="#555555",
        )

    # No caso degenerado a escala já foi fixada em torno do degrau; sobrescrevê-la
    # com o zero à esquerda esconderia de novo a única informação da figura.
    if not constante:
        eixo.set_xlim(0, limite_x)
    eixo.set_xlabel("Potência de pico diário (kW)")
    eixo.set_ylabel("Probabilidade de excedência (%)")
    eixo.set_title("Excedência de pico da carga de backup, com os limites de cada inversor")
    eixo.set_ylim(0, 100)
    eixo.legend(fontsize=7, frameon=False, loc="upper right", ncol=1)
    return _salvar(fig, destino, "excedencia_pico")


def _grafico_dia_medio(estudo: ResultadoEstudo, destino: Path) -> Path:
    """
    Carga total, carga de backup e — quando há — geração no mesmo par de eixos.

    Mostrar só a carga de backup engana quem acabou de ver a curva da
    instalação inteira na análise de consumo: um quadro de backup composto só
    de servidores e CFTV é uma reta perfeita, e a reta sozinha parece defeito
    de simulação. Com as duas curvas juntas, ela vira informação — é a
    diferença entre elas que justifica o recorte de circuitos essenciais.

    Num estudo sem energia solar a faixa de geração não aparece. Desenhá-la em
    cinza, ou com legenda dizendo "se houvesse", seria pior que omitir: quem lê
    uma proposta entende que o que está no gráfico faz parte dela.
    """
    fig, eixo = _figura(9.0, 5.2)
    total = estudo.ensemble_total
    backup = estudo.ensemble_backup
    horas_total = np.arange(total.passos_por_dia) * total.passo_min / 60.0
    horas_backup = np.arange(backup.passos_por_dia) * backup.passo_min / 60.0

    eixo.plot(horas_total, total.perfil_medio_w() / 1000.0,
              color=_SEM["carga"], lw=2.2, label="carga total da instalação")
    eixo.plot(horas_backup, backup.perfil_medio_w() / 1000.0,
              color=_SEM["backup"], lw=2.0, ls="--", label="carga do quadro de backup")
    for i, estacao in enumerate(total.estacoes):
        eixo.plot(horas_total, total.perfil_medio_w(estacao) / 1000.0,
                  color=_SEM["carga"], lw=0.7, alpha=0.35,
                  label="carga total, por estação" if i == 0 else None)

    if estudo.com_solar:
        geracao = np.mean(
            [estudo.serie.janela_media_por_hora(e) for e in total.estacoes], axis=0
        ) * estudo.potencia_fv_kwp / 1000.0
        eixo.fill_between(np.arange(24) + 0.5, 0, geracao, color=_SEM["geracao"], alpha=0.30,
                          label=f"geração média de {estudo.potencia_fv_kwp:.0f} kWp")
        for i, estacao in enumerate(total.estacoes):
            eixo.plot(np.arange(24) + 0.5,
                      estudo.serie.janela_media_por_hora(estacao) * estudo.potencia_fv_kwp / 1000.0,
                      color=_SEM["geracao"], lw=0.9, ls=":", alpha=0.85,
                      label="geração, por estação" if i == 0 else None)

    eixo.set_xlabel("Hora do dia")
    eixo.set_ylabel("Potência (kW)")
    eixo.set_xlim(0, 24)
    eixo.set_ylim(bottom=0)
    eixo.set_xticks(range(0, 25, 3))
    eixo.set_title(
        "Dia médio: carga da instalação, carga de backup e geração"
        if estudo.com_solar else "Dia médio: carga da instalação e carga de backup"
    )
    eixo.legend(fontsize=7.5, ncol=2, frameon=False)
    return _salvar(fig, destino, "dia_medio")


def _grafico_mapa(resultado, destino: Path) -> Path:
    estacoes = list(dict.fromkeys(resultado.tabela["estacao"]))
    fig, eixos = plt.subplots(1, len(estacoes), figsize=(3.2 * len(estacoes), 4.6), dpi=140, sharey=True)
    eixos = np.atleast_1d(eixos)
    for eixo, estacao in zip(eixos, estacoes):
        matriz = resultado.matriz_hora_duracao(estacao)
        imagem = eixo.imshow(
            matriz.to_numpy() * 100.0, aspect="auto", origin="lower",
            cmap="RdYlGn", vmin=0, vmax=100,
        )
        eixo.set_xticks(range(len(matriz.columns)))
        eixo.set_xticklabels([f"{c:g}h" for c in matriz.columns], fontsize=7)
        eixo.set_yticks(range(0, len(matriz.index), 3))
        eixo.set_yticklabels(matriz.index[::3], fontsize=7)
        eixo.set_title(estacao, fontsize=9)
        eixo.set_xlabel("duração", fontsize=8)
    eixos[0].set_ylabel("hora de início do apagão", fontsize=8)
    fig.colorbar(imagem, ax=eixos, label="probabilidade de atravessar (%)", fraction=0.03)
    # Característica nominal, não marca: a referência comercial fica no anexo.
    fig.suptitle(resultado.conjunto.especificacao_curta(), fontsize=9)
    return _salvar(fig, destino, "mapa_atendimento")


def _grafico_soc(estudo: ResultadoEstudo, resultado, destino: Path) -> Path:
    """Trajetória do estado de carga no apagão longo mais desfavorável."""
    from .apagao import _lote_de_carga, _lote_de_geracao
    from .despacho import simular_ilhamento

    malha = resultado.malha
    duracao = max(malha.duracoes_h)
    pior = resultado.pior_janela(duracao)
    estacao, hora = str(pior["estacao"]), int(pior["hora_inicio"])
    n_passos = int(round(duracao * 60 / malha.passo_min))

    rng = np.random.default_rng(99)
    carga, pico = _lote_de_carga(estudo.ensemble_backup, estacao, [hora], n_passos, 200, rng)
    geracao = _lote_de_geracao(
        estudo.serie, estacao, [hora], n_passos, malha.passo_min, 200,
        resultado.conjunto.potencia_fv_aproveitavel_kw, rng,
    )
    despacho = simular_ilhamento(
        carga, pico, geracao, resultado.limites, malha.passo_min,
        malha.soc_inicial_frac, guardar_trajetoria=True,
    )
    trajetoria = despacho.trajetoria_soc_frac * 100.0
    horas = np.arange(n_passos) * malha.passo_min / 60.0

    fig, eixo = _figura()
    eixo.fill_between(
        horas, np.percentile(trajetoria, 5, axis=0), np.percentile(trajetoria, 95, axis=0),
        color=_CORES[0], alpha=0.18, label="faixa P5–P95",
    )
    eixo.plot(horas, np.percentile(trajetoria, 50, axis=0), color=_CORES[0], lw=2.0, label="mediana")
    eixo.plot(horas, np.percentile(trajetoria, 5, axis=0), color=_CORES[4], lw=1.0, ls="--", label="P5")
    eixo.set_xlabel(f"Horas desde o início do apagão (começando às {hora:02d}h, {estacao})")
    eixo.set_ylabel("Estado de carga (%)")
    eixo.set_ylim(0, 100)
    eixo.set_title(
        f"Apagão de {duracao:g} h — {resultado.conjunto.especificacao_curta()}")
    eixo.legend(fontsize=8, frameon=False)
    return _salvar(fig, destino, "estado_de_carga")


def _grafico_fronteira(estudo: ResultadoEstudo, destino: Path) -> Path:
    """
    Energia útil × autonomia garantida: onde a curva satura por potência.

    Dois cuidados que a versão anterior não tinha e que produziam uma figura
    enganosa:

    * **A autonomia é censurada.** Ela só pode valer o que a malha simulou. Um
      conjunto marcado com 36 h significa "pelo menos 36 h" — não se mediu
      além. Ligar esses pontos como se fossem medidas exatas sugere uma
      precisão que não existe, então eles saem com marcador de seta.
    * **Os candidatos são poucos e espaçados.** Uma reta traçada entre 14 kWh e
      79 kWh atravessa 65 kWh onde nada foi avaliado. O traço fica pontilhado
      nos vãos grandes, para não passar por medição contínua.
    """
    fig, eixo = _figura(9.0, 4.8)
    cfg = estudo.configuracao
    ranking = estudo.ranking.sort_values("energia_util_kwh")
    teto = max(cfg.malha.duracoes_h)

    x = ranking["energia_util_kwh"].to_numpy(dtype=float)
    for coluna, cor, marcador, rotulo, estilo in (
        ("autonomia_garantida_h", _CORES[0], "o", "hoje", "-"),
        ("autonomia_ano10_h", _CORES[1], "s", "ano 10", "--"),
    ):
        if coluna not in ranking or not ranking[coluna].notna().any():
            continue
        y = ranking[coluna].to_numpy(dtype=float)
        # Segmento a segmento: vão maior que o dobro do passo típico vira
        # pontilhado, porque ali não houve avaliação nenhuma.
        if len(x) > 1:
            passo_tipico = float(np.median(np.diff(x))) if len(x) > 2 else float(np.diff(x)[0])
            for i in range(len(x) - 1):
                vao = x[i + 1] - x[i]
                eixo.plot(
                    x[i:i + 2], y[i:i + 2], color=cor,
                    ls=":" if vao > 2.5 * max(passo_tipico, 1e-6) else estilo,
                    lw=1.5, alpha=0.9 if vao <= 2.5 * passo_tipico else 0.55,
                )
        no_teto = y >= teto - 1e-9
        eixo.plot(x[~no_teto], y[~no_teto], marker=marcador, ls="none",
                  color=cor, label=rotulo)
        if no_teto.any():
            eixo.plot(x[no_teto], y[no_teto], marker="^", ls="none", color=cor,
                      markersize=9, markerfacecolor="none", markeredgewidth=1.8)

    eixo.axhline(cfg.autonomia_alvo_h, color=_CORES[4], ls=":", lw=1.6,
                 label=f"meta {cfg.autonomia_alvo_h:g} h")
    eixo.axhline(teto, color="#888888", ls="-.", lw=1.0)
    eixo.text(
        eixo.get_xlim()[1] if False else x.max(), teto,
        f"  limite do que foi simulado ({teto:g} h)",
        fontsize=7.5, color="#666666", va="bottom", ha="right",
    )
    eixo.set_ylim(0, teto * 1.12)
    eixo.set_xlabel("Energia útil do banco (kWh)")
    eixo.set_ylabel(f"Autonomia garantida a {cfg.confiabilidade_alvo:.0%} (h)")
    eixo.set_title("Fronteira de dimensionamento — onde mais kWh deixa de resolver")
    eixo.legend(fontsize=8, frameon=False, loc="lower right")
    eixo.text(
        0.01, 0.97,
        "△ = atingiu o teto simulado; a autonomia real pode ser maior\n"
        "··· = vão sem conjunto avaliado",
        transform=eixo.transAxes, fontsize=7, color="#666666", va="top",
    )
    return _salvar(fig, destino, "fronteira")


def _grafico_degradacao(estudo: ResultadoEstudo, chave: str, destino: Path) -> Path:
    fig, eixo = _figura(9.0, 4.6)
    trajetoria = estudo.degradacao[chave]
    eixo.plot(trajetoria["ano"], trajetoria["retencao"] * 100.0, color=_CORES[0], lw=2.0, label="retenção")
    eixo.fill_between(
        trajetoria["ano"], 0, trajetoria["fade_calendario"] * 100.0,
        color=_CORES[2], alpha=0.25, label="perda por calendário",
    )
    eixo.fill_between(
        trajetoria["ano"], trajetoria["fade_calendario"] * 100.0,
        (trajetoria["fade_calendario"] + trajetoria["fade_ciclagem"]) * 100.0,
        color=_CORES[1], alpha=0.25, label="perda por ciclagem",
    )
    autonomias = estudo.resiliencia_degradada.get(chave, {})
    if autonomias:
        gemeo = eixo.twinx()
        anos = sorted(autonomias)
        gemeo.plot([a for a in anos], [autonomias[a] for a in anos],
                   marker="D", color=_CORES[4], lw=1.6, label="autonomia garantida")
        gemeo.set_ylabel("Autonomia garantida (h)", color=_CORES[4])
        gemeo.tick_params(axis="y", labelcolor=_CORES[4])
    eixo.set_xlabel("Ano")
    eixo.set_ylabel("Capacidade (% da original)")
    eixo.set_title(f"Envelhecimento — {chave}")
    eixo.legend(fontsize=8, loc="lower left", frameon=False)
    return _salvar(fig, destino, "degradacao")


# ----------------------------------------------------------------------------
# Relatório
# ----------------------------------------------------------------------------
def _grafico_perfis_ocupacao(estudo: ResultadoEstudo, destino: Path) -> Path:
    """
    Os dois dias da mesma casa, com o sol por cima.

    É o gráfico que resume a decisão inteira, e por isso vem no começo. Ele
    mostra três coisas de uma vez:

    * **as duas curvas não são a mesma casa em escala** — no dia de semana o
      meio do dia é um vale, e no fim de semana é onde a carga mora;
    * **o sol nasce no vale de um e no pico do outro** — o autoconsumo do fim
      de semana é naturalmente maior, e é o que faz o solar render mais em
      casa cheia;
    * **o que sobra depois do sol** é o que a bateria tem de guardar, e a
      diferença entre os dois dias é o tamanho do banco.

    A carga de backup entra tracejada porque é a única das três curvas que o
    sistema promete atender no apagão — e é uma fração pequena das outras.
    """
    ocupacao = getattr(estudo.configuracao, "ocupacao", None) or {}
    curvas: dict[str, np.ndarray] = ocupacao.get("curvas") or {}
    passo_min = int(ocupacao.get("passo_min") or 1)

    fig, eixo = _figura(10.0, 5.4)

    # A faixa de todos os arranjos de horário, no fundo de tudo.
    #
    # Sem ela o gráfico mostra dois dias em UM arranjo cada, e o leitor entende
    # que a casa vive entre aquelas duas linhas. Ela não vive: a varredura mede
    # dezenas de arranjos, e a faixa entre eles é bem mais larga que a distância
    # entre as duas curvas. Desenhar só as curvas é dizer "é isto" quando o que
    # se sabe é "está em algum lugar aqui dentro".
    arranjos = 0
    envelope = getattr(estudo, "envelope_rotina", None)
    if envelope is not None and len(envelope):
        from ..demanda.rotina import banda_do_envelope

        banda = banda_do_envelope(envelope)
        if banda is not None:
            passos = len(banda["minimo"])
            horas_banda = np.arange(passos) * (24.0 / passos)
            arranjos = int(banda["arranjos"][0])
            eixo.fill_between(
                horas_banda, banda["minimo"] / 1000.0, banda["maximo"] / 1000.0,
                color=marca.APOIO["azul"], alpha=0.13, zorder=0, linewidth=0,
                label=f"faixa de {arranjos} arranjos de horário",
            )
            eixo.fill_between(
                horas_banda, banda["p10"] / 1000.0, banda["p90"] / 1000.0,
                color=marca.APOIO["azul"], alpha=0.18, zorder=0, linewidth=0,
            )

    # A geração depois, ao fundo: é a área, e as curvas passam por cima.
    if estudo.com_solar:
        media = np.mean(
            [estudo.serie.janela_media_por_hora(e) for e in estudo.ensemble_total.estacoes],
            axis=0,
        ) * estudo.potencia_fv_kwp / 1000.0
        eixo.fill_between(
            np.arange(24) + 0.5, 0, media, color=_SEM["geracao"], alpha=0.30,
            label=f"geração de {estudo.potencia_fv_kwp:.1f} kWp", zorder=1,
        )

    # O âmbar é do sol e de mais ninguém neste gráfico: uma curva de carga da
    # mesma cor da área de geração some dentro dela.
    paleta = [_SEM["carga"], marca.APOIO["azul"], marca.APOIO["roxo"]]
    estilos = [("-", 2.4), ((0, (7, 2)), 2.4), ((0, (2, 2)), 2.0)]
    for i, (nome, curva) in enumerate(curvas.items()):
        horas = np.arange(len(curva)) * passo_min / 60.0
        traco, largura = estilos[i % len(estilos)]
        eixo.plot(
            horas, np.asarray(curva) / 1000.0,
            color=paleta[i % len(paleta)], lw=largura, ls=traco, label=nome, zorder=3,
        )

    backup = estudo.ensemble_backup
    eixo.plot(
        np.arange(backup.passos_por_dia) * backup.passo_min / 60.0,
        backup.perfil_medio_w() / 1000.0,
        color=_SEM["backup"], lw=1.8, ls=(0, (4, 2, 1, 2)),
        label="quadro de backup", zorder=4,
    )

    eixo.set_xlabel("Hora do dia")
    eixo.set_ylabel("Potência (kW)")
    eixo.set_xlim(0, 24)
    eixo.set_ylim(bottom=0)
    eixo.set_xticks(range(0, 25, 3))
    # O título dizia "e o sol por cima" mesmo num estudo sem solar, onde não há
    # curva de sol nenhuma no desenho. Texto e figura discordando é o mesmo
    # defeito de sempre, e aqui a frase era simplesmente falsa.
    dias = "Os dois dias da mesma casa" if len(curvas) > 1 else "O dia desta casa"
    if estudo.com_solar:
        titulo = f"{dias}, e o sol por cima"
    elif arranjos:
        titulo = f"{dias}, dentro da faixa de {arranjos} arranjos de horário"
    else:
        titulo = dias
    eixo.set_title(titulo)
    eixo.legend(fontsize=8, ncol=2, frameon=False)
    return _salvar(fig, destino, "perfis_ocupacao")


def _grafico_cenarios_de_uso(uso, destino: Path) -> Path:
    """
    Os três cenários de uso, e o sistema que cada um exige.

    À esquerda a curva de cada um — é onde se vê que não são a mesma casa em
    escala: o dia levantado em campo é quase todo noturno, a casa cheia enche
    o meio do dia, e o dia de semana fica no meio com os dois cumes de
    refeição. À direita o que isso cobra em equipamento.

    A geração entra atrás das curvas porque a pergunta que o desenho responde
    é de quanto sol cada cenário precisa — e sol nasce na hora em que os três
    cenários mais diferem entre si.
    """
    cenarios = list(uso.cenarios)
    fig, eixos = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=140)
    for eixo in eixos:
        eixo.grid(alpha=0.25, linewidth=0.6)
        eixo.set_axisbelow(True)

    paleta = [_SEM["backup"], marca.APOIO["azul"], marca.APOIO["roxo"]]
    estilos = [(0, (2, 2)), "-", (0, (7, 2))]
    for i, cenario in enumerate(cenarios):
        curva = np.asarray(cenario.curva_w) / 1000.0
        horas = np.arange(len(curva)) * cenario.passo_min / 60.0
        eixos[0].plot(horas, curva, color=paleta[i % len(paleta)], lw=2.2,
                      ls=estilos[i % len(estilos)], label=cenario.nome)
    eixos[0].set_xlabel("Hora do dia")
    eixos[0].set_ylabel("Potência (kW)")
    eixos[0].set_xlim(0, 24)
    eixos[0].set_ylim(bottom=0)
    eixos[0].set_xticks(range(0, 25, 4))
    eixos[0].set_title("A carga de cada cenário")
    eixos[0].legend(fontsize=7.5, frameon=False)

    posicoes = np.arange(len(cenarios))
    # Sem solar, a barra de kWp seria uma linha no zero em todos os cenários —
    # e uma legenda de "solar" num estudo sem solar. Fica só o banco, centrado.
    com_solar = any(c.potencia_fv_kwp > 0 for c in cenarios)
    largura = 0.38 if com_solar else 0.5
    if com_solar:
        eixos[1].bar(posicoes - largura / 2, [c.potencia_fv_kwp for c in cenarios],
                     largura, color=_SEM["geracao"], label="solar (kWp)")
        eixos[1].bar(posicoes + largura / 2, [c.banco_kwh for c in cenarios],
                     largura, color=marca.APOIO["azul"], label="banco (kWh úteis)")
    else:
        eixos[1].bar(posicoes, [c.banco_kwh for c in cenarios],
                     largura, color=marca.APOIO["azul"], label="banco (kWh úteis)")
    for i, cenario in enumerate(cenarios):
        eixos[1].annotate(
            f"{cenario.energia_diaria_kwh:.0f} kWh/dia",
            (i, max(cenario.potencia_fv_kwp, cenario.banco_kwh)),
            textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7.5,
        )
    eixos[1].set_xticks(posicoes)
    # Quebra em duas linhas: os nomes são frases, e três frases lado a lado
    # no eixo se sobrepõem.
    eixos[1].set_xticklabels(
        [chr(10).join(_quebrar(c.nome, 16)) for c in cenarios], fontsize=7.0)
    eixos[1].set_title("O sistema que cada um exige" if com_solar
                       else "O banco que cada um exige")
    eixos[1].legend(fontsize=8, frameon=False)

    return _salvar(fig, destino, "cenarios_de_uso")


def _grafico_escopos(escopos, destino: Path) -> Path:
    """
    O essencial e o ampliado, nas duas dimensões que decidem o banco.

    Dois painéis, e não um índice único, porque são duas grandezas que se
    resolvem com equipamentos diferentes. A **energia** do dia diz quantos
    módulos de bateria; o **pico** diz qual inversor. Um quadro que dobra de
    energia e não muda de pico custa módulo; um que dobra de pico exige
    inversor maior, e nenhum módulo extra resolve.

    A curva de cada escopo entra por cima porque a diferença entre eles não é
    só de tamanho: os preferíveis costumam ser cargas de conforto, e elas se
    concentram nas horas em que a família está em casa -- exatamente quando o
    essencial também está no seu máximo.
    """
    medidas = list(escopos.medidas)
    fig, eixos = plt.subplots(1, 2, figsize=(11.0, 4.6), dpi=140)
    for eixo in eixos:
        eixo.grid(alpha=0.25, linewidth=0.6)
        eixo.set_axisbelow(True)

    # Painel 1: as curvas médias, uma por escopo.
    cores = [_SEM["backup"], marca.APOIO["azul"]]
    for i, medida in enumerate(medidas):
        curva = np.asarray(medida.curva_w) / 1000.0
        horas = np.arange(len(curva)) * medida.passo_min / 60.0
        eixos[0].plot(horas, curva, color=cores[i % len(cores)], lw=2.2,
                      label=f"{medida.escopo.nome} ({medida.escopo.rotulo_dos_niveis})")
    eixos[0].set_xlabel("Hora do dia")
    eixos[0].set_ylabel("Potência (kW)")
    eixos[0].set_xlim(0, 24)
    eixos[0].set_ylim(bottom=0)
    eixos[0].set_xticks(range(0, 25, 4))
    eixos[0].set_title("A carga de cada quadro")
    eixos[0].legend(fontsize=8, frameon=False)

    # Painel 2: energia e pico lado a lado, com o banco de cada um.
    rotulos = [m.escopo.nome for m in medidas]
    posicoes = np.arange(len(medidas))
    largura = 0.36
    eixos[1].bar(posicoes - largura / 2, [m.energia_diaria_kwh for m in medidas],
                 largura, color=_SEM["backup"], label="energia do dia (kWh)")
    eixos[1].bar(posicoes + largura / 2, [m.energia_util_kwh for m in medidas],
                 largura, color=marca.APOIO["azul"], label="banco necessário (kWh úteis)")
    for i, medida in enumerate(medidas):
        eixos[1].annotate(
            f"pico {medida.pico_p95_kw:.2f} kW\n{medida.autonomia_h:.0f} h",
            (i, max(medida.energia_diaria_kwh, medida.energia_util_kwh)),
            textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7.5,
        )
    eixos[1].set_xticks(posicoes)
    eixos[1].set_xticklabels(rotulos, fontsize=8)
    eixos[1].set_ylabel("kWh")
    eixos[1].set_title("O que cada quadro consome e exige")
    eixos[1].legend(fontsize=8, frameon=False)

    return _salvar(fig, destino, "escopos_backup")


def _grafico_cenarios(comparacao, destino: Path) -> Path:
    """
    Os arranjos lado a lado, nas duas dimensões que decidem: conta e apagão.

    Dois painéis porque são duas perguntas diferentes, e juntá-las num índice
    único esconderia justamente o que o quadro existe para mostrar -- que
    solar resolve a conta e não resolve o apagão, e que a bateria faz o
    contrário. Um eixo só obrigaria a inventar um peso entre reais por ano e
    quilowatt-hora que faltam, e esse peso é do cliente, não do estudo.

    O painel da direita mede a energia que falta numa interrupção média, e não
    a autonomia garantida. A autonomia é a promessa que se pode assinar -- e
    ela empata em zero para todo mundo sempre que a carga essencial é pesada,
    deixando o painel em branco justamente no caso em que a comparação mais
    importa. A energia que falta separa os arranjos em qualquer situação; a
    autonomia entra como rótulo em cada barra.
    """
    cenarios = list(comparacao.cenarios)
    nomes = [c.nome.replace("Rede + ", "") for c in cenarios]
    posicoes = np.arange(len(cenarios))

    fig, (esq, dir_) = plt.subplots(
        1, 2, figsize=(11.5, 0.55 * len(cenarios) + 2.4), dpi=140, sharey=True
    )
    for eixo in (esq, dir_):
        eixo.grid(alpha=0.25, linewidth=0.6, axis="x")
        eixo.set_axisbelow(True)

    # -- esquerda: a conta de energia --------------------------------------
    base = comparacao.base.conta_anual_brl
    contas = [c.conta_anual_brl for c in cenarios]
    cores = [_SEM["geracao"] if c.capex_brl > 0 else _SEM["neutro"] for c in cenarios]
    esq.barh(posicoes, contas, color=cores, alpha=0.85)
    esq.axvline(base, color=_SEM["alerta"], lw=1.6, ls="--", label="conta de hoje")
    for y, (conta, cenario) in enumerate(zip(contas, cenarios)):
        # Só rotula quem de fato mexeu na conta: "−0 R$/ano" é ruído com cara
        # de resultado.
        if cenario.economia_anual_brl > max(1.0, 0.01 * base):
            esq.text(
                conta + base * 0.02, y,
                f"−{_milhar(cenario.economia_anual_brl)} R$/ano",
                va="center", fontsize=8, color="#333333",
            )
    esq.set_xlim(0, base * 1.45)
    esq.set_yticks(posicoes)
    esq.set_yticklabels(nomes, fontsize=9)
    esq.invert_yaxis()
    esq.set_xlabel("Conta de energia (R$/ano)")
    esq.set_title("O que sobra na conta", fontsize=10)
    esq.legend(fontsize=8, loc="lower right")

    # -- direita: o que falta no apagão ------------------------------------
    faltas = [float(np.nan_to_num(c.ens_por_evento_kwh)) for c in cenarios]
    sem_nada = faltas[0] if faltas else 0.0
    cores_falta = [_SEM["carga"] if c.capex_brl > 0 else _SEM["neutro"] for c in cenarios]
    dir_.barh(posicoes, faltas, color=cores_falta, alpha=0.85)
    if sem_nada > 0:
        dir_.axvline(sem_nada, color=_SEM["alerta"], lw=1.6, ls="--", label="sem nenhuma fonte")
        dir_.legend(fontsize=8, loc="lower right")
    teto = max(faltas) if faltas and max(faltas) > 0 else 1.0
    for y, (falta, cenario) in enumerate(zip(faltas, cenarios)):
        horas = cenario.autonomia_garantida_h
        # Sem autonomia garantida, o rótulo útil é o próprio número: as barras
        # pequenas são ilegíveis, e "nada garantido" oito vezes não informa.
        rotulo = f"{horas:g} h garantidas" if horas > 0 else f"{falta:,.1f} kWh".replace(",", ".")
        dir_.text(falta + teto * 0.03, y, rotulo, va="center", fontsize=8, color="#333333")
    dir_.set_xlim(0, teto * 1.55)
    duracao = comparacao.premissas.duracao_media_interrupcao_h
    dir_.set_xlabel(f"Energia que falta numa interrupção de {duracao:g} h (kWh)")
    dir_.set_title("O que acontece quando a luz cai", fontsize=10)

    fig.suptitle(
        f"Cenários de fontes: o mesmo cliente, {len(cenarios)} arranjos", fontsize=12
    )
    fig.tight_layout()
    return _salvar(fig, destino, "cenarios")


def _grafico_gerador(comparacao, custo: pd.DataFrame, destino: Path) -> Path:
    """
    O que custa rodar o grupo, por duração de falta.

    Com o grupo dimensionado para suprir a carga, a potência sai da discussão
    e sobra o combustível. A curva cresce com a duração porque é energia, não
    potência, que se queima — e é por isso que a decisão entre gerador e
    bateria muda de lado conforme a falta é curta ou longa.

    As duas curvas, quando existem, são o argumento econômico da dupla: com
    banco, o grupo entra depois do sol e da bateria e queima menos.
    """
    fig, eixo = _figura(9.0, 5.0)
    grupo = comparacao.gerador

    # Traços diferentes por arranjo: sem sol ilhado, "gerador" e
    # "solar + gerador" queimam exatamente o mesmo, e uma linha some debaixo da
    # outra deixando na legenda uma cor que não aparece em lugar nenhum.
    tracos = [(0, ()), (0, (6, 2)), (0, (2, 2)), (0, (7, 2, 1, 2))]
    fim_x = float(custo["duracao_h"].max())
    for i, (nome, bloco) in enumerate(custo.groupby("cenario", sort=False)):
        bloco = bloco.sort_values("duracao_h")
        eixo.plot(
            bloco["duracao_h"], bloco["custo_do_evento_brl"],
            marker="o", ms=4.0, lw=2.0, color=_CORES[i % len(_CORES)],
            ls=tracos[i % len(tracos)],
            label=nome.replace("Rede + ", ""),
        )
        # Só o último ponto recebe rótulo: com quatro curvas quase coincidentes,
        # rotular todos os pontos vira uma nuvem ilegível.
        ultimo = bloco.iloc[-1]
        eixo.annotate(
            f"R$ {_milhar(ultimo.custo_do_evento_brl)}",
            (ultimo.duracao_h, ultimo.custo_do_evento_brl),
            textcoords="offset points", xytext=(8, -2),
            ha="left", va="center", fontsize=8,
            color=_CORES[i % len(_CORES)],
        )
    eixo.set_xlim(0, fim_x * 1.28)

    eixo.set_xlabel("Duração da falta (h)")
    eixo.set_ylabel("Custo do combustível no evento (R$)")
    eixo.set_ylim(bottom=0)
    eixo.set_title(
        f"Custo de rodar o {grupo.descricao()} a "
        f"R$ {grupo.custo_energia_brl_kwh:.2f}/kWh".replace(".", ",")
    )
    eixo.legend(fontsize=8, frameon=False)
    return _salvar(fig, destino, "gerador")


def _tabela_md(df: pd.DataFrame, casas: int = 2) -> str:
    return df.round(casas).to_markdown(index=False)


def escrever_relatorio(
    estudo: ResultadoEstudo,
    destino: str | Path,
    com_graficos: bool = True,
    com_latex: bool = True,
) -> dict[str, Path]:
    """
    Grava o pacote completo: figuras, planilhas CSV, ``estudo.md`` e ``estudo.tex``.

    Devolve o mapa de tudo que foi escrito, para que a interface e a CLI
    saibam o que oferecer sem adivinhar nomes de arquivo.
    """
    pasta = Path(destino)
    pasta.mkdir(parents=True, exist_ok=True)
    escritos: dict[str, Path] = {}

    figuras = gerar_graficos(estudo, pasta) if com_graficos else {}
    escritos.update({f"figura_{k}": v for k, v in figuras.items()})

    tabelas = {
        "diagnostico_inversores": estudo.diagnostico_inversores,
        "excedencia_potencia": estudo.tabela_excedencia,
        "excedencia_horaria": estudo.tabela_horaria,
        "ranking": estudo.ranking,
    }
    if estudo.cenarios is not None:
        custo_grupo = estudo_do_gerador(estudo.cenarios)
        if not custo_grupo.empty:
            tabelas["custo_do_gerador"] = custo_grupo
        # A triagem completa de inversores e o quadro de cenários vão para a
        # planilha, não para o documento: quem vende precisa dos dois, quem
        # lê a proposta precisa da conclusão.
        tabelas["cenarios_de_fontes"] = estudo.cenarios.tabela()
    for resultado in estudo.resiliencia:
        nome = _fatiar(resultado.conjunto.especificacao_curta())
        tabelas[f"resiliencia_{nome}"] = resultado.tabela
    for chave, economico in estudo.economia.items():
        tabelas[f"fluxo_{_fatiar(chave)}"] = economico.fluxo
    for nome, tabela in tabelas.items():
        caminho = pasta / f"{nome}.csv"
        tabela.to_csv(caminho, index=False, encoding="utf-8-sig", sep=";", decimal=",")
        escritos[f"csv_{nome}"] = caminho

    markdown = _montar_markdown(estudo, figuras)
    caminho_md = pasta / "estudo.md"
    caminho_md.write_text(markdown, encoding="utf-8")
    escritos["markdown"] = caminho_md

    if com_latex:
        caminho_tex = pasta / "estudo.tex"
        caminho_tex.write_text(_montar_latex(estudo, figuras), encoding="utf-8")
        escritos["latex"] = caminho_tex

    return escritos


def _quebrar(texto: str, largura: int = 16) -> list[str]:
    """Quebra uma frase em linhas curtas, sem partir palavra."""
    linhas, atual = [], ""
    for palavra in str(texto).split():
        if atual and len(atual) + 1 + len(palavra) > largura:
            linhas.append(atual)
            atual = palavra
        else:
            atual = f"{atual} {palavra}".strip()
    if atual:
        linhas.append(atual)
    return linhas or [""]


def _fatiar(texto: str) -> str:
    """Nome de arquivo seguro a partir da descrição do conjunto."""
    limpo = "".join(c if c.isalnum() or c in "-_" else "_" for c in texto)
    return limpo.strip("_")[:60]


def _montar_markdown(estudo: ResultadoEstudo, figuras: dict[str, Path]) -> str:
    cfg = estudo.configuracao
    partes: list[str] = []
    partes.append(f"# Estudo de armazenamento — {cfg.nome}\n")
    partes.append(
        f"Local: {cfg.latitude:.4f}, {cfg.longitude:.4f} · "
        f"Sistema FV: **{estudo.potencia_fv_kwp:.1f} kWp** "
        f"({estudo.serie.anual_kwh_por_kwp():.0f} kWh/kWp·ano, fonte `{estudo.serie.fonte}`)\n"
    )
    partes.append(f"Malha simulada: {cfg.malha.descricao()}\n")

    if estudo.avisos:
        partes.append("## Avisos\n")
        partes.extend(f"- {a}\n" for a in estudo.avisos)

    telhado = cfg.telhado
    layout = cfg.layout
    if telhado is not None and layout is not None:
        partes.append("\n## 0. O telhado\n")
        partes.append(
            f"Área marcada em planta: **{_milhar(telhado.area_m2)} m²**"
            f" · superfície real do telhado: {_milhar(telhado.area_inclinada_m2)} m²"
            f" · face **{telhado.orientacao}** ({telhado.azimute_deg:.0f}°)"
            f" · inclinação {telhado.inclinacao_deg:.0f}°.\n\n"
        )
        partes.append(
            f"Cabem **{layout.quantidade} módulos** de "
            f"{layout.modulo} em {layout.orientacao_modulo}, somando "
            f"**{layout.potencia_kwp:.1f} kWp** — ocupação de "
            f"{layout.taxa_ocupacao:.0%} da área bruta, "
            f"{layout.densidade_wp_m2:.0f} Wp/m². "
            f"A área aproveitável já desconta {(1 - telhado.fator_obstaculos):.0%} "
            "por obstáculos não visíveis na imagem.\n"
        )
        if telhado.montagem == "coplanar":
            partes.append(
                "\n> A orientação da face vem do contorno desenhado, e o contorno não "
                "distingue para qual dos dois lados a água cai — a escolha aqui é a mais "
                "próxima do ótimo da latitude. Confirme em campo antes de emitir a proposta.\n"
            )
        if "telhado" in figuras:
            partes.append(f"\n![Croqui do telhado]({figuras['telhado'].name})\n")

    partes.append("\n## 1. A carga\n")
    resumo_backup = estudo.ensemble_backup.resumo()
    partes.append(
        f"Carga do quadro de backup: pico médio **{resumo_backup['geral']['pico_medio_kw']:.1f} kW**, "
        f"pico P95 **{resumo_backup['geral']['pico_p95_kw']:.1f} kW**, "
        f"consumo diário médio **{resumo_backup['geral']['energia_diaria_media_kwh']:.1f} kWh**.\n"
    )
    partes.append("\n### Potência exigida por probabilidade de excedência\n")
    partes.append(_tabela_md(estudo.tabela_excedencia) + "\n")
    partes.append(
        "\n> `pico_diario_kw` é a potência excedida em uma fração dos **dias**; "
        "`carga_instantanea_kw`, a excedida em uma fração do **tempo**. A primeira "
        "dimensiona, a segunda diz por quanto tempo o inversor ficaria saturado.\n"
    )

    partes.append("\n## 2. Triagem dos inversores\n")
    colunas = [
        "fabricante", "modelo", "nominal_kw", "pico_kw", "duracao_pico_s",
        "prob_pico_diario_acima_nominal", "prob_pico_diario_acima_surto",
        "fracao_do_tempo_acima_nominal", "duracao_media_min", "veredito",
    ]
    partes.append(_tabela_md(estudo.diagnostico_inversores[colunas], 3) + "\n")

    if "excedencia" in figuras:
        partes.append(f"\n![Curva de excedência]({figuras['excedencia'].name})\n")

    partes.append("\n## 3. Resiliência a apagões\n")
    for resultado in estudo.resiliencia:
        partes.append(f"\n### {resultado.conjunto.descricao()}\n")
        partes.append(_tabela_md(resultado.por_duracao(), 3) + "\n")
        partes.append(
            f"\nAutonomia garantida a {cfg.confiabilidade_alvo:.0%} "
            f"(pior estação e pior hora): **{resultado.autonomia_garantida_h(cfg.confiabilidade_alvo):g} h**. "
            f"Na média das horas: {resultado.autonomia_garantida_h(cfg.confiabilidade_alvo, False):g} h.\n"
        )
    for chave in ("mapa_atendimento", "soc", "dia_medio", "fronteira"):
        if chave in figuras:
            partes.append(f"\n![{chave}]({figuras[chave].name})\n")

    partes.append("\n## 4. Economia e vida útil\n")
    if not estudo.ranking.empty:
        colunas_rank = [
            "conjunto", "energia_util_kwh", "potencia_kw", "autonomia_garantida_h",
            "autonomia_ano5_h", "autonomia_ano10_h", "capex_brl", "vpl_brl",
            "payback_anos", "lcos_brl_kwh", "ciclos_ano", "vida_util_anos", "atende_meta",
        ]
        partes.append(_tabela_md(estudo.ranking[colunas_rank]) + "\n")
    if "degradacao" in figuras:
        partes.append(f"\n![Degradação]({figuras['degradacao'].name})\n")

    partes.append("\n## 5. Recomendação\n")
    if estudo.recomendado is not None:
        conjunto = estudo.recomendado.conjunto
        economico = estudo.economia[conjunto.descricao()]
        partes.append(
            f"**{conjunto.descricao()}** — o conjunto de menor CAPEX que cumpre "
            f"{cfg.autonomia_alvo_h:g} h com {cfg.confiabilidade_alvo:.0%} de confiabilidade"
            f"{' no pior par estação/hora' if cfg.exigir_pior_caso else ''}.\n\n"
            f"- Energia útil: {conjunto.energia_util_kwh:.1f} kWh\n"
            f"- Potência contínua / de pico: {conjunto.potencia_descarga_kw:.1f} kW / "
            f"{conjunto.potencia_pico_kw:.1f} kW por {conjunto.inversor.duracao_pico_s:.0f} s\n"
            f"- CAPEX estimado: R$ {economico.capex_brl:,.0f}\n"
            f"- VPL em {economico.premissas.anos_analise if economico.premissas else 15} anos: "
            f"R$ {economico.vpl_brl:,.0f}\n"
            f"- Ciclos por ano: {economico.operacao.ciclos_equivalentes:.0f} · "
            f"vida útil estimada: {economico.vida_util_anos:.1f} anos\n"
        )
    else:
        partes.append(
            "Nenhum conjunto do catálogo cumpre o critério declarado. "
            "Ver os avisos acima para os caminhos possíveis.\n"
        )
    partes.append(
        "\n---\n\n*Gerado por `aurum.bateria`. A demanda vem do modelo Monte Carlo do "
        "D² (Demanda e Dados); a geração, da série horária do PVGIS. "
        "Confira o catálogo `BDBaterias.xlsx` antes de usar em proposta comercial: "
        "os campos marcados `a conferir` são ordens de grandeza de mercado, não datasheets.*\n"
    )
    return "".join(partes)


def _montar_latex(estudo: ResultadoEstudo, figuras: dict[str, Path]) -> str:
    cfg = estudo.configuracao
    linhas: list[str] = [
        "% Seção de armazenamento gerada por aurum.bateria",
        "\\section{Estudo de armazenamento de energia}",
        "",
        f"A instalação analisada ({_escapar_latex(cfg.nome)}) foi modelada com "
        f"{cfg.simulacoes} simulações de Monte Carlo por estação do ano, cruzadas com a "
        f"série horária de geração de um sistema de "
        f"{estudo.potencia_fv_kwp:.1f}~kWp no local.",
        "",
        "\\subsection{Excedência de pico e escolha do inversor}",
        "",
        "\\begin{tabular}{lrrrr}",
        "\\hline",
        "Inversor & Nominal (kW) & Pico (kW) & P(pico $>$ nominal) & Tempo acima (\\%) \\\\",
        "\\hline",
    ]
    for linha in estudo.diagnostico_inversores.itertuples():
        linhas.append(
            f"{_escapar_latex(linha.modelo)} & {linha.nominal_kw:.1f} & {linha.pico_kw:.1f} & "
            f"{linha.prob_pico_diario_acima_nominal:.1%} & "
            f"{linha.fracao_do_tempo_acima_nominal:.2%} \\\\".replace("%", "\\%")
        )
    linhas += ["\\hline", "\\end{tabular}", ""]

    if "excedencia" in figuras:
        linhas += [
            "\\begin{figure}[H]\\centering",
            f"\\includegraphics[width=\\linewidth]{{{figuras['excedencia'].name}}}",
            "\\caption{Curva de excedência do pico diário da carga de backup.}",
            "\\end{figure}",
            "",
        ]

    linhas += ["\\subsection{Autonomia frente a faltas de energia}", ""]
    if estudo.recomendado is not None:
        conjunto = estudo.recomendado.conjunto
        linhas += [
            f"Conjunto recomendado: {_escapar_latex(conjunto.descricao())}.",
            "",
            "\\begin{tabular}{lrr}",
            "\\hline",
            "Duração da falta & Probabilidade de atravessar & Energia não suprida (kWh) \\\\",
            "\\hline",
        ]
        for linha in estudo.recomendado.por_duracao().itertuples():
            linhas.append(
                f"{linha.duracao_h:g} h & {linha.prob_atendimento:.1%} & "
                f"{linha.ens_medio_kwh:.1f} \\\\".replace("%", "\\%")
            )
        linhas += ["\\hline", "\\end{tabular}", ""]
    else:
        linhas += [
            "Nenhum conjunto do catálogo atendeu ao critério de autonomia declarado.",
            "",
        ]

    if "mapa_atendimento" in figuras:
        linhas += [
            "\\begin{figure}[H]\\centering",
            f"\\includegraphics[width=\\linewidth]{{{figuras['mapa_atendimento'].name}}}",
            "\\caption{Probabilidade de atravessar a falta, por hora de início e duração.}",
            "\\end{figure}",
        ]
    return "\n".join(linhas) + "\n"
