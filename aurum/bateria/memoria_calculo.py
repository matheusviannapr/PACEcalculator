"""
Memória de cálculo do estudo: cada premissa escrita, cada conta refeita.

O dossiê defende o dimensionamento e a apresentação vende; nenhum dos dois é
lugar para mostrar todas as contas. Este documento é. Ele existe para que
alguém que não rodou o software consiga, com uma calculadora, chegar aos
mesmos números da proposta — e para que, quando não chegar, fique claro em
qual linha a divergência nasce.

Duas regras organizam o texto:

1. **Todo número traz a sua origem.** Medido (fatura, vistoria), declarado
   (tela), consultado (PVGIS, tabela de kit) ou calculado — e, quando
   calculado, a fórmula com os valores substituídos.
2. **As conferências vêm primeiro.** Antes das contas, uma lista de
   verificações automáticas diz se as partes do estudo conversam entre si:
   o balanço de energia fecha, a conta refeita bate com a do cenário, o
   payback do slide é o investimento dividido pela economia do mesmo slide.
   Uma conferência que falha é defeito, não ressalva.

O texto sai em Markdown, que abre em qualquer editor, renderiza no GitHub e
cola em e-mail; as tabelas de fluxo saem também em CSV, ao lado.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Sequence

import numpy as np

from .. import __version__
from ..pv.financials import CRONOGRAMA_FIO_B, FIO_B_APOS_TRANSICAO, fator_fio_b
from ..util.numfmt import br_float, br_int, br_money

__all__ = ["Conferencia", "conferencias", "montar_memoria"]


# ----------------------------------------------------------------------------
# Formatação
# ----------------------------------------------------------------------------
def _r(valor: Any, casas: int = 0) -> str:
    # Arredondar antes evita o "R$ -0" de uma economia de -0,003.
    try:
        valor = round(float(valor), casas) + 0.0
    except (TypeError, ValueError):
        pass
    return br_money(valor, casas=casas)


def _n(valor: Any, casas: int = 0, unidade: str = "") -> str:
    texto = br_float(valor, casas)
    return f"{texto} {unidade}".strip() if unidade else texto


def _pct(fracao: Any, casas: int = 1) -> str:
    try:
        return f"{br_float(float(fracao) * 100.0, casas)}%"
    except (TypeError, ValueError):
        return "—"


def _tabela(cabecalho: Sequence[str], linhas: Sequence[Sequence[Any]]) -> str:
    """Tabela Markdown; a primeira coluna à esquerda, as demais à direita."""
    alinhar = [":--"] + ["--:"] * (len(cabecalho) - 1)
    saida = [
        "| " + " | ".join(cabecalho) + " |",
        "| " + " | ".join(alinhar) + " |",
    ]
    for linha in linhas:
        saida.append("| " + " | ".join(str(c).replace("|", "/") for c in linha) + " |")
    return "\n".join(saida)


def _anos(anos: float | None) -> str:
    if anos is None:
        return "não se paga no horizonte"
    return f"{br_float(anos, 1)} anos ({br_int(anos * 12)} meses)"


# ----------------------------------------------------------------------------
# Conferências
# ----------------------------------------------------------------------------
@dataclass
class Conferencia:
    """Uma verificação de coerência entre duas partes do estudo."""

    nome: str
    esperado: str
    obtido: str
    ok: bool
    #: ``False`` quando a falha é ressalva (dado frágil) e não defeito de conta.
    grave: bool = True
    explicacao: str = ""

    @property
    def situacao(self) -> str:
        if self.ok:
            return "✅ confere"
        return "❌ não confere" if self.grave else "⚠️ atenção"


def _perto(a: float, b: float, rel: float = 0.005, abs_: float = 0.5) -> bool:
    return abs(float(a) - float(b)) <= max(abs_, rel * max(abs(float(a)), abs(float(b))))


def conferencias(estudo, comercial=None) -> list[Conferencia]:
    """
    Verifica, com números, que as partes do estudo conversam entre si.

    Cada item compara duas maneiras de chegar ao mesmo número. Se o software
    um dia voltar a calcular a mesma coisa em dois lugares, e os dois lugares
    divergirem, é aqui que aparece — antes de o cliente ver.
    """
    itens: list[Conferencia] = []
    comparacao = estudo.cenarios
    cfg = estudo.configuracao

    # -- a série solar ---------------------------------------------------------
    if estudo.com_solar:
        confiavel = bool(getattr(estudo.serie, "confiavel", True))
        itens.append(Conferencia(
            "Série de geração vem do PVGIS",
            "PVGIS (irradiação medida por satélite)",
            str(estudo.serie.fonte),
            confiavel, grave=False,
            explicacao="" if confiavel else
            "A série é sintética, reconstruída por geometria solar: serve para estudo "
            "interno, e a geração precisa ser confirmada antes da proposta.",
        ))

    if comparacao is not None:
        produtividade = estudo.serie.anual_kwh_por_kwp()
        for cenario in comparacao.cenarios:
            b = cenario.balanco
            if not b:
                continue
            # -- energia: o que entra é o que sai --------------------------------
            entra = b["autoconsumo"] + b["compra"] + b["gerador"]
            itens.append(Conferencia(
                f"Balanço de consumo fecha — {cenario.nome}",
                f"consumo {_n(b['consumo'], 0, 'kWh')}",
                f"autoconsumo + compra + gerador = {_n(entra, 0, 'kWh')}",
                _perto(entra, b["consumo"]),
            ))
            if cenario.composicao.solar and b.get("geracao", 0) > 0:
                destino = b.get("autoconsumo_direto", 0) + b.get("recarga", 0) + b["injecao"]
                itens.append(Conferencia(
                    f"Destino da geração fecha — {cenario.nome}",
                    f"geração {_n(b['geracao'], 0, 'kWh')}",
                    f"direto + recarga do banco + injeção = {_n(destino, 0, 'kWh')}",
                    _perto(destino, b["geracao"]),
                ))
                esperado = estudo.potencia_fv_kwp * produtividade
                bruta = b.get("geracao_bruta", b["geracao"])
                itens.append(Conferencia(
                    f"Geração do balanço = kWp × produtividade — {cenario.nome}",
                    f"{_n(estudo.potencia_fv_kwp, 2)} kWp × {_n(produtividade, 0)} = "
                    f"{_n(esperado, 0, 'kWh')}",
                    f"{_n(bruta, 0, 'kWh')}",
                    _perto(bruta, esperado, rel=0.01),
                    explicacao="Diferença acima de 1% indica que o balanço não percorreu a "
                    "série inteira (modo amostrado).",
                ))
                if bruta - b["geracao"] > 0.01 * max(bruta, 1.0):
                    itens.append(Conferencia(
                        f"Geração inteira aproveitada pelo inversor — {cenario.nome}",
                        "nenhum corte na entrada FV do híbrido",
                        f"{_n(bruta - b['geracao'], 0, 'kWh')} cortados por ano",
                        False, grave=False,
                        explicacao="O gerador passa da entrada FV do inversor híbrido; o "
                        "excedente é perdido. Um inversor maior, ou acoplamento CA, o "
                        "recupera.",
                    ))
            # -- dinheiro: a conta refeita --------------------------------------
            detalhe = comparacao.fatura.detalhar_conta(b["compra"], b["injecao"])
            itens.append(Conferencia(
                f"Conta refeita pelas parcelas — {cenario.nome}",
                f"{_r(cenario.conta_anual_brl)}/ano",
                f"{_n(detalhe['faturado_kwh'], 0)} kWh × {_r(detalhe['tarifa_brl_kwh'], 4)}"
                f" + {_n(detalhe['compensado_kwh'], 0)} kWh × {_r(detalhe['fio_b_brl_kwh'], 4)}"
                f" = {_r(detalhe['conta_brl'])}",
                _perto(detalhe["conta_brl"], cenario.conta_anual_brl),
            ))
            if cenario.composicao.n_fontes > 0:
                economia = comparacao.base.conta_anual_brl - cenario.conta_anual_brl
                itens.append(Conferencia(
                    f"Economia = conta de hoje − conta com o sistema — {cenario.nome}",
                    f"{_r(cenario.economia_anual_brl)}",
                    f"{_r(comparacao.base.conta_anual_brl)} − {_r(cenario.conta_anual_brl)}"
                    f" = {_r(economia)}",
                    _perto(economia, cenario.economia_anual_brl),
                ))
                sobra = detalhe["credito_sem_uso_kwh"]
                if sobra > 0.02 * max(b["injecao"], 1.0):
                    itens.append(Conferencia(
                        f"Todo crédito injetado é aproveitado — {cenario.nome}",
                        "crédito sem uso ≈ 0",
                        f"{_n(sobra, 0, 'kWh')}/ano sem uso",
                        False, grave=False,
                        explicacao="O sistema injeta mais do que a conta consegue abater "
                        "acima do custo de disponibilidade. Esse excedente não vira "
                        "economia: o sistema está maior que a conta.",
                    ))

    # -- as propostas -----------------------------------------------------------
    financeiro = getattr(estudo, "financeiro", None) or {}
    for topologia, f in financeiro.items():
        if f.payback_simples_anos is not None:
            itens.append(Conferencia(
                f"Payback simples = investimento ÷ economia — {f.nome}",
                _anos(f.payback_simples_anos),
                f"{_r(f.capex_brl)} ÷ {_r(f.economia_ano1_brl)} = "
                f"{br_float(f.capex_brl / f.economia_ano1_brl, 2)} anos",
                _perto(f.capex_brl / f.economia_ano1_brl, f.payback_simples_anos, abs_=0.01),
            ))
        if f.tir is not None:
            caixa = [-f.capex_brl, *f.fluxo["fluxo_brl"].tolist()]
            residuo = sum(v / (1.0 + f.tir) ** i for i, v in enumerate(caixa))
            itens.append(Conferencia(
                f"TIR zera o valor presente do mesmo fluxo — {f.nome}",
                "VPL à TIR ≈ R$ 0",
                f"{_r(residuo, 2)}",
                abs(residuo) < max(1.0, 1e-4 * f.capex_brl),
            ))
        cenario = comparacao.por_chave(f.cenario) if comparacao is not None else None
        if cenario is not None:
            itens.append(Conferencia(
                f"Economia da proposta = economia do cenário — {f.nome}",
                f"{_r(cenario.economia_anual_brl)}",
                f"{_r(f.economia_ano1_brl)}",
                _perto(cenario.economia_anual_brl, f.economia_ano1_brl),
            ))

    if comparacao is not None and financeiro:
        for chave in {f.cenario for f in financeiro.values()}:
            cenario = comparacao.por_chave(chave)
            opcoes = [f for f in financeiro.values() if f.cenario == chave]
            if cenario is None or not opcoes:
                continue
            menor = min(opcoes, key=lambda f: f.capex_brl)
            itens.append(Conferencia(
                f"Investimento do cenário = proposta mais barata — {cenario.nome}",
                f"{_r(menor.capex_brl)} ({menor.nome})",
                f"{_r(cenario.capex_brl)}",
                _perto(menor.capex_brl, cenario.capex_brl, abs_=1.0),
                explicacao="O quadro de arranjos e a proposta precisam falar do mesmo "
                "investimento; se não, o banco foi contado de dois jeitos.",
            ))

    # -- o slide principal -------------------------------------------------------
    if comparacao is not None and estudo.chave_principal() is not None:
        try:
            from .apresentacao import OpcoesComerciais, dados_da_apresentacao

            slide = dados_da_apresentacao(estudo, opcoes=comercial or OpcoesComerciais())
        except Exception as exc:  # noqa: BLE001 — a conferência registra, não derruba
            itens.append(Conferencia(
                "Slide principal montado", "sem erro", f"erro: {exc}", False))
        else:
            principal = estudo.financeiro_do_cenario(slide["cenario"])
            e = slide["economia"]
            itens.append(Conferencia(
                "Slide principal = financeiro da proposta principal",
                f"{_r(principal.capex_brl)} · {_r(principal.economia_ano1_brl)}/ano · "
                f"{_anos(principal.payback_simples_anos)}",
                f"{_r(e['capex_brl'])} · {_r(e['economia_anual_brl'])}/ano · "
                f"{_anos(e['payback_anos'])}",
                _perto(e["capex_brl"], principal.capex_brl, abs_=1.0)
                and _perto(e["economia_anual_brl"], principal.economia_ano1_brl)
                and (e["payback_anos"] or 0) == (principal.payback_simples_anos or 0),
            ))
            cenario = comparacao.por_chave(slide["cenario"])
            if cenario is not None and cenario.composicao.solar:
                itens.append(Conferencia(
                    "Geração do slide = geração do balanço",
                    f"{_n(cenario.geracao_fv_kwh_ano, 0, 'kWh')}",
                    f"{_n(slide['sistema']['geracao_anual_kwh'], 0, 'kWh')}",
                    _perto(cenario.geracao_fv_kwh_ano, slide["sistema"]["geracao_anual_kwh"]),
                ))

    # -- a cobertura -------------------------------------------------------------
    dados = getattr(estudo, "compensacao", None)
    if dados and estudo.com_solar and dados.get("definida_por") == "consumo a compensar":
        pedida = float(dados["fracao_pedida"])
        obtida = float(dados["cobertura"])
        itens.append(Conferencia(
            "Cobertura pedida foi atingida (consumo menos custo de disponibilidade)",
            f"{_pct(pedida, 0)} de {_n(dados.get('compensavel_kwh', 0.0), 0, 'kWh')}",
            _pct(obtida, 0),
            abs(obtida - pedida) <= 0.10, grave=False,
            explicacao="A diferença vem do arredondamento em módulos inteiros e do menor "
            "arranjo válido do inversor escolhido.",
        ))

    # -- o consumo de referência ---------------------------------------------------
    if comparacao is not None and comparacao.fatura.consumo_anual_kwh:
        base = comparacao.base.balanco.get("consumo", 0.0)
        itens.append(Conferencia(
            "Consumo da conta de hoje = consumo de referência",
            f"{_n(comparacao.fatura.consumo_anual_kwh, 0, 'kWh/ano')}",
            f"{_n(base, 0, 'kWh/ano')}",
            _perto(base, comparacao.fatura.consumo_anual_kwh, rel=0.01),
        ))

    if cfg.fatura is None:
        itens.append(Conferencia(
            "Tarifa do cliente informada", "tarifa da fatura",
            "tarifa de referência do software", False, grave=False,
            explicacao="Sem a tarifa da fatura, economia e payback são de referência.",
        ))
    return itens


# ----------------------------------------------------------------------------
# O documento
# ----------------------------------------------------------------------------
def _curva_horaria_kw(ensemble, fator: float = 1.0) -> np.ndarray:
    """Perfil médio das quatro estações, em kW, agregado por hora."""
    media = np.mean([ensemble.perfil_medio_w(e) for e in ensemble.estacoes], axis=0)
    por_hora = media.reshape(24, -1).mean(axis=1)
    return por_hora * float(fator) / 1000.0


def _secao_conferencias(itens: list[Conferencia]) -> list[str]:
    falhas = [c for c in itens if not c.ok and c.grave]
    atencao = [c for c in itens if not c.ok and not c.grave]
    resumo = (
        "Todas as conferências de conta fecharam."
        if not falhas else
        f"**{len(falhas)} conferência(s) de conta não fecharam** — o estudo tem "
        "números que não conversam entre si e não deve ser apresentado antes da correção."
    )
    if atencao:
        resumo += f" {len(atencao)} ponto(s) de atenção sobre a qualidade dos dados."
    partes = [
        "## 0. Conferências automáticas",
        "",
        "Cada linha compara duas maneiras de chegar ao mesmo número. ✅ confere; "
        "⚠️ é ressalva sobre o dado (não é erro de conta); ❌ é divergência entre "
        "partes do estudo.",
        "",
        resumo,
        "",
        _tabela(
            ["Conferência", "Esperado", "Obtido", "Situação"],
            [(c.nome, c.esperado, c.obtido, c.situacao) for c in itens],
        ),
    ]
    explicadas = [c for c in itens if not c.ok and c.explicacao]
    if explicadas:
        partes += ["", *[f"- **{c.nome}.** {c.explicacao}" for c in explicadas]]
    return partes


def _secao_identificacao(estudo) -> list[str]:
    cfg = estudo.configuracao
    if cfg.ajuste_conta is not None:
        origem = "curva típica do segmento ajustada à conta de luz (sem levantamento de equipamentos)"
    elif cfg.comodos:
        origem = "levantamento de cargas equipamento a equipamento, simulado por Monte Carlo"
    else:
        origem = f"cenário de demonstração ({cfg.cenario_demo}) — não é o cliente"
    linhas = [
        ("Cliente / local", cfg.nome),
        ("Coordenada", f"{cfg.latitude:.5f}, {cfg.longitude:.5f}"),
        ("Tensão de linha da rede", _n(cfg.tensao_rede_v, 0, "V")),
        ("Origem da carga", origem),
        ("Simulações de Monte Carlo por estação", str(cfg.simulacoes)),
        ("Semente aleatória", str(cfg.semente)),
    ]
    if cfg.ocupacao:
        linhas.append(("Perfil de ocupação que dimensiona",
                       str(cfg.ocupacao.get("dimensionante", "—"))))
    return [
        "## 1. Identificação e origem dos dados",
        "",
        _tabela(["Item", "Valor"], linhas),
    ]


def _secao_consumo(estudo) -> list[str]:
    comparacao = estudo.cenarios
    simulado = (
        float(comparacao.consumo_simulado_kwh)
        if comparacao is not None and comparacao.consumo_simulado_kwh > 0
        else float(np.mean(estudo.ensemble_total.energia_diaria_kwh())) * 365.25
    )
    referencia = None
    origem = "a própria curva simulada"
    if comparacao is not None and comparacao.fatura.consumo_anual_kwh:
        referencia = float(comparacao.fatura.consumo_anual_kwh)
        cfg = estudo.configuracao
        if cfg.fatura is not None and cfg.fatura.consumo_mensal_kwh:
            origem = "consumo médio da fatura informado na tela (medido)"
        elif cfg.objetivo_fv == "compensar":
            origem = "consumo anual digitado como alvo de compensação"
        elif cfg.ajuste_conta is not None:
            origem = "consumo da conta de luz que deu tamanho à curva"
        else:
            origem = "consumo anual ponderado entre os perfis de ocupação"
    fator = (
        comparacao.fator_consumo if comparacao is not None
        else (referencia / simulado if referencia and simulado > 0 else 1.0)
    )
    geral_total = estudo.ensemble_total.resumo()["geral"]
    geral_backup = estudo.ensemble_backup.resumo()["geral"]
    linhas = [
        ("Consumo anual da curva simulada",
         f"{_n(simulado, 0, 'kWh')}",
         "dias simulados da curva, somados no balanço e levados a 365,25 dias"),
        ("Consumo anual de referência da conta",
         f"{_n(referencia or simulado, 0, 'kWh')}", origem),
        ("Fator aplicado à curva para a conta",
         f"{br_float(fator, 3)}",
         "referência ÷ simulado — muda o tamanho, preserva a forma"),
        ("Pico P95 da instalação", f"{_n(geral_total['pico_p95_kw'], 2, 'kW')}",
         "curva simulada, sem o fator — é ela que descreve o pico"),
        ("Consumo diário do quadro de backup",
         f"{_n(geral_backup['energia_diaria_media_kwh'], 1, 'kWh/dia')}",
         "ensemble do backup"),
        ("Pico P95 do quadro de backup", f"{_n(geral_backup['pico_p95_kw'], 2, 'kW')}",
         "dimensiona o inversor híbrido"),
    ]
    partes = [
        "## 2. Consumo",
        "",
        "A conta de luz de cada cenário é calculada dia a dia, com a **forma** da curva "
        "de carga e o **tamanho** do consumo de referência. É a forma que decide quanto "
        "da geração é consumida na hora (autoconsumo, vale a tarifa cheia) e quanto é "
        "injetada (vale a tarifa menos o Fio B). Uma curva concentrada à noite injeta "
        "mais e economiza menos que uma curva diurna com o mesmo consumo anual.",
        "",
        _tabela(["Grandeza", "Valor", "Origem / fórmula"], linhas),
    ]
    return partes


def _secao_geracao(estudo) -> list[str]:
    if not estudo.com_solar:
        return ["## 3. Geração solar", "", "Estudo sem geração solar."]
    cfg = estudo.configuracao
    serie = estudo.serie
    produtividade = serie.anual_kwh_por_kwp()
    kwp = estudo.potencia_fv_kwp
    dados = getattr(estudo, "compensacao", None) or {}
    linhas = [
        ("Potência do gerador", f"{_n(kwp, 2, 'kWp')}",
         f"definida por: {dados.get('definida_por', '—')}"),
    ]
    memoria = cfg.memoria
    if memoria is not None and getattr(memoria, "viavel", False):
        linhas.append((
            "Arranjo",
            f"{memoria.modulos_do_sistema} × {memoria.modulo.potencia_wp:.0f} Wp",
            f"{memoria.modulos_por_string} módulos por string × {memoria.strings_do_sistema} "
            f"strings; {memoria.inversores} × {memoria.inversor.modelo}; razão CC/CA "
            f"{br_float(memoria.razao_cc_ca, 2)}",
        ))
    linhas += [
        ("Fonte da série horária", str(serie.fonte),
         f"{serie.n_dias} dias ({serie.datas[0].isoformat()} a {serie.datas[-1].isoformat()})"),
        ("Inclinação / azimute", f"{_n(serie.inclinacao_deg, 1)}° / {_n(serie.azimute_deg, 0)}°",
         "azimute 0° = Norte (convenção interna)"),
        ("Perdas do sistema", f"{_n(serie.perdas_percent, 1)}%",
         "sujeira, cabos, mismatch, temperatura e inversor — parâmetro do PVGIS"),
        ("Produtividade", f"{_n(produtividade, 0, 'kWh/kWp/ano')}",
         "Σ energia diária da série ÷ nº de dias × 365,25"),
        ("Geração anual", f"{_n(kwp * produtividade, 0, 'kWh')}",
         f"{_n(kwp, 2)} kWp × {_n(produtividade, 0)} kWh/kWp"),
    ]
    por_estacao = serie.resumo().get("energia_diaria_media_kwh_por_kwp", {})
    partes = [
        "## 3. Geração solar",
        "",
        _tabela(["Grandeza", "Valor", "Origem / fórmula"], linhas),
    ]
    if por_estacao:
        partes += [
            "",
            _tabela(
                ["Estação", "kWh/kWp/dia", "Geração média do sistema (kWh/dia)"],
                [(e, br_float(v, 2), br_float(v * kwp, 1)) for e, v in por_estacao.items()],
            ),
        ]
    if len(getattr(cfg, "aguas", None) or []) > 1:
        partes += [
            "",
            "Mais de uma água: a série é a combinação das séries de cada orientação, "
            "ponderada pela potência de cada uma.",
        ]
    return partes


def _secao_curva(estudo) -> list[str]:
    """A curva adotada e o sol, hora a hora — o que decide autoconsumo e injeção."""
    comparacao = estudo.cenarios
    if comparacao is None or not estudo.com_solar:
        return []
    carga = _curva_horaria_kw(estudo.ensemble_total, comparacao.fator_consumo)
    sol = estudo.serie.janela_media_por_hora() * estudo.potencia_fv_kwp / 1000.0
    simultaneo = np.minimum(carga, sol)
    linhas = [
        (f"{h:02d}h", br_float(carga[h], 2), br_float(sol[h], 2),
         br_float(simultaneo[h], 2), br_float(max(0.0, sol[h] - carga[h]), 2))
        for h in range(24)
    ]
    linhas.append(("Total do dia (kWh)", br_float(carga.sum(), 1), br_float(sol.sum(), 1),
                   br_float(simultaneo.sum(), 1),
                   br_float(np.maximum(0.0, sol - carga).sum(), 1)))
    return [
        "## 4. A curva adotada e o sol, hora a hora",
        "",
        "Médias do ano (quatro estações) para leitura; **o balanço não usa estas "
        "médias** — ele roda dia a dia, pareando cada dia real da série solar com um "
        "dia simulado de carga. A média suaviza os dias nublados e os picos de carga, "
        "e por isso o autoconsumo desta tabela é otimista em relação ao do balanço.",
        "",
        _tabela(["Hora", "Carga média (kW)", "Geração média (kW)",
                 "Consumo simultâneo (kW)", "Excedente (kW)"], linhas),
    ]


def _secao_balanco(estudo) -> list[str]:
    comparacao = estudo.cenarios
    if comparacao is None:
        return []
    linhas = []
    for c in comparacao.cenarios:
        b = c.balanco
        if not b:
            continue
        linhas.append((
            c.nome.replace("Rede + ", ""),
            br_int(b["consumo"]), br_int(b["geracao"]),
            br_int(b.get("autoconsumo_direto", 0.0)), br_int(b.get("descarga", 0.0)),
            br_int(b["injecao"]), br_int(b["compra"]),
            _pct(c.autoconsumo_frac, 0) if b["geracao"] > 0 else "—",
        ))
    return [
        "## 5. Balanço energético do ano típico (kWh/ano)",
        "",
        "Ordem de despacho, a cada passo do dia: o sol atende a carga; o excedente "
        "carrega o banco (quando há); o que sobra é injetado; à noite o banco devolve o "
        "que guardou, respeitando a reserva de backup. Sem carga do banco pela rede: "
        "na tarifa simples a arbitragem só perde nas duas conversões.",
        "",
        _tabela(["Arranjo", "Consumo", "Geração", "Autoconsumo direto",
                 "Via bateria", "Injeção", "Compra da rede", "Autoconsumo"], linhas),
        "",
        "Fechamentos conferidos na seção 0: consumo = autoconsumo direto + via bateria "
        "+ compra; geração = autoconsumo direto + recarga do banco + injeção.",
    ]


def _secao_conta(estudo) -> list[str]:
    comparacao = estudo.cenarios
    if comparacao is None:
        return []
    f = comparacao.fatura
    cronograma = ", ".join(
        f"{ano}: {br_int(p * 100)}%" for ano, p in sorted(CRONOGRAMA_FIO_B.items())
    ) + f"; a partir de {max(CRONOGRAMA_FIO_B) + 1}: {br_int(FIO_B_APOS_TRANSICAO * 100)}% (hipótese)"
    premissas = [
        ("Tarifa (R$/kWh, com tributos)", _r(f.tarifa_brl_kwh, 4), "fatura do cliente / tela"),
        ("Adicional de bandeira (R$/kWh)", _r(f.bandeira_brl_kwh, 4),
         "zero quando a tarifa já é a média da fatura"),
        ("Tarifa efetiva do kWh comprado", _r(f.tarifa_efetiva_brl_kwh, 4), "tarifa + bandeira"),
        ("Custo de disponibilidade", f"{_n(f.custo_disponibilidade_kwh, 0)} kWh/mês",
         "30 mono, 50 bi, 100 trifásico — piso de 12 × ao ano"),
        ("Participação do Fio B na tarifa", _pct(f.participacao_fio_b, 0),
         "varia por distribuidora — confirmar na resolução tarifária"),
        ("Ano de conexão", str(f.ano_conexao),
         f"Fio B cobrado sobre o compensado: {_pct(fator_fio_b(f.ano_conexao), 0)} neste ano"),
        ("Fio B por kWh compensado (ano 1)", _r(f.fio_b_brl_kwh(), 4),
         f"{_r(f.tarifa_brl_kwh, 4)} × {_pct(f.participacao_fio_b, 0)} × "
         f"{_pct(fator_fio_b(f.ano_conexao), 0)}"),
        ("Valor do kWh injetado e compensado (ano 1)", _r(f.valor_injecao_brl_kwh, 4),
         "tarifa efetiva − Fio B"),
        ("Reajuste tarifário", f"{_pct(f.escalada_tarifaria_ano, 1)} a.a.", "premissa"),
    ]
    partes = [
        "## 6. Conta de luz e compensação da energia injetada",
        "",
        "Regra aplicada (Lei 14.300/2022, sistema conectado a partir de 2023):",
        "",
        "1. A energia consumida **na hora** em que é gerada não passa pelo medidor: "
        "evita a tarifa cheia, sem Fio B.",
        "2. A energia **injetada** vira crédito em kWh e abate o que se compra da rede — "
        "inclusive a bandeira, que incide sobre o faturado depois da compensação.",
        "3. Cada kWh **compensado** paga a fração do Fio B do ano: " + cronograma + ".",
        "4. A conta nunca fica abaixo do custo de disponibilidade (doze mínimos por "
        "ano); o crédito que passaria disso não tem onde ser usado.",
        "5. Compensação anual: o crédito de um mês abate outro (validade de 60 meses). "
        "Descreve o ano típico em regime, não o primeiro mês.",
        "",
        "Fórmulas:",
        "",
        "- compensado = mín(injeção, máx(0, compra − piso))",
        "- faturado = máx(piso, compra − compensado)",
        "- conta = faturado × tarifa efetiva + compensado × Fio B do ano",
        "",
        _tabela(["Premissa", "Valor", "Origem / fórmula"], premissas),
        "",
    ]
    linhas = []
    for c in comparacao.cenarios:
        b = c.balanco
        if not b:
            continue
        d = f.detalhar_conta(b["compra"], b["injecao"])
        linhas.append((
            c.nome.replace("Rede + ", ""),
            br_int(d["compra_kwh"]), br_int(d["injecao_kwh"]), br_int(d["compensado_kwh"]),
            br_int(d["credito_sem_uso_kwh"]), br_int(d["faturado_kwh"]),
            _r(d["energia_brl"]), _r(d["fio_b_brl"]), _r(d["conta_brl"]),
            _r(c.economia_anual_brl) if c.composicao.n_fontes else "—",
        ))
    partes += [
        "Conta do primeiro ano, por arranjo:",
        "",
        _tabela(["Arranjo", "Compra (kWh)", "Injeção (kWh)", "Compensado (kWh)",
                 "Crédito sem uso (kWh)", "Faturado (kWh)", "Energia (R$)", "Fio B (R$)",
                 "Conta (R$/ano)", "Economia (R$/ano)"], linhas),
    ]
    return partes


def _secao_investimento(estudo) -> list[str]:
    precos = getattr(estudo, "precos_por_topologia", None) or {}
    cfg = estudo.configuracao
    partes = ["## 7. Investimento", ""]
    if not precos:
        comparacao = estudo.cenarios
        if comparacao is None:
            return []
        partes.append(
            "Sem tabela de kit nesta potência: o investimento de cada arranjo vem da "
            f"curva de R$/kWp do padrão de obra '{cfg.padrao_capex}'"
            + (" ou do valor informado na tela" if cfg.capex_fv_brl else "")
            + ", mais o armazenamento pelo catálogo."
        )
        partes += ["", _tabela(
            ["Arranjo", "Investimento"],
            [(c.nome, _r(c.capex_brl)) for c in comparacao.cenarios if c.capex_brl > 0],
        )]
        return partes

    from ..pv.kits import APURADO_EM, FONTE, TABELA_KIT, preco_kit

    kwp = estudo.potencia_fv_kwp
    degraus = sorted(TABELA_KIT)
    abaixo = max([p for p in degraus if p <= kwp], default=degraus[0])
    acima = min([p for p in degraus if p >= kwp], default=degraus[-1])
    mao = float(cfg.mao_de_obra_brl_kwp or 0.0)
    ca = float(cfg.material_ca_brl_kwp or 0.0)
    bloco_kwh = float(cfg.premissas.bloco_bateria_kwh or 5.0)
    bloco_brl = float(cfg.premissas.bloco_bateria_brl or 0.0)
    partes += [
        f"Fonte: {FONTE}, apurada em {APURADO_EM.strftime('%d/%m/%Y')}. Entre dois "
        f"degraus o preço é interpolado em linha reta — aqui, entre {br_float(abaixo, 1)} e "
        f"{br_float(acima, 1)} kWp para {br_float(kwp, 2)} kWp. A tabela é **equipamento "
        "posto**: mão de obra e material CA entram somados por kWp; o banco, em blocos.",
        "",
    ]
    linhas = []
    for topologia, dados in precos.items():
        if dados.get("capex_brl") is None:
            linhas.append((dados["nome"], "—", "—", "—", "—", "—", "sem preço de tabela"))
            continue
        kit = preco_kit(kwp, str(dados["coluna"])) or 0.0
        blocos = int(dados.get("blocos") or 0) if dados["com_bateria"] else 0
        banco = blocos * bloco_brl if blocos else 0.0
        hibrido = float(dados.get("inversor_a_parte_brl") or 0.0)
        linhas.append((
            dados["nome"],
            f"{_r(kit)} (coluna {dados['coluna']})",
            _r(mao * kwp), _r(ca * kwp),
            f"{blocos} × {_r(bloco_brl)} = {_r(banco)}" if blocos else "—",
            _r(hibrido) if hibrido else "—",
            _r(dados["capex_brl"]),
        ))
    partes.append(_tabela(
        ["Proposta", "Kit", f"Mão de obra ({_r(mao)}/kWp)", f"Material CA ({_r(ca)}/kWp)",
         f"Banco (blocos de {br_float(bloco_kwh, 1)} kWh)", "Híbrido à parte", "Total"],
        linhas,
    ))
    avisos = [f"- {d['nome']}: {d['aviso']}" for d in precos.values() if d.get("aviso")]
    if avisos:
        partes += ["", *avisos]
    if not mao and not ca:
        partes += ["", "⚠️ Mão de obra e material CA estão em **zero**: o investimento "
                   "é só o material. Informe os dois na tela antes de apresentar preço de "
                   "obra entregue."]
    return partes


def _secao_fluxo(estudo) -> list[str]:
    comparacao = estudo.cenarios
    if comparacao is None:
        return []
    financeiros = list((getattr(estudo, "financeiro", None) or {}).values())
    if not financeiros:
        principal = estudo.financeiro_do_cenario()
        financeiros = [principal] if principal is not None else []
    if not financeiros:
        return []
    p = comparacao.premissas
    f = comparacao.fatura
    partes = [
        "## 8. Fluxo de caixa de cada proposta",
        "",
        _tabela(["Premissa", "Valor"], [
            ("Horizonte", f"{p.anos_analise} anos"),
            ("Taxa de desconto (VPL)", f"{_pct(p.taxa_desconto_ano, 1)} a.a."),
            ("Reajuste tarifário", f"{_pct(f.escalada_tarifaria_ano, 1)} a.a."),
            ("Perda de rendimento do módulo", f"{_pct(p.degradacao_fv_ano, 2)} a.a."),
            ("Operação e manutenção", f"{_pct(p.opex_percent_do_capex_ano, 1)} do investimento ao ano, reajustado"),
            ("Queda do preço da bateria (troca)", f"{_pct(p.queda_preco_bateria_ano, 1)} a.a."),
        ]),
        "",
        "Como cada ano é montado:",
        "",
        "1. **Economia a preços do ano 1**: a conta de luz refeita com o Fio B do ano "
        "calendário e com a geração reduzida pela perda do módulo (o autoconsumo "
        "perdido volta a ser compra; a injeção encolhe na mesma proporção).",
        "2. **Economia nominal** = economia a preços do ano 1 × (1 + reajuste)^(ano − 1).",
        "3. **Caixa** = economia nominal − operação − troca do banco (no ano em que a "
        "retenção cruza o fim de vida).",
        "4. **Payback simples** = investimento ÷ economia do ano 1. **Retorno pelo "
        "fluxo** = ano em que o caixa acumulado cruza zero. **TIR** e **VPL** saem do "
        "mesmo caixa.",
        "5. O **valor da resiliência** (energia que deixa de faltar × custo da "
        "interrupção) não entra no caixa — não aparece na conta de luz. Aparece à parte, "
        "no VPL com resiliência.",
    ]
    for fin in financeiros:
        cenario = comparacao.por_chave(fin.cenario)
        anuais = {int(a["ano"]): a for a in (cenario.economias_por_ano if cenario else [])}
        linhas = []
        acumulado = -fin.capex_brl
        for linha in fin.fluxo.itertuples():
            ano = int(linha.ano)
            a = anuais.get(ano, {})
            acumulado += float(linha.fluxo_brl)
            linhas.append((
                str(ano), str(a.get("ano_calendario", "—")),
                _r(a.get("fio_b_brl_kwh"), 4) if a else "—",
                br_float(a.get("fator_geracao", 1.0), 3) if a else "—",
                _r(a.get("economia_brl")) if a else "—",
                _r(linha.economia_conta_brl), _r(linha.custo_operacional_brl),
                _r(linha.troca_brl) if linha.troca_brl else "—",
                _r(linha.fluxo_brl), _r(acumulado),
            ))
        economia = fin.economia_ano1_brl
        partes += [
            "",
            f"### {fin.nome}",
            "",
            _tabela(["Indicador", "Valor", "Conta"], [
                ("Investimento", _r(fin.capex_brl), "seção 7"),
                ("Economia na conta, ano 1", _r(economia), "seção 6"),
                ("Payback simples", _anos(fin.payback_simples_anos),
                 f"{_r(fin.capex_brl)} ÷ {_r(economia)}" if economia > 0 else "—"),
                ("Retorno pelo fluxo de caixa", _anos(fin.payback_anos), "acumulado cruza zero"),
                ("TIR", _pct(fin.tir, 1) if fin.tir is not None else "—", "caixa abaixo"),
                ("VPL do caixa", _r(fin.vpl_brl), f"a {_pct(p.taxa_desconto_ano, 0)} a.a."),
                ("Valor da resiliência, ano 1", _r(fin.valor_resiliencia_ano1_brl),
                 "fora do caixa"),
                ("VPL somando a resiliência", _r(fin.vpl_com_resiliencia_brl), "decisão de arranjo"),
                ("Autonomia garantida", f"{br_float(fin.autonomia_h, 1)} h" if fin.autonomia_h else "nenhuma",
                 "95% dos casos, pior par estação/hora"),
            ]),
            "",
            _tabela(["Ano", "Calendário", "Fio B (R$/kWh)", "Fator de geração",
                     "Economia (preços ano 1)", "Economia nominal", "Operação", "Troca",
                     "Caixa", "Acumulado"], linhas),
        ]
    return partes


def _secao_bateria(estudo) -> list[str]:
    if estudo.recomendado is None and not estudo.resiliencia:
        return []
    cfg = estudo.configuracao
    p = estudo.cenarios.premissas if estudo.cenarios is not None else cfg.premissas
    partes = [
        "## 9. Bateria e resiliência",
        "",
        f"Critério: o conjunto de menor investimento que atravessa "
        f"{br_float(cfg.autonomia_alvo_h, 0)} h de falta em pelo menos "
        f"{_pct(cfg.confiabilidade_alvo, 0)} dos casos"
        + (", no pior par de estação e hora de início." if cfg.exigir_pior_caso else "."),
        "",
    ]
    if estudo.recomendado is not None:
        c = estudo.recomendado.conjunto
        partes.append(_tabela(["Item", "Valor"], [
            ("Banco", f"{c.modulos} × {' '.join(x for x in (c.bateria.fabricante, c.bateria.modelo) if x)}"),
            ("Capacidade nominal / útil",
             f"{br_float(c.capacidade_nominal_kwh, 1)} / {br_float(c.energia_util_kwh, 1)} kWh"),
            ("Inversor híbrido", f"{c.inversor.fabricante} {c.inversor.modelo}"),
            ("Potência contínua / de pico",
             f"{br_float(c.potencia_descarga_kw, 1)} / {br_float(c.potencia_pico_kw, 1)} kW"),
            ("Autonomia garantida",
             f"{br_float(estudo.recomendado.autonomia_garantida_h(cfg.confiabilidade_alvo, cfg.exigir_pior_caso), 1)} h"),
            ("Reserva de backup (não ciclada)", _pct(p.reserva_backup_frac, 0)),
        ]))
    else:
        partes.append("Nenhum conjunto do catálogo cumpriu a meta; ver o dossiê.")
    unidade = (
        f"{_r(p.custo_interrupcao_brl_evento)} por evento evitado"
        if p.custo_interrupcao_brl_evento > 0
        else f"{_r(p.custo_interrupcao_brl_kwh, 2)} por kWh não suprido"
    )
    partes += [
        "",
        f"Valor da resiliência: {unidade}, com {br_float(p.interrupcoes_por_ano, 1)} "
        f"interrupções por ano de {br_float(p.duracao_media_interrupcao_h, 1)} h em média. "
        "É o parâmetro mais sensível da análise do banco e vem do cliente; não entra no "
        "caixa das propostas.",
    ]
    return partes


def _secao_comercial(estudo, comercial) -> list[str]:
    if comercial is None:
        return []
    from .apresentacao import parcela_price

    principal = estudo.financeiro_do_cenario()
    if principal is None:
        return []
    entrada = min(max(0.0, float(comercial.entrada_financiamento_brl)), principal.capex_brl)
    financiado = principal.capex_brl - entrada
    parcela = parcela_price(financiado, comercial.juros_financiamento_am,
                            comercial.prazo_financiamento_meses,
                            comercial.carencia_financiamento_meses)
    servicos = comercial.seguro_brl_mes + comercial.gerenciamento_brl_mes
    return [
        "## 10. Condições comerciais da apresentação",
        "",
        "Premissas de venda, não resultado de cálculo — conferir antes de enviar.",
        "",
        _tabela(["Item", "Valor", "Conta"], [
            ("Proposta principal", principal.nome, ""),
            ("Valor do sistema", _r(principal.capex_brl, 2), "seção 7"),
            ("Entrada", _r(entrada, 2), "tela"),
            ("Valor financiado", _r(financiado, 2), "sistema − entrada"),
            ("Juros", f"{br_float(comercial.juros_financiamento_am * 100, 2)}% a.m.", "tela"),
            ("Carência", f"{comercial.carencia_financiamento_meses} meses",
             "juros capitalizados no saldo"),
            ("Prazo", f"{comercial.prazo_financiamento_meses} meses", "tela"),
            ("Parcela (tabela Price)", _r(parcela, 2),
             "saldo × i ÷ (1 − (1 + i)^−n), saldo = financiado × (1 + i)^carência"),
            ("Seguro + gerenciamento", f"{_r(servicos, 2)}/mês", "tela"),
            ("Economia mensal média, ano 1", _r(principal.economia_mensal_brl, 2),
             "economia do ano 1 ÷ 12"),
            ("Saldo mensal financiado, ano 1",
             _r(principal.economia_mensal_brl - servicos - parcela, 2),
             "economia − serviços − parcela"),
        ]),
    ]


def _secao_limites() -> list[str]:
    return [
        "## 12. O que este cálculo não modela",
        "",
        "- **Tarifa branca e Grupo A** no quadro de arranjos: a conta é de tarifa simples. "
        "Posto tarifário e demanda contratada ficam de fora da comparação.",
        "- **Bandeira mês a mês**: entra como adicional médio (R$/kWh) sobre todo kWh "
        "faturado.",
        "- **Tributos sobre a energia compensada**: seguem a tarifa informada; diferenças "
        "de ICMS entre estados sobre a TUSD compensada não são modeladas.",
        "- **Mês a mês**: a compensação é anual, com o piso de doze mínimos. Um mês de "
        "consumo abaixo do mínimo com crédito sobrando noutro é tratado como a "
        "distribuidora trata no ano em regime, não mês a mês.",
        "- **Sombreamento**: só o fator de área aproveitável do telhado; não há estudo de "
        "sombra hora a hora.",
        "- **Regra do Fio B a partir de 2029**: a lei a remete a regulamentação futura; "
        "adotou-se 100% como hipótese conservadora.",
    ]


def montar_memoria(estudo, comercial=None) -> str:
    """A memória de cálculo inteira, em Markdown."""
    cfg = estudo.configuracao
    itens = conferencias(estudo, comercial)
    partes: list[str] = [
        f"# Memória de cálculo — {cfg.nome}",
        "",
        f"Gerada em {date.today().strftime('%d/%m/%Y')} pela PACEcalculator v{__version__}. "
        "Este documento lista cada premissa e refaz cada conta que sustenta o estudo e a "
        "proposta. Os números do dossiê, da tela e da apresentação saem dos mesmos "
        "objetos que estão aqui; a seção 0 confere isso automaticamente.",
        "",
        *_secao_conferencias(itens),
        "",
        *_secao_identificacao(estudo),
        "",
        *_secao_consumo(estudo),
        "",
        *_secao_geracao(estudo),
    ]
    for secao in (_secao_curva(estudo), _secao_balanco(estudo), _secao_conta(estudo),
                  _secao_investimento(estudo), _secao_fluxo(estudo),
                  _secao_bateria(estudo), _secao_comercial(estudo, comercial)):
        if secao:
            partes += ["", *secao]
    if estudo.avisos:
        partes += ["", "## 11. Avisos do estudo", "",
                   *[f"- {a}" for a in dict.fromkeys(estudo.avisos)]]
    partes += ["", *_secao_limites(), ""]
    return "\n".join(partes)
