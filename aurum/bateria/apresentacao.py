"""
A apresentação comercial: os 19 slides que acompanham o dossiê.

O dossiê (:mod:`aurum.bateria.documento`) é o documento que defende o
dimensionamento. Esta é a peça que vende: capa, quem somos, o sistema, três
formas de pagar, o retorno, e o anexo técnico. O esqueleto em LaTeX vive em
``apresentacao/`` na raiz do repositório e **não é escrito por este módulo** —
ele só escreve o ``dados.tex``, o arquivo com tudo que muda de proposta para
proposta, e copia o esqueleto e as imagens fixas para junto dele.

A fronteira entre os dois arquivos é o que faz a automação ser simples: o
esqueleto é diagramação, muda por commit; o ``dados.tex`` são valores já
formatados, muda por cliente. Nada aqui produz LaTeX de layout.

De onde vem cada número
-----------------------

**Do estudo**, sem intervenção: o cliente e o local, a potência, o módulo e a
quantidade, o inversor, a área ocupada, a geração anual, o banco de baterias,
o investimento, a economia do primeiro ano, o payback, a TIR e o fluxo de
caixa ano a ano — tudo do cenário de fontes que a proposta apresenta.

**Do operador**, porque o estudo não tem como saber: quem assina (a
:class:`~aurum.bateria.documento.DadosCapa`) e as condições comerciais
(:class:`OpcoesComerciais`) — juros e prazo do financiamento, a mensalidade
do leasing, o preço do seguro e do gerenciamento, as garantias. Cada uma tem
um padrão declarado, para a apresentação sair mesmo sem ninguém preencher;
mas são premissas de venda, não resultado de cálculo, e a proposta que for
para o cliente precisa tê-las conferidas.
"""
from __future__ import annotations

import logging
import shutil
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Sequence

from .. import marca
from ..proposal.latex import esc
from ..proposal.render import compilar_pdf, encontrar_compilador
from ..util.numfmt import br_float, br_int, br_money
from .documento import DadosCapa
from .estudo import ResultadoEstudo

LOGGER = logging.getLogger(__name__)

__all__ = [
    "OpcoesComerciais",
    "PASTA_MODELO",
    "dados_da_apresentacao",
    "dados_tex",
    "escrever_apresentacao",
]

#: Onde está o esqueleto: ``apresentacao/`` na raiz do repositório.
PASTA_MODELO = Path(__file__).resolve().parents[2] / "apresentacao"

#: Os anos que a tabela e o gráfico do fluxo acumulado mostram. Os primeiros
#: três porque é onde o financiado fica negativo; 5 e 10 porque é onde os
#: cenários se cruzam; o último porque é o horizonte.
ANOS_FLUXO_PADRAO: tuple[int, ...] = (1, 2, 3, 5, 10, 25)

MESES_PT = (
    "janeiro", "fevereiro", "março", "abril", "maio", "junho",
    "julho", "agosto", "setembro", "outubro", "novembro", "dezembro",
)


@dataclass
class OpcoesComerciais:
    """
    As condições de venda. O estudo calcula o sistema; isto é como se paga.

    Os padrões existem para a apresentação compilar de primeira, e são os
    da proposta que serviu de modelo. Nenhum deles vem de cálculo.
    """

    #: Nome que aparece na capa e nos títulos. ``None`` usa o nome do estudo.
    cliente: str | None = None
    data: date = field(default_factory=date.today)
    #: Chave do cenário de fontes que a proposta apresenta (``solar``,
    #: ``solar+bateria``...). ``None`` escolhe: solar com o banco recomendado
    #: quando o estudo tem os dois, senão só solar.
    cenario: str | None = None
    validade_dias: int = 15

    # -- serviços mensais, cobrados nas compras à vista e financiada ---------
    seguro_brl_mes: float = 40.0
    gerenciamento_brl_mes: float = 40.0

    # -- financiamento (tabela Price, juros capitalizados na carência) --------
    juros_financiamento_am: float = 0.0149
    prazo_financiamento_meses: int = 60
    carencia_financiamento_meses: int = 3

    # -- leasing ("as a service") ---------------------------------------------
    #: A mensalidade como fração da economia mensal do primeiro ano. Seguro,
    #: manutenção e gerenciamento estão dentro dela.
    leasing_fracao_da_economia: float = 0.86
    prazo_leasing_anos: int = 10
    carencia_leasing_meses: int = 3

    # -- o que vai na lista de equipamentos ----------------------------------
    estrutura: str = "Telhado"
    garantia_inversor_anos: int = 10
    garantia_modulo_anos: str = "10 / 20 anos"
    garantia_bateria_anos: int = 5
    garantia_estrutura_anos: int = 20

    # -- números institucionais do slide de monitoramento --------------------
    clientes_geridos: str = "+6.000"
    potencia_gerida: str = "150 MWp"

    #: Os anos mostrados no fluxo acumulado. São filtrados pelo horizonte do
    #: estudo: pedir o ano 25 num estudo de 15 anos não inventa dez anos.
    anos_fluxo: Sequence[int] = ANOS_FLUXO_PADRAO

    def as_dict(self) -> dict[str, Any]:
        dados = asdict(self)
        dados["data"] = self.data.isoformat()
        dados["anos_fluxo"] = list(self.anos_fluxo)
        return dados


# ----------------------------------------------------------------------------
# Finanças da proposta: o que o estudo não calcula
# ----------------------------------------------------------------------------
def parcela_price(principal: float, juros_am: float, prazo_meses: int, carencia_meses: int = 0) -> float:
    """
    A prestação fixa de um financiamento, com os juros da carência somados ao
    saldo antes de começar a amortizar — que é como o banco faz.
    """
    if principal <= 0 or prazo_meses <= 0:
        return 0.0
    saldo = principal * (1.0 + juros_am) ** carencia_meses
    if juros_am <= 0:
        return saldo / prazo_meses
    return saldo * juros_am / (1.0 - (1.0 + juros_am) ** (-prazo_meses))


def _parcelas_no_ano(ano: int, carencia_meses: int, prazo_meses: int) -> int:
    """Quantas prestações caem no ano ``ano`` (1 = primeiro ano da operação)."""
    inicio, fim = 12 * (ano - 1) + 1, 12 * ano
    primeira, ultima = carencia_meses + 1, carencia_meses + prazo_meses
    return max(0, min(fim, ultima) - max(inicio, primeira) + 1)


def _acumulado(valores: Sequence[float]) -> list[float]:
    saldo, saida = 0.0, []
    for v in valores:
        saldo += v
        saida.append(saldo)
    return saida


# ----------------------------------------------------------------------------
# Os dados, como valores — antes de virar LaTeX
# ----------------------------------------------------------------------------
def _cenario_da_proposta(estudo: ResultadoEstudo, chave: str | None):
    """
    O cenário de fontes que a apresentação vende.

    Sem escolha explícita, é o que o estudo concluiu: solar mais o banco
    recomendado quando os dois existem, senão só o solar. Apresentar o
    cenário de melhor VPL seria vender coisa diferente da que o dossiê
    defende.
    """
    comparacao = estudo.cenarios
    if comparacao is None:
        raise ValueError("O estudo não tem comparação de cenários; a apresentação precisa dela.")
    if chave is None:
        chave = "solar+bateria" if (estudo.com_solar and estudo.recomendado is not None) else "solar"
        if comparacao.por_chave(chave) is None and estudo.com_solar:
            chave = "solar"
    cenario = comparacao.por_chave(chave)
    if cenario is None:
        disponiveis = ", ".join(c.composicao.chave for c in comparacao.cenarios)
        raise ValueError(f"Cenário '{chave}' não existe no estudo (há: {disponiveis}).")
    if cenario.capex_brl <= 0:
        raise ValueError(f"O cenário '{chave}' não investe nada; não há proposta a apresentar.")
    return cenario


def _data_por_extenso(d: date) -> str:
    return f"{d.day} de {MESES_PT[d.month - 1]} de {d.year}"


def dados_da_apresentacao(
    estudo: ResultadoEstudo,
    capa: DadosCapa | None = None,
    opcoes: OpcoesComerciais | None = None,
) -> dict[str, Any]:
    """
    Tudo que a apresentação mostra, como números e textos — ainda não LaTeX.

    É esta a função que os testes conferem, porque é aqui que um número do
    estudo pode ir para o slide errado. A conversão para ``dados.tex`` é só
    formatação.
    """
    capa = capa or DadosCapa()
    opcoes = opcoes or OpcoesComerciais()
    cfg = estudo.configuracao
    cenario = _cenario_da_proposta(estudo, opcoes.cenario)
    com_bateria = cenario.composicao.bateria and estudo.recomendado is not None
    conjunto = estudo.recomendado.conjunto if com_bateria else None
    layout = cfg.layout
    memoria = cfg.memoria
    fatura = estudo.cenarios.fatura
    premissas = estudo.cenarios.premissas

    # -- o sistema ------------------------------------------------------------
    if memoria is not None:
        potencia_kwp = memoria.potencia_do_sistema_kwp
        qtd_modulos = memoria.modulos_do_sistema
    elif layout is not None:
        potencia_kwp = layout.potencia_kwp
        qtd_modulos = layout.quantidade
    else:
        potencia_kwp = estudo.potencia_fv_kwp
        qtd_modulos = None
    modulo = layout.modulo if layout is not None else None

    # O inversor do gerador solar é o da memória de cálculo. O híbrido do
    # conjunto de baterias é outro equipamento — o que faz o backup — e vai
    # na linha do banco. Só quando o estudo não dimensionou arranjo (potência
    # vinda do consumo) o híbrido responde sozinho pela linha do inversor.
    inversor_kw: float | None = None
    inversor_nome: str | None = None
    inversor_tipo: str | None = None
    inversores = 1
    hibrido_kw: float | None = None
    hibrido_nome: str | None = None
    if memoria is not None:
        inversor_kw = memoria.inversor.potencia_ca_w / 1000.0
        inversor_nome = f"{memoria.inversor.fabricante} {memoria.inversor.modelo}"
        inversores = memoria.inversores
        inversor_tipo = "string"
    if conjunto is not None:
        hibrido_kw = conjunto.inversor.potencia_ca_nominal_kw
        hibrido_nome = f"{conjunto.inversor.fabricante} {conjunto.inversor.modelo}"
        if memoria is None:
            inversor_kw, inversor_nome, inversor_tipo = hibrido_kw, hibrido_nome, "híbrido"
            hibrido_kw = hibrido_nome = None

    # -- o dinheiro -----------------------------------------------------------
    capex = float(cenario.capex_brl)
    economia_anual = float(cenario.economia_anual_brl)
    economia_mensal = economia_anual / 12.0
    servicos_mes = opcoes.seguro_brl_mes + opcoes.gerenciamento_brl_mes
    parcela = parcela_price(
        capex, opcoes.juros_financiamento_am,
        opcoes.prazo_financiamento_meses, opcoes.carencia_financiamento_meses,
    )
    mensalidade_leasing = economia_mensal * opcoes.leasing_fracao_da_economia

    # O fluxo do estudo já traz a escalada tarifária e o custo de operação. A
    # apresentação só lhe soma o que o estudo não conhece: os serviços
    # mensais e, em cada forma de pagamento, o que se paga ao banco ou ao
    # locador. Os três cenários saem da mesma curva, e por isso comparáveis.
    horizonte = int(premissas.anos_analise)
    fluxo_estudo = [float(v) for v in cenario.fluxo["fluxo_brl"].tolist()][:horizonte]
    a_vista = [-capex] + [f - 12 * servicos_mes for f in fluxo_estudo]
    financiado = [0.0] + [
        f - 12 * servicos_mes
        - parcela * _parcelas_no_ano(ano, opcoes.carencia_financiamento_meses,
                                     opcoes.prazo_financiamento_meses)
        for ano, f in enumerate(fluxo_estudo, start=1)
    ]
    leasing = [0.0] + [
        f - mensalidade_leasing * _parcelas_no_ano(
            ano, opcoes.carencia_leasing_meses, 12 * opcoes.prazo_leasing_anos)
        for ano, f in enumerate(fluxo_estudo, start=1)
    ]
    acumulados = {
        "a_vista": _acumulado(a_vista),
        "financiado": _acumulado(financiado),
        "leasing": _acumulado(leasing),
    }
    anos = [a for a in opcoes.anos_fluxo if 1 <= a <= horizonte]
    if not anos or anos[-1] != horizonte:
        anos.append(horizonte)
    fluxo_acumulado = [
        {"ano": a, **{k: v[a] for k, v in acumulados.items()}} for a in anos
    ]

    cliente = opcoes.cliente or cfg.nome
    return {
        "cliente": cliente,
        "local": cfg.nome,
        "data": opcoes.data,
        "responsavel": capa.responsavel,
        "crea": capa.crea,
        "credenciais": capa.credenciais,
        "nomear_marcas": bool(capa.nomear_marcas),
        "cenario": cenario.composicao.chave,
        "sistema": {
            "potencia_kwp": potencia_kwp,
            "modulo_wp": modulo.potencia_wp if modulo is not None else None,
            "modulo_nome": str(modulo) if modulo is not None else None,
            "modulos": qtd_modulos,
            "inversor_kw": inversor_kw,
            "inversor_nome": inversor_nome,
            "inversor_tipo": inversor_tipo,
            "inversores": inversores,
            "hibrido_kw": hibrido_kw,
            "hibrido_nome": hibrido_nome,
            "area_ocupada_m2": layout.area_ocupada_m2 if layout is not None else None,
            "geracao_anual_kwh": float(cenario.geracao_fv_kwh_ano),
            "bess_blocos": conjunto.modulos if conjunto is not None else 0,
            "bess_kwh_util": conjunto.energia_util_kwh if conjunto is not None else 0.0,
            "bess_kwh_nominal": conjunto.capacidade_nominal_kwh if conjunto is not None else 0.0,
            "bess_nome": (f"{conjunto.bateria.fabricante} {conjunto.bateria.modelo}"
                          if conjunto is not None else None),
            "estrutura": opcoes.estrutura,
        },
        "economia": {
            "capex_brl": capex,
            "economia_anual_brl": economia_anual,
            "economia_mensal_brl": economia_mensal,
            "payback_anos": cenario.payback_anos,
            "tir": cenario.tir,
            "horizonte_anos": horizonte,
            "escalada_tarifaria_ano": float(fatura.escalada_tarifaria_ano),
            "taxa_desconto_ano": float(premissas.taxa_desconto_ano),
            "seguro_brl_mes": opcoes.seguro_brl_mes,
            "gerenciamento_brl_mes": opcoes.gerenciamento_brl_mes,
            "parcela_brl_mes": parcela,
            "mensalidade_leasing_brl_mes": mensalidade_leasing,
        },
        "fluxo_acumulado": fluxo_acumulado,
        "opcoes": opcoes.as_dict(),
    }


# ----------------------------------------------------------------------------
# dados.tex
# ----------------------------------------------------------------------------
def _brl(valor: float | None, casas: int = 0) -> str:
    """``R\\$ 1.234`` com o cifrão já escapado; negativo com travessão curto."""
    if valor is None:
        return "---"
    texto = br_money(abs(valor), casas=casas, simbolo="")
    return rf"{'--' if valor < 0 else ''}R\$ {texto}"


def _saldo(valor: float) -> str:
    """Célula do semáforo: verde de zero para cima, vermelho abaixo."""
    return rf"\{'negativo' if valor < 0 else 'positivo'}{{{_brl(valor, 2)}}}"


def _cmd(nome: str, valor: str) -> str:
    return rf"\newcommand{{\{nome}}}{{{valor}}}"


def dados_tex(dados: dict[str, Any]) -> str:
    """O ``dados.tex`` a partir do dicionário de :func:`dados_da_apresentacao`."""
    s = dados["sistema"]
    e = dados["economia"]
    o = dados["opcoes"]
    nomear = dados["nomear_marcas"]

    if s["modulos"] is not None and s["potencia_kwp"]:
        potencia = f"{br_float(s['potencia_kwp'], 2)} kWp"
    else:
        potencia = f"{br_float(s['potencia_kwp'], 1)} kWp"
    inversor_kw = f"{br_float(s['inversor_kw'], 1)} kW" if s["inversor_kw"] else "---"
    inversor = f"{inversor_kw} {s['inversor_tipo']}" if s["inversor_kw"] else "---"
    inversor_generico = f"Inversor {s['inversor_tipo']}" if s["inversor_tipo"] else "Inversor"
    if s["bess_blocos"]:
        bess = (f"{s['bess_blocos']} x {br_float(s['bess_kwh_nominal'] / s['bess_blocos'], 1)} kWh "
                f"({br_float(s['bess_kwh_util'], 1)} kWh úteis)")
        if s["hibrido_kw"]:
            bess += f" + híbrido {br_float(s['hibrido_kw'], 1)} kW"
    else:
        bess = "---"
    payback = (f"{br_int(round(e['payback_anos'] * 12))} meses" if e["payback_anos"]
               else "não se paga no horizonte")
    tir = f"{br_float(e['tir'] * 100, 1)}\\% a.a." if e["tir"] is not None else "---"
    servicos = e["seguro_brl_mes"] + e["gerenciamento_brl_mes"]
    parcela, leasing = e["parcela_brl_mes"], e["mensalidade_leasing_brl_mes"]
    em = e["economia_mensal_brl"]

    def nome_ou_generico(nome: str | None, generico: str) -> str:
        return esc(nome) if (nomear and nome) else generico

    linhas_equip = [
        rf"  \linhaequipamento{{Inversor}}{{{nome_ou_generico(s['inversor_nome'], inversor_generico)}}}"
        rf"{{{s['inversores']}}}{{{inversor_kw}}}{{{o['garantia_inversor_anos']} anos}}",
    ]
    if s["hibrido_kw"]:
        linhas_equip.append(
            rf"  \linhaequipamento{{Inversor híbrido}}{{{nome_ou_generico(s['hibrido_nome'], 'Inversor híbrido')}}}"
            rf"{{1}}{{{br_float(s['hibrido_kw'], 1)} kW}}{{{o['garantia_inversor_anos']} anos}}"
        )
    if s["modulo_wp"]:
        linhas_equip.append(
            rf"  \linhaequipamento{{Módulo}}{{{nome_ou_generico(s['modulo_nome'], 'Módulo bifacial')}}}"
            rf"{{{s['modulos'] if s['modulos'] is not None else '---'}}}{{{br_int(s['modulo_wp'])} Wp}}"
            rf"{{{esc(o['garantia_modulo_anos'])}}}"
        )
    if s["bess_blocos"]:
        linhas_equip.append(
            rf"  \linhaequipamento{{Bateria}}{{{nome_ou_generico(s['bess_nome'], 'Bloco LFP')}}}"
            rf"{{{s['bess_blocos']}}}{{{br_float(s['bess_kwh_nominal'], 1)} kWh}}"
            rf"{{{o['garantia_bateria_anos']} anos}}"
        )
    linhas_equip.append(
        rf"  \linhaequipamento{{Estrutura}}{{{esc(s['estrutura'])}}}{{---}}{{---}}"
        rf"{{{o['garantia_estrutura_anos']} anos}}"
    )
    fotos = [r"\includegraphics[height=2.6cm]{estaticas/equip-inversor-trifasico.png}"]
    if s["bess_blocos"]:
        fotos.append(r"\includegraphics[height=2.6cm]{estaticas/equip-bess-parede.png}")
    fotos.append(r"\includegraphics[height=2.6cm]{estaticas/equip-modulo.png}")

    anos = [f["ano"] for f in dados["fluxo_acumulado"]]
    coords = {
        k: " ".join(f"({f['ano']},{f[k]:.2f})" for f in dados["fluxo_acumulado"])
        for k in ("a_vista", "financiado", "leasing")
    }
    linhas_fluxo = [
        rf"  \linhafluxo{{{f['ano']}}}{{{_saldo(f['a_vista'])}}}{{{_saldo(f['financiado'])}}}{{{_saldo(f['leasing'])}}}"
        for f in dados["fluxo_acumulado"]
    ]
    nota_fluxo = (
        f"Fluxo nominal do estudo: escalada tarifária de "
        f"{br_float(e['escalada_tarifaria_ano'] * 100, 1)}\\% a.a. e horizonte de "
        f"{e['horizonte_anos']} anos. Financiamento a "
        f"{br_float(o['juros_financiamento_am'] * 100, 2)}\\% a.m.; "
        f"leasing com seguro, manutenção e gerenciamento inclusos."
    )

    partes = [
        "% dados.tex — gerado pela PACEcalculator. Não edite: rode o estudo de novo.",
        f"% Cenário apresentado: {dados['cenario']}",
        r"\renewcommand{\pastalogos}{estaticas/}",
        "",
        "% ---- Identificação",
        _cmd("cliente", esc(dados["cliente"])),
        _cmd("dataproposta", _data_por_extenso(dados["data"])),
        _cmd("tituloproposta", "Proposta de Inteligência Energética"),
        _cmd("responsavel", esc(dados["responsavel"]) or "PACE Inteligência Energética"),
        _cmd("responsavelregistro", esc(dados["crea"])),
        _cmd("responsavelcredenciais", esc(dados["credenciais"])),
        rf"\renewcommand{{\rodapereferencia}}{{Proposta para \cliente}}",
        "",
        "% ---- Sistema",
        _cmd("subtitulosistema",
             "Dimensionado sobre o consumo e o telhado levantados no estudo" if s["area_ocupada_m2"]
             else "Dimensionado sobre o consumo levantado no estudo"),
        _cmd("potenciaprojeto", potencia),
        _cmd("potenciamodulo", f"{br_int(s['modulo_wp'])} Wp" if s["modulo_wp"] else "---"),
        _cmd("qtdmodulos", str(s["modulos"]) if s["modulos"] is not None else "---"),
        _cmd("potenciainversor", inversor),
        _cmd("qtdinversores", str(s["inversores"])),
        _cmd("areaocupada",
             rf"{br_int(s['area_ocupada_m2'])} m\textsuperscript{{2}}" if s["area_ocupada_m2"] else "---"),
        _cmd("producaoanual", f"{br_float(s['geracao_anual_kwh'] / 1000, 2)} MWh"),
        _cmd("bess", bess),
        _cmd("fototelhado", "figuras/foto_telhado.png"),
        "",
        "% ---- Compra à vista",
        _cmd("solucao", esc(s["estrutura"])),
        _cmd("geracaoanual", f"{br_int(s['geracao_anual_kwh'])} kWh"),
        _cmd("investimento", _brl(e["capex_brl"], 2)),
        _cmd("economiaanual", f"{_brl(e['economia_anual_brl'], 2)} no 1º ano"),
        _cmd("rotulopayback", "Payback simples"),
        _cmd("payback", payback),
        _cmd("gerenciamentomensal", f"{_brl(e['gerenciamento_brl_mes'], 2)}/mês"),
        _cmd("seguromensal", f"{_brl(e['seguro_brl_mes'], 2)}/mês"),
        _cmd("rotulotir", f"Retorno anual (TIR {e['horizonte_anos']} anos)"),
        _cmd("tir", tir),
        "",
        "% ---- Compra financiada",
        _cmd("investimentocurto", _brl(e["capex_brl"])),
        _cmd("economiamensal", _brl(em)),
        _cmd("mensalidadefinanciada",
             f"{_brl(parcela)} ({br_float(o['juros_financiamento_am'] * 100, 2)}\\% a.m.)"),
        _cmd("segurofinanciado", _brl(e["seguro_brl_mes"])),
        _cmd("gerenciamentofinanciado", _brl(e["gerenciamento_brl_mes"])),
        _cmd("prazofinanciamento", f"{o['prazo_financiamento_meses']} meses"),
        _cmd("carenciafinanciamento", f"{o['carencia_financiamento_meses']} meses"),
        _cmd("validadeproposta", f"{o['validade_dias']} dias"),
        "",
        "% ---- Leasing",
        _cmd("mensalidadeleasing", _brl(leasing)),
        _cmd("prazoleasing", f"{o['prazo_leasing_anos']} anos"),
        _cmd("carencialeasing", f"{o['carencia_leasing_meses']} meses"),
        "",
        "% ---- Desembolso mensal no primeiro ano",
        r"\newcommand{\linhasdesembolso}{%",
        rf"  \linhadesembolso{{À vista}}{{{_brl(e['capex_brl'])}}}{{---}}{{{_brl(servicos, 2)}}}"
        rf"{{{_brl(em, 2)}}}{{{_saldo(em - servicos)}}}",
        rf"  \linhadesembolso{{Financiado}}{{---}}{{{_brl(parcela, 2)}}}{{{_brl(servicos, 2)}}}"
        rf"{{{_brl(em, 2)}}}{{{_saldo(em - servicos - parcela)}}}",
        rf"  \linhadesembolso{{As a Service}}{{---}}{{{_brl(leasing, 2)}}}{{---}}"
        rf"{{{_brl(em, 2)}}}{{{_saldo(em - leasing)}}}",
        "}",
        "",
        "% ---- Fluxo de caixa acumulado",
        r"\newcommand{\linhasfluxo}{%",
        *linhas_fluxo,
        "}",
        _cmd("fluxoanos", ",".join(str(a) for a in anos)),
        _cmd("fluxozero", f"({anos[0]},0) ({anos[-1]},0)"),
        _cmd("fluxoavista", coords["a_vista"]),
        _cmd("fluxofinanciado", coords["financiado"]),
        _cmd("fluxoservico", coords["leasing"]),
        _cmd("notafluxo", nota_fluxo),
        "",
        "% ---- Lista de equipamentos",
        r"\newcommand{\linhasequipamentos}{%",
        *linhas_equip,
        "}",
        r"\newcommand{\fotosequipamentos}{%",
        "\\hspace{0.6cm}%\n".join(f"  {f}" for f in fotos),
        "}",
        "",
        "% ---- Institucional",
        _cmd("clientesgeridos", esc(o["clientes_geridos"])),
        _cmd("potenciagerida", esc(o["potencia_gerida"])),
        "",
    ]
    return "\n".join(partes)


# ----------------------------------------------------------------------------
# Escrita
# ----------------------------------------------------------------------------
def escrever_apresentacao(
    estudo: ResultadoEstudo,
    destino: str | Path,
    capa: DadosCapa | None = None,
    opcoes: OpcoesComerciais | None = None,
    foto_telhado: Path | None = None,
    compilar: bool = True,
) -> dict[str, Path]:
    """
    Monta a pasta da apresentação: esqueleto, imagens, ``dados.tex`` e o PDF.

    A pasta sai autocontida — compila no Overleaf sem o resto do repositório,
    porque as logos são copiadas para ``estaticas/`` e o ``dados.tex`` aponta
    para lá.
    """
    if not (PASTA_MODELO / "apresentacao.tex").exists():
        raise FileNotFoundError(f"Esqueleto da apresentação não encontrado em {PASTA_MODELO}")

    pasta = Path(destino)
    pasta.mkdir(parents=True, exist_ok=True)
    (pasta / "figuras").mkdir(exist_ok=True)
    for nome in ("apresentacao.tex", "pace-apresentacao.sty"):
        shutil.copy(PASTA_MODELO / nome, pasta / nome)
    shutil.copytree(PASTA_MODELO / "estaticas", pasta / "estaticas", dirs_exist_ok=True)
    for variante in ("clara", "transparente", "escura"):
        logo = marca.logo(variante)
        if logo is not None:
            shutil.copy(logo, pasta / "estaticas" / logo.name)
    if foto_telhado is not None and Path(foto_telhado).exists():
        shutil.copy(foto_telhado, pasta / "figuras" / "foto_telhado.png")

    dados = dados_da_apresentacao(estudo, capa, opcoes)
    caminho_tex = pasta / "dados.tex"
    caminho_tex.write_text(dados_tex(dados), encoding="utf-8")
    resultado: dict[str, Path] = {
        "apresentacao_tex": pasta / "apresentacao.tex",
        "apresentacao_dados": caminho_tex,
    }
    if compilar and encontrar_compilador():
        # Duas passadas: capa, notas e passos usam `remember picture`.
        pdf = compilar_pdf(pasta / "apresentacao.tex", passadas=2)
        if pdf is not None:
            resultado["apresentacao_pdf"] = pdf
        for extensao in (".aux", ".log", ".nav", ".out", ".snm", ".toc"):
            (pasta / f"apresentacao{extensao}").unlink(missing_ok=True)
    return resultado
