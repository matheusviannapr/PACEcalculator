"""
Coerência de ponta a ponta: um número, um cálculo, o mesmo em todo lugar.

A auditoria de setembro de 2026 achou o mesmo sistema com três paybacks —
na tela, no slide principal e nos slides de valor —, a TIR de uma topologia
impressa ao lado do preço de outra, a injeção valorada a dois preços no mesmo
estudo e a tarifa da vistoria descartada na importação. Estes testes existem
para que nenhuma dessas divergências volte sem alguém perceber.
"""
from __future__ import annotations

import io
import zipfile
from datetime import date

import pytest

from aurum.bateria.apagao import MalhaApagao
from aurum.bateria.apresentacao import OpcoesComerciais, dados_da_apresentacao
from aurum.bateria.documento import zip_do_dossie
from aurum.bateria.economia import PremissasBateria
from aurum.bateria.estudo import ConfiguracaoEstudo, executar_estudo
from aurum.bateria.fontes import Fatura, _balanco_anual
from aurum.bateria.geracao import serie_sintetica
from aurum.bateria.memoria_calculo import conferencias, montar_memoria
from aurum.demanda import cenario_exemplo, simular_ensemble
from aurum.demanda.cenario import Cenario


# ----------------------------------------------------------------------------
# A conta de luz com compensação
# ----------------------------------------------------------------------------
def test_fio_b_segue_o_cronograma_da_lei_14300():
    fatura = Fatura(tarifa_brl_kwh=1.0, participacao_fio_b=0.28, ano_conexao=2026)
    assert fatura.fio_b_brl_kwh(2026) == pytest.approx(0.28 * 0.60)
    assert fatura.fio_b_brl_kwh(2027) == pytest.approx(0.28 * 0.75)
    assert fatura.fio_b_brl_kwh(2028) == pytest.approx(0.28 * 0.90)
    assert fatura.fio_b_brl_kwh(2031) == pytest.approx(0.28 * 1.00)
    assert fatura.valor_injecao_brl_kwh == pytest.approx(1.0 - 0.28 * 0.60)


def test_compensado_paga_fio_b_e_nao_passa_do_piso():
    fatura = Fatura(tarifa_brl_kwh=1.0, custo_disponibilidade_kwh=50.0,
                    participacao_fio_b=0.25, ano_conexao=2029)
    d = fatura.detalhar_conta(compra_kwh_ano=4_000.0, injecao_kwh_ano=10_000.0)
    # O piso é de 600 kWh/ano: só 3.400 kWh podem ser compensados.
    assert d["compensado_kwh"] == pytest.approx(3_400.0)
    assert d["credito_sem_uso_kwh"] == pytest.approx(6_600.0)
    assert d["faturado_kwh"] == pytest.approx(600.0)
    assert d["conta_brl"] == pytest.approx(600.0 * 1.0 + 3_400.0 * 0.25)


def test_bandeira_encarece_a_compra_e_a_compensacao_a_evita():
    sem = Fatura(tarifa_brl_kwh=1.0, custo_disponibilidade_kwh=0.0, ano_conexao=2026)
    com = Fatura(tarifa_brl_kwh=1.0, bandeira_brl_kwh=0.05,
                 custo_disponibilidade_kwh=0.0, ano_conexao=2026)
    assert com.conta_anual_brl(1_000.0) == pytest.approx(1_050.0)
    # O kWh compensado evita a bandeira: vale R$ 0,05 a mais com ela.
    assert com.valor_injecao_brl_kwh - sem.valor_injecao_brl_kwh == pytest.approx(0.05)


# ----------------------------------------------------------------------------
# O balanço percorre a série inteira
# ----------------------------------------------------------------------------
def test_balanco_exato_gera_kwp_vezes_produtividade_e_fecha():
    comodos, instancias = cenario_exemplo("residencia")
    ensemble = simular_ensemble(comodos, instancias, 20).reamostrar(15)
    serie = serie_sintetica(-23.0, -43.2, 0.0, 20.0, 14.0, anos=(2019, 2020))
    b = _balanco_anual(ensemble, serie, 5.0, None, None, Fatura(), 0.3, None, 7)
    assert b["geracao"] == pytest.approx(5.0 * serie.anual_kwh_por_kwp(), rel=1e-6)
    assert b["autoconsumo"] + b["compra"] == pytest.approx(b["consumo"], rel=1e-9)
    assert b["autoconsumo_direto"] + b["recarga"] + b["injecao"] == pytest.approx(
        b["geracao"], rel=1e-9)


# ----------------------------------------------------------------------------
# Um estudo residencial com as três propostas
# ----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def estudo():
    cenario = Cenario.de_segmento("residencia", "Casa Coerente")
    cenario.criticidades_essenciais = ("MC", "C")
    cfg = ConfiguracaoEstudo(
        latitude=-22.95, longitude=-43.18, nome="Casa Coerente", tensao_rede_v=220,
        comodos=cenario.para_comodos(), instancias_por_comodo=cenario.instancias_de(),
        comodos_backup=cenario.para_comodos(True) if cenario.tem_criticidade else None,
        instancias_backup=cenario.instancias_de(True) if cenario.tem_criticidade else None,
        simulacoes=30, potencia_fv_kwp=4.0, max_candidatos=3, autonomia_alvo_h=2.0,
        malha=MalhaApagao(duracoes_h=(1.0, 2.0), horas_inicio=(0, 12, 18), amostras=15),
        premissas=PremissasBateria(custo_interrupcao_brl_kwh=10.0),
        fatura=Fatura(tarifa_brl_kwh=1.02, consumo_mensal_kwh=450,
                      custo_disponibilidade_kwh=50.0, ano_conexao=2026),
        topologia_kit="splitphase", considerar_gerador=False,
    )
    return executar_estudo(cfg)


def test_toda_conferencia_de_conta_fecha(estudo):
    falhas = [c for c in conferencias(estudo) if not c.ok and c.grave]
    assert not falhas, "\n".join(f"{c.nome}: {c.esperado} | {c.obtido}" for c in falhas)


def test_cada_proposta_tem_o_financeiro_do_proprio_investimento(estudo):
    assert estudo.financeiro, "o estudo precificou as topologias e não fez o financeiro"
    for topologia, f in estudo.financeiro.items():
        preco = estudo.precos_por_topologia[topologia]["capex_brl"]
        assert f.capex_brl == pytest.approx(preco)
        assert f.payback_simples_anos == pytest.approx(f.capex_brl / f.economia_ano1_brl)
        # A TIR é a deste capex: o valor presente à TIR, com ele, zera.
        caixa = [-f.capex_brl, *f.fluxo["fluxo_brl"].tolist()]
        assert sum(v / (1 + f.tir) ** i for i, v in enumerate(caixa)) == pytest.approx(0.0, abs=1.0)
    # Duas topologias do mesmo cenário: mesma economia, investimentos e TIRs diferentes.
    micro, split = estudo.financeiro["microinversor"], estudo.financeiro["splitphase"]
    assert micro.economia_ano1_brl == pytest.approx(split.economia_ano1_brl)
    assert micro.capex_brl != pytest.approx(split.capex_brl)
    assert micro.tir != pytest.approx(split.tir)


def test_slides_leem_o_mesmo_financeiro(estudo):
    dados = dados_da_apresentacao(estudo, opcoes=OpcoesComerciais(data=date(2026, 9, 23)))
    principal = estudo.financeiro_do_cenario(dados["cenario"])
    e = dados["economia"]
    assert e["capex_brl"] == pytest.approx(principal.capex_brl)
    assert e["economia_anual_brl"] == pytest.approx(principal.economia_ano1_brl)
    # "Payback simples" no slide é investimento ÷ economia do mesmo slide.
    assert e["payback_anos"] == pytest.approx(e["capex_brl"] / e["economia_anual_brl"])
    assert e["tir"] == pytest.approx(principal.tir)
    for topologia, bloco in dados["comparativo"]["topologias"].items():
        f = estudo.financeiro[topologia]
        assert bloco["capex_brl"] == pytest.approx(f.capex_brl)
        assert bloco["tir"] == pytest.approx(f.tir)
        assert bloco["payback_anos"] == pytest.approx(f.payback_simples_anos)


def test_resiliencia_fica_fora_do_caixa(estudo):
    com_banco = estudo.financeiro["splitphase_bateria"]
    assert com_banco.valor_resiliencia_ano1_brl > 0
    assert com_banco.vpl_com_resiliencia_brl > com_banco.vpl_brl
    assert "valor_resiliencia_brl" in com_banco.fluxo
    assert com_banco.fluxo["fluxo_brl"].iloc[0] == pytest.approx(
        com_banco.fluxo["economia_conta_brl"].iloc[0]
        - com_banco.fluxo["custo_operacional_brl"].iloc[0]
        - com_banco.fluxo["troca_brl"].iloc[0])


def test_economia_do_banco_usa_a_injecao_da_fatura(estudo):
    """A economia isolada do banco e a conta de luz valoram o mesmo kWh igual."""
    fatura = estudo.configuracao.fatura
    for economico in estudo.economia.values():
        assert economico.premissas.valor_injecao_brl_kwh == pytest.approx(
            fatura.valor_injecao_brl_kwh)


def test_fio_b_crescente_reduz_a_economia_real(estudo):
    solar = estudo.cenarios.por_chave("solar")
    anuais = [a["economia_brl"] for a in solar.economias_por_ano]
    assert anuais[0] == pytest.approx(solar.economia_anual_brl)
    assert anuais[3] < anuais[0], "o Fio B de 2029 tem de pesar mais que o de 2026"


def test_memoria_vai_no_pacote_e_cita_os_numeros_da_proposta(estudo):
    texto = montar_memoria(estudo)
    for titulo in ("Conferências automáticas", "Conta de luz e compensação",
                   "Fluxo de caixa de cada proposta", "Investimento"):
        assert titulo in texto
    principal = estudo.financeiro_do_cenario()
    assert f"{principal.capex_brl:,.0f}".replace(",", ".") in texto
    arquivo = zip_do_dossie(estudo, compilar=False)
    nomes = zipfile.ZipFile(io.BytesIO(arquivo)).namelist()
    assert any(n.endswith("memoria-de-calculo.md") for n in nomes)
    assert any("fluxo_proposta_splitphase_bateria" in n for n in nomes)


# ----------------------------------------------------------------------------
# A vistoria traz a conta de luz
# ----------------------------------------------------------------------------
def test_vistoria_traz_tarifa_e_consumo_da_fatura():
    from aurum.demanda.vistoria import ler_backup

    dados = {
        "formato": "vistoria-eletrica/1",
        "vistoria": {"cliente": "Fulano", "tensao": 127, "sistema": "127/220 V"},
        "fatura": {"tarifaBrlKwh": 1.07, "consumoMensalKwh": 380},
        "itens": [{"equipamento": "Geladeira", "comodo": "Cozinha", "potTipica": 150,
                   "qtd": 1, "criticidade": "MC"}],
    }
    vistoria = ler_backup(dados)
    assert vistoria.tarifa_brl_kwh == pytest.approx(1.07)
    assert vistoria.consumo_mensal_kwh == pytest.approx(380)


def test_botoes_da_tela_navegam_pelo_nome_do_passo():
    from aurum.pagina_estudo import PASSOS, _passo_chamado

    assert PASSOS[_passo_chamado("Meta") - 1][1] == "Meta"
    assert PASSOS[_passo_chamado("Resultado") - 1][1] == "Resultado"


def test_compensar_a_conta_desconta_o_custo_de_disponibilidade():
    """
    "Compensar 100%" é gerar o consumo menos os doze mínimos do ano.

    Gerando o consumo inteiro, o crédito acima do piso não tem onde ser usado:
    num caso de 6.000 kWh/ano em ligação trifásica, ~1.200 kWh/ano de geração
    viravam crédito perdido, e o sistema saía maior que a conta.
    """
    comodos, instancias = cenario_exemplo("residencia")
    cfg = ConfiguracaoEstudo(
        latitude=-22.95, longitude=-43.18, nome="Compensar",
        comodos=comodos, instancias_por_comodo=instancias, simulacoes=20,
        consumo_anual_kwh=6_000.0, objetivo_fv="compensar", compensar_frac=1.0,
        max_candidatos=2, autonomia_alvo_h=1.0,
        malha=MalhaApagao(duracoes_h=(1.0,), horas_inicio=(0, 18), amostras=10),
        fatura=Fatura(tarifa_brl_kwh=1.0, custo_disponibilidade_kwh=100.0),
        considerar_gerador=False,
    )
    estudo = executar_estudo(cfg)
    produtividade = estudo.serie.anual_kwh_por_kwp()
    # A fatura sem consumo usa o consumo declarado; o piso é 100 × 12.
    assert estudo.potencia_fv_kwp == pytest.approx((6_000.0 - 1_200.0) / produtividade, rel=1e-6)
    assert estudo.compensacao["cobertura"] == pytest.approx(1.0, abs=0.01)
    solar = estudo.cenarios.por_chave("solar")
    sem_uso = estudo.cenarios.fatura.detalhar_conta(
        solar.balanco["compra"], solar.balanco["injecao"])["credito_sem_uso_kwh"]
    assert sem_uso < 0.05 * solar.balanco["injecao"]
