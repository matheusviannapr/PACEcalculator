"""
Testes do estudo de baterias.

Concentrados no que pode quebrar em silêncio: conservação de energia no
despacho, a distinção entre falha de potência e falha de energia, a correção da
janela que atravessa a meia-noite, e a monotonia que qualquer modelo de
armazenamento tem que respeitar (mais kWh nunca pode reduzir a autonomia).
Nenhum teste aqui toca a rede.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from aurum.bateria.apagao import MalhaApagao, avaliar_conjunto
from aurum.bateria.catalogo import ConjuntoArmazenamento, carregar_catalogo
from aurum.bateria.degradacao import ModeloDegradacao, trajetoria_de_vida
from aurum.bateria.despacho import (
    CAUSA_ENERGIA,
    CAUSA_POTENCIA,
    LimitesDespacho,
    simular_ilhamento,
)
from aurum.bateria.economia import PremissasBateria, avaliar_economia, simular_operacao_anual
from aurum.bateria.excedencia import diagnosticar_inversor
from aurum.bateria.geracao import (
    deslocamento_utc_horas,
    estacao_da_data,
    serie_sintetica,
)
from aurum.demanda import cenario_exemplo, simular_ensemble
from aurum.demanda.correcoes import intervalos_circulares
from aurum.demanda.nucleo import Equipamento, parse_intervalo_fixo


# ----------------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def catalogo():
    return carregar_catalogo()


@pytest.fixture(scope="module")
def ensemble():
    """Ensemble pequeno, no passo do despacho, para os testes serem rápidos."""
    comodos, instancias = cenario_exemplo("hotel")
    area_comum = [c for c in comodos if c.nome == "Área Comum"]
    return simular_ensemble(area_comum, {"Área Comum": 1}, num_simulacoes=40).reamostrar(15)


@pytest.fixture(scope="module")
def serie():
    return serie_sintetica(-25.43, -49.27, 0.0, 25.0, 14.0, anos=(2018, 2019))


@pytest.fixture
def limites():
    """10 kWh úteis, 5 kW contínuos, 10 kW de surto por 10 s, sem FV."""
    return LimitesDespacho(
        energia_util_dc_kwh=10.0,
        potencia_descarga_kw=5.0,
        potencia_pico_kw=10.0,
        duracao_pico_s=10.0,
        potencia_carga_kw=5.0,
        potencia_fv_kw=0.0,
        eficiencia_descarga=1.0,
        eficiencia_carga=1.0,
    )


# ----------------------------------------------------------------------------
# Correção da janela que atravessa a meia-noite
# ----------------------------------------------------------------------------
def test_janela_noturna_some_sem_correcao():
    """Documenta o bug do D²: 18:00 as 06:00 produz carga zero."""
    eq = Equipamento(
        nome="Iluminação", potencia=120, quantidade=10,
        intervalos=parse_intervalo_fixo("18:00 as 06:00"), probabilidade=1.0,
    )
    assert eq.simula_carga(1440).sum() == 0.0


def test_correcao_restaura_a_carga_noturna():
    eq = Equipamento(
        nome="Iluminação", potencia=120, quantidade=10,
        intervalos=parse_intervalo_fixo("18:00 as 06:00"), probabilidade=1.0,
    )
    with intervalos_circulares():
        carga = eq.simula_carga(1440)
    # 12 h de 1,2 kW = 14,4 kWh, ligado em 720 dos 1440 minutos.
    assert int((carga > 0).sum()) == 720
    assert carga.sum() / 60 / 1000 == pytest.approx(14.4, rel=1e-9)
    # Acende de fato nas duas pontas do dia, não só numa.
    assert carga[0] > 0 and carga[1080] > 0 and carga[720] == 0


def test_correcao_nao_altera_janela_dentro_do_dia():
    eq = Equipamento(
        nome="dia", potencia=100, quantidade=1,
        intervalos=parse_intervalo_fixo("08:00 as 18:00"), probabilidade=1.0,
    )
    original = eq.simula_carga(1440)
    with intervalos_circulares():
        corrigida = eq.simula_carga(1440)
    np.testing.assert_array_equal(original, corrigida)


# ----------------------------------------------------------------------------
# Ensemble e curvas de excedência
# ----------------------------------------------------------------------------
def test_reamostragem_preserva_energia_e_pico():
    comodos, instancias = cenario_exemplo("escritorio")
    fino = simular_ensemble(comodos, instancias, num_simulacoes=20)
    grosso = fino.reamostrar(15)
    assert grosso.passos_por_dia == 96
    np.testing.assert_allclose(
        fino.energia_diaria_kwh(), grosso.energia_diaria_kwh(), rtol=1e-9
    )
    np.testing.assert_allclose(fino.picos_diarios_w(), grosso.picos_diarios_w(), rtol=1e-9)


def test_reamostragem_recusa_passo_que_nao_divide_o_dia():
    comodos, instancias = cenario_exemplo("escritorio")
    ensemble = simular_ensemble(comodos, instancias, num_simulacoes=5)
    with pytest.raises(ValueError, match="janelas iguais"):
        ensemble.reamostrar(7)


def test_curva_de_excedencia_e_seu_inverso_sao_consistentes():
    """
    Amostras contínuas de propósito: a carga do cenário demo é fortemente
    quantizada (poucos valores distintos), e nenhuma distribuição empírica
    discreta consegue inverter exatamente uma probabilidade arbitrária.
    """
    from aurum.demanda.ensemble import CurvaExcedencia

    rng = np.random.default_rng(11)
    curva = CurvaExcedencia.de_amostras(rng.lognormal(10.0, 0.35, size=5000))
    for p in (0.05, 0.2, 0.5):
        valor = curva.valor_para_probabilidade(p)
        assert curva.prob_excedencia(valor) == pytest.approx(p, abs=0.01)
    assert curva.prob_excedencia(curva.amostras[-1] + 1) == 0.0
    assert curva.prob_excedencia(-1.0) == 1.0


def test_excedencia_instantanea_nunca_supera_a_de_pico(ensemble):
    """
    Todo dia cujo *qualquer* minuto passa de x tem pico maior que x; o inverso
    não vale. A desigualdade entre as duas curvas é estrutural.
    """
    pico = ensemble.curva_excedencia_pico()
    instantanea = ensemble.curva_excedencia_instantanea()
    for limite in np.percentile(ensemble.picos_janela(), [50, 75, 90, 99]):
        assert instantanea.prob_excedencia(limite) <= pico.prob_excedencia(limite) + 1e-12


# ----------------------------------------------------------------------------
# Despacho
# ----------------------------------------------------------------------------
def test_despacho_conserva_energia(limites):
    """Atendida + não suprida tem que fechar com a demandada, sempre."""
    rng = np.random.default_rng(3)
    carga = rng.uniform(0.5, 8.0, size=(40, 96))
    geracao = np.zeros_like(carga)
    r = simular_ilhamento(carga, carga, geracao, limites, passo_min=15)
    atendida = r.energia_bateria_kwh + r.energia_pv_kwh
    np.testing.assert_allclose(atendida + r.ens_kwh, r.energia_demandada_kwh, rtol=1e-9)


def test_falha_por_energia_quando_o_banco_esvazia(limites):
    """Carga baixa e longa: o inversor dá conta, o banco é que acaba."""
    carga = np.full((1, 96), 2.0)  # 2 kW por 24 h = 48 kWh contra 10 kWh úteis
    r = simular_ilhamento(carga, carga, np.zeros_like(carga), limites, passo_min=15)
    assert not r.atendido[0]
    assert r.causa[0] == CAUSA_ENERGIA
    assert r.minutos_ate_falha[0] == pytest.approx(5 * 60, abs=15)  # 10 kWh / 2 kW


def test_falha_por_potencia_com_banco_cheio(limites):
    """Carga acima do surto no primeiro passo: falha imediata, banco intacto."""
    carga = np.full((1, 4), 12.0)  # acima dos 10 kW de pico
    r = simular_ilhamento(carga, carga, np.zeros_like(carga), limites, passo_min=1)
    assert r.causa[0] == CAUSA_POTENCIA
    assert r.minutos_ate_falha[0] == 0
    assert r.soc_min_frac[0] > 0.9  # sobrou bateria: o gargalo não era energia
    assert r.eventos_desarme[0] == 4  # desarma em todos os quatro passos


def test_surto_cobre_pico_curto_e_se_esgota_no_longo(limites):
    """
    7 kW está entre o contínuo (5) e o surto (10). Um passo curto passa pelo
    crédito de surto; passos seguidos o consomem e a entrega cai ao contínuo.
    """
    passo = 1  # 60 s por passo, contra 10 s de crédito
    curto = np.array([[7.0, 1.0, 1.0, 1.0]])
    r_curto = simular_ilhamento(curto, curto, np.zeros_like(curto), limites, passo_min=passo)
    assert r_curto.atendido[0]

    longo = np.full((1, 10), 7.0)
    r_longo = simular_ilhamento(longo, longo, np.zeros_like(longo), limites, passo_min=passo)
    assert not r_longo.atendido[0]
    assert r_longo.causa[0] == CAUSA_POTENCIA


def test_sol_recarrega_durante_o_apagao(limites):
    """Com geração, o mesmo apagão que esvaziava o banco passa a ser atendido."""
    # 1 kW por 24 h = 24 kWh contra 10 kWh úteis: sem sol, o banco morre às 10 h.
    carga = np.full((1, 96), 1.0)
    sem_sol = simular_ilhamento(carga, carga, np.zeros_like(carga), limites, passo_min=15)
    assert not sem_sol.atendido[0]

    com_fv = LimitesDespacho(**{**limites.__dict__, "potencia_fv_kw": 8.0})
    sol = np.zeros_like(carga)
    sol[:, 24:56] = 6.0  # sol das 6 h às 14 h, antes de o banco esvaziar
    com_sol = simular_ilhamento(carga, carga, sol, com_fv, passo_min=15)

    assert sem_sol.ens_kwh[0] > com_sol.ens_kwh[0]
    assert com_sol.energia_recarregada_kwh[0] > 0
    assert com_sol.atendido[0]


def test_excedente_sem_lugar_para_ir_e_contabilizado(limites):
    """Ilhado não exporta: o que sobra depois de encher o banco é cortado."""
    com_fv = LimitesDespacho(**{**limites.__dict__, "potencia_fv_kw": 20.0})
    carga = np.full((1, 96), 0.5)
    sol = np.full((1, 96), 15.0)
    r = simular_ilhamento(carga, carga, sol, com_fv, passo_min=15)
    assert r.atendido[0]
    assert r.energia_desperdicada_kwh[0] > 0


# ----------------------------------------------------------------------------
# Catálogo
# ----------------------------------------------------------------------------
def test_energia_util_desconta_dod_e_rendimento(catalogo):
    chumbo = catalogo.bateria_por_modelo("Clean Nano 220Ah (banco 48V)")
    litio = catalogo.bateria_por_modelo("Tower T10")
    assert chumbo is not None and litio is not None
    # Nominais parecidas (10,56 e 10,66 kWh), úteis muito diferentes.
    assert chumbo.capacidade_nominal_kwh == pytest.approx(litio.capacidade_nominal_kwh, rel=0.02)
    assert chumbo.energia_util_kwh < litio.energia_util_kwh * 0.55


def test_conjunto_limita_pelo_menor_dos_dois(catalogo):
    inversor = catalogo.inversor_por_modelo("SUN-12K-SG04LP3")  # 12 kW
    bateria = catalogo.bateria_por_modelo("US3000C")  # 1,78 kW por módulo
    um = ConjuntoArmazenamento(inversor, bateria, 1)
    muitos = ConjuntoArmazenamento(inversor, bateria, 10)
    assert um.potencia_descarga_kw == pytest.approx(1.78)  # gargalo na bateria
    assert muitos.potencia_descarga_kw == pytest.approx(12.0)  # gargalo no inversor


def test_inversor_sem_mppt_mantem_fv_zero(catalogo):
    """Zero é valor legítimo e não pode virar o padrão de 1,3 × nominal."""
    victron = catalogo.inversor_por_modelo("MultiPlus-II 48/5000/70")
    assert victron.potencia_fv_max_kw == 0.0


def test_banco_alem_do_limite_do_fabricante_e_recusado(catalogo):
    inversor = catalogo.inversor_por_modelo("SUN-5K-SG03LP1")
    bateria = catalogo.bateria_por_modelo("Tower T10")  # máximo 8 em paralelo
    with pytest.raises(ValueError, match="máximo"):
        ConjuntoArmazenamento(inversor, bateria, 20)


# ----------------------------------------------------------------------------
# Geração
# ----------------------------------------------------------------------------
def test_estacoes_do_hemisferio_sul():

    assert estacao_da_data(date(2020, 1, 15)) == "verão"
    assert estacao_da_data(date(2020, 3, 15)) == "verão"
    assert estacao_da_data(date(2020, 4, 1)) == "outono"
    assert estacao_da_data(date(2020, 7, 1)) == "inverno"
    assert estacao_da_data(date(2020, 10, 1)) == "primavera"
    assert estacao_da_data(date(2020, 12, 25)) == "verão"
    # No hemisfério norte, tudo se inverte.
    assert estacao_da_data(date(2020, 7, 1), hemisferio_sul=False) == "verão"


def test_serie_sintetica_tem_verao_mais_produtivo_que_inverno(serie):
    diaria = serie.energia_diaria_kwh_por_kwp()
    verao = diaria[serie.dias_da_estacao("verão")].mean()
    inverno = diaria[serie.dias_da_estacao("inverno")].mean()
    assert verao > inverno
    assert 900 < serie.anual_kwh_por_kwp() < 2000  # faixa brasileira plausível


def test_serie_nao_gera_energia_de_madrugada(serie):
    madrugada = serie.potencia_w_por_kwp[:, :4]
    assert madrugada.max() == 0.0


def test_amostragem_de_janela_respeita_forma_e_hora(serie):
    rng = np.random.default_rng(0)
    janela = serie.amostrar_janelas("inverno", 22, n_passos=48, passo_min=5, quantidade=7, rng=rng)
    assert janela.shape == (7, 48)
    # 22h + 4h = até 2h da manhã: tudo no escuro.
    assert janela.max() == 0.0


def test_fuso_da_longitude_bate_com_as_capitais():
    # O PVGIS carimba a série em UTC; sem converter, o pico de geração de
    # Curitiba caía às 15 h e toda a sobreposição com a carga saía errada.
    assert deslocamento_utc_horas(-49.27) == -3   # Curitiba
    assert deslocamento_utc_horas(-38.52) == -3   # Salvador
    assert deslocamento_utc_horas(-60.02) == -4   # Manaus
    assert deslocamento_utc_horas(-46.63) == -3   # São Paulo


def test_alinhamento_desloca_a_serie_e_descarta_as_bordas():
    from aurum.bateria.geracao import _alinhar_ao_relogio_local

    # Um "sol" artificial que só brilha às 15 h UTC, como a série crua.
    dias = tuple(date(2020, 1, 1) + timedelta(days=i) for i in range(10))
    matriz = np.zeros((10, 24))
    matriz[:, 15] = 500.0

    datas, alinhada, aviso = _alinhar_ao_relogio_local(dias, matriz, longitude=-49.27)

    assert alinhada.shape == (8, 24), "o primeiro e o último dia saem, por circularidade"
    assert datas == dias[1:-1]
    assert int(alinhada.mean(axis=0).argmax()) == 12, "15 h UTC são 12 h em Curitiba"
    assert aviso is None, "meio-dia passa no teste de sanidade"


def test_alinhamento_avisa_quando_o_resultado_nao_e_plausivel():
    from aurum.bateria.geracao import _alinhar_ao_relogio_local

    # Geração concentrada às 3 h UTC: depois da correção sobra a madrugada, e
    # o estudo precisa dizer isso em vez de seguir calado com número errado.
    dias = tuple(date(2020, 1, 1) + timedelta(days=i) for i in range(10))
    matriz = np.zeros((10, 24))
    matriz[:, 3] = 500.0

    _, _, aviso = _alinhar_ao_relogio_local(dias, matriz, longitude=-49.27)
    assert aviso is not None and "fuso" in aviso


def test_janela_longa_demais_para_a_serie_falha_claramente():
    curta = serie_sintetica(-25.43, -49.27, 0.0, 25.0, 14.0, anos=(2020, 2020))
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="curta demais"):
        curta.amostrar_janelas("inverno", 0, n_passos=200 * 24 * 12, passo_min=5, quantidade=1, rng=rng)


# ----------------------------------------------------------------------------
# Excedência contra inversor
# ----------------------------------------------------------------------------
def test_inversor_folgado_aprova_e_apertado_reprova(catalogo, ensemble):
    grande = diagnosticar_inversor(catalogo.inversor_por_modelo("SUN-12K-SG04LP3"), ensemble)
    assert grande.aprovado
    assert grande.prob_pico_diario_acima_nominal == 0.0
    assert "folga" in grande.veredito

    # Um inversor artificialmente pequeno para a mesma carga.
    from dataclasses import replace

    minusculo = replace(
        catalogo.inversor_por_modelo("SUN-5K-SG03LP1"),
        potencia_ca_nominal_kw=0.3, potencia_ca_pico_kw=0.5,
    )
    diagnostico = diagnosticar_inversor(minusculo, ensemble)
    assert not diagnostico.aprovado
    assert "reprovado" in diagnostico.veredito
    assert diagnostico.duracao_media_min > 0


# ----------------------------------------------------------------------------
# Resiliência
# ----------------------------------------------------------------------------
def test_mais_energia_nunca_reduz_a_autonomia(catalogo, ensemble, serie):
    """
    Monotonia: com o mesmo inversor, dobrar o banco não pode piorar. É a
    invariante que pega quase qualquer erro de sinal no balanço do despacho.
    """
    inversor = catalogo.inversor_por_modelo("SUN-8K-SG04LP3")
    bateria = catalogo.bateria_por_modelo("US5000")
    malha = MalhaApagao(duracoes_h=(3.0, 12.0, 24.0), horas_inicio=(0, 12, 18), amostras=40, passo_min=15)

    autonomias = []
    for modulos in (1, 2, 4):
        conjunto = ConjuntoArmazenamento(inversor, bateria, modulos, potencia_fv_kwp=10.0)
        resultado = avaliar_conjunto(conjunto, ensemble, serie, malha)
        autonomias.append(resultado.autonomia_garantida_h(0.90))
    assert autonomias == sorted(autonomias)


def test_apagao_noturno_e_pior_que_diurno(catalogo, ensemble, serie):
    """Com FV, começar às 18 h é sempre pelo menos tão ruim quanto às 6 h."""
    conjunto = ConjuntoArmazenamento(
        catalogo.inversor_por_modelo("SUN-8K-SG04LP3"),
        catalogo.bateria_por_modelo("US5000"), 1, potencia_fv_kwp=10.0,
    )
    malha = MalhaApagao(duracoes_h=(12.0,), horas_inicio=(6, 18), amostras=60, passo_min=15)
    tabela = avaliar_conjunto(conjunto, ensemble, serie, malha).tabela
    manha = tabela[tabela["hora_inicio"] == 6]["prob_atendimento"].mean()
    noite = tabela[tabela["hora_inicio"] == 18]["prob_atendimento"].mean()
    assert manha >= noite


def test_capacidade_degradada_reduz_a_autonomia(catalogo, ensemble, serie):
    conjunto = ConjuntoArmazenamento(
        catalogo.inversor_por_modelo("SUN-8K-SG04LP3"),
        catalogo.bateria_por_modelo("US5000"), 2, potencia_fv_kwp=5.0,
    )
    malha = MalhaApagao(duracoes_h=(6.0, 12.0), horas_inicio=(18,), amostras=40, passo_min=15)
    novo = avaliar_conjunto(conjunto, ensemble, serie, malha)
    velho = avaliar_conjunto(conjunto, ensemble, serie, malha, fator_capacidade=0.5)
    assert velho.tabela["ens_medio_kwh"].sum() >= novo.tabela["ens_medio_kwh"].sum()


# ----------------------------------------------------------------------------
# Degradação
# ----------------------------------------------------------------------------
def test_retencao_cai_com_tempo_e_com_ciclos():
    modelo = ModeloDegradacao(ciclos_nominais=6000, retencao_fim_vida=0.8)
    assert modelo.retencao(0, 0) == 1.0
    assert modelo.retencao(10, 0) < modelo.retencao(1, 0)
    assert modelo.retencao(1, 3000) < modelo.retencao(1, 0)
    assert modelo.retencao(100, 100000) == 0.0  # com piso, nunca negativa


def test_vida_util_encurta_com_uso_intenso():
    modelo = ModeloDegradacao(ciclos_nominais=6000, retencao_fim_vida=0.8)
    leve = modelo.vida_util_anos(ciclos_por_ano=100)
    pesado = modelo.vida_util_anos(ciclos_por_ano=600)
    assert pesado < leve
    assert 8 < pesado < 12  # 6000 ciclos a 600/ano, menos o calendário


def test_trajetoria_marca_o_fim_de_vida():
    modelo = ModeloDegradacao(ciclos_nominais=2000, retencao_fim_vida=0.8)
    tabela = trajetoria_de_vida(modelo, ciclos_por_ano=400, energia_util_inicial_kwh=10.0, anos=15)
    assert tabela["retencao"].is_monotonic_decreasing
    assert tabela["fim_de_vida"].any()
    assert tabela.loc[0, "energia_util_kwh"] == pytest.approx(10.0)


# ----------------------------------------------------------------------------
# Economia
# ----------------------------------------------------------------------------
def test_operacao_anual_nao_cicla_de_forma_absurda(catalogo, ensemble, serie):
    """
    Um ciclo por dia é o teto físico de um banco bem operado. Passar muito
    disso é sintoma de laço de carga e descarga no mesmo passo — foi o que
    aconteceu antes da janela de preparo antes da ponta existir.
    """
    conjunto = ConjuntoArmazenamento(
        catalogo.inversor_por_modelo("SUN-5K-SG03LP1"),
        catalogo.bateria_por_modelo("US5000"), 2, potencia_fv_kwp=8.0,
    )
    operacao = simular_operacao_anual(conjunto, ensemble, serie, PremissasBateria(), dias_por_estacao=20)
    assert 0 <= operacao.ciclos_equivalentes <= 730  # até dois ciclos por dia
    assert operacao.export_com_kwh <= operacao.export_sem_kwh + 1e-6
    assert operacao.import_ponta_com_kwh <= operacao.import_ponta_sem_kwh + 1e-6


def test_valor_da_resiliencia_entra_no_fluxo(catalogo, ensemble, serie):
    conjunto = ConjuntoArmazenamento(
        catalogo.inversor_por_modelo("SUN-5K-SG03LP1"),
        catalogo.bateria_por_modelo("US5000"), 2, potencia_fv_kwp=8.0,
    )
    malha = MalhaApagao(duracoes_h=(3.0,), horas_inicio=(18,), amostras=30, passo_min=15)
    resiliencia = avaliar_conjunto(conjunto, ensemble, serie, malha)

    sem_valor = avaliar_economia(
        conjunto, ensemble, serie,
        PremissasBateria(custo_interrupcao_brl_kwh=0.0), resiliencia, dias_por_estacao=20,
    )
    com_valor = avaliar_economia(
        conjunto, ensemble, serie,
        PremissasBateria(custo_interrupcao_brl_kwh=50.0), resiliencia, dias_por_estacao=20,
    )
    assert sem_valor.valor_resiliencia_ano1_brl == 0.0
    assert com_valor.valor_resiliencia_ano1_brl > 0.0
    assert com_valor.vpl_brl > sem_valor.vpl_brl
    assert com_valor.capex_brl == sem_valor.capex_brl > 0


def test_premissa_com_modo_invalido_falha_cedo():
    with pytest.raises(ValueError, match="modo deve ser"):
        PremissasBateria(modo="turbo")


# ----------------------------------------------------------------------------
# Estudo completo
# ----------------------------------------------------------------------------
def test_estudo_completo_gera_relatorio(tmp_path):
    """Fumaça de ponta a ponta, com malha mínima para caber em segundos."""
    from aurum.bateria.estudo import ConfiguracaoEstudo, executar_estudo
    from aurum.bateria.relatorio import escrever_relatorio

    comodos, instancias = cenario_exemplo("hotel")
    cfg = ConfiguracaoEstudo(
        latitude=-25.43, longitude=-49.27, nome="Teste",
        comodos=comodos, instancias_por_comodo=instancias,
        comodos_essenciais=["Área Comum"],
        simulacoes=30, potencia_fv_kwp=10.0, anos_serie=(2019, 2020),
        max_candidatos=2, autonomia_alvo_h=3.0,
        malha=MalhaApagao(duracoes_h=(1.0, 3.0), horas_inicio=(0, 12, 18), amostras=20, passo_min=15),
    )
    estudo = executar_estudo(cfg)

    assert not estudo.ranking.empty
    assert len(estudo.resiliencia) == 2
    assert set(estudo.economia) == {r.conjunto.descricao() for r in estudo.resiliencia}
    # Com rede, a série vem do PVGIS; sem rede, cai para a sintética — e nesse
    # caso o aviso é obrigatório, porque a proposta não pode sair sem ele.
    if estudo.serie.fonte == "sintetica":
        assert any("sintética" in a or "PVGIS" in a for a in estudo.avisos)
    else:
        assert estudo.serie.confiavel

    escritos = escrever_relatorio(estudo, tmp_path)
    assert escritos["markdown"].exists()
    assert escritos["latex"].exists()
    assert (tmp_path / "excedencia_pico.png").exists()
    texto = escritos["markdown"].read_text(encoding="utf-8")
    assert "Estudo de armazenamento" in texto and "excedência" in texto


def test_estudo_sem_comodo_essencial_valido_falha_com_a_lista():
    from aurum.bateria.estudo import ConfiguracaoEstudo, executar_estudo

    comodos, instancias = cenario_exemplo("hotel")
    cfg = ConfiguracaoEstudo(
        latitude=-25.43, longitude=-49.27,
        comodos=comodos, instancias_por_comodo=instancias,
        comodos_essenciais=["Sala Inexistente"], simulacoes=5,
    )
    with pytest.raises(ValueError, match="disponíveis"):
        executar_estudo(cfg)


# ----------------------------------------------------------------------------
# A poda de candidatos, e o preço do banco quando não há kit solar
# ----------------------------------------------------------------------------
def test_a_poda_guarda_o_banco_que_a_meta_exige():
    """
    Podar por índice não olha para a carga, e por isso pula a resposta.

    Com 23 bancos em bloco e teto de 8, a amostragem uniforme saltou o de 2
    blocos — 9,3 kWh úteis, R\u00a037.500, que cumpre a meta de 6 h — e ofereceu o
    de 3, R\u00a012.000 mais caro. O estudo recomendou o maior porque o menor
    nunca chegou a ser avaliado, e a seção de escopos, que não poda nada,
    encontrou o menor: o mesmo documento passou a trazer dois bancos para a
    mesma casa e a mesma meta.
    """
    from aurum.bateria.catalogo import candidatos_em_blocos
    from aurum.bateria.estudo import _podar_perto_do_alvo

    candidatos = candidatos_em_blocos(carregar_catalogo())
    assert len(candidatos) > 8, "o caso de poda precisa de mais candidatos que o teto"
    candidatos.sort(key=lambda c: (c.modulos, c.potencia_descarga_kw))

    alvo = 5.75  # o que 0,96 kW de carga essencial exigem em 6 h
    podados = _podar_perto_do_alvo(candidatos, 8, alvo)
    assert len(podados) == 8

    # O menor que passa do alvo tem de sobreviver: é ele que a meta compra.
    acima = [c for c in candidatos if c.energia_util_kwh >= alvo]
    menor_que_serve = min(acima, key=lambda c: c.energia_util_kwh)
    assert any(
        c.energia_util_kwh == menor_que_serve.energia_util_kwh for c in podados
    ), "a poda jogou fora justamente o banco que cumpre a meta"

    # As pontas continuam: o menor mostra o que basta, o maior mostra o teto.
    assert podados[0] is candidatos[0]
    assert podados[-1] is candidatos[-1]


def test_sem_alvo_a_poda_volta_a_ser_uniforme():
    """Sem saber o que se procura, espalhar pela faixa é o melhor disponível."""
    from aurum.bateria.catalogo import candidatos_em_blocos
    from aurum.bateria.estudo import _podar_perto_do_alvo

    candidatos = sorted(candidatos_em_blocos(carregar_catalogo()),
                        key=lambda c: (c.modulos, c.potencia_descarga_kw))
    podados = _podar_perto_do_alvo(candidatos, 5, 0.0)
    assert len(podados) == 5
    assert podados[0] is candidatos[0] and podados[-1] is candidatos[-1]


def test_a_poda_nao_toca_em_lista_que_cabe_no_teto():
    from aurum.bateria.catalogo import candidatos_em_blocos
    from aurum.bateria.estudo import _podar_perto_do_alvo

    candidatos = candidatos_em_blocos(carregar_catalogo())
    assert _podar_perto_do_alvo(candidatos, 999, 5.0) == candidatos


def test_o_bloco_de_bateria_nao_depende_de_haver_kit_solar():
    """
    O preço do banco e a existência de um kit fotovoltaico são independentes.

    Quem compra só bateria compra o mesmo módulo de 5 kWh, pelo mesmo preço,
    sem comprar painel nenhum. A herança do bloco estava presa a
    ``topologia_kit == "splitphase"``; num estudo sem solar não há topologia a
    declarar, e o banco caía no modelo genérico — R$/kWh de catálogo mais
    inversor, tudo vezes 1,35 de instalação. Além de mais caro, isso cobra
    instalação de um módulo que já vem instalado.
    """
    from aurum.bateria.economia import PremissasBateria, _capex
    from aurum.pv.kits import BATERIA_BLOCO_BRL, BATERIA_BLOCO_KWH

    catalogo = carregar_catalogo()
    conjunto = next(
        c for c in catalogo.combinacoes(modulos=(2,))
        if c.energia_util_kwh > 0
    )

    generico = _capex(conjunto, PremissasBateria())[0]
    em_blocos = _capex(conjunto, PremissasBateria(
        bloco_bateria_kwh=BATERIA_BLOCO_KWH,
        bloco_bateria_brl=BATERIA_BLOCO_BRL,
        inversor_no_kit_fv=False,
    ))[0]
    assert em_blocos != generico, "o bloco tem de mudar o preço, senão não serve"

    # Sem kit, o inversor é compra à parte e entra na conta.
    com_kit = _capex(conjunto, PremissasBateria(
        bloco_bateria_kwh=BATERIA_BLOCO_KWH,
        bloco_bateria_brl=BATERIA_BLOCO_BRL,
        inversor_no_kit_fv=True,
    ))[0]
    assert em_blocos > com_kit, "sem kit fotovoltaico o inversor não está pago"
