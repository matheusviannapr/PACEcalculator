"""Motor fotovoltaico: recurso solar, equipamentos, layout, arranjo e economia."""
from __future__ import annotations

import math

import pytest
from shapely.geometry import Polygon

from aurum.pv import consumption, financials
from aurum.pv.equipment import EquipmentError, carregar_base
from aurum.pv.layout import (
    calcular_layout,
    estimar_por_area,
    fator_obstaculos_para,
    melhor_layout,
    passo_entre_fileiras,
)
from aurum.pv.sizing import (
    DC_AC_MAX,
    TEMP_MIN_PROJETO_C,
    melhor_sistema,
    modulos_por_string,
    strings_por_mppt,
    vmp_corrigida,
    voc_corrigida,
    dimensionar,
    dimensionar_por_consumo,
)
from aurum.pv.solar import (
    azimute_otimo,
    azimute_para_pvgis,
    inclinacao_otima,
    nome_orientacao,
    rendimento_estimado_kwh_kwp_ano,
)


# ----------------------------------------------------------------------
# Recurso solar — o bug do azimute
# ----------------------------------------------------------------------
def test_hemisferio_sul_aponta_para_o_norte():
    """
    Correção central: no Brasil o azimute ótimo é 0 (Norte), não 180 (Sul).
    A versão anterior usava 180 como padrão, o que custava ~26% da geração.
    """
    assert azimute_otimo(-25.49) == 0.0
    assert azimute_otimo(-3.10) == 0.0
    assert azimute_otimo(40.71) == 180.0


def test_conversao_de_azimute_para_pvgis():
    """PVGIS usa 0 = Sul; a convenção interna usa 0 = Norte."""
    assert azimute_para_pvgis(0) == 180.0     # Norte
    assert azimute_para_pvgis(180) == 0.0     # Sul
    assert azimute_para_pvgis(90) == -90.0    # Leste
    assert azimute_para_pvgis(270) == 90.0    # Oeste
    assert all(-180 <= azimute_para_pvgis(a) <= 180 for a in range(0, 360, 15))


def test_nome_das_orientacoes():
    assert nome_orientacao(0) == "Norte"
    assert nome_orientacao(180) == "Sul"
    assert nome_orientacao(359) == "Norte"


def test_inclinacao_respeita_piso_e_teto():
    """Piso de 10° para a chuva lavar o módulo; teto de 40° por esforço de vento."""
    assert inclinacao_otima(-3.1) == 10.0
    assert inclinacao_otima(-25.49) == pytest.approx(25.49)
    assert inclinacao_otima(-60.0) == 40.0


def test_rendimento_offline_dentro_da_faixa_brasileira():
    for latitude in (-33.0, -25.5, -15.8, -3.1):
        assert 1150 <= rendimento_estimado_kwh_kwp_ano(latitude) <= 1800
    # menor latitude absoluta rende mais
    assert rendimento_estimado_kwh_kwp_ano(-9.4) > rendimento_estimado_kwh_kwp_ano(-30.0)


def test_perfil_escala_linearmente(perfil_curitiba):
    assert perfil_curitiba.anual(100) == pytest.approx(100 * perfil_curitiba.anual_kwh_por_kwp)
    assert sum(perfil_curitiba.mensal(50)) == pytest.approx(perfil_curitiba.anual(50), rel=1e-9)
    assert 0.05 < perfil_curitiba.fator_capacidade() < 0.30


# ----------------------------------------------------------------------
# Equipamentos
# ----------------------------------------------------------------------
def test_catalogo_carrega(base_equipamentos):
    assert not base_equipamentos.vazia
    assert base_equipamentos.modulos and base_equipamentos.inversores


def test_dimensoes_do_modulo_batem_com_a_eficiencia(base_equipamentos):
    """
    Área derivada = Pmax / (eficiência x 1000 W/m²), pela definição de
    eficiência nas condições padrão de ensaio.
    """
    for modulo in base_equipamentos.modulos:
        if not modulo.dimensoes_derivadas or modulo.eficiencia_percent <= 0:
            continue
        esperada = modulo.potencia_wp / (modulo.eficiencia_percent / 100 * 1000)
        assert modulo.area_m2 == pytest.approx(esperada, rel=0.01)
        assert 1.0 < modulo.area_m2 < 4.0, "módulo fora de qualquer tamanho plausível"


def test_planilha_ausente_levanta_erro_claro():
    with pytest.raises(EquipmentError, match="não encontrada"):
        carregar_base("nao_existe_em_lugar_nenhum.xlsx")


# ----------------------------------------------------------------------
# Correção de temperatura — o bug que queima inversor
# ----------------------------------------------------------------------
def test_voc_sobe_no_frio(base_equipamentos):
    modulo = base_equipamentos.modulos[0]
    assert voc_corrigida(modulo, 0.0) > modulo.voc
    assert voc_corrigida(modulo, 25.0) == pytest.approx(modulo.voc)
    assert voc_corrigida(modulo, 0.0) / modulo.voc == pytest.approx(1.07, abs=0.005)


def test_vmp_cai_no_calor(base_equipamentos):
    modulo = base_equipamentos.modulos[0]
    assert vmp_corrigida(modulo, 70.0) < modulo.vmp


def test_string_maxima_nao_estoura_a_tensao_do_inversor(base_equipamentos):
    """
    Sem correção de temperatura, a string máxima ultrapassa a tensão do
    inversor numa manhã fria. Este teste garante que a corrigida não passa.
    """
    houve_par_valido = False
    for modulo in base_equipamentos.modulos:
        for inversor in base_equipamentos.inversores:
            _minimo, maximo = modulos_por_string(modulo, inversor)
            if maximo == 0:
                continue
            houve_par_valido = True
            tensao_fria = maximo * voc_corrigida(modulo, TEMP_MIN_PROJETO_C)
            assert tensao_fria <= inversor.tensao_max_cc, (
                f"{modulo.modelo} em {inversor.modelo}: {tensao_fria:.0f} V a "
                f"{TEMP_MIN_PROJETO_C:.0f} °C excede {inversor.tensao_max_cc:.0f} V"
            )
    assert houve_par_valido


def test_correcao_de_temperatura_e_mais_conservadora(base_equipamentos):
    """A lógica antiga (Voc de catálogo) permitiria mais módulos por string."""
    modulo = base_equipamentos.modulo_por_modelo("CS7L-575MS")
    inversor = base_equipamentos.inversor_por_modelo("MAX 75KTL3 LV")
    _minimo, maximo_corrigido = modulos_por_string(modulo, inversor)
    maximo_ingenuo = int(inversor.tensao_max_cc // modulo.voc)
    assert maximo_corrigido < maximo_ingenuo
    # e o ingênuo de fato estouraria
    assert maximo_ingenuo * voc_corrigida(modulo, 0.0) > inversor.tensao_max_cc


def test_string_minima_parte_o_inversor(base_equipamentos):
    """A string mínima precisa superar a tensão de partida mesmo no calor."""
    for modulo in base_equipamentos.modulos:
        for inversor in base_equipamentos.inversores:
            minimo, maximo = modulos_por_string(modulo, inversor)
            if maximo == 0:
                continue
            assert minimo * vmp_corrigida(modulo, 70.0) >= inversor.tensao_start


def test_corrente_por_mppt_respeita_os_dois_limites(base_equipamentos):
    for modulo in base_equipamentos.modulos:
        for inversor in base_equipamentos.inversores:
            n = strings_por_mppt(modulo, inversor)
            if n == 0:
                continue
            assert n * modulo.imp <= inversor.corrente_max_mppt
            assert n * modulo.isc <= inversor.corrente_curto_max_mppt


# ----------------------------------------------------------------------
# Layout — o ganho sobre a heurística de área
# ----------------------------------------------------------------------
def test_forma_do_telhado_altera_a_quantidade(base_equipamentos, galpao_utm, telhado_em_l_utm):
    """
    Duas coberturas de mesma área rendem números diferentes de módulos.
    A heurística antiga (área x fator / área do módulo) dava o mesmo para as
    duas, e era exatamente esse o erro.
    """
    modulo = base_equipamentos.modulo_mais_potente()
    retangular = calcular_layout(galpao_utm, modulo, montagem="coplanar", tilt_deg=8.0)
    # recorta o retângulo para a mesma área do L
    proporcional = calcular_layout(
        Polygon([(0, 0), (100, 0), (100, 40), (0, 40)]), modulo, montagem="coplanar", tilt_deg=8.0
    )
    em_l = calcular_layout(telhado_em_l_utm, modulo, montagem="coplanar", tilt_deg=8.0)

    assert telhado_em_l_utm.area == pytest.approx(4000, rel=0.01)
    assert em_l.quantidade < proporcional.quantidade
    assert em_l.quantidade < retangular.quantidade


def test_patio_interno_reduz_a_quantidade(base_equipamentos, galpao_utm):
    modulo = base_equipamentos.modulo_mais_potente()
    com_patio = Polygon(
        [(0, 0), (100, 0), (100, 50), (0, 50)],
        [[(40, 20), (60, 20), (60, 35), (40, 35)]],
    )
    cheio = calcular_layout(galpao_utm, modulo, montagem="coplanar", tilt_deg=8.0)
    vazado = calcular_layout(com_patio, modulo, montagem="coplanar", tilt_deg=8.0)
    assert vazado.quantidade < cheio.quantidade


def test_montagem_inclinada_cabe_menos_que_coplanar(base_equipamentos, galpao_utm):
    """Estrutura inclinada precisa de espaço entre fileiras contra sombreamento."""
    modulo = base_equipamentos.modulo_mais_potente()
    coplanar = calcular_layout(galpao_utm, modulo, montagem="coplanar", tilt_deg=8.0)
    inclinado = calcular_layout(galpao_utm, modulo, montagem="inclinado", tilt_deg=20.0)
    assert inclinado.quantidade < coplanar.quantidade
    # A taxa de ocupação, e não o passo entre fileiras, é o que se compara aqui:
    # cada montagem escolhe livremente retrato ou paisagem, e o passo medido em
    # retrato não é comparável com o medido em paisagem. A fração do telhado
    # coberta sobrevive a essa escolha.
    assert inclinado.taxa_ocupacao < coplanar.taxa_ocupacao


def test_passo_entre_fileiras_segue_a_ocupacao_do_solo(base_equipamentos):
    modulo = base_equipamentos.modulo_mais_potente()
    profundidade, _lateral = passo_entre_fileiras(modulo, "inclinado", 20.0, 0.45, 0.02, "retrato")
    esperado = modulo.comprimento_m * math.cos(math.radians(20.0)) / 0.45
    assert profundidade == pytest.approx(esperado, rel=1e-6)


def test_modulos_cabem_inteiros_dentro_do_telhado(base_equipamentos, galpao_utm):
    """Nenhum módulo do croqui pode ficar pendurado para fora da cobertura."""
    modulo = base_equipamentos.modulo_mais_potente()
    layout = calcular_layout(galpao_utm, modulo, montagem="coplanar", tilt_deg=8.0)
    assert layout.modulos_geom
    for retangulo in layout.modulos_geom[:80]:
        assert galpao_utm.contains(retangulo.buffer(-1e-9))


def test_recuo_de_borda_e_respeitado(base_equipamentos, galpao_utm):
    modulo = base_equipamentos.modulo_mais_potente()
    sem_recuo = calcular_layout(galpao_utm, modulo, montagem="coplanar", tilt_deg=8.0, recuo_m=0.0)
    com_recuo = calcular_layout(galpao_utm, modulo, montagem="coplanar", tilt_deg=8.0, recuo_m=3.0)
    assert com_recuo.quantidade < sem_recuo.quantidade
    assert com_recuo.area_util_m2 < sem_recuo.area_util_m2


def test_fator_de_obstaculos_reduz_e_mantem_o_croqui_coerente(base_equipamentos, galpao_utm):
    modulo = base_equipamentos.modulo_mais_potente()
    layout = calcular_layout(
        galpao_utm, modulo, montagem="coplanar", tilt_deg=8.0, fator_obstaculos=0.80
    )
    assert layout.quantidade < layout.quantidade_geometrica
    assert len(layout.modulos_geom) == layout.quantidade
    assert layout.quantidade == pytest.approx(layout.quantidade_geometrica * 0.80, rel=0.02)


def test_densidade_de_potencia_e_fisicamente_plausivel(base_equipamentos, galpao_utm):
    """Módulo de ~20% de eficiência não passa de ~200 Wp por m² de cobertura."""
    modulo = base_equipamentos.modulo_mais_potente()
    layout = calcular_layout(
        galpao_utm, modulo, montagem="coplanar", tilt_deg=8.0,
        fator_obstaculos=fator_obstaculos_para("warehouse"),
    )
    assert 80 < layout.densidade_wp_m2 < 200


def test_telhado_minusculo_nao_explode(base_equipamentos):
    modulo = base_equipamentos.modulo_mais_potente()
    layout = calcular_layout(Polygon([(0, 0), (3, 0), (3, 2), (0, 2)]), modulo, tilt_deg=15.0)
    assert layout.quantidade >= 0
    assert layout.avisos


def test_recuo_maior_que_o_telhado_devolve_zero(base_equipamentos):
    modulo = base_equipamentos.modulo_mais_potente()
    layout = calcular_layout(Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]), modulo, recuo_m=5.0)
    assert layout.quantidade == 0
    assert any("recuo" in aviso.lower() for aviso in layout.avisos)


def test_estimativa_por_area_fica_na_mesma_ordem_de_grandeza(base_equipamentos, galpao_utm):
    modulo = base_equipamentos.modulo_mais_potente()
    grade = calcular_layout(galpao_utm, modulo, montagem="coplanar", tilt_deg=8.0)
    aproximada = estimar_por_area(galpao_utm, modulo, montagem="coplanar", tilt_deg=8.0)
    assert 0.5 < aproximada.quantidade / grade.quantidade < 1.5


def test_melhor_layout_maximiza_potencia(base_equipamentos, galpao_utm):
    melhor = melhor_layout(galpao_utm, base_equipamentos.modulos, montagem="coplanar", tilt_deg=8.0)
    for modulo in base_equipamentos.modulos:
        candidato = calcular_layout(galpao_utm, modulo, montagem="coplanar", tilt_deg=8.0)
        assert melhor.potencia_kwp >= candidato.potencia_kwp - 1e-6


# ----------------------------------------------------------------------
# Arranjo elétrico
# ----------------------------------------------------------------------
def test_sistema_respeita_os_modulos_disponiveis(base_equipamentos):
    modulo = base_equipamentos.modulo_por_modelo("CS3W-450MS")
    sistema = dimensionar(500, modulo, base_equipamentos)
    assert sistema is not None
    assert sistema.modulos_totais <= 500
    assert sistema.potencia_cc_kwp == pytest.approx(sistema.modulos_totais * modulo.potencia_wp / 1000)


def test_sistema_nao_ultrapassa_a_potencia_fv_do_inversor(base_equipamentos):
    modulo = base_equipamentos.modulo_por_modelo("CS3W-450MS")
    sistema = dimensionar(2000, modulo, base_equipamentos)
    assert sistema is not None
    for arranjo in sistema.arranjos:
        potencia_por_unidade = arranjo.modulos_totais / arranjo.quantidade * modulo.potencia_wp
        assert potencia_por_unidade <= arranjo.inversor.potencia_fv_max_w
    assert sistema.razao_dc_ac <= DC_AC_MAX + 1e-6


def test_telhado_pequeno_nao_recebe_inversor_gigante(base_equipamentos):
    """Um telhado de ~10 kWp não pode ser atendido por um inversor de 75 kW."""
    quantidades = {m.modelo: 24 for m in base_equipamentos.modulos}
    sistema = melhor_sistema(quantidades, base_equipamentos)
    assert sistema is not None
    assert sistema.potencia_ca_kw < 30, sistema.descricao_inversores()


def test_telhado_grande_usa_inversores_grandes(base_equipamentos):
    quantidades = {m.modelo: 1500 for m in base_equipamentos.modulos}
    sistema = melhor_sistema(quantidades, base_equipamentos)
    assert sistema is not None
    assert max(a.inversor.potencia_ca_kw for a in sistema.arranjos) >= 75


def test_dimensionar_por_consumo_declara_quando_o_telhado_limita(base_equipamentos):
    modulo = base_equipamentos.modulo_por_modelo("CS3W-450MS")
    limitado = dimensionar_por_consumo(240_000, 1250.0, modulo, base_equipamentos, modulos_maximos=200)
    assert limitado is not None
    assert any("limita" in aviso for aviso in limitado.avisos)


def test_sem_modulos_disponiveis_nao_ha_sistema(base_equipamentos):
    modulo = base_equipamentos.modulos[0]
    assert dimensionar(0, modulo, base_equipamentos) is None


# ----------------------------------------------------------------------
# Consumo estimado
# ----------------------------------------------------------------------
def test_consumo_cresce_com_pavimentos():
    um = consumption.estimar(1000, "office", pavimentos=1)
    dez = consumption.estimar(1000, "office", pavimentos=10)
    assert dez.consumo_anual_kwh == pytest.approx(10 * um.consumo_anual_kwh)
    assert any("pavimentos" in aviso for aviso in dez.avisos)


def test_faixa_contem_a_estimativa_tipica():
    estimativa = consumption.estimar(2000, "supermarket")
    minimo, maximo = estimativa.faixa_mensal_kwh
    assert minimo <= estimativa.consumo_mensal_kwh <= maximo


def test_segmento_desconhecido_e_marcado_com_confianca_baixa():
    estimativa = consumption.estimar(3000, "yes")
    assert estimativa.confianca == "muito baixa"
    assert estimativa.avisos


def test_autoconsumo_reflete_o_horario_de_operacao():
    """Indústria diurna aproveita mais do que igreja, que opera à noite."""
    assert consumption.fracao_autoconsumo("industrial") > consumption.fracao_autoconsumo("church")
    assert 0 < consumption.fracao_autoconsumo("desconhecido") <= 1


def test_grande_consumidor_cai_no_grupo_a():
    _tarifa_pequeno, grupo_pequeno = consumption.tarifa_referencia("retail", 5_000)
    tarifa_grande, grupo_grande = consumption.tarifa_referencia("industrial", 200_000)
    assert "B3" in grupo_pequeno
    assert "A4" in grupo_grande
    assert tarifa_grande < 0.80


# ----------------------------------------------------------------------
# Economia
# ----------------------------------------------------------------------
def _premissas(**kwargs) -> financials.PremissasEconomicas:
    padrao = dict(tarifa_brl_kwh=0.92, capex_brl=2_000_000.0, fracao_autoconsumo=0.6,
                  ano_conexao=2026, consumo_anual_kwh=1_200_000.0)
    padrao.update(kwargs)
    return financials.PremissasEconomicas(**padrao)


def test_capex_cai_por_kwp_com_a_escala():
    pequeno = financials.estimar_capex(10) / 10
    grande = financials.estimar_capex(1000) / 1000
    assert grande < pequeno
    assert 1500 < grande < 5000



def test_padrao_alto_reproduz_a_obra_real():
    """
    O nível "alto" é calibrado, não chutado.

    Ed. Mourisco (Rio, 2026): 312 kWp fechados a R$ 1.280.400, ou
    R$ 4.104/kWp. É o único ponto de calibração que existe para o padrão alto,
    e se a curva deixar de passar por ele o número perde a procedência que o
    documento afirma ter.
    """
    capex = financials.estimar_capex(312.0, padrao="alto")
    assert capex == pytest.approx(1_280_400.0, rel=0.01)
    assert capex / 312.0 == pytest.approx(4_104.0, rel=0.01)


def test_os_tres_padroes_estao_em_ordem():
    kwp = 200.0
    basico = financials.estimar_capex(kwp, padrao="basico")
    padrao = financials.estimar_capex(kwp, padrao="padrao")
    alto = financials.estimar_capex(kwp, padrao="alto")
    assert basico < padrao < alto
    # A diferença entre as pontas é grande o bastante para mudar a decisão, e
    # é por isso que o documento tem de declarar qual foi usada.
    assert alto / basico > 1.5


def test_cotacao_informada_passa_na_frente_da_curva():
    """Preço de verdade é melhor que qualquer curva, e tem de vencer."""
    assert financials.estimar_capex(
        312.0, referencia_brl_kwp=5_000.0, padrao="basico"
    ) == pytest.approx(312.0 * 5_000.0 * (3.12 ** -0.12), rel=1e-6)


def test_padrao_desconhecido_falha_dizendo_os_conhecidos():
    with pytest.raises(ValueError, match="basico"):
        financials.estimar_capex(100.0, padrao="luxo")


def test_sem_padrao_vale_a_configuracao():
    """Os chamadores antigos não podem mudar de resposta."""
    from aurum.config import get_settings

    referencia = get_settings().solar.capex_reference_brl_per_kwp
    assert financials.estimar_capex(100.0) == pytest.approx(referencia * 100.0)

def test_composicao_do_capex_fecha_no_total():
    total = 1_994_661.32
    itens = financials.composicao_capex(total)
    assert sum(itens.values()) == pytest.approx(total, abs=0.01)


def test_cronograma_do_fio_b():
    assert financials.fator_fio_b(2022) == 0.0
    assert financials.fator_fio_b(2023) == 0.15
    assert financials.fator_fio_b(2026) == 0.60
    assert financials.fator_fio_b(2035) == 1.0


def test_lei_14300_reduz_a_economia():
    """Ignorar o Fio B superestima o retorno — era o que a versão anterior fazia."""
    geracao = 1_000_000.0
    com = financials.calcular(geracao, _premissas(aplicar_lei_14300=True))
    sem = financials.calcular(geracao, _premissas(aplicar_lei_14300=False))
    assert com.economia_ano1_brl < sem.economia_ano1_brl
    assert com.vpl_brl < sem.vpl_brl
    assert all(linha.custo_fio_b_brl > 0 for linha in com.fluxo)
    assert all(linha.custo_fio_b_brl == 0 for linha in sem.fluxo)


def test_autoconsumo_total_nao_paga_fio_b():
    """Sem injeção não há Fio B: a lei incide sobre a energia que vai para a rede."""
    resultado = financials.calcular(1_000_000.0, _premissas(fracao_autoconsumo=1.0))
    assert all(linha.custo_fio_b_brl == 0 for linha in resultado.fluxo)


def test_economia_nao_ultrapassa_o_consumo():
    """Gerar o dobro do que se consome não dobra a economia."""
    premissas = _premissas(consumo_anual_kwh=500_000.0)
    modesto = financials.calcular(500_000.0, premissas)
    exagerado = financials.calcular(1_000_000.0, premissas)
    assert exagerado.economia_ano1_brl == pytest.approx(modesto.economia_ano1_brl)
    assert any("supera o consumo" in aviso for aviso in exagerado.avisos)


def test_fluxo_tem_o_horizonte_pedido_e_degrada():
    resultado = financials.calcular(1_000_000.0, _premissas(anos=25))
    assert len(resultado.fluxo) == 25
    assert resultado.fluxo[-1].geracao_kwh < resultado.fluxo[0].geracao_kwh
    assert resultado.fluxo[-1].tarifa_brl_kwh > resultado.fluxo[0].tarifa_brl_kwh


def test_vpl_acumulado_bate_com_a_soma_dos_descontados():
    resultado = financials.calcular(1_000_000.0, _premissas())
    soma = -resultado.premissas.capex_brl + sum(l.fluxo_descontado_brl for l in resultado.fluxo)
    assert resultado.vpl_brl == pytest.approx(soma, rel=1e-9)


def test_tir_zera_o_vpl_do_proprio_fluxo():
    """A TIR precisa ser a do fluxo que a proposta mostra, não de outro."""
    resultado = financials.calcular(1_000_000.0, _premissas())
    assert resultado.tir_anual is not None
    caixa = [-resultado.premissas.capex_brl] + [l.fluxo_liquido_brl for l in resultado.fluxo]
    vpl = sum(f / ((1 + resultado.tir_anual) ** i) for i, f in enumerate(caixa))
    assert vpl == pytest.approx(0.0, abs=abs(resultado.premissas.capex_brl) * 1e-4)


def test_projeto_inviavel_nao_inventa_payback():
    resultado = financials.calcular(1_000_000.0, _premissas(tarifa_brl_kwh=0.02))
    assert resultado.vpl_brl < 0
    assert resultado.payback_descontado_anos is None
    assert any("não se torna positivo" in aviso for aviso in resultado.avisos)


def test_lcoe_abaixo_da_tarifa_indica_viabilidade():
    resultado = financials.calcular(1_000_000.0, _premissas())
    assert resultado.lcoe_brl_kwh is not None
    assert resultado.lcoe_brl_kwh < resultado.premissas.tarifa_brl_kwh


def test_co2_usa_o_fator_do_sin():
    resultado = financials.calcular(1_000_000.0, _premissas())
    esperado = resultado.geracao_total_kwh / 1000 * financials.FATOR_EMISSAO_SIN_TCO2_MWH
    assert resultado.co2_evitado_t == pytest.approx(esperado)
    assert resultado.arvores_equivalentes > 0


def test_sensibilidade_move_todos_os_indicadores():
    """
    A sensibilidade antiga só multiplicava a economia. Payback, VPL e TIR
    também precisam responder à tarifa.
    """
    cenarios = financials.sensibilidade_tarifa(1_000_000.0, _premissas())
    assert set(cenarios) >= {"-20%", "base", "+20%"}
    assert cenarios["-20%"]["vpl"] < cenarios["base"]["vpl"] < cenarios["+20%"]["vpl"]
    assert cenarios["-20%"]["payback"] > cenarios["+20%"]["payback"]
    assert cenarios["-20%"]["tir"] < cenarios["+20%"]["tir"]


def test_empacotamento_de_galpao_grande_e_rapido(base_equipamentos):
    """
    A prospecção empacota cada modelo de módulo em cada telhado do lote.
    Com o laço posição a posição, um galpão de 25.000 m² levava mais de 4 s
    para os quatro módulos; a versão vetorizada fica em fração de segundo.
    """
    import time

    galpao = Polygon([(0, 0), (250, 0), (250, 100), (0, 100)])
    inicio = time.perf_counter()
    for modulo in base_equipamentos.modulos:
        calcular_layout(galpao, modulo, montagem="coplanar", tilt_deg=8.0)
    decorrido = time.perf_counter() - inicio
    assert decorrido < 2.0, f"empacotamento levou {decorrido:.1f}s"


def test_croqui_vetorizado_mantem_os_modulos_dentro(base_equipamentos):
    """A rotação em lote precisa devolver os módulos na posição original."""
    telhado = Polygon([(10, 20), (90, 20), (90, 60), (10, 60)])
    modulo = base_equipamentos.modulo_mais_potente()
    layout = calcular_layout(
        telhado, modulo, montagem="coplanar", tilt_deg=8.0, azimute_fileiras_deg=37.0
    )
    assert layout.quantidade > 0
    assert len(layout.modulos_geom) == layout.quantidade
    for retangulo in layout.modulos_geom:
        assert telhado.contains(retangulo.buffer(-1e-9))
        # a área de cada módulo se conserva sob rotação
        assert retangulo.area == pytest.approx(modulo.area_m2, rel=0.01)
