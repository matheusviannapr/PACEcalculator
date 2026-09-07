"""
Testes da biblioteca de cargas e do cenário editável.

Dois grupos com papéis distintos:

* **Integridade do catálogo e dos modelos** — que nenhum modelo aponte para
  equipamento inexistente e que todo rascunho de segmento nasça simulável. É o
  que garante que o passo 2 da interface nunca abra quebrado.
* **Coerência física dos modelos semeados** — cada segmento é simulado e
  comparado contra a curva de referência do próprio setor. Foi esse teste que
  pegou o quarto de hotel com pico ao meio-dia (hora em que o quarto está
  vazio) e a academia com pico às 7 h.

A validação do cenário tem testes próprios para as três armadilhas silenciosas
do D², porque cada uma delas produz um número plausível e errado.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aurum.demanda import simular_ensemble
from aurum.demanda.biblioteca import (
    CATEGORIAS,
    EQUIPAMENTOS,
    LIMITE_DIVERGENCIA_PP,
    MODELOS,
    PERIODOS,
    calibrar_por_conta,
    catalogo_como_dataframe,
    comparar_com_perfil,
    equipamento,
    linha_de_planilha,
    perfil_tipico,
    perfis_tipicos,
    segmento_do_perfil,
)
from aurum.demanda.cenario import Cenario
from aurum.demanda.ensemble import ensemble_de_curva_tipica


# ----------------------------------------------------------------------------
# Catálogo
# ----------------------------------------------------------------------------
def test_catalogo_tem_categorias_e_campos_completos():
    assert len(EQUIPAMENTOS) > 50
    assert len(CATEGORIAS) >= 10
    for item in EQUIPAMENTOS:
        assert item["potencia_w"] > 0, item["nome"]
        assert 0 < item["probabilidade"] <= 1, item["nome"]
        assert 0 < item["fd"] <= 1.0, item["nome"]
        assert item["tipo_intervalo"] in ("fixo", "dinâmico"), item["nome"]


def test_equipamento_dinamico_sempre_traz_duracao():
    """Sem duração, o tipo dinâmico cai no parser legado e vira 1 h à meia-noite."""
    for item in EQUIPAMENTOS:
        if item["tipo_intervalo"] == "dinâmico":
            assert item["duracao_min"] is not None, item["nome"]
            assert item["duracao_max"] is not None, item["nome"]


def test_nomes_do_catalogo_sao_unicos():
    nomes = [e["nome"] for e in EQUIPAMENTOS]
    assert len(nomes) == len(set(nomes))


def test_ajuste_sobrescreve_o_catalogo():
    padrao = linha_de_planilha("Ar-condicionado split 9.000 BTU")
    ajustada = linha_de_planilha(
        "Ar-condicionado split 9.000 BTU", 2, intervalo="18:00 as 10:00", probabilidade=0.8)
    assert padrao["intervalo"] != ajustada["intervalo"]
    assert ajustada["Quantidade"] == 2
    assert ajustada["probabilidade"] == 0.8
    # O ajuste não pode contaminar o catálogo para a próxima chamada.
    assert linha_de_planilha("Ar-condicionado split 9.000 BTU")["intervalo"] == padrao["intervalo"]


def test_duracao_ausente_vira_nan_e_nao_none():
    """`None` numa coluna numérica aparece como o texto 'None' na grade."""
    linha = linha_de_planilha("Lâmpada LED bulbo 9 W")
    assert isinstance(linha["duracao_min"], float) and np.isnan(linha["duracao_min"])


def test_catalogo_como_dataframe_filtra_por_categoria():
    tudo = catalogo_como_dataframe()
    climatizacao = catalogo_como_dataframe("Climatização")
    assert len(climatizacao) < len(tudo)
    assert set(climatizacao["categoria"]) == {"Climatização"}


# ----------------------------------------------------------------------------
# Modelos de segmento
# ----------------------------------------------------------------------------
def test_todo_modelo_referencia_equipamento_existente():
    for identificador, segmento in MODELOS.items():
        for comodo in segmento.comodos:
            for item in comodo.equipamentos:
                equipamento(item[0])  # levanta KeyError se não existir


def test_todo_modelo_tem_perfil_de_referencia():
    for identificador in MODELOS:
        assert segmento_do_perfil(identificador) is not None, identificador


def test_todo_modelo_tem_ao_menos_um_comodo_essencial():
    """Sem sugestão de essenciais, o passo do backup abre com o prédio inteiro."""
    for identificador, segmento in MODELOS.items():
        assert any(c.essencial for c in segmento.comodos), identificador


@pytest.mark.parametrize("identificador", list(MODELOS))
def test_rascunho_do_segmento_nasce_valido(identificador):
    cenario = Cenario.de_segmento(identificador, "teste")
    problemas = [p for p in cenario.validar() if p.impede]
    assert not problemas, [str(p) for p in problemas]
    assert cenario.potencia_instalada_w() > 0
    # Menor ou igual: no hospital, todos os ambientes são essenciais — e isso
    # é a resposta certa, não um defeito do modelo.
    assert 0 < cenario.potencia_instalada_w(True) <= cenario.potencia_instalada_w()


def test_hotel_tem_backup_bem_menor_que_o_total():
    """O recorte precisa recortar de verdade em quem tem carga dispensável."""
    cenario = Cenario.de_segmento("hotel", "teste")
    assert cenario.potencia_instalada_w(True) < 0.5 * cenario.potencia_instalada_w()


@pytest.mark.parametrize("identificador", list(MODELOS))
def test_modelo_semeado_e_coerente_com_a_curva_do_setor(identificador):
    """
    A energia de cada período de 6 h tem que cair perto do padrão do segmento.

    Não é a correlação hora a hora: essa é frágil e dispara com um único pico
    legítimo. É a distribuição entre madrugada, manhã, tarde e noite, que só
    se desloca quando a energia foi mesmo parar na hora errada.
    """
    cenario = Cenario.de_segmento(identificador, "teste")
    ensemble = simular_ensemble(
        cenario.para_comodos(), cenario.instancias_de(), num_simulacoes=40
    ).reamostrar(5)
    comparacao = comparar_com_perfil(
        ensemble.perfil_medio_w(), segmento_do_perfil(identificador), 5)
    assert comparacao["coerente"], (
        f"{identificador}: {comparacao['divergencia_do_pior_periodo_pp']:.1f} pp de desvio "
        f"na {comparacao['pior_periodo']} (limite {LIMITE_DIVERGENCIA_PP} pp)"
    )


# ----------------------------------------------------------------------------
# Curvas típicas
# ----------------------------------------------------------------------------
def test_curvas_tipicas_somam_um():
    perfis = perfis_tipicos()
    assert len(perfis) >= 10
    for perfil in perfis.values():
        assert perfil.curva_pu.shape == (24,)
        assert perfil.curva_pu.sum() == pytest.approx(1.0, abs=1e-9)
        assert (perfil.curva_pu >= 0).all()


def test_perfis_sinteticos_sao_declarados_como_tal():
    """A base marca os perfis como sintéticos; o código não pode esconder isso."""
    assert not perfil_tipico("hotel").confiavel


def test_calibracao_pela_conta_fecha_a_energia():
    perfil = perfil_tipico("comercial_generico")
    resultado = calibrar_por_conta(perfil, consumo_mensal_kwh=9000.0, dias_operacao_mes=30)
    assert resultado["consumo_diario_kwh"] == pytest.approx(300.0)
    assert float(np.sum(resultado["curva_kw"])) == pytest.approx(300.0)
    assert "não medição" in resultado["aviso"] or "hipótese" in resultado["aviso"] \
        or "subestima" in resultado["aviso"]


def test_ensemble_de_curva_tipica_tem_cauda():
    """Sem dispersão, a curva de excedência degenera num único valor."""
    perfil = perfil_tipico("comercial_generico")
    calibrado = calibrar_por_conta(perfil, 9000.0, 30)
    ensemble = ensemble_de_curva_tipica(calibrado["curva_w_por_minuto"], num_simulacoes=200)
    picos = ensemble.picos_diarios_w()
    assert picos.std() > 0
    assert ensemble.metadados["origem"] == "curva_tipica_calibrada"
    # A energia média tem que continuar batendo com a conta.
    assert ensemble.energia_diaria_kwh().mean() == pytest.approx(300.0, rel=0.05)


def test_comparacao_detecta_energia_na_hora_errada():
    """
    Trocar dia por noite tem que reprovar.

    O deslocamento de 12 h, e não a inversão temporal, é o erro que se quer
    pegar: uma curva comercial invertida no tempo continua com a energia de
    dia, porque ela é quase simétrica em torno do meio-dia — e nesse caso o
    veredito "coerente" está certo.
    """
    perfil = perfil_tipico("comercial_generico")
    trocada = np.repeat(np.roll(perfil.curva_pu, 12), 60) * 1000.0
    comparacao = comparar_com_perfil(trocada, perfil, passo_min=1)
    assert not comparacao["coerente"]
    assert comparacao["divergencia_do_pior_periodo_pp"] > LIMITE_DIVERGENCIA_PP


def test_inversao_temporal_de_curva_simetrica_nao_alarma():
    """O contrapositivo: a métrica não pode disparar por simetria."""
    perfil = perfil_tipico("comercial_generico")
    invertida = np.repeat(perfil.curva_pu[::-1], 60) * 1000.0
    assert comparar_com_perfil(invertida, perfil, passo_min=1)["coerente"]


def test_comparacao_aprova_a_propria_curva():
    perfil = perfil_tipico("hotel")
    igual = np.repeat(perfil.curva_pu, 60) * 1000.0
    comparacao = comparar_com_perfil(igual, perfil, passo_min=1)
    assert comparacao["coerente"]
    assert comparacao["correlacao"] == pytest.approx(1.0)
    assert set(comparacao["energia_por_periodo_pu"]) == set(PERIODOS)


# ----------------------------------------------------------------------------
# Cenário: as três armadilhas do D²
# ----------------------------------------------------------------------------
def _cenario_com(linha: list) -> Cenario:
    tabela = pd.DataFrame([linha], columns=[
        "Equipamento", "Potência", "Quantidade", "Tipo de intervalo", "intervalo",
        "probabilidade", "FD", "duracao_min", "duracao_max", "modo_fixo"])
    return Cenario(nome="t", comodos={"Sala": Cenario._normalizar(tabela)},
                   instancias={"Sala": 1})


def test_dinamico_sem_duracao_e_erro_e_explica_o_silencio():
    cenario = _cenario_com(
        ["Bomba", 750, 1, "dinâmico", "08:00 as 18:00", 1.0, 1.0, np.nan, np.nan, np.nan])
    problemas = [p for p in cenario.validar() if p.impede]
    assert problemas
    assert "meia-noite" in problemas[0].mensagem
    assert not cenario.valido


def test_janela_noturna_e_aviso_nao_erro():
    cenario = _cenario_com(
        ["Luz", 500, 1, "fixo", "22:00 as 05:00", 1.0, 1.0, np.nan, np.nan, "FIXO_100%"])
    problemas = cenario.validar()
    assert cenario.valido
    assert any("meia-noite" in p.mensagem for p in problemas)


def test_duracao_maior_que_a_janela_avisa_do_encurtamento():
    cenario = _cenario_com(
        ["Forno", 5000, 1, "dinâmico", "08:00 as 10:00", 1.0, 1.0, 5.0, 8.0, np.nan])
    assert cenario.valido
    assert any("encurtada" in p.mensagem for p in cenario.validar())


def test_erros_triviais_de_digitacao():
    cenario = _cenario_com(
        ["", 0, 0, "fixo", "08:00 as 18:00", 1.5, 0.0, np.nan, np.nan, "FIXO_100%"])
    campos = {p.campo for p in cenario.validar() if p.impede}
    assert {"Equipamento", "Potência", "Quantidade", "probabilidade", "FD"} <= campos


def test_para_comodos_recusa_cenario_invalido():
    cenario = _cenario_com(
        ["Sem potência", 0, 1, "fixo", "08:00 as 18:00", 1.0, 1.0, np.nan, np.nan, "FIXO_100%"])
    with pytest.raises(ValueError, match="erro"):
        cenario.para_comodos()


# ----------------------------------------------------------------------------
# Cenário: edição e planilha
# ----------------------------------------------------------------------------
def test_planilha_faz_ida_e_volta(tmp_path):
    original = Cenario.de_segmento("escritorio", "Escritório X")
    caminho = tmp_path / "cargas.xlsx"
    original.para_planilha(caminho)

    lido = Cenario.de_planilha(caminho, "Escritório X")
    assert set(lido.comodos) == set(original.comodos)
    assert lido.instancias == original.instancias
    assert lido.total_de_equipamentos() == original.total_de_equipamentos()
    assert lido.potencia_instalada_w() == pytest.approx(original.potencia_instalada_w())
    assert not [p for p in lido.validar() if p.impede]


def test_comodo_com_barra_no_nome_ainda_exporta(tmp_path):
    """
    O erro que derrubava a tela inteira no download da planilha.

    O Excel proíbe ``\\ / * ? : [ ]`` no nome da aba, e a vistoria devolve
    cômodos como "Sala de TV / Cinema" e "Espaço Gourmet / Churrasqueira". A
    exceção vinha lá de dentro do openpyxl e não dizia qual cômodo era o
    culpado — o usuário via só um app quebrado.
    """
    cenario = Cenario.de_segmento("escritorio", "X")
    primeiro = next(iter(cenario.comodos))
    cenario.comodos["Sala de TV / Cinema"] = cenario.comodos.pop(primeiro)
    cenario.instancias["Sala de TV / Cinema"] = cenario.instancias.pop(primeiro, 1)

    caminho = tmp_path / "cargas.xlsx"
    cenario.para_planilha(caminho)

    lido = Cenario.de_planilha(caminho, "X")
    assert "Sala de TV - Cinema" in lido.comodos, "a barra vira hífen"
    assert lido.total_de_equipamentos() == cenario.total_de_equipamentos()


def test_nomes_longos_parecidos_nao_colidem_na_mesma_aba():
    """
    O corte em 31 caracteres pode colar dois cômodos no mesmo nome.

    Sem desambiguar, o Excel recusa a segunda aba — ou, dependendo da versão,
    a sobrescreve em silêncio, que é o pior dos dois: a planilha sai com um
    cômodo a menos e ninguém percebe.
    """
    from aurum.demanda.cenario import LIMITE_NOME_DE_ABA, nome_de_aba

    longo = "Area de servico e lavanderia dos fundos"
    usados: set[str] = set()
    nomes = [nome_de_aba(longo, usados) for _ in range(3)]
    assert len(set(nomes)) == 3
    assert all(len(n) <= LIMITE_NOME_DE_ABA for n in nomes)


def test_nome_de_aba_nunca_sai_vazio():
    """Aba sem nome é recusada pelo Excel tanto quanto aba com barra."""
    from aurum.demanda.cenario import nome_de_aba

    assert nome_de_aba("") == "comodo"
    assert nome_de_aba("   ") == "comodo"
    assert nome_de_aba("///") == "---"


def test_adicionar_e_remover_comodo():
    cenario = Cenario.de_segmento("varejo", "Loja")
    antes = len(cenario.comodos)
    cenario.adicionar_comodo("Depósito", instancias=2)
    assert len(cenario.comodos) == antes + 1
    cenario.adicionar_equipamento("Depósito", "Luminária LED tubular 20 W", 10)
    assert len(cenario.comodos["Depósito"]) == 1

    with pytest.raises(ValueError, match="já existe"):
        cenario.adicionar_comodo("Depósito")

    cenario.remover_comodo("Depósito")
    assert len(cenario.comodos) == antes
    assert "Depósito" not in cenario.instancias


def test_essenciais_recortam_a_potencia():
    cenario = Cenario.de_segmento("hotel", "Hotel")
    total = cenario.potencia_instalada_w()
    backup = cenario.potencia_instalada_w(apenas_essenciais=True)
    assert 0 < backup < total
    assert set(cenario.instancias_de(True)) == set(cenario.essenciais)


def test_grade_nao_mostra_a_palavra_none():
    """
    Célula vazia tem que sair em branco na grade de edição, não como "None".

    O caminho que produzia isso: o `st.data_editor` devolve colunas `object`
    depois de uma edição, e `None` numa coluna de texto é desenhado como a
    palavra. A normalização na volta é o que mantém o tipo e o branco.
    """
    cenario = Cenario.de_segmento("hotel", "teste")
    for tabela in cenario.comodos.values():
        assert int((tabela.astype(str) == "None").sum().sum()) == 0
        # Ida e volta pela grade, que devolve tudo como objeto.
        volta = Cenario._normalizar(tabela.astype(object))
        assert int((volta.astype(str) == "None").sum().sum()) == 0
        assert volta["duracao_min"].dtype.kind == "f"
        assert volta["Potência"].dtype.kind == "f"


def test_modo_fixo_so_aparece_onde_se_aplica():
    """O D² só lê `modo_fixo` quando o uso é fixo; na linha dinâmica confunde."""
    cenario = Cenario.de_segmento("hotel", "teste")
    for tabela in cenario.comodos.values():
        dinamicas = tabela["Tipo de intervalo"].str.strip().str.lower() == "dinâmico"
        assert tabela.loc[dinamicas, "modo_fixo"].isna().all()
        assert tabela.loc[~dinamicas, "modo_fixo"].notna().all()


def test_todo_segmento_tem_custo_de_interrupcao_positivo():
    """
    Nenhum segmento pode sugerir que ficar sem energia é de graça.

    O valor é ordem de grandeza e o usuário troca — mas o padrão precisa ser
    uma afirmação defensável, e zero não é: zero diz que a resiliência não
    vale nada, e é a premissa que decide se a bateria se paga.
    """
    from aurum.demanda.biblioteca import MODELOS

    for chave, segmento in MODELOS.items():
        assert segmento.custo_interrupcao_brl_kwh > 0, chave

    # A escala precisa ordenar o que a realidade ordena: carga refrigerada
    # perde estoque, escritório perde hora de trabalho, escola remarca a aula.
    assert MODELOS["frigorifico"].custo_interrupcao_brl_kwh > MODELOS["escritorio"].custo_interrupcao_brl_kwh
    assert MODELOS["escritorio"].custo_interrupcao_brl_kwh > MODELOS["escola"].custo_interrupcao_brl_kwh
    assert MODELOS["hospital"].custo_interrupcao_brl_kwh == max(
        s.custo_interrupcao_brl_kwh for s in MODELOS.values()
    ), "hospital é o teto: ali o número não é econômico, é de segurança"


# ----------------------------------------------------------------------------
# Janelas múltiplas
# ----------------------------------------------------------------------------
def _casa_com_microondas(intervalo: str) -> pd.DataFrame:
    """Um micro-ondas de duração intervalar, e nada mais, para isolar a janela."""
    return pd.DataFrame([{
        "Equipamento": "Micro-ondas", "Potência": 1200.0, "Quantidade": 1,
        "Tipo de intervalo": "dinâmico", "intervalo": intervalo,
        "probabilidade": 1.0, "FD": 1.0,
        "duracao_min": 0.25, "duracao_max": 0.25,
    }])


def _energia_por_hora(intervalo: str, simulacoes: int = 400) -> np.ndarray:
    """A curva média em kW, hora a hora, de uma casa com um micro-ondas só."""
    from aurum.demanda.nucleo import cria_comodo_da_planilha
    comodo = cria_comodo_da_planilha(_casa_com_microondas(intervalo), "Cozinha")
    ensemble = simular_ensemble([comodo], {"Cozinha": 1}, simulacoes)
    ensemble = ensemble.reamostrar(60) if ensemble.passo_min == 1 else ensemble
    return ensemble.perfil_medio_w() / 1000.0


def test_a_segunda_janela_declarada_nao_e_descartada():
    """
    O bug que produzia uma casa com um cume só, no horário errado.

    ``parse_janela_operacao`` devolvia ``intervalos[0]`` e jogava fora o resto
    sem avisar. Para janela única -- que é quase todo equipamento levantado em
    campo -- nunca apareceu; para um micro-ondas declarado no almoço **e** no
    jantar, o jantar simplesmente não acontecia, e ninguém tinha como notar
    olhando o número.
    """
    curva = _energia_por_hora("11:30 as 14:00 e 18:30 as 22:00")
    almoco = curva[11:14].sum()
    jantar = curva[18:22].sum()
    assert almoco > 0.0, "o almoço acontece"
    assert jantar > 0.0, "e o jantar também — era isto que se perdia"
    # Fora das duas janelas a casa está desligada.
    assert curva[15:18].sum() == pytest.approx(0.0, abs=1e-6)


def test_uma_janela_so_continua_como_era():
    """A correção não pode mexer no caso comum, que é a janela única."""
    curva = _energia_por_hora("18:30 as 22:00")
    assert curva[11:14].sum() == pytest.approx(0.0, abs=1e-6)
    assert curva[18:22].sum() > 0.0


def test_duas_janelas_sao_dois_usos_e_nao_um_sorteado():
    """
    Quem declara duas janelas está dizendo que o aparelho é usado nas duas.

    Sortear qual delas devolveria a mesma energia diária num cume só, deslocado
    de um dia para o outro — o que na média some, e no pico não.
    """
    uma = _energia_por_hora("18:30 as 22:00").sum()
    duas = _energia_por_hora("11:30 as 14:00 e 18:30 as 22:00").sum()
    assert duas == pytest.approx(2.0 * uma, rel=0.15)
