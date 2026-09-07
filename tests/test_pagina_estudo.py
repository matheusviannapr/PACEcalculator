"""
Testes da interface do estudo de energia.

Percorrem os oito passos com o `AppTest` do Streamlit, que executa o script de
verdade e os mesmos callbacks que o navegador dispara. O que estes testes pegam
e um teste de biblioteca não pega é o erro de **estado entre reruns** — foi
assim que apareceram os `session_state.get(chave, padrão)` devolvendo o `None`
já guardado em vez do padrão.

O outro papel deles é de contrato de experiência, em dois níveis:

* **A fronteira entre as fases.** O dimensionamento tem que fechar sozinho,
  sem que nenhuma pergunta sobre apagão apareça antes dele. Se um passo da
  fase 2 vazar para a fase 1, o teste que atravessa a fase 1 quebra.
* **Os padrões.** Se um passo passar a exigir uma decisão que antes tinha
  padrão, o teste que atravessa só apertando "Continuar" quebra — e é isso que
  se quer que quebre.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
APP = str(RAIZ / "app.py")

# Numeração dos passos, para os testes falarem a mesma língua da interface.
CLIENTE, CARGAS, CONSUMO, TELHADO, EQUIPAMENTO, BACKUP, META, RESULTADO = range(1, 9)


def _por_rotulo(colecao, trecho: str):
    """Acha um widget pelo rótulo: índice posicional quebra a cada ajuste de layout."""
    trecho = trecho.lower()
    for widget in colecao:
        if trecho in (getattr(widget, "label", None) or "").lower():
            return widget
    rotulos = [getattr(w, "label", "?") for w in colecao]
    raise AssertionError(f"widget com '{trecho}' não encontrado. Existentes: {rotulos}")


def _abrir() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=400).run()
    assert not at.exception, at.exception
    at.sidebar.radio[0].set_value("Estudo de energia").run()
    assert not at.exception, at.exception
    return at


def _continuar(at: AppTest) -> AppTest:
    _por_rotulo(at.button, "Continuar").click().run()
    assert not at.exception, at.exception
    return at


def _ate_o_consumo(at: AppTest, simulacoes: int = 50) -> AppTest:
    """Passos 1 a 3: nome, rascunho de cargas e a análise do consumo."""
    _por_rotulo(at.text_input, "Nome do cliente").set_value("Hotel Teste").run()
    _continuar(at)
    _continuar(at)
    _por_rotulo(at.slider, "Simulações por estação").set_value(simulacoes).run()
    _por_rotulo(at.button, "Analisar o consumo").click().run()
    assert not at.exception, at.exception
    return at


def _ate_o_equipamento(at: AppTest, kwp: float = 50.0) -> AppTest:
    """Segue do consumo até o dimensionamento fechado."""
    _continuar(at)  # consumo -> telhado
    at.radio[0].set_value("Já sei a potência").run()
    _por_rotulo(at.number_input, "Potência do sistema").set_value(kwp).run()
    _continuar(at)  # telhado -> equipamento
    return at


# ----------------------------------------------------------------------------
# Estrutura em duas fases
# ----------------------------------------------------------------------------
def test_abre_no_passo_1_sem_erro():
    at = _abrir()
    assert at.session_state["e_passo"] == CLIENTE
    assert any("onde fica" in t.value.lower() for t in at.title)
    # A prospecção não pode vazar para dentro do estudo.
    assert not any("Varrer região" in b.label for b in at.button)


def test_a_fase_de_dimensionamento_nao_pergunta_sobre_apagao():
    """
    A fronteira entre as fases é o ponto da reestruturação.

    Quem só quer dimensionar não deve topar com autonomia, apagão ou bateria
    antes de o dimensionamento fechar.
    """
    at = _abrir()
    proibidos = ("apagão", "autonomia", "bateria", "backup", "falta de energia")
    for _ in range(3):  # passos 1, 2 e 3
        # Só o corpo da página: a barra lateral lista as duas fases de
        # propósito, para o usuário saber o que vem depois.
        laterais = {id(w) for w in at.sidebar.button}
        rotulos = " ".join(
            (getattr(w, "label", "") or "")
            for grupo in (at.button, at.radio, at.slider, at.selectbox, at.number_input)
            for w in grupo
            if id(w) not in laterais
        ).lower()
        titulos = " ".join(t.value.lower() for t in at.title)
        for termo in proibidos:
            assert termo not in rotulos, f"'{termo}' apareceu na fase de dimensionamento"
            assert termo not in titulos, f"'{termo}' no título da fase de dimensionamento"
        if at.session_state["e_passo"] == CLIENTE:
            _por_rotulo(at.text_input, "Nome do cliente").set_value("X").run()
        _continuar(at)


def test_continuar_bloqueado_explica_o_que_falta():
    """Botão desabilitado tem que dizer o motivo, não sumir."""
    at = _abrir()
    seguir = _por_rotulo(at.button, "Continuar")
    assert seguir.disabled
    assert any("Falta:" in c.value for c in at.caption)


def test_fase_1_fecha_num_dimensionamento():
    at = _ate_o_equipamento(_ate_o_consumo(_abrir()))
    assert at.session_state["e_passo"] == EQUIPAMENTO

    memoria = at.session_state["e_memoria"]
    assert memoria is not None and memoria.viavel
    assert memoria.modulos_por_string > 0 and memoria.strings_do_sistema > 0
    assert memoria.potencia_do_sistema_kwp > 0
    # A memória tem que ser conferível: fórmula, substituição e resultado.
    assert len(memoria.passos) >= 8
    for passo in memoria.passos:
        assert passo.formula and passo.substituicao and passo.resultado

    # E o botão de saída da fase 1 anuncia a fase 2.
    assert any("Analisar com bateria" in b.label for b in at.button)


def test_a_ponte_entre_as_fases_leva_ao_backup():
    at = _ate_o_equipamento(_ate_o_consumo(_abrir()))
    _por_rotulo(at.button, "Analisar com bateria").click().run()
    assert not at.exception, at.exception
    assert at.session_state["e_passo"] == BACKUP


# ----------------------------------------------------------------------------
# A análise do consumo
# ----------------------------------------------------------------------------
def test_analise_do_consumo_traz_o_que_o_d2_traz():
    at = _ate_o_consumo(_abrir())
    analise = at.session_state["e_analise"]
    assert analise is not None

    # Estatística dos picos, com a incerteza do próprio Monte Carlo.
    assert analise.estatisticas.percentis[95] > 0
    assert analise.estatisticas.erro_padrao_p95 > 0
    assert 0 < analise.estatisticas.coeficiente_variacao < 2

    # Indicadores de energia e os três fatores.
    indicadores = analise.indicadores
    assert indicadores.consumo_diario_kwh > 0
    assert 0 < indicadores.fator_de_carga <= 1
    assert 0 < indicadores.fator_de_demanda <= 1
    assert 0 < indicadores.fator_de_coincidencia <= 1

    # Composição do pico: o que o D² calculava e não levava ao relatório.
    composicao = analise.composicao
    assert composicao is not None
    assert not composicao.por_equipamento.empty
    assert 0 <= composicao.hora_mais_provavel <= 23
    assert composicao.por_equipamento["participacao"].max() > 0

    # As quatro estações em separado.
    assert len(analise.por_estacao) == 4


def test_analise_agrupa_as_instancias_do_comodo():
    """40 apartamentos são um cômodo repetido, não 40 linhas de tabela."""
    at = _ate_o_consumo(_abrir())
    composicao = at.session_state["e_analise"].composicao
    nomes = set(composicao.por_comodo["comodo"])
    assert not any("." in n and n.rsplit(".", 1)[-1].isdigit() for n in nomes), nomes
    assert len(nomes) <= 6


def test_analise_refeita_quando_as_cargas_mudam():
    at = _ate_o_consumo(_abrir())
    assert at.session_state["e_assinatura_analise"] is not None

    at.session_state["e_passo"] = CARGAS
    at.run()
    cenario = at.session_state["e_cenario"]
    cenario.adicionar_equipamento(next(iter(cenario.comodos)), "Chuveiro elétrico 7.500 W", 5)
    _continuar(at)
    assert at.session_state["e_passo"] == CONSUMO
    assert any("mudaram desde a última análise" in i.value for i in at.info)


# ----------------------------------------------------------------------------
# Escolha de equipamento
# ----------------------------------------------------------------------------
def test_troca_de_inversor_muda_a_memoria():
    at = _ate_o_equipamento(_ate_o_consumo(_abrir()))
    primeira = at.session_state["e_memoria"]

    # O AppTest expõe `options` já formatadas em texto e `value` como o objeto:
    # comparar os dois diretamente escolhe sempre a primeira opção, que é a
    # atual, e o teste passaria a não testar nada.
    seletor = _por_rotulo(at.selectbox, "Inversor de rede")
    atual = str(seletor.value)
    outro = next(o for o in seletor.options if o != atual)
    seletor.set_value(outro).run()
    assert not at.exception, at.exception

    segunda = at.session_state["e_memoria"]
    assert segunda.inversor.modelo != primeira.inversor.modelo
    assert segunda.passos, "a memória tem que ser recalculada para o novo inversor"


def test_memoria_respeita_o_limite_do_telhado(monkeypatch):
    """Com telhado marcado, o arranjo não pode usar mais módulos do que cabem."""
    from aurum.pv.equipment import carregar_base
    from aurum.pv.memoria import memoria_do_arranjo

    base = carregar_base()
    modulo = base.modulo_por_modelo("LR7-72HGD-620M")
    for inversor in base.inversores_ordenados():
        memoria = memoria_do_arranjo(modulo, inversor, modulos_disponiveis=288)
        assert memoria.modulos_do_sistema <= 288, inversor.modelo


# ----------------------------------------------------------------------------
# Caminhos alternativos
# ----------------------------------------------------------------------------
def test_caminho_da_conta_de_luz_avisa_o_que_perde():
    at = _abrir()
    _por_rotulo(at.text_input, "Nome do cliente").set_value("Padaria Teste").run()
    _continuar(at)

    at.radio[0].set_value("conta").run()
    assert not at.exception, at.exception
    assert any("variabilidade" in w.value.lower() for w in at.warning), \
        "o caminho sem levantamento tem que declarar o que se perde"

    _continuar(at)
    assert at.session_state["e_passo"] == CONSUMO
    ensemble = at.session_state["e_ensemble_total"]
    assert ensemble.metadados["origem"] == "curva_tipica_calibrada"
    # A dispersão assumida tem que produzir uma cauda, não uma curva única.
    assert ensemble.picos_diarios_w().std() > 0


def test_erro_de_carga_bloqueia_e_aponta_a_linha():
    import numpy as np

    at = _abrir()
    _por_rotulo(at.text_input, "Nome do cliente").set_value("Hotel Teste").run()
    _continuar(at)

    cenario = at.session_state["e_cenario"]
    primeiro = next(iter(cenario.comodos))
    tabela = cenario.comodos[primeiro].copy()
    tabela.loc[len(tabela)] = [
        "Bomba sem duração", 750, 1, "dinâmico", "08:00 as 18:00",
        1.0, 1.0, np.nan, np.nan, np.nan,
    ]
    cenario.comodos[primeiro] = tabela
    at.run()
    assert not at.exception, at.exception

    assert _por_rotulo(at.button, "Continuar").disabled, "erro de carga tem que bloquear"
    texto = " ".join(m.value for m in at.markdown)
    assert "meia-noite" in texto, "a mensagem tem que explicar o que o D² faria em silêncio"


def test_backup_menor_que_o_total():
    at = _ate_o_equipamento(_ate_o_consumo(_abrir()))
    _por_rotulo(at.button, "Analisar com bateria").click().run()
    at.slider[0].set_value(50).run()
    _por_rotulo(at.button, "Simular a demanda").click().run()
    assert not at.exception, at.exception

    total = at.session_state["e_ensemble_total"].energia_diaria_kwh().mean()
    backup = at.session_state["e_ensemble_backup"].energia_diaria_kwh().mean()
    assert 0 < backup < total


def test_resultado_sem_estudo_avisa_em_vez_de_quebrar():
    at = _abrir()
    at.session_state["e_passo"] = RESULTADO
    at.run()
    assert not at.exception, at.exception
    assert any("ainda não foi rodado" in w.value for w in at.warning)


# ----------------------------------------------------------------------------
# De ponta a ponta
# ----------------------------------------------------------------------------
@pytest.mark.rede
def _ate_a_meta(at):
    """Atravessa as duas fases até a tela onde as fontes são escolhidas."""
    at = _ate_o_equipamento(_ate_o_consumo(at), kwp=20.0)
    _por_rotulo(at.button, "Analisar com bateria").click().run()
    at.slider[0].set_value(50).run()
    _por_rotulo(at.button, "Simular a demanda").click().run()
    _continuar(at)
    assert at.session_state["e_passo"] == META
    return at


def test_a_meta_pergunta_as_fontes_menos_solar():
    """
    Bateria e gerador ficam na Meta; solar não.

    Solar é perguntada no passo 1, onde a resposta ainda poda o fluxo.
    Repetir a pergunta no fim deixaria o usuário desmarcar depois de já ter
    marcado o telhado no mapa e escolhido o inversor.
    """
    at = _ate_a_meta(_abrir())
    rotulos = [c.label for c in at.checkbox]
    for fonte in ("Bateria", "Grupo gerador"):
        assert any(fonte in r for r in rotulos), f"falta a fonte '{fonte}': {rotulos}"
    assert not any("Energia solar" in r for r in rotulos), (
        "solar não pode ser perguntada de novo no fim do fluxo"
    )
    # E a conta de luz, que é o referencial de toda a comparação.
    _por_rotulo(at.number_input, "Tarifa (R$/kWh)")
    _por_rotulo(at.number_input, "Consumo médio na fatura")
    _por_rotulo(at.selectbox, "Custo de disponibilidade")


def test_gerador_so_pede_os_dados_dele_quando_e_marcado():
    """Campo que não vai ser usado é ruído: o grupo aparece sob demanda."""
    at = _ate_a_meta(_abrir())
    rotulos = [n.label for n in at.number_input]
    assert not any("Potência do grupo" in r for r in rotulos)

    _por_rotulo(at.checkbox, "Grupo gerador").set_value(True).run()
    assert not at.exception, at.exception
    _por_rotulo(at.number_input, "Potência do grupo")
    _por_rotulo(at.number_input, "Custo do kWh gerado")
    _por_rotulo(at.number_input, "Investimento no grupo")


def test_sem_solar_os_passos_de_solar_somem():
    """
    O exemplo que motivou a mudança.

    Perguntar "tem solar?" no fim e mesmo assim obrigar a marcar o telhado no
    mapa e escolher inversor era o fluxo às avessas: o usuário atravessava dois
    passos de dimensionamento para depois dizer que o cliente não vai instalar
    painel nenhum.
    """
    at = _abrir()
    _por_rotulo(at.text_input, "Nome do cliente").set_value("Sem Sol").run()
    _por_rotulo(at.radio, "Tem solar").set_value(False).run()
    assert not at.exception, at.exception
    assert at.session_state["e_com_solar"] is False

    # O botão do passo do consumo passa a atravessar direto para a fase 2.
    at = _ate_o_consumo(at)
    assert at.session_state["e_passo"] == CONSUMO
    # O rótulo muda sozinho: o consumo virou o último passo da fase 1.
    _por_rotulo(at.button, "Analisar com bateria").click().run()
    assert not at.exception, at.exception
    assert at.session_state["e_passo"] == BACKUP, "telhado e equipamento têm de sumir"

    # E voltar não pode cair num passo que não existe mais.
    _por_rotulo(at.button, "Voltar").click().run()
    assert at.session_state["e_passo"] == CONSUMO


def test_desligar_solar_descarta_o_que_ja_foi_dimensionado():
    """
    Deixar o telhado para trás produziria um relatório com croqui e sem
    seção de solar — dois documentos no mesmo PDF.
    """
    at = _ate_o_equipamento(_ate_o_consumo(_abrir()), kwp=20.0)
    assert at.session_state["e_kwp"] > 0, "a fase 1 fechou num sistema dimensionado"

    _por_rotulo(at.sidebar.button, "Cliente").click().run()
    _por_rotulo(at.radio, "Tem solar").set_value(False).run()
    assert not at.exception, at.exception
    assert at.session_state["e_kwp"] == 0.0
    assert at.session_state["e_telhado"] is None


def test_tirar_um_ambiente_do_backup_e_definitivo():
    """
    Desmarcar tinha de ser feito duas vezes, e às vezes voltava tudo.

    O widget usava `default=`, que reimpõe o padrão a cada re-execução. Como
    lista vazia é falsa em Python, `[] or cenario.essenciais or todos` caía no
    terceiro termo: tirar o último ambiente devolvia a lista inteira. Com
    `key=`, o estado é do widget e a escolha do usuário manda.
    """
    at = _ate_o_consumo(_abrir())
    _continuar(at)                       # consumo -> telhado
    at.radio[0].set_value("Já sei a potência").run()
    _por_rotulo(at.number_input, "Potência do sistema").set_value(20.0).run()
    _continuar(at)                       # telhado -> equipamento
    _por_rotulo(at.button, "Analisar com bateria").click().run()
    assert at.session_state["e_passo"] == BACKUP

    ambientes = at.multiselect[0]
    assert len(ambientes.value) > 1, "o modelo precisa vir com mais de um ambiente"

    # Tira todos menos um, e depois o último.
    ambientes.set_value([ambientes.value[0]]).run()
    assert not at.exception, at.exception
    assert len(at.session_state["e_essenciais"]) == 1

    at.multiselect[0].set_value([]).run()
    assert not at.exception, at.exception
    assert at.session_state["e_essenciais"] == [], "o vazio tem de permanecer vazio"
    # E a tela diz o que falta, em vez de repovoar a lista sozinha.
    assert any("ao menos um ambiente" in str(c.value).lower() for c in at.caption)


def test_custo_da_interrupcao_nasce_sugerido_e_nao_zerado():
    """
    Zero afirma que ficar sem energia não custa nada.

    Com o campo nascendo em zero, o valor da resiliência saía zero em todo
    estudo e a bateria aparecia sempre como investimento sem retorno — o que
    é uma afirmação financeira, não uma ausência de dado.
    """
    from aurum.demanda.biblioteca import MODELOS

    at = _ate_a_meta(_abrir())
    campo = _por_rotulo(at.number_input, "Quanto custa ao cliente 1 kWh que faltou")
    esperado = MODELOS[at.session_state["e_segmento"]].custo_interrupcao_brl_kwh
    assert campo.value == pytest.approx(esperado)
    assert campo.value > 0


def test_resiliencia_vira_dinheiro_no_estudo():
    """O caminho todo: sugestão na tela -> premissa -> valor no quadro."""
    at = _ate_a_meta(_abrir())
    _por_rotulo(at.select_slider, "Autonomia alvo").set_value(1).run()
    _por_rotulo(at.multiselect, "Durações de falta").set_value([1.0]).run()
    _por_rotulo(at.slider, "Amostras por combinação").set_value(20).run()
    _por_rotulo(at.slider, "Conjuntos a avaliar").set_value(2).run()
    _por_rotulo(at.button, "Rodar o estudo").click().run()
    assert not at.exception, at.exception

    estudo = at.session_state["e_estudo"]
    assert estudo.configuracao.premissas.custo_interrupcao_brl_kwh > 0
    com_bateria = estudo.cenarios.por_chave("solar+bateria")
    assert com_bateria.valor_resiliencia_brl_ano > 0, (
        "a bateria evita energia não suprida; isso tem de virar dinheiro"
    )


def test_a_meta_pergunta_o_padrao_da_obra():
    """
    O R$/kWp é a premissa mais sensível do payback, e ela tem de ser escolha.

    Entre obra básica e alto padrão a diferença passa de 70% no investimento, e
    atravessa inteira para o payback. Deixar isso fixo num número médio faria o
    software dar a resposta certa só para o cliente médio, que não existe.
    """
    at = _ate_a_meta(_abrir())
    seletor = _por_rotulo(at.selectbox, "Padrão da obra")
    assert any("Alto padrão" in str(o) for o in seletor.options), seletor.options
    _por_rotulo(at.number_input, "Ou informe o investimento no solar")


def test_padrao_da_obra_chega_ao_estudo():
    """De nada adianta o seletor existir se a configuração ignora a escolha."""
    at = _ate_a_meta(_abrir())
    seletor = _por_rotulo(at.selectbox, "Padrão da obra")
    alto = next(o for o in seletor.options if "Alto padrão" in str(o))
    seletor.set_value(alto).run()
    assert not at.exception, at.exception

    _por_rotulo(at.select_slider, "Autonomia alvo").set_value(1).run()
    _por_rotulo(at.multiselect, "Durações de falta").set_value([1.0]).run()
    _por_rotulo(at.slider, "Amostras por combinação").set_value(20).run()
    _por_rotulo(at.slider, "Conjuntos a avaliar").set_value(2).run()
    _por_rotulo(at.button, "Rodar o estudo").click().run()
    assert not at.exception, at.exception

    estudo = at.session_state["e_estudo"]
    assert estudo.configuracao.padrao_capex == "alto"
    # E o preço estimado tem de refletir a escolha, não a curva média.
    from aurum.pv.financials import estimar_capex

    solar = estudo.cenarios.por_chave("solar")
    assert solar.capex_brl == pytest.approx(
        estimar_capex(estudo.potencia_fv_kwp, padrao="alto"), rel=1e-6
    )


def test_estudo_completo_pela_interface():
    """As duas fases inteiras. `rede` porque o passo do telhado consulta o PVGIS."""
    at = _ate_o_equipamento(_ate_o_consumo(_abrir()), kwp=20.0)
    _por_rotulo(at.button, "Analisar com bateria").click().run()

    at.slider[0].set_value(50).run()
    _por_rotulo(at.button, "Simular a demanda").click().run()
    _continuar(at)
    assert at.session_state["e_passo"] == META

    _por_rotulo(at.select_slider, "Autonomia alvo").set_value(3).run()
    _por_rotulo(at.multiselect, "Durações de falta").set_value([1.0, 3.0]).run()
    _por_rotulo(at.slider, "Amostras por combinação").set_value(20).run()
    _por_rotulo(at.slider, "Conjuntos a avaliar").set_value(2).run()
    assert not at.exception, at.exception

    _por_rotulo(at.button, "Rodar o estudo").click().run()
    assert not at.exception, at.exception
    assert at.session_state["e_passo"] == RESULTADO

    estudo = at.session_state["e_estudo"]
    assert estudo is not None and len(estudo.resiliencia) == 2
    # A conclusão vem antes dos dados: sucesso ou erro, mas nunca só tabela.
    assert at.success or at.error

    # O quadro de cenários sai do mesmo estudo, com o referencial "só a rede"
    # e um arranjo por combinação de fontes marcada.
    comparacao = estudo.cenarios
    assert comparacao is not None, "o estudo tem bateria: os cenários deviam existir"
    chaves = {c.composicao.chave for c in comparacao.cenarios}
    assert "rede" in chaves and "solar+bateria" in chaves
    assert comparacao.base.capex_brl == 0.0
    so_solar = comparacao.por_chave("solar")
    assert so_solar.autonomia_garantida_h == 0.0, "solar sem banco não ilha"

    _por_rotulo(at.button, "Gerar dossiê").click().run()
    assert not at.exception, at.exception
    pacote = at.session_state["e_pacote"]
    assert pacote and pacote[:2] == b"PK"

    with zipfile.ZipFile(io.BytesIO(pacote)) as zf:
        nomes = zf.namelist()
    assert any(n.endswith("estudo.tex") for n in nomes)
    assert any(n.endswith("resumo.md") for n in nomes)
    assert any("/figuras/" in n and n.endswith(".png") for n in nomes)
    assert any("/tabelas/" in n and n.endswith(".csv") for n in nomes)
    assert "LEIA-ME.txt" in nomes


# ----------------------------------------------------------------------------
# A primeira tela recebe a vistoria
# ----------------------------------------------------------------------------
def test_a_vistoria_vem_antes_dos_campos_que_ela_preenche():
    """
    Ordem de desenho, e não de gosto: importar escreve `e_nome`.

    O Streamlit proíbe escrever numa chave já instanciada como widget no mesmo
    run. Com o campo de nome desenhado antes do bloco da vistoria, importar
    derrubava a página inteira com StreamlitWidgetAlreadyInstantiatedError —
    e o erro só aparecia depois do upload, em produção.
    """
    fonte = (RAIZ / "aurum" / "pagina_estudo.py").read_text(encoding="utf-8")
    corpo = fonte[fonte.index("def _passo_cliente("):]
    corpo = corpo[: corpo.index("\ndef ")]
    assert corpo.index("_vistoria_na_primeira_tela()") < corpo.index('key="e_nome"'), (
        "a vistoria preenche e_nome; desenhar o campo antes torna a escrita ilegal")


def test_o_primeiro_passo_oferece_a_vistoria_e_abre_em_residencia():
    """
    As duas coisas que a primeira tela precisa oferecer de saída.

    Residência é o segmento em que o estudo de bateria mais é pedido, e a
    vistoria é a origem mais completa que o software aceita — pedir os dados à
    mão primeiro inverte a ordem da confiança.
    """
    at = _abrir()
    assert not at.exception, at.exception

    rotulos = " | ".join(str(getattr(e, "label", "") or "") for e in at.expander)
    assert "vistoria" in rotulos.lower(), f"expansor da vistoria ausente: {rotulos}"

    tipo = _por_rotulo(at.selectbox, "Tipo")
    assert tipo.value == "residencia", f"o padrão devia ser residência, veio {tipo.value}"
