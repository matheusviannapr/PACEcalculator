"""
Estudo de energia: cargas → solar → bateria, num fluxo só.

O desenho da experiência segue quatro regras, e elas explicam quase todas as
decisões deste arquivo:

1. **Uma pergunta por tela, em português.** O cabeçalho de cada passo é a
   pergunta que o usuário faria a um engenheiro — "o que não pode faltar
   quando a luz cai?" —, não o nome do parâmetro que a pergunta preenche.
2. **Nada de folha em branco.** O passo das cargas já vem montado a partir do
   segmento escolhido. Corrigir um rascunho é muito mais barato que partir do
   zero, e é o único jeito de alguém terminar o levantamento na primeira vez.
3. **Padrão em tudo.** Dá para atravessar o fluxo inteiro só apertando
   "Continuar" e ainda assim sair com um estudo defensável. O que exige
   decisão fica visível; o resto vai para "Ajustes avançados", fechado.
4. **Nunca um beco sem saída.** Quando falta alguma coisa, o botão de avançar
   não some: ele diz o que falta. Quando o estudo não encontra solução, a tela
   diz por quê e o que dá para fazer.

A barra lateral carrega um resumo do que já foi decidido, que vai crescendo.
É o que responde "onde eu estou?" sem o usuário ter que voltar telas.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import streamlit as st

from .bateria.apagao import MalhaApagao
from .bateria.catalogo import CatalogoError, carregar_catalogo, criar_planilha_modelo
from .bateria.documento import DadosCapa, zip_do_dossie
from .bateria.economia import PremissasBateria
from .bateria.estudo import ConfiguracaoEstudo, executar_estudo
from .bateria.excedencia import avaliar_inversores, potencia_para_excedencia
from .bateria.fontes import Fatura, Gerador
from .config import get_settings
from .demanda import simular_ensemble
from .demanda.analise import analisar
from .demanda.biblioteca import (
    CATEGORIAS,
    EQUIPAMENTOS,
    MODELOS,
    calibrar_por_conta,
    comparar_com_perfil,
    segmento_do_perfil,
)
from .demanda.cenario import MODOS_FIXOS, TIPOS_INTERVALO, Cenario
from .demanda.ensemble import ensemble_de_curva_tipica
from .proposal.render import encontrar_compilador

# O fluxo tem duas fases, e a fronteira entre elas é real, não decorativa: a
# primeira dimensiona o sistema fotovoltaico e termina num projeto fechado, com
# equipamento escolhido e memória de cálculo. Só depois disso a segunda pergunta
# o que acontece quando a rede cai. Misturar as duas era o defeito do fluxo
# anterior — quem só queria dimensionar tinha de atravessar perguntas sobre
# apagão, e quem queria a análise de bateria não via o dimensionamento fechar.
FASES: tuple[tuple[str, str], ...] = (
    ("Dimensionamento", "Do levantamento de cargas ao sistema fotovoltaico fechado"),
    ("Análise com bateria", "O que acontece quando a rede cai, e quanto isso vale"),
)

#: (fase, título curto, pergunta da tela, por que o passo existe)
PASSOS: tuple[tuple[int, str, str, str], ...] = (
    (0, "Cliente", "Para quem é o estudo, e onde fica?",
     "O local define o sol disponível; o tipo de instalação define o ponto de partida das cargas."),
    (0, "Cargas", "O que consome energia nesse lugar?",
     "Comece de um modelo, da sua planilha, ou monte equipamento por equipamento."),
    (0, "Consumo", "Como esse lugar consome energia?",
     "A análise completa da demanda: distribuição dos picos, quem os causa, e quando."),
    (0, "Telhado", "Onde os módulos vão ficar?",
     "Marque a cobertura no mapa; daí saem a área, a orientação e quantos módulos cabem."),
    (0, "Equipamento", "Que módulo e que inversor?",
     "Escolha do catálogo e a memória de cálculo do arranjo, string por string."),
    (1, "Backup", "O que não pode faltar quando a luz cai?",
     "É a escolha que mais muda o preço: alimentar o prédio inteiro em ilha custa três vezes mais."),
    (1, "Meta", "Quanto tempo sem luz precisa aguentar?",
     "A partir daqui é só cálculo: a varredura testa cada duração começando em cada hora do ano."),
    (1, "Resultado", "O que comprar",
     "A resposta, e tudo que a sustenta."),
)

#: Índice (1-based) do primeiro passo da segunda fase.
PRIMEIRO_PASSO_ANALISE = next(i for i, p in enumerate(PASSOS, start=1) if p[0] == 1)

_PADROES: dict[str, Any] = {
    "e_passo": 1,
    "e_nome": "",
    "e_segmento": "residencia",
    "e_com_solar": True,
    "e_tensao_rede": 380.0,
    "e_lat": -25.4284,
    "e_lon": -49.2733,
    "e_endereco": "",
    #: Candidatos da última busca de endereço, à espera de escolha.
    "e_lugares": [],
    "e_modo_carga": "modelo",
    "e_cenario": None,
    "e_vistoria": None,
    "e_conta_kwh": 5000.0,
    "e_dias_operacao": 30,
    "e_ensemble_total": None,
    "e_ensemble_backup": None,
    "e_essenciais": [],
    "e_essenciais_sel": None,
    "e_opcoes_essenciais": None,
    "e_kwp": 0.0,
    "e_estudo": None,
    "e_pacote": None,
    #: Figuras do relatório desenhadas para a tela, e de qual estudo vieram.
    "e_figuras": None,
    "e_figuras_de": None,
    "e_avisos_carga": [],
    "e_assinatura_carga": None,
    "e_assinatura_analise": None,
    "e_analise": None,
    #: Qual dos estudos de ocupação foi escolhido, ou "" para usar as janelas
    #: da vistoria como elas vieram.
    "e_estudo_ocupacao": "",
    #: O que a escolha produziu: tabela dos perfis, curvas, dimensionante e o
    #: consumo diário ponderado. Vai inteiro para o dossiê.
    "e_ocupacao": None,
    #: O cenário reescrito pelo perfil que dimensiona. O de `e_cenario`
    #: continua sendo o levantado em campo — é o dado, e o anexo o mostra.
    "e_cenario_ocupacao": None,
    "e_modulo": None,
    "e_inversor": None,
    "e_memoria": None,
    "e_telhado": None,
    "e_layout": None,
    #: As águas marcadas, na ordem em que foram desenhadas. Cada uma é um
    #: dicionário com nome, telhado, arranjo automático, arranjo editado,
    #: croqui e edição — tudo que a água precisa para ser editada sozinha.
    "e_aguas": [],
    #: Qual delas está espelhada nas chaves acima e recebe os cliques do mapa.
    "e_agua_ativa": 0,
    #: Produtividade e geração por água, quando calculadas.
    "e_geracao_aguas": None,
    "e_croqui": None,
    "e_edicao": None,
    "e_layout_base": None,
    "e_modo_edicao": "Remover",
    "e_azimute_fileiras": None,
    "e_inclinacao": 0.0,
    "e_azimute": 0.0,
    "e_exigir_pvgis": False,
    #: Coluna da tabela de kit; vazio volta para a curva de R$/kWp.
    "e_topologia_kit": "",
    #: O bloco de expansão de bateria: quanto de energia e quanto custa.
    "e_bateria_bloco_kwh": None,
    "e_bateria_bloco_brl": None,
    #: As duas parcelas que a obra soma ao kit, em R$/kWp.
    "e_mao_de_obra": None,
    "e_material_ca": None,
}


# ============================================================================
# Infraestrutura da página
# ============================================================================
def _iniciar() -> None:
    for chave, valor in _PADROES.items():
        st.session_state.setdefault(chave, valor)


#: Passos que só fazem sentido num estudo com energia solar.
PASSOS_DE_SOL = (4, 5)  # Telhado, Equipamento


def _passos_ativos() -> list[int]:
    """
    Os passos deste estudo, na ordem — sem os que não se aplicam.

    Perguntar "tem solar?" no fim e mesmo assim obrigar a marcar o telhado no
    mapa e escolher inversor era o fluxo às avessas. A resposta vem no passo 1
    e poda o que vem depois.
    """
    if st.session_state.get("e_com_solar", True):
        return list(range(1, len(PASSOS) + 1))
    return [i for i in range(1, len(PASSOS) + 1) if i not in PASSOS_DE_SOL]


def _ir(passo: int) -> None:
    """Vai para o passo pedido, ou para o ativo mais próximo depois dele."""
    ativos = _passos_ativos()
    alvo = max(1, min(len(PASSOS), passo))
    if alvo not in ativos:
        # Pulou para um passo podado: segue na direção em que estava indo.
        adiante = [i for i in ativos if i > alvo]
        atras = [i for i in ativos if i < alvo]
        indo_para_frente = alvo > st.session_state.get("e_passo", 1)
        if indo_para_frente:
            alvo = adiante[0] if adiante else (atras[-1] if atras else 1)
        else:
            alvo = atras[-1] if atras else (adiante[0] if adiante else 1)
    st.session_state["e_passo"] = alvo
    st.rerun()


def _milhar(valor: float, casas: int = 0) -> str:
    """
    Número com ponto de milhar, sem tocar no resto da frase.

    Aplicar ``.replace(",", ".")`` na frase inteira é o atalho óbvio e come as
    vírgulas do texto: "1.003 m², somando" virava "1.003 m². somando".
    """
    return f"{valor:,.{casas}f}".replace(",", ".")


def _cartao(coluna, rotulo: str, valor: str) -> None:
    coluna.markdown(
        f'<div class="bloco-metrica"><div class="bloco-valor">{valor}</div>'
        f'<div class="bloco-rotulo">{rotulo}</div></div>',
        unsafe_allow_html=True,
    )


def _cabecalho(indice: int) -> None:
    fase, titulo, pergunta, porque = PASSOS[indice - 1]
    nome_fase, subtitulo_fase = FASES[fase]
    ativos = _passos_ativos()
    da_fase = [i for i, p in enumerate(PASSOS, start=1) if p[0] == fase and i in ativos]
    posicao = da_fase.index(indice) + 1
    st.progress(
        posicao / len(da_fase),
        text=f"Fase {fase + 1} · {nome_fase} — passo {posicao} de {len(da_fase)}: {titulo}",
    )
    st.title(pergunta)
    st.caption(porque)


def _rodape(
    pendencia: str | None = None,
    rotulo_avancar: str = "Continuar →",
    ao_avancar: Callable[[], bool] | None = None,
) -> None:
    """
    Navegação no mesmo lugar em todas as telas.

    ``pendencia`` não esconde o botão: mostra o motivo pelo qual ele não vai
    funcionar ainda. Botão que some deixa o usuário procurando; botão que
    explica encerra a dúvida.
    """
    st.divider()
    passo = st.session_state["e_passo"]
    # Sem solar, o passo do consumo é o último da fase 1 — e o botão precisa
    # dizer isso, senão o usuário atravessa a fronteira sem perceber.
    ativos = _passos_ativos()
    seguintes = [i for i in ativos if i > passo]
    if (
        rotulo_avancar == "Continuar →"
        and seguintes
        and PASSOS[seguintes[0] - 1][0] != PASSOS[passo - 1][0]
    ):
        rotulo_avancar = "Analisar com bateria →"
    esquerda, direita = st.columns([1, 2])
    if passo > 1 and esquerda.button("← Voltar", width="stretch", key=f"voltar{passo}"):
        _ir(passo - 1)
    if pendencia:
        direita.button(rotulo_avancar, width="stretch", disabled=True, key=f"seguir{passo}")
        direita.caption(f"Falta: {pendencia}")
        return
    if direita.button(rotulo_avancar, type="primary", width="stretch", key=f"seguir{passo}"):
        if ao_avancar is None or ao_avancar():
            _ir(passo + 1)


@st.cache_resource(show_spinner=False)
def _catalogo(caminho: str | None = None):
    return carregar_catalogo(caminho)


# ============================================================================
# Barra lateral: o que já sei
# ============================================================================
def _barra_lateral() -> None:
    passo_atual = st.session_state["e_passo"]
    ativos = _passos_ativos()
    fase_anterior = -1
    for indice, (fase, titulo, _, _) in enumerate(PASSOS, start=1):
        if indice not in ativos:
            continue
        if fase != fase_anterior:
            st.sidebar.markdown(f"**Fase {fase + 1} · {FASES[fase][0]}**")
            fase_anterior = fase
        icone = "✅" if indice < passo_atual else ("▶️" if indice == passo_atual else "⚪")
        if st.sidebar.button(f"{icone} {titulo}", key=f"nav{indice}", width="stretch"):
            if indice <= passo_atual:
                _ir(indice)

    st.sidebar.divider()
    st.sidebar.markdown("**O que já sei**")
    linhas: list[str] = []
    estado = st.session_state
    if estado["e_nome"]:
        linhas.append(f"Cliente · **{estado['e_nome']}**")
    if estado["e_segmento"]:
        linhas.append(f"Tipo · {MODELOS[estado['e_segmento']].nome}")
    if estado["e_endereco"] or estado["e_lat"]:
        linhas.append(f"Local · {estado['e_lat']:.3f}, {estado['e_lon']:.3f}")
    cenario: Cenario | None = estado["e_cenario"]
    if cenario is not None:
        linhas.append(f"Cargas · {cenario.total_de_equipamentos()} equipamentos, "
                      f"{cenario.potencia_instalada_w() / 1000:.0f} kW instalados")
    if estado["e_ensemble_backup"] is not None:
        geral = estado["e_ensemble_backup"].resumo()["geral"]
        linhas.append(f"Backup · pico P95 {geral['pico_p95_kw']:.1f} kW, "
                      f"{geral['energia_diaria_media_kwh']:.0f} kWh/dia")
    if estado.get("e_telhado") is not None:
        telhado = estado["e_telhado"]
        linhas.append(f"Telhado · {_milhar(telhado.area_m2)} m², face {telhado.orientacao}")
    if estado["e_kwp"]:
        linhas.append(f"Solar · {estado['e_kwp']:.1f} kWp")
    st.sidebar.caption("\n\n".join(linhas) if linhas else "Nada ainda — comece pelo passo 1.")

    st.sidebar.divider()
    try:
        base = _catalogo()
        resumo = base.resumo()
        st.sidebar.caption(
            f"Catálogo de armazenamento: {resumo['baterias']} baterias, "
            f"{resumo['inversores']} inversores híbridos"
        )
        if resumo["a_conferir"]:
            st.sidebar.caption(
                f"⚠️ {resumo['a_conferir']} itens marcados *a conferir* em "
                f"`{base.caminho.name}` — são ordens de grandeza de mercado, não datasheets."
            )
    except CatalogoError as exc:
        st.sidebar.error(f"Catálogo indisponível: {exc}")
        if st.sidebar.button("Criar planilha modelo", width="stretch"):
            criar_planilha_modelo()
            _catalogo.clear()
            st.rerun()

    if st.sidebar.button("Recomeçar do zero", width="stretch"):
        for chave, valor in _PADROES.items():
            st.session_state[chave] = valor
        st.rerun()


# ============================================================================
# Passo 1 — cliente e local
# ============================================================================
def _resumo_do_catalogo(tensao_rede_v: float) -> None:
    """
    Diz quantos inversores sobram nessa rede, e de que faixa.

    Serve de conferência imediata: se o número for pequeno demais para a carga
    do cliente, é melhor descobrir agora do que no passo do resultado, depois
    de rodar a simulação inteira.
    """
    try:
        base = _catalogo()
    except CatalogoError:
        return
    atendem = [i for i in base.inversores if i.atende_rede(tensao_rede_v)]
    if not atendem:
        st.warning(
            f"Nenhum inversor do catálogo atende {tensao_rede_v:.0f} V. O estudo vai "
            "rodar com o catálogo inteiro e avisar — a solução exige transformador ou "
            "cadastrar o equipamento certo na planilha."
        )
        return
    menor = min(i.potencia_ca_nominal_kw for i in atendem)
    maior = max(i.potencia_ca_nominal_kw for i in atendem)
    st.caption(
        f"**{len(atendem)}** de {len(base.inversores)} inversores do catálogo atendem "
        f"essa rede, de {menor:g} a {maior:g} kW."
    )


def _vistoria_na_primeira_tela() -> None:
    """
    O atalho: quem tem o backup da vistoria não precisa responder nada.

    O arquivo traz cliente, cidade, tensão da rede e a criticidade de cada
    equipamento. Pedir isso à mão antes, e só aceitar o arquivo no passo
    seguinte, invertia a ordem da confiança: a resposta digitada vencia a
    medida em campo por chegar primeiro. Aqui ela chega antes, e os campos
    abaixo já nascem preenchidos.

    Continua sendo opcional, e continua disponível no passo de cargas — quem
    não tem vistoria não é obrigado a passar por aqui.
    """
    if st.session_state.get("e_vistoria") is not None:
        vistoria = st.session_state["e_vistoria"]
        st.success(
            f"✅ Vistoria de **{vistoria.cliente}** carregada — "
            f"{vistoria.cenario.total_de_equipamentos()} equipamentos, rede de "
            f"{vistoria.tensao_rede_v:.0f} V, {vistoria.local or 'local não informado'}. "
            "Os campos abaixo já vieram dela."
        )
        return

    with st.expander("Tenho o backup de uma vistoria técnica (.json)"):
        st.caption(
            "É a origem mais completa que este software aceita: traz o cliente, a "
            "cidade, a tensão da rede e a criticidade **de cada equipamento** — que "
            "é o que monta o quadro de backup. Carregando aqui, o resto da tela já "
            "vem preenchido."
        )
        _cargas_da_vistoria()


def _buscar_endereco(consulta: str, limite: int = 5) -> list[dict]:
    """
    Os lugares que o geocodificador achou, para o usuário escolher.

    Devolve lista de ``{"nome", "lat", "lon"}``. Uma lista, e não o primeiro
    resultado: "Rua São João" existe em quinze cidades, e escolher sozinho
    põe o estudo inteiro na cidade errada sem avisar ninguém.
    """
    from .geo.nominatim import GeocodingError, NominatimClient

    try:
        lugares = NominatimClient(get_settings()).search(
            consulta, limit=limite, with_geometry=False)
    except GeocodingError as exc:
        st.error(f"Não achei esse endereço: {exc}")
        return []
    if not lugares:
        st.warning(
            f"Nenhum lugar encontrado para “{consulta}”. Tente incluir a cidade "
            "e o estado, ou digite as coordenadas na outra aba."
        )
    return [
        {"nome": lugar.display_name, "lat": float(lugar.lat), "lon": float(lugar.lon)}
        for lugar in lugares
    ]


def _fixar_local(lugar: dict) -> None:
    """
    Coordenada e rótulo andam juntos — é o que faltava na importação.

    Os campos da aba de coordenadas também são atualizados: eles são widgets
    com chave própria, e deixá-los para trás faria a busca por endereço mover
    o mapa enquanto os números na tela continuavam mostrando o lugar anterior.
    """
    st.session_state.update({
        "e_lat": float(lugar["lat"]),
        "e_lon": float(lugar["lon"]),
        "e_endereco": lugar["nome"],
        "e_lugares": [],
    })
    for chave, valor in (("e_lat_digitada", lugar["lat"]),
                         ("e_lon_digitada", lugar["lon"])):
        if chave in st.session_state:
            st.session_state[chave] = float(valor)


def _passo_cliente() -> None:
    # A vistoria vem **antes** de qualquer widget desta tela, e não é questão
    # de gosto: importar preenche `e_nome`, `e_tensao_rede` e as coordenadas,
    # e o Streamlit proíbe escrever numa chave já instanciada como widget no
    # mesmo run. Com o campo de nome desenhado primeiro, importar derrubava a
    # página com StreamlitWidgetAlreadyInstantiatedError.
    #
    # A ordem também é a que faz sentido ler: carregue o arquivo, e o resto da
    # tela já nasce respondido.
    _vistoria_na_primeira_tela()

    # `key=` em vez de atribuir o retorno: a barra lateral é desenhada antes
    # do corpo da página e leria o valor anterior, ficando um rerun atrás do
    # que o usuário acabou de digitar. Com a chave, o Streamlit guarda o valor
    # no estado antes de qualquer widget rodar.
    st.text_input(
        "Nome do cliente ou do local",
        key="e_nome",
        placeholder="Ex.: Hotel Central",
    )

    st.markdown("##### Que tipo de instalação é?")
    st.caption(
        "Serve para duas coisas: montar o rascunho das cargas e escolher a curva de "
        "consumo de referência com que o resultado será conferido depois."
    )
    ids = list(MODELOS)
    atual = st.session_state["e_segmento"]
    escolha = st.selectbox(
        "Tipo", ids, index=ids.index(atual) if atual in ids else 0,
        format_func=lambda i: MODELOS[i].nome, label_visibility="collapsed",
    )
    if escolha != st.session_state["e_segmento"]:
        # Trocar de segmento invalida o rascunho de cargas montado do anterior.
        st.session_state["e_segmento"] = escolha
        st.session_state["e_cenario"] = None
        st.session_state["e_ensemble_backup"] = None

    modelo = MODELOS[escolha]
    st.info(
        f"**{modelo.nome}** vem com {len(modelo.comodos)} cômodos prontos: "
        + ", ".join(c.nome for c in modelo.comodos)
        + ". Você ajusta tudo no próximo passo."
    )

    st.divider()
    st.markdown("##### Este estudo tem energia solar?")
    st.caption(
        "Se não tiver, os passos de telhado e de equipamento fotovoltaico não "
        "aparecem — e o relatório não fala de geração em lugar nenhum."
    )
    com_solar = st.radio(
        "Tem solar",
        [True, False],
        index=0 if st.session_state.get("e_com_solar", True) else 1,
        horizontal=True,
        format_func=lambda v: (
            "Sim — dimensionar o sistema fotovoltaico" if v
            else "Não — só armazenamento e continuidade"
        ),
        label_visibility="collapsed",
    )
    if com_solar != st.session_state.get("e_com_solar", True):
        st.session_state["e_com_solar"] = com_solar
        if not com_solar:
            # O que foi dimensionado deixa de valer, e deixar para trás
            # produziria um relatório com telhado marcado e sem seção de solar.
            st.session_state.update({
                "e_telhado": None, "e_layout": None, "e_layout_base": None,
                "e_croqui": None, "e_edicao": None, "e_memoria": None, "e_kwp": 0.0,
            })
        st.rerun()

    st.divider()
    st.markdown("##### Qual é a tensão da rede?")
    st.caption(
        "Decide que inversores podem ser recomendados. A maior parte dos híbridos "
        "trifásicos do mercado é de 220/380 V e **não liga** numa rede de 127/220 — "
        "recomendar um deles é um erro que só aparece na entrega."
    )
    opcoes_rede = [220.0, 380.0]
    st.session_state["e_tensao_rede"] = st.radio(
        "Tensão de linha",
        opcoes_rede,
        index=opcoes_rede.index(float(st.session_state.get("e_tensao_rede") or 380.0)),
        horizontal=True,
        format_func=lambda v: (
            "127/220 V — baixa tensão, a rede de boa parte do país" if v == 220.0
            else "220/380 V — trifásico de 380 V"
        ),
        label_visibility="collapsed",
    )
    _resumo_do_catalogo(float(st.session_state["e_tensao_rede"]))

    st.divider()
    st.markdown("##### Onde fica?")
    st.caption("A geração solar do local vem daqui — série horária de anos reais do PVGIS.")

    aba_endereco, aba_coordenadas = st.tabs(["Buscar por endereço", "Digitar coordenadas"])
    with aba_endereco:
        consulta = st.text_input(
            "Endereço", value=st.session_state["e_endereco"],
            placeholder="Ex.: Avenida Sete de Setembro, Curitiba",
        )
        if st.button("Localizar", disabled=not consulta):
            st.session_state["e_lugares"] = _buscar_endereco(consulta)
            st.rerun()

        # As alternativas ficam à vista. Pegar a primeira em silêncio era o
        # que fazia "Rua São João" cair na cidade errada -- e o erro só
        # aparecia depois, na produtividade do PVGIS, onde ninguém procura.
        lugares = st.session_state.get("e_lugares") or []
        if len(lugares) > 1:
            escolha = st.radio(
                "Achei mais de um lugar. Qual é?",
                range(len(lugares)),
                format_func=lambda i: lugares[i]["nome"],
            )
            if st.button("Usar este", type="primary"):
                _fixar_local(lugares[escolha])
                st.rerun()
        elif len(lugares) == 1:
            _fixar_local(lugares[0])
            st.session_state["e_lugares"] = []
            st.rerun()

        if st.session_state["e_endereco"]:
            st.success(
                f"📍 {st.session_state['e_endereco']}  \n"
                f"`{st.session_state['e_lat']:.5f}, {st.session_state['e_lon']:.5f}`"
            )

    with aba_coordenadas:
        st.caption(
            "Cole a coordenada exata do telhado. É o caminho mais confiável: "
            "endereço depende do que o geocodificador entende, e coordenada não "
            "depende de nada."
        )
        colunas = st.columns([1, 1, 1])
        lat = colunas[0].number_input(
            "Latitude", min_value=-90.0, max_value=90.0, step=0.001, format="%.5f",
            value=float(st.session_state["e_lat"]), key="e_lat_digitada")
        lon = colunas[1].number_input(
            "Longitude", min_value=-180.0, max_value=180.0, step=0.001, format="%.5f",
            value=float(st.session_state["e_lon"]), key="e_lon_digitada")
        # Um botão, e não a escrita direta no estado: o `number_input` só
        # entrega o valor quando o campo perde o foco, e quem digitava e
        # clicava direto em "Continuar" levava o valor anterior sem perceber.
        # O botão torna o instante visível e encerra a dúvida.
        colunas[2].markdown("<div style='height:1.85rem'></div>", unsafe_allow_html=True)
        if colunas[2].button("Usar esta coordenada", width="stretch"):
            st.session_state.update({
                "e_lat": float(lat), "e_lon": float(lon), "e_lugares": [],
                # O rótulo do lugar anterior morre aqui. Deixá-lo vivo fazia a
                # tela dizer "Região Sudeste" com a coordenada certa embaixo.
                "e_endereco": f"coordenada informada ({lat:.5f}, {lon:.5f})",
            })
            st.rerun()

    st.caption(
        f"Coordenadas em uso: **{st.session_state['e_lat']:.5f}, "
        f"{st.session_state['e_lon']:.5f}** — é daqui que sai a série do PVGIS e "
        "é aqui que o mapa do telhado abre."
    )
    _rodape(pendencia="dar um nome ao cliente" if not st.session_state["e_nome"] else None)


# ============================================================================
# Passo 2 — cargas
# ============================================================================
_OPCOES_CARGA = {
    "vistoria": "Tenho uma vistoria técnica",
    "modelo": "Usar o rascunho do modelo",
    "planilha": "Tenho a planilha do D²",
    "conta": "Não tenho levantamento",
}


def _passo_cargas() -> None:
    escolha = st.radio(
        "De onde vêm as cargas",
        list(_OPCOES_CARGA),
        format_func=lambda k: _OPCOES_CARGA[k],
        index=list(_OPCOES_CARGA).index(st.session_state["e_modo_carga"]),
        horizontal=True,
        label_visibility="collapsed",
    )
    st.session_state["e_modo_carga"] = escolha

    if escolha == "conta":
        _cargas_pela_conta()
        return
    if escolha == "vistoria":
        _cargas_da_vistoria()
    if escolha == "planilha":
        _cargas_da_planilha()
    if st.session_state["e_cenario"] is None and escolha == "modelo":
        st.session_state["e_cenario"] = Cenario.de_segmento(
            st.session_state["e_segmento"], st.session_state["e_nome"] or "instalação"
        )
    cenario: Cenario | None = st.session_state["e_cenario"]
    if cenario is None:
        _rodape(pendencia="carregar a planilha de cargas")
        return

    _painel_da_vistoria()
    _editor_de_cargas(cenario)


def _cargas_da_vistoria() -> None:
    """
    Recebe o backup da vistoria técnica e traz tudo que ele sabe.

    O backup em JSON é a fonte: as duas planilhas exportadas ao lado dele são
    derivadas e perdem informação — a do simulador descarta cômodo e
    criticidade, que é o que define o quadro de backup, e o inventário sai com
    o texto corrompido pelo Excel.

    Além das cargas, o backup responde três perguntas que a tela faria adiante
    e que o vistoriador já respondeu em campo: a tensão da rede, se já existe
    geração, e qual equipamento é crítico. Redigitá-las seria pedir duas vezes
    o mesmo dado — e a segunda resposta é a que costuma vir errada.
    """
    import json

    from .demanda.contrato_vistoria import Gravidade, conferir
    from .demanda.vistoria import CRITICIDADES, DESCRICAO_CRITICIDADE, ler_backup

    arquivo = st.file_uploader(
        "Backup da vistoria técnica (.json)", type=["json"],
        help="O arquivo que a vistoria exporta com nome terminado em '-backup.json'. "
             "É a fonte completa: as planilhas ao lado dele perdem cômodo, "
             "criticidade e a tensão da rede.",
    )
    if arquivo is None:
        return

    try:
        dados = json.loads(arquivo.getvalue().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 — arquivo do usuário, erro legível
        st.error(f"Não consegui ler o JSON: {exc}")
        return

    # A conferência antes da importação: é melhor saber o que falta agora do
    # que descobrir no relatório que uma hipótese entrou no lugar de um dado.
    conferencia = conferir(dados)
    if not conferencia.serve:
        st.error(f"**{conferencia.resumo()}**")
        for achado in conferencia.bloqueios:
            st.markdown(f"- `{achado.campo.caminho}` — {achado.campo.sem_ele}")
        return

    importantes = conferencia.por_gravidade(Gravidade.IMPORTANTE)
    if importantes:
        with st.expander(f"⚠️ {len(importantes)} campo(s) importante(s) em falta — "
                         "o estudo roda, mas com hipótese no lugar"):
            for achado in importantes:
                st.markdown(f"- **`{achado.campo.caminho}`** — {achado.campo.sem_ele}")

    if not st.button("Importar esta vistoria", type="primary", width="stretch"):
        return

    try:
        vistoria = ler_backup(dados)
    except ValueError as exc:
        st.error(f"Não consegui importar: {exc}")
        return

    cenario = vistoria.cenario
    # O quadro de backup nasce do que a vistoria marcou como crítico.
    cenario.criticidades_essenciais = ("MC", "C")
    st.session_state.update({
        "e_cenario": cenario,
        "e_vistoria": vistoria,
        "e_nome": vistoria.nome,
        "e_tensao_rede": vistoria.tensao_rede_v,
        "e_ensemble_backup": None,
        "e_assinatura_carga": None,
    })
    # A cidade da vistoria vira coordenada, e não só rótulo. Escrever o texto
    # e deixar a coordenada no padrão de Curitiba fazia a tela dizer "Rio de
    # Janeiro / RJ" com o mapa aberto no Paraná — e toda a geração solar do
    # estudo saía do lugar errado, sem nenhum aviso.
    #
    # Mas **só com cidade**. Uma UF sozinha não é endereço: o geocodificador,
    # obrigado a responder alguma coisa, devolve o centroide de uma região
    # inteira — foi assim que "RJ" virou "Região Sudeste", a centenas de
    # quilômetros do imóvel. Uma coordenada plausível e errada é pior que
    # nenhuma, porque não se anuncia.
    if vistoria.cidade:
        st.session_state["e_endereco"] = vistoria.local
        achados = _buscar_endereco(vistoria.local, limite=1)
        if achados:
            _fixar_local(achados[0])
        else:
            st.warning(
                f"A vistoria diz “{vistoria.local}”, mas não consegui converter "
                "isso em coordenada. Confira o local no passo 1 antes de seguir: "
                "a geração solar sai dali."
            )
    elif vistoria.uf:
        st.warning(
            f"A vistoria informou apenas o estado ({vistoria.uf}), sem cidade. "
            "**A coordenada não foi alterada** — um estado inteiro não vira "
            "ponto no mapa, e chutar o centro dele poria a geração solar a "
            "centenas de quilômetros do imóvel. Informe o local no passo 1."
        )
    st.rerun()


def _painel_da_vistoria() -> None:
    """O que a vistoria trouxe, para conferência antes de seguir."""
    from .demanda.vistoria import CRITICIDADES, DESCRICAO_CRITICIDADE

    vistoria = st.session_state.get("e_vistoria")
    if vistoria is None:
        return
    cenario = st.session_state["e_cenario"]

    st.success(
        f"**{vistoria.cliente}** · vistoria de {vistoria.data} por "
        f"{vistoria.vistoriador or 'não informado'} · rede "
        f"{vistoria.tensao_rede_v:.0f} V · {vistoria.local or 'local não informado'}"
    )

    tabela = cenario.por_criticidade()
    if tabela.empty:
        return
    agregado = (
        tabela.groupby("criticidade")
        .agg(itens=("equipamento", "size"), potencia_w=("potencia_w", "sum"))
        .reindex([c for c in CRITICIDADES if c in set(tabela["criticidade"])])
    )
    total = float(agregado["potencia_w"].sum()) or 1.0
    st.markdown("##### O que a vistoria classificou")
    st.dataframe(
        pd.DataFrame({
            "criticidade": agregado.index,
            "o que significa": [DESCRICAO_CRITICIDADE.get(c, "") for c in agregado.index],
            "equipamentos": agregado["itens"].values,
            "potência (kW)": (agregado["potencia_w"] / 1000).round(2).values,
            "fração": [f"{v/total:.0%}" for v in agregado["potencia_w"].values],
        }),
        width="stretch", hide_index=True,
    )
    st.caption(
        "O quadro de backup é montado **equipamento a equipamento**, e não por "
        "ambiente: numa cozinha, a geladeira crítica entra e o forno elétrico não. "
        "Recortar por cômodo levaria os dois."
    )
    if vistoria.descartados:
        with st.expander(f"{len(vistoria.descartados)} item(ns) fora da simulação"):
            for linha in vistoria.descartados:
                st.markdown(f"- {linha}")


def _cargas_da_planilha() -> None:
    arquivo = st.file_uploader(
        "Planilha do D² — uma aba por cômodo", type=["xlsx", "xlsm"],
        help="O mesmo formato que o D² já lê. Uma aba opcional chamada 'instancias' "
             "define quantas vezes cada cômodo se repete.",
    )
    if arquivo is not None and st.button("Carregar planilha", type="primary"):
        try:
            st.session_state["e_cenario"] = Cenario.de_planilha(
                arquivo, st.session_state["e_nome"] or "instalação"
            )
        except Exception as exc:  # noqa: BLE001 — arquivo do usuário, erro tem que ser legível
            st.error(f"Não consegui ler a planilha: {exc}")
        else:
            st.session_state["e_ensemble_backup"] = None
            st.rerun()


def _cargas_pela_conta() -> None:
    perfil = segmento_do_perfil(st.session_state["e_segmento"])
    if perfil is None:
        st.error("Este segmento não tem curva de referência. Use o rascunho do modelo.")
        _rodape(pendencia="escolher outro caminho")
        return

    st.warning(
        "**O que você perde por este caminho.** A forma do dia vem da curva típica do "
        "segmento e o tamanho, da sua conta de luz. Falta a variabilidade dia a dia — "
        "e é ela que produz o pico raro que decide o inversor. O estudo assume uma "
        "dispersão de 15%, que é hipótese, não medição. Serve para uma primeira "
        "conversa; não substitui o levantamento antes de fechar a venda."
    )
    colunas = st.columns(2)
    colunas[0].number_input(
        "Consumo mensal da conta (kWh)", min_value=50.0, max_value=5_000_000.0,
        step=100.0, key="e_conta_kwh",
    )
    colunas[1].number_input(
        "Dias de operação por mês", min_value=1, max_value=31, key="e_dias_operacao",
        help="Comércio que fecha domingo opera ~26 dias; indústria de turno único, ~22.",
    )

    calibrado = calibrar_por_conta(
        perfil, st.session_state["e_conta_kwh"], st.session_state["e_dias_operacao"]
    )
    colunas = st.columns(3)
    _cartao(colunas[0], "Consumo diário", f"{calibrado['consumo_diario_kwh']:.0f} kWh")
    _cartao(colunas[1], "Demanda média", f"{calibrado['demanda_media_kw']:.1f} kW")
    _cartao(colunas[2], "Pico da curva", f"{calibrado['demanda_maxima_kw']:.1f} kW")

    st.caption(f"Curva de referência: **{perfil.nome}**"
               + ("" if perfil.confiavel else " — perfil sintético na origem da base."))
    st.line_chart(
        pd.DataFrame({"kW": calibrado["curva_kw"]}, index=pd.RangeIndex(24, name="hora")),
        height=200,
    )

    def _seguir() -> bool:
        ensemble = ensemble_de_curva_tipica(calibrado["curva_w_por_minuto"])
        st.session_state["e_cenario"] = None
        st.session_state["e_ensemble_total"] = ensemble
        st.session_state["e_ensemble_backup"] = ensemble
        st.session_state["e_essenciais"] = []
        st.session_state["e_avisos_carga"] = [calibrado["aviso"], ensemble.metadados["aviso"]]
        return True

    _rodape(ao_avancar=_seguir)


def _configuracao_de_colunas() -> dict[str, Any]:
    return {
        "Equipamento": st.column_config.TextColumn("Equipamento", width="medium", required=True),
        "Potência": st.column_config.NumberColumn(
            "Potência (W)", min_value=1, step=10, format="%d",
            help="Potência de placa, em watts. Não desconte o uso aqui — isso é o FD.",
        ),
        "Quantidade": st.column_config.NumberColumn("Qtd.", min_value=1, step=1, format="%d"),
        "Tipo de intervalo": st.column_config.SelectboxColumn(
            "Uso", options=list(TIPOS_INTERVALO), width="small",
            help="fixo: liga e desliga em horário certo. dinâmico: roda um tempo sorteado "
                 "dentro da janela.",
        ),
        "intervalo": st.column_config.TextColumn(
            "Janela", width="small",
            help="Formato \"08:00 as 18:00\". Pode atravessar a meia-noite: \"22:00 as 06:00\".",
        ),
        "probabilidade": st.column_config.NumberColumn(
            "Prob.", min_value=0.0, max_value=1.0, step=0.05, format="%.2f",
            help="Chance de o equipamento ser usado no dia. 1,0 = todo dia.",
        ),
        "FD": st.column_config.NumberColumn(
            "FD", min_value=0.01, max_value=1.5, step=0.05, format="%.2f",
            help="Fator de demanda: quanto da potência de placa o aparelho puxa em média "
                 "enquanto está ligado. Geladeira fica em ~0,35 por causa do compressor.",
        ),
        "duracao_min": st.column_config.NumberColumn(
            "Dur. mín (h)", min_value=0.0, step=0.25, format="%.2f",
            help="Obrigatório no tipo dinâmico.",
        ),
        "duracao_max": st.column_config.NumberColumn(
            "Dur. máx (h)", min_value=0.0, step=0.25, format="%.2f"),
        "modo_fixo": st.column_config.SelectboxColumn(
            "Modo", options=list(MODOS_FIXOS), width="small",
            help="Só vale no tipo fixo.",
        ),
    }


def _editor_de_cargas(cenario: Cenario) -> None:
    st.caption(
        "Cada cômodo é um ambiente típico; a quantidade multiplica ele. Um hotel com 40 "
        "apartamentos iguais tem **um** cômodo e 40 instâncias — não 40 cômodos."
    )

    remover: str | None = None
    for indice, nome in enumerate(list(cenario.comodos)):
        tabela = cenario.comodos[nome]
        potencia = sum(
            (float(l.get("Potência") or 0) * float(l.get("Quantidade") or 0))
            for _, l in tabela.iterrows()
        ) * cenario.instancias.get(nome, 1) / 1000.0
        rotulo = f"{nome} — {len(tabela)} equipamentos · {potencia:.1f} kW instalados"
        with st.expander(rotulo, expanded=indice == 0):
            topo = st.columns([3, 1, 1])
            novo_nome = topo[0].text_input("Nome do cômodo", value=nome, key=f"nome_{indice}")
            cenario.instancias[nome] = topo[1].number_input(
                "Quantos existem", min_value=1, max_value=5000,
                value=int(cenario.instancias.get(nome, 1)), key=f"inst_{indice}",
            )
            topo[2].write("")
            if topo[2].button("Remover", key=f"rm_{indice}", width="stretch"):
                remover = nome

            editada = st.data_editor(
                tabela, num_rows="dynamic", width="stretch", key=f"ed_{indice}",
                column_config=_configuracao_de_colunas(),
            )
            # A grade devolve colunas como `object` quando o usuário apaga uma
            # célula, e aí um `None` volta a ser desenhado como o texto "None"
            # no rerun seguinte. Normalizar na volta mantém os tipos estáveis e
            # reaplica a regra de qual coluna se aplica a qual tipo de uso.
            cenario.comodos[nome] = Cenario._normalizar(editada)

            adicionar = st.columns([2, 3, 1, 1])
            categoria = adicionar[0].selectbox(
                "Categoria", CATEGORIAS, key=f"cat_{indice}", label_visibility="collapsed")
            opcoes = [e["nome"] for e in EQUIPAMENTOS if e["categoria"] == categoria]
            escolhido = adicionar[1].selectbox(
                "Equipamento", opcoes, key=f"eq_{indice}", label_visibility="collapsed")
            quantidade = adicionar[2].number_input(
                "Qtd", min_value=1, max_value=999, value=1,
                key=f"qtd_{indice}", label_visibility="collapsed")
            if adicionar[3].button("Adicionar", key=f"add_{indice}", width="stretch"):
                cenario.adicionar_equipamento(nome, escolhido, int(quantidade))
                st.rerun()

            if novo_nome != nome and novo_nome.strip():
                cenario.comodos = {
                    (novo_nome if k == nome else k): v for k, v in cenario.comodos.items()
                }
                cenario.instancias[novo_nome] = cenario.instancias.pop(nome)
                if nome in cenario.essenciais:
                    cenario.essenciais = [novo_nome if e == nome else e for e in cenario.essenciais]
                st.rerun()

    if remover:
        cenario.remover_comodo(remover)
        st.rerun()

    colunas = st.columns([2, 1, 1])
    novo = colunas[0].text_input("Nome do novo cômodo", placeholder="Ex.: Cozinha",
                                 label_visibility="collapsed")
    if colunas[1].button("+ Novo cômodo", width="stretch", disabled=not novo.strip()):
        cenario.adicionar_comodo(novo.strip())
        st.rerun()
    colunas[2].download_button(
        "⬇️ Planilha", data=cenario.para_planilha(),
        file_name=f"cargas-{_slug(cenario.nome)}.xlsx", width="stretch",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    # -- conferência -------------------------------------------------------
    st.divider()
    problemas = cenario.validar()
    erros = [p for p in problemas if p.impede]
    avisos = [p for p in problemas if not p.impede]

    colunas = st.columns(3)
    _cartao(colunas[0], "Equipamentos", str(cenario.total_de_equipamentos()))
    _cartao(colunas[1], "Potência instalada", f"{cenario.potencia_instalada_w() / 1000:.1f} kW")
    _cartao(colunas[2], "Problemas", f"{len(erros)} erros · {len(avisos)} avisos")

    if erros:
        st.error(f"**{len(erros)} problema(s) que impedem a simulação:**")
        for problema in erros[:12]:
            st.markdown(f"- {problema}")
    if avisos:
        with st.expander(f"⚠️ {len(avisos)} aviso(s) — não impedem, mas vale conferir"):
            for problema in avisos:
                st.markdown(f"- {problema}")

    st.session_state["e_cenario"] = cenario
    st.session_state["e_avisos_carga"] = [str(p) for p in avisos]
    _rodape(pendencia=f"corrigir {len(erros)} erro(s) acima" if erros else None)



# ============================================================================
# Passo 3 — a análise do consumo
# ============================================================================
def _passo_analise() -> None:
    """
    A análise completa da demanda, antes de qualquer decisão de equipamento.

    É a etapa que o fluxo anterior não tinha: ele simulava a carga e ia direto
    dimensionar. Mas o número que sai daqui — o pico, quem o causa e a que hora
    — é o que justifica cada escolha adiante, e o cliente merece vê-lo antes de
    ver um orçamento.
    """
    cenario: Cenario | None = st.session_state["e_cenario"]
    ensemble = st.session_state.get("e_ensemble_total")

    chave_ocupacao = _seletor_de_ocupacao(cenario)

    with st.expander("Ajustes da simulação"):
        colunas = st.columns(3)
        simulacoes = colunas[0].slider(
            "Simulações por estação", 50, 1000, value=300, step=50,
            help="Mais simulações estreitam a incerteza da cauda, que é onde o pico mora.")
        inicio_ponta = colunas[1].number_input("Início da ponta (h)", 0, 23, 18)
        fim_ponta = colunas[2].number_input("Fim da ponta (h)", 0, 23, 21)

    assinatura = (
        simulacoes, int(inicio_ponta), int(fim_ponta),
        cenario.total_de_equipamentos() if cenario else 0,
        round(cenario.potencia_instalada_w(), 3) if cenario else 0.0,
        chave_ocupacao,
    )
    atual = st.session_state.get("e_assinatura_analise") == assinatura

    if not atual:
        if st.session_state.get("e_analise") is not None:
            st.info("As cargas mudaram desde a última análise. Rode de novo para atualizar.")
        rotulo = "Analisar o consumo" if st.session_state.get("e_analise") is None else "Analisar de novo"
        if st.button(rotulo, type="primary", width="stretch"):
            with st.spinner("Simulando as quatro estações e decompondo o pico…"):
                if cenario is not None:
                    # Com um estudo de ocupação escolhido, quem é simulado é a
                    # casa reescrita pelo perfil que dimensiona — não a casa do
                    # formulário. O levantado em campo continua em `e_cenario`.
                    usado = _aplicar_ocupacao(cenario, chave_ocupacao, simulacoes)
                    st.session_state["e_cenario_ocupacao"] = (
                        usado if usado is not cenario else None
                    )
                    novo = simular_ensemble(
                        usado.para_comodos(), usado.instancias_de(),
                        num_simulacoes=simulacoes)
                    st.session_state["e_ensemble_total"] = novo
                    st.session_state["e_analise"] = analisar(
                        novo, usado.para_comodos(), usado.instancias_de(),
                        potencia_instalada_w=usado.potencia_instalada_w(),
                        inicio_ponta_h=int(inicio_ponta), fim_ponta_h=int(fim_ponta))
                else:
                    st.session_state["e_analise"] = analisar(
                        ensemble, inicio_ponta_h=int(inicio_ponta),
                        fim_ponta_h=int(fim_ponta), detalhar_pico=False)
            st.session_state["e_assinatura_analise"] = assinatura
            st.session_state["e_assinatura_carga"] = None
            st.rerun()
        _rodape(pendencia="rodar a análise do consumo")
        return

    analise = st.session_state["e_analise"]
    _mostrar_ocupacao()
    _mostrar_analise(analise)
    _rodape()


def _seletor_de_ocupacao(cenario: Cenario | None) -> str:
    """
    Como esta casa é usada — a pergunta que decide o tamanho de tudo.

    Só aparece para residência, porque só ali a diferença entre o dia cheio e o
    dia vazio é de formato e não de escala. Num hotel ou num galpão a curva
    levantada já é a curva de operação.

    A opção de não escolher nada continua primeira e continua legítima: se a
    vistoria levantou janelas reais em vez das do formulário, reescrevê-las é
    piorar o dado.
    """
    from .demanda import ocupacao

    if cenario is None or (cenario.segmento or "").lower() != "residencia":
        st.session_state["e_estudo_ocupacao"] = ""
        return ""

    rotulos = {"": "Usar as janelas da vistoria como vieram"}
    rotulos.update({c: e["nome"] for c, e in ocupacao.ESTUDOS.items()})
    chaves = list(rotulos)

    escolha = st.selectbox(
        "Como esta casa é usada?",
        chaves,
        index=chaves.index(st.session_state.get("e_estudo_ocupacao", "") or ""),
        format_func=lambda c: rotulos[c],
        key="e_estudo_ocupacao",
        help="Uma residência não tem uma curva de carga, tem pelo menos duas. "
             "O equipamento é dimensionado pelo dia mais cheio; a conta de "
             "energia, pela média dos sete dias.",
    )
    if escolha:
        st.caption(ocupacao.ESTUDOS[escolha]["para_que"])
    return escolha


def _aplicar_ocupacao(cenario: Cenario, chave: str, simulacoes: int) -> Cenario:
    """
    Simula cada perfil do estudo e devolve o cenário que dimensiona.

    A conta de energia sai da média ponderada dos perfis, e o equipamento, do
    perfil de maior pico. Misturar os dois — dimensionar solar pelo fim de
    semana ou bateria pela média — é o erro que este cálculo existe para
    evitar: no primeiro caso o gerador fica 40% grande demais, no segundo o
    banco não atravessa o sábado.
    """
    from .demanda import ocupacao

    st.session_state["e_ocupacao"] = None
    if not chave or chave not in ocupacao.ESTUDOS:
        return cenario

    definicao = ocupacao.ESTUDOS[chave]
    pesos: dict[str, int] = definicao["perfis"]

    medidos: dict[str, dict[str, Any]] = {}
    for identificador in pesos:
        perfil = ocupacao.PERFIS[identificador]
        ajustado = ocupacao.aplicar(cenario, perfil)
        ajustado.criticidades_essenciais = cenario.criticidades_essenciais
        # Menos simulações que a análise: aqui só se quer a curva média e o
        # consumo do dia, e nenhum dos dois mora na cauda.
        conjunto = simular_ensemble(
            ajustado.para_comodos(), ajustado.instancias_de(),
            num_simulacoes=max(50, simulacoes // 3),
        )
        suave = conjunto.reamostrar(15) if conjunto.passo_min == 1 else conjunto
        geral = conjunto.resumo()["geral"]
        medidos[identificador] = {
            "perfil": perfil, "cenario": ajustado,
            "curva": suave.perfil_medio_w(), "passo_min": suave.passo_min,
            "pico_p95_kw": geral["pico_p95_kw"],
            "energia_diaria_kwh": geral["energia_diaria_media_kwh"],
        }

    dimensionante = max(pesos, key=lambda i: medidos[i]["pico_p95_kw"])
    diaria = sum(medidos[i]["energia_diaria_kwh"] * n for i, n in pesos.items())
    diaria /= sum(pesos.values())

    comparacao = ocupacao.comparar(cenario, [medidos[i]["perfil"] for i in pesos])
    comparacao["pico_p95_kw"] = [medidos[i]["pico_p95_kw"] for i in pesos]
    comparacao["energia_diaria_kwh"] = [medidos[i]["energia_diaria_kwh"] for i in pesos]

    st.session_state["e_ocupacao"] = {
        "estudo": chave,
        "tabela": comparacao,
        "dimensionante": medidos[dimensionante]["perfil"].nome,
        "para_que": definicao["para_que"],
        "curvas": {medidos[i]["perfil"].nome: medidos[i]["curva"] for i in pesos},
        "passo_min": medidos[dimensionante]["passo_min"],
        "reencaixados": int(comparacao["janelas_alteradas"].max()),
        "diaria_ponderada_kwh": diaria,
    }
    return medidos[dimensionante]["cenario"]


def _mostrar_ocupacao() -> None:
    """A tabela dos perfis, para conferência antes de seguir."""
    dados = st.session_state.get("e_ocupacao")
    if not dados:
        return

    st.markdown("##### Como esta casa é usada")
    quadro = dados["tabela"][[
        "perfil", "acordado", "refeicoes", "uso_diurno", "dias_por_semana",
        "pico_p95_kw", "energia_diaria_kwh",
    ]].rename(columns={
        "perfil": "perfil", "acordado": "acordado", "refeicoes": "refeições",
        "uso_diurno": "uso diurno", "dias_por_semana": "dias/semana",
        "pico_p95_kw": "pico P95 (kW)", "energia_diaria_kwh": "consumo (kWh/dia)",
    }).round(2)
    st.dataframe(quadro, width="stretch", hide_index=True)

    diaria = float(dados["diaria_ponderada_kwh"])
    st.caption(
        f"O equipamento é dimensionado por **{dados['dimensionante']}**, que é o "
        f"pior caso. A conta de energia usa a média ponderada dos sete dias: "
        f"**{diaria:.1f} kWh/dia**, ou {diaria * 30:.0f} kWh/mês."
    )


def _mostrar_analise(analise) -> None:
    estatisticas = analise.estatisticas
    indicadores = analise.indicadores

    colunas = st.columns(4)
    _cartao(colunas[0], "Consumo diário", f"{_milhar(indicadores.consumo_diario_kwh)} kWh")
    _cartao(colunas[1], "Demanda média", f"{indicadores.demanda_media_kw:.1f} kW")
    _cartao(colunas[2], "Pico P95", f"{estatisticas.percentis[95] / 1000:.1f} kW")
    _cartao(colunas[3], "Fator de carga", f"{indicadores.fator_de_carga:.2f}")

    st.caption(
        f"O pico de dimensionamento é **{estatisticas.percentis[95] / 1000:.1f} kW** "
        f"(P95), com incerteza de ±{estatisticas.erro_padrao_p95 / 1000:.1f} kW pelo "
        f"próprio Monte Carlo. {estatisticas.interpretacao.capitalize()}."
    )

    abas = st.tabs([
        "Distribuição dos picos", "Quem causa o pico", "Curva de duração",
        "Perfil do dia", "Por estação", "Indicadores",
    ])

    with abas[0]:
        picos = analise.ensemble.picos_diarios_w() / 1000.0
        st.bar_chart(
            pd.DataFrame({"dias": np.histogram(picos, bins=25)[0]},
                         index=pd.Index(np.round(np.histogram(picos, bins=25)[1][:-1], 1),
                                        name="pico (kW)")),
            height=260)
        st.dataframe(analise.tabela_percentis().round(2), width="stretch", hide_index=True)
        st.caption(
            "O P95 é a referência consagrada de dimensionamento: garante que 95% dos dias "
            "simulados fiquem abaixo dele. A distância entre o P95 e o P99 é o argumento a "
            "favor ou contra subir um degrau de equipamento."
        )

    with abas[1]:
        if analise.composicao is None:
            st.info("A composição do pico exige o levantamento de equipamentos.")
        else:
            composicao = analise.composicao
            st.markdown(
                f"O pico acontece com mais frequência às **{composicao.hora_mais_provavel}h**."
            )
            st.dataframe(
                composicao.por_equipamento.head(12).round(3),
                width="stretch", hide_index=True,
                column_config={
                    "carga_media_kw": st.column_config.NumberColumn("kW no pico", format="%.2f"),
                    "presenca": st.column_config.ProgressColumn(
                        "Presença nos picos", min_value=0, max_value=1, format="%.0f%%"),
                    "participacao": st.column_config.ProgressColumn(
                        "Fatia do pico", min_value=0, max_value=1, format="%.1f%%"),
                },
            )
            st.caption(
                "Saber que o pico é de 60 kW não diz o que fazer. Saber que dois terços "
                "dele são chuveiros elétricos às 8 h diz — e abre a conversa sobre "
                "aquecimento solar ou deslocamento de carga antes de comprar inversor maior."
            )
            st.bar_chart(
                composicao.por_comodo.set_index("comodo")["carga_media_kw"], height=220)

    with abas[2]:
        fracao, valores = analise.curva_de_duracao()
        amostra = np.linspace(0, len(valores) - 1, 400).astype(int)
        st.line_chart(
            pd.DataFrame({"kW": valores[amostra] / 1000.0},
                         index=pd.Index(fracao[amostra] * 100, name="% do tempo")),
            height=260)
        st.caption(
            "Lida da esquerda para a direita: a parte alta e estreita é a carga que só "
            "existe em poucas horas do ano — exatamente a que um banco de baterias corta "
            "sem que ninguém perceba."
        )

    with abas[3]:
        passo = analise.ensemble.reamostrar(5) if analise.ensemble.passo_min == 1 else analise.ensemble
        horas = np.arange(passo.passos_por_dia) * passo.passo_min / 60.0
        st.line_chart(
            pd.DataFrame({e: passo.perfil_medio_w(e) / 1000.0 for e in passo.estacoes},
                         index=pd.Index(horas, name="hora")),
            height=260)
        if analise.perfis_por_comodo is not None:
            st.markdown("##### Composição hora a hora, por ambiente")
            st.area_chart(analise.perfis_por_comodo.iloc[::15], height=260)
        st.bar_chart(
            pd.DataFrame({"fator de carga": analise.fator_de_carga_horario()},
                         index=pd.Index(range(24), name="hora")),
            height=200)
        st.caption(
            "O fator de carga por hora separa o que é carga firme do que é pico: hora com "
            "fator alto tem consumo parecido todo dia; hora com fator baixo depende de "
            "coincidência."
        )

    with abas[4]:
        st.dataframe(analise.por_estacao.round(2), width="stretch", hide_index=True)
        st.caption(
            "O mesmo prédio tem picos diferentes em janeiro e em julho. É o pior deles que "
            "dimensiona, e é por isso que a simulação roda as quatro estações em separado."
        )

    with abas[5]:
        st.dataframe(
            pd.DataFrame([
                {"indicador": "Fator de carga", "valor": f"{indicadores.fator_de_carga:.2f}",
                 "o que significa": indicadores.interpretacao_fator_carga},
                {"indicador": "Fator de demanda",
                 "valor": f"{indicadores.fator_de_demanda:.2f}",
                 "o que significa": "demanda máxima sobre potência instalada — o clássico da NBR 5410"},
                {"indicador": "Fator de coincidência",
                 "valor": f"{indicadores.fator_de_coincidencia:.2f}",
                 "o que significa": "quanto a operação real economiza por os equipamentos não ligarem juntos"},
                {"indicador": "Coeficiente de variação do pico",
                 "valor": f"{estatisticas.coeficiente_variacao:.2f}",
                 "o que significa": estatisticas.interpretacao},
                {"indicador": "Fração do consumo na ponta",
                 "valor": f"{indicadores.fracao_na_ponta:.0%}",
                 "o que significa": f"janela de {analise.inicio_ponta_h}h às {analise.fim_ponta_h}h"},
            ]),
            width="stretch", hide_index=True)
        colunas = st.columns(3)
        _cartao(colunas[0], "Potência instalada",
                f"{_milhar(indicadores.potencia_instalada_kw)} kW")
        _cartao(colunas[1], "Soma dos picos por ambiente",
                f"{_milhar(indicadores.soma_dos_picos_individuais_kw)} kW")
        _cartao(colunas[2], "Pico do conjunto",
                f"{indicadores.demanda_maxima_kw:.1f} kW")
        st.caption(
            "Os três números contam a mesma história em escalas diferentes: a potência de "
            "placa somada, o que cada ambiente pediria isolado, e o que a instalação "
            "realmente pede junta. A distância entre o primeiro e o último é o que separa "
            "um projeto viável de um superdimensionado."
        )


# ============================================================================
# Passo 5 — escolha do equipamento e memória de cálculo
# ============================================================================
def _passo_equipamento() -> None:
    from .pv.equipment import EquipmentError, carregar_base
    from .pv.memoria import memoria_do_arranjo

    try:
        base = carregar_base()
    except EquipmentError as exc:
        st.error(f"Catálogo de equipamentos indisponível: {exc}")
        _rodape(pendencia="corrigir o catálogo BDFotovoltaica.xlsx")
        return

    layout = st.session_state.get("e_layout")
    disponiveis = layout.quantidade if layout is not None else None
    if disponiveis:
        st.caption(
            f"O telhado marcado comporta **{disponiveis} módulos**. A escolha abaixo "
            "define quantos deles o arranjo consegue de fato usar."
        )
    else:
        st.caption(
            "Sem telhado marcado, a memória descreve o arranjo cheio que o inversor "
            "aceita — não o que cabe na cobertura."
        )

    colunas = st.columns(2)
    modulos = sorted(base.modulos, key=lambda m: -m.potencia_wp)
    inversores = base.inversores_ordenados()
    # `key=` em vez de `index=`: passar o índice a cada rerun sobrescreve a
    # escolha que o usuário acabou de fazer com a que estava guardada antes —
    # o seletor volta sozinho para o valor anterior.
    st.session_state.setdefault("e_modulo_sel", modulos[0])
    st.session_state.setdefault("e_inversor_sel", inversores[0])
    escolha_modulo = colunas[0].selectbox(
        "Módulo fotovoltaico", modulos, format_func=str, key="e_modulo_sel")
    escolha_inversor = colunas[1].selectbox(
        "Inversor de rede", inversores, format_func=str, key="e_inversor_sel")
    st.session_state["e_modulo"] = escolha_modulo.modelo
    st.session_state["e_inversor"] = escolha_inversor.modelo

    memoria = memoria_do_arranjo(escolha_modulo, escolha_inversor, disponiveis)
    st.session_state["e_memoria"] = memoria

    if not memoria.viavel:
        st.error(
            "Este par módulo/inversor não fecha um arranjo válido. A memória abaixo "
            "mostra em que passo a conta trava."
        )
    else:
        colunas = st.columns(4)
        _cartao(colunas[0], "Arranjo",
                f"{memoria.modulos_por_string} × {memoria.strings_do_sistema}")
        _cartao(colunas[1], "Inversores", f"{memoria.inversores} × {escolha_inversor.modelo}")
        _cartao(colunas[2], "Potência do gerador",
                f"{memoria.potencia_do_sistema_kwp:.1f} kWp")
        _cartao(colunas[3], "Razão CC/CA", f"{memoria.razao_cc_ca:.2f}")
        st.session_state["e_kwp"] = float(memoria.potencia_do_sistema_kwp)

    st.markdown("##### Memória de cálculo")
    for passo in memoria.passos:
        with st.container(border=True):
            st.markdown(f"**{passo.titulo}**")
            st.latex(_para_latex(passo.formula))
            st.code(passo.substituicao, language=None)
            marca = "🔴" if passo.critico else "→"
            st.markdown(f"{marca} **{passo.resultado}**")
            if passo.comentario:
                st.caption(passo.comentario)

    for aviso in memoria.avisos:
        st.markdown(f'<div class="aviso-suave">⚠️ {aviso}</div>', unsafe_allow_html=True)

    st.divider()
    st.success(
        f"**Dimensionamento fechado.** {memoria.modulos_do_sistema} módulos "
        f"{escolha_modulo.modelo} e {memoria.inversores} inversor(es) "
        f"{escolha_inversor.modelo}, somando {memoria.potencia_do_sistema_kwp:.1f} kWp."
        if memoria.viavel else
        "**O dimensionamento não fechou.** Escolha outro par de equipamentos."
    )
    _rodape(
        pendencia=None if memoria.viavel else "escolher um par de equipamentos compatível",
        rotulo_avancar="Analisar com bateria →",
    )


def _para_latex(formula: str) -> str:
    """Converte a fórmula legível em algo que o `st.latex` desenhe sem susto."""
    trocas = {
        "×": r"\times", "÷": r"\div", "≤": r"\le", "≥": r"\ge",
        "⌊": r"\lfloor ", "⌋": r" \rfloor", "⌈": r"\lceil ", "⌉": r" \rceil",
        "−": "-", "°C": r"^\circ C", "β": r"\beta", "_": r"\_",
    }
    saida = formula
    for antes, depois in trocas.items():
        saida = saida.replace(antes, depois)
    return r"\text{" + saida.replace("{", "").replace("}", "") + "}"


# ============================================================================
# Passo 3 — backup
# ============================================================================
def _passo_backup() -> None:
    cenario: Cenario | None = st.session_state["e_cenario"]
    if cenario is None:
        st.info(
            "Você escolheu o caminho da conta de luz, sem levantamento de equipamentos. "
            "Não dá para separar circuitos essenciais sem saber quais são — o estudo vai "
            "considerar a instalação inteira no backup, que é o cenário mais caro."
        )
        fracao = st.slider(
            "Que fração da carga total ficaria no quadro de backup?",
            0.1, 1.0, value=1.0, step=0.05,
            help="Um recorte grosseiro, já que não há lista de circuitos. Metade da carga "
                 "é o típico quando só entram iluminação, refrigeração e TI.",
        )
        base = st.session_state["e_ensemble_total"]
        if fracao < 1.0:
            from .demanda.ensemble import EnsembleCarga

            st.session_state["e_ensemble_backup"] = EnsembleCarga(
                perfis_w={e: base.perfis_w[e] * fracao for e in base.estacoes},
                passo_min=base.passo_min,
                metadados=dict(base.metadados, fracao_backup=fracao),
            )
        else:
            st.session_state["e_ensemble_backup"] = base
        _mostrar_carga_de_backup()
        _rodape()
        return

    st.caption(
        "Marque os ambientes que ficam ligados no inversor quando a rede cai. Um hotel "
        "que põe só elevador, bombas e circulação no backup precisa de **um terço** do "
        "inversor que a carga total pediria — e é o inversor que domina o preço."
    )

    # `key=` em vez de `default=`. Com `default=`, cada re-execução reimpõe o
    # padrão por cima do que o usuário acabou de fazer: desmarcar o último
    # ambiente devolvia a lista inteira, porque `[] or cenario.essenciais or
    # todos` cai no terceiro termo — lista vazia é falsa em Python. O sintoma
    # era ter de clicar duas vezes para tirar um ambiente.
    opcoes = list(cenario.comodos)
    if st.session_state.get("e_opcoes_essenciais") != opcoes:
        # As opções mudaram (o usuário mexeu nas cargas): reaproveita o que
        # ainda existe em vez de deixar o Streamlit quebrar com um valor órfão.
        anterior = st.session_state.get("e_essenciais_sel")
        if anterior is None:
            anterior = st.session_state["e_essenciais"] or cenario.essenciais or opcoes
        st.session_state["e_essenciais_sel"] = [c for c in anterior if c in opcoes]
        st.session_state["e_opcoes_essenciais"] = opcoes

    escolhidos = st.multiselect(
        "Ambientes essenciais",
        options=opcoes,
        key="e_essenciais_sel",
        placeholder="Escolha ao menos um ambiente",
        label_visibility="collapsed",
    )
    cenario.essenciais = escolhidos
    st.session_state["e_essenciais"] = escolhidos

    if escolhidos:
        total = cenario.potencia_instalada_w() / 1000.0
        backup = cenario.potencia_instalada_w(True) / 1000.0
        colunas = st.columns(3)
        _cartao(colunas[0], "Instalado total", f"{total:.1f} kW")
        _cartao(colunas[1], "No quadro de backup", f"{backup:.1f} kW")
        _cartao(colunas[2], "Fração no backup", f"{backup / total:.0%}" if total else "—")

    with st.expander("Ajustes avançados"):
        simulacoes = st.slider(
            "Simulações de Monte Carlo por estação", 50, 1000, value=300, step=50,
            help="Mais simulações estreitam a incerteza da cauda — que é onde o inversor "
                 "é decidido. 300 é suficiente para a maioria dos casos.",
        )

    # Simular e avançar são dois botões, não um. O painel que a simulação
    # produz — qual inversor aguenta, se a curva bate com o padrão do setor —
    # é metade do valor deste passo; num botão só, o usuário passa direto por
    # ele sem nunca ver.
    assinatura = (tuple(escolhidos), int(simulacoes), cenario.total_de_equipamentos(),
                  round(cenario.potencia_instalada_w(True), 3))
    atualizado = st.session_state.get("e_assinatura_carga") == assinatura

    if not atualizado:
        if st.session_state["e_ensemble_backup"] is not None:
            st.info("As cargas mudaram desde a última simulação. Rode de novo para atualizar.")
        rotulo = "Simular a demanda" if st.session_state["e_ensemble_backup"] is None \
            else "Simular de novo"
        if st.button(rotulo, type="primary", width="stretch", disabled=not escolhidos):
            with st.spinner("Simulando a demanda nas quatro estações…"):
                st.session_state["e_ensemble_total"] = simular_ensemble(
                    cenario.para_comodos(), cenario.instancias_de(), num_simulacoes=simulacoes)
                st.session_state["e_ensemble_backup"] = simular_ensemble(
                    cenario.para_comodos(True), cenario.instancias_de(True),
                    num_simulacoes=simulacoes)
            st.session_state["e_assinatura_carga"] = assinatura
            st.rerun()

    if atualizado and st.session_state["e_ensemble_backup"] is not None:
        _mostrar_carga_de_backup()

    if not escolhidos:
        pendencia = "escolher ao menos um ambiente"
    elif not atualizado:
        pendencia = "simular a demanda com as cargas atuais"
    else:
        pendencia = None
    _rodape(pendencia=pendencia)


def _mostrar_carga_de_backup() -> None:
    ensemble = st.session_state["e_ensemble_backup"]
    if ensemble is None:
        return
    st.divider()
    st.markdown("##### A carga que o sistema vai ter que sustentar")
    geral = ensemble.resumo()["geral"]
    colunas = st.columns(3)
    _cartao(colunas[0], "Pico médio do dia", f"{geral['pico_medio_kw']:.1f} kW")
    _cartao(colunas[1], "Pico P95", f"{geral['pico_p95_kw']:.1f} kW")
    _cartao(colunas[2], "Consumo diário", f"{geral['energia_diaria_media_kwh']:.0f} kWh")

    passo = ensemble.reamostrar(5) if ensemble.passo_min == 1 else ensemble
    aba_curva, aba_inv, aba_conf = st.tabs(
        ["Curva do dia", "Que inversor aguenta", "Bate com o padrão do setor?"])

    with aba_curva:
        horas = np.arange(passo.passos_por_dia) * passo.passo_min / 60.0
        dados = pd.DataFrame(
            {e: passo.perfil_medio_w(e) / 1000.0 for e in passo.estacoes},
            index=pd.Index(horas, name="hora"),
        )
        st.line_chart(dados, height=260)
        st.caption("Média das simulações, por estação. O pico que dimensiona não está aqui — "
                   "está na cauda, na aba ao lado.")

    with aba_inv:
        try:
            base = _catalogo()
        except CatalogoError as exc:
            st.error(str(exc))
        else:
            diagnostico = avaliar_inversores(base, passo)
            aprovados = int(diagnostico["aprovado"].sum())
            if aprovados == 0:
                st.error(
                    "Nenhum inversor do catálogo aguenta essa carga sem desarmar. "
                    "Ou o quadro de backup precisa encolher, ou o catálogo precisa de "
                    "modelos maiores."
                )
            else:
                menor = diagnostico[diagnostico["aprovado"]].iloc[0]
                st.success(
                    f"O menor inversor que aguenta é o **{menor['fabricante']} "
                    f"{menor['modelo']}** ({menor['nominal_kw']:.0f} kW). "
                    f"{aprovados} de {len(diagnostico)} modelos passam."
                )
            st.dataframe(
                diagnostico[["fabricante", "modelo", "nominal_kw", "pico_kw",
                             "prob_pico_diario_acima_nominal", "duracao_media_min", "veredito"]]
                .round(3),
                width="stretch", hide_index=True,
            )
            st.dataframe(potencia_para_excedencia(passo).round(2), width="stretch", hide_index=True)
            st.caption(
                "`pico_diario_kw`: potência excedida numa fração dos **dias** — dimensiona. "
                "`carga_instantanea_kw`: excedida numa fração do **tempo** — diz por quanto "
                "tempo o inversor ficaria saturado."
            )

    with aba_conf:
        perfil = segmento_do_perfil(st.session_state["e_segmento"])
        total = st.session_state["e_ensemble_total"]
        if perfil is None:
            st.info("Este segmento não tem curva de referência na base.")
        elif total is None:
            st.info("Simule a demanda para comparar.")
        else:
            # A comparação usa a carga **total**, não a de backup: a curva de
            # referência descreve a instalação inteira, e um subconjunto de
            # circuitos não tem por que se parecer com ela. Comparar o recorte
            # contra o todo geraria um alarme falso a cada estudo.
            referencia = total.reamostrar(5) if total.passo_min == 1 else total
            st.caption(
                "Comparação feita sobre a **carga total** da instalação — a curva de "
                "referência descreve o prédio inteiro, não o recorte do quadro de backup."
            )
            comparacao = comparar_com_perfil(
                referencia.perfil_medio_w(), perfil, referencia.passo_min)
            colunas = st.columns(3)
            _cartao(colunas[0], f"Maior desvio ({comparacao['pior_periodo']})",
                    f"{comparacao['divergencia_do_pior_periodo_pp']:.0f} pp")
            _cartao(colunas[1], "Hora do pico",
                    f"{comparacao['hora_de_pico_simulada']}h vs "
                    f"{comparacao['hora_de_pico_referencia']}h")
            _cartao(colunas[2], "Correlação (informativa)", f"{comparacao['correlacao']:.2f}")
            st.line_chart(
                pd.DataFrame(
                    {"seu levantamento": comparacao["curva_simulada_pu"],
                     f"padrão {perfil.nome}": comparacao["curva_referencia_pu"]},
                    index=pd.RangeIndex(24, name="hora"),
                ),
                height=240,
            )
            if not comparacao["coerente"]:
                periodo = comparacao["pior_periodo"]
                desvio = comparacao["divergencia_por_periodo_pp"][periodo]
                direcao = "mais" if desvio > 0 else "menos"
                st.warning(
                    f"Seu levantamento coloca **{abs(desvio):.0f} pontos percentuais {direcao}** "
                    f"do consumo diário no período da **{periodo}** do que o padrão do setor. "
                    "Pode ser uma instalação atípica — ou uma janela de uso digitada errado. "
                    "Vale conferir os equipamentos que operam nesse período."
                )
            else:
                st.success(
                    "A distribuição da energia ao longo do dia bate com o padrão do setor."
                )
            st.dataframe(
                pd.DataFrame([
                    {"período": nome,
                     "seu levantamento": f"{comparacao['energia_por_periodo_pu'][nome]:.0%}",
                     "padrão do setor":
                         f"{comparacao['energia_por_periodo_pu'][nome] - comparacao['divergencia_por_periodo_pp'][nome] / 100:.0%}",
                     "diferença": f"{comparacao['divergencia_por_periodo_pp'][nome]:+.0f} pp"}
                    for nome in comparacao["energia_por_periodo_pu"]
                ]),
                width="stretch", hide_index=True,
            )
            st.caption(
                "O veredito sai da energia por período, não da correlação hora a hora: "
                "um pico legítimo que a curva de referência não modela — o chuveiro "
                "elétrico às 7 h, por exemplo — derruba a correlação sem que nada esteja "
                "errado."
            )
            if not perfil.confiavel:
                st.caption(
                    "A curva de referência está marcada como sintética na origem da base: "
                    "serve de sanidade, não de prova."
                )


# ============================================================================
# Passo 4 — solar
# ============================================================================
def _passo_solar() -> None:
    consumo_anual = float(np.mean(st.session_state["e_ensemble_total"].energia_diaria_kwh())) * 365
    st.caption(f"Consumo anual estimado da instalação: **{_milhar(consumo_anual)} kWh**.")

    modo = st.radio(
        "Como definir o sistema",
        ["Marcar o telhado no mapa (recomendado)", "Compensar o consumo", "Já sei a potência"],
        horizontal=True, label_visibility="collapsed",
    )

    if modo.startswith("Marcar"):
        _solar_pelo_telhado()
    elif modo.startswith("Compensar"):
        st.info(
            "O sistema será dimensionado para compensar o consumo anual, usando a "
            "produtividade real do local. É o padrão de uma proposta de geração "
            "distribuída — mas não confere se cabe no telhado."
        )
        st.session_state["e_kwp"] = 0.0  # zero = o estudo dimensiona sozinho
        st.session_state["e_telhado"] = None
    else:
        st.session_state["e_kwp"] = st.number_input(
            "Potência do sistema (kWp)", 0.5, 5000.0,
            float(st.session_state["e_kwp"] or 30.0), 0.5)
        st.session_state["e_telhado"] = None

    with st.expander("Ajustes avançados"):
        colunas = st.columns(3)
        st.session_state["e_inclinacao"] = colunas[0].number_input(
            "Inclinação (°)", 0.0, 60.0,
            float(st.session_state.get("e_inclinacao") or 0.0), 1.0,
            help="Zero usa o ótimo da latitude — ou a inclinação do telhado marcado.")
        st.session_state["e_azimute"] = colunas[1].number_input(
            "Azimute (° — 0 = Norte)", 0.0, 359.0,
            float(st.session_state.get("e_azimute") or 0.0), 5.0)
        st.session_state["e_exigir_pvgis"] = colunas[2].checkbox(
            "Exigir dados do PVGIS", value=False,
            help="Sem rede, o padrão é cair para uma série sintética marcada como tal.")

    # Sem telhado marcado o estudo ainda roda — só não confere se cabe. Avisar
    # é melhor que bloquear: quem não tem a imagem do telhado à mão continua, e
    # a ressalva vai junto para o relatório em vez de virar uma promessa muda.
    if modo.startswith("Marcar") and st.session_state.get("e_telhado") is None:
        st.caption(
            "Sem telhado marcado, o sistema será dimensionado pelo consumo anual — "
            "e o estudo não terá como afirmar que ele cabe na cobertura."
        )
    _rodape()


_TILES_SATELITE = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/"
    "tile/{z}/{y}/{x}"
)


def _solar_pelo_telhado() -> None:
    """
    Desenhe a água do telhado; o programa mede, orienta e empacota os módulos.

    O croqui de volta no mapa não é enfeite: ver onde cada painel caiu denuncia
    uma orientação errada ou um recuo mal escolhido mais rápido que qualquer
    tabela de números.
    """
    import folium
    from folium.plugins import Draw
    from streamlit_folium import st_folium

    from .pv.equipment import EquipmentError, carregar_base
    from .pv.telhado import dimensionar_no_telhado, telhado_de_geojson

    colunas = st.columns(3)
    montagem = colunas[0].selectbox(
        "Tipo de cobertura", ["coplanar", "inclinado"],
        format_func=lambda m: (
            "Telhado inclinado (módulo acompanha a água)" if m == "coplanar"
            else "Laje plana (estrutura inclinada)"
        ),
        help="Coplanar: o módulo deita no telhado e herda a inclinação e a orientação "
             "dele. Laje: a estrutura aponta para onde se quiser, mas as fileiras "
             "precisam de espaçamento para não se sombrearem.",
    )
    inclinacao = colunas[1].number_input(
        "Inclinação do telhado (°)", 0.0, 45.0,
        20.0 if montagem == "coplanar" else 15.0, 1.0,
        help="Telha cerâmica fica entre 25° e 35°; fibrocimento, entre 10° e 20°.",
    )
    obstaculos = colunas[2].slider(
        "Área aproveitável", 0.5, 1.0, 0.90, 0.05, format="%.2f",
        help="Desconto do que a imagem de satélite não mostra: caixa d'água, "
             "claraboia, chaminé, condensadoras, sombra de platibanda.",
    )

    st.caption(
        "Use a ferramenta de **polígono** ou **retângulo** e contorne **uma água** do "
        "telhado. Cada água tem uma orientação própria; marque a que vai receber os "
        "módulos — e repita para as demais."
    )
    # Dizer onde o mapa está aberto. Quando a coordenada estava errada, o
    # sintoma era o usuário desenhar um telhado sobre a cidade errada e só
    # descobrir na produtividade do PVGIS, onde ninguém procura.
    st.caption(
        f"Aberto em **{st.session_state.get('e_endereco') or 'coordenada informada'}** "
        f"· `{st.session_state['e_lat']:.5f}, {st.session_state['e_lon']:.5f}`. "
        "Não é aqui? Volte ao passo 1 e corrija o local."
    )

    centro = [st.session_state["e_lat"], st.session_state["e_lon"]]
    mapa = folium.Map(location=centro, zoom_start=19, max_zoom=22, tiles=None)
    folium.TileLayer(
        tiles=_TILES_SATELITE, attr="Esri World Imagery",
        name="Satélite", max_zoom=22, max_native_zoom=19,
    ).add_to(mapa)
    Draw(
        draw_options={
            "polyline": False, "circle": False, "marker": False,
            "circlemarker": False, "polygon": True, "rectangle": True,
        },
        edit_options={"edit": True, "remove": True},
    ).add_to(mapa)

    aguas = st.session_state["e_aguas"]
    ativa = st.session_state["e_agua_ativa"]
    telhado_atual = st.session_state.get("e_telhado")

    # As águas que não estão sendo editadas entram apagadas, contorno e
    # módulos: continuam visíveis — é preciso ver o telhado inteiro para
    # decidir onde marcar a próxima — sem competir com a que recebe o clique.
    for indice, agua in enumerate(aguas):
        if indice == ativa:
            continue
        folium.GeoJson(
            agua["telhado"].as_dict()["geojson"],
            style_function=lambda _: {
                "color": "#a1a1aa", "weight": 1.5, "dashArray": "4,3",
                "fillOpacity": 0.05},
            name=agua["nome"],
        ).add_to(mapa)
        if agua.get("croqui"):
            folium.GeoJson(
                agua["croqui"], name=f"módulos · {agua['nome']}",
                style_function=lambda f: {
                    "color": "#94a3b8", "weight": 0.5,
                    "fillColor": "#94a3b8",
                    "fillOpacity": 0.45 if f["properties"].get("ativo", True) else 0.10},
            ).add_to(mapa)

    if telhado_atual is not None:
        folium.GeoJson(
            telhado_atual.as_dict()["geojson"],
            style_function=lambda _: {"color": "#f59e0b", "weight": 2, "fillOpacity": 0.10},
            name="telhado marcado",
        ).add_to(mapa)
        croqui = st.session_state.get("e_croqui")
        if croqui:
            # Desligado sai em cinza, e não some: um módulo que desaparece do
            # mapa ao ser removido não tem como voltar com um clique.
            folium.GeoJson(
                croqui, name="módulos",
                style_function=lambda f: (
                    {"color": "#1d4ed8", "weight": 0.6,
                     "fillColor": "#1d4ed8", "fillOpacity": 0.75}
                    if f["properties"].get("ativo", True)
                    else {"color": "#9ca3af", "weight": 0.6, "dashArray": "3,3",
                          "fillColor": "#9ca3af", "fillOpacity": 0.20}
                ),
            ).add_to(mapa)
        mapa.fit_bounds(folium.GeoJson(telhado_atual.as_dict()["geojson"]).get_bounds())

    # `center` e `zoom` como argumentos, e não só no `folium.Map`.
    #
    # O `location` do `folium.Map` decide a vista na **primeira** montagem do
    # componente e mais nenhuma: depois disso o `st_folium` guarda centro e
    # zoom do lado do navegador e os devolve a cada rerun. Trocar o endereço
    # movia o estado do Python e deixava o mapa onde estava — e o usuário
    # desenhava o telhado sobre a cidade errada sem nenhum aviso. Passados
    # como argumento, a vista é dirigida pelo Python.
    #
    # A chave continua carregando a coordenada porque as duas coisas se
    # somam: a chave descarta o desenho da localização anterior, que não tem
    # sentido na nova.
    zoom = 19 if not st.session_state["e_aguas"] else 18
    resultado = st_folium(
        mapa, height=460, width=None,
        center=(centro[0], centro[1]), zoom=zoom,
        key=f"mapa_telhado_{centro[0]:.5f}_{centro[1]:.5f}",
        returned_objects=["all_drawings", "last_active_drawing", "last_clicked"],
    )
    if telhado_atual is not None:
        _tratar_clique_no_arranjo(resultado)

    desenho = (resultado or {}).get("last_active_drawing")
    rotulo_botao = (
        "Medir e dimensionar esta água" if not st.session_state["e_aguas"]
        else f"Acrescentar esta água (já há {len(st.session_state['e_aguas'])})"
    )
    if desenho and st.button(rotulo_botao, type="primary", width="stretch"):
        try:
            telhado = telhado_de_geojson(
                desenho, nome=f"Água {len(st.session_state['e_aguas']) + 1}",
                montagem=montagem, inclinacao_deg=inclinacao, fator_obstaculos=obstaculos,
            )
        except ValueError as exc:
            st.error(str(exc))
            return
        try:
            base = carregar_base()
        except EquipmentError as exc:
            st.error(f"Catálogo de módulos indisponível: {exc}")
            return
        layout = dimensionar_no_telhado(telhado, base)
        if layout is None or layout.quantidade == 0:
            st.error(
                "Nenhum módulo do catálogo cabe nesse contorno. Ou o desenho ficou pequeno "
                "demais, ou o recuo de borda comeu a área toda."
            )
            return
        from .pv.edicao import EdicaoLayout, croqui_editavel

        st.session_state["e_aguas"].append({
            "nome": telhado.nome,
            "telhado": telhado,
            # O arranjo automático fica guardado inteiro: é sobre a numeração
            # dele que a edição manual se apoia, e é dele que sai a conta de
            # quanto a edição custou em potência.
            "layout_base": layout,
            "layout": layout,
            "edicao": EdicaoLayout(),
            "croqui": croqui_editavel(layout, telhado, None),
        })
        # A recém-marcada vira a ativa: é nela que o usuário vai querer mexer.
        _ativar_agua(len(st.session_state["e_aguas"]) - 1)
        st.session_state["e_geracao_aguas"] = None
        st.rerun()

    if not st.session_state["e_aguas"]:
        return

    _quadro_das_aguas()

    telhado_atual = st.session_state["e_telhado"]
    layout = st.session_state["e_layout"]
    colunas = st.columns(4)
    _cartao(colunas[0], "Área marcada", f"{_milhar(telhado_atual.area_m2)} m²")
    _cartao(colunas[1], "Módulos que cabem", f"{layout.quantidade}")
    _cartao(colunas[2], "Potência", f"{layout.potencia_kwp:.1f} kWp")
    _cartao(colunas[3], "Orientação", f"{telhado_atual.orientacao} ({telhado_atual.azimute_deg:.0f}°)")

    st.info(telhado_atual.sugerir_azimute()[1])
    if telhado_atual.montagem == "coplanar":
        opcoes = telhado_atual.azimutes_possiveis()
        if len(opcoes) > 1:
            escolha = st.radio(
                "A água do telhado aponta para:",
                opcoes, horizontal=True,
                index=opcoes.index(telhado_atual.azimute_deg) if telhado_atual.azimute_deg in opcoes else 0,
                format_func=lambda a: f"{_orientacao(a)} ({a:.0f}°)",
                key="azimute_agua",
            )
            if escolha != telhado_atual.azimute_deg:
                telhado_atual.azimute_deg = float(escolha)
                st.session_state["e_azimute"] = float(escolha)
                st.rerun()

    detalhes = st.columns(4)
    _cartao(detalhes[0], "Superfície real do telhado",
            f"{_milhar(telhado_atual.area_inclinada_m2)} m²")
    _cartao(detalhes[1], "Módulo escolhido",
            f"{layout.modulo.potencia_wp:.0f} Wp")
    _cartao(detalhes[2], "Ocupação do telhado", f"{layout.taxa_ocupacao:.0%}")
    _cartao(detalhes[3], "Densidade", f"{layout.densidade_wp_m2:.0f} Wp/m²")
    st.caption(
        f"Módulo: **{layout.modulo}** em {layout.orientacao_modulo}, "
        f"fileiras a {layout.azimute_fileiras_deg:.0f}°, passo de "
        f"{layout.passo_fileira_m:.2f} m. Área aproveitável já descontada em "
        f"{(1 - telhado_atual.fator_obstaculos):.0%} por obstáculos."
    )
    for aviso in [*telhado_atual.avisos, *layout.avisos]:
        st.markdown(f'<div class="aviso-suave">⚠️ {aviso}</div>', unsafe_allow_html=True)

    _editor_do_arranjo(telhado_atual)


def _ativar_agua(indice: int) -> None:
    """
    Espelha uma água nas chaves antigas do estado.

    O editor de arranjo, o clique no mapa e o recálculo foram escritos quando
    havia uma água só, e continuam falando com ``e_telhado``/``e_layout``.
    Espelhar em vez de reescrever os três é o que mantém a edição manual —
    a parte mais delicada da tela — sem tocar numa linha.
    """
    aguas = st.session_state["e_aguas"]
    if not aguas:
        return
    indice = max(0, min(int(indice), len(aguas) - 1))
    agua = aguas[indice]
    st.session_state.update({
        "e_agua_ativa": indice,
        "e_telhado": agua["telhado"],
        "e_layout": agua["layout"],
        "e_layout_base": agua["layout_base"],
        "e_edicao": agua["edicao"],
        "e_croqui": agua["croqui"],
        "e_inclinacao": float(agua["telhado"].inclinacao_deg),
        "e_azimute": float(agua["telhado"].azimute_deg),
        "e_azimute_fileiras": float(agua["layout"].azimute_fileiras_deg),
    })
    _somar_aguas()


def _guardar_agua_ativa() -> None:
    """O caminho de volta: o que a edição mudou volta para a lista."""
    aguas = st.session_state["e_aguas"]
    indice = st.session_state["e_agua_ativa"]
    if not aguas or not (0 <= indice < len(aguas)):
        return
    aguas[indice].update({
        "telhado": st.session_state["e_telhado"],
        "layout": st.session_state["e_layout"],
        "layout_base": st.session_state["e_layout_base"],
        "edicao": st.session_state["e_edicao"],
        "croqui": st.session_state["e_croqui"],
    })
    _somar_aguas()


def _somar_aguas() -> None:
    """A potência do sistema é a soma das águas, e não a da água em foco."""
    st.session_state["e_kwp"] = float(sum(
        float(getattr(a["layout"], "potencia_kwp", 0.0) or 0.0)
        for a in st.session_state["e_aguas"]
    ))


def _quadro_das_aguas() -> None:
    """
    A lista das águas, com a geração de cada uma.

    Duas águas opostas não somam a uma água média: a do nascente enche de
    manhã e a do poente à tarde, e o conjunto é mais plano que qualquer uma
    delas. A tabela mostra a produtividade de cada orientação justamente para
    que essa diferença apareça antes de virar um número só.
    """
    aguas = st.session_state["e_aguas"]
    ativa = st.session_state["e_agua_ativa"]
    geracao = st.session_state.get("e_geracao_aguas") or {}

    linhas = []
    for indice, agua in enumerate(aguas):
        telhado, layout = agua["telhado"], agua["layout"]
        chave = (round(float(telhado.azimute_deg), 1),
                 round(float(telhado.inclinacao_deg), 1))
        produtividade = geracao.get(chave)
        kwp = float(layout.potencia_kwp)
        linhas.append({
            "": "◉" if indice == ativa else "",
            "água": agua["nome"],
            "área (m²)": round(telhado.area_m2),
            "orientação": f"{telhado.orientacao} ({telhado.azimute_deg:.0f}°)",
            "inclinação": f"{telhado.inclinacao_deg:.0f}°",
            "módulos": layout.quantidade,
            "kWp": round(kwp, 2),
            "kWh/kWp·ano": round(produtividade) if produtividade else None,
            "kWh/ano": round(kwp * produtividade) if produtividade else None,
        })

    quadro = pd.DataFrame(linhas)
    st.markdown("##### As águas marcadas")
    st.dataframe(quadro, width="stretch", hide_index=True)

    total_kwp = sum(float(a["layout"].potencia_kwp) for a in aguas)
    total_kwh = sum(l["kWh/ano"] or 0 for l in linhas)
    colunas = st.columns(4)
    _cartao(colunas[0], "Águas", f"{len(aguas)}")
    _cartao(colunas[1], "Módulos", f"{sum(a['layout'].quantidade for a in aguas)}")
    _cartao(colunas[2], "Potência total", f"{total_kwp:.1f} kWp")
    _cartao(colunas[3], "Geração", f"{_milhar(total_kwh)} kWh/ano" if total_kwh else "—")

    acoes = st.columns([2, 2, 1])
    escolha = acoes[0].selectbox(
        "Água em edição", range(len(aguas)),
        index=min(ativa, len(aguas) - 1),
        format_func=lambda i: f"{aguas[i]['nome']} · {aguas[i]['layout'].potencia_kwp:.1f} kWp",
        key="agua_em_edicao",
    )
    if escolha != ativa:
        _ativar_agua(escolha)
        st.rerun()

    if acoes[1].button("Calcular a geração de cada água", width="stretch"):
        _calcular_geracao_das_aguas()
        st.rerun()

    if acoes[2].button("Remover", width="stretch",
                       help="Apaga a água em edição"):
        aguas.pop(ativa)
        st.session_state["e_geracao_aguas"] = None
        if aguas:
            _ativar_agua(min(ativa, len(aguas) - 1))
        else:
            st.session_state.update({
                "e_telhado": None, "e_layout": None, "e_layout_base": None,
                "e_edicao": None, "e_croqui": None, "e_kwp": 0.0,
                "e_agua_ativa": 0,
            })
        st.rerun()

    if total_kwh and len(aguas) > 1:
        st.caption(
            "A produtividade difere de uma água para a outra porque a orientação "
            "difere. O estudo não usa a média dos ângulos — busca uma série horária "
            "por orientação e as combina ponderadas pela potência, que é a única "
            "forma de a geração da manhã e a da tarde aparecerem nas horas certas."
        )


def _calcular_geracao_das_aguas() -> None:
    """
    Uma consulta por orientação distinta, e não uma por água.

    Duas águas com o mesmo par de ângulos têm a mesma série -- é a mesma
    consulta ao PVGIS, e o cache em disco é por par. Num galpão de seis águas
    em duas orientações isso é a diferença entre duas consultas e seis.
    """
    from .bateria.geracao import obter_serie_horaria

    aguas = st.session_state["e_aguas"]
    if not aguas:
        return
    pares = {
        (round(float(a["telhado"].azimute_deg), 1),
         round(float(a["telhado"].inclinacao_deg), 1))
        for a in aguas
    }
    resultado: dict[tuple[float, float], float] = {}
    with st.spinner(f"Buscando a série horária de {len(pares)} orientação(ões)…"):
        for azimute, inclinacao in pares:
            try:
                serie = obter_serie_horaria(
                    st.session_state["e_lat"], st.session_state["e_lon"],
                    azimute_deg=azimute, inclinacao_deg=inclinacao,
                    permitir_fallback=not st.session_state.get("e_exigir_pvgis", False),
                )
            except Exception as exc:  # noqa: BLE001 — a tela não pode cair por isso
                st.warning(f"Não foi possível obter a geração de {azimute:.0f}°: {exc}")
                continue
            resultado[(azimute, inclinacao)] = serie.anual_kwh_por_kwp()
    st.session_state["e_geracao_aguas"] = resultado


def _recalcular_arranjo() -> None:
    """
    Reaplica a edição sobre o arranjo automático e atualiza o que dela depende.

    Um lugar só para isso porque são quatro coisas que precisam andar juntas —
    arranjo, croqui, potência e o que o mapa desenha — e esquecer uma delas
    produz uma tela que mostra 84 módulos e um estudo que dimensiona 90.
    """
    from .pv.edicao import aplicar_edicao, croqui_editavel

    base = st.session_state.get("e_layout_base")
    telhado = st.session_state.get("e_telhado")
    edicao = st.session_state.get("e_edicao")
    if base is None or telhado is None:
        return
    layout = aplicar_edicao(base, telhado, edicao)
    st.session_state["e_layout"] = layout
    st.session_state["e_croqui"] = croqui_editavel(base, telhado, edicao)
    # A potência do sistema é a soma das águas — `_guardar_agua_ativa` devolve
    # esta água à lista e refaz a soma. Escrever `e_kwp` aqui direto, como era
    # quando havia uma água só, apagaria as outras do total.
    _guardar_agua_ativa()


def _tratar_clique_no_arranjo(resultado: dict | None) -> None:
    """
    Um clique no mapa vira uma operação de edição sobre o arranjo.

    O ``st_folium`` devolve o último ponto clicado, e não o objeto — então quem
    decide o que foi clicado é a geometria: se o ponto cai dentro de um módulo,
    o modo "Remover" o desliga; se cai no vazio, o modo "Acrescentar" põe um
    módulo ali. É a interação mais direta que o Streamlit permite, e cobre as
    duas operações que aparecem na prática.

    O mesmo clique volta em toda re-execução do script, então ele é guardado e
    tratado uma vez só — sem isso, um clique num módulo o ligaria e desligaria
    alternadamente a cada rerun, sem ninguém tocar em nada.
    """
    from .pv.edicao import indice_no_ponto, ponto_para_utm

    clique = (resultado or {}).get("last_clicked")
    if not clique:
        return
    chave = (round(float(clique["lat"]), 8), round(float(clique["lng"]), 8))
    if st.session_state.get("e_ultimo_clique") == chave:
        return
    st.session_state["e_ultimo_clique"] = chave

    base = st.session_state.get("e_layout_base")
    edicao = st.session_state.get("e_edicao")
    if base is None or edicao is None:
        return

    modo = st.session_state.get("e_modo_edicao", "Remover")
    if modo == "Remover":
        indice = indice_no_ponto(base, chave[0], chave[1], edicao)
        if indice is None:
            return
        edicao.alternar(indice)
    elif modo == "Acrescentar":
        edicao.extras_utm.append(ponto_para_utm(base, chave[0], chave[1]))
    else:
        return
    _recalcular_arranjo()
    st.rerun()


def _editor_do_arranjo(telhado) -> None:
    """
    Os controles que transformam o empacotamento automático em projeto.

    A grade não sabe onde fica a caixa d'água nem por onde se anda para fazer
    manutenção. Quem sabe está olhando a imagem — e aqui tem como dizer.
    """
    from .pv.edicao import EdicaoLayout

    base = st.session_state.get("e_layout_base")
    if base is None:
        return
    edicao: EdicaoLayout = st.session_state.get("e_edicao") or EdicaoLayout()
    st.session_state["e_edicao"] = edicao
    layout = st.session_state["e_layout"]

    st.markdown("##### Ajustar o arranjo")
    st.caption(
        "O empacotamento automático é o ponto de partida. Clique nos módulos do mapa "
        "para tirar ou pôr, desloque a grade para alinhá-la com o telhado real, e gire "
        "as fileiras quando a orientação sugerida não for a da água."
    )

    colunas = st.columns([1.4, 1, 1, 1])
    opcoes = ["Remover", "Acrescentar", "Nada"]
    st.session_state["e_modo_edicao"] = colunas[0].radio(
        "O clique no mapa",
        opcoes,
        index=opcoes.index(st.session_state.get("e_modo_edicao", "Remover")),
        help="Com 'Remover', clicar num módulo o desliga — ele fica cinza e volta com "
             "outro clique. Com 'Acrescentar', clicar num vazio põe um módulo ali, se "
             "couber sem invadir a borda nem outro painel. Use 'Nada' para navegar no "
             "mapa sem alterar o arranjo.",
    )
    leste = colunas[1].slider(
        "Deslocar para leste (m)", -20.0, 20.0, float(edicao.deslocamento_leste_m), 0.25,
        help="Positivo vai para leste. Módulos empurrados para fora do telhado saem do "
             "arranjo, e aparecem em cinza antes de sair.")
    norte = colunas[2].slider(
        "Deslocar para norte (m)", -20.0, 20.0, float(edicao.deslocamento_norte_m), 0.25)
    azimute_fileiras = colunas[3].slider(
        "Ângulo das fileiras (°)", 0.0, 179.0,
        float(st.session_state.get("e_azimute_fileiras") or base.azimute_fileiras_deg), 1.0,
        help="Direção das fileiras, medida do Norte. Em telhado de duas águas é a "
             "cumeeira; em laje plana, 90° mantém as faces voltadas ao Norte.")

    if (float(leste), float(norte)) != (
        float(edicao.deslocamento_leste_m), float(edicao.deslocamento_norte_m)
    ):
        edicao.deslocamento_leste_m = float(leste)
        edicao.deslocamento_norte_m = float(norte)
        _recalcular_arranjo()
        st.rerun()

    # A comparação é contra a quantidade FINAL do automático, e não contra a
    # geométrica: a geométrica é anterior ao desconto de obstáculos, e usá-la
    # aqui mostrava "-112" num arranjo em que ninguém tinha tocado.
    cartoes = st.columns(4)
    _cartao(cartoes[0], "Módulos no arranjo", f"{layout.quantidade}")
    _cartao(cartoes[1], "O automático propôs", f"{base.quantidade}")
    _cartao(cartoes[2], "Diferença", f"{layout.quantidade - base.quantidade:+d}")
    _cartao(cartoes[3], "Potência", f"{layout.potencia_kwp:.1f} kWp")
    st.caption(f"Edições: {edicao.resumo()}.")

    # A inclinação entra aqui, e não só na hora de desenhar, porque ela muda
    # duas coisas de uma vez: o quanto o sol rende no ano e o espaçamento entre
    # fileiras numa laje. Ter de apagar o desenho para experimentar 15° contra
    # 25° tornava a comparação cara demais para alguém fazer.
    inclinacao = st.slider(
        "Inclinação dos módulos (°)", 0.0, 45.0, float(telhado.inclinacao_deg), 1.0,
        help="Coplanar: é a inclinação da própria água do telhado. Laje plana: é o "
             "ângulo da estrutura — mais inclinação rende mais no inverno e afasta as "
             "fileiras, tirando módulos do telhado.",
    )
    mudou_inclinacao = abs(float(inclinacao) - float(telhado.inclinacao_deg)) > 0.5
    mudou_fileiras = abs(float(azimute_fileiras) - float(base.azimute_fileiras_deg)) > 0.5

    if mudou_inclinacao or mudou_fileiras:
        mudancas = []
        if mudou_inclinacao:
            mudancas.append(
                f"inclinação de {telhado.inclinacao_deg:.0f}° para {inclinacao:.0f}°"
            )
        if mudou_fileiras:
            mudancas.append(
                f"fileiras de {base.azimute_fileiras_deg:.0f}° para {azimute_fileiras:.0f}°"
            )
        st.warning(
            "Mudança pedida: " + "; ".join(mudancas) + ". Reempacotar refaz a grade do "
            "zero e **descarta as remoções e os acréscimos manuais** — a numeração dos "
            "módulos deixa de valer."
        )
        if st.button("Reempacotar com os novos ângulos", width="stretch"):
            _reempacotar(telhado, float(azimute_fileiras), float(inclinacao))

    if not edicao.vazia and st.button("Desfazer todas as edições", width="stretch"):
        st.session_state["e_edicao"] = EdicaoLayout()
        _recalcular_arranjo()
        st.rerun()


def _reempacotar(telhado, azimute_fileiras: float, inclinacao_deg: float) -> None:
    """
    Refaz o empacotamento com outros ângulos.

    Girar a grade ou mudar a inclinação não é uma edição sobre o resultado: é
    um resultado novo. Por isso a função troca o arranjo base e zera a edição,
    em vez de tentar preservar índices que deixaram de existir.

    A inclinação é gravada no telhado, e não só no arranjo: é ela que vai para
    o PVGIS pedir a série de geração, e deixar as duas divergirem produziria um
    croqui de 25° com uma produtividade calculada a 20°.
    """
    from .pv.edicao import EdicaoLayout, croqui_editavel
    from .pv.layout import calcular_layout

    telhado.inclinacao_deg = float(inclinacao_deg)
    novo = calcular_layout(
        telhado.poligono_utm, st.session_state["e_layout"].modulo,
        montagem=telhado.montagem,
        tilt_deg=float(inclinacao_deg),
        azimute_fileiras_deg=float(azimute_fileiras),
        utm_epsg=telhado.utm_epsg,
        fator_obstaculos=telhado.fator_obstaculos,
    )
    if novo.quantidade == 0:
        st.error(
            f"Com as fileiras a {azimute_fileiras:.0f}° e {inclinacao_deg:.0f}° de "
            "inclinação, nenhum módulo cabe no contorno. Os ângulos anteriores foram "
            "mantidos."
        )
        return
    st.session_state.update({
        "e_layout_base": novo,
        "e_layout": novo,
        "e_edicao": EdicaoLayout(),
        "e_croqui": croqui_editavel(novo, telhado, None),
        "e_kwp": float(novo.potencia_kwp),
        "e_azimute_fileiras": float(azimute_fileiras),
        "e_inclinacao": float(inclinacao_deg),
    })
    st.rerun()


def _orientacao(azimute: float) -> str:
    from .pv.solar import nome_orientacao

    return nome_orientacao(azimute)


# ============================================================================
# Passo 5 — meta e execução
# ============================================================================
def _passo_meta() -> None:
    colunas = st.columns(2)
    with colunas[0]:
        st.markdown("##### Quanto tempo sem luz?")
        autonomia = st.select_slider(
            "Autonomia alvo", options=[1, 2, 3, 6, 12, 24, 36], value=12,
            format_func=lambda h: f"{h} h", label_visibility="collapsed",
        )
        st.caption(
            {1: "Queda rápida — o suficiente para não perder o que estava rodando.",
             2: "Interrupção curta, o caso mais comum na área urbana.",
             3: "Interrupção curta com folga.",
             6: "Uma tarde inteira sem rede.",
             12: "Uma noite inteira, ou um dia inteiro de trabalho.",
             24: "Um dia completo — exige o sol recarregando no meio.",
             36: "Um dia e meio: atravessa duas noites, e é a segunda que decide."}[autonomia]
        )
    with colunas[1]:
        st.markdown("##### Com que garantia?")
        confiabilidade = st.select_slider(
            "Confiabilidade", options=[0.80, 0.90, 0.95, 0.99], value=0.95,
            format_func=lambda p: f"{p:.0%}", label_visibility="collapsed",
        )
        st.caption(
            f"O conjunto tem que atravessar em **{confiabilidade:.0%}** dos casos — "
            "e no par mais desfavorável de estação e horário, não na média."
        )

    st.divider()
    st.markdown("##### Quais fontes entram na comparação?")
    st.caption(
        "O estudo monta um cenário para cada combinação e mede todos contra a conta "
        "que o cliente paga hoje. Desmarque o que ele já descartou — quatro linhas "
        "verdadeiras valem mais que oito com metade irrelevante."
    )
    # Solar não é caixa aqui: foi perguntado no passo 1, onde a resposta ainda
    # pode podar o fluxo. Repetir a pergunta no fim deixaria o usuário
    # desmarcar depois de ter marcado o telhado no mapa.
    com_solar = bool(st.session_state.get("e_com_solar", True))
    if not com_solar:
        st.caption(
            "Este estudo não tem energia solar — definido no passo 1. Os cenários "
            "com solar não são gerados."
        )
    colunas = st.columns(2)
    com_bateria = colunas[0].checkbox(
        "Bateria", value=True,
        help="Dá backup. Na tarifa simples economiza pouco — o ganho é recuperar o "
             "que o Fio B retém da energia injetada.")
    com_gerador = colunas[1].checkbox(
        "Grupo gerador", value=False,
        help="Energia praticamente ilimitada, potência limitada, custo por kWh que "
             "só aparece quando ele roda.")

    gerador = None
    if com_gerador:
        colunas = st.columns(4)
        potencia_grupo = colunas[0].number_input("Potência do grupo (kW)", 1.0, 2000.0, 40.0, 1.0)
        custo_grupo = colunas[1].number_input(
            "Custo do kWh gerado (R$)", 0.10, 20.0, 2.30, 0.05,
            help="Combustível, lubrificante e manutenção por kWh gerado.")
        capex_grupo = colunas[2].number_input(
            "Investimento no grupo (R$)", 0.0, 5_000_000.0, 52_000.0, 1_000.0)
        opex_grupo = colunas[3].number_input(
            "Manutenção fixa (R$/ano)", 0.0, 500_000.0, 4_000.0, 500.0,
            help="Ensaios mensais, troca de óleo e filtros — existe mesmo sem apagão.")
        partida = st.slider(
            "Tempo até o grupo assumir a carga (s)", 0, 120, 30, 5,
            help="Num grupo com QTA são 10 a 30 s. O que acontece nesse intervalo "
                 "depende de haver ou não bateria.")
        gerador = Gerador(
            potencia_kw=float(potencia_grupo),
            custo_energia_brl_kwh=float(custo_grupo),
            capex_brl=float(capex_grupo),
            opex_fixo_brl_ano=float(opex_grupo),
            atraso_partida_min=float(partida) / 60.0,
        )

    st.markdown("##### A conta de luz do cliente")
    st.caption(
        "Tarifa simples (convencional, Grupo B). É contra essa conta que a economia "
        "de cada cenário é medida."
    )
    colunas = st.columns(3)
    tarifa_simples = colunas[0].number_input(
        "Tarifa (R$/kWh)", 0.10, 5.0, 0.98, 0.01,
        help="A tarifa cheia da fatura, com impostos.")
    consumo_fatura = colunas[1].number_input(
        "Consumo médio na fatura (kWh/mês)", 0.0, 500_000.0, 0.0, 50.0,
        help="Zero usa o consumo que a simulação de cargas produziu. Preenchido, a "
             "simulação é escalada para bater com a fatura — ela é medida, o "
             "levantamento é estimativa.")
    disponibilidade = colunas[2].selectbox(
        "Custo de disponibilidade", [30, 50, 100], index=2,
        format_func=lambda k: {30: "30 kWh (monofásico)", 50: "50 kWh (bifásico)",
                               100: "100 kWh (trifásico)"}[k],
        help="É o piso da conta: nenhum sistema leva a fatura a zero.")

    fatura = Fatura(
        tarifa_brl_kwh=float(tarifa_simples),
        consumo_mensal_kwh=float(consumo_fatura) or None,
        custo_disponibilidade_kwh=float(disponibilidade),
    )

    colunas = st.columns([1.2, 1])
    padrao_capex = colunas[0].selectbox(
        "Padrão da obra (define o R$/kWp do sistema solar)",
        ["basico", "padrao", "alto"],
        index=1,
        format_func=lambda k: {
            "basico": "Básico — telhado metálico, acesso livre",
            "padrao": "Padrão — o caso médio do mercado",
            "alto": "Alto padrão — prédio ocupado, projeto executivo",
        }[k],
        help="O mesmo kWp custa coisas muito diferentes conforme a obra. O que muda "
             "não é o módulo: é estrutura, acabamento, prazo em prédio ocupado, ART, "
             "seguro e exigência de projeto. O nível 'alto' foi calibrado numa obra "
             "real de 312 kWp fechada a R$ 4.104/kWp.",
    )
    capex_informado = colunas[1].number_input(
        "Ou informe o investimento no solar (R$)", 0.0, 100_000_000.0, 0.0, 10_000.0,
        help="Zero usa a curva do padrão escolhido. Cotação de verdade passa na "
             "frente de qualquer curva.",
    )
    topologia_kit, mao_de_obra, material_ca = _controles_do_kit()

    referencia = _referencia_capex(
        padrao_capex, st.session_state.get("e_kwp") or 0.0,
        topologia_kit, mao_de_obra, material_ca,
    )
    if referencia:
        colunas[0].caption(referencia)

    with st.expander("Para calcular o retorno financeiro (opcional)"):
        st.caption(
            "Sem preencher, o estudo entrega a engenharia e deixa o dinheiro de fora. "
            "O campo que mais muda a conta é o último."
        )
        colunas = st.columns(3)
        tarifa = colunas[0].number_input("Tarifa fora de ponta (R$/kWh)", 0.10, 5.0, 0.78, 0.01)
        tarifa_ponta = colunas[1].number_input("Tarifa de ponta (R$/kWh)", 0.10, 8.0, 1.35, 0.01)
        tarifa_demanda = colunas[2].number_input(
            "Demanda contratada (R$/kW/mês)", 0.0, 200.0, 0.0, 1.0,
            help="Zero para clientes do Grupo B (baixa tensão).")
        # O campo nascia em zero, e zero afirma que ficar sem energia não custa
        # nada — o que nenhuma instalação com quadro de backup acredita. Com o
        # padrão em zero, o valor da resiliência saía zero em todo estudo, e a
        # bateria aparecia sempre como investimento sem retorno. A sugestão por
        # segmento põe a conversa no lugar: o número final vem do cliente, mas
        # a pergunta agora é "esse valor está certo?" em vez de "que valor é
        # esse?".
        sugerido = float(MODELOS[st.session_state["e_segmento"]].custo_interrupcao_brl_kwh)
        custo_interrupcao = st.number_input(
            "Quanto custa ao cliente 1 kWh que faltou (R$)", 0.0, 1000.0, sugerido, 1.0,
            help="Câmara fria parada, hotel sem elevador, produção interrompida. É o "
                 "número que transforma resiliência em dinheiro, e só o cliente sabe. O "
                 "valor sugerido é a ordem de grandeza do segmento escolhido — confirme "
                 "antes de propor.",
        )
        if custo_interrupcao == 0:
            st.warning(
                "Com zero aqui, o valor da resiliência não entra em nenhum payback: o "
                "estudo passa a comparar só economia de conta, e a bateria aparece pior "
                "do que é para quem não pode ficar sem energia."
            )
        elif abs(custo_interrupcao - sugerido) < 0.01:
            st.caption(
                f"Sugestão para **{MODELOS[st.session_state['e_segmento']].nome}**: "
                f"R$ {sugerido:.0f}/kWh. É ordem de grandeza, não cotação — o número "
                "que vale é o que o cliente disser."
            )
        colunas = st.columns(2)
        interrupcoes = colunas[0].number_input(
            "Interrupções por ano na região", 0.0, 200.0, 8.0, 1.0,
            help="Use o FEC do alimentador quando houver.")
        anos = colunas[1].number_input("Horizonte da análise (anos)", 5, 30, 15)

    with st.expander("Ajustes avançados da varredura"):
        colunas = st.columns(3)
        amostras = colunas[0].slider("Amostras por combinação", 20, 500, 150, 10)
        passo = colunas[1].selectbox("Passo do despacho (min)", [1, 5, 10, 15], index=1)
        candidatos = colunas[2].slider("Conjuntos a avaliar", 2, 20, 8)
        duracoes = st.multiselect(
            "Durações de falta a simular (h)",
            [1.0, 2.0, 3.0, 6.0, 12.0, 24.0, 36.0, 48.0],
            default=[1.0, 2.0, 3.0, 6.0, 12.0, 24.0, 36.0],
        )
        reserva = st.slider(
            "Reserva de backup (fração do banco que não é ciclada)", 0.0, 0.9, 0.30, 0.05,
            help="Reserva alta dá resiliência e mata a arbitragem tarifária; zero faz o contrário.")

    if float(autonomia) not in duracoes:
        duracoes = sorted({*duracoes, float(autonomia)})

    cenarios = len(duracoes) * 24 * 4 * amostras
    st.caption(
        f"A varredura testa **{_milhar(cenarios)} cenários** por conjunto candidato "
        f"({len(duracoes)} durações × 24 horas de início × 4 estações × {amostras} sorteios). "
        f"Leva de 10 a 60 segundos."
    )

    def _rodar() -> bool:
        cfg = _montar_configuracao(
            autonomia, confiabilidade, duracoes, amostras, passo, candidatos,
            tarifa, tarifa_ponta, tarifa_demanda, custo_interrupcao, interrupcoes,
            anos, reserva,
            fatura=fatura, gerador=gerador,
            considerar_solar=com_solar, considerar_bateria=com_bateria,
            considerar_gerador=com_gerador,
            padrao_capex=padrao_capex,
            topologia_kit=topologia_kit,
            mao_de_obra_brl_kwp=mao_de_obra,
            material_ca_brl_kwp=material_ca,
            capex_fv_brl=float(capex_informado) or None,
        )
        barra = st.progress(0.0, text="Preparando…")
        try:
            st.session_state["e_estudo"] = executar_estudo(
                cfg, progresso=lambda m, f: barra.progress(min(1.0, f), text=m))
        except Exception as exc:  # noqa: BLE001 — a interface mostra a mensagem, não o traceback
            barra.empty()
            st.error(f"O estudo não pôde ser concluído: {exc}")
            return False
        barra.empty()
        st.session_state["e_pacote"] = None
        return True

    _rodape(rotulo_avancar="Rodar o estudo →", ao_avancar=_rodar)


def _aba_cenarios(estudo) -> None:
    """
    O quadro de cenários: com e sem cada fonte, contra a conta de hoje.

    Vem como primeira aba porque é a pergunta que antecede todas as outras. As
    abas seguintes detalham o arranjo escolhido; esta é a que diz se vale a
    pena escolher algum.
    """
    comparacao = estudo.cenarios
    if comparacao is None:
        st.info(
            "A comparação de cenários precisa de pelo menos uma fonte além da rede: "
            "marque bateria ou gerador no passo anterior e rode de novo."
        )
        return

    base = comparacao.base
    colunas = st.columns(3)
    _cartao(colunas[0], "Conta de hoje", f"{_reais(base.conta_anual_brl)}/ano")
    melhor = comparacao.melhor_vpl
    if melhor is not None:
        _cartao(colunas[1], "Maior valor presente", melhor.nome.replace("Rede + ", ""))
        _cartao(colunas[2], "VPL desse arranjo", _reais(melhor.vpl_brl))

    quadro = pd.DataFrame([
        {
            "arranjo": c.nome.replace("Rede + ", ""),
            "investimento (R$)": round(c.capex_brl),
            "conta (R$/ano)": round(c.conta_anual_brl),
            "economia (R$/ano)": round(c.economia_anual_brl),
            "resiliência (R$/ano)": round(c.valor_resiliencia_brl_ano),
            "autonomia (h)": c.autonomia_garantida_h,
            "sem energia (kWh)": round(c.ens_por_evento_kwh, 1),
            "VPL (R$)": round(c.vpl_brl),
            "payback (anos)": round(c.payback_anos, 1) if c.payback_anos else None,
        }
        for c in comparacao.cenarios
    ])
    st.dataframe(quadro, width="stretch", hide_index=True)
    st.caption(
        "`autonomia` é a maior falta atravessada em 95% dos casos no pior par de "
        "estação e hora — ela empata em zero sempre que nenhum arranjo cumpre isso. "
        "`sem energia` é o que falta à carga essencial numa interrupção média, e "
        "separa os arranjos mesmo quando a autonomia empata."
    )

    st.markdown("##### O que cada fonte resolve")
    st.bar_chart(
        quadro.set_index("arranjo")[["economia (R$/ano)", "resiliência (R$/ano)"]],
        height=280,
    )
    st.caption(
        "Solar aparece na coluna da esquerda e não na da direita; bateria e gerador, "
        "o contrário. É a diferença que decide a compra, e ela some quando os dois "
        "benefícios são somados num número só."
    )

    for aviso in comparacao.avisos:
        st.warning(aviso)


def _controles_do_kit() -> tuple[str | None, float, float]:
    """
    A topologia do kit e as duas parcelas que a obra acrescenta.

    Abaixo de 40 kWp o preço do equipamento não sai de curva de escala: sai da
    tabela do distribuidor, e a coluna importa tanto quanto a linha — um kit
    split-phase custa mais de 50% acima de um mono da mesma potência.

    A mão de obra e o material CA entram **somados por kWp**, e não como
    percentual sobre o kit: a equipe leva o mesmo tempo para instalar os dois
    kits, e um percentual cobraria mais caro pela instalação só porque o
    equipamento é mais caro.
    """
    from .pv.kits import (
        BATERIA_BLOCO_BRL,
        BATERIA_BLOCO_KWH,
        DESCRICAO_TOPOLOGIA,
        MAO_DE_OBRA_BRL_KWP,
        MATERIAL_CA_BRL_KWP,
        POTENCIA_MAXIMA_KWP,
        TOPOLOGIAS,
        potencia_maxima,
        topologia_para_rede,
    )

    kwp = float(st.session_state.get("e_kwp") or 0.0)
    sugerida = topologia_para_rede(float(st.session_state.get("e_tensao_rede") or 380.0))
    chaves = ["", *TOPOLOGIAS]

    # Com `key=`, o Streamlit lê o estado e ignora o `value=` do widget. Como
    # o estado nasce vazio, o número entrava como None e o widget quebrava na
    # primeira abertura. Semear antes de desenhar é o caminho previsto — e
    # mantém a constante viva num lugar só, em `aurum.pv.kits`.
    for chave, padrao in (("e_mao_de_obra", MAO_DE_OBRA_BRL_KWP),
                          ("e_material_ca", MATERIAL_CA_BRL_KWP)):
        if st.session_state.get(chave) is None:
            st.session_state[chave] = padrao

    with st.expander(
        f"Preço de kit e obra (a tabela vai até {POTENCIA_MAXIMA_KWP:.0f} kWp)",
        expanded=0 < kwp <= POTENCIA_MAXIMA_KWP,
    ):
        st.caption(
            "A tabela de kit é **equipamento posto**: não traz mão de obra, projeto, "
            "ART, homologação nem o material do lado CA. Os dois campos abaixo somam "
            "isso ao preço do kit, por kWp — é o que transforma material em usina "
            "ligada."
        )
        colunas = st.columns([1.4, 1, 1])
        topologia = colunas[0].selectbox(
            "Topologia do kit",
            chaves,
            index=chaves.index(sugerida) if sugerida in chaves else 0,
            format_func=lambda c: (
                "Não usar a tabela — estimar pela curva de R$/kWp" if not c
                else f"{TOPOLOGIAS[c]} — até {potencia_maxima(c):.0f} kWp"
            ),
            key="e_topologia_kit",
            help="A sugestão vem da tensão da rede: 380 V pede trifásico; 220 V, "
                 "split-phase ou mono/bifásico. Cada coluna tem o seu alcance — "
                 "acima de 40 kWp a entrada é trifásica e só o trifásico continua, "
                 "até 125 kWp. Fora da tabela, a curva volta a valer sozinha.",
        )
        mao_de_obra = colunas[1].number_input(
            "Mão de obra (R$/kWp)", 0.0, 5_000.0, step=25.0,
            key="e_mao_de_obra",
            help="Equipe, estrutura fora do kit, projeto, ART e homologação.",
        )
        material_ca = colunas[2].number_input(
            "Material CA (R$/kWp)", 0.0, 5_000.0, step=25.0,
            key="e_material_ca",
            help="Cabo CA até o quadro, disjuntores, DPS, eletroduto e aterramento. "
                 "Inversor longe do padrão de entrada sobe este número.",
        )
        if topologia:
            st.caption(DESCRICAO_TOPOLOGIA.get(topologia, ""))

        # A bateria entra em bloco, e não em R$/kWh contínuo: é assim que se
        # compra, e é o que a própria tabela mostra — a coluna "Split + 5kWh"
        # é o kit split-phase mais um bloco.
        bloco = st.columns([1, 1, 2])
        st.session_state["e_bateria_bloco_kwh"] = bloco[0].number_input(
            "Bloco de bateria (kWh)", 1.0, 50.0, step=0.5,
            key="e_bateria_bloco_kwh_w",
            value=float(st.session_state.get("e_bateria_bloco_kwh") or BATERIA_BLOCO_KWH),
        )
        st.session_state["e_bateria_bloco_brl"] = bloco[1].number_input(
            "Preço do bloco (R$)", 0.0, 200_000.0, step=500.0,
            key="e_bateria_bloco_brl_w",
            value=float(st.session_state.get("e_bateria_bloco_brl") or BATERIA_BLOCO_BRL),
        )
        bloco[2].caption(
            "A cada bloco, mais autonomia e mais conforto. O inversor híbrido já "
            "vem no kit split-phase e **não** é cobrado de novo aqui — é por isso "
            "que o preço do armazenamento é o do bloco, e não o de um sistema "
            "inteiro."
        )

        _mostrar_composicao_do_kit(topologia, kwp, mao_de_obra, material_ca)

    return (topologia or None), float(mao_de_obra), float(material_ca)


def _mostrar_composicao_do_kit(
    topologia: str, kwp: float, mao_de_obra: float, material_ca: float,
) -> None:
    """As três parcelas à vista, para o número não ter que ser aceito de fé."""
    from .pv.kits import (
        TOPOLOGIAS,
        composicao_de_kit,
        potencia_maxima,
        preco_kit_com_bateria,
    )

    if not topologia or kwp <= 0:
        return
    partes = composicao_de_kit(
        kwp, topologia, mao_de_obra, material_ca,
        bateria_kwh=float(st.session_state.get("e_energia_util_kwh") or 0.0),
        bloco_kwh=float(st.session_state.get("e_bateria_bloco_kwh") or 5.0),
        bloco_brl=float(st.session_state.get("e_bateria_bloco_brl") or 10_000.0),
    )
    if partes is None:
        st.caption(
            f"A tabela não cobre {kwp:.1f} kWp em {TOPOLOGIAS.get(topologia, topologia)} "
            f"— essa coluna vai até {potencia_maxima(topologia):.0f} kWp. O "
            "investimento sai da curva do padrão de obra escolhido."
        )
        return

    total = sum(partes.values())
    rotulos = {"kit": "Kit (equipamento)", "mao_de_obra": "Mão de obra",
               "material_ca": "Material CA", "bateria": "Banco (blocos)"}
    colunas = st.columns(len(partes) + 1)
    for i, (chave, valor) in enumerate(partes.items()):
        _cartao(colunas[i], rotulos.get(chave, chave), _reais(valor))
    _cartao(colunas[-1], "Investimento", f"{_reais(total)} · {_reais(total / kwp)}/kWp")

    com_bateria = preco_kit_com_bateria(kwp)
    if com_bateria:
        st.caption(
            f"Conferência: o kit split-phase com 5 kWh de bateria embutidos custa "
            f"{_reais(com_bateria)} nessa potência. O estudo **não** usa esse "
            "número — ele dimensiona e precifica o armazenamento à parte, e somar "
            "os dois cobraria a bateria duas vezes. Serve para comparar: se o "
            "solar mais o banco do estudo saírem muito acima disto, ou o banco "
            "ficou grande, ou o preço de armazenamento está velho."
        )


def _referencia_capex(
    padrao: str, kwp: float,
    topologia: str | None = None,
    mao_de_obra: float | None = None,
    material_ca: float | None = None,
) -> str:
    """
    Mostra o R$/kWp que a escolha produz nessa potência.

    A curva tem ganho de escala, então a referência de tabela (medida em
    100 kWp) não é o número que vai sair. Esconder isso faria o usuário
    escolher "alto padrão" esperando R$ 4.705/kWp e receber R$ 4.105.

    Com uma topologia de kit escolhida e potência dentro da tabela, quem manda
    é a tabela, e o padrão de obra deixa de valer — dizer isso aqui evita que
    o usuário mexa no seletor de padrão e não veja número nenhum mudar.
    """
    from .pv.financials import estimar_capex

    if kwp <= 0:
        return ""
    total = estimar_capex(
        kwp, padrao=padrao, topologia=topologia,
        mao_de_obra_brl_kwp=mao_de_obra, material_ca_brl_kwp=material_ca,
    )
    if topologia:
        from .pv.kits import capex_de_kit

        if capex_de_kit(kwp, topologia, mao_de_obra or 0.0, material_ca or 0.0) is not None:
            return (
                f"Nessa potência ({kwp:.1f} kWp) o preço vem da **tabela de kit**, "
                f"não do padrão de obra: {_reais(total / kwp)}/kWp, "
                f"{_reais(total)} no total."
            )
    return (
        f"Nessa potência ({kwp:.0f} kWp): {_reais(total / kwp)}/kWp, "
        f"{_reais(total)} no total."
    )


def _montar_configuracao(
    autonomia, confiabilidade, duracoes, amostras, passo, candidatos,
    tarifa, tarifa_ponta, tarifa_demanda, custo_interrupcao, interrupcoes, anos, reserva,
    fatura: Fatura | None = None,
    gerador: Gerador | None = None,
    considerar_solar: bool = True,
    considerar_bateria: bool = True,
    considerar_gerador: bool = False,
    padrao_capex: str = "padrao",
    topologia_kit: str | None = None,
    mao_de_obra_brl_kwp: float | None = None,
    material_ca_brl_kwp: float | None = None,
    capex_fv_brl: float | None = None,
) -> ConfiguracaoEstudo:
    estado = st.session_state
    levantado: Cenario | None = estado["e_cenario"]
    # O cenário reescrito pelo perfil de ocupação, quando houve um. O levantado
    # continua sendo o dado: é ele que vai para o anexo de cargas.
    cenario: Cenario | None = estado.get("e_cenario_ocupacao") or levantado
    comodos = cenario.para_comodos() if cenario else None

    # O quadro de backup por equipamento, quando a vistoria classificou. Sem
    # isto o estudo volta a recortar por ambiente, e a geladeira crítica
    # arrasta o forno de 4 kW junto.
    comodos_backup = instancias_backup = criticidade = None
    if cenario is not None and cenario.tem_criticidade:
        from .demanda.vistoria import DESCRICAO_CRITICIDADE

        comodos_backup = cenario.para_comodos(True)
        instancias_backup = cenario.instancias_de(True)
        criticidade = {
            "tabela": (levantado or cenario).por_criticidade(),
            "corte": cenario.criticidades_essenciais,
            "descricoes": DESCRICAO_CRITICIDADE,
        }

    ocupacao_dados = estado.get("e_ocupacao")
    consumo_anual = None
    if ocupacao_dados and ocupacao_dados.get("diaria_ponderada_kwh"):
        consumo_anual = float(ocupacao_dados["diaria_ponderada_kwh"]) * 365.0

    return ConfiguracaoEstudo(
        latitude=estado["e_lat"], longitude=estado["e_lon"],
        nome=estado["e_nome"] or "instalação",
        tensao_rede_v=float(estado.get("e_tensao_rede") or 380.0),
        comodos=comodos,
        instancias_por_comodo=cenario.instancias_de() if cenario else None,
        comodos_essenciais=estado["e_essenciais"] or None,
        comodos_backup=comodos_backup,
        instancias_backup=instancias_backup,
        criticidade=criticidade,
        # Ligado sozinho quando a vistoria classificou: se o levantamento
        # separou o preferível do crítico, foi para que alguém decidisse entre
        # os dois, e perguntar de novo na tela seria pedir a mesma informação
        # duas vezes.
        comparar_escopos=criticidade is not None,
        tabelas_cenario=(levantado or cenario).comodos if cenario else None,
        ocupacao=ocupacao_dados,
        consumo_anual_kwh=consumo_anual,
        simulacoes=300,
        potencia_fv_kwp=estado["e_kwp"] or None,
        telhado=estado.get("e_telhado"),
        layout=estado.get("e_layout"),
        aguas=estado.get("e_aguas") or None,
        memoria=estado.get("e_memoria"),
        analise=estado.get("e_analise"),
        inclinacao_deg=estado.get("e_inclinacao") or None,
        azimute_deg=estado.get("e_azimute") or None,
        permitir_serie_sintetica=not estado.get("e_exigir_pvgis", False),
        max_candidatos=candidatos,
        fatura=fatura,
        gerador=gerador if considerar_gerador else None,
        considerar_solar=considerar_solar and bool(estado.get("e_com_solar", True)),
        considerar_bateria=considerar_bateria,
        considerar_gerador=considerar_gerador,
        padrao_capex=padrao_capex,
        topologia_kit=topologia_kit,
        mao_de_obra_brl_kwp=mao_de_obra_brl_kwp,
        material_ca_brl_kwp=material_ca_brl_kwp,
        capex_fv_brl=capex_fv_brl,
        autonomia_alvo_h=float(autonomia),
        confiabilidade_alvo=float(confiabilidade),
        exigir_pior_caso=True,
        malha=MalhaApagao(
            duracoes_h=tuple(sorted(duracoes)), amostras=amostras, passo_min=int(passo)),
        premissas=PremissasBateria(
            tarifa_fora_ponta_brl_kwh=tarifa,
            tarifa_ponta_brl_kwh=tarifa_ponta,
            tarifa_demanda_brl_kw_mes=tarifa_demanda,
            custo_interrupcao_brl_kwh=custo_interrupcao,
            interrupcoes_por_ano=interrupcoes,
            reserva_backup_frac=reserva,
            anos_analise=int(anos),
        ),
    )


# ============================================================================
# Passo 6 — resultado
# ============================================================================
def _passo_resultado() -> None:
    estudo = st.session_state["e_estudo"]
    if estudo is None:
        st.warning("O estudo ainda não foi rodado.")
        _rodape(rotulo_avancar="—", pendencia="voltar ao passo 5 e rodar o estudo")
        return

    cfg = estudo.configuracao
    _resposta_em_uma_frase(estudo)

    abas = st.tabs([
        "Cenários", "Resiliência", "Alternativas", "Excedência", "Dinheiro",
        "Vida útil", "Baixar",
    ])

    # As mesmas figuras do dossiê, e não uma segunda versão desenhada de
    # outro jeito — é assim que um relatório e uma interface passam a
    # discordar sobre o mesmo estudo.
    figuras = _figuras_do_estudo(estudo)

    with abas[0]:
        _aba_cenarios(estudo)
        _mostrar_figura(
            figuras, "cenarios",
            "Cada arranjo nas duas dimensões que decidem: o que faz na conta de luz "
            "e o que faz no apagão. Solar aparece num painel e não no outro; bateria, "
            "o contrário.")
        _mostrar_figura(
            figuras, "perfis_ocupacao",
            "Os padrões de ocupação da instalação e a geração solar no mesmo eixo.")

    with abas[1]:
        # A fronteira e o estado de carga são as duas figuras que explicam o
        # tamanho do banco, e viviam só dentro do PDF: quem lia o resultado na
        # tela via tabela e mais tabela, e o número tinha de ser aceito de fé.
        _mostrar_figura(
            figuras, "fronteira",
            "A fronteira entre energia e potência: onde o banco falta por kWh e "
            "onde falta por kW. São dois problemas diferentes, e a solução de um "
            "não resolve o outro.")
        _mostrar_figura(
            figuras, "soc",
            "O estado de carga do banco ao longo do apagão — quanto sobra, e quando "
            "acaba.")
        _mostrar_figura(
            figuras, "mapa_atendimento",
            "A probabilidade de atravessar, por hora em que a luz cai e por duração "
            "da interrupção.")
        _mostrar_escopos(estudo, figuras)
        if estudo.resiliencia:
            nomes = [r.conjunto.descricao() for r in estudo.resiliencia]
            padrao = estudo.recomendado.conjunto.descricao() if estudo.recomendado else nomes[0]
            escolha = st.selectbox("Conjunto", nomes, index=nomes.index(padrao))
            resultado = next(r for r in estudo.resiliencia if r.conjunto.descricao() == escolha)
            st.dataframe(resultado.por_duracao().round(3), width="stretch", hide_index=True)
            st.caption(
                "`falhas_por_potencia` e `falhas_por_energia` separam os dois modos: "
                "energia pede mais kWh, potência pede inversor maior ou corte de carga. "
                "Confundir os dois faz comprar a coisa errada."
            )
            estacao = st.selectbox("Estação", list(estudo.ensemble_backup.estacoes))
            matriz = resultado.matriz_hora_duracao(estacao)
            st.markdown("##### Probabilidade de atravessar, por hora em que a luz cai")
            st.dataframe(
                matriz.style.background_gradient(cmap="RdYlGn", vmin=0, vmax=1).format("{:.0%}"),
                width="stretch",
            )

    with abas[2]:
        colunas = [
            "conjunto", "energia_util_kwh", "potencia_kw", "autonomia_garantida_h",
            "autonomia_ano10_h", "capex_brl", "vpl_brl", "payback_anos", "atende_meta",
        ]
        st.dataframe(estudo.ranking[colunas].round(2), width="stretch", hide_index=True)
        st.caption(
            "A coluna do ano 10 é a mesma autonomia com o banco já degradado — a "
            "diferença entre uma promessa e uma promessa com data."
        )

    with abas[3]:
        _mostrar_figura(
            figuras, "excedencia",
            "A curva de excedência de pico: qual potência é ultrapassada em que "
            "fração dos dias. É ela que dimensiona o inversor, e não a média.")
        st.dataframe(estudo.tabela_excedencia.round(2), width="stretch", hide_index=True)
        st.dataframe(
            estudo.diagnostico_inversores[
                ["fabricante", "modelo", "nominal_kw", "pico_kw", "duracao_pico_s",
                 "prob_pico_diario_acima_nominal", "fracao_do_tempo_acima_nominal",
                 "duracao_media_min", "veredito"]
            ].round(3),
            width="stretch", hide_index=True,
        )

    with abas[4]:
        nomes = list(estudo.economia)
        escolha = st.selectbox("Conjunto ", nomes, key="eco")
        economico = estudo.economia[escolha]
        colunas = st.columns(4)
        _cartao(colunas[0], "CAPEX", _reais(economico.capex_brl))
        _cartao(colunas[1], "Economia no ano 1", _reais(economico.economia_ano1_brl))
        _cartao(colunas[2], "Valor da resiliência", _reais(economico.valor_resiliencia_ano1_brl))
        _cartao(colunas[3], "Payback",
                f"{economico.payback_anos:.1f} anos" if economico.payback_anos else "não paga")
        st.dataframe(economico.fluxo.round(2), width="stretch", hide_index=True)
        if economico.premissas and economico.premissas.custo_interrupcao_brl_kwh == 0:
            st.info(
                "O custo da interrupção ficou em zero, então a resiliência não entrou no "
                "fluxo de caixa — só o ganho tarifário. Em cliente que perde produção ou "
                "carga refrigerada, a resiliência costuma ser o maior dos benefícios."
            )

    with abas[5]:
        _mostrar_figura(
            figuras, "degradacao",
            "A perda de capacidade do banco ao longo dos anos, e o que ela faz com "
            "a autonomia prometida.")
        nomes = list(estudo.degradacao)
        escolha = st.selectbox("Conjunto  ", nomes, key="deg")
        st.dataframe(estudo.degradacao[escolha].round(4), width="stretch", hide_index=True)
        autonomias = estudo.resiliencia_degradada.get(escolha, {})
        if autonomias:
            st.dataframe(
                pd.DataFrame([{"ano": a, "autonomia garantida (h)": h}
                              for a, h in sorted(autonomias.items())]),
                width="stretch", hide_index=True,
            )

    with abas[6]:
        st.markdown(
            "O pacote traz o **dossiê completo em LaTeX** — capa, sumário, a foto aérea "
            "do telhado com os módulos, croqui cotado, recurso solar, demanda, "
            "excedência de pico, resiliência a apagões, degradação, análise econômica e "
            "a procedência de cada dado — mais o PDF compilado, um resumo em Markdown, "
            "todas as figuras em PNG e as tabelas em CSV."
        )
        colunas = st.columns([2, 1])
        with colunas[1]:
            st.caption("Dados que vão na capa")
            empresa = st.text_input("Empresa", value="PACE Inteligência Energética", key="capa_empresa")
            responsavel = st.text_input("Responsável técnico", key="capa_resp")
            crea = st.text_input("CREA", key="capa_crea")
            contato = st.text_input("E-mail ou telefone", key="capa_contato")

        with colunas[0]:
            if not encontrar_compilador():
                st.info(
                    "Nenhuma distribuição LaTeX encontrada nesta máquina: o pacote sai "
                    "com o `.tex` e as figuras, que compilam no Overleaf sem ajuste."
                )
            if st.button("Gerar dossiê completo", type="primary", width="stretch"):
                with st.spinner("Desenhando figuras, buscando a imagem aérea e compilando…"):
                    st.session_state["e_pacote"] = _montar_zip(
                        estudo,
                        DadosCapa(
                            empresa=empresa, responsavel=responsavel, crea=crea,
                            email=contato, referencia=cfg.nome,
                        ),
                    )
            if st.session_state["e_pacote"]:
                st.download_button(
                    "⬇️ Baixar o estudo (.zip)", data=st.session_state["e_pacote"],
                    file_name=f"estudo-{_slug(cfg.nome)}.zip", mime="application/zip",
                    width="stretch", type="primary",
                )

    st.divider()
    colunas = st.columns(2)
    if colunas[0].button("← Mudar a meta e rodar de novo", width="stretch"):
        _ir(5)
    if colunas[1].button("Novo estudo", width="stretch"):
        for chave, valor in _PADROES.items():
            st.session_state[chave] = valor
        _ir(1)


def _resposta_em_uma_frase(estudo) -> None:
    """A conclusão antes dos dados — quem lê quer saber o que comprar."""
    cfg = estudo.configuracao
    avisos = list(estudo.avisos) + list(st.session_state.get("e_avisos_carga") or [])

    if estudo.recomendado is not None:
        conjunto = estudo.recomendado.conjunto
        economico = estudo.economia[conjunto.descricao()]
        pior = estudo.recomendado.pior_janela(cfg.autonomia_alvo_h)
        st.success(
            f"### {conjunto.descricao()}\n\n"
            f"Atravessa **{cfg.autonomia_alvo_h:g} h** de falta em pelo menos "
            f"**{cfg.confiabilidade_alvo:.0%}** dos casos — inclusive no pior horário "
            f"({int(pior['hora_inicio'])}h, {pior['estacao']}), onde ainda entrega "
            f"{pior['prob_atendimento']:.0%}. É o mais barato do catálogo que consegue isso."
        )
        colunas = st.columns(4)
        _cartao(colunas[0], "Banco", f"{conjunto.energia_util_kwh:.1f} kWh úteis")
        _cartao(colunas[1], "Inversor",
                f"{conjunto.potencia_descarga_kw:.0f} / {conjunto.potencia_pico_kw:.0f} kW")
        _cartao(colunas[2], "Solar", f"{estudo.potencia_fv_kwp:.1f} kWp")
        _cartao(colunas[3], "Investimento", _reais(economico.capex_brl))
    else:
        _diagnostico_da_falha(estudo)

    for aviso in avisos[:6]:
        st.markdown(f'<div class="aviso-suave">⚠️ {aviso}</div>', unsafe_allow_html=True)
    if avisos:
        st.write("")


def _diagnostico_da_falha(estudo) -> None:
    """
    Quando nada atende, dizer **por quê** e **quanto falta**.

    "Nenhuma combinação atende" sozinho manda o usuário adivinhar. E o palpite
    natural é o errado: pedir mais kWh. Se o gargalo é potência, mais kWh não
    muda uma casa decimal — o banco maior descarrega pelo mesmo inversor
    pequeno. Distinguir os dois casos aqui é a diferença entre um orçamento
    útil e um orçamento caro e inútil.
    """
    cfg = estudo.configuracao
    ranking = estudo.ranking.sort_values(
        ["autonomia_garantida_h", "energia_util_kwh"], ascending=[False, True])
    melhor_nome = ranking.iloc[0]["conjunto"]
    melhor = next(r for r in estudo.resiliencia if r.conjunto.descricao() == melhor_nome)

    recorte = melhor.tabela[np.isclose(melhor.tabela["duracao_h"], cfg.autonomia_alvo_h)]
    if recorte.empty:
        recorte = melhor.tabela
    por_potencia = float(recorte["falhas_por_potencia"].mean())
    por_energia = float(recorte["falhas_por_energia"].mean())

    maior_banco = ranking.sort_values("energia_util_kwh").iloc[-1]
    exigencia = estudo.tabela_excedencia
    linha = exigencia[np.isclose(exigencia["prob_excedencia"], 0.01)]
    kw_necessario = float(linha["pico_diario_kw"].iloc[0]) if not linha.empty else float("nan")
    maior_inversor = float(estudo.diagnostico_inversores["nominal_kw"].max())

    if por_potencia > por_energia:
        motivo = (
            f"**O gargalo é potência, não energia.** Mesmo o maior banco avaliado "
            f"({maior_banco['energia_util_kwh']:.0f} kWh) não atravessa, porque "
            f"{por_potencia:.0%} das falhas acontecem com bateria sobrando: o inversor "
            f"não dá conta do pico. A carga de backup pede **{kw_necessario:.0f} kW** para "
            f"cobrir 99% dos dias, e o maior inversor do catálogo entrega "
            f"{maior_inversor:.0f} kW. **Comprar mais kWh não muda esse número.**"
        )
        caminhos = (
            "**(1)** tirar carga do quadro de backup até o pico caber — volte ao passo 3 "
            "e veja a aba *Que inversor aguenta*; **(2)** cadastrar um inversor de pelo "
            f"menos {kw_necessario:.0f} kW em `BDBaterias.xlsx`; **(3)** prever corte "
            "seletivo, desligando cargas não críticas durante a falta."
        )
    else:
        motivo = (
            f"**O gargalo é energia.** {por_energia:.0%} das falhas acontecem com o banco "
            f"vazio, e o maior avaliado tem {maior_banco['energia_util_kwh']:.0f} kWh úteis "
            f"contra um consumo de backup de "
            f"{estudo.ensemble_backup.energia_diaria_kwh().mean():.0f} kWh por dia."
        )
        caminhos = (
            "**(1)** aumentar o banco — cadastre mais módulos ou baterias maiores em "
            "`BDBaterias.xlsx`; **(2)** aumentar o sistema solar, que recarrega durante a "
            "falta (passo 4); **(3)** reduzir a meta ou a confiabilidade, no passo 5."
        )

    st.error(
        f"### Nenhuma combinação do catálogo atravessa {cfg.autonomia_alvo_h:g} h "
        f"com {cfg.confiabilidade_alvo:.0%} de garantia\n\n{motivo}\n\nCaminhos: {caminhos}"
    )
    colunas = st.columns(3)
    _cartao(colunas[0], "Pico exigido (99% dos dias)", f"{kw_necessario:.0f} kW")
    _cartao(colunas[1], "Maior inversor do catálogo", f"{maior_inversor:.0f} kW")
    _cartao(colunas[2], "Falhas por potência",
            f"{por_potencia:.0%}" if por_potencia + por_energia else "—")


# ============================================================================
# Utilidades
# ============================================================================
def _reais(valor: float) -> str:
    return f"R$ {_milhar(valor)}"


def _slug(texto: str) -> str:
    limpo = "".join(c if c.isalnum() else "-" for c in str(texto).lower())
    return "-".join(p for p in limpo.split("-") if p)[:50] or "estudo"


def _mostrar_escopos(estudo, figuras: dict) -> None:
    """
    O quadro essencial contra o ampliado: quanto custa levar o desejável.

    Só aparece quando a vistoria classificou alguma coisa como preferível.
    Sem isso os dois quadros são o mesmo, e mostrar duas linhas iguais sugere
    uma escolha que não existe.
    """
    escopos = getattr(estudo, "escopos", None)
    if escopos is None or not escopos.tem_preferiveis:
        return

    st.markdown("##### E se o quadro levasse também o que é preferível?")
    st.dataframe(escopos.tabela(), width="stretch", hide_index=True)
    _mostrar_figura(
        figuras, "escopos_backup",
        "À esquerda, a carga de cada quadro; à direita, o que cada um consome e o "
        "banco que exige.")

    marginal = escopos.marginal()
    if marginal:
        colunas = st.columns(4)
        _cartao(colunas[0], "Consumo a mais",
                f"{marginal['energia_diaria_kwh']:.1f} kWh/dia")
        _cartao(colunas[1], "Pico a mais", f"{marginal['pico_kw']:.2f} kW")
        _cartao(colunas[2], "Banco a mais",
                f"{marginal['energia_util_kwh']:.1f} kWh")
        _cartao(colunas[3], "Investimento a mais", _reais(marginal["capex_brl"]))

    limitante = escopos.limitante
    if limitante == "potencia":
        st.warning(
            "O que impede o banco do quadro essencial de carregar também os "
            "preferíveis é **potência**, e não energia: o pico do quadro ampliado "
            "passa da descarga do banco, e o sistema desarma no primeiro instante "
            "em vez de esvaziar devagar. Acrescentar módulo de bateria não resolve "
            "— é preciso inversor maior, que é outro equipamento e outro preço."
        )
    elif limitante == "energia":
        st.info(
            f"O banco do quadro essencial atravessa "
            f"{escopos.autonomia_do_base_no_ampliado_h:.1f} h carregando também os "
            "preferíveis, com a folga que já veio dos degraus do catálogo. Daí em "
            "diante o que falta é energia, e energia se resolve com módulo de "
            "bateria no mesmo inversor."
        )


def _figuras_do_estudo(estudo) -> dict[str, Path]:
    """
    As figuras do relatório, desenhadas uma vez e reaproveitadas na tela.

    São as mesmas do dossiê -- não uma segunda versão, desenhada de outro
    jeito, que é como um relatório e uma interface passam a discordar. Ficam
    numa pasta temporária da sessão, e o cache é a própria pasta: refazer
    treze gráficos de matplotlib a cada clique numa aba tornaria a tela
    inutilizável.
    """
    import tempfile

    from .bateria.relatorio import escrever_relatorio

    marca_atual = id(estudo)
    if st.session_state.get("e_figuras_de") == marca_atual:
        return st.session_state.get("e_figuras") or {}

    destino = Path(tempfile.mkdtemp(prefix="pace-figuras-"))
    try:
        escritos = escrever_relatorio(estudo, destino, com_graficos=True, com_latex=False)
    except Exception as exc:  # noqa: BLE001 — a tela não pode cair por um gráfico
        st.warning(f"Não consegui desenhar as figuras: {exc}")
        return {}
    figuras = {
        chave.removeprefix("figura_"): caminho
        for chave, caminho in escritos.items()
        if chave.startswith("figura_")
    }
    st.session_state["e_figuras"] = figuras
    st.session_state["e_figuras_de"] = marca_atual
    return figuras


def _mostrar_figura(figuras: dict, chave: str, legenda: str) -> None:
    """Uma figura do relatório, quando ela existe para este estudo."""
    caminho = figuras.get(chave)
    if caminho and Path(caminho).exists():
        st.image(str(caminho), caption=legenda, width="stretch")


def _montar_zip(estudo, capa: DadosCapa | None = None) -> bytes:
    """O dossiê completo — LaTeX, PDF, figuras e tabelas — como ZIP em memória."""
    return zip_do_dossie(estudo, capa)


# ============================================================================
def renderar() -> None:
    """Ponto de entrada chamado pelo ``app.py``."""
    _iniciar()
    _barra_lateral()
    passo = st.session_state["e_passo"]
    _cabecalho(passo)
    {
        1: _passo_cliente,
        2: _passo_cargas,
        3: _passo_analise,
        4: _passo_solar,
        5: _passo_equipamento,
        6: _passo_backup,
        7: _passo_meta,
        8: _passo_resultado,
    }[passo]()
