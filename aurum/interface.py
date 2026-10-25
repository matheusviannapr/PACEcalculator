"""Camada visual e orientação. Não contém regras de dimensionamento."""
from html import escape

import streamlit as st


def aplicar_estilo():
    st.markdown("""<style>
    .stMainBlockContainer {max-width:1180px; padding-top:5rem; padding-bottom:3rem;}
    h1 {font-size:2.15rem !important; letter-spacing:-.045em; max-width:850px;}
    h2,h3 {letter-spacing:-.025em;}
    [data-testid="stSidebar"] {border-right:1px solid #393934;}
    [data-testid="stSidebar"] [data-testid="stImage"] {max-width:190px; margin-bottom:1.4rem;}
    [data-testid="stSidebar"] button {text-align:left;}
    .stButton button,.stDownloadButton button {min-height:44px; border-radius:10px;}
    button:focus-visible,a:focus-visible {outline:3px solid #F8C112 !important; outline-offset:3px;}
    [data-testid="stExpander"] {border-radius:12px;}
    [data-testid="stMetric"] {background:#242422; border:1px solid #41413b; border-radius:14px; padding:1rem;}
    .pace-eyebrow {color:#F8C112; font-size:.75rem; letter-spacing:.13em; font-weight:700; margin-bottom:.6rem;}
    .pace-guide {border:1px solid #494638; background:linear-gradient(110deg,#302d20,#242422); border-radius:16px; padding:1.1rem 1.3rem; margin:1rem 0 1.3rem;}
    .pace-guide strong {display:block; color:#FAFAF8; margin-bottom:.4rem; font-size:1.05rem;}
    .pace-guide p {color:#d3d3c9; margin:0; line-height:1.65;}
    .pace-steps {display:flex; gap:8px; flex-wrap:wrap; margin:.7rem 0 1.5rem;}
    .pace-step {font-size:.8rem; border:1px solid #45453e; border-radius:8px; padding:7px 10px; color:#b9b9ae;}
    .pace-step.active {background:#F8C112; color:#161615; border-color:#F8C112; font-weight:700;}
    .bloco-metrica {background:#242422; border:1px solid #45453e; border-radius:14px; padding:1.1rem; text-align:left;}
    .bloco-valor {font-size:1.65rem; font-weight:700; color:#F8C112;}
    .bloco-rotulo {font-size:.85rem; color:#c4c4b9; margin-top:4px;}
    @media(max-width:640px) {.stMainBlockContainer{padding:5rem 1rem 2rem;} h1{font-size:1.7rem !important;} .pace-step{font-size:.72rem;} }
    </style>""", unsafe_allow_html=True)


GUIAS = {
    1: ("Comece com os dados do local", "Dê um nome ao estudo, escolha o tipo de imóvel e confirme a localização. Tenha uma conta de luz por perto: ela ajuda a conferir o consumo e a tensão da rede."),
    2: ("Escolha o caminho que combina com os seus dados", "Tem apenas a fatura? Use a conta de luz. Sabe quais aparelhos existem? Comece pelo modelo e ajuste quantidades e horários. Os modelos são exemplos: revise antes de continuar."),
    3: ("Confira o consumo antes de escolher os equipamentos", "Clique em Analisar o consumo. Depois, compare a estimativa com a conta de luz e confira os aparelhos que mais consomem. Você pode voltar e corrigir os dados."),
    4: ("Informe o espaço disponível ou a potência desejada", "Use o mapa para desenhar cada parte do telhado. Se já tiver a potência de um projeto, escolha Já sei a potência. kWp é a potência nominal dos painéis solares."),
    5: ("Confira a combinação de painéis e inversor", "Os painéis produzem energia; o inversor a converte para uso no imóvel. Confira os avisos de compatibilidade. A memória de cálculo mostra como o sistema foi dimensionado."),
    6: ("Selecione só o que precisa continuar funcionando", "Backup é a alimentação durante uma queda de energia. Comece pelos aparelhos essenciais, como iluminação e geladeira. A seleção define o consumo que a bateria terá de atender."),
    7: ("Defina o tempo de atendimento e simule", "Escolha quantas horas os equipamentos essenciais devem funcionar sem rede. Confira as fontes e os custos antes de calcular. Os ajustes avançados já têm valores iniciais."),
    8: ("Leia o resumo, compare e baixe", "Comece pela recomendação abaixo. Compare as opções em Cenários e consulte Custos e economia. Em Baixar relatório, gere os documentos com os dados do cliente."),
}


def guia(passo):
    titulo, texto = GUIAS[passo]
    st.markdown(f'<div class="pace-guide"><strong>{escape(titulo)}</strong><p>{escape(texto)}</p></div>', unsafe_allow_html=True)


def etapas(ativos, atual, nomes):
    itens = []
    for numero, indice in enumerate(ativos, 1):
        classe = "pace-step active" if indice == atual else "pace-step"
        aria = ' aria-current="step"' if indice == atual else ''
        itens.append(f'<span class="{classe}"{aria}>{numero}. {escape(nomes[indice-1])}</span>')
    st.markdown('<div class="pace-steps" aria-label="Etapas do estudo">'+''.join(itens)+'</div>', unsafe_allow_html=True)


def ajuda(passo):
    termos = {
        1: "**Tensão da rede:** a voltagem disponível no imóvel. Confira na conta ou com o responsável pela instalação. **Localização:** determina os dados de sol usados no estudo.",
        2: "**Carga:** um aparelho que consome energia. **W e kW:** potência do aparelho (1 kW = 1.000 W). **kWh:** energia consumida ao longo do tempo; aparece na conta de luz.",
        3: "**Pico de consumo:** maior potência exigida ao mesmo tempo. **P95:** valor que não foi ultrapassado em 95% dos casos simulados. **Simulação:** estima diferentes combinações de uso dos aparelhos.",
        4: "**Água do telhado:** cada face inclinada da cobertura. **Azimute:** direção para a qual a face aponta. **kWp:** potência nominal do conjunto de painéis.",
        5: "**Módulo:** painel solar. **String:** sequência de painéis ligados em série. **Inversor:** converte a energia dos painéis para uso na instalação.",
        6: "**Backup:** energia de reserva. **Cargas essenciais:** aparelhos selecionados para continuar funcionando. **kW:** potência simultânea que o conjunto precisa suportar.",
        7: "**Autonomia:** tempo desejado de funcionamento sem rede. **Confiabilidade:** proporção de casos simulados em que a meta deve ser atendida; não é garantia de atendimento real. **Tarifa:** preço da energia em R$/kWh.",
        8: "**kWh de bateria:** capacidade de armazenamento. **Excedência:** situações em que a demanda supera um limite. **CAPEX:** investimento inicial. **Payback:** tempo estimado para a economia recuperar o investimento.",
    }
    with st.expander("Entenda os termos desta etapa"):
        st.markdown(termos[passo])
