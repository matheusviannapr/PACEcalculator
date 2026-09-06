"""
PACE — interface de prospecção solar e de estudo de armazenamento.

Executar com:

    streamlit run app.py

Dois modos, escolhidos no alto da barra lateral:

* **Prospecção de telhados** — quatro passos: definir a região, escolher os
  telhados no mapa, ajustar as premissas e baixar as propostas.
* **Estudo de energia** — seis passos, um cliente conhecido do começo ao fim:
  quem é e onde fica, o que consome, o que não pode faltar na falta de luz,
  quanto de solar, quanto tempo de autonomia, e o resultado. Vive em
  :mod:`aurum.pagina_estudo`; é um fluxo diferente o bastante para não caber no
  mesmo assistente, e o import de lá puxa matplotlib, que a prospecção não tem
  por que pagar.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import streamlit as st

from aurum import __version__
from aurum.config import get_settings
from aurum.geo import segments
from aurum.geo.nominatim import GeocodingError, NominatimClient
from aurum.pipeline import (
    AreaGrandeDemais,
    OpcoesDimensionamento,
    dimensionar_lote,
    prospectar,
    resumo_da_area,
)
from aurum.proposal.context import DadosEmissor
from aurum import marca
from aurum.proposal.render import encontrar_compilador, zip_em_memoria
from aurum.pv.equipment import EquipmentError, carregar_base

logging.basicConfig(level=logging.WARNING)

st.set_page_config(
    page_title=f"{marca.NOME} — {marca.TAGLINE}",
    page_icon=str(marca.logo("clara") or "⚡"),
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
<style>
.bloco-metrica {{ background:{marca.GRAFITE}; border:1px solid #3A3A36; border-radius:10px;
                 padding:0.8rem 1rem; text-align:center; }}
.bloco-valor  {{ font-size:1.5rem; font-weight:700; color:{marca.AMARELO}; }}
.bloco-rotulo {{ font-size:0.75rem; color:{marca.CINZA}; margin-top:2px; }}
.aviso-suave  {{ background:#2E2718; border-left:3px solid {marca.AMARELO};
                padding:0.6rem 0.9rem; border-radius:4px; font-size:0.86rem; }}
</style>
""",
    unsafe_allow_html=True,
)

CONFIG = get_settings()
PASSOS = ["1. Região", "2. Telhados", "3. Premissas", "4. Propostas"]


# ----------------------------------------------------------------------
# Estado
# ----------------------------------------------------------------------
def iniciar_estado() -> None:
    padroes = {
        "passo": 1,
        "prospeccao": None,
        "selecionados": set(),
        "contextos": None,
        "falhas": [],
        "erro": None,
    }
    for chave, valor in padroes.items():
        st.session_state.setdefault(chave, valor)


iniciar_estado()


@st.cache_resource(show_spinner=False)
def carregar_equipamentos():
    """Catálogo de equipamentos, carregado uma vez por sessão."""
    return carregar_base()


def metrica(coluna, rotulo: str, valor: str) -> None:
    coluna.markdown(
        f'<div class="bloco-metrica"><div class="bloco-valor">{valor}</div>'
        f'<div class="bloco-rotulo">{rotulo}</div></div>',
        unsafe_allow_html=True,
    )


def ir_para(passo: int) -> None:
    st.session_state["passo"] = passo
    st.rerun()


# ----------------------------------------------------------------------
# Barra lateral
# ----------------------------------------------------------------------
_logo = marca.logo("escura")
if _logo:
    st.sidebar.image(str(_logo), width="stretch")
else:
    st.sidebar.title(marca.NOME)
st.sidebar.caption(f"Prospecção solar e armazenamento · v{__version__}")

MODO = st.sidebar.radio(
    "Ferramenta",
    ["Estudo de energia", "Prospecção de telhados"],
    label_visibility="collapsed",
    help=(
        "Estudo de energia: um cliente conhecido, do levantamento de cargas ao "
        "dimensionamento de solar e bateria. Prospecção: varre uma região no mapa "
        "atrás de telhados aproveitáveis."
    ),
)
st.sidebar.divider()

if MODO == "Estudo de energia":
    # O import fica aqui dentro, e não no topo: a página do estudo carrega
    # matplotlib, e quem entrou para prospectar não deve pagar esse tempo.
    from aurum.pagina_estudo import renderar as renderar_estudo

    renderar_estudo()
    st.stop()

for indice, nome in enumerate(PASSOS, start=1):
    atual = indice == st.session_state["passo"]
    concluido = indice < st.session_state["passo"]
    icone = "✅" if concluido else ("▶️" if atual else "⚪")
    if st.sidebar.button(f"{icone} {nome}", key=f"nav{indice}", use_container_width=True):
        if indice <= st.session_state["passo"] or st.session_state["prospeccao"] is not None:
            ir_para(indice)

st.sidebar.divider()

try:
    base_equipamentos = carregar_equipamentos()
    resumo = base_equipamentos.resumo()
    st.sidebar.success(
        f"Catálogo: {resumo['modulos']} módulos, {resumo['inversores']} inversores "
        f"({resumo['potencia_inversor_min_kw']:.0f}–{resumo['potencia_inversor_max_kw']:.0f} kW)"
    )
except EquipmentError as exc:
    base_equipamentos = None
    st.sidebar.error(f"Catálogo indisponível: {exc}")

if encontrar_compilador():
    st.sidebar.info("LaTeX detectado: os PDFs serão compilados.")
else:
    st.sidebar.warning("Sem LaTeX no PATH. Serão entregues os arquivos .tex (compiláveis no Overleaf).")

with st.sidebar.expander("Dados do emissor"):
    emissor = DadosEmissor(
        empresa=st.text_input("Empresa", value="PACE Inteligência Energética"),
        responsavel_tecnico=st.text_input("Responsável técnico", value=""),
        crea=st.text_input("CREA", value=""),
        telefone=st.text_input("Telefone", value=""),
        email=st.text_input("E-mail", value=""),
        site=st.text_input("Site", value=""),
    )

st.title("Prospecção e proposta automática de geração solar")


# ======================================================================
# PASSO 1 — Região
# ======================================================================
if st.session_state["passo"] == 1:
    st.subheader("Passo 1 — Escolha a região a varrer")
    st.write(
        "O sistema consulta o OpenStreetMap, identifica os telhados aproveitáveis da região "
        "e tenta descobrir qual empresa ocupa cada um."
    )

    modo = st.radio(
        "Como definir a área",
        ["Por nome (cidade, bairro, distrito)", "Por coordenadas (bbox)"],
        horizontal=True,
    )

    regiao = bbox = None
    if modo.startswith("Por nome"):
        regiao = st.text_input(
            "Região",
            placeholder="Ex.: Cidade Industrial de Curitiba",
            help="Bairros e distritos industriais funcionam melhor que cidades inteiras.",
        )
        if regiao and st.button("Conferir localização e tamanho", use_container_width=False):
            try:
                lugares = NominatimClient(CONFIG).search(regiao, limit=4)
            except GeocodingError as exc:
                st.error(str(exc))
            else:
                principal = resumo_da_area(bbox=lugares[0].bbox, settings=CONFIG)
                st.write(f"**{lugares[0].display_name}**")
                colunas = st.columns(3)
                colunas[0].metric("Área", f"{principal['area_km2']:,.0f} km²".replace(",", "."))
                colunas[1].metric("Consultas ao OSM", f"{principal['tiles']:,}".replace(",", "."))
                colunas[2].metric("Tempo estimado", f"{principal['minutos_estimados']:.0f} min")

                if principal["grande_demais"]:
                    st.error(
                        "Região grande demais para varrer de uma vez. Quase sempre isso é um "
                        "município inteiro quando o que interessa é um distrito. Tente o nome "
                        "do bairro ou do distrito industrial, ou marque a opção abaixo para "
                        "varrer assim mesmo."
                    )
                elif principal["lenta"]:
                    st.warning("Região grande: a varredura vai levar alguns minutos.")

                if len(lugares) > 1:
                    st.caption("Outras correspondências para o mesmo nome:")
                    for lugar in lugares[1:]:
                        st.caption(f"• {lugar.display_name}  ·  bbox `{lugar.bbox_str}`")
    else:
        bbox = st.text_input(
            "Bbox (min_lon, min_lat, max_lon, max_lat)",
            placeholder="-49.3465,-25.5100,-49.3225,-25.4860",
            help="Obtenha um bbox em bboxfinder.com",
        )

    col1, col2 = st.columns(2)
    area_min = col1.number_input(
        "Área mínima do telhado (m²)", min_value=100, max_value=50_000, value=1000, step=100,
        help="Abaixo de 500 m² o volume de leads cresce muito e a qualidade cai.",
    )
    limite = col2.number_input("Máximo de telhados a listar", min_value=5, max_value=500, value=50, step=5)

    st.markdown("**Tipo de local**")
    segmentos_escolhidos = st.multiselect(
        "Segmentos a procurar",
        options=list(segments.POR_CHAVE.keys()) + ["desconhecido"],
        default=[],
        format_func=lambda chave: (
            segments.SEGMENTO_DESCONHECIDO.rotulo
            if chave == "desconhecido"
            else segments.POR_CHAVE[chave].rotulo
        ),
        placeholder="Deixe vazio para procurar todos os tipos",
        help=(
            "Escolha um ou mais segmentos. Vazio procura tudo. "
            "'Não identificado' são as edificações que o OpenStreetMap não classificou — "
            "no Brasil elas são a maioria, e muitas valem a pena."
        ),
    )
    alvo = segmentos_escolhidos or "todos"

    if segmentos_escolhidos:
        descricoes = [
            (segments.SEGMENTO_DESCONHECIDO.descricao
             if c == "desconhecido" else segments.POR_CHAVE[c].descricao)
            for c in segmentos_escolhidos
        ]
        st.caption(" · ".join(descricoes))
        if "desconhecido" not in segmentos_escolhidos:
            st.caption(
                "⚠️ Filtrar por tipo descarta as edificações sem classificação no mapa. "
                "Se a região render pouco, marque também **Não identificado**."
            )

    col4, col5 = st.columns(2)
    rigoroso = col4.checkbox(
        "Filtro geométrico rigoroso",
        help="Menos telhados, geometria mais confiável. Útil em regiões com mapeamento irregular.",
    )
    enriquecer = col5.checkbox(
        "Buscar dados das empresas", value=True,
        help="Nome, telefone, site e endereço a partir do OSM e do Nominatim.",
    )
    area_grande = st.checkbox(
        "Permitir região muito grande (acima de 400 km²)",
        help=(
            "Um município inteiro passa de mil km² e leva dezenas de minutos, "
            "sobrecarregando os servidores públicos do OpenStreetMap. "
            "Marque só se for realmente essa a intenção."
        ),
    )

    if st.button("🔍 Varrer região", type="primary", use_container_width=True):
        if not regiao and not bbox:
            st.error("Informe uma região ou um bbox.")
        else:
            barra = st.progress(0.0, text="Iniciando…")
            try:
                resultado = prospectar(
                    regiao=regiao or None,
                    bbox=bbox or None,
                    min_area_m2=float(area_min),
                    alvo=alvo,
                    rigoroso=rigoroso,
                    limite=int(limite),
                    enriquecer=enriquecer,
                    confirmar_area_grande=area_grande,
                    settings=CONFIG,
                    progresso=lambda m, f: barra.progress(min(f, 1.0), text=m),
                )
            except AreaGrandeDemais as exc:
                barra.empty()
                st.error(
                    exc.diagnostico
                    + "\n\nPara varrer assim mesmo, marque "
                    "**Permitir região muito grande** logo acima."
                )
            except Exception as exc:  # noqa: BLE001 - a mensagem vai para a tela
                barra.empty()
                st.error(f"Falha na varredura: {exc}")
            else:
                barra.empty()
                if not resultado.leads:
                    st.warning(
                        "Nenhum telhado atendeu aos critérios. Tente reduzir a área mínima, "
                        "usar o segmento 'todos' ou ampliar a região."
                    )
                st.session_state["prospeccao"] = resultado
                st.session_state["selecionados"] = set()
                st.session_state["contextos"] = None
                if resultado.leads:
                    ir_para(2)


# ======================================================================
# PASSO 2 — Seleção de telhados
# ======================================================================
elif st.session_state["passo"] == 2:
    resultado = st.session_state["prospeccao"]
    if resultado is None:
        st.warning("Faça a varredura de uma região primeiro.")
        st.stop()

    st.subheader("Passo 2 — Escolha os telhados")
    st.caption(
        "Até aqui nada foi dimensionado: a lista abaixo vem só do mapa. "
        "O estudo técnico e econômico roda depois, e apenas nos telhados que você marcar."
    )

    estatisticas = resultado.estatisticas
    colunas = st.columns(4)
    metrica(colunas[0], "Telhados encontrados", f"{resultado.total}")
    metrica(colunas[1], "Área varrida", f"{estatisticas.get('area_km2', 0):.1f} km²")
    metrica(colunas[2], "Edificações analisadas", f"{estatisticas.get('elementos', 0):,}".replace(",", "."))
    enriquecimento = estatisticas.get("enriquecimento", {})
    metrica(colunas[3], "Com contato identificado", f"{enriquecimento.get('com_contato', 0)}")

    with st.expander("Detalhes da varredura"):
        st.json(estatisticas)

    ordem = st.radio(
        "Ordenar por",
        ["Pontuação do lead", "Maior área"],
        horizontal=True,
        help="A pontuação combina área, forma, segmento e qualidade dos dados cadastrais.",
    )
    leads = resultado.por_area() if ordem == "Maior área" else resultado.leads

    # --- Mapa -------------------------------------------------------
    try:
        import folium
        from folium.plugins import MarkerCluster
        from streamlit_folium import st_folium

        centro = [
            sum(l.centroid_lat for l in leads) / len(leads),
            sum(l.centroid_lon for l in leads) / len(leads),
        ]
        mapa = folium.Map(location=centro, zoom_start=14, tiles="OpenStreetMap")
        agrupador = MarkerCluster().add_to(mapa)
        maior_area = max(l.metrics.area_m2 for l in leads)

        for posicao, lead in enumerate(leads, start=1):
            empresa = lead.company
            popup = folium.Popup(
                f"<b>#{posicao} — {lead.name}</b><br>"
                f"Área: {lead.metrics.area_m2:,.0f} m²<br>"
                f"Pontuação: {lead.score:.1f}<br>"
                f"Segmento: {lead.building_type}<br>"
                f"{empresa.get('endereco', '')}<br>"
                f"{empresa.get('telefone', '')}<br>"
                f"<a href='{lead.maps_url}' target='_blank'>Ver no Google Maps</a>",
                max_width=320,
            )
            # Telhados maiores em vermelho: é onde está o contrato grande.
            intensidade = lead.metrics.area_m2 / maior_area
            cor = "#c0392b" if intensidade > 0.6 else ("#e67e22" if intensidade > 0.3 else "#2980b9")
            folium.GeoJson(
                lead.to_geojson_feature(),
                style_function=lambda _f, c=cor: {
                    "fillColor": c, "color": c, "weight": 2, "fillOpacity": 0.45
                },
                popup=popup,
                tooltip=f"#{posicao} · {lead.metrics.area_m2:,.0f} m²",
            ).add_to(agrupador)

        st_folium(mapa, height=460, use_container_width=True, returned_objects=[])
    except ImportError:
        st.info("Instale folium e streamlit-folium para visualizar o mapa.")

    # --- Tabela de seleção -------------------------------------------
    st.markdown("#### Selecione os telhados para dimensionar")
    linhas = []
    for posicao, lead in enumerate(leads, start=1):
        empresa = lead.company
        linhas.append({
            "Selecionar": lead.lead_id in st.session_state["selecionados"],
            "#": posicao,
            "Nome": lead.name,
            "Segmento": lead.segmento_rotulo,
            "Tipo OSM": lead.building_type,
            "Área (m²)": round(lead.metrics.area_m2),
            "Pontuação": lead.score,
            "Telefone": empresa.get("telefone") or "",
            "Site": empresa.get("site") or "",
            "Endereço": empresa.get("endereco") or "",
            "Mapa": lead.maps_url,
            "_id": lead.lead_id,
        })
    tabela = pd.DataFrame(linhas)

    col_a, col_b, col_c, col_d = st.columns([1, 1, 1, 1.4])
    if col_a.button("Marcar todos", use_container_width=True):
        st.session_state["selecionados"] = {l.lead_id for l in leads}
        st.rerun()
    if col_b.button("Limpar seleção", use_container_width=True):
        st.session_state["selecionados"] = set()
        st.rerun()
    quantos_top = col_c.number_input(
        "Quantos do topo", min_value=1, max_value=len(leads), value=min(10, len(leads)),
        help="Marca os primeiros da lista, na ordem escolhida acima.",
    )
    # O botão fica sempre visível: aparecer só depois de digitar um número
    # esconde a ação de quem não sabe que ela existe.
    col_d.write("")
    if col_d.button(f"Marcar os {quantos_top} do topo", use_container_width=True):
        st.session_state["selecionados"] = {l.lead_id for l in leads[:quantos_top]}
        st.rerun()

    editada = st.data_editor(
        tabela,
        hide_index=True,
        use_container_width=True,
        height=380,
        column_config={
            "Selecionar": st.column_config.CheckboxColumn(required=True),
            "Área (m²)": st.column_config.NumberColumn(format="%d"),
            "Pontuação": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f"),
            "Site": st.column_config.LinkColumn(display_text="abrir"),
            "Mapa": st.column_config.LinkColumn(display_text="ver"),
            "_id": None,
        },
        disabled=[c for c in tabela.columns if c != "Selecionar"],
        key="editor_leads",
    )
    st.session_state["selecionados"] = set(editada.loc[editada["Selecionar"], "_id"])

    escolhidos = st.session_state["selecionados"]
    area_total = sum(l.metrics.area_m2 for l in leads if l.lead_id in escolhidos)
    st.info(f"**{len(escolhidos)}** telhados selecionados · {area_total:,.0f} m² de cobertura total"
            .replace(",", "."))

    st.download_button(
        "⬇️ Baixar lista completa (CSV)",
        data=pd.DataFrame(resultado.to_records()).to_csv(index=False).encode("utf-8-sig"),
        file_name="telhados_prospectados.csv",
        mime="text/csv",
    )

    if st.button("Continuar para as premissas ➡️", type="primary", disabled=not escolhidos):
        ir_para(3)


# ======================================================================
# PASSO 3 — Premissas
# ======================================================================
elif st.session_state["passo"] == 3:
    resultado = st.session_state["prospeccao"]
    escolhidos = st.session_state["selecionados"]
    if resultado is None or not escolhidos:
        st.warning("Selecione ao menos um telhado no passo anterior.")
        st.stop()

    st.subheader(f"Passo 3 — Premissas para {len(escolhidos)} telhados")
    st.caption(
        f"O estudo será calculado para os {len(escolhidos)} telhados que você marcou, "
        "e para mais nenhum. Ajuste as premissas e mande calcular."
    )

    aba_tec, aba_eco = st.tabs(["Técnicas", "Econômicas"])

    with aba_tec:
        col1, col2 = st.columns(2)
        montagem = col1.selectbox(
            "Tipo de cobertura",
            ["coplanar", "inclinado"],
            format_func=lambda v: (
                "Telhado inclinado (módulos rentes)" if v == "coplanar"
                else "Laje plana (estrutura inclinada)"
            ),
            help="Coplanar ocupa mais o telhado; estrutura inclinada precisa de espaço entre fileiras.",
        )
        inclinacao = col2.number_input(
            "Inclinação dos módulos (°)", min_value=0.0, max_value=40.0,
            value=8.0 if montagem == "coplanar" else 20.0, step=1.0,
            help="Para telhado coplanar, use a inclinação real da cobertura.",
        )
        col3, col4 = st.columns(2)
        fator_obstaculos = col3.slider(
            "Aproveitamento da cobertura", min_value=0.40, max_value=1.00, value=0.85, step=0.01,
            help="Desconta climatização, claraboias e circulação, que o OSM não mapeia.",
        )
        limitar = col4.checkbox(
            "Limitar o sistema ao consumo estimado",
            help="Sem isto, o sistema ocupa o telhado inteiro, o que pode exceder o consumo.",
        )
        eui = st.number_input(
            "Intensidade de consumo (kWh/m²/mês) — 0 usa a referência do segmento",
            min_value=0.0, max_value=80.0, value=0.0, step=0.5,
        )

    with aba_eco:
        col1, col2, col3 = st.columns(3)
        tarifa = col1.number_input(
            "Tarifa (R$/kWh) — 0 usa a referência do segmento",
            min_value=0.0, max_value=3.0, value=0.0, step=0.01, format="%.4f",
        )
        capex_kwp = col2.number_input(
            "Custo (R$/kWp) — 0 usa a curva de escala",
            min_value=0.0, max_value=12000.0, value=0.0, step=100.0,
        )
        anos = col3.number_input("Horizonte de análise (anos)", min_value=5, max_value=30, value=25)

        col4, col5, col6 = st.columns(3)
        taxa_desconto = col4.slider("Taxa de desconto (% a.a.)", 1.0, 25.0, 10.0, 0.5) / 100
        escalada = col5.slider("Escalada tarifária (% a.a.)", 0.0, 15.0, 5.0, 0.5) / 100
        ano_conexao = col6.number_input("Ano de conexão", min_value=2023, max_value=2040, value=2026)

        lei = st.checkbox(
            "Aplicar Lei 14.300/2022 (cobrança do Fio B sobre a energia injetada)",
            value=True,
            help="Desligar superestima a economia. Mantenha ligado para proposta realista.",
        )

    if st.button("⚙️ Calcular e gerar propostas", type="primary", use_container_width=True):
        leads = [l for l in resultado.leads if l.lead_id in escolhidos]
        opcoes = OpcoesDimensionamento(
            montagem=montagem,
            tilt_deg=inclinacao,
            fator_obstaculos=fator_obstaculos,
            limitar_ao_consumo=limitar,
            tarifa_brl_kwh=tarifa or None,
            capex_brl_por_kwp=capex_kwp or None,
            taxa_desconto=taxa_desconto,
            escalada_tarifa=escalada,
            anos=int(anos),
            ano_conexao=int(ano_conexao),
            aplicar_lei_14300=lei,
            eui_personalizado=eui or None,
        )
        barra = st.progress(0.0, text="Dimensionando…")
        contextos, falhas = dimensionar_lote(
            leads,
            opcoes=opcoes,
            emissor=emissor,
            settings=CONFIG,
            base=base_equipamentos,
            progresso=lambda m, f: barra.progress(min(f, 1.0), text=m),
        )
        barra.empty()
        st.session_state["contextos"] = contextos
        st.session_state["falhas"] = falhas
        ir_para(4)


# ======================================================================
# PASSO 4 — Propostas
# ======================================================================
elif st.session_state["passo"] == 4:
    contextos = st.session_state["contextos"]
    if not contextos:
        st.warning("Nenhuma proposta calculada. Volte ao passo anterior.")
        st.stop()

    st.subheader(f"Passo 4 — {len(contextos)} propostas geradas")

    resumos = [ctx.resumo() for ctx in contextos]
    colunas = st.columns(4)
    metrica(colunas[0], "Potência total", f"{sum(r['potencia_kwp'] for r in resumos):,.0f} kWp".replace(",", "."))
    metrica(colunas[1], "Geração total", f"{sum(r['geracao_anual_kwh'] for r in resumos) / 1e6:,.2f} GWh/ano")
    metrica(colunas[2], "Investimento total", f"R$ {sum(r['capex_brl'] for r in resumos) / 1e6:,.2f} mi")
    paybacks = [r["payback_anos"] for r in resumos if r["payback_anos"]]
    metrica(colunas[3], "Payback médio", f"{sum(paybacks) / len(paybacks):.1f} anos" if paybacks else "—")

    if st.session_state["falhas"]:
        with st.expander(f"⚠️ {len(st.session_state['falhas'])} telhados sem proposta"):
            for falha in st.session_state["falhas"]:
                st.write(f"**{falha['nome']}** ({falha['lead_id']}): {falha['erro']}")

    st.markdown("#### Resultado por telhado")
    st.dataframe(
        pd.DataFrame(resumos)[[
            "referencia", "cliente", "tipo_edificacao", "area_telhado_m2", "potencia_kwp",
            "geracao_anual_kwh", "cobertura_consumo", "capex_brl", "economia_ano1_brl",
            "payback_anos", "tir_percent", "telefone", "site",
        ]],
        hide_index=True,
        use_container_width=True,
        column_config={
            "referencia": "Referência",
            "cliente": "Cliente",
            "tipo_edificacao": "Segmento",
            "area_telhado_m2": st.column_config.NumberColumn("Área (m²)", format="%d"),
            "potencia_kwp": st.column_config.NumberColumn("kWp", format="%.1f"),
            "geracao_anual_kwh": st.column_config.NumberColumn("kWh/ano", format="%d"),
            "cobertura_consumo": st.column_config.NumberColumn("Cobertura", format="%.0f%%"),
            "capex_brl": st.column_config.NumberColumn("Investimento", format="R$ %.0f"),
            "economia_ano1_brl": st.column_config.NumberColumn("Economia ano 1", format="R$ %.0f"),
            "payback_anos": st.column_config.NumberColumn("Payback", format="%.1f a"),
            "tir_percent": st.column_config.NumberColumn("TIR", format="%.1f%%"),
            "site": st.column_config.LinkColumn("Site", display_text="abrir"),
        },
    )

    col1, col2 = st.columns(2)
    col1.download_button(
        "📦 Baixar todas as propostas (.tex + .json)",
        data=zip_em_memoria(contextos),
        file_name="propostas-aurum.zip",
        mime="application/zip",
        type="primary",
        use_container_width=True,
    )
    col2.download_button(
        "📊 Baixar índice comercial (CSV)",
        data=pd.DataFrame(resumos).to_csv(index=False).encode("utf-8-sig"),
        file_name="indice-propostas.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.markdown("#### Gravar em disco e compilar os PDFs")
    destino_texto = st.text_input("Pasta de destino", value=str(CONFIG.output_dir))
    if st.button("💾 Gravar e compilar"):
        from aurum.pipeline import escrever_geojson, escrever_indice
        from aurum.proposal.render import escrever_proposta, montar_zip

        destino = Path(destino_texto)
        barra = st.progress(0.0, text="Gravando…")
        for indice, ctx in enumerate(contextos, start=1):
            barra.progress(indice / len(contextos), text=f"{ctx.referencia} ({indice}/{len(contextos)})")
            escrever_proposta(ctx, destino / ctx.referencia, compilar=True)
        escrever_indice(contextos, destino, st.session_state["falhas"])
        escrever_geojson(st.session_state["prospeccao"].leads, destino / "leads.geojson")
        caminho_zip = montar_zip(destino)
        barra.empty()
        st.success(f"Arquivos gravados em `{destino.resolve()}` · pacote: `{caminho_zip.name}`")

    st.markdown("#### Detalhe de uma proposta")
    escolha = st.selectbox(
        "Proposta", range(len(contextos)),
        format_func=lambda i: f"{contextos[i].referencia} — {contextos[i].titulo_cliente}",
    )
    ctx = contextos[escolha]

    aba1, aba2, aba3 = st.tabs(["Resumo", "Ressalvas", "Dados completos"])
    with aba1:
        col1, col2 = st.columns(2)
        col1.write(f"**Cliente:** {ctx.titulo_cliente}")
        col1.write(f"**Endereço:** {ctx.endereco}")
        col1.write(f"**Contato:** {ctx.lead.company.get('telefone') or '—'}")
        col1.write(f"**Site:** {ctx.lead.company.get('site') or '—'}")
        col2.write(f"**Sistema:** {ctx.sistema.potencia_cc_kwp:,.1f} kWp · "
                   f"{ctx.sistema.modulos_totais} módulos")
        col2.write(f"**Inversores:** {ctx.sistema.descricao_inversores()}")
        col2.write(f"**Geração:** {ctx.sistema.geracao_anual_kwh:,.0f} kWh/ano "
                   f"(fonte: {ctx.perfil_solar.fonte})")
        col2.write(f"**Ocupação do telhado:** {ctx.layout.taxa_ocupacao:.0%}")

        # O ano vai como texto: tratado como número, o gráfico o formata com
        # separador de milhar e mostra "2.026".
        fluxo = pd.DataFrame([
            {"Ano": str(l.ano_calendario), "VPL acumulado (R$)": l.vpl_acumulado_brl,
             "Fluxo líquido (R$)": l.fluxo_liquido_brl}
            for l in ctx.economia.fluxo
        ])
        st.line_chart(fluxo.set_index("Ano"))
    with aba2:
        for item in ctx.avisos_consolidados():
            st.markdown(f'<div class="aviso-suave">{item}</div>', unsafe_allow_html=True)
            st.write("")
    with aba3:
        st.json(ctx.as_dict(), expanded=False)
