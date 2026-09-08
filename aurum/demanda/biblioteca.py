"""
Biblioteca de cargas: equipamentos típicos, modelos de cômodo e curvas de referência.

Três coisas diferentes, que juntas tiram do usuário a parte chata do
levantamento e dão um jeito de conferir o resultado:

1. **Catálogo de equipamentos** (:data:`EQUIPAMENTOS`) — potência, janela de
   uso típica, probabilidade e fator de demanda de cada aparelho que costuma
   aparecer numa instalação brasileira. Serve para montar o cenário clicando
   em vez de digitar dez colunas por linha.

2. **Modelos de cômodo por segmento** (:data:`MODELOS`) — "Quarto de hotel",
   "Cozinha industrial", "Sala de aula", "Câmara fria" já vêm com os
   equipamentos dentro. Um hotel sai montado em dois cliques e depois é
   ajustado, que é a ordem certa: corrigir um rascunho é muito mais barato
   que partir da folha em branco.

3. **Curvas de carga típicas** (:mod:`aurum.demanda.biblioteca` +
   ``data/load_profiles/``) — as curvas normalizadas de 24 h que vieram do
   Aurumcalc, uma por segmento, somando 1,0 em p.u. Elas têm dois usos e os
   dois importam:

   * **Conferência.** Depois de simular, comparar a curva do Monte Carlo com o
     padrão do segmento. Se o levantamento diz que um supermercado consome de
     madrugada tanto quanto ao meio-dia, alguma coisa foi digitada errado — e é
     melhor descobrir isso antes de dimensionar bateria em cima.
   * **Caminho alternativo.** Sem levantamento de equipamentos, a curva típica
     calibrada pela conta de luz (:func:`calibrar_por_conta`) já dá uma curva
     horária utilizável. Pior que o levantamento, muito melhor que nada — e o
     estudo declara qual dos dois usou.

**Aviso herdado da base:** os perfis por setor em ``data/load_profiles/`` estão
marcados ``synthetic_for_testing`` na própria origem. São formas plausíveis,
não medições. :func:`perfil_confiavel` diz quais são quais.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "EQUIPAMENTOS",
    "LIMITE_DIVERGENCIA_PP",
    "MODELOS",
    "PERIODOS",
    "SEGMENTOS",
    "PerfilTipico",
    "calibrar_por_conta",
    "catalogo_como_dataframe",
    "comparar_com_perfil",
    "equipamento",
    "modelo_de_comodo",
    "perfil_confiavel",
    "perfil_tipico",
    "perfis_tipicos",
    "segmento_do_perfil",
]

RAIZ_PERFIS = Path(__file__).resolve().parent.parent.parent / "data" / "load_profiles"
ARQUIVO_TIPICOS = RAIZ_PERFIS / "typical_load_profiles.json"

#: Colunas que o parser do D² espera numa aba de cômodo, na ordem em que fazem
#: sentido para quem preenche.
COLUNAS = [
    "Equipamento", "Potência", "Quantidade", "Tipo de intervalo", "intervalo",
    "probabilidade", "FD", "duracao_min", "duracao_max", "modo_fixo",
    "criticidade",
]

#: Criticidade por categoria, quando o nome do equipamento não diz mais.
#:
#: É o que decide o que entra no quadro de backup, e portanto o tamanho do
#: inversor. Recortar por ambiente — o comportamento anterior para cenários de
#: modelo — levava a geladeira e o forno de 4 kW juntos por estarem na mesma
#: cozinha, e dobrava o inversor por causa de um equipamento que ninguém
#: precisa no apagão.
CRITICIDADE_POR_CATEGORIA: dict[str, str] = {
    "Refrigeração": "C",       # comida estraga, e o prejuízo não volta
    "Elevação": "C",           # gente presa
    "Saúde": "MC",             # segurança de vida
    "Escritório e TI": "P",
    "Iluminação": "P",
    "Climatização": "P",       # conforto: o cliente sente falta e a casa não para
    "Outros": "P",
    "Motores e bombas": "NC",
    "Cozinha": "NC",           # alta potência, e pode esperar a rede voltar
    "Aquecimento de água": "NC",
    "Lavanderia": "NC",
    "Industrial": "NC",
}

#: As exceções, por nome. Onde a categoria erraria, e erraria feio.
CRITICIDADE_POR_EQUIPAMENTO: dict[str, str] = {
    # Rede e segurança não admitem interrupção, e consomem quase nada — é essa
    # combinação que faz o quadro de backup ser barato.
    "Nobreak / rack de rede": "MC",
    "Servidor de rack": "MC",
    "Sistema de segurança / CFTV": "MC",
    "Central de alarme": "MC",
    "Iluminação de emergência": "MC",
    # Água: a casa fica sem água antes de ficar sem luz, na percepção de quem
    # mora nela.
    "Bomba d'água 1 CV": "C",
    "Bomba d'água 3 CV": "C",
    "Bomba de recalque 5 CV": "C",
    # Câmara fria é refrigeração de estoque, e o degelo é o que a mantém.
    "Câmara fria (unidade condensadora)": "C",
    "Degelo de câmara fria": "C",
    "Portão automático": "P",
    # Piscina é conforto que espera; carro elétrico carrega quando a luz volta.
    "Bomba de piscina": "NC",
    "Carregador de veículo elétrico 7 kW": "NC",
}

#: Onde a criticidade não é conhecida. Fica de fora do backup por omissão, que
#: é o lado seguro: um equipamento a mais no quadro encarece o inversor, e um a
#: menos aparece na conversa com o cliente antes da compra.
CRITICIDADE_PADRAO = "NC"


def criticidade_sugerida(nome: str) -> str:
    """
    O nível sugerido para um equipamento do catálogo.

    Sugestão, e não veredito: a coluna é editável na tela de cargas, e a
    vistoria técnica, quando existe, passa por cima dela. Quem esteve no
    imóvel sabe mais que uma regra por categoria — mas uma regra por categoria
    sabe mais que o silêncio, que era o que havia antes.
    """
    especifica = CRITICIDADE_POR_EQUIPAMENTO.get(nome)
    if especifica:
        return especifica
    item = next((e for e in EQUIPAMENTOS if e["nome"] == nome), None)
    if item is None:
        return CRITICIDADE_PADRAO
    return CRITICIDADE_POR_CATEGORIA.get(item["categoria"], CRITICIDADE_PADRAO)

#: Blocos de seis horas usados na conferência contra a curva de referência.
PERIODOS = ("madrugada", "manhã", "tarde", "noite")

#: Acima desta diferença na fatia de energia de um período, em pontos
#: percentuais do consumo diário, a curva simulada merece conferência. Quinze
#: pontos é bastante: significa que um sexto do dia inteiro mudou de lugar.
LIMITE_DIVERGENCIA_PP = 15.0


# ============================================================================
# Catálogo de equipamentos
# ============================================================================
def _eq(
    nome: str,
    categoria: str,
    potencia: float,
    intervalo: str,
    probabilidade: float = 1.0,
    fd: float = 1.0,
    tipo: str = "fixo",
    duracao_min: float | None = None,
    duracao_max: float | None = None,
    modo_fixo: str | None = "FIXO_100%",
    nota: str = "",
) -> dict[str, Any]:
    return {
        "nome": nome, "categoria": categoria, "potencia_w": potencia,
        "tipo_intervalo": tipo, "intervalo": intervalo,
        "probabilidade": probabilidade, "fd": fd,
        "duracao_min": duracao_min, "duracao_max": duracao_max,
        "modo_fixo": modo_fixo, "nota": nota,
    }


#: Equipamentos típicos. Potências são de placa, em watts; o fator de demanda
#: (`fd`) traz a média de operação, e a probabilidade, a chance de o aparelho
#: ser usado naquele dia. Separar os dois é o que evita o erro clássico de
#: somar potência instalada e chamar de demanda.
EQUIPAMENTOS: list[dict[str, Any]] = [
    # -- Climatização --------------------------------------------------------
    _eq("Ar-condicionado split 9.000 BTU", "Climatização", 900, "08:00 as 18:00",
        0.7, 0.65, "dinâmico", 2.0, 6.0, None,
        "Inverter moderno; o FD traz o ciclo do compressor."),
    _eq("Ar-condicionado split 12.000 BTU", "Climatização", 1200, "08:00 as 18:00",
        0.7, 0.65, "dinâmico", 2.0, 6.0, None),
    _eq("Ar-condicionado split 18.000 BTU", "Climatização", 1800, "08:00 as 18:00",
        0.7, 0.65, "dinâmico", 2.0, 6.0, None),
    _eq("Ar-condicionado split 24.000 BTU", "Climatização", 2400, "08:00 as 18:00",
        0.7, 0.65, "dinâmico", 2.0, 6.0, None),
    _eq("Ar-condicionado janela 10.000 BTU", "Climatização", 1400, "08:00 as 18:00",
        0.6, 0.8, "dinâmico", 2.0, 5.0, None,
        "Sem inverter: liga e desliga em degrau, FD alto."),
    _eq("Self-contained 5 TR", "Climatização", 6000, "08:00 as 18:00", 0.9, 0.75, "dinâmico", 6.0, 10.0, None),
    _eq("Ventilador de teto", "Climatização", 130, "08:00 as 22:00", 0.6, 1.0),
    _eq("Exaustor de banheiro", "Climatização", 60, "06:00 as 23:00", 0.4, 0.5),
    _eq("Cortina de ar", "Climatização", 500, "08:00 as 20:00", 1.0, 0.9),

    # -- Iluminação ----------------------------------------------------------
    _eq("Lâmpada LED bulbo 9 W", "Iluminação", 9, "18:00 as 23:00"),
    _eq("Luminária LED tubular 20 W", "Iluminação", 20, "08:00 as 18:00"),
    _eq("Luminária LED 2×20 W", "Iluminação", 40, "08:00 as 18:00"),
    _eq("Luminária LED alta potência 150 W", "Iluminação", 150, "18:00 as 06:00",
        1.0, 1.0, "fixo", None, None, "FIXO_100%", "Galpão e pátio; janela noturna."),
    _eq("Iluminação de emergência", "Iluminação", 15, "00:00 as 23:59"),
    _eq("Letreiro / fachada", "Iluminação", 300, "18:00 as 23:00"),

    # -- Refrigeração --------------------------------------------------------
    _eq("Geladeira doméstica", "Refrigeração", 150, "00:00 as 23:59", 1.0, 0.35,
        "fixo", None, None, "FIXO_100%", "FD 0,35 é o ciclo do compressor ao longo do dia."),
    _eq("Frigobar", "Refrigeração", 90, "00:00 as 23:59", 1.0, 0.3),
    _eq("Freezer horizontal", "Refrigeração", 350, "00:00 as 23:59", 1.0, 0.4),
    _eq("Expositor refrigerado", "Refrigeração", 600, "00:00 as 23:59", 1.0, 0.5),
    _eq("Câmara fria (unidade condensadora)", "Refrigeração", 3000, "00:00 as 23:59", 1.0, 0.55),
    _eq("Degelo de câmara fria", "Refrigeração", 2200, "22:00 as 02:00",
        1.0, 1.0, "dinâmico", 1.0, 2.0, None, "Janela que vira a meia-noite."),
    _eq("Chiller 10 TR", "Refrigeração", 12000, "07:00 as 19:00", 1.0, 0.7, "dinâmico", 8.0, 12.0, None),

    # -- Cozinha -------------------------------------------------------------
    _eq("Forno elétrico industrial", "Cozinha", 6000, "10:00 as 14:00", 0.9, 0.6, "dinâmico", 1.0, 3.0, None),
    _eq("Fritadeira elétrica", "Cozinha", 5000, "11:00 as 22:00", 0.9, 0.5, "dinâmico", 2.0, 6.0, None),
    _eq("Chapa / grill", "Cozinha", 4000, "11:00 as 22:00", 0.9, 0.5, "dinâmico", 2.0, 6.0, None),
    _eq("Forno de micro-ondas", "Cozinha", 1400, "11:00 as 14:00", 0.8, 0.25, "dinâmico", 0.2, 0.6, None),
    _eq("Cafeteira industrial", "Cozinha", 2000, "06:00 as 17:00", 0.9, 0.3, "dinâmico", 0.5, 2.0, None),
    _eq("Lava-louças industrial", "Cozinha", 6000, "12:00 as 23:00", 0.9, 0.6, "dinâmico", 1.0, 3.0, None),
    _eq("Coifa / exaustão de cozinha", "Cozinha", 750, "10:00 as 23:00", 1.0, 0.9),
    _eq("Liquidificador industrial", "Cozinha", 800, "08:00 as 18:00", 0.6, 0.2, "dinâmico", 0.1, 0.5, None),

    # -- Aquecimento de água -------------------------------------------------
    _eq("Chuveiro elétrico 5.500 W", "Aquecimento de água", 5500, "06:00 as 09:00",
        0.9, 1.0, "dinâmico", 0.15, 0.4, None,
        "O clássico causador de pico: potência alta, duração curta."),
    _eq("Chuveiro elétrico 7.500 W", "Aquecimento de água", 7500, "06:00 as 09:00",
        0.9, 1.0, "dinâmico", 0.15, 0.4, None),
    _eq("Boiler elétrico 3.000 W", "Aquecimento de água", 3000, "05:00 as 09:00", 1.0, 0.7, "dinâmico", 2.0, 4.0, None),
    _eq("Torneira elétrica", "Aquecimento de água", 4500, "06:00 as 22:00", 0.7, 1.0, "dinâmico", 0.05, 0.2, None),

    # -- Motores e bombas ----------------------------------------------------
    _eq("Bomba d'água 1 CV", "Motores e bombas", 750, "00:00 as 23:59", 1.0, 0.25, "dinâmico", 2.0, 6.0, None),
    _eq("Bomba d'água 3 CV", "Motores e bombas", 2200, "00:00 as 23:59", 1.0, 0.3, "dinâmico", 3.0, 8.0, None),
    _eq("Bomba de recalque 5 CV", "Motores e bombas", 3700, "00:00 as 23:59", 1.0, 0.35, "dinâmico", 4.0, 10.0, None),
    _eq("Bomba de piscina", "Motores e bombas", 1100, "08:00 as 16:00", 1.0, 1.0),
    _eq("Compressor de ar 10 CV", "Motores e bombas", 7500, "08:00 as 18:00", 1.0, 0.6, "dinâmico", 6.0, 10.0, None),
    _eq("Portão automático", "Motores e bombas", 500, "06:00 as 22:00", 0.9, 0.05, "dinâmico", 0.05, 0.2, None),

    # -- Elevação ------------------------------------------------------------
    _eq("Elevador social", "Elevação", 7500, "06:00 as 23:00", 1.0, 0.3, "dinâmico", 0.05, 0.3, None,
        "Partida puxa muito mais que o nominal: é o caso que testa o surto do inversor."),
    _eq("Elevador de carga", "Elevação", 11000, "07:00 as 19:00", 0.8, 0.25, "dinâmico", 0.05, 0.3, None),
    _eq("Plataforma elevatória", "Elevação", 2200, "07:00 as 19:00", 0.5, 0.1, "dinâmico", 0.05, 0.2, None),

    # -- Escritório e TI -----------------------------------------------------
    _eq("Computador desktop", "Escritório e TI", 180, "08:00 as 18:00", 0.9, 0.8),
    _eq("Notebook", "Escritório e TI", 65, "08:00 as 18:00", 0.9, 0.7),
    _eq("Monitor", "Escritório e TI", 35, "08:00 as 18:00", 0.9, 0.9),
    _eq("Impressora multifuncional", "Escritório e TI", 600, "08:00 as 18:00", 0.7, 0.12),
    _eq("Servidor de rack", "Escritório e TI", 500, "00:00 as 23:59", 1.0, 0.8),
    _eq("Nobreak / rack de rede", "Escritório e TI", 300, "00:00 as 23:59", 1.0, 0.7),
    _eq("Projetor", "Escritório e TI", 280, "08:00 as 18:00", 0.4, 0.9, "dinâmico", 1.0, 3.0, None),
    _eq("TV LED 50\"", "Escritório e TI", 110, "18:00 as 23:00", 0.8, 1.0),

    # -- Lavanderia ----------------------------------------------------------
    _eq("Máquina de lavar doméstica", "Lavanderia", 600, "08:00 as 20:00", 0.5, 0.6, "dinâmico", 1.0, 2.0, None),
    _eq("Lavadora industrial 20 kg", "Lavanderia", 4000, "07:00 as 17:00", 0.9, 0.7, "dinâmico", 2.0, 6.0, None),
    _eq("Secadora industrial 20 kg", "Lavanderia", 9000, "08:00 as 18:00", 0.9, 0.8, "dinâmico", 2.0, 5.0, None),
    _eq("Calandra", "Lavanderia", 6000, "08:00 as 17:00", 0.7, 0.7, "dinâmico", 2.0, 5.0, None),

    # -- Saúde ---------------------------------------------------------------
    _eq("Autoclave", "Saúde", 3500, "07:00 as 19:00", 0.8, 0.5, "dinâmico", 1.0, 3.0, None),
    _eq("Raio-X", "Saúde", 5000, "08:00 as 18:00", 0.5, 0.08, "dinâmico", 0.05, 0.3, None),
    _eq("Monitor multiparâmetro", "Saúde", 120, "00:00 as 23:59", 1.0, 0.9),
    _eq("Bomba de infusão", "Saúde", 40, "00:00 as 23:59", 1.0, 0.8),

    # -- Industrial ----------------------------------------------------------
    _eq("Motor trifásico 5 CV", "Industrial", 3700, "07:00 as 17:00", 1.0, 0.7),
    _eq("Motor trifásico 15 CV", "Industrial", 11000, "07:00 as 17:00", 1.0, 0.7),
    _eq("Máquina de solda", "Industrial", 8000, "07:00 as 17:00", 0.7, 0.3, "dinâmico", 1.0, 4.0, None),
    _eq("Estufa industrial", "Industrial", 15000, "07:00 as 19:00", 1.0, 0.6, "dinâmico", 6.0, 12.0, None),
    _eq("Ponte rolante", "Industrial", 9000, "07:00 as 17:00", 0.6, 0.15, "dinâmico", 0.1, 0.5, None),

    # -- Outros --------------------------------------------------------------
    _eq("Sistema de segurança / CFTV", "Outros", 200, "00:00 as 23:59"),
    _eq("Central de alarme", "Outros", 50, "00:00 as 23:59"),
    _eq("Carregador de veículo elétrico 7 kW", "Outros", 7400, "18:00 as 06:00",
        0.5, 1.0, "dinâmico", 3.0, 6.0, None, "Janela noturna, que vira a meia-noite."),
    _eq("Máquina de café expresso", "Outros", 1600, "07:00 as 18:00", 0.9, 0.2, "dinâmico", 0.1, 0.4, None),
    _eq("Bebedouro", "Outros", 100, "00:00 as 23:59", 1.0, 0.4),
]

CATEGORIAS = tuple(dict.fromkeys(e["categoria"] for e in EQUIPAMENTOS))


def equipamento(nome: str) -> dict[str, Any]:
    """Um equipamento do catálogo, por nome exato."""
    for item in EQUIPAMENTOS:
        if item["nome"] == nome:
            return dict(item)
    raise KeyError(f"equipamento não está no catálogo: {nome}")


def catalogo_como_dataframe(categoria: str | None = None) -> pd.DataFrame:
    itens = [e for e in EQUIPAMENTOS if categoria in (None, e["categoria"])]
    return pd.DataFrame(itens)


def linha_de_planilha(nome_equipamento: str, quantidade: int = 1, **ajustes: Any) -> dict[str, Any]:
    """
    Converte um item do catálogo numa linha no formato do D².

    ``ajustes`` sobrescreve campos do catálogo para o contexto do cômodo. O
    mesmo ar-condicionado de 9.000 BTU roda das 8 h às 18 h numa sala de aula e
    das 18 h às 10 h num quarto de hotel — é o mesmo aparelho e são horários
    opostos, porque quem o liga é a ocupação, não o equipamento.
    """
    item = equipamento(nome_equipamento)
    item.update(ajustes)
    return {
        "Equipamento": item["nome"],
        "Potência": float(item["potencia_w"]),
        "Quantidade": int(quantidade),
        "Tipo de intervalo": item["tipo_intervalo"],
        "intervalo": item["intervalo"],
        "probabilidade": float(item["probabilidade"]),
        "FD": float(item["fd"]),
        # NaN e não None: numa coluna numérica, `None` vira o texto "None" na
        # grade de edição, enquanto NaN aparece como célula vazia — que é o que
        # "este equipamento não tem duração sorteada" quer dizer.
        "duracao_min": float(item["duracao_min"]) if item["duracao_min"] is not None else np.nan,
        "duracao_max": float(item["duracao_max"]) if item["duracao_max"] is not None else np.nan,
        "modo_fixo": item["modo_fixo"],
        # A criticidade acompanha a linha desde a origem. Sem ela, o quadro de
        # backup de um cenário de modelo tinha de ser recortado por ambiente —
        # e recortar por ambiente leva a geladeira e o forno de 4 kW juntos,
        # por estarem na mesma cozinha.
        "criticidade": ajustes.get("criticidade") or criticidade_sugerida(item["nome"]),
    }


# ============================================================================
# Modelos de cômodo por segmento
# ============================================================================
@dataclass(frozen=True)
class ModeloComodo:
    """Um cômodo pré-montado: nome, quantas vezes se repete e o que tem dentro."""

    nome: str
    instancias: int
    #: ``(nome no catálogo, quantidade)`` ou ``(nome, quantidade, {ajustes})``.
    equipamentos: tuple[tuple, ...]
    essencial: bool = False

    def para_dataframe(self) -> pd.DataFrame:
        linhas = [
            linha_de_planilha(item[0], item[1], **(item[2] if len(item) > 2 else {}))
            for item in self.equipamentos
        ]
        return pd.DataFrame(linhas, columns=COLUNAS)


@dataclass(frozen=True)
class Segmento:
    """Um tipo de instalação, com seus cômodos e a curva típica de referência."""

    id: str
    nome: str
    perfil_tipico_id: str
    comodos: tuple[ModeloComodo, ...]
    #: Consumo específico anual, em kWh/m²·ano — só para conferência grosseira.
    eui_kwh_m2_ano: float | None = None
    #: Ordem de grandeza do que custa ao cliente 1 kWh que faltou, em R$.
    #:
    #: É o parâmetro mais sensível de todo o estudo de bateria e o único que
    #: não se estima de fora: depende do que o cliente perde parado. A
    #: sugestão existe porque a alternativa era pior -- o campo nascia em zero,
    #: e zero afirma que ficar sem energia não custa nada, o que nenhuma
    #: instalação com quadro de backup acredita. Sugerir uma ordem de grandeza
    #: e dizer que é sugestão põe a conversa no lugar certo: o número final
    #: vem do cliente.
    #:
    #: A escala separa três situações: quem só perde conforto, quem perde
    #: faturamento enquanto está parado, e quem perde estoque -- carga
    #: refrigerada estraga, e o prejuízo não volta quando a luz volta.
    custo_interrupcao_brl_kwh: float = 15.0

    def montar(self) -> tuple[list[str], dict[str, pd.DataFrame], dict[str, int], list[str]]:
        """Devolve (ordem dos cômodos, tabelas, instâncias, essenciais sugeridos)."""
        tabelas = {c.nome: c.para_dataframe() for c in self.comodos}
        instancias = {c.nome: c.instancias for c in self.comodos}
        essenciais = [c.nome for c in self.comodos if c.essencial]
        return [c.nome for c in self.comodos], tabelas, instancias, essenciais


MODELOS: dict[str, Segmento] = {
    "residencia": Segmento(
        "residencia", "Residência", "residencial",
        (
            # A sala é o centro da noite: TV, luz e o ar que só liga quando a
            # família se junta. O rack de rede e o CFTV são o que não desliga
            # nunca, e é por isso que os dois vão para o quadro de backup.
            ModeloComodo("Sala de estar", 1, (
                ("TV LED 50\"", 1, {"intervalo": "18:00 as 23:30"}),
                ("Lâmpada LED bulbo 9 W", 6, {"intervalo": "18:00 as 23:30"}),
                ("Ventilador de teto", 1,
                 {"intervalo": "12:00 as 23:00", "probabilidade": 0.5}),
                ("Ar-condicionado split 12.000 BTU", 1,
                 {"intervalo": "19:00 as 23:30", "duracao_min": 2.0,
                  "duracao_max": 4.5, "probabilidade": 0.55}),
                ("Nobreak / rack de rede", 1),
                ("Sistema de segurança / CFTV", 1),
            ), essencial=True),
            # A cozinha tem a única carga que roda 24 h e as duas refeições. O
            # micro-ondas entra duas vezes de propósito: almoço e jantar são
            # eventos separados, e uma janela esticada das 11 às 21 h faria a
            # casa esquentar a comida às 16 h.
            ModeloComodo("Cozinha", 1, (
                ("Geladeira doméstica", 1),
                ("Freezer horizontal", 1),
                ("Forno de micro-ondas", 1,
                 {"intervalo": "11:30 as 14:00", "probabilidade": 0.8}),
                ("Forno de micro-ondas", 1,
                 {"intervalo": "18:30 as 21:30", "probabilidade": 0.8}),
                ("Coifa / exaustão de cozinha", 1,
                 {"intervalo": "18:30 as 21:30", "probabilidade": 0.6}),
                ("Lâmpada LED bulbo 9 W", 4, {"intervalo": "17:30 as 22:30"}),
                ("Torneira elétrica", 1,
                 {"intervalo": "06:00 as 22:00", "duracao_min": 0.15,
                  "duracao_max": 0.4, "probabilidade": 0.6}),
            ), essencial=True),
            # Três quartos. O ar da noite é a carga que mais separa uma casa
            # de alto padrão de uma casa média -- e é também a que mais
            # facilmente estraga o modelo: uma janela das 21:30 às 07:00 com
            # 4 a 8 h de duração jogava 30% da energia da casa na madrugada,
            # contra 12% da curva de referência. Quem dorme desliga o ar de
            # madrugada, ou o termostato o faz.
            ModeloComodo("Quarto", 3, (
                ("Ar-condicionado split 9.000 BTU", 1,
                 {"intervalo": "21:30 as 03:00", "duracao_min": 2.5,
                  "duracao_max": 5.0, "probabilidade": 0.55}),
                ("Ventilador de teto", 1,
                 {"intervalo": "13:00 as 18:00", "probabilidade": 0.4}),
                ("Lâmpada LED bulbo 9 W", 2, {"intervalo": "18:30 as 23:30"}),
                ("Notebook", 1, {"intervalo": "13:00 as 23:30",
                                 "probabilidade": 0.7}),
            )),
            # O chuveiro é o pico da manhã da casa brasileira, e o segundo
            # banho da noite é real: sem ele o modelo devolve uma manhã
            # pesada demais em relação à noite.
            ModeloComodo("Banheiro", 2, (
                ("Chuveiro elétrico 5.500 W", 1,
                 {"intervalo": "06:00 as 09:00", "duracao_min": 0.12,
                  "duracao_max": 0.25, "probabilidade": 0.9}),
                ("Chuveiro elétrico 5.500 W", 1,
                 {"intervalo": "18:00 as 22:30", "duracao_min": 0.12,
                  "duracao_max": 0.3, "probabilidade": 0.6}),
                ("Lâmpada LED bulbo 9 W", 1, {"intervalo": "06:00 as 23:00"}),
                ("Exaustor de banheiro", 1, {"intervalo": "06:00 as 23:00"}),
            )),
            ModeloComodo("Área de serviço", 1, (
                ("Máquina de lavar doméstica", 1,
                 {"intervalo": "08:00 as 19:00", "duracao_min": 0.8,
                  "duracao_max": 1.5, "probabilidade": 0.5}),
                ("Lâmpada LED bulbo 9 W", 2, {"intervalo": "17:30 as 21:00"}),
            )),
            # A bomba d'água é contínua no catálogo e liga por minutos: é ela
            # que põe carga na madrugada sem inventar ninguém acordado. A da
            # piscina é o contrário -- roda no meio do dia, e é a carga que
            # dá à tarde da casa o peso que a curva de referência mostra e que
            # um modelo só de gente presente não consegue explicar.
            ModeloComodo("Área externa", 1, (
                ("Bomba d'água 1 CV", 1),
                ("Bomba de piscina", 1,
                 {"intervalo": "09:00 as 16:00", "duracao_min": 3.0,
                  "duracao_max": 5.0, "probabilidade": 0.8}),
                ("Portão automático", 1),
                ("Luminária LED alta potência 150 W", 2,
                 {"intervalo": "18:00 as 06:00"}),
            )),
        ),
        eui_kwh_m2_ano=45.0,
        # Casa parada custa conforto, e não faturamento: comida estragando na
        # geladeira é o único prejuízo que não volta quando a luz volta.
        custo_interrupcao_brl_kwh=10.0,
    ),
    "hotel": Segmento(
        "hotel", "Hotel / pousada", "hotel",
        (
            # O hóspede sai de manhã e volta à noite: climatização, TV e luz do
            # quarto seguem a ocupação, não o horário comercial. Usar a janela
            # padrão do catálogo (08:00-18:00) punha o pico do hotel ao meio-dia,
            # que é justamente a hora em que o quarto está vazio.
            ModeloComodo("Apartamento", 40, (
                # O retorno começa às 16 h, não às 18: a janela precisa abrir
                # antes do horário de pico, porque o sorteio do início é
                # uniforme dentro dela. Abrindo às 18 h, quase ninguém tinha
                # ligado o ar às 18 h — e o modelo cavava um vale às 17 h, que
                # é justamente quando o hotel volta a encher.
                ("Ar-condicionado split 9.000 BTU", 1,
                 {"intervalo": "16:00 as 10:00", "duracao_min": 5.0, "duracao_max": 10.0,
                  "probabilidade": 0.8}),
                ("Lâmpada LED bulbo 9 W", 4, {"intervalo": "17:00 as 23:30"}),
                ("TV LED 50\"", 1, {"intervalo": "19:00 as 23:30"}),
                ("Frigobar", 1),
                # Ocupação diurna residual: check-in cedo, quarto de descanso,
                # hóspede a trabalho. Sem esta linha o modelo abre um buraco
                # entre 10 h e 18 h que nenhum hotel real tem.
                ("Ar-condicionado split 9.000 BTU", 1,
                 {"intervalo": "10:00 as 18:00", "duracao_min": 1.0, "duracao_max": 3.0,
                  "probabilidade": 0.2}),
                ("Chuveiro elétrico 5.500 W", 1,
                 {"intervalo": "06:00 as 09:30", "probabilidade": 0.7}),
                ("Chuveiro elétrico 5.500 W", 1,
                 {"intervalo": "18:00 as 22:00", "probabilidade": 0.5}),
            )),
            ModeloComodo("Áreas comuns", 1, (
                # Corredor de hotel fica aceso o tempo todo — é isso que
                # sustenta a carga da tarde e da madrugada.
                ("Luminária LED tubular 20 W", 40, {"intervalo": "00:00 as 23:59"}),
                ("Lâmpada LED bulbo 9 W", 30),
                ("Elevador social", 2),
                ("Sistema de segurança / CFTV", 1),
                ("Bomba de recalque 5 CV", 2),
                ("Iluminação de emergência", 20),
            ), essencial=True),
            ModeloComodo("Cozinha e restaurante", 1, (
                ("Câmara fria (unidade condensadora)", 1),
                ("Forno elétrico industrial", 1,
                 {"intervalo": "06:00 as 22:30", "duracao_min": 3.0, "duracao_max": 6.0}),
                ("Coifa / exaustão de cozinha", 1),
                ("Lava-louças industrial", 1),
                ("Expositor refrigerado", 2),
            ), essencial=True),
            ModeloComodo("Lavanderia", 1, (
                ("Lavadora industrial 20 kg", 2),
                ("Secadora industrial 20 kg", 2),
                ("Calandra", 1),
            )),
        ),
        eui_kwh_m2_ano=150.0,
        # Hóspede sem elevador e sem ar não volta, e reclama onde todo mundo lê.
        custo_interrupcao_brl_kwh=25.0,
    ),
    "escritorio": Segmento(
        "escritorio", "Escritório", "escritorio",
        (
            ModeloComodo("Estação de trabalho", 30, (
                ("Computador desktop", 1),
                ("Monitor", 2),
                ("Luminária LED 2×20 W", 1),
            )),
            ModeloComodo("Sala de reunião", 3, (
                ("Ar-condicionado split 18.000 BTU", 1),
                ("Projetor", 1),
                ("Luminária LED 2×20 W", 4),
            )),
            ModeloComodo("Infraestrutura", 1, (
                ("Servidor de rack", 2),
                ("Nobreak / rack de rede", 1),
                ("Sistema de segurança / CFTV", 1),
                ("Iluminação de emergência", 10),
            ), essencial=True),
            ModeloComodo("Copa", 1, (
                ("Geladeira doméstica", 1),
                ("Forno de micro-ondas", 1),
                ("Máquina de café expresso", 1),
                ("Bebedouro", 2),
            )),
        ),
        eui_kwh_m2_ano=120.0,
        # Hora parada de equipe é o prejuízo, e ela custa mais que a energia.
        custo_interrupcao_brl_kwh=18.0,
    ),
    "supermercado": Segmento(
        "supermercado", "Mercado / supermercado", "mercado_supermercado",
        (
            ModeloComodo("Salão de vendas", 1, (
                ("Luminária LED tubular 20 W", 120),
                ("Expositor refrigerado", 20),
                ("Ar-condicionado split 24.000 BTU", 6),
                ("Cortina de ar", 2),
            )),
            ModeloComodo("Câmaras frias", 1, (
                ("Câmara fria (unidade condensadora)", 3),
                ("Degelo de câmara fria", 3),
            ), essencial=True),
            ModeloComodo("Padaria e açougue", 1, (
                ("Forno elétrico industrial", 2),
                ("Freezer horizontal", 4),
                ("Coifa / exaustão de cozinha", 1),
            ), essencial=True),
            ModeloComodo("Retaguarda", 1, (
                ("Computador desktop", 6),
                ("Sistema de segurança / CFTV", 1),
                ("Iluminação de emergência", 15),
                ("Bomba d'água 3 CV", 1),
            ), essencial=True),
        ),
        eui_kwh_m2_ano=450.0,
        # Câmara fria parada estraga estoque, e o prejuízo não volta com a luz.
        custo_interrupcao_brl_kwh=60.0,
    ),
    "restaurante": Segmento(
        "restaurante", "Restaurante", "restaurante",
        (
            ModeloComodo("Cozinha", 1, (
                # A janela padrão do forno (10-14 h) sozinha jogava o pico do
                # restaurante para o almoço; na maioria deles o jantar é maior.
                ("Forno elétrico industrial", 1,
                 {"intervalo": "10:00 as 22:30", "duracao_min": 2.0, "duracao_max": 5.0}),
                ("Fritadeira elétrica", 2),
                ("Chapa / grill", 1),
                ("Coifa / exaustão de cozinha", 1),
                ("Lava-louças industrial", 1),
                ("Câmara fria (unidade condensadora)", 1),
            ), essencial=True),
            ModeloComodo("Salão", 1, (
                ("Ar-condicionado split 24.000 BTU", 3),
                ("Luminária LED 2×20 W", 20),
                ("TV LED 50\"", 2),
            )),
            ModeloComodo("Apoio", 1, (
                ("Freezer horizontal", 2),
                ("Expositor refrigerado", 2),
                ("Computador desktop", 2),
                ("Sistema de segurança / CFTV", 1),
            ), essencial=True),
        ),
        eui_kwh_m2_ano=350.0,
        # Serviço interrompido no horário de pico, mais o que estraga na câmara.
        custo_interrupcao_brl_kwh=45.0,
    ),
    "escola": Segmento(
        "escola", "Escola / colégio", "colegio",
        (
            ModeloComodo("Sala de aula", 20, (
                ("Ar-condicionado split 18.000 BTU", 1,
                 {"intervalo": "07:00 as 17:30", "duracao_min": 3.0, "duracao_max": 6.0}),
                ("Luminária LED 2×20 W", 6, {"intervalo": "07:00 as 17:30"}),
                ("Projetor", 1, {"intervalo": "07:00 as 17:30"}),
                ("Ventilador de teto", 2, {"intervalo": "07:00 as 17:30"}),
            )),
            ModeloComodo("Administração", 1, (
                ("Computador desktop", 8),
                ("Impressora multifuncional", 2),
                ("Ar-condicionado split 12.000 BTU", 3),
            )),
            ModeloComodo("Infraestrutura", 1, (
                ("Bomba d'água 3 CV", 2),
                ("Sistema de segurança / CFTV", 1),
                ("Iluminação de emergência", 25),
                ("Servidor de rack", 1),
            ), essencial=True),
            ModeloComodo("Cantina", 1, (
                ("Geladeira doméstica", 2),
                ("Forno elétrico industrial", 1),
                ("Freezer horizontal", 1),
            )),
        ),
        eui_kwh_m2_ano=80.0,
        # Aula suspensa se remarca; o prejuízo é de reputação, não de estoque.
        custo_interrupcao_brl_kwh=8.0,
    ),
    "hospital": Segmento(
        "hospital", "Hospital / clínica", "hospital_clinica",
        (
            ModeloComodo("Leito", 30, (
                ("Monitor multiparâmetro", 1),
                ("Bomba de infusão", 2),
                ("Lâmpada LED bulbo 9 W", 3),
                ("Ar-condicionado split 9.000 BTU", 1),
            ), essencial=True),
            ModeloComodo("Centro cirúrgico", 2, (
                ("Ar-condicionado split 24.000 BTU", 2),
                ("Luminária LED alta potência 150 W", 2),
                ("Monitor multiparâmetro", 2),
            ), essencial=True),
            ModeloComodo("Apoio e diagnóstico", 1, (
                ("Autoclave", 2),
                ("Raio-X", 1),
                ("Computador desktop", 10),
                ("Servidor de rack", 2),
            ), essencial=True),
            ModeloComodo("Infraestrutura", 1, (
                ("Bomba de recalque 5 CV", 2),
                ("Elevador social", 2),
                ("Iluminação de emergência", 40),
                ("Sistema de segurança / CFTV", 1),
            ), essencial=True),
        ),
        eui_kwh_m2_ano=250.0,
        # Risco assistencial; aqui o número não é econômico, é de segurança.
        custo_interrupcao_brl_kwh=150.0,
    ),
    "industria": Segmento(
        "industria", "Indústria leve", "industria_generica",
        (
            ModeloComodo("Produção", 1, (
                ("Motor trifásico 15 CV", 6),
                ("Compressor de ar 10 CV", 2),
                ("Ponte rolante", 1),
                ("Luminária LED alta potência 150 W", 40),
                ("Máquina de solda", 3),
            )),
            ModeloComodo("Administração", 1, (
                ("Computador desktop", 10),
                ("Ar-condicionado split 18.000 BTU", 3),
                ("Servidor de rack", 1),
            )),
            ModeloComodo("Utilidades", 1, (
                ("Bomba de recalque 5 CV", 2),
                ("Sistema de segurança / CFTV", 1),
                ("Iluminação de emergência", 30),
            ), essencial=True),
        ),
        eui_kwh_m2_ano=200.0,
        # Linha parada, mais o refugo do lote interrompido e o religamento.
        custo_interrupcao_brl_kwh=40.0,
    ),
    "frigorifico": Segmento(
        "frigorifico", "Frigorífico / câmara fria", "frigorifico",
        (
            ModeloComodo("Câmaras", 1, (
                ("Câmara fria (unidade condensadora)", 6),
                ("Degelo de câmara fria", 6),
                ("Chiller 10 TR", 2),
            ), essencial=True),
            ModeloComodo("Processamento", 1, (
                ("Motor trifásico 5 CV", 8),
                ("Luminária LED alta potência 150 W", 20),
                ("Compressor de ar 10 CV", 1),
            )),
            ModeloComodo("Utilidades", 1, (
                ("Bomba de recalque 5 CV", 2),
                ("Sistema de segurança / CFTV", 1),
                ("Iluminação de emergência", 20),
            ), essencial=True),
        ),
        eui_kwh_m2_ano=600.0,
        # A carga refrigerada é o próprio negócio, e ela não espera.
        custo_interrupcao_brl_kwh=120.0,
    ),
    "varejo": Segmento(
        "varejo", "Loja de varejo", "comercial_generico",
        (
            ModeloComodo("Loja", 1, (
                ("Luminária LED tubular 20 W", 60),
                ("Ar-condicionado split 24.000 BTU", 4),
                ("Letreiro / fachada", 2),
                ("Computador desktop", 4),
            )),
            ModeloComodo("Estoque e apoio", 1, (
                ("Luminária LED alta potência 150 W", 8),
                ("Sistema de segurança / CFTV", 1),
                ("Iluminação de emergência", 12),
                ("Bebedouro", 1),
            ), essencial=True),
        ),
        eui_kwh_m2_ano=180.0,
        # Venda perdida na hora, sem estoque perecível para piorar.
        custo_interrupcao_brl_kwh=20.0,
    ),
    "academia": Segmento(
        "academia", "Academia", "academia",
        (
            # Academia tem dois horários, e o da tarde é muito maior. Com as
            # janelas padrão do catálogo o pico caía às 7 h, puxado pelos
            # chuveiros do vestiário — o oposto do movimento real.
            ModeloComodo("Salão de musculação", 1, (
                ("Ar-condicionado split 24.000 BTU", 6,
                 {"intervalo": "16:00 as 22:30", "duracao_min": 4.0, "duracao_max": 6.0,
                  "probabilidade": 0.95}),
                ("Ar-condicionado split 24.000 BTU", 2,
                 {"intervalo": "06:00 as 10:00", "duracao_min": 2.0, "duracao_max": 4.0,
                  "probabilidade": 0.8}),
                # A academia não fecha ao meio-dia: o movimento cai, não zera.
                ("Ar-condicionado split 24.000 BTU", 3,
                 {"intervalo": "10:00 as 16:30", "duracao_min": 2.0, "duracao_max": 5.0,
                  "probabilidade": 0.7}),
                ("Luminária LED tubular 20 W", 50, {"intervalo": "06:00 as 22:30"}),
                ("TV LED 50\"", 6, {"intervalo": "06:00 as 22:30"}),
                ("Ventilador de teto", 8, {"intervalo": "16:00 as 22:30"}),
            )),
            ModeloComodo("Vestiários", 2, (
                ("Chuveiro elétrico 7.500 W", 6,
                 {"intervalo": "17:00 as 22:30", "probabilidade": 0.9}),
                ("Chuveiro elétrico 7.500 W", 2,
                 {"intervalo": "06:30 as 10:00", "probabilidade": 0.6}),
                ("Boiler elétrico 3.000 W", 1, {"intervalo": "14:00 as 20:00"}),
                ("Exaustor de banheiro", 4, {"intervalo": "06:00 as 22:30"}),
                ("Lâmpada LED bulbo 9 W", 10, {"intervalo": "06:00 as 22:30"}),
            )),
            ModeloComodo("Recepção e apoio", 1, (
                ("Computador desktop", 2),
                ("Sistema de segurança / CFTV", 1),
                ("Bebedouro", 2),
                ("Iluminação de emergência", 8),
            ), essencial=True),
        ),
        eui_kwh_m2_ano=160.0,
        # Aula cancelada e desconto na mensalidade; sem perecível.
        custo_interrupcao_brl_kwh=12.0,
    ),
}

SEGMENTOS = tuple(MODELOS)


def modelo_de_comodo(segmento: str, comodo: str) -> ModeloComodo:
    for modelo in MODELOS[segmento].comodos:
        if modelo.nome == comodo:
            return modelo
    raise KeyError(f"cômodo '{comodo}' não está no modelo de {segmento}")


# ============================================================================
# Curvas de carga típicas (base do Aurumcalc)
# ============================================================================
@dataclass(frozen=True)
class PerfilTipico:
    """Curva normalizada de 24 h, em p.u. do consumo diário (soma 1,0)."""

    id: str
    nome: str
    curva_pu: np.ndarray
    descricao: str = ""
    observacao: str = ""
    origem: str = "typical_load_profiles"

    @property
    def confiavel(self) -> bool:
        """False quando a base declara o perfil como sintético."""
        texto = f"{self.observacao} {self.origem}".lower()
        return "sintétic" not in texto and "synthetic" not in texto

    def curva_kw(self, consumo_diario_kwh: float) -> np.ndarray:
        """A curva escalada para um consumo diário conhecido, em kW por hora."""
        return self.curva_pu * float(consumo_diario_kwh)

    def por_minuto(self, consumo_diario_kwh: float) -> np.ndarray:
        """Mesma curva em 1.440 pontos, em watts — degrau de uma hora."""
        return np.repeat(self.curva_kw(consumo_diario_kwh), 60) * 1000.0

    def resumo(self) -> dict[str, Any]:
        return {
            "id": self.id, "nome": self.nome, "confiavel": self.confiavel,
            "hora_de_pico": int(np.argmax(self.curva_pu)),
            "fracao_no_pico": float(self.curva_pu.max()),
            "fator_de_carga": float(self.curva_pu.mean() / self.curva_pu.max()),
            "fracao_noturna_22_06": float(
                self.curva_pu[np.r_[22, 23, 0, 1, 2, 3, 4, 5]].sum()
            ),
        }


@lru_cache(maxsize=1)
def perfis_tipicos() -> dict[str, PerfilTipico]:
    """Todas as curvas típicas da base, indexadas por id."""
    if not ARQUIVO_TIPICOS.exists():
        return {}
    dados = json.loads(ARQUIVO_TIPICOS.read_text(encoding="utf-8"))
    perfis: dict[str, PerfilTipico] = {}
    for bruto in dados.get("profiles", []):
        curva = np.asarray(bruto.get("curva_24h_pu", []), dtype=float)
        if curva.size != 24 or curva.sum() <= 0:
            continue
        perfis[bruto["id"]] = PerfilTipico(
            id=bruto["id"],
            nome=bruto.get("nome", bruto["id"]),
            # Renormaliza: a base declara soma 1,0, mas arredondamento no JSON
            # deixa resíduo, e ele viraria erro na energia diária.
            curva_pu=curva / curva.sum(),
            descricao=bruto.get("descricao", ""),
            observacao=bruto.get("observacao", ""),
        )
    return perfis


def perfil_tipico(identificador: str) -> PerfilTipico:
    perfis = perfis_tipicos()
    if identificador not in perfis:
        raise KeyError(
            f"perfil típico '{identificador}' não existe. Disponíveis: {', '.join(sorted(perfis))}"
        )
    return perfis[identificador]


def perfil_confiavel(identificador: str) -> bool:
    try:
        return perfil_tipico(identificador).confiavel
    except KeyError:
        return False


def segmento_do_perfil(segmento: str) -> PerfilTipico | None:
    """A curva típica associada a um segmento do :data:`MODELOS`."""
    modelo = MODELOS.get(segmento)
    if modelo is None:
        return None
    try:
        return perfil_tipico(modelo.perfil_tipico_id)
    except KeyError:
        return None


# ----------------------------------------------------------------------------
# Calibração pela conta de luz
# ----------------------------------------------------------------------------
def calibrar_por_conta(
    perfil: PerfilTipico,
    consumo_mensal_kwh: float,
    dias_operacao_mes: int = 30,
) -> dict[str, Any]:
    """
    Escala a curva típica para bater com a conta de luz.

    É o caminho para quem não tem levantamento de equipamentos: a forma vem do
    padrão do segmento, o tamanho vem da fatura. O resultado é uma curva
    horária utilizável — pior que o levantamento, porque não tem variabilidade
    nem cauda (todo dia sai igual), e muito melhor que chutar.

    A ausência de cauda é a limitação que importa aqui: uma curva média não tem
    pico raro, e é o pico raro que decide o inversor. Um estudo feito só sobre
    ela subestima a potência necessária.
    """
    dias = max(1, int(dias_operacao_mes))
    diario = float(consumo_mensal_kwh) / dias
    curva = perfil.curva_kw(diario)
    return {
        "perfil": perfil.id,
        "confiavel": perfil.confiavel,
        "consumo_diario_kwh": diario,
        "curva_kw": curva,
        "curva_w_por_minuto": np.repeat(curva, 60) * 1000.0,
        "demanda_media_kw": float(curva.mean()),
        "demanda_maxima_kw": float(curva.max()),
        "fator_de_carga": float(curva.mean() / curva.max()) if curva.max() > 0 else 0.0,
        "aviso": (
            "Curva típica calibrada pela conta: sem variabilidade dia a dia e sem cauda "
            "de pico. Serve para estimar energia; subestima a potência de pico que o "
            "inversor precisa. Substitua pelo levantamento de equipamentos antes de "
            "fechar o dimensionamento."
        ),
    }


def comparar_com_perfil(
    perfil_medio_w: Sequence[float],
    perfil: PerfilTipico,
    passo_min: int = 1,
) -> dict[str, Any]:
    """
    Confere a curva simulada contra o padrão do segmento.

    Compara a **forma**, não o tamanho: as duas são normalizadas antes. O que
    interessa é se o levantamento coloca a energia nas mesmas horas que o
    padrão coloca — um supermercado que consome de madrugada tanto quanto ao
    meio-dia é quase sempre um erro de digitação numa janela de uso, e é muito
    mais barato descobrir isso aqui do que depois de dimensionar a bateria.

    O veredito sai da **energia por período do dia**, não da correlação hora a
    hora. A correlação sobre 24 pontos é frágil: um único pico legítimo que a
    curva de referência não modela — o chuveiro elétrico de um hotel às 7 h,
    por exemplo — derruba o coeficiente sem que nada esteja errado, e um alarme
    que dispara no caminho padrão é pior que alarme nenhum. Já a distribuição
    entre madrugada, manhã, tarde e noite só se desloca quando a energia foi
    mesmo parar na hora errada, que é o erro que se quer pegar.

    A correlação continua no resultado, como informação; ``divergencia_maxima``
    e ``divergencia_por_periodo_pp`` é que decidem.
    """
    serie = np.asarray(perfil_medio_w, dtype=float)
    por_hora = max(1, int(round(60 / passo_min)))
    horaria = serie.reshape(24, por_hora).mean(axis=1)
    total = horaria.sum()
    if total <= 0:
        raise ValueError("perfil simulado é nulo")
    simulada_pu = horaria / total

    diferenca = simulada_pu - perfil.curva_pu
    hora_pico_sim = int(np.argmax(simulada_pu))
    hora_pico_ref = int(np.argmax(perfil.curva_pu))

    blocos = simulada_pu.reshape(4, 6).sum(axis=1)
    blocos_ref = perfil.curva_pu.reshape(4, 6).sum(axis=1)
    por_periodo = {
        nome: float((blocos[i] - blocos_ref[i]) * 100)
        for i, nome in enumerate(PERIODOS)
    }
    pior_periodo = max(por_periodo, key=lambda k: abs(por_periodo[k]))

    return {
        "perfil": perfil.id,
        "perfil_nome": perfil.nome,
        "perfil_confiavel": perfil.confiavel,
        "curva_simulada_pu": simulada_pu,
        "curva_referencia_pu": perfil.curva_pu,
        "diferenca_pu": diferenca,
        "divergencia_maxima_pp": float(np.abs(diferenca).max() * 100),
        "hora_da_maior_divergencia": int(np.argmax(np.abs(diferenca))),
        "hora_de_pico_simulada": hora_pico_sim,
        "hora_de_pico_referencia": hora_pico_ref,
        "deslocamento_do_pico_h": hora_pico_sim - hora_pico_ref,
        "erro_absoluto_medio_pp": float(np.abs(diferenca).mean() * 100),
        "correlacao": float(np.corrcoef(simulada_pu, perfil.curva_pu)[0, 1]),
        "energia_por_periodo_pu": {n: float(blocos[i]) for i, n in enumerate(PERIODOS)},
        "divergencia_por_periodo_pp": por_periodo,
        "pior_periodo": pior_periodo,
        "divergencia_do_pior_periodo_pp": abs(por_periodo[pior_periodo]),
        "coerente": abs(por_periodo[pior_periodo]) <= LIMITE_DIVERGENCIA_PP,
    }
