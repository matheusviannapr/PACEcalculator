"""
Núcleo de simulação de demanda do **D² — Demanda e Dados**, tornado importável.

Este arquivo é uma extração fiel das linhas 381-999 de
``monte_carlo_hotel_app_final_v3.py`` do repositório
https://github.com/matheusviannapr/DemandaDados — o modelo de equipamentos, os
parsers de intervalo de uso, o ajuste sazonal e o laço Monte Carlo, sem uma
linha de Streamlit, matplotlib ou fpdf.

Por que copiar em vez de importar: lá o núcleo convive no mesmo módulo que a
interface, que executa ``st.set_page_config`` no import. Importar o arquivo
original obrigaria o estudo de baterias a subir um servidor Streamlit para
calcular uma curva de carga. A cópia é intencionalmente literal para que um
``diff`` contra o original continue legível quando o D² evoluir.

**Não edite este arquivo para mudar comportamento.** Correções de modelagem
pertencem ao D²; adaptações para o estudo de baterias pertencem a
:mod:`aurum.demanda.ensemble`.

Convenção herdada, válida em todo o módulo: potência em **watts**, tempo em
**minutos do dia** (0..1439), perfis com um ponto por minuto.
"""
from __future__ import annotations

import copy
import random
import re
from dataclasses import dataclass
from datetime import time
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Constantes de simulação e dimensionamento (originais, linhas 19-24)
TEMPO_TOTAL_PADRAO_MIN = 1440
PERCENTIL_DIMENSIONAMENTO = 95
MARGEM_SEGURANCA_DIMENSIONAMENTO = 1.20
IC_INFERIOR_PERCENTIL = 2.5
IC_SUPERIOR_PERCENTIL = 97.5


# --- Núcleo de simulação ---
IntervalType = Union[List[Tuple[int, int]], Callable[[], List[Tuple[int, int]]]]
ESTACOES_ANO = ["verão", "outono", "inverno", "primavera"]


def parse_time(time_str: str) -> int:
    parts = time_str.strip().split(":")
    if len(parts) == 1:
        hours = int(parts[0])
        minutes = 0
    else:
        hours = int(parts[0])
        minutes = int(parts[1])
    return hours * 60 + minutes


def parse_intervalo_fixo(interval_str: str) -> List[Tuple[int, int]]:
    interval_str = interval_str.replace("às", "as")
    parts = interval_str.split(" e ")
    intervals = []
    for part in parts:
        times = part.split(" as ")
        if len(times) != 2:
            continue
        start = parse_time(times[0])
        end = parse_time(times[1])
        intervals.append((start, end))
    return intervals


def _duracao_horas_para_passos(duracao_h: float, dt_min: int) -> int:
    return max(1, int(round((duracao_h * 60) / dt_min)))


def gerar_intervalo_uso(
    janela_inicio: str,
    janela_fim: str,
    duracao_min_h: float,
    duracao_max_h: float,
    dt_min: int = 1,
    probabilidade: float = 1.0,
    seed: Optional[int] = None,
    on_overflow: str = "clamp",
) -> Optional[Tuple[int, int]]:
    """
    Retorna (inicio_idx, fim_idx) em passos discretos (não em minutos).

    Regras implementadas:
    - probabilidade é aplicada antes do sorteio de duração/início;
    - duração é sorteada de forma uniforme contínua em horas;
    - n_passos = round(duracao_h * 60 / dt_min);
    - se duração não couber na janela: aplica clamp (reduz para caber) quando on_overflow='clamp'.
    """
    if dt_min <= 0:
        raise ValueError("dt_min deve ser > 0")

    rng = random.Random(seed) if seed is not None else random

    if rng.random() > probabilidade:
        return None

    if duracao_min_h > duracao_max_h:
        raise ValueError("duracao_min_h não pode ser maior que duracao_max_h")

    janela_inicio_min = parse_time(janela_inicio)
    janela_fim_min = parse_time(janela_fim)
    if janela_fim_min <= janela_inicio_min:
        raise ValueError("janela_fim deve ser maior que janela_inicio")

    duracao_h = rng.uniform(duracao_min_h, duracao_max_h)
    n_passos = _duracao_horas_para_passos(duracao_h, dt_min)

    janela_inicio_idx = janela_inicio_min // dt_min
    janela_fim_idx = janela_fim_min // dt_min
    tamanho_janela_passos = janela_fim_idx - janela_inicio_idx

    if n_passos > tamanho_janela_passos:
        if on_overflow == "clamp":
            n_passos = tamanho_janela_passos
        else:
            return None

    latest_start = janela_fim_idx - n_passos
    inicio_idx = rng.randint(janela_inicio_idx, latest_start)
    fim_idx = inicio_idx + n_passos
    return inicio_idx, fim_idx


def parse_intervalo_dinamico_split(interval_str: str) -> Callable[[], List[Tuple[int, int]]]:
    """
    Compatibilidade com formato legado:
      "Início entre 10:30-14:00, duração 2"
    onde duração está em horas e é fragmentada em 1 a 3 segmentos.
    """
    parts = interval_str.split(";")

    def dynamic_intervals():
        intervals = []
        for part in parts:
            part = part.strip()
            m = re.search(r"(?i)entre\s*([\d:]+)\s*-\s*([\d:]+)", part)
            m2 = re.search(r"(?i)duração\s*(\d+(?:[\.,]\d+)?)", part)
            if m and m2:
                start_lower = parse_time(m.group(1))
                start_upper = parse_time(m.group(2))
                total_duration_minutes = int(float(m2.group(1).replace(',', '.')) * 60)

                num_segments = random.randint(1, 3)
                if total_duration_minutes < 2:
                    segments = [total_duration_minutes]
                else:
                    available = total_duration_minutes - 1
                    num_cuts = min(num_segments - 1, available)
                    if num_cuts > 0:
                        cut_points = sorted(random.sample(range(1, total_duration_minutes), num_cuts))
                        segments = []
                        previous = 0
                        for cp in cut_points:
                            segments.append(cp - previous)
                            previous = cp
                        segments.append(total_duration_minutes - previous)
                    else:
                        segments = [total_duration_minutes]

                first_start = random.randint(start_lower, start_upper)
                current_start = first_start
                for seg_duration in segments:
                    intervals.append((current_start, current_start + seg_duration))
                    current_start = current_start + seg_duration + random.randint(0, 30)
        return intervals if intervals else [(0, 60)]

    return dynamic_intervals


def parse_janelas_operacao(intervalo_str: str) -> List[Tuple[int, int]]:
    """
    Todas as janelas de operação declaradas, em minutos.

    O formato aceita mais de uma janela separada por ``" e "``. Um micro-ondas
    de "11:30 as 14:00 e 18:30 as 21:30" é usado no almoço **e** no jantar; até
    aqui a segunda janela era descartada em silêncio, e a carga da casa saía
    com um cume só.
    """
    intervalos = parse_intervalo_fixo(intervalo_str)
    if not intervalos:
        raise ValueError(
            "Para duração intervalar, use intervalo no formato 'HH:MM as HH:MM' para janela de operação."
        )
    return intervalos


def parse_janela_operacao(intervalo_str: str) -> Tuple[int, int]:
    """A primeira janela declarada. Mantido para quem só sabe lidar com uma."""
    return parse_janelas_operacao(intervalo_str)[0]


def criar_gerador_duracao_intervalar(
    janela_inicio_min: int,
    janela_fim_min: int,
    duracao_min_h: float,
    duracao_max_h: float,
    dt_min: int = 1,
    probabilidade: float = 1.0,
    seed: Optional[int] = None,
    on_overflow: str = "clamp",
) -> Callable[[], List[Tuple[int, int]]]:
    """Gera um único intervalo (em minutos) por execução, com duração aleatória em intervalo."""

    def _generator() -> List[Tuple[int, int]]:
        intervalo = gerar_intervalo_uso(
            janela_inicio=f"{janela_inicio_min // 60:02d}:{janela_inicio_min % 60:02d}",
            janela_fim=f"{janela_fim_min // 60:02d}:{janela_fim_min % 60:02d}",
            duracao_min_h=duracao_min_h,
            duracao_max_h=duracao_max_h,
            dt_min=dt_min,
            probabilidade=probabilidade,
            seed=seed,
            on_overflow=on_overflow,
        )
        if intervalo is None:
            return []
        inicio_idx, fim_idx = intervalo
        return [(inicio_idx * dt_min, fim_idx * dt_min)]

    return _generator


@dataclass
class Equipamento:
    nome: str
    potencia: float
    quantidade: int
    intervalos: IntervalType
    probabilidade: float = 1.0
    fator_demanda: float = 1.0
    probabilisticado_no_intervalo: bool = False
    #: Como refazer o gerador de janelas com outra probabilidade.
    #:
    #: Um item de intervalo dinâmico com duração não guarda a probabilidade num
    #: campo: ela está dentro do gerador, que sorteia se o dia tem utilização
    #: antes de sortear a hora. Mexer em `probabilidade` depois de construído
    #: não muda nada — e era o que a sazonalidade por probabilidade fazia,
    #: silenciosamente, justamente no ar-condicionado.
    fabrica_intervalos: Optional[Callable[[float], IntervalType]] = None

    def simula_carga(self, tempo_total: int = 1440):
        carga = np.zeros(tempo_total)
        intervals = self.intervalos() if callable(self.intervalos) else self.intervalos

        if self.probabilisticado_no_intervalo or np.random.rand() < self.probabilidade:
            potencia_efetiva = self.potencia * self.fator_demanda
            for inicio, fim in intervals:
                inicio = max(0, inicio)
                fim = min(tempo_total, fim)
                carga[inicio:fim] += potencia_efetiva * self.quantidade
        return carga


@dataclass
class Comodo:
    nome: str
    equipamentos: List[Equipamento]

    def simula_carga(self, tempo_total: int = 1440):
        carga_total = np.zeros(tempo_total)
        for eq in self.equipamentos:
            carga_total += eq.simula_carga(tempo_total)
        return carga_total


def _get_duration_bounds_h(row: pd.Series) -> Tuple[Optional[float], Optional[float]]:
    duracao_min = row.get("duracao_min")
    duracao_max = row.get("duracao_max")
    duracao_legacy = row.get("duracao")

    if pd.notna(duracao_min) and pd.notna(duracao_max):
        return float(duracao_min), float(duracao_max)
    if pd.notna(duracao_legacy):
        d = float(duracao_legacy)
        return d, d
    return None, None


def criar_gerador_multijanela(
    janelas: List[Tuple[int, int]],
    duracao_min_h: float,
    duracao_max_h: float,
    dt_min: int = 1,
    probabilidade: float = 1.0,
    on_overflow: str = "clamp",
) -> Callable[[], List[Tuple[int, int]]]:
    """
    Uma utilização por janela declarada, cada uma com seu próprio sorteio.

    Com uma janela só o comportamento é idêntico ao de
    :func:`criar_gerador_duracao_intervalar` -- é o caso de quase todo
    equipamento levantado em campo. Com duas, o aparelho pode ser usado nas
    duas, e é isso que produz os dois cumes de refeição: sortear qual das
    janelas devolveria de novo um cume só, deslocado de um dia para o outro.
    """
    geradores = [
        criar_gerador_duracao_intervalar(
            janela_inicio_min=inicio,
            janela_fim_min=fim,
            duracao_min_h=duracao_min_h,
            duracao_max_h=duracao_max_h,
            dt_min=dt_min,
            probabilidade=probabilidade,
            on_overflow=on_overflow,
        )
        for inicio, fim in janelas
    ]

    def _generator() -> List[Tuple[int, int]]:
        intervalos: List[Tuple[int, int]] = []
        for gerar in geradores:
            intervalos.extend(gerar())
        return intervalos

    return _generator


def cria_comodo_da_planilha(sheet_df: pd.DataFrame, comodo_nome: str) -> Comodo:
    equipamentos = []
    for _, row in sheet_df.iterrows():
        nome = row["Equipamento"]
        potencia = float(row["Potência"])
        quantidade = int(row["Quantidade"])
        tipo_intervalo = str(row["Tipo de intervalo"]).strip().lower()
        intervalo_str = str(row["intervalo"]).strip()
        probabilidade = float(row["probabilidade"])
        fd = float(row["FD"])
        modo_fixo = str(row.get("modo_fixo", "FIXO_100%")).strip().upper()

        duracao_min_h, duracao_max_h = _get_duration_bounds_h(row)

        if duracao_min_h is not None and duracao_max_h is not None and duracao_min_h > duracao_max_h:
            raise ValueError(
                f"Equipamento '{nome}' (cômodo '{comodo_nome}'): a duração mínima "
                f"({duracao_min_h}h) é maior que a duração máxima ({duracao_max_h}h). "
                "Corrija as colunas de duração na planilha."
            )

        if not (0.0 <= probabilidade <= 1.0):
            raise ValueError(
                f"Equipamento '{nome}' (cômodo '{comodo_nome}'): probabilidade inválida "
                f"({probabilidade}). O valor deve estar entre 0.0 e 1.0."
            )

        probabilisticado_no_intervalo = False
        fabrica = None

        def _fabrica_multijanela(intervalo_str=intervalo_str,
                                 duracao_min_h=duracao_min_h,
                                 duracao_max_h=duracao_max_h):
            """Gerador de janelas em função da probabilidade, para a sazonalidade."""
            def refazer(prob: float):
                return criar_gerador_multijanela(
                    janelas=parse_janelas_operacao(intervalo_str),
                    duracao_min_h=duracao_min_h,
                    duracao_max_h=duracao_max_h,
                    probabilidade=max(0.0, min(1.0, float(prob))),
                    dt_min=1,
                    on_overflow="clamp",
                )
            return refazer

        if tipo_intervalo == "fixo":
            if duracao_min_h is not None and modo_fixo == "FIXO_DURACAO_INTERVALAR":
                fabrica = _fabrica_multijanela()
                intervalos = fabrica(probabilidade)
                probabilisticado_no_intervalo = True
            else:
                intervalos = parse_intervalo_fixo(intervalo_str)
        elif tipo_intervalo == "dinâmico":
            if duracao_min_h is not None:
                fabrica = _fabrica_multijanela()
                intervalos = fabrica(probabilidade)
                probabilisticado_no_intervalo = True
            else:
                intervalos = parse_intervalo_dinamico_split(intervalo_str)
        else:
            intervalos = parse_intervalo_fixo(intervalo_str)

        eq = Equipamento(
            nome=nome,
            potencia=potencia,
            quantidade=quantidade,
            intervalos=intervalos,
            probabilidade=probabilidade,
            fator_demanda=fd,
            probabilisticado_no_intervalo=probabilisticado_no_intervalo,
            fabrica_intervalos=fabrica,
        )
        equipamentos.append(eq)
    return Comodo(nome=comodo_nome, equipamentos=equipamentos)


def cria_comodos_do_excel(uploaded_file) -> List[Comodo]:
    sheets = pd.read_excel(uploaded_file, sheet_name=None)
    return [cria_comodo_da_planilha(df, sheet_name) for sheet_name, df in sheets.items()]


def cria_comodos_do_dataframe(df_dict: dict) -> List[Comodo]:
    comodos = []
    for comodo_nome, df in df_dict.items():
        if not df.empty:
            comodos.append(cria_comodo_da_planilha(df, comodo_nome))
    return comodos


def criar_cenario_exemplo(tipo: str = "hotel") -> dict:
    """Retorna um dicionário {nome_comodo: DataFrame} com dados fictícios
    prontos para uso, para que o usuário veja o programa funcionando sem
    precisar configurar nada."""
    if tipo == "escritorio":
        sala_reuniao = pd.DataFrame({
            'Equipamento': ['Ar Condicionado', 'Iluminação', 'Projetor'],
            'Potência': [1800, 80, 250],
            'Quantidade': [1, 6, 1],
            'Tipo de intervalo': ['dinâmico', 'fixo', 'fixo'],
            'intervalo': ['08:00 as 18:00', '08:00 as 19:00', '09:00 as 17:00'],
            'probabilidade': [0.6, 1.0, 0.4],
            'FD': [0.7, 1.0, 0.9],
            'duracao_min': [1.0, np.nan, np.nan],
            'duracao_max': [2.0, np.nan, np.nan],
            'modo_fixo': [np.nan, 'FIXO_100%', 'FIXO_100%'],
        })
        estacao_trabalho = pd.DataFrame({
            'Equipamento': ['Computador', 'Monitor', 'Iluminação'],
            'Potência': [180, 40, 60],
            'Quantidade': [1, 2, 2],
            'Tipo de intervalo': ['fixo', 'fixo', 'fixo'],
            'intervalo': ['08:00 as 18:00', '08:00 as 18:00', '08:00 as 19:00'],
            'probabilidade': [0.9, 0.9, 1.0],
            'FD': [0.85, 0.85, 1.0],
            'duracao_min': [np.nan, np.nan, np.nan],
            'duracao_max': [np.nan, np.nan, np.nan],
            'modo_fixo': ['FIXO_100%', 'FIXO_100%', 'FIXO_100%'],
        })
        return {"Sala de Reunião": sala_reuniao, "Estação de Trabalho": estacao_trabalho}

    # tipo == "hotel" (padrão)
    quarto_standard = pd.DataFrame({
        'Equipamento': ['Ar Condicionado', 'Iluminação', 'TV'],
        'Potência': [2000, 100, 150],
        'Quantidade': [1, 4, 1],
        'Tipo de intervalo': ['dinâmico', 'fixo', 'fixo'],
        'intervalo': ['07:00 as 22:00', '18:00 as 23:00', '19:00 as 23:00'],
        'probabilidade': [0.8, 1.0, 0.9],
        'FD': [0.8, 1.0, 1.0],
        'duracao_min': [1.0, np.nan, np.nan],
        'duracao_max': [3.0, np.nan, np.nan],
        'modo_fixo': [np.nan, 'FIXO_100%', 'FIXO_DURACAO_INTERVALAR'],
    })
    area_comum = pd.DataFrame({
        'Equipamento': ['Iluminação', 'Elevador', 'Frigobar'],
        'Potência': [120, 3500, 90],
        'Quantidade': [10, 1, 1],
        'Tipo de intervalo': ['fixo', 'dinâmico', 'fixo'],
        'intervalo': ['18:00 as 06:00', '06:00 as 23:00', '00:00 as 23:59'],
        'probabilidade': [1.0, 0.3, 1.0],
        'FD': [1.0, 0.6, 1.0],
        'duracao_min': [np.nan, 0.05, np.nan],
        'duracao_max': [np.nan, 0.2, np.nan],
        'modo_fixo': ['FIXO_100%', np.nan, 'FIXO_100%'],
    })
    return {"Quarto Standard": quarto_standard, "Área Comum": area_comum}


def cria_comodos_individualizados(comodos: List[Comodo], instancias_por_comodo: dict) -> List[Comodo]:
    comodos_individualizados = []
    for comodo in comodos:
        qtd = instancias_por_comodo.get(comodo.nome, 1)
        for i in range(qtd):
            equipamentos_copia = []
            for eq in comodo.equipamentos:
                equipamentos_copia.append(
                    Equipamento(
                        nome=eq.nome,
                        potencia=eq.potencia,
                        quantidade=eq.quantidade,
                        intervalos=eq.intervalos,
                        probabilidade=eq.probabilidade,
                        fator_demanda=eq.fator_demanda,
                        probabilisticado_no_intervalo=eq.probabilisticado_no_intervalo,
                    )
                )
            comodos_individualizados.append(Comodo(nome=f"{comodo.nome}.{i+1}", equipamentos=equipamentos_copia))
    return comodos_individualizados


def listar_equipamentos(comodos: List[Comodo]) -> List[dict]:
    """Lista equipamentos por cômodo, preservando a estrutura atual do app."""
    equipamentos = []
    for comodo in comodos:
        for eq in comodo.equipamentos:
            equipamentos.append(
                {
                    "id": f"{comodo.nome}::{eq.nome}",
                    "comodo": comodo.nome,
                    "equipamento": eq.nome,
                }
            )
    return equipamentos


def aplicar_ajuste_sazonal(
    comodos: List[Comodo],
    ajustes_sazonais: Dict[str, dict],
    estacao: str,
) -> List[Comodo]:
    """
    Retorna uma cópia dos cômodos com ajuste sazonal aplicado.

    O ajuste aceita um ``"modo"``, e a escolha muda a forma da curva:

    ``"fator_demanda"`` (padrão)
        O fator multiplica a **intensidade**. Serve para carga que de fato
        modula com a estação sem mudar de horário — um chuveiro que esquenta
        mais no inverno, uma bomba que trabalha contra água mais fria.

    ``"probabilidade"``
        O fator multiplica a **frequência de uso**, e só transborda para a
        intensidade quando a probabilidade satura em 1,0. É o certo para
        climatização: no inverno o ar-condicionado não liga mais fraco, ele não
        liga. Aplicado sobre o fator de demanda, −45% de inverno vira "roda as
        mesmas 5,5 horas todas as noites, a 55% da potência", e o modelo põe um
        platô de ar-condicionado em todas as madrugadas de julho. Quem
        dimensiona banco de bateria pela madrugada recebia uma carga que não
        existe.

    O padrão preserva o comportamento de todo ajuste já escrito.
    """
    comodos_sazonais = copy.deepcopy(comodos)

    for comodo in comodos_sazonais:
        for eq in comodo.equipamentos:
            chave_eq = f"{comodo.nome}::{eq.nome}"
            ajuste_eq = ajustes_sazonais.get(chave_eq, {})
            if not ajuste_eq.get("ativo", False):
                continue

            percentual = float(ajuste_eq.get(estacao, 0.0))
            multiplicador = 1.0 + (percentual / 100.0)
            multiplicador = max(0.0, multiplicador)

            modo = str(ajuste_eq.get("modo", "fator_demanda")).strip().lower()
            if modo != "probabilidade":
                eq.fator_demanda *= multiplicador
                continue

            alvo = eq.probabilidade * multiplicador
            eq.probabilidade = min(1.0, alvo)
            if alvo > 1.0:
                # O que não coube na frequência vira intensidade: um verão que
                # dobra o uso de um aparelho já ligado toda noite só pode
                # aparecer como mais potência e mais tempo.
                eq.fator_demanda *= alvo

            if eq.probabilisticado_no_intervalo:
                # Aqui a probabilidade mora dentro do gerador de janelas, e o
                # campo é só registro. Sem refazer o gerador, o ajuste não
                # alcançaria justamente o ar-condicionado, que é dinâmico.
                if eq.fabrica_intervalos is None:
                    eq.fator_demanda *= multiplicador
                else:
                    eq.intervalos = eq.fabrica_intervalos(eq.probabilidade)

    return comodos_sazonais


def simula_carga_total(
    comodos: List[Comodo],
    instancias_por_comodo: dict,
    num_simulacoes: int = 1000,
    tempo_total: int = 1440,
    coletar_detalhes_pico: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[List[dict]]]:
    picos = []
    perfis = []
    consumos_diarios = []
    detalhes_pico = [] if coletar_detalhes_pico else None
    comodos_individualizados = cria_comodos_individualizados(comodos, instancias_por_comodo)

    for _ in range(num_simulacoes):
        load_total = np.zeros(tempo_total)
        cargas_equipamentos = []

        for comodo in comodos_individualizados:
            for eq in comodo.equipamentos:
                carga_eq = eq.simula_carga(tempo_total)
                load_total += carga_eq
                if coletar_detalhes_pico:
                    cargas_equipamentos.append((comodo.nome, eq.nome, carga_eq))

        minuto_pico = int(np.argmax(load_total))
        valor_pico = float(np.max(load_total))

        picos.append(valor_pico)
        perfis.append(load_total)
        consumos_diarios.append(np.sum(load_total) / 1000 / 60)

        if coletar_detalhes_pico:
            equipamentos_ativos = []
            for nome_comodo, nome_equipamento, carga_eq in cargas_equipamentos:
                carga_no_pico = float(carga_eq[minuto_pico])
                if carga_no_pico > 0:
                    equipamentos_ativos.append({
                        "comodo": nome_comodo,
                        "equipamento": nome_equipamento,
                        "carga_w": carga_no_pico,
                    })

            equipamentos_ativos.sort(key=lambda item: item["carga_w"], reverse=True)
            detalhes_pico.append(
                {
                    "minuto_pico": minuto_pico,
                    "valor_pico": valor_pico,
                    "equipamentos_ativos": equipamentos_ativos,
                }
            )

    return np.array(picos), np.array(perfis), np.array(consumos_diarios), detalhes_pico


def _tempo_para_minutos(valor_tempo: time) -> int:
    return int(valor_tempo.hour) * 60 + int(valor_tempo.minute)


def _mascara_horario_ponta(tempo_total: int, inicio_ponta_min: int, fim_ponta_min: int) -> np.ndarray:
    """Cria máscara booleana por ponto da curva (True = ponta)."""
    minutos_do_dia = np.arange(tempo_total) % 1440

    if inicio_ponta_min == fim_ponta_min:
        return np.zeros(tempo_total, dtype=bool)

    if inicio_ponta_min < fim_ponta_min:
        return (minutos_do_dia >= inicio_ponta_min) & (minutos_do_dia < fim_ponta_min)

    # Caso ponta atravesse meia-noite (ex.: 22:00 às 01:00)
    return (minutos_do_dia >= inicio_ponta_min) | (minutos_do_dia < fim_ponta_min)


def calcular_indicadores_ponta_fora(
    perfis: np.ndarray,
    tempo_total: int,
    inicio_ponta_min: int,
    fim_ponta_min: int,
) -> dict:
    """
    Calcula indicadores a partir da curva típica (perfil médio).
    Energia em kWh: soma(Potência_W * Δt_h) / 1000.
    """
    perfil_medio = np.mean(perfis, axis=0)
    mascara_ponta = _mascara_horario_ponta(tempo_total, inicio_ponta_min, fim_ponta_min)
    mascara_fora = ~mascara_ponta
    delta_t_h = 1.0 / 60.0  # passo da simulação em minutos

    demanda_media_total = float(np.mean(perfil_medio))
    demanda_media_ponta = float(np.mean(perfil_medio[mascara_ponta])) if np.any(mascara_ponta) else 0.0
    demanda_media_fora = float(np.mean(perfil_medio[mascara_fora])) if np.any(mascara_fora) else 0.0

    demanda_max_ponta = float(np.max(perfil_medio[mascara_ponta])) if np.any(mascara_ponta) else 0.0
    demanda_max_fora = float(np.max(perfil_medio[mascara_fora])) if np.any(mascara_fora) else 0.0

    energia_total = float(np.sum(perfil_medio) * delta_t_h / 1000.0)
    energia_ponta = float(np.sum(perfil_medio[mascara_ponta]) * delta_t_h / 1000.0) if np.any(mascara_ponta) else 0.0
    energia_fora = float(np.sum(perfil_medio[mascara_fora]) * delta_t_h / 1000.0) if np.any(mascara_fora) else 0.0

    return {
        "demanda_media_total_w": demanda_media_total,
        "demanda_media_ponta_w": demanda_media_ponta,
        "demanda_media_fora_w": demanda_media_fora,
        "demanda_max_ponta_w": demanda_max_ponta,
        "demanda_max_fora_w": demanda_max_fora,
        "energia_total_kwh": energia_total,
        "energia_ponta_kwh": energia_ponta,
        "energia_fora_kwh": energia_fora,
    }


def calcular_estatisticas_resumo(
    resultados: dict,
    instancias_por_comodo: dict,
    comodos_originais: Optional[List["Comodo"]],
) -> dict:
    """
    Centraliza o cálculo das estatísticas-resumo reutilizadas pelo PDF e pelo LaTeX,
    evitando duplicação e divergência entre os dois formatos de relatório.
    """
    picos = resultados["picos"]
    perfis = resultados["perfis"]
    consumos = resultados["consumos"]

    pico_medio = float(np.mean(picos))
    pico_max = float(np.max(picos))
    pico_min = float(np.min(picos))
    pico_95 = float(np.percentile(picos, PERCENTIL_DIMENSIONAMENTO))
    desvio_padrao = float(np.std(picos))
    coef_variacao = (desvio_padrao / pico_medio) * 100 if pico_medio > 0 else 0.0
    consumo_medio = float(np.mean(consumos))

    media_por_minuto = np.mean(perfis, axis=0)
    fator_carga_medio = (
        (np.mean(media_por_minuto) / np.max(media_por_minuto)) * 100
        if np.max(media_por_minuto) > 0
        else 0.0
    )

    ic_inferior = float(np.percentile(picos, IC_INFERIOR_PERCENTIL))
    ic_superior = float(np.percentile(picos, IC_SUPERIOR_PERCENTIL))
    ic_consumo_inferior = float(np.percentile(consumos, IC_INFERIOR_PERCENTIL))
    ic_consumo_superior = float(np.percentile(consumos, IC_SUPERIOR_PERCENTIL))
    capacidade_recomendada = pico_95 * MARGEM_SEGURANCA_DIMENSIONAMENTO

    total_potencia_instalada = 0.0
    if comodos_originais:
        for comodo_obj in comodos_originais:
            total_potencia_instalada += sum(
                eq.potencia * eq.quantidade for eq in comodo_obj.equipamentos
            ) * instancias_por_comodo.get(comodo_obj.nome, 1)
    fator_diversidade = pico_medio / total_potencia_instalada if total_potencia_instalada > 0 else 0.0

    if coef_variacao < 15:
        interpretacao_diversidade = "baixa variabilidade, comportamento previsível e estável"
    elif coef_variacao < 30:
        interpretacao_diversidade = "variabilidade moderada, comportamento típico para instalações hoteleiras"
    else:
        interpretacao_diversidade = "alta variabilidade, requer monitoramento e análise adicional"

    if fator_carga_medio > 70:
        interpretacao_fator_carga = "excelente utilização da infraestrutura instalada"
    elif fator_carga_medio > 50:
        interpretacao_fator_carga = "boa utilização da infraestrutura"
    else:
        interpretacao_fator_carga = "utilização baixa, com grande diferença entre carga média e pico"

    return {
        "pico_medio": pico_medio,
        "pico_max": pico_max,
        "pico_min": pico_min,
        "pico_95": pico_95,
        "desvio_padrao": desvio_padrao,
        "coef_variacao": coef_variacao,
        "consumo_medio": consumo_medio,
        "fator_carga_medio": fator_carga_medio,
        "ic_inferior": ic_inferior,
        "ic_superior": ic_superior,
        "ic_consumo_inferior": ic_consumo_inferior,
        "ic_consumo_superior": ic_consumo_superior,
        "capacidade_recomendada": capacidade_recomendada,
        "fator_diversidade": fator_diversidade,
        "interpretacao_diversidade": interpretacao_diversidade,
        "interpretacao_fator_carga": interpretacao_fator_carga,
    }

