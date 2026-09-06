"""
Subcomandos de bateria da linha de comando do Aurum.

    aurum bateria estudo --lat -25.43 --lon -49.27 --kwp 40 --saida estudos/hotel
    aurum bateria excedencia --cargas cargas.xlsx
    aurum bateria catalogo --criar

Registrado em :mod:`aurum.cli`; este módulo não define ``main`` próprio.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def _progresso(mensagem: str, fracao: float) -> None:
    print(f"[{fracao * 100:3.0f}%] {mensagem}", file=sys.stderr)


def _mostrar(df: pd.DataFrame, casas: int = 3) -> None:
    with pd.option_context("display.width", 200, "display.max_columns", 40):
        print(df.round(casas).to_string(index=False))


# ----------------------------------------------------------------------------
def comando_catalogo(args: argparse.Namespace) -> int:
    """Mostra — ou cria — o catálogo de baterias e inversores híbridos."""
    from .catalogo import CatalogoError, carregar_catalogo, criar_planilha_modelo

    if args.criar:
        caminho = criar_planilha_modelo(args.planilha)
        print(f"Planilha semente gravada em {caminho}")
        print("Todos os campos vêm marcados 'a conferir'. Substitua pelos datasheets")
        print("dos seus fornecedores antes de usar em proposta comercial.")
        return 0

    try:
        base = carregar_catalogo(args.planilha)
    except CatalogoError as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 1

    print(f"Base: {base.caminho}\n")
    print(f"{'BATERIA':<46}{'kWh':>7}{'ÚTIL':>7}{'DoD':>6}{'ηRT':>6}{'kW':>7}{'CICLOS':>8}")
    print("-" * 90)
    for bateria in sorted(base.baterias, key=lambda b: -b.capacidade_nominal_kwh):
        print(
            f"{str(bateria)[:45]:<46}{bateria.capacidade_nominal_kwh:>7.2f}"
            f"{bateria.energia_util_kwh:>7.2f}{bateria.profundidade_descarga_percent:>5.0f}%"
            f"{bateria.eficiencia_roundtrip_percent:>5.0f}%"
            f"{bateria.potencia_descarga_max_kw:>7.2f}{bateria.ciclos_vida:>8}"
        )
    print()
    print(f"{'INVERSOR HÍBRIDO':<46}{'kW':>7}{'PICO':>7}{'SURTO':>7}{'kW FV':>7}{'FASES':>7}")
    print("-" * 90)
    for inversor in base.inversores_ordenados():
        print(
            f"{str(inversor)[:45]:<46}{inversor.potencia_ca_nominal_kw:>7.1f}"
            f"{inversor.potencia_ca_pico_kw:>7.1f}{inversor.duracao_pico_s:>6.0f}s"
            f"{inversor.potencia_fv_max_kw:>7.1f}{inversor.fases:>7}"
        )
    resumo = base.resumo()
    if resumo["a_conferir"]:
        print(
            f"\n{resumo['a_conferir']} de {resumo['baterias'] + resumo['inversores']} itens "
            "ainda estão marcados 'a conferir' na planilha."
        )
    return 0


# ----------------------------------------------------------------------------
def _ensemble_dos_args(args: argparse.Namespace):
    from ..demanda import carregar_cenario_excel, cenario_exemplo, simular_ensemble

    if args.cargas:
        comodos = carregar_cenario_excel(args.cargas)
        instancias = {c.nome: 1 for c in comodos}
    else:
        comodos, instancias = cenario_exemplo(args.demo)
        print(
            f"Nenhuma planilha de cargas informada: usando o cenário de demonstração "
            f"'{args.demo}' do D². Use --cargas para o levantamento real.",
            file=sys.stderr,
        )
    if args.essenciais:
        nomes = {n.strip().lower() for n in args.essenciais.split(",")}
        comodos = [c for c in comodos if c.nome.strip().lower() in nomes]
        if not comodos:
            raise SystemExit("--essenciais não bateu com nenhum cômodo da planilha.")
        instancias = {c.nome: instancias.get(c.nome, 1) for c in comodos}
    return simular_ensemble(comodos, instancias, num_simulacoes=args.simulacoes)


def comando_excedencia(args: argparse.Namespace) -> int:
    """Cruza a curva de excedência de pico com o catálogo de inversores."""
    from .catalogo import carregar_catalogo
    from .excedencia import avaliar_inversores, potencia_para_excedencia, tabela_por_faixa_horaria

    ensemble = _ensemble_dos_args(args).reamostrar(args.passo)
    base = carregar_catalogo(args.catalogo)

    print("\n=== POTÊNCIA POR PROBABILIDADE DE EXCEDÊNCIA ===")
    _mostrar(potencia_para_excedencia(ensemble), 2)
    print("\n  pico_diario_kw   : excedida em uma fração dos DIAS  (dimensiona)")
    print("  carga_instantanea: excedida em uma fração do TEMPO  (satura o inversor)")

    print("\n=== DIAGNÓSTICO DOS INVERSORES ===")
    diagnostico = avaliar_inversores(base, ensemble)
    _mostrar(
        diagnostico[
            [
                "modelo", "nominal_kw", "pico_kw",
                "prob_pico_diario_acima_nominal", "prob_pico_diario_acima_surto",
                "fracao_do_tempo_acima_nominal", "duracao_media_min", "veredito",
            ]
        ]
    )

    if args.horaria:
        print("\n=== EXCEDÊNCIA POR HORA DO DIA ===")
        potencias = sorted({round(i.potencia_ca_nominal_kw, 1) for i in base.inversores})
        _mostrar(tabela_por_faixa_horaria(ensemble, potencias), 3)

    if args.saida:
        pasta = Path(args.saida)
        pasta.mkdir(parents=True, exist_ok=True)
        diagnostico.to_csv(pasta / "diagnostico_inversores.csv", index=False, sep=";", decimal=",")
        print(f"\nGravado em {pasta.resolve()}")
    return 0


# ----------------------------------------------------------------------------
def comando_estudo(args: argparse.Namespace) -> int:
    """Roda o estudo completo e grava o pacote de relatório."""
    from ..demanda import carregar_cenario_excel, cenario_exemplo
    from .apagao import MalhaApagao
    from .economia import PremissasBateria
    from .estudo import ConfiguracaoEstudo, executar_estudo
    from .relatorio import escrever_relatorio

    if args.cargas:
        comodos = carregar_cenario_excel(args.cargas)
        instancias = {c.nome: 1 for c in comodos}
    else:
        comodos, instancias = cenario_exemplo(args.demo)

    duracoes = tuple(float(d) for d in args.duracoes.split(",")) if args.duracoes else None
    malha = MalhaApagao(
        amostras=args.amostras,
        passo_min=args.passo,
        soc_inicial_frac=args.soc_inicial,
        **({"duracoes_h": duracoes} if duracoes else {}),
    )
    premissas = PremissasBateria(
        tarifa_fora_ponta_brl_kwh=args.tarifa,
        tarifa_ponta_brl_kwh=args.tarifa_ponta,
        custo_interrupcao_brl_kwh=args.custo_interrupcao,
        tarifa_demanda_brl_kw_mes=args.tarifa_demanda,
        reserva_backup_frac=args.reserva_backup,
        anos_analise=args.anos,
    )
    cfg = ConfiguracaoEstudo(
        latitude=args.lat,
        longitude=args.lon,
        nome=args.nome,
        comodos=comodos,
        instancias_por_comodo=instancias,
        comodos_essenciais=(
            [n.strip() for n in args.essenciais.split(",")] if args.essenciais else None
        ),
        simulacoes=args.simulacoes,
        potencia_fv_kwp=args.kwp,
        inclinacao_deg=args.inclinacao,
        azimute_deg=args.azimute,
        anos_serie=(args.ano_inicial, args.ano_final),
        permitir_serie_sintetica=not args.exigir_pvgis,
        catalogo=args.catalogo,
        max_candidatos=args.candidatos,
        autonomia_alvo_h=args.autonomia,
        confiabilidade_alvo=args.confiabilidade,
        exigir_pior_caso=not args.pela_media,
        malha=malha,
        premissas=premissas,
    )

    estudo = executar_estudo(cfg, progresso=_progresso)

    print("\n=== RANKING ===")
    _mostrar(
        estudo.ranking[
            [
                "conjunto", "energia_util_kwh", "potencia_kw", "autonomia_garantida_h",
                "autonomia_ano10_h", "capex_brl", "vpl_brl", "payback_anos", "atende_meta",
            ]
        ],
        1,
    )
    if estudo.recomendado is not None:
        print(f"\nRecomendado: {estudo.recomendado.conjunto.descricao()}")
    for aviso in estudo.avisos:
        print(f"\nAVISO: {aviso}", file=sys.stderr)

    escritos = escrever_relatorio(estudo, args.saida, com_graficos=not args.sem_graficos)
    print(f"\nPacote gravado em {Path(args.saida).resolve()}")
    print(f"  Relatório : {escritos['markdown'].name}")
    if "latex" in escritos:
        print(f"  LaTeX     : {escritos['latex'].name}")
    print(f"  Figuras   : {sum(1 for k in escritos if k.startswith('figura_'))}")
    print(f"  Planilhas : {sum(1 for k in escritos if k.startswith('csv_'))}")
    return 0


# ----------------------------------------------------------------------------
def registrar(sub: argparse._SubParsersAction) -> None:
    """Pendura o subcomando ``bateria`` no parser principal do Aurum."""
    parser = sub.add_parser(
        "bateria",
        help="Estudo de armazenamento: resiliência, potência e economia",
        description=(
            "Cruza a demanda probabilística do D² com a geração solar horária do local "
            "para dimensionar banco e inversor híbrido por critério de resiliência."
        ),
    )
    interno = parser.add_subparsers(dest="subcomando_bateria", required=True)

    # -- catálogo ---------------------------------------------------------
    p_cat = interno.add_parser("catalogo", help="Mostra ou cria o BDBaterias.xlsx")
    p_cat.add_argument("--planilha", help="Caminho da planilha (padrão: BDBaterias.xlsx)")
    p_cat.add_argument("--criar", action="store_true", help="Grava a planilha semente e sai")
    p_cat.add_argument("-v", "--verboso", action="store_true")
    p_cat.set_defaults(funcao=comando_catalogo)

    # -- excedência -------------------------------------------------------
    p_exc = interno.add_parser(
        "excedencia",
        help="Curva de excedência de pico contra os limites dos inversores",
        description=(
            "Barato e rápido: não simula apagão nenhum, só cruza a distribuição de picos "
            "da carga com o nominal e a sobrecarga de cada inversor do catálogo."
        ),
    )
    _argumentos_de_carga(p_exc)
    p_exc.add_argument("--catalogo", help="Caminho do BDBaterias.xlsx")
    p_exc.add_argument("--passo", type=int, default=1, help="Passo de análise em minutos")
    p_exc.add_argument("--horaria", action="store_true", help="Inclui a tabela hora a hora")
    p_exc.add_argument("--saida", help="Diretório para gravar os CSV")
    p_exc.add_argument("-v", "--verboso", action="store_true")
    p_exc.set_defaults(funcao=comando_excedencia)

    # -- estudo -----------------------------------------------------------
    p_est = interno.add_parser("estudo", help="Estudo completo, com relatório")
    _argumentos_de_carga(p_est)
    p_est.add_argument("--lat", type=float, required=True, help="Latitude do local")
    p_est.add_argument("--lon", type=float, required=True, help="Longitude do local")
    p_est.add_argument("--nome", default="instalação", help="Nome do cliente ou do local")
    p_est.add_argument("--kwp", type=float, help="Potência FV (padrão: dimensiona pelo consumo)")
    p_est.add_argument("--inclinacao", type=float, help="Inclinação dos módulos em graus")
    p_est.add_argument("--azimute", type=float, help="Azimute (0 = Norte)")
    p_est.add_argument("--ano-inicial", dest="ano_inicial", type=int, default=2016)
    p_est.add_argument("--ano-final", dest="ano_final", type=int, default=2020)
    p_est.add_argument("--exigir-pvgis", action="store_true",
                       help="Falha em vez de cair para a série sintética")
    p_est.add_argument("--catalogo", help="Caminho do BDBaterias.xlsx")
    p_est.add_argument("--candidatos", type=int, default=8,
                       help="Quantos conjuntos avaliar (padrão: 8)")
    p_est.add_argument("--duracoes", help="Durações em horas, ex.: 1,2,3,6,12,24,36")
    p_est.add_argument("--amostras", type=int, default=200,
                       help="Sorteios por combinação de estação, duração e hora")
    p_est.add_argument("--passo", type=int, default=5, help="Passo do despacho em minutos")
    p_est.add_argument("--soc-inicial", dest="soc_inicial", type=float, default=1.0,
                       help="Estado de carga no instante do apagão (0 a 1)")
    p_est.add_argument("--autonomia", type=float, default=12.0,
                       help="Autonomia alvo em horas (padrão: 12)")
    p_est.add_argument("--confiabilidade", type=float, default=0.95,
                       help="Probabilidade mínima de atravessar (padrão: 0,95)")
    p_est.add_argument("--pela-media", action="store_true",
                       help="Exige a meta na média das horas, não no pior par estação/hora")
    p_est.add_argument("--tarifa", type=float, default=0.78, help="Tarifa fora de ponta, R$/kWh")
    p_est.add_argument("--tarifa-ponta", dest="tarifa_ponta", type=float, default=1.35)
    p_est.add_argument("--tarifa-demanda", dest="tarifa_demanda", type=float, default=0.0,
                       help="R$/kW/mês de demanda contratada (Grupo A)")
    p_est.add_argument("--custo-interrupcao", dest="custo_interrupcao", type=float, default=0.0,
                       help="Quanto vale 1 kWh que faltou, em R$ — pergunte ao cliente")
    p_est.add_argument("--reserva-backup", dest="reserva_backup", type=float, default=0.30,
                       help="Fração do banco reservada para apagão (não é ciclada)")
    p_est.add_argument("--anos", type=int, default=15, help="Horizonte da análise econômica")
    p_est.add_argument("--saida", required=True, help="Diretório do pacote de relatório")
    p_est.add_argument("--sem-graficos", action="store_true", help="Pula as figuras")
    p_est.add_argument("-v", "--verboso", action="store_true")
    p_est.set_defaults(funcao=comando_estudo)


def _argumentos_de_carga(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cargas", help="Planilha de cargas do D² (uma aba por cômodo)")
    parser.add_argument("--demo", default="hotel", choices=["hotel", "escritorio"],
                        help="Cenário de demonstração, quando não há --cargas")
    parser.add_argument("--essenciais", metavar="A,B",
                        help="Cômodos ligados ao quadro de backup (padrão: todos)")
    parser.add_argument("--simulacoes", type=int, default=300,
                        help="Simulações de Monte Carlo por estação")
