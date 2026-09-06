"""
Interface de linha de comando do Aurum.

O fluxo é de duas etapas, e é assim de propósito: **nenhum estudo é calculado
antes de você escolher os telhados**. A prospecção é barata e olha a região
inteira; o dimensionamento é caro e só roda no que você marcou.

    1) aurum prospectar --regiao "..." --saida pasta
         varre a região e grava pasta/leads.csv com a coluna "selecionar"

    2) abra pasta/leads.csv, escreva x nas linhas que interessam, salve

    3) aurum propor --da-lista pasta --saida pasta/propostas
         calcula e gera a proposta só dos marcados

Subcomandos:

    aurum prospectar   varre uma região e lista os telhados aproveitáveis
    aurum propor       gera as propostas dos telhados que você escolheu
    aurum equipamentos mostra o catálogo de módulos e inversores carregado
    aurum bateria      estudo de armazenamento (resiliência, potência, economia)

Exemplos::

    python -m aurum.cli prospectar --regiao "Cidade Industrial de Curitiba" --saida leads/cic
    python -m aurum.cli propor --da-lista leads/cic --saida propostas/cic
    python -m aurum.cli propor --da-lista leads/cic --ids way/447998570,way/357014707
    python -m aurum.cli bateria estudo --lat -25.43 --lon -49.27 --kwp 40 --saida estudos/hotel

Para pular a escolha manual e aceitar os N melhores automaticamente, é preciso
dizer isso explicitamente:

    python -m aurum.cli propor --regiao "..." --auto-top 10
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import get_settings
from .geo import segments
from .pipeline import (
    AreaGrandeDemais,
    OpcoesDimensionamento,
    carregar_leads,
    dimensionar_lote,
    escrever_geojson,
    escrever_indice,
    escrever_leads_csv,
    filtrar_selecionados,
    ler_selecao_csv,
    prospectar,
)
from .proposal.context import DadosEmissor
from .proposal.render import encontrar_compilador, escrever_proposta, montar_zip


def _configurar_log(verboso: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verboso else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def _plural(quantidade: int, singular: str, plural: str | None = None) -> str:
    """Concordância simples, para as mensagens não saírem com '1 telhados'."""
    return singular if quantidade == 1 else (plural or singular + "s")


def _emissor_de_args(args: argparse.Namespace) -> DadosEmissor:
    return DadosEmissor(
        empresa=args.empresa,
        responsavel_tecnico=args.responsavel,
        crea=args.crea,
        telefone=args.telefone,
        email=args.email,
        site=args.site,
    )


def _argumentos_comuns(parser: argparse.ArgumentParser, area_obrigatoria: bool = True) -> None:
    area = parser.add_mutually_exclusive_group(required=area_obrigatoria)
    area.add_argument("--regiao", help="Nome da região, bairro ou cidade")
    area.add_argument("--bbox", help="min_lon,min_lat,max_lon,max_lat")

    parser.add_argument("--min-area", type=float, default=500.0,
                        help="Área mínima de telhado em m² (padrão: 500)")
    parser.add_argument(
        "--alvo", default="todos", metavar="SEGMENTO",
        help=(
            "Segmentos alvo, separados por vírgula. Use 'todos' para não filtrar "
            "ou 'aurum segmentos' para ver a lista. Ex.: --alvo agro,educacao"
        ),
    )
    parser.add_argument("--top", type=int, default=20,
                        help="Quantos telhados considerar (padrão: 20)")
    parser.add_argument("--ordenar", default="score", choices=["score", "area"],
                        help="Ordenar por pontuação de lead ou por área (padrão: score)")
    parser.add_argument("--rigoroso", action="store_true",
                        help="Filtro geométrico conservador: menos leads, mais confiáveis")
    parser.add_argument("--sem-enriquecimento", action="store_true",
                        help="Pula a busca de nome, telefone e site das empresas")
    parser.add_argument("--area-grande", action="store_true",
                        help="Autoriza varrer região acima de 400 km² (município inteiro)")
    parser.add_argument("-v", "--verboso", action="store_true")


def _validar_alvo(alvo: str) -> None:
    """
    Recusa segmento inexistente antes de varrer a região.

    Um erro de digitação em --alvo devolveria zero telhados depois de minutos
    de consulta, e parecia região vazia em vez de nome errado.
    """
    entradas = [p.strip().lower() for p in str(alvo).split(",") if p.strip()]
    validas = set(segments.chaves_validas())
    desconhecidas = [e for e in entradas if e not in validas]
    if desconhecidas:
        raise SystemExit(
            f"Segmento desconhecido: {', '.join(desconhecidas)}.\n"
            f"Disponíveis: {', '.join(sorted(segments.POR_CHAVE))}.\n"
            "Rode 'python -m aurum.cli segmentos' para ver a descrição de cada um."
        )


def comando_segmentos(args: argparse.Namespace) -> int:
    """Lista os segmentos disponíveis para o filtro."""
    print("Segmentos disponíveis para --alvo (separe vários por vírgula):\n")
    print(f"  {'chave':<16}{'peso':>5}  descrição")
    print("  " + "-" * 94)
    for segmento in sorted(segments.SEGMENTOS, key=lambda s: -s.peso_comercial):
        print(f"  {segmento.chave:<16}{segmento.peso_comercial:>5.0f}  "
              f"{segmento.rotulo} — {segmento.descricao}")
    print(f"\n  {'todos':<16}{'':>5}  sem filtro")
    print(f"  {'desconhecido':<16}{segments.SEGMENTO_DESCONHECIDO.peso_comercial:>5.0f}  "
          "Não identificado — edificações sem tipo declarado no OpenStreetMap")
    print(
        "\nO peso é a contribuição do segmento na pontuação do lead (0 a 20). "
        "Reflete\nconsumo diurno, porte típico e facilidade de decisão comercial."
    )
    print("\nExemplos:")
    print("  --alvo agro")
    print("  --alvo educacao,saude")
    print("  --alvo industrial,logistica,frigorifico")
    return 0


def comando_prospectar(args: argparse.Namespace) -> int:
    """Lista os telhados de uma região, sem dimensionar nada."""
    _validar_alvo(args.alvo)
    resultado = prospectar(
        regiao=args.regiao,
        bbox=args.bbox,
        min_area_m2=args.min_area,
        alvo=args.alvo,
        rigoroso=args.rigoroso,
        limite=args.top,
        enriquecer=not args.sem_enriquecimento,
        confirmar_area_grande=args.area_grande,
        progresso=lambda m, f: print(f"[{f * 100:3.0f}%] {m}", file=sys.stderr),
    )

    leads = resultado.por_area() if args.ordenar == "area" else resultado.leads
    estatisticas = resultado.estatisticas
    print(f"\n{len(leads)} telhados em {estatisticas.get('regiao') or 'bbox informado'}\n")
    print(f"{'#':>3}  {'ID OSM':<16}{'ÁREA m²':>10}{'SCORE':>7}  {'SEGMENTO':<26}{'NOME / CONTATO'}")
    print("-" * 118)
    for i, lead in enumerate(leads, 1):
        empresa = lead.company
        print(f"{i:>3}  {lead.lead_id:<16}{lead.metrics.area_m2:>10,.0f}{lead.score:>7.1f}  "
              f"{lead.segmento_rotulo[:25]:<26}{lead.name[:38]}")
        detalhes = [d for d in (empresa.get("telefone"), empresa.get("site"), empresa.get("endereco")) if d]
        if detalhes:
            print(f"{'':>5}{' | '.join(str(d)[:105] for d in detalhes)}")

    distribuicao = estatisticas.get("segmentos") or {}
    if distribuicao and len(distribuicao) > 1:
        print("\nDistribuição por segmento:")
        for chave, quantidade in sorted(distribuicao.items(), key=lambda kv: -kv[1]):
            segmento = segments.POR_CHAVE.get(chave)
            rotulo = segmento.rotulo if segmento else segments.SEGMENTO_DESCONHECIDO.rotulo
            print(f"  {quantidade:>4}  {rotulo}")
        if distribuicao.get("desconhecido"):
            print("\n  'Não identificado' são edificações sem tipo declarado no OpenStreetMap.")
            print("  Muitas valem a pena: confira no satélite antes de descartar.")

    if args.saida:
        destino = Path(args.saida)
        destino.mkdir(parents=True, exist_ok=True)
        escrever_geojson(leads, destino / "leads.geojson")
        caminho_csv = escrever_leads_csv(leads, destino / "leads.csv")
        (destino / "prospeccao.json").write_text(
            json.dumps(resultado.estatisticas, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nArquivos gravados em {destino.resolve()}")
        print("\nNenhum estudo foi calculado ainda. Para escolher os telhados:")
        print(f"  1. abra {caminho_csv.name} e escreva  x  na coluna 'selecionar'")
        print("  2. salve o arquivo")
        print(f"  3. rode:  python -m aurum.cli propor --da-lista {args.saida}")
    else:
        print("\nNenhum estudo foi calculado. Use --saida <pasta> para gravar a lista "
              "e depois escolher os telhados.")

    return 0


def _selecionar_leads(args: argparse.Namespace) -> tuple[list, str]:
    """
    Resolve quais telhados vão virar estudo. Devolve (leads, origem).

    Três caminhos, todos explícitos: uma lista já prospectada com marcação na
    planilha, identificadores passados na linha de comando, ou a aceitação
    declarada dos N melhores. Não existe caminho implícito -- o custo de
    calcular um estudo que ninguém pediu é do usuário, não do programa.
    """
    if args.da_lista:
        leads = carregar_leads(args.da_lista)
        if not leads:
            raise SystemExit(f"Nenhum telhado em {args.da_lista}.")

        if args.ids:
            escolhidos = filtrar_selecionados(leads, args.ids.split(","))
            origem = (f"{len(escolhidos)} {_plural(len(escolhidos), 'telhado')} "
                      f"{_plural(len(escolhidos), 'indicado')} por --ids")
        else:
            marcados = ler_selecao_csv(args.da_lista)
            if not marcados:
                raise SystemExit(
                    f"Nenhuma linha marcada em {Path(args.da_lista).name}/leads.csv.\n"
                    "Abra a planilha, escreva  x  na coluna 'selecionar' das linhas que "
                    "interessam, salve e rode de novo.\n"
                    "Para indicar direto pela linha de comando, use --ids way/123,way/456."
                )
            escolhidos = filtrar_selecionados(leads, marcados)
            origem = (f"{len(escolhidos)} de {len(leads)} {_plural(len(leads), 'telhado')} "
                      f"{_plural(len(escolhidos), 'marcado')} na planilha")

        if not escolhidos:
            raise SystemExit(
                "Os identificadores marcados não batem com nenhum telhado da lista. "
                "Confira a coluna lead_id."
            )
        return escolhidos, origem

    # Caminho automático: só roda porque foi pedido com todas as letras.
    _validar_alvo(args.alvo)
    resultado = prospectar(
        regiao=args.regiao,
        bbox=args.bbox,
        min_area_m2=args.min_area,
        alvo=args.alvo,
        rigoroso=args.rigoroso,
        enriquecer=not args.sem_enriquecimento,
        confirmar_area_grande=args.area_grande,
        progresso=lambda m, f: print(f"[{f * 100:3.0f}%] {m}", file=sys.stderr),
    )
    escolhidos = (
        resultado.por_area(args.auto_top) if args.ordenar == "area" else resultado.top(args.auto_top)
    )
    origem = (
        f"{len(escolhidos)} {_plural(len(escolhidos), 'melhor', 'melhores')} de "
        f"{resultado.total} {_plural(resultado.total, 'encontrado')} "
        f"(seleção automática por {args.ordenar})"
    )
    return escolhidos, origem


def comando_propor(args: argparse.Namespace) -> int:
    """Gera as propostas dos telhados escolhidos."""
    if not args.da_lista and not args.auto_top:
        raise SystemExit(
            "Escolha os telhados antes de gerar o estudo. Duas formas:\n\n"
            "  1) Fluxo com seleção (recomendado)\n"
            "       aurum prospectar --regiao \"...\" --saida pasta\n"
            "       (marque x na coluna 'selecionar' de pasta/leads.csv)\n"
            "       aurum propor --da-lista pasta\n\n"
            "  2) Aceitar os N melhores automaticamente\n"
            "       aurum propor --regiao \"...\" --auto-top 10"
        )
    if args.da_lista and (args.regiao or args.bbox):
        raise SystemExit("--da-lista já traz os telhados; não combine com --regiao ou --bbox.")

    if encontrar_compilador() is None and not args.sem_pdf:
        print(
            "Nenhuma distribuição LaTeX encontrada. Os arquivos .tex serão gerados "
            "e podem ser compilados no Overleaf.",
            file=sys.stderr,
        )

    leads, origem = _selecionar_leads(args)
    verbo = _plural(len(leads), "Será calculado", "Serão calculados")
    print(f"\n{verbo}: {origem}", file=sys.stderr)
    for lead in leads[:12]:
        print(f"  · {lead.metrics.area_m2:>9,.0f} m²  {lead.name[:52]}", file=sys.stderr)
    if len(leads) > 12:
        print(f"  · … e mais {len(leads) - 12}", file=sys.stderr)

    opcoes = OpcoesDimensionamento(
        montagem=args.montagem,
        tilt_deg=args.inclinacao,
        limitar_ao_consumo=args.limitar_ao_consumo,
        tarifa_brl_kwh=args.tarifa,
        capex_brl_por_kwp=args.capex_kwp,
        taxa_desconto=args.taxa_desconto,
        anos=args.anos,
        ano_conexao=args.ano_conexao,
        aplicar_lei_14300=not args.sem_lei_14300,
    )

    contextos, falhas = dimensionar_lote(
        leads,
        opcoes=opcoes,
        emissor=_emissor_de_args(args),
        progresso=lambda m, f: print(f"[{f * 100:3.0f}%] {m}", file=sys.stderr),
    )

    destino = Path(args.saida)
    destino.mkdir(parents=True, exist_ok=True)
    for ctx in contextos:
        escrever_proposta(ctx, destino / ctx.referencia, compilar=not args.sem_pdf)
    indice = escrever_indice(contextos, destino, falhas)
    escrever_geojson(leads, destino / "leads.geojson")
    caminho_zip = montar_zip(destino)

    print(f"\nTelhados escolhidos : {len(leads)}")
    print(f"Propostas geradas   : {len(contextos)}")
    if falhas:
        print(f"Sem proposta        : {len(falhas)}")
        for falha in falhas:
            print(f"  - {falha['nome'][:50]}: {falha['erro'][:70]}")
    print(f"\nDiretório : {destino.resolve()}")
    print(f"Índice    : {indice['md']}")
    print(f"Pacote    : {caminho_zip}")
    return 0


def comando_equipamentos(args: argparse.Namespace) -> int:
    """Mostra o catálogo carregado, para conferir a planilha."""
    from .pv.equipment import EquipmentError, carregar_base

    try:
        base = carregar_base(args.planilha)
    except EquipmentError as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 1

    print(f"Base: {base.caminho}\n")
    print(f"{'MÓDULO':<34}{'Wp':>7}{'Voc':>7}{'Isc':>7}{'Imp':>7}{'DIMENSÕES':>18}{'ÁREA':>8}")
    print("-" * 90)
    for modulo in sorted(base.modulos, key=lambda m: -m.potencia_wp):
        marca = " *" if modulo.dimensoes_derivadas else "  "
        print(f"{str(modulo)[:33]:<34}{modulo.potencia_wp:>7.0f}{modulo.voc:>7.1f}"
              f"{modulo.isc:>7.2f}{modulo.imp:>7.2f}"
              f"{modulo.comprimento_m:>9.3f} x{modulo.largura_m:>6.3f}{modulo.area_m2:>8.2f}{marca}")
    print("\n* dimensões derivadas da eficiência declarada, não do datasheet\n")

    print(f"{'INVERSOR':<34}{'kW CA':>8}{'kW FV':>8}{'Vmax':>7}{'Vstart':>8}{'MPPT':>6}{'A/MPPT':>8}")
    print("-" * 90)
    for inversor in base.inversores_ordenados():
        print(f"{str(inversor)[:33]:<34}{inversor.potencia_ca_kw:>8.1f}"
              f"{inversor.potencia_fv_max_w / 1000:>8.1f}{inversor.tensao_max_cc:>7.0f}"
              f"{inversor.tensao_start:>8.0f}{inversor.num_mppt:>6}{inversor.corrente_max_mppt:>8.1f}")
    return 0


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aurum",
        description="Prospecção de telhados e geração automática de propostas de energia solar.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="comando", required=True)

    p_prosp = sub.add_parser("prospectar", help="Lista telhados de uma região")
    _argumentos_comuns(p_prosp)
    p_prosp.add_argument("--saida", help="Diretório para gravar CSV e GeoJSON")
    p_prosp.set_defaults(funcao=comando_prospectar)

    p_prop = sub.add_parser(
        "propor",
        help="Gera as propostas dos telhados que você escolheu",
        description=(
            "Calcula o estudo apenas dos telhados escolhidos. Use --da-lista com uma "
            "prospecção já gravada, ou --auto-top N para aceitar os N melhores."
        ),
    )
    _argumentos_comuns(p_prop, area_obrigatoria=False)
    p_prop.add_argument("--da-lista", dest="da_lista", metavar="PASTA",
                        help="Pasta de uma prospecção anterior (com leads.geojson e leads.csv)")
    p_prop.add_argument("--ids", metavar="ID,ID",
                        help="Telhados a calcular, ex.: way/447998570,way/357014707")
    p_prop.add_argument("--auto-top", dest="auto_top", type=int, metavar="N",
                        help="Aceita automaticamente os N melhores, sem seleção manual")
    p_prop.add_argument("--saida", default=str(get_settings().output_dir),
                        help="Diretório de saída das propostas")
    p_prop.add_argument("--montagem", default="coplanar", choices=["coplanar", "inclinado"],
                        help="coplanar: telhado inclinado; inclinado: laje plana com estrutura")
    p_prop.add_argument("--inclinacao", type=float,
                        help="Inclinação dos módulos em graus (padrão: automático)")
    p_prop.add_argument("--tarifa", type=float,
                        help="Tarifa em R$/kWh (padrão: referência do segmento)")
    p_prop.add_argument("--capex-kwp", type=float,
                        help="Custo em R$/kWp (padrão: curva de escala)")
    p_prop.add_argument("--taxa-desconto", type=float, default=0.10)
    p_prop.add_argument("--anos", type=int, default=25)
    p_prop.add_argument("--ano-conexao", type=int, default=2026)
    p_prop.add_argument("--limitar-ao-consumo", action="store_true",
                        help="Dimensiona para o consumo estimado em vez de encher o telhado")
    p_prop.add_argument("--sem-lei-14300", action="store_true",
                        help="Desliga a cobrança do Fio B no fluxo de caixa")
    p_prop.add_argument("--sem-pdf", action="store_true", help="Gera apenas os .tex")
    p_prop.add_argument("--empresa", default="PACE Inteligência Energética")
    p_prop.add_argument("--responsavel", help="Responsável técnico")
    p_prop.add_argument("--crea", help="Registro CREA do responsável")
    p_prop.add_argument("--telefone")
    p_prop.add_argument("--email")
    p_prop.add_argument("--site")
    p_prop.set_defaults(funcao=comando_propor)

    p_seg = sub.add_parser("segmentos", help="Lista os segmentos disponíveis para --alvo")
    p_seg.add_argument("-v", "--verboso", action="store_true")
    p_seg.set_defaults(funcao=comando_segmentos)

    p_equip = sub.add_parser("equipamentos", help="Mostra o catálogo de equipamentos")
    p_equip.add_argument("--planilha", help="Caminho da planilha (padrão: BDFotovoltaica.xlsx)")
    p_equip.add_argument("-v", "--verboso", action="store_true")
    p_equip.set_defaults(funcao=comando_equipamentos)

    # O estudo de armazenamento traz os próprios subcomandos. O import fica
    # aqui dentro porque ele carrega matplotlib, e `aurum prospectar` não tem
    # por que pagar esse custo.
    from .bateria.cli import registrar as registrar_bateria

    registrar_bateria(sub)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = construir_parser()
    args = parser.parse_args(argv)
    _configurar_log(getattr(args, "verboso", False))
    try:
        return int(args.funcao(args))
    except KeyboardInterrupt:
        print("\nInterrompido.", file=sys.stderr)
        return 130
    except AreaGrandeDemais as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - a CLI deve falhar com mensagem, não com traceback
        logging.getLogger(__name__).debug("Falha na execução", exc_info=True)
        print(f"Erro: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
