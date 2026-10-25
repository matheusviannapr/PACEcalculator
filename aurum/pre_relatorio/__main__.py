"""
Pré-relatório pela linha de comando.

    python -m aurum.pre_relatorio estudo-cliente-PACE-X9KYZI.json --kwh 450 --tarifa 1.05

Lê o JSON que chegou pelo CRM ou pelo e-mail, roda a curva híbrida e o estudo, e
grava ao lado do JSON:

* ``<nome>-pre.png`` — a imagem para o WhatsApp;
* ``<nome>-pre.json`` — os números da imagem e os avisos, para conferir.

``--kwh`` e ``--tarifa`` são lidos na conta anexada: kWh faturados no mês e a
tarifa efetiva (valor total ÷ kWh). Sem eles o pré-relatório sai, mas avisa que
estimou — e a imagem diz de onde veio o consumo.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .entrada import montar_entrada
from .estudo import OpcoesPreRelatorio, gerar


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m aurum.pre_relatorio", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pacote", type=Path, help="JSON do questionário (pace-coleta/1)")
    p.add_argument("--kwh", type=float, help="consumo do mês na conta, em kWh")
    p.add_argument("--tarifa", type=float, help="tarifa efetiva, R$/kWh (valor total ÷ kWh)")
    p.add_argument("--tensao", type=float, default=220, choices=[220, 380], help="tensão de linha da rede")
    p.add_argument("--disponibilidade", type=float, default=50, choices=[30, 50, 100],
                   help="custo de disponibilidade em kWh: 30 mono, 50 bi, 100 trifásico")
    p.add_argument("--autonomia", type=float, help="horas de autonomia alvo (padrão: a do questionário)")
    p.add_argument("--mostrar-investimento", action="store_true", help="põe investimento e retorno na imagem")
    p.add_argument("--saida", type=Path, help="pasta de saída (padrão: a do JSON)")
    a = p.parse_args(argv)

    entrada = montar_entrada(a.pacote, consumo_mensal_kwh=a.kwh, tarifa_brl_kwh=a.tarifa)
    opcoes = OpcoesPreRelatorio(tensao_v=a.tensao, disponibilidade_kwh=a.disponibilidade,
                                autonomia_alvo_h=a.autonomia, mostrar_investimento=a.mostrar_investimento)
    pr = gerar(entrada, opcoes, progresso=lambda msg, frac: print(f"[{frac:4.0%}] {msg}", flush=True))

    pasta = a.saida or a.pacote.parent
    pasta.mkdir(parents=True, exist_ok=True)
    base = pasta / (a.pacote.stem + "-pre")
    dados = pr.resumo.as_dict()
    dados["avisos"] = pr.avisos
    dados["conferencia_conta"] = pr.curva.ajuste.erros_pct
    dados["peso_respostas"] = pr.curva.peso_respostas
    base.with_suffix(".json").write_text(json.dumps(dados, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"Números: {base.with_suffix('.json')}")

    try:
        from .imagem import desenhar
    except ImportError:
        print("Imagem: módulo aurum.pre_relatorio.imagem ainda não existe; só os números foram gravados.")
        return 0
    desenhar(pr.resumo, base.with_suffix(".png"))
    print(f"Imagem:  {base.with_suffix('.png')}")
    for aviso in pr.avisos:
        print("  aviso:", aviso)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
