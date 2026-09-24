"""Executa o PACE Calculator diretamente a partir de um pacote pace-coleta/1."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from .coleta import importar, ler_pacote


@dataclass
class Opcoes:
    tarifa: float
    tensao: float
    autonomia: float = 6.0
    confiabilidade: float = 0.95
    disponibilidade: float = 50.0
    simulacoes: int = 300
    amostras: int = 200
    candidatos: int = 12
    permitir_sintetica: bool = False
    compilar_pdf: bool = True

    def validar(self):
        for nome in ('tarifa', 'tensao', 'autonomia', 'confiabilidade', 'disponibilidade'):
            valor = getattr(self, nome)
            if isinstance(valor, bool) or not math.isfinite(valor) or valor <= 0:
                raise ValueError(f'{nome} deve ser um número positivo.')
        if self.confiabilidade > 1 or self.tensao not in (220, 380):
            raise ValueError('Confiabilidade até 1; tensão de linha 220 ou 380 V.')
        for nome in ('simulacoes', 'amostras', 'candidatos'):
            if type(getattr(self, nome)) is not int or getattr(self, nome) < 1:
                raise ValueError(f'{nome} deve ser inteiro positivo.')


def configurar(pacote: dict, opcoes: Opcoes):
    from .bateria.apagao import MalhaApagao
    from .bateria.economia import PremissasBateria
    from .bateria.estudo import ConfiguracaoEstudo
    from .bateria.fontes import Fatura
    from .demanda.vistoria import DESCRICAO_CRITICIDADE
    from .pv.telhado import telhado_de_geojson, dimensionar_no_telhado
    from .pv.equipment import carregar_base

    opcoes.validar()
    dados = ler_pacote(pacote)
    c = dados.cenario
    consumo = pacote['vistoria'].get('consumoKwh')
    if consumo not in (None, ''):
        consumo = float(str(consumo).replace(',', '.'))
        if not math.isfinite(consumo) or consumo <= 0:
            raise ValueError('Consumo mensal inválido.')
    else:
        consumo = None
    backup = c.para_comodos(True) if c.tem_criticidade else c.para_comodos()
    if not backup:
        raise ValueError('Selecione ao menos um equipamento C ou MC para o estudo de backup.')
    telhado = layout = None
    if dados.telhado_geojson:
        telhado = telhado_de_geojson(dados.telhado_geojson, nome=dados.local)
        layout = dimensionar_no_telhado(telhado, carregar_base())
        if layout is None or layout.quantidade == 0:
            raise ValueError('O contorno não comporta módulos do catálogo; revise o telhado.')
    cfg = ConfiguracaoEstudo(
        latitude=dados.latitude, longitude=dados.longitude,
        nome=f'{dados.cliente} — {dados.local}', tensao_rede_v=opcoes.tensao,
        comodos=c.para_comodos(), instancias_por_comodo=c.instancias_de(),
        comodos_backup=backup, instancias_backup=c.instancias_de(c.tem_criticidade),
        criticidade={'tabela': c.por_criticidade(), 'corte': c.criticidades_essenciais,
                     'descricoes': DESCRICAO_CRITICIDADE} if c.tem_criticidade else None,
        tabelas_cenario=c.comodos, telhado=telhado, layout=layout,
        inclinacao_deg=telhado.inclinacao_deg if telhado else None,
        azimute_deg=telhado.azimute_deg if telhado else None,
        consumo_anual_kwh=consumo * 12 if consumo else None,
        fatura=Fatura(tarifa_brl_kwh=opcoes.tarifa, consumo_mensal_kwh=consumo,
                      custo_disponibilidade_kwh=opcoes.disponibilidade),
        premissas=PremissasBateria(tarifa_fora_ponta_brl_kwh=opcoes.tarifa),
        simulacoes=opcoes.simulacoes, max_candidatos=opcoes.candidatos,
        autonomia_alvo_h=opcoes.autonomia, confiabilidade_alvo=opcoes.confiabilidade,
        permitir_serie_sintetica=opcoes.permitir_sintetica,
        malha=MalhaApagao(amostras=opcoes.amostras,
                         duracoes_h=tuple(sorted({1., 3., 6., 12., 24., opcoes.autonomia}))),
    )
    return dados, cfg


def executar(pacote: dict, destino: str | Path, opcoes: Opcoes, progresso=None) -> Path:
    from .bateria.estudo import executar_estudo
    from .bateria.apresentacao import OpcoesComerciais
    from .bateria.documento import DadosCapa, escrever_dossie

    dados, cfg = configurar(pacote, opcoes)
    destino = Path(destino).resolve()
    # A importação valida os anexos e recusa sobrescrever um estudo anterior.
    importar(pacote, destino)
    manifesto = {'situacao': 'executando', 'entrada_sha256': hashlib.sha256(
        json.dumps(pacote, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
        'opcoes': asdict(opcoes), 'avisos': dados.avisos + [
            'Conta anexada preservada, sem OCR. Tarifa e tensão informadas pelo operador.',
            'Um consumo mensal informado é extrapolado para 12 meses.',
            'Layout usa capacidade da cobertura e premissas geométricas do Calculator.']}
    arquivo = destino / 'execucao.json'
    def registrar():
        arquivo.write_text(json.dumps(manifesto, ensure_ascii=False, indent=2), encoding='utf-8')
    registrar()
    try:
        estudo = executar_estudo(cfg, progresso=progresso)
        estudo.avisos.extend(manifesto['avisos'])
        # A apresentação sai com as condições comerciais padrão: o pacote de
        # coleta não as traz, e elas são premissa de venda, não de cálculo.
        escritos = escrever_dossie(estudo, destino / 'estudo',
                                  capa=DadosCapa(referencia='Estudo residencial'),
                                  compilar=opcoes.compilar_pdf,
                                  comercial=OpcoesComerciais(cliente=dados.cliente))
        shutil.make_archive(str(destino / 'estudo-pace'), 'zip', destino / 'estudo')
        manifesto.update(situacao='concluido', atende_meta=bool(estudo.atende_meta),
                         pdf_gerado='pdf' in escritos,
                         arquivos={k: str(v.relative_to(destino)) for k, v in escritos.items()},
                         avisos=estudo.avisos)
        registrar()
    except Exception as exc:
        manifesto.update(situacao='falhou', erro=str(exc))
        registrar()
        raise
    return destino


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pacote', type=Path)
    parser.add_argument('--saida', type=Path, required=True)
    parser.add_argument('--tarifa', type=float, required=True, help='R$/kWh conferidos')
    parser.add_argument('--tensao', type=float, required=True, choices=[220, 380])
    parser.add_argument('--autonomia', type=float, default=6)
    parser.add_argument('--disponibilidade', type=float, choices=[30, 50, 100], default=50)
    parser.add_argument('--permitir-sintetica', action='store_true')
    parser.add_argument('--simulacoes', type=int, default=300)
    parser.add_argument('--amostras', type=int, default=200)
    parser.add_argument('--candidatos', type=int, default=12)
    args = vars(parser.parse_args())
    pacote = json.loads(args.pop('pacote').read_text(encoding='utf-8-sig'))
    saida = args.pop('saida')
    executar(pacote, saida, Opcoes(**args), lambda msg, frac: print(f'[{frac:.0%}] {msg}', flush=True))
    print(f'Estudo gerado em {saida}')


if __name__ == '__main__':
    main()
