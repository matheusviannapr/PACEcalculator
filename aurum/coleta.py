"""Ponte da coleta HTML para o cenário PACE, sem executar simulações.

Uso: python -m aurum.coleta pacote.json --destino outputs/coleta-cliente
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import re
from pathlib import Path

import pandas as pd

from .demanda.biblioteca import COLUNAS
from .demanda.cenario import Cenario


def validar_telhado(telhado: dict) -> None:
    """Valida o anel GeoJSON fechado; coleta antiga sem telhado segue legível."""
    geometria = telhado.get('geojson') or {}
    aneis = geometria.get('coordinates')
    if geometria.get('type') != 'Polygon' or not isinstance(aneis, list) or len(aneis) != 1:
        raise ValueError('Telhado deve conter um polígono simples.')
    anel = aneis[0]
    if not isinstance(anel, list) or not 4 <= len(anel) <= 501 or anel[0] != anel[-1]:
        raise ValueError('Contorno do telhado deve estar fechado.')
    for ponto in anel:
        if not isinstance(ponto, list) or len(ponto) != 2:
            raise ValueError('Coordenadas do telhado inválidas.')
        for n, limite in zip(ponto, (180, 90)):
            if isinstance(n, bool) or not isinstance(n, (int, float)) or not math.isfinite(n) or abs(n) > limite:
                raise ValueError('Coordenadas do telhado inválidas.')
    pontos = anel[:-1]
    if len(set(map(tuple, pontos))) != len(pontos):
        raise ValueError('Vértices repetidos no telhado.')
    def cruz(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
    def toca(a, b, p):
        return abs(cruz(a, b, p)) < 1e-16 and all(min(a[k], b[k]) <= p[k] <= max(a[k], b[k]) for k in (0, 1))
    for i in range(len(pontos)):
        a, b = anel[i:i+2]
        for j in range(i+1, len(pontos)):
            if j == i+1 or (i == 0 and j == len(pontos)-1):
                continue
            c, d = anel[j:j+2]
            if (cruz(a,b,c)*cruz(a,b,d) < 0 and cruz(c,d,a)*cruz(c,d,b) < 0) or any((toca(a,b,c), toca(a,b,d), toca(c,d,a), toca(c,d,b))):
                raise ValueError('O contorno do telhado cruza a si próprio.')
    if abs(sum(cruz(pontos[0], anel[i], anel[i+1]) for i in range(1, len(pontos)))) < 1e-14:
        raise ValueError('Contorno do telhado sem área.')
    if not telhado.get('confirmadoEm'):
        raise ValueError('Confirme o contorno do telhado.')


def converter(pacote: dict) -> Cenario:
    if pacote.get('formato') != 'pace-coleta/1':
        raise ValueError('Formato de coleta não reconhecido.')
    v = pacote['vistoria']
    if v.get('telhado') is not None:
        validar_telhado(v['telhado'])
    loc = v.get('localizacao') or {}
    for chave, limite in [('latitude', 90), ('longitude', 180)]:
        n = loc.get(chave)
        if isinstance(n, bool) or not isinstance(n, (float, int)) or not math.isfinite(n) or abs(n) > limite:
            raise ValueError('Coordenadas inválidas.')
    if not loc.get('confirmadoEm') or not v.get('aceiteDados'):
        raise ValueError('Localização ou uso dos dados sem confirmação.')
    if any(not str(v.get(k, '')).strip() for k in ('cliente', 'contato', 'endereco', 'cidade')):
        raise ValueError('Cadastro incompleto.')
    # Nomes únicos e seguros para abas; preserve a correspondência por ID.
    nomes, usados = {}, {'instancias', 'instâncias', 'config', 'leia-me'}
    for a in pacote['ambientes']:
        if a.get('vistoriaId') != v['id'] or a['id'] in nomes:
            raise ValueError('Ambiente duplicado ou de outra vistoria.')
        raiz = re.sub(r"[\\/*?:\[\]]", '-', str(a['nome'])).strip(" '") or 'Ambiente'
        nome, contador = raiz[:31], 1
        while nome.lower() in usados:
            contador += 1
            nome = raiz[:25] + f' ({contador})'
        usados.add(nome.lower())
        nomes[a['id']] = nome
    linhas: dict[str, list] = {}
    for i in pacote['itens']:
        if not i.get('presente') or i.get('simular') is False:
            continue
        if i.get('vistoriaId') != v['id'] or i.get('ambienteId') not in nomes:
            raise ValueError('Equipamento sem ambiente válido.')
        potencia = i.get('potInformada')
        if potencia is None or potencia == '':
            potencia = i.get('potTipica')
        # A criticidade entra junto, na ordem de COLUNAS. A coleta HTML já a
        # traz por equipamento, e é ela que monta o quadro de backup: sem
        # trazê-la aqui, o recorte voltaria a ser por ambiente e levaria a
        # geladeira e o forno de 4 kW no mesmo quadro.
        valores = [i['equipamento'], potencia, i.get('quantidade'), i.get('tipoInt'),
                   i.get('intervalo'), i.get('prob'), i.get('fd'), i.get('durMin'),
                   i.get('durMax'), i.get('modo'), i.get('criticidade') or 'NC']
        # Impede fórmulas originadas de texto ao abrir a planilha no Excel.
        if isinstance(valores[0], str) and valores[0].lstrip().startswith(('=', '+', '-', '@')):
            valores[0] = "'" + valores[0]
        linhas.setdefault(nomes[i['ambienteId']], []).append(valores)
    if not linhas:
        raise ValueError('Nenhum equipamento para o motor.')
    cenario = Cenario(nome=v['cliente'], segmento='residential',
                      comodos={n: Cenario._normalizar(pd.DataFrame(l, columns=COLUNAS)) for n, l in linhas.items()},
                      instancias={n: 1 for n in linhas})
    erros = [str(p) for p in cenario.validar() if p.impede]
    if erros:
        raise ValueError('Revise os parâmetros do levantamento: ' + '; '.join(erros))
    return cenario


def ler_conta(pacote: dict) -> tuple[str, bytes]:
    conta = pacote['vistoria'].get('conta') or {}
    tipos = {'application/pdf':('.pdf', b'%PDF-'), 'image/jpeg':('.jpg', b'\xff\xd8\xff'),
             'image/png':('.png', b'\x89PNG\r\n\x1a\n'), 'image/webp':('.webp', b'RIFF')}
    tipo = conta.get('tipo')
    if tipo not in tipos:
        raise ValueError('Tipo de conta inválido.')
    prefixo = f'data:{tipo};base64,'
    url = conta.get('dataUrl', '')
    if not url.startswith(prefixo) or len(url) > 14_000_000:
        raise ValueError('Anexo inválido ou grande demais.')
    try:
        dados = base64.b64decode(url[len(prefixo):], validate=True)
    except ValueError as e:
        raise ValueError('Anexo base64 inválido.') from e
    extensao, assinatura = tipos[tipo]
    if not dados.startswith(assinatura) or len(dados) > 10 * 1024 * 1024:
        raise ValueError('Conteúdo da conta não corresponde ao tipo informado.')
    if tipo == 'image/webp' and dados[8:12] != b'WEBP':
        raise ValueError('WebP inválido.')
    return extensao, dados


def importar(pacote: dict, destino: Path) -> None:
    cenario = converter(pacote)
    extensao, conta = ler_conta(pacote)
    # Valida e gera em memória antes de criar a pasta de entrega.
    planilha = cenario.para_planilha()
    metadados = dict(pacote['vistoria'])
    metadados['conta'] = {k: v for k, v in metadados['conta'].items() if k != 'dataUrl'}
    metadados['situacao'] = 'coletado_aguardando_conferencia'
    metadados['avisos'] = ['Não houve leitura automática da fatura nem simulação.',
                          'Potências típicas e parâmetros herdados do catálogo são estimativas.',
                          'O contorno desenhado é uma projeção em planta; inclinação, área útil e sombreamento exigem conferência.']
    destino.mkdir(parents=True, exist_ok=False)
    (destino / 'cargas.xlsx').write_bytes(planilha)
    (destino / ('conta' + extensao)).write_bytes(conta)
    (destino / 'cadastro.json').write_text(json.dumps(metadados, ensure_ascii=False, indent=2), encoding='utf-8')
    (destino / 'coleta-original.json').write_text(json.dumps(pacote, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pacote', type=Path)
    parser.add_argument('--destino', type=Path, required=True)
    args = parser.parse_args()
    importar(json.loads(args.pacote.read_text(encoding='utf-8-sig')), args.destino)
    print(f'Coleta preparada em {args.destino}. Nenhuma simulação executada.')
