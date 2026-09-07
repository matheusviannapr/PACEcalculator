import base64
import copy
import io

import pytest

from aurum.coleta import converter, importar, ler_conta, validar_telhado
from aurum.demanda.cenario import Cenario


def pacote():
    return {'formato':'pace-coleta/1', 'vistoria': {
        'id':'v1', 'cliente':'Teste', 'contato':'teste@example.com', 'endereco':'Rua de teste, 1',
        'cidade':'Teste', 'aceiteDados':'2026-09-07',
        'localizacao':{'latitude':-22.9, 'longitude':-43.2, 'confirmadoEm':'2026-09-07'},
        'conta':{'tipo':'application/pdf', 'dataUrl':'data:application/pdf;base64,' + base64.b64encode(b'%PDF-1.4\nfixture').decode()}},
        'ambientes':[{'id':'a1','vistoriaId':'v1','nome':'Sala'}],
        'itens':[{'ambienteId':'a1','vistoriaId':'v1','presente':True,'equipamento':'TV',
                  'potTipica':100,'quantidade':1,'tipoInt':'fixo','intervalo':'18:00 as 22:00',
                  'prob':1,'fd':1,'modo':'FIXO_100%'}]}


def test_planilha_e_anexo_preservados(tmp_path):
    p = pacote()
    destino = tmp_path / 'recebido'
    importar(p, destino)
    c = Cenario.de_planilha(destino / 'cargas.xlsx')
    assert c.potencia_instalada_w() == 100
    assert (destino / 'conta.pdf').read_bytes() == b'%PDF-1.4\nfixture'
    assert 'dataUrl' not in (destino / 'cadastro.json').read_text()
    with pytest.raises(FileExistsError):
        importar(p, destino)


def test_coordenadas_e_confirmacao():
    p = pacote()
    p['vistoria']['localizacao']['latitude'] = float('nan')
    with pytest.raises(ValueError, match='Coordenadas'):
        converter(p)
    p = pacote()
    p['vistoria']['localizacao']['confirmadoEm'] = None
    with pytest.raises(ValueError, match='confirmação'):
        converter(p)


def test_nomes_de_abas_e_cargas_nao_se_misturam():
    p = pacote()
    p['ambientes'][0]['nome'] = 'Quarto/A'
    p['ambientes'].append({'id':'a2','vistoriaId':'v1','nome':'Quarto/A'})
    outro = copy.deepcopy(p['itens'][0]); outro['ambienteId'] = 'a2'; outro['potInformada'] = 250
    p['itens'].append(outro)
    c = converter(p)
    assert len(c.comodos) == 2
    assert Cenario.de_planilha(io.BytesIO(c.para_planilha())).potencia_instalada_w() == 350


def test_anexo_falso_e_carga_invalida():
    p = pacote(); p['vistoria']['conta']['dataUrl'] = 'data:application/pdf;base64,SGVsbG8='
    with pytest.raises(ValueError, match='Conteúdo'):
        ler_conta(p)
    p = pacote(); p['itens'][0]['quantidade'] = -1
    with pytest.raises(ValueError):
        converter(p)


def test_poligono_preservado_e_validado(tmp_path):
    p = pacote()
    p['vistoria']['telhado'] = {'geojson': {'type':'Polygon', 'coordinates':[
        [[-43,-22],[-42.9999,-22],[-42.9999,-21.9999],[-43,-21.9999],[-43,-22]]]},
        'confirmadoEm':'2026-09-07','origem':'desenho_cliente_satelite'}
    importar(p, tmp_path / 'telhado')
    import json
    salvo = json.loads((tmp_path / 'telhado/cadastro.json').read_text(encoding='utf-8'))
    assert salvo['telhado'] == p['vistoria']['telhado']


@pytest.mark.parametrize('anel', [
    [[0,0],[1,1],[0,1],[1,0],[0,0]],
    [[0,0],[1,0],[2,0],[0,0]],
    [[0,0],[1,0],[1,1]],
    [[0,0],[float('nan'),0],[1,1],[0,0]],
])
def test_poligonos_invalidos(anel):
    with pytest.raises(ValueError):
        validar_telhado({'geojson':{'type':'Polygon','coordinates':[anel]},'confirmadoEm':'2026-09-07'})
