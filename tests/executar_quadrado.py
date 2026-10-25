"""Exemplo reproduzível: JSON de vistoria fictícia com telhado de 10 x 10 m."""
import base64
import json
from pathlib import Path
from pyproj import Transformer
from aurum.estudo_coleta import configurar, executar, Opcoes

raiz = Path('outputs/teste-quadrado-20260909')
raiz.mkdir(parents=True, exist_ok=True)
ida = Transformer.from_crs(4326, 32723, always_xy=True)
volta = Transformer.from_crs(32723, 4326, always_xy=True)
x, y = ida.transform(-43.2, -22.9)
anel = [list(volta.transform(x+dx,y+dy)) for dx,dy in
        [(-5,-5),(5,-5),(5,5),(-5,5),(-5,-5)]]
stamp = '2026-09-09T12:00:00Z'
pacote = {'formato':'pace-coleta/1','exportadoEm':stamp,
 'teste':True,
 'vistoria':{'id':'TESTE-QUADRADO-100M2','cliente':'TESTE FICTÍCIO — telhado quadrado',
 'contato':'teste@example.com','endereco':'Endereço fictício, sem imóvel identificado',
 'cidade':'Rio de Janeiro','uf':'RJ','aceiteDados':stamp,'consumoKwh':600,
 'localizacao':{'latitude':-22.9,'longitude':-43.2,'confirmadoEm':stamp},
 'telhado':{'geojson':{'type':'Polygon','coordinates':[anel]},'confirmadoEm':stamp,
 'origem':'desenho_cliente_satelite'},
 'conta':{'nome':'conta-ficticia.pdf','tipo':'application/pdf',
 'dataUrl':'data:application/pdf;base64,'+base64.b64encode(
 b'%PDF-1.4\n% ANEXO FICTICIO PARA TESTE DO FLUXO, NAO E FATURA\n%%EOF').decode()}},
 'ambientes':[{'id':'a1','vistoriaId':'TESTE-QUADRADO-100M2','nome':'Casa'}], 'itens':[]}
for nome,watts,horario,criticidade,fd in [
 ('Geladeira',150,'00:00 as 23:59','C',0.4),
 ('Iluminação',100,'18:00 as 23:00','C',1),
 ('Roteador',15,'00:00 as 23:59','MC',1),
 ('TV',150,'18:00 as 23:00','P',1),
 ('Ar-condicionado',1200,'22:00 as 23:59','NC',0.7),
 ('Chuveiro',5500,'07:00 as 08:00','NC',1)]:
 pacote['itens'].append({'id':f'i{len(pacote["itens"])}','ambienteId':'a1',
 'vistoriaId':pacote['vistoria']['id'],'presente':True,'equipamento':nome,
 'potTipica':watts,'quantidade':1,'tipoInt':'fixo','intervalo':horario,
 'prob':1,'fd':fd,'modo':'FIXO_100%','criticidade':criticidade})
entrada = raiz/'levantamento-quadrado.json'
entrada.write_text(json.dumps(pacote,ensure_ascii=False,indent=2),encoding='utf-8')
opcoes = Opcoes(tarifa=0.98,tensao=220,autonomia=6,
                simulacoes=50,amostras=30,candidatos=4,permitir_sintetica=True)
dados,cfg = configurar(json.loads(entrada.read_text(encoding='utf-8')),opcoes)
resumo = {'teste':True,'lado_m':10,'area_m2':cfg.telhado.area_m2,
          'layout':cfg.layout.as_dict(),'consumo_mensal_kwh':600,
          'premissas':'Dados fictícios. Amostragem reduzida para teste; não é projeto executivo.'}
(raiz/'geometria-e-layout.json').write_text(json.dumps(resumo,ensure_ascii=False,indent=2),encoding='utf-8')
assert abs(cfg.telhado.area_m2 - 100)<0.01
print(json.dumps(resumo,ensure_ascii=False),flush=True)
executar(pacote,raiz/'resultado',opcoes,
         lambda msg,fracao:print(f'[{fracao:.0%}] {msg}',flush=True))
