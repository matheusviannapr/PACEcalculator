"""
A coleta residencial entrando inteira na calculadora.

O aplicativo de coleta roda no celular do cliente e devolve um pacote com
cadastro, a coordenada que ele confirmou sobre a própria casa, o contorno do
telhado que ele desenhou sobre a imagem de satélite, a conta de luz e o
levantamento por ambiente.

Até aqui, `converter` validava a localização e o contorno e devolvia **só** o
cenário de cargas: os dois dados mais caros de obter eram verificados e
descartados, e a calculadora pedia ambos de novo — o telhado sendo redesenhado
no escritório por quem nunca esteve na casa. O que estes testes garantem é que
nada disso se perde mais.
"""
from __future__ import annotations

import json

import pytest

from aurum.coleta import converter, ler_pacote


def _pacote(**ajustes) -> dict:
    """Um pacote mínimo no formato ``pace-coleta/1``, como o app o entrega."""
    vistoria = {
        "id": "v1",
        "cliente": "Família Teste",
        "contato": "teste@example.com",
        "endereco": "Rua das Palmeiras, 100",
        "cidade": "Petrópolis",
        "uf": "RJ",
        "localizacao": {
            "latitude": -22.4127, "longitude": -43.1421,
            "confirmadoEm": "2026-09-08",
        },
        "telhado": {
            # Um quadrado de ~30 m de lado, fechado, sem auto-interseção.
            "geojson": {
                "type": "Polygon",
                "coordinates": [[
                    [-43.14210, -22.41270],
                    [-43.14180, -22.41270],
                    [-43.14180, -22.41300],
                    [-43.14210, -22.41300],
                    [-43.14210, -22.41270],
                ]],
            },
            "confirmadoEm": "2026-09-08",
            "origem": "desenho",
        },
        "conta": {"nome": "conta.pdf", "tipo": "application/pdf",
                  "dataUrl": "data:application/pdf;base64,AA=="},
        "aceiteDados": "2026-09-08",
    }
    vistoria.update(ajustes)
    return {
        "formato": "pace-coleta/1",
        "vistoria": vistoria,
        "ambientes": [
            {"id": "a1", "nome": "Cozinha", "vistoriaId": "v1"},
            {"id": "a2", "nome": "Sala", "vistoriaId": "v1"},
        ],
        "itens": [
            {"vistoriaId": "v1", "ambienteId": "a1", "equipamento": "Geladeira",
             "potTipica": 150, "quantidade": 1, "tipoInt": "fixo",
             "intervalo": "00:00 as 23:59", "prob": 1.0, "fd": 1.0,
             "modo": "FIXO_100%", "criticidade": "C", "presente": True,
             "simular": True},
            {"vistoriaId": "v1", "ambienteId": "a2", "equipamento": "Chuveiro",
             "potTipica": 5500, "quantidade": 1, "tipoInt": "fixo",
             "intervalo": "06:00 as 09:00", "prob": 0.9, "fd": 1.0,
             "modo": "FIXO_100%", "criticidade": "NC", "presente": True,
             "simular": True},
        ],
    }


# ----------------------------------------------------------------------------
def test_o_pacote_devolve_tudo_e_nao_so_as_cargas():
    """
    O que `converter` validava e jogava fora.

    Coordenada e contorno do telhado eram verificados — a validação sempre foi
    rigorosa — e nenhum dos dois chegava a quem ia usar. A calculadora então
    pedia os dois de novo, e o telhado acabava redesenhado no escritório.
    """
    dados = ler_pacote(_pacote())

    assert dados.cliente == "Família Teste"
    assert dados.cidade == "Petrópolis" and dados.uf == "RJ"
    assert dados.local.startswith("Rua das Palmeiras")
    assert dados.tem_coordenada
    assert dados.latitude == pytest.approx(-22.4127)
    assert dados.longitude == pytest.approx(-43.1421)
    assert dados.telhado_geojson is not None
    assert dados.telhado_geojson["type"] == "Polygon"
    assert dados.conta_nome == "conta.pdf"


def test_a_criticidade_do_cliente_atravessa_ate_o_quadro_de_backup():
    """
    O app pergunta a criticidade item a item, e ela decide o inversor.

    Sem ela, o recorte volta a ser por ambiente: a geladeira crítica arrastaria
    o chuveiro de 5.500 W da mesma casa, e o inversor multiplicaria por causa
    de um equipamento que ninguém precisa no apagão.
    """
    cenario = ler_pacote(_pacote()).cenario
    assert cenario.tem_criticidade
    cenario.criticidades_essenciais = ("MC", "C")
    assert cenario.potencia_instalada_w() == pytest.approx(5650.0)
    assert cenario.potencia_instalada_w(True) == pytest.approx(150.0)


def test_o_segmento_e_residencia():
    """
    O aplicativo de coleta só existe para residência.

    O segmento escolhe a curva de referência contra a qual o resultado é
    conferido; adotar outra faria a conferência acusar divergência de um modelo
    que está certo.
    """
    assert ler_pacote(_pacote()).cenario.segmento == "residencia"


def test_o_contorno_do_cliente_vira_telhado_medido():
    """
    O dado mais caro do pacote: quem desenhou estava na frente da casa.

    Redesenhá-lo no escritório é pior — quem redesenha nunca viu o imóvel — e
    mais lento. O contorno vira área medida, orientação deduzida e arranjo de
    módulos sem ninguém tocar no mapa.
    """
    from aurum.pv.equipment import carregar_base
    from aurum.pv.telhado import dimensionar_no_telhado, telhado_de_geojson

    dados = ler_pacote(_pacote())
    telhado = telhado_de_geojson(
        {"type": "Feature", "geometry": dados.telhado_geojson},
        nome="Água 1", montagem="coplanar", inclinacao_deg=20.0,
        fator_obstaculos=0.90,
    )
    assert telhado.area_m2 > 100.0, "um quadrado de ~30 m de lado"
    assert telhado.area_inclinada_m2 > telhado.area_m2, "a projeção encolhe pelo cosseno"

    layout = dimensionar_no_telhado(telhado, carregar_base())
    assert layout is not None and layout.quantidade > 0
    assert layout.potencia_kwp > 0


def test_coleta_sem_telhado_continua_valendo():
    """
    Coleta antiga, anterior à exigência do contorno, não pode ser recusada.

    O estudo roda sem telhado — apenas sem afirmar que o sistema cabe na
    cobertura —, e é isso que o aviso diz.
    """
    dados = ler_pacote(_pacote(telhado=None))
    assert dados.telhado_geojson is None
    assert any("não afirmará" in a or "não traz contorno" in a for a in dados.avisos)


def test_pacote_sem_confirmacao_e_recusado():
    """
    Localização não confirmada é palpite do geocodificador, não do cliente.

    O estudo inteiro sai dessa coordenada — a série de geração vem dela —, e
    aceitar uma não confirmada seria construir tudo sobre um chute silencioso.
    """
    pacote = _pacote()
    pacote["vistoria"]["localizacao"]["confirmadoEm"] = ""
    with pytest.raises(ValueError, match="confirmação"):
        ler_pacote(pacote)


def test_o_caminho_antigo_continua_funcionando():
    """`converter` é usado pela linha de comando e não muda de contrato."""
    cenario = converter(_pacote())
    assert set(cenario.comodos) == {"Cozinha", "Sala"}
    assert cenario.total_de_equipamentos() == 2


def test_aceita_dicionario_ou_arquivo(tmp_path):
    caminho = tmp_path / "pace-VIST.json"
    caminho.write_text(json.dumps(_pacote()), encoding="utf-8")
    assert ler_pacote(caminho).cliente == ler_pacote(_pacote()).cliente
