"""
Testes da imagem do pré-relatório.

O que se garante aqui é o contrato com o WhatsApp — um PNG de 1080×1350, o
retrato 4:5 que aparece inteiro na conversa — e que o desenho não quebra com
o resumo magro: estudo sem bateria recomendada, sem curvas, sem conta solar.
Esse é justamente o caso em que a imagem tem de sair mesmo assim, só mais
curta, e sem inventar número nenhum para preencher o buraco.

Se a imagem está *bonita* não se testa aqui: isso se olha.
"""
from __future__ import annotations

import io
from dataclasses import replace

import pytest
from PIL import Image

from aurum.pre_relatorio.imagem import _lista, _pct, _reais, desenhar
from aurum.pre_relatorio.resumo import Consumidor, ResumoPreRelatorio, exemplo

PNG = b"\x89PNG\r\n\x1a\n"


def _tamanho(dados: bytes) -> tuple[int, int]:
    return Image.open(io.BytesIO(dados)).size


@pytest.mark.parametrize("tipo", ["casa", "negocio"])
def test_exemplo_sai_png_1080x1350(tipo, tmp_path):
    destino = tmp_path / f"{tipo}.png"
    dados = desenhar(exemplo(tipo), destino)
    assert dados.startswith(PNG)
    assert _tamanho(dados) == (1080, 1350)
    # Gravou o mesmo que devolveu.
    assert destino.read_bytes() == dados


def test_resumo_vazio_nao_quebra():
    """Sem banco, sem consumidores, sem curvas, sem solar: sai a imagem curta."""
    vazio = ResumoPreRelatorio(cliente="João", protocolo="PACE-0", cidade_uf="Rio/RJ",
                               origem_consumo="respostas")
    assert vazio.autonomia_h is None and not vazio.consumidores
    dados = desenhar(vazio)
    assert dados.startswith(PNG)
    assert _tamanho(dados) == (1080, 1350)


def test_textos_longos_nao_quebram():
    """Nome enorme, quatro críticos e seis consumidores: tem de caber, não estourar."""
    r = replace(
        exemplo("negocio"),
        empresa="Escola Municipal de Educação Infantil Professora Maria Aparecida dos Santos",
        setor="Escola ou creche",
        criticos=["o freezer da merenda", "a internet do laboratório",
                  "as câmeras de segurança", "a bomba d'água"],
        consumidores=[Consumidor(f"Carga número {i} com nome comprido demais", 0.1) for i in range(6)],
        investimento_brl=1_850_000, payback_anos=3.8, origem_consumo="valor",
    )
    assert _tamanho(desenhar(r)) == (1080, 1350)


def test_formatacao_pt_br():
    assert _reais(1250.4).replace(" ", " ") == "R$ 1.250"
    assert _reais(1_850_000).replace(" ", " ") == "R$ 1.850.000"
    assert _pct(0.62) == "62%"
    assert _lista(["a geladeira", "a internet", "o ar"]) == "a geladeira, a internet e o ar"
    assert _lista(["a geladeira"]) == "a geladeira"
    assert _lista([]) == ""
