"""
Testes de integração.

O teste de compilação exige uma distribuição LaTeX instalada; sem ela o teste
é pulado em vez de falhar, para não quebrar a suíte em máquina sem TeX.
Os testes marcados com ``rede`` só rodam com ``--rede`` e tocam OpenStreetMap
e PVGIS de verdade.
"""
from __future__ import annotations

import pytest

from aurum.proposal.render import compilar_pdf, encontrar_compilador, escrever_proposta

@pytest.fixture(scope="module")
def contexto_compilavel(base_equipamentos_modulo, perfil_curitiba_modulo):
    """Mesmo contexto de referência usado nos testes de proposta."""
    from tests.conftest import construir_contexto

    return construir_contexto(base_equipamentos_modulo, perfil_curitiba_modulo)


@pytest.mark.skipif(encontrar_compilador() is None, reason="nenhuma distribuição LaTeX instalada")
def test_proposta_compila_em_pdf(contexto_compilavel, tmp_path):
    """
    O teste que importa de verdade: o .tex gerado precisa virar PDF.

    A versão anterior produzia arquivos que abortavam a compilação por escape
    quebrado, f-string esquecida e Unicode não suportado.
    """
    gerados = escrever_proposta(contexto_compilavel, tmp_path, compilar=True, manter_auxiliares=True)
    log = gerados["tex"].with_suffix(".log")

    if gerados["pdf"] is None:  # falhou: mostra os erros do LaTeX no relatório
        texto = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
        erros = [linha for linha in texto.splitlines() if linha.startswith("!")]
        pytest.fail("pdflatex não gerou PDF:\n" + "\n".join(erros[:15]))

    assert gerados["pdf"].exists()
    assert gerados["pdf"].stat().st_size > 40_000, "PDF pequeno demais para ter o conteúdo esperado"


@pytest.mark.skipif(encontrar_compilador() is None, reason="nenhuma distribuição LaTeX instalada")
def test_compilacao_de_tex_invalido_nao_levanta(tmp_path):
    """Um .tex quebrado deve devolver None, não derrubar o lote de propostas."""
    quebrado = tmp_path / "quebrado.tex"
    quebrado.write_text(r"\documentclass{article}\begin{document}\comandoInexistente", encoding="utf-8")
    assert compilar_pdf(quebrado) is None


# ----------------------------------------------------------------------
# Rede
# ----------------------------------------------------------------------
def _pular_sem_rede(request):
    if not request.config.getoption("--rede", default=False):
        pytest.skip("use --rede para executar os testes de rede")


@pytest.mark.rede
def test_pvgis_devolve_producao_plausivel_no_brasil(request):
    _pular_sem_rede(request)
    from aurum.pv.solar import SolarResourceClient

    perfil = SolarResourceClient().perfil(-25.49, -49.33)
    assert perfil.fonte in {"pvgis", "pvwatts"}, f"nenhuma base respondeu: {perfil.aviso}"
    assert 900 < perfil.anual_kwh_por_kwp < 2000
    assert len(perfil.mensal_kwh_por_kwp) == 12
    assert perfil.azimuth_deg == 0.0, "hemisfério sul deve apontar ao Norte"


@pytest.mark.rede
def test_azimute_norte_supera_o_sul_no_brasil(request):
    """Mede o erro que existia na versão anterior contra a base real."""
    _pular_sem_rede(request)
    from aurum.pv.solar import SolarResourceClient

    cliente = SolarResourceClient()
    norte = cliente.perfil(-25.49, -49.33, azimuth_deg=0)
    sul = cliente.perfil(-25.49, -49.33, azimuth_deg=180)
    if not (norte.confiavel and sul.confiavel):
        pytest.skip("base de irradiação indisponível")
    assert sul.anual_kwh_por_kwp < norte.anual_kwh_por_kwp * 0.85


@pytest.mark.rede
def test_prospeccao_real_encontra_telhados(request):
    _pular_sem_rede(request)
    from aurum.pipeline import prospectar

    resultado = prospectar(
        bbox=(-49.3465, -25.5100, -49.3225, -25.4860),
        min_area_m2=1500,
        enriquecer=False,
        limite=10,
    )
    assert resultado.leads, "área industrial conhecida deveria render telhados"
    assert all(l.metrics.area_m2 >= 1500 for l in resultado.leads)
    assert resultado.estatisticas["tiles_ok"] > 0
