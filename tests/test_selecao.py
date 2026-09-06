"""
Separação entre prospectar e dimensionar.

A regra que estes testes protegem: **nenhum estudo é calculado antes de
alguém escolher os telhados**. Prospectar uma região é barato e olha tudo;
dimensionar é caro e só pode rodar no que foi escolhido.
"""
from __future__ import annotations

import csv
import json

import pytest

from aurum.geo.roofs import RoofLead
from aurum.pipeline import (
    OpcoesDimensionamento,
    carregar_leads,
    escrever_geojson,
    escrever_leads_csv,
    executar,
    filtrar_selecionados,
    ler_selecao_csv,
)


# ----------------------------------------------------------------------
# Ida e volta pelo disco
# ----------------------------------------------------------------------
def test_lead_sobrevive_ao_geojson(leads):
    """
    Sem essa reconstrução não dá para separar as duas etapas: a proposta
    precisaria varrer o OpenStreetMap de novo só para saber a geometria.
    """
    original = max(leads, key=lambda l: l.metrics.area_m2)
    original.company = {"nome": "Metalúrgica Teste", "telefone": "+55 41 3000-0000"}

    recuperado = RoofLead.from_geojson_feature(original.to_geojson_feature())

    assert recuperado is not None
    assert recuperado.lead_id == original.lead_id
    assert recuperado.building_type == original.building_type
    assert recuperado.metrics.area_m2 == pytest.approx(original.metrics.area_m2, rel=1e-6)
    assert recuperado.company["nome"] == "Metalúrgica Teste"
    assert recuperado.tags == original.tags
    assert recuperado.utm_epsg == original.utm_epsg


def test_geojson_invalido_devolve_none():
    assert RoofLead.from_geojson_feature({"properties": {}}) is None
    assert RoofLead.from_geojson_feature({"geometry": None, "properties": {}}) is None


def test_carregar_leads_de_pasta(leads, tmp_path):
    escrever_geojson(leads, tmp_path / "leads.geojson")
    # aceita a pasta ou o arquivo
    assert len(carregar_leads(tmp_path)) == len(leads)
    assert len(carregar_leads(tmp_path / "leads.geojson")) == len(leads)


def test_carregar_leads_de_pasta_inexistente(tmp_path):
    with pytest.raises(FileNotFoundError):
        carregar_leads(tmp_path / "nao_existe")


# ----------------------------------------------------------------------
# A planilha de seleção
# ----------------------------------------------------------------------
def test_planilha_sai_com_coluna_de_selecao_em_branco(leads, tmp_path):
    caminho = escrever_leads_csv(leads, tmp_path / "leads.csv")
    with caminho.open(encoding="utf-8-sig") as arquivo:
        linhas = list(csv.DictReader(arquivo))

    assert list(linhas[0].keys())[0] == "selecionar", "a coluna deve vir primeiro, à vista"
    assert all(linha["selecionar"] == "" for linha in linhas), "nada marcado por padrão"
    assert len(linhas) == len(leads)


def test_planilha_vazia_nao_quebra(tmp_path):
    caminho = escrever_leads_csv([], tmp_path / "leads.csv")
    assert caminho.exists()
    assert ler_selecao_csv(caminho) == set()


@pytest.mark.parametrize("marca", ["x", "X", "sim", "SIM", " s ", "1", "true", "ok", "y"])
def test_marcas_aceitas(leads, tmp_path, marca):
    """Quem preenche a planilha à mão escreve de tudo; todas essas valem."""
    caminho = escrever_leads_csv(leads, tmp_path / "leads.csv")
    _marcar(caminho, {leads[0].lead_id: marca})
    assert ler_selecao_csv(caminho) == {leads[0].lead_id}


@pytest.mark.parametrize("marca", ["", "  ", "nao", "0", "-"])
def test_marcas_recusadas(leads, tmp_path, marca):
    caminho = escrever_leads_csv(leads, tmp_path / "leads.csv")
    _marcar(caminho, {leads[0].lead_id: marca})
    assert ler_selecao_csv(caminho) == set()


def test_planilha_sem_a_coluna_avisa(leads, tmp_path):
    caminho = tmp_path / "leads.csv"
    with caminho.open("w", newline="", encoding="utf-8-sig") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=["lead_id", "nome"])
        escritor.writeheader()
        escritor.writerow({"lead_id": leads[0].lead_id, "nome": "x"})
    with pytest.raises(ValueError, match="não tem a coluna"):
        ler_selecao_csv(caminho)


def test_filtrar_mantem_apenas_os_escolhidos(leads):
    escolhido = leads[0].lead_id
    resultado = filtrar_selecionados(leads, [escolhido, "way/000000", " "])
    assert [l.lead_id for l in resultado] == [escolhido]


def test_selecao_completa_pelo_disco(leads, tmp_path):
    """O caminho inteiro: gravar, marcar dois, recarregar e filtrar."""
    escrever_geojson(leads, tmp_path / "leads.geojson")
    caminho = escrever_leads_csv(leads, tmp_path / "leads.csv")

    escolhidos = {leads[0].lead_id, leads[-1].lead_id}
    _marcar(caminho, {identificador: "x" for identificador in escolhidos})

    recarregados = carregar_leads(tmp_path)
    marcados = ler_selecao_csv(tmp_path)
    selecionados = filtrar_selecionados(recarregados, marcados)

    assert {l.lead_id for l in selecionados} == escolhidos
    assert len(selecionados) < len(recarregados)


# ----------------------------------------------------------------------
# A regra em si
# ----------------------------------------------------------------------
def test_executar_recusa_calcular_sem_selecao():
    """
    Sem 'selecao' nem 'top', a função não pode sair dimensionando a região
    inteira. Precisa recusar antes de tocar a rede.
    """
    with pytest.raises(ValueError, match="quais telhados calcular"):
        executar(regiao="Curitiba")


def test_prospectar_nao_importa_o_motor_de_dimensionamento():
    """
    Guarda de arquitetura: a etapa de prospecção não pode acabar chamando o
    dimensionamento por descuido numa refatoração futura.
    """
    import ast
    import inspect

    from aurum import pipeline

    fonte = inspect.getsource(pipeline.prospectar)
    arvore = ast.parse(fonte)
    chamadas = {
        no.func.id if isinstance(no.func, ast.Name) else getattr(no.func, "attr", "")
        for no in ast.walk(arvore)
        if isinstance(no, ast.Call)
    }
    proibidas = {"dimensionar_lead", "dimensionar_lote", "melhor_sistema", "calcular_layout"}
    assert not (chamadas & proibidas), f"prospectar() está calculando estudo: {chamadas & proibidas}"


def _marcar(caminho, marcas: dict[str, str]) -> None:
    """Escreve marcas na coluna de seleção, como faria quem abre a planilha."""
    with caminho.open(encoding="utf-8-sig") as arquivo:
        linhas = list(csv.DictReader(arquivo))
        campos = list(linhas[0].keys())
    for linha in linhas:
        if linha["lead_id"] in marcas:
            linha["selecionar"] = marcas[linha["lead_id"]]
    with caminho.open("w", newline="", encoding="utf-8-sig") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(linhas)


# ----------------------------------------------------------------------
# Guarda de área
# ----------------------------------------------------------------------
def test_regiao_grande_demais_e_recusada_antes_de_consultar():
    """
    Um município inteiro passa de mil km² e levaria centenas de consultas ao
    Overpass. A recusa precisa vir antes de qualquer tráfego de rede, senão o
    usuário só descobre o engano depois de meia hora.
    """
    from unittest.mock import patch

    from aurum.pipeline import AreaGrandeDemais, prospectar

    enorme = (-50.5, -25.5, -49.5, -24.5)  # ~1 grau de lado
    with patch("aurum.pipeline.OverpassClient") as cliente:
        with pytest.raises(AreaGrandeDemais) as erro:
            prospectar(bbox=enorme)
        cliente.assert_not_called()

    assert erro.value.area_km2 > 400
    assert erro.value.tiles_estimados > 100
    assert "Recorte em bairros" in str(erro.value)


def test_regiao_grande_passa_com_confirmacao():
    """A recusa é um freio, não uma proibição."""
    from unittest.mock import MagicMock, patch

    from aurum.pipeline import prospectar

    enorme = (-50.5, -25.5, -49.5, -24.5)
    falso = MagicMock()
    falso.fetch_buildings.return_value = ([], {"tiles_ok": 0, "tiles_total": 0})
    falso.fetch_uso_do_solo.return_value = []

    with patch("aurum.pipeline.OverpassClient", return_value=falso):
        resultado = prospectar(bbox=enorme, confirmar_area_grande=True, enriquecer=False)

    assert resultado.total == 0
    assert falso.fetch_buildings.called


def test_resumo_da_area_antecipa_o_custo():
    """A interface precisa poder avisar antes, não depois."""
    from aurum.pipeline import resumo_da_area

    pequena = resumo_da_area(bbox=(-49.35, -25.51, -49.32, -25.48))
    enorme = resumo_da_area(bbox=(-50.5, -25.5, -49.5, -24.5))

    assert not pequena["grande_demais"]
    assert enorme["grande_demais"]
    assert enorme["tiles"] > pequena["tiles"]
    assert enorme["minutos_estimados"] > pequena["minutos_estimados"]
