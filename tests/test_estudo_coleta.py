import json
import pytest
from aurum.estudo_coleta import Opcoes, configurar, executar
from test_coleta import pacote


def entrada():
    p = pacote()
    p['itens'][0]['criticidade'] = 'C'
    p['vistoria']['consumoKwh'] = '300,5'
    return p


def test_configuracao_preserva_dados():
    dados, cfg = configurar(entrada(), Opcoes(tarifa=0.9, tensao=220))
    assert cfg.latitude == -22.9
    assert cfg.longitude == -43.2
    assert cfg.fatura.consumo_mensal_kwh == 300.5
    assert cfg.consumo_anual_kwh == 3606
    assert cfg.tensao_rede_v == 220
    assert len(cfg.comodos_backup) == 1
    assert cfg.permitir_serie_sintetica is False


def test_entrada_invalida_nao_cria_saida(tmp_path):
    p = entrada()
    p['vistoria']['consumoKwh'] = 'NaN'
    with pytest.raises(ValueError):
        executar(p, tmp_path/'erro', Opcoes(tarifa=1, tensao=220))
    assert not (tmp_path/'erro').exists()


def test_falha_do_motor_registrada(tmp_path, monkeypatch):
    from aurum.bateria import estudo
    def falhar(*a, **k):
        raise RuntimeError('PVGIS indisponível')
    monkeypatch.setattr(estudo, 'executar_estudo', falhar)
    destino = tmp_path/'falha'
    with pytest.raises(RuntimeError):
        executar(entrada(), destino, Opcoes(tarifa=1,tensao=220))
    m = json.loads((destino/'execucao.json').read_text(encoding='utf-8'))
    assert m['situacao'] == 'falhou'
    assert not (destino/'estudo-pace.zip').exists()


def test_estudo_real_com_recurso_solar_de_teste(tmp_path, monkeypatch):
    # Só a fonte solar externa é substituída; cálculo e dossiê são reais.
    from aurum.bateria import estudo
    from aurum.bateria.geracao import serie_sintetica
    solar = serie_sintetica(-22.9, -43.2, 0, 20, 14, anos=(2020,2020),
                           aviso='TESTE: recurso solar sintético, não é estudo de cliente.')
    monkeypatch.setattr(estudo, 'obter_serie_horaria', lambda *a, **k: solar)
    destino = executar(entrada(), tmp_path/'entrega', Opcoes(
        tarifa=1, tensao=220, simulacoes=3, amostras=2, candidatos=1, compilar_pdf=False))
    assert (destino/'estudo/estudo.tex').stat().st_size > 1000
    assert (destino/'estudo-pace.zip').is_file()
    m = json.loads((destino/'execucao.json').read_text(encoding='utf-8'))
    assert m['situacao'] == 'concluido'
    assert m['pdf_gerado'] is False
    with pytest.raises(FileExistsError):
        executar(entrada(), destino, Opcoes(tarifa=1,tensao=220))
