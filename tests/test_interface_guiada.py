"""Contratos da navegação guiada e da proteção contra perda de dados."""
from pathlib import Path
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / 'app.py')


def abrir():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    return at


def test_etapas_futuras_explicitas_e_bloqueadas():
    at = abrir()
    assert at.button(key='nav1').disabled is False
    assert at.button(key='nav2').disabled is True
    assert any('Local ainda não confirmado' in x.value for x in at.warning)
    at.text_input(key='e_nome').set_value('Estudo de demonstração').run()
    at.button(key='seguir1').click().run()
    assert not at.exception
    assert at.session_state['e_passo'] == 2
    assert at.button(key='nav1').disabled is False
    assert at.button(key='nav3').disabled is True
    at.button(key='nav1').click().run()
    assert at.text_input(key='e_nome').value == 'Estudo de demonstração'


def test_reinicio_exige_confirmacao_e_limpa_widgets():
    at = abrir()
    at.text_input(key='e_nome').set_value('Não perder sem confirmar').run()
    reiniciar = next(b for b in at.button if b.label == 'Recomeçar do zero')
    assert reiniciar.disabled
    at.checkbox(key='confirmar_reinicio').check().run()
    next(b for b in at.button if b.label == 'Recomeçar do zero').click().run()
    assert not at.exception
    assert at.session_state['e_passo'] == 1
    assert at.text_input(key='e_nome').value == ''
    assert not at.checkbox(key='confirmar_reinicio').value
