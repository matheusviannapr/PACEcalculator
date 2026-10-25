"""
O questionário do cliente (vistoria/cliente) gera um pacote que a calculadora lê.

O aplicativo traduz "o que fez falta", a rotina e o resto da casa em itens do
catálogo da vistoria. Se a tradução gerar um item que o cenário recusa, o
cliente responde tudo e o escritório não consegue abrir o arquivo — por isso o
teste roda o JavaScript de verdade no Node e passa o resultado pelo mesmo
:func:`aurum.coleta.ler_pacote` que a tela usa.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from aurum.coleta import ler_pacote

RAIZ = Path(__file__).resolve().parents[1] / "vistoria"
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(NODE is None, reason="Node.js ausente")

# PNG de 1 px: a conta precisa passar pela checagem de assinatura do anexo.
PNG = ("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
       "z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")

RESPOSTAS = {
    "falta": {"geladeira": 1, "internet": 1, "ar": 2, "bomba": 1, "saude": 1},
    "tolerancia": {"geladeira": "C", "internet": "MC", "ar": "P", "bomba": "C"},
    "bomTer": {"tv": 1, "trabalho": 2},
    "casa": {"geladeira": 1, "tv": 1, "microondas": 1, "chuveiro": 2, "forno": 1, "carro": 1},
    "rotina": {"pessoas": "4", "diaSemana": "uma", "almoco": "sempre", "fimDeSemana": "ficam",
               "acorda": "cedo", "janta": "tarde", "dorme": "tarde", "arNoite": "noite_toda",
               "madrugada": "as_vezes"},
    "apagao": {"frequencia": "mensal", "duracao": "meio_dia"},
    "fontes": {"nobreak": True},
    "conta": {"nome": "conta.png", "tipo": "image/png", "dataUrl": PNG},
    "contaValor": "1.250,50",
    "dados": {"nome": "Maria Teste", "whatsapp": "(21) 99999-0000", "email": "",
              "endereco": "Rua A, 10", "cidade": "Rio de Janeiro", "uf": "RJ"},
    "local": {"latitude": -22.95, "longitude": -43.19, "origem": "toque_cliente_satelite",
              "confirmadoEm": "2026-09-26T12:00:00Z"},
    "aceite": True,
}

SCRIPT = """
global.window = globalThis;
require(process.argv[1] + '/dados.js');
require(process.argv[1] + '/cliente/negocio.js');
require(process.argv[1] + '/cliente/cliente.js');
const M = globalThis.PQ.modelo;
const e = Object.assign(M.novoEstado(), JSON.parse(process.argv[2]));
process.stdout.write(JSON.stringify(M.montarPacote(e, '2026-09-26T12:00:00.000Z')));
"""


def _pacote(respostas: dict) -> dict:
    saida = subprocess.run([NODE, "-e", SCRIPT, str(RAIZ), json.dumps(respostas)],
                           capture_output=True, text=True, encoding="utf-8", check=True)
    return json.loads(saida.stdout)


@pytest.fixture(scope="module")
def pacote() -> dict:
    return _pacote(RESPOSTAS)


def test_a_calculadora_le_o_pacote(pacote):
    dados = ler_pacote(pacote)
    assert dados.cliente == "Maria Teste"
    assert dados.tem_coordenada
    assert dados.cenario.total_de_equipamentos() > 0


def test_o_que_fez_falta_leva_a_criticidade_respondida(pacote):
    crit = {(i["equipId"], i["origem"]): i["criticidade"] for i in pacote["itens"]}
    assert crit[("EQ0138", "fez_falta")] == "MC"   # internet: nem um minuto
    assert crit[("EQ0048", "fez_falta")] == "C"    # geladeira: até uma hora
    assert crit[("EQ0151", "fez_falta")] == "MC"   # saúde sem resposta: padrão MC
    assert crit[("EQ0028", "bom_ter")] == "P"
    assert crit[("EQ0113", "tem_em_casa")] == "NC"


def test_escolha_repetida_nao_duplica(pacote):
    # A geladeira fez falta e também está em "tem em casa": entra uma vez só.
    geladeiras = [i for i in pacote["itens"] if i["equipId"] == "EQ0048"]
    assert len(geladeiras) == 1 and geladeiras[0]["origem"] == "fez_falta"


def test_nenhuma_janela_atravessa_a_meia_noite(pacote):
    for item in pacote["itens"]:
        inicio, fim = item["intervalo"].split(" as ")
        assert inicio < fim, item["equipamento"]


def test_rotina_desloca_as_janelas(pacote):
    por_nome = {i["equipamento"]: i for i in pacote["itens"]}
    assert por_nome["Chuveiro elétrico 5.500 W"]["intervalo"] == "05:00 as 08:00"      # acorda cedo
    assert por_nome["Micro-ondas"]["intervalo"] == "19:30 as 23:30"                   # janta tarde
    assert por_nome["Micro-ondas — almoço"]["intervalo"] == "11:00 as 14:00"          # almoça em casa
    assert por_nome["Ar-condicionado split 12.000 BTU/h — madrugada"]["intervalo"] == "00:00 as 06:00"
    assert por_nome["Micro-ondas"]["comodo"] == "Cozinha"  # o perfil de ocupação acha a cozinha pelo nome


def test_cada_ajuste_aparece_uma_vez(pacote):
    # O item do almoço tem janela própria: não passa pelo deslocamento do jantar.
    ajustes = pacote["questionario"]["ajustesAplicados"]
    assert len(ajustes) == len(set(ajustes))


def test_questionario_vai_cru_no_pacote(pacote):
    q = pacote["questionario"]
    assert q["formato"] == "pace-questionario/1"
    assert q["rotina"]["diaSemana"] == "uma"
    assert q["autonomiaSugeridaH"] == 6
    assert pacote["vistoria"]["contaValorMensal"] == 1250.5
    assert pacote["vistoria"]["ups"] is True


def test_protocolo_e_estavel_e_vai_no_pacote():
    respostas = dict(RESPOSTAS, id="CLI-MUJ43X9K-YZI")
    a, b = _pacote(respostas), _pacote(respostas)
    # Reenviar o mesmo estudo dá o mesmo protocolo: é por ele que o receptor deduplica.
    assert a["protocolo"] == b["protocolo"] == a["vistoria"]["protocolo"] == "PACE-X9KYZI"


FUNCOES = """
global.window = globalThis;
require(process.argv[1] + '/cliente/negocio.js');
require(process.argv[1] + '/cliente/cliente.js');
const M = globalThis.PQ.modelo;
const e = Object.assign(M.novoEstado(), JSON.parse(process.argv[2]));
const t = M.triagem(e);
process.stdout.write(JSON.stringify({triagem: t, mensagem: M.mensagemWhatsApp(e, {remetente: 'Matheus'}),
  valores: ['450', 'R$ 1.250,50', '1250.50', '1.250', ''].map(M.valorEmReais),
  zaps: ['(21) 99999-0000', '5521999990000', '999'].map(M.whatsappInternacional)}));
"""


def _funcoes(respostas: dict) -> dict:
    saida = subprocess.run([NODE, "-e", FUNCOES, str(RAIZ), json.dumps(respostas)],
                           capture_output=True, text=True, encoding="utf-8", check=True)
    return json.loads(saida.stdout)


def test_valor_da_conta_e_whatsapp_sao_lidos_como_o_cliente_digita():
    r = _funcoes(RESPOSTAS)
    assert r["valores"] == [450, 1250.5, 1250.5, 1250, None]
    assert r["zaps"] == ["5521999990000", "5521999990000", ""]


def test_triagem_quente_e_mensagem_personalizada():
    r = _funcoes(RESPOSTAS)   # conta R$ 1.250, falta luz todo mês, aparelho de saúde
    assert r["triagem"]["classe"] == "quente"
    msg = r["mensagem"]
    assert msg.startswith("Oi, Maria! Aqui é Matheus, da PACE.")
    assert "todo mês" in msg and "Rio de Janeiro" in msg
    # O mais crítico primeiro: internet e aparelho de saúde são MC.
    assert "o que mais faz falta é a internet, o aparelho de saúde e a geladeira" in msg
    assert "ligação" in msg          # quente recebe convite para conversar
    assert "R$" not in msg           # nenhum número antes da conversa


def test_triagem_morna_e_fria():
    base = dict(RESPOSTAS, falta={"geladeira": 1}, tolerancia={"geladeira": "C"})
    morno = _funcoes(dict(base, contaValor="500", apagao={"frequencia": "rara", "duracao": "minutos"}))
    frio = _funcoes(dict(base, contaValor="200", apagao={"frequencia": "rara", "duracao": "minutos"}))
    sem_conta = _funcoes(dict(base, contaValor="", apagao={"frequencia": "semanal", "duracao": "horas"}))
    assert morno["triagem"]["classe"] == "morno"
    assert frio["triagem"]["classe"] == "frio"
    assert sem_conta["triagem"]["classe"] == "morno"   # falta luz toda semana já basta
    assert "ligação" not in frio["mensagem"]


def test_pacote_leva_a_triagem(pacote):
    assert pacote["questionario"]["triagem"]["rotulo"] == "QUENTE"


OUTROS = dict(
    RESPOSTAS,
    outros={"outro-a1": {"nome": "Aquário marinho", "grupo": "falta"},
            "outro-b2": {"nome": "Sauna", "grupo": "casa"},
            "outro-zz": {"nome": "Desmarcado", "grupo": "casa"}},   # sem quantidade: não entra
    falta=dict(RESPOSTAS["falta"], **{"outro-a1": 2}),
    tolerancia=dict(RESPOSTAS["tolerancia"], **{"outro-a1": "MC"}),
    casa=dict(RESPOSTAS["casa"], **{"outro-b2": 1}),
    observacoes="  A luz cai sempre que chove.\nQuero carregar o carro à noite.  ",
)


def test_outro_equipamento_entra_sem_simular_e_a_calculadora_le():
    p = _pacote(OUTROS)
    outros = {i["equipamento"]: i for i in p["itens"] if i["equipamento"].startswith("Outro:")}
    assert set(outros) == {"Outro: Aquário marinho", "Outro: Sauna"}
    aq = outros["Outro: Aquário marinho"]
    assert (aq["quantidade"], aq["criticidade"], aq["simular"], aq["equipId"]) == (2, "MC", False, "EQ0246")
    assert outros["Outro: Sauna"]["criticidade"] == "NC"
    # Potência zero travaria o cenário: sem simular, ele é pulado e o resto segue.
    dados = ler_pacote(p)
    nomes = {n for df in dados.cenario.comodos.values() for n in df.iloc[:, 0]}
    assert not any(str(n).startswith("Outro:") for n in nomes)
    assert set(p["questionario"]["outros"]) == {"outro-a1", "outro-b2"}


def test_observacoes_vao_no_pacote():
    p = _pacote(OUTROS)
    esperado = "A luz cai sempre que chove.\nQuero carregar o carro à noite."
    assert p["vistoria"]["obs"] == esperado
    assert p["questionario"]["observacoes"] == esperado


def test_outro_critico_aparece_na_mensagem():
    r = _funcoes(OUTROS)
    assert "o aquário marinho" in r["mensagem"]


def test_casa_minima_tambem_fecha():
    minimo = dict(RESPOSTAS, falta={}, tolerancia={}, bomTer={}, casa={}, rotina={})
    dados = ler_pacote(_pacote(minimo))
    # Sem nada marcado, a casa ainda tem luz e tomadas estimadas.
    assert dados.cenario.total_de_equipamentos() >= 2


# ---------------------------------------------------------------------------
# Pequenos negócios: o mesmo app, com setor e funcionamento no lugar da rotina.
# ---------------------------------------------------------------------------
PADARIA = dict(
    RESPOSTAS,
    tipo="negocio", setor="padaria",
    falta={"n_forno_paes": 1, "n_expositor": 2, "n_pdv": 1},
    tolerancia={"n_forno_paes": "C", "n_expositor": "MC", "n_pdv": "MC"},
    bomTer={"n_internet": 1},
    casa={"n_masseira": 1, "n_letreiro": 1, "n_ar": 2},
    rotina={},
    negocio={"dias": "todos", "abre": "madrugada", "fecha": "noite", "pico": "manha",
             "producao": "muito", "tamanho": "2", "impacto": {"mercadoria": True, "vendas": True}},
    contaValor="2.400",
    dados=dict(RESPOSTAS["dados"], empresa="Padaria Pão Quente"),
)


def test_negocio_a_calculadora_le_com_o_segmento_do_setor():
    p = _pacote(PADARIA)
    dados = ler_pacote(p)
    assert dados.cenario.segmento == "restaurante"          # padaria usa o modelo de restaurante
    assert p["questionario"]["perfilTipico"] == "padaria"   # e a curva típica de padaria
    assert p["vistoria"]["imovel"] == "Padaria ou confeitaria"
    assert p["vistoria"]["empresa"] == "Padaria Pão Quente"


def test_negocio_horario_de_funcionamento_vira_janela():
    p = _pacote(PADARIA)
    por_nome = {i["equipamento"]: i for i in p["itens"]}
    # Abre antes das 6h (5h) e produz 3h30 antes: o forno vai das 01:30 às 07:00.
    assert por_nome["Forno elétrico de pães"]["intervalo"] == "01:30 as 07:00"
    assert por_nome["Masseira"]["intervalo"] == "01:30 as 06:00"
    assert por_nome["Caixa e maquininha (PDV)"]["intervalo"] == "05:00 as 22:00"   # aberto
    assert por_nome["Expositor refrigerado"]["intervalo"] == "00:00 as 23:59"      # contínuo
    assert por_nome["Letreiro / fachada"]["intervalo"] == "18:00 as 23:59"
    assert por_nome["Expositor refrigerado"]["quantidade"] == 2
    assert por_nome["Expositor refrigerado"]["criticidade"] == "MC"
    assert "confirmar" in por_nome["Forno elétrico de pães"]["obs"]   # potência típica, não do motor
    for item in p["itens"]:
        ini, fim = item["intervalo"].split(" as ")
        assert ini < fim, item["equipamento"]


def test_escola_turnos_viram_uma_janela():
    escola = dict(PADARIA, setor="escola", falta={"n_ar_salas": 10}, tolerancia={}, bomTer={}, casa={"n_computadores": 20},
                  negocio={"dias": "seg_sex", "turnos": {"manha": True, "tarde": True}, "tamanho": "3", "impacto": {"aulas": True}},
                  dados=dict(RESPOSTAS["dados"], empresa="Colégio Teste"))
    p = _pacote(escola)
    por_nome = {i["equipamento"]: i for i in p["itens"]}
    assert por_nome["Ar-condicionado split 18.000 BTU"]["intervalo"] == "07:00 as 18:00"
    assert por_nome["Ar-condicionado split 18.000 BTU"]["quantidade"] == 10
    assert ler_pacote(p).cenario.segmento == "escola"


def test_negocio_triagem_e_mensagem():
    r = _funcoes(PADARIA)   # conta R$ 2.400, falta todo mês, perde mercadoria
    assert r["triagem"]["classe"] == "quente"
    assert "perde mercadoria" in r["triagem"]["motivos"]
    msg = r["mensagem"]
    assert "da sua padaria — Padaria Pão Quente" in msg
    assert "sem perder mercadoria nem ficar sem vender" in msg
    assert "o balcão refrigerado" in msg
    frio = _funcoes(dict(PADARIA, contaValor="600", apagao={"frequencia": "rara", "duracao": "minutos"},
                         negocio=dict(PADARIA["negocio"], impacto={})))
    assert frio["triagem"]["classe"] == "frio"


TODOS_OS_SETORES = """
global.window = globalThis;
require(process.argv[1] + '/dados.js');
require(process.argv[1] + '/cliente/negocio.js');
require(process.argv[1] + '/cliente/cliente.js');
const M = globalThis.PQ.modelo, N = globalThis.PQN, base = JSON.parse(process.argv[2]);
const saida = N.SETORES.map(s => {
  // Cada setor com o que vem marcado de fábrica e as três primeiras respostas de cada pergunta.
  const negocio = {impacto: {[s.impactos[0]]: true}};
  N.perguntasFuncionamento(s.id).forEach(p => { negocio[p.id] = p.multi ? {[p.opcoes[0][0]]: true} : p.opcoes[Math.min(1, p.opcoes.length - 1)][0]; });
  const cartoes = N.cartoesDoSetor(s.id);
  const e = Object.assign(M.novoEstado(), base, {tipo: 'negocio', setor: s.id, negocio,
    falta: {[cartoes[0].id]: 1}, tolerancia: {[cartoes[0].id]: 'MC'}, bomTer: {}, casa: Object.assign({}, s.padrao)});
  return {setor: s.id, pacote: M.montarPacote(e, '2026-09-27T12:00:00.000Z'), mensagem: M.mensagemWhatsApp(e)};
});
process.stdout.write(JSON.stringify(saida));
"""


def test_todos_os_setores_geram_pacote_que_a_calculadora_le():
    base = dict(RESPOSTAS, dados=dict(RESPOSTAS["dados"], empresa="Negócio Teste"))
    saida = subprocess.run([NODE, "-e", TODOS_OS_SETORES, str(RAIZ), json.dumps(base)],
                           capture_output=True, text=True, encoding="utf-8", check=True)
    casos = json.loads(saida.stdout)
    assert len(casos) == 16
    for caso in casos:
        p = caso["pacote"]
        dados = ler_pacote(p)
        assert dados.cenario.segmento == p["questionario"]["segmentoMotor"], caso["setor"]
        for item in p["itens"]:
            ini, fim = item["intervalo"].split(" as ")
            assert ini < fim, (caso["setor"], item["equipamento"])
        assert "Negócio Teste" in caso["mensagem"], caso["setor"]
