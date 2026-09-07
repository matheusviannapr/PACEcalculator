"""
Perfis de ocupação: a mesma casa cheia e a mesma casa quase vazia.

Uma residência não tem *uma* curva de carga. Tem pelo menos duas, e elas não
se parecem:

* **Casa cheia.** Todo mundo em casa e acordado. Qualquer equipamento pode ser
  usado a qualquer hora do dia — não há razão para o notebook ser das 8 às 18
  e a TV só à noite quando ninguém saiu. A exceção é a cozinha, que **mantém
  os picos de almoço e jantar**: refeição tem hora, e é ela que faz os dois
  cumes da curva.
* **Casa quase vazia.** A família sai para trabalhar e estudar, mas a casa não
  fica zerada: sobra uma pessoa, ou alguém volta no meio do dia, e a
  refrigeração não sabe que dia é hoje. O consumo diurno **cai**, e não some.

Modelar a casa vazia como um zero entre 8 h e 18 h foi o primeiro erro deste
módulo — e é o erro que faz o dimensionamento parecer mais folgado do que é.
O que muda no meio do dia é a **intensidade**, não a existência.

O que dimensiona o equipamento é a casa cheia; o dia de semana entra na conta
de energia, porque acontece cinco vezes em sete.

**O que não é tocado:** equipamento de janela contínua. Geladeira, freezer e
roteador não sabem que dia é hoje, e mexer neles seria inventar consumo.
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd

from .cenario import Cenario

__all__ = [
    "ESTUDOS",
    "PERFIS",
    "PerfilOcupacao",
    "aplicar",
    "comparar",
]

#: Uma janela que cobre este tanto do dia é tratada como contínua e fica
#: intocada. 23 h em vez de 24 para pegar o ``00:00 as 23:59`` da vistoria.
_HORAS_CONTINUO = 23.0

#: Quanto de uma janela precisa cair dentro de uma refeição para que ela seja
#: tratada como carga de refeição. Metade: um equipamento que liga das 18 h às
#: 22 h é do jantar; um que liga das 7 h às 23 h não é de refeição nenhuma.
_FRACAO_REFEICAO = 0.5

_FAIXA = re.compile(r"(\d{1,2}):(\d{2})\s*as\s*(\d{1,2}):(\d{2})", re.I)

#: Onde a comida é feita. A cozinha guarda os picos; o resto da casa não.
_AMBIENTES_DE_REFEICAO = ("cozinha", "gourmet", "churrasq", "copa", "restaurante")


@dataclass(frozen=True)
class PerfilOcupacao:
    """
    Quem está em casa, e com que intensidade cada hora é usada.

    Duas alavancas, e elas fazem coisas diferentes:

    ``espalhar`` decide se as janelas levantadas são **alargadas** para a
    janela em que a casa está acordada. Com a casa cheia, sim: qualquer
    equipamento pode ser usado a qualquer hora, e manter o notebook das 8 às 18
    seria repetir o horário de escritório num dia em que ninguém foi trabalhar.

    ``fator_diurno`` decide o que acontece com o que é usado **durante o
    expediente**. Com a casa quase vazia ele reduz a probabilidade — sobra uma
    pessoa, e não a família inteira — em vez de zerar a janela. Zerar produzia
    um vale artificial que nenhuma casa tem.
    """

    id: str
    nome: str
    descricao: str
    #: A faixa em que a casa está acordada.
    janela_acordado: tuple[float, float]
    #: Almoço e jantar. Equipamento de cozinha nestas faixas mantém a janela:
    #: é o que produz os dois cumes da curva.
    janelas_refeicao: tuple[tuple[float, float], ...] = ()
    #: Alargar as janelas de presença para a janela acordada.
    espalhar: bool = False
    #: Desdobrar a carga de cozinha em almoço **e** jantar.
    #:
    #: A vistoria de campo costuma levantar só o jantar — é a refeição que o
    #: morador lembra de citar, e o formulário aceita uma janela por item. Numa
    #: casa cheia almoça-se em casa, e o forno, o micro-ondas e o lava-louças
    #: rodam duas vezes. Sem este desdobramento a curva tem um cume só, e o
    #: pico do meio-dia — que é quando o sol está no máximo — não aparece.
    desdobrar_refeicoes: bool = False
    #: Multiplica a probabilidade de tudo que é movido a presença.
    fator_probabilidade: float = 1.0
    #: Teto do ganho de probabilidade concedido a quem teve a janela alargada.
    #:
    #: Alargar a janela sem mexer na probabilidade **conserva** a energia do
    #: dia: o sorteio de uso vem antes do sorteio de horário, e a duração não
    #: depende da janela. Uma janela três vezes maior devolve, portanto, uma
    #: curva média três vezes mais rasa -- e a casa cheia terminava a noite
    #: abaixo da casa quase vazia, que é o contrário do que se quer mostrar.
    #: Compensar pelo alargamento inteiro seria o outro exagero: uma família em
    #: casa usa a TV mais tempo, não três vezes mais tempo. O teto de 2,0 é o
    #: meio-termo, e é a única constante deste módulo que é arbitrada.
    alargamento_max: float = 2.0
    #: Multiplica a probabilidade do que é usado durante o expediente.
    fator_diurno: float = 1.0
    #: Que fração da madrugada o item ligado na hora de dormir atravessa.
    #:
    #: 0,6 numa casa cheia — os quartos estão todos ocupados e o ar roda até
    #: quase o amanhecer; 0,35 numa casa quase vazia, onde metade dos quartos
    #: está fechada e o que roda é menos e por menos tempo. Zero
    #: desliga a regra. O primeiro esboço tinha o parâmetro sem uso e uma
    #: janela de tamanho fixo, e a simulação saía com 25% da energia na
    #: madrugada contra 12% da referência — o erro oposto ao que a regra veio
    #: corrigir.
    #:
    #: O formulário da vistoria pede uma janela por equipamento, o morador
    #: responde "das 18 às 23", e o ar-condicionado do quarto some às 23:00 em
    #: ponto — o que nenhum ar-condicionado faz. A simulação saía com 1% da
    #: energia na madrugada contra 12% da curva residencial de referência.
    arrasto_noturno: float = 0.0
    #: A que hora a casa acorda — o fim da madrugada.
    hora_de_acordar: float = 6.0
    #: Duração mínima, em horas, para um item atravessar a noite. Climatização
    #: e refrigeração atravessam; lâmpada e televisor, não — e sem este filtro
    #: a luz da sala passaria a acender às 3 h da manhã.
    duracao_minima_para_arrasto_h: float = 1.0
    #: Multiplica a probabilidade das cargas de refeição, separado do resto.
    #:
    #: Almoço e jantar acontecem em qualquer dia — mudam de tamanho, não de
    #: existência. Numa casa quase vazia alguém cozinha para um, e o forno roda
    #: menos; tratar a refeição com o mesmo fator do resto da casa apagaria o
    #: almoço do dia de semana, que é justamente onde o sol está no máximo.
    fator_refeicao: float = 1.0
    #: A faixa considerada expediente, para efeito do fator diurno.
    janela_expediente: tuple[float, float] = (8.0, 18.0)
    probabilidade_max: float = 0.98
    #: Quantos dias da semana este perfil representa — pesa a conta de energia.
    dias_por_semana: int = 5

    @property
    def horas_acordado(self) -> float:
        return self.janela_acordado[1] - self.janela_acordado[0]

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "nome": self.nome,
            "acordado": f"{_hhmm(self.janela_acordado[0])}–{_hhmm(self.janela_acordado[1])}",
            "refeicoes": [f"{_hhmm(a)}–{_hhmm(b)}" for a, b in self.janelas_refeicao],
            "espalhar": self.espalhar,
            "fator_probabilidade": self.fator_probabilidade,
            "fator_diurno": self.fator_diurno,
            "dias_por_semana": self.dias_por_semana,
        }


def _hhmm(hora: float) -> str:
    h = int(hora)
    m = int(round((hora - h) * 60))
    if m == 60:
        h, m = h + 1, 0
    return f"{min(h, 23):02d}:{m:02d}"


#: As refeições de uma residência brasileira. O café da manhã não entra: ele
#: é curto, de baixa potência, e já está na janela da manhã que a vistoria
#: levanta para o boiler e o chuveiro.
#:
#: As duas têm a **mesma largura**, e isso não é estética. A energia de um
#: item dinâmico não depende da largura da janela — só o instante depende —,
#: então uma janela mais estreita concentra a mesma carga num cume mais alto.
#: Com o almoço em 2,5 h e o jantar em 3,5 h, o cume do meio-dia passava o da
#: noite, e a casa aparecia com o pico às 12 h contra as 20 h da curva
#: residencial de referência. Numa casa brasileira o pico é o jantar.
_ALMOCO = (11.0, 14.5)
_JANTAR = (18.5, 22.0)

PERFIS: dict[str, PerfilOcupacao] = {
    "levantado": PerfilOcupacao(
        id="levantado",
        nome="Como a vistoria levantou",
        descricao=(
            "As janelas de uso exatamente como o vistoriador as anotou, sem "
            "nenhuma transformação. É a leitura mais fiel ao campo e a mais "
            "concentrada à noite, porque é de noite que o morador lembra de "
            "usar as coisas quando responde ao formulário."
        ),
        # Nenhuma alavanca acionada: janela intacta, probabilidade intacta.
        # A janela acordada existe só para a tabela do relatório ter o que
        # mostrar na coluna; ela não é aplicada a nada.
        janela_acordado=(6.0, 23.5),
        espalhar=False,
        fator_probabilidade=1.0,
        fator_diurno=1.0,
        dias_por_semana=7,
    ),
    "casa_cheia": PerfilOcupacao(
        id="casa_cheia",
        nome="Casa cheia",
        descricao=(
            "Todo mundo em casa e acordado. Qualquer equipamento pode ser usado "
            "a qualquer hora, com os picos de almoço e jantar na cozinha. É este "
            "perfil que dimensiona o sistema, porque é nele que o pior caso "
            "acontece."
        ),
        janela_acordado=(7.0, 23.5),
        janelas_refeicao=(_ALMOCO, _JANTAR),
        espalhar=True,
        desdobrar_refeicoes=True,
        arrasto_noturno=0.6,
        fator_probabilidade=1.35,
        dias_por_semana=2,
    ),
    "casa_quase_vazia": PerfilOcupacao(
        id="casa_quase_vazia",
        nome="Casa quase vazia",
        descricao=(
            "A família sai para trabalhar e estudar, mas a casa não fica zerada: "
            "sobra alguém durante o dia. O consumo diurno cai — não some — e a "
            "manhã e a noite continuam cheias."
        ),
        janela_acordado=(6.0, 23.5),
        janelas_refeicao=(_ALMOCO, _JANTAR),
        # As janelas levantadas já descrevem um dia de semana: não se alarga.
        espalhar=False,
        # Mas almoça-se e janta-se em qualquer dia. A vistoria costuma anotar
        # só o jantar, e sem o desdobramento o dia de semana sai com um vale no
        # meio-dia que nenhuma casa tem — justamente na hora em que o sol está
        # no máximo, que é onde o autoconsumo se decide.
        desdobrar_refeicoes=True,
        arrasto_noturno=0.35,
        fator_probabilidade=1.0,
        # Alguém cozinha para um, e não para a família: a refeição acontece,
        # menor. Mas **menos reduzida que o uso geral do dia**: quem fica em
        # casa deixa de usar metade da casa e não deixa de almoçar. Um fator
        # de refeição abaixo do diurno — foi o primeiro valor tentado — dizia
        # o contrário, e nenhuma casa faz isso.
        fator_refeicao=0.72,
        # Uma pessoa em vez da família. O primeiro valor tentado, 0,45,
        # empurrava 25 pontos percentuais de energia a mais para a noite do
        # que a curva residencial de referência tem: uma casa quase vazia não
        # é uma casa desligada, e o que sobra durante o dia — refrigeração,
        # rede, uma pessoa em casa — é mais que isso.
        fator_diurno=0.62,
        dias_por_semana=5,
    ),
}

#: O perfil que dimensiona o equipamento. É o pior caso, e é o que a proposta
#: promete cumprir.
PERFIL_DIMENSIONANTE = "casa_cheia"

#: Os três níveis de uso que o estudo entrega, sempre os três.
#:
#: **Ninguém escolhe entre eles antes do estudo.** Escolher antes é o que
#: transforma uma premissa em conclusão: quem decide "esta casa é de pouco
#: uso" no começo recebe um sistema dimensionado para isso e nenhuma forma de
#: saber o que perdeu. Os três saem juntos, com o sistema de cada um, e a
#: escolha passa a ser uma conversa com números na mesa.
#:
#: O que **não** é um cenário: a janela que a vistoria levantou. Ela é
#: recortada pelo que o morador lembra de responder — quase sempre a noite,
#: quase nunca o almoço — e por isso é o insumo dos três, e não um deles. O
#: perfil ``levantado`` continua existindo para a seção de ocupação mostrar a
#: transformação lado a lado, que é o que a torna auditável.
ESTUDOS: dict[str, dict[str, Any]] = {
    "pouco_uso": {
        "nome": "Pouco uso",
        "perfis": {"casa_quase_vazia": 7},
        "para_que": (
            "A casa esvazia de segunda a domingo: casal que trabalha fora e "
            "viaja no fim de semana, ou segunda residência ocupada de vez em "
            "quando. É o piso do consumo, e o sistema mais enxuto que faz "
            "sentido propor."
        ),
    },
    "uso_comum": {
        "nome": "Uso comum",
        "perfis": {"casa_quase_vazia": 5, "casa_cheia": 2},
        "para_que": (
            "Cinco dias de casa quase vazia e dois cheios — a rotina de quem "
            "trabalha fora. É onde a maioria das famílias cai — o que não faz "
            "dele a resposta: quem conhece a casa é o cliente."
        ),
    },
    "muito_uso": {
        "nome": "Muito uso",
        "perfis": {"casa_cheia": 7},
        "para_que": (
            "Todo mundo em casa todos os dias: férias, home office da família "
            "inteira, casa de veraneio na temporada. É o teto, e é o cenário "
            "que a proposta tem de cumprir se o cliente disser que é o dele."
        ),
    },
}

#: A ordem em que os três são apresentados — do mais leve ao mais pesado.
ORDEM_DOS_ESTUDOS = ("pouco_uso", "uso_comum", "muito_uso")

#: O cenário do meio. Não é uma recomendação do estudo — o estudo apresenta os
#: três e não escolhe —, é só a posição dele na ordem, útil para o texto do
#: relatório dizer qual fica entre os outros dois.
ESTUDO_DO_MEIO = "uso_comum"


# ----------------------------------------------------------------------------
# Transformação
# ----------------------------------------------------------------------------
def _parse(intervalo: str) -> tuple[float, float] | None:
    achado = _FAIXA.search(str(intervalo or ""))
    if not achado:
        return None
    h1, m1, h2, m2 = (int(g) for g in achado.groups())
    return h1 + m1 / 60.0, h2 + m2 / 60.0


def _duracao(faixa: tuple[float, float]) -> float:
    inicio, fim = faixa
    return (fim - inicio) if fim >= inicio else (24.0 - inicio + fim)


def _e_continuo(faixa: tuple[float, float]) -> bool:
    return _duracao(faixa) >= _HORAS_CONTINUO


def _sobreposicao(a: tuple[float, float], b: tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def _e_de_refeicao(faixa: tuple[float, float], comodo: str,
                   perfil: PerfilOcupacao) -> bool:
    """
    A janela é de refeição, e por isso preservada?

    Duas condições, e as duas precisam valer: o equipamento está num ambiente
    onde se faz comida, e a janela dele cai principalmente dentro de almoço ou
    jantar. Sem a primeira, a TV da sala ligada às 20 h viraria "carga de
    jantar"; sem a segunda, a geladeira da cozinha também viraria.
    """
    nome = str(comodo or "").lower()
    if not any(marca in nome for marca in _AMBIENTES_DE_REFEICAO):
        return False
    duracao = _duracao(faixa) or 1.0
    return any(
        _sobreposicao(faixa, refeicao) / duracao >= _FRACAO_REFEICAO
        for refeicao in perfil.janelas_refeicao
    )


def _fracao_no_expediente(faixa: tuple[float, float], perfil: PerfilOcupacao) -> float:
    """Quanto da janela cai no horário em que a casa esvazia."""
    duracao = _duracao(faixa) or 1.0
    return _sobreposicao(faixa, perfil.janela_expediente) / duracao


def _atravessa_a_noite(
    faixa: tuple[float, float], linha: "pd.Series", perfil: PerfilOcupacao,
) -> bool:
    """
    O item estava ligado na hora de dormir, e é do tipo que não desliga?

    Dois filtros, e os dois são necessários. O primeiro pergunta se a janela
    termina junto com a hora de dormir — se termina, o item estava ligado ali.
    O segundo pergunta se ele tem duração longa por utilização: climatização e
    refrigeração atravessam a noite, lâmpada e televisor não. Sem o segundo, a
    luz da sala passaria a acender às 3 h da manhã.
    """
    if perfil.arrasto_noturno <= 0:
        return False
    fim_da_noite = perfil.janela_acordado[1]
    # Termina na última hora e meia da vigília — foi dormir com aquilo ligado.
    if not (fim_da_noite - 1.5 <= faixa[1] <= fim_da_noite + 0.5):
        return False
    duracao = linha.get("duracao_max")
    if duracao is None or pd.isna(duracao):
        duracao = linha.get("duracao_min")
    if duracao is None or pd.isna(duracao):
        return False
    return float(duracao) >= perfil.duracao_minima_para_arrasto_h


def _e_de_formulario(faixa: tuple[float, float], perfil: PerfilOcupacao) -> bool:
    """
    A janela parece o padrão do catálogo, ou parece observação de campo?

    Uma janela que cai principalmente dentro do expediente é suspeita numa
    residência: ninguém opera nada por horário comercial dentro de casa, e
    08:00-18:00 é o padrão que o catálogo traz e que o formulário aceita sem
    discussão. Essa é a janela que a casa cheia tem o direito de reescrever.

    Já uma janela ancorada na noite ou na manhã é o que o morador observou, e
    o morador estava lá. Reescrevê-la também — que era o comportamento antes —
    apagava a concentração natural do jantar e levava o pico da casa para as
    12 h, contra as 20 h da curva residencial de referência. Espalhar tudo
    parecia mais isento e era menos fiel.
    """
    return _fracao_no_expediente(faixa, perfil) >= 0.6


def _janela_das_refeicoes(faixa: tuple[float, float], perfil: PerfilOcupacao) -> str:
    """
    A janela que cobre almoço **e** jantar, no formato que o núcleo entende.

    O D² aceita janelas múltiplas separadas por ``" e "`` -- é o que permite
    dizer "das 11:30 às 14:00 e das 18:30 às 22:00" numa linha só, em vez de
    esticar uma janela das 11:30 às 22:00 e deixar o micro-ondas rodar às 16 h.
    A distinção importa: esticar produziria um platô, e o que a casa tem são
    dois cumes.

    As duas refeições usam a janela canônica, e não a levantada. Preservar a
    levantada era o primeiro instinto -- é o dado de campo -- mas as janelas
    saem com larguras diferentes: um jantar anotado das 18:00 às 22:00 tem 4 h
    e o almoço inserido tem 2,5 h. Como a energia diária de um item dinâmico
    não depende da largura da janela, a mesma carga saía diluída na janela
    larga e concentrada na estreita, e a curva ficava com o cume do almoço
    maior que o do jantar. Numa casa é o contrário. A janela levantada continua
    no anexo de cargas, que é onde o dado de campo pertence.
    """
    partes = sorted(perfil.janelas_refeicao)
    return " e ".join(f"{_hhmm(a)} as {_hhmm(b)}" for a, b in partes)


def aplicar(cenario: Cenario, perfil: PerfilOcupacao | str) -> Cenario:
    """
    Devolve o cenário reescrito para um perfil de ocupação.

    Não altera o original: o cenário levantado em campo é o dado, e os perfis
    são leituras dele. Poder voltar ao levantado é o que permite mostrar a
    transformação no relatório em vez de pedir confiança.
    """
    if isinstance(perfil, str):
        if perfil not in PERFIS:
            raise ValueError(f"perfil desconhecido: {perfil!r}. Conhecidos: {sorted(PERFIS)}")
        perfil = PERFIS[perfil]

    comodos: dict[str, pd.DataFrame] = {}
    for nome, tabela in cenario.comodos.items():
        nova = tabela.copy(deep=True)
        comodo_do_item = str(nova.get("comodo", pd.Series([nome] * len(nova))).iloc[0]
                             if "comodo" in nova.columns and len(nova) else nome)
        for indice, linha in nova.iterrows():
            faixa = _parse(linha.get("intervalo"))
            if faixa is None or _e_continuo(faixa):
                continue  # contínuo não sabe que dia é hoje

            comodo = str(linha.get("comodo") or comodo_do_item or nome)
            de_refeicao = _e_de_refeicao(faixa, comodo, perfil)

            # 1. A janela. Refeição tem hora e é preservada; o resto se
            #    espalha pela janela acordada quando a casa está cheia — mas
            #    só o que parece padrão de formulário. Ver `_e_de_formulario`.
            alargamento = 1.0
            if perfil.espalhar and not de_refeicao and _e_de_formulario(faixa, perfil):
                nova.at[indice, "intervalo"] = (
                    f"{_hhmm(perfil.janela_acordado[0])} as "
                    f"{_hhmm(perfil.janela_acordado[1])}"
                )
                alargamento = min(
                    perfil.horas_acordado / (_duracao(faixa) or 1.0),
                    perfil.alargamento_max,
                )

            # 2. A refeição que falta. Uma janela de jantar numa casa cheia
            #    também é almoço: o equipamento passa a cobrir as duas.
            if de_refeicao and perfil.desdobrar_refeicoes:
                nova.at[indice, "intervalo"] = _janela_das_refeicoes(faixa, perfil)

            # 2b. A madrugada de quem foi dormir com o ar ligado.
            elif _atravessa_a_noite(faixa, linha, perfil):
                ate = perfil.hora_de_acordar * perfil.arrasto_noturno
                atual = str(nova.at[indice, "intervalo"])
                nova.at[indice, "intervalo"] = f"00:00 as {_hhmm(ate)} e {atual}"

            # 3. A intensidade. O fator diurno pesa pela fração da janela que
            #    cai no expediente: um equipamento das 8 às 18 leva o fator
            #    inteiro, um das 18 às 23 não leva nada.
            prob = linha.get("probabilidade")
            if pd.notna(prob):
                fator = perfil.fator_probabilidade * max(alargamento, 1.0)
                if de_refeicao:
                    fator *= perfil.fator_refeicao
                if perfil.fator_diurno != 1.0 and not de_refeicao:
                    dentro = _fracao_no_expediente(faixa, perfil)
                    fator *= 1.0 - dentro * (1.0 - perfil.fator_diurno)
                nova.at[indice, "probabilidade"] = float(
                    min(float(prob) * fator, perfil.probabilidade_max)
                )
        comodos[nome] = nova

    novo = copy.copy(cenario)
    novo.comodos = comodos
    novo.instancias = dict(cenario.instancias)
    novo.essenciais = list(cenario.essenciais)
    novo.nome = f"{cenario.nome} · {perfil.nome}"
    return novo


def comparar(cenario: Cenario, perfis: Iterable[PerfilOcupacao | str] | None = None) -> pd.DataFrame:
    """
    Uma linha por perfil, com o que muda entre eles.

    A potência instalada não muda — é a mesma casa. O que muda é a janela em
    que cada equipamento pode ligar e a chance de ele ligar, e a consequência
    disso só aparece depois de simular. Esta tabela mostra a **entrada** da
    simulação; a saída é o pico e o consumo, e vão no relatório.
    """
    escolhidos = [
        PERFIS[p] if isinstance(p, str) else p
        for p in (perfis if perfis is not None else PERFIS.values())
    ]
    linhas = []
    for perfil in escolhidos:
        ajustado = aplicar(cenario, perfil)
        janelas = probabilidades = 0
        for nome, tabela in ajustado.comodos.items():
            original = cenario.comodos[nome]
            janelas += int((tabela["intervalo"].astype(str)
                            != original["intervalo"].astype(str)).sum())
            probabilidades += int(
                (tabela["probabilidade"].fillna(-1).round(4)
                 != original["probabilidade"].fillna(-1).round(4)).sum()
            )
        linhas.append({
            "perfil": perfil.nome,
            "acordado": f"{_hhmm(perfil.janela_acordado[0])}–"
                        f"{_hhmm(perfil.janela_acordado[1])}",
            "refeicoes": ", ".join(
                f"{_hhmm(a)}–{_hhmm(b)}" for a, b in perfil.janelas_refeicao
            ),
            "uso_diurno": (
                "cheio" if perfil.fator_diurno == 1.0
                else f"{perfil.fator_diurno:.0%} do normal"
            ),
            "dias_por_semana": perfil.dias_por_semana,
            "janelas_alteradas": janelas,
            "probabilidades_alteradas": probabilidades,
            "potencia_instalada_kw": round(ajustado.potencia_instalada_w() / 1000.0, 2),
        })
    return pd.DataFrame(linhas)
