"""
Deslocamento de rotina: a mesma casa, em dias que não são iguais.

O Monte Carlo do D² sorteia **se** um equipamento é usado e **quanto tempo**,
mas não sorteia **quando a rotina acontece**. Toda simulação de todo dia coloca
o almoço na mesma janela, o jantar na mesma janela e a hora de dormir na mesma
hora. Uma casa não é assim: almoça-se ao meio-dia e às duas, janta-se às sete e
às dez, e de vez em quando alguém vira a noite trabalhando.

**O que isso corrige, e o que não corrige.** Não corrige energia: deslocar uma
janela no tempo não muda quanto o equipamento consome no dia. Corrige a
**coincidência**, que é outra coisa e vale dinheiro. Com janelas fixas, os
equipamentos de um mesmo cômodo começam sempre no mesmo minuto do relógio em
todas as simulações, e o perfil médio do conjunto sai mais pontudo do que
qualquer casa real — o cume das 20 h vira uma agulha em vez de uma corcova. Quem
lê esse perfil dimensiona transformador, cabo e inversor para uma sincronia que
não existe.

**A correlação é o ponto, e é onde é fácil errar.** Sortear um deslocamento
independente por equipamento seria pior que não sortear nada: destruiria a
coincidência que de fato existe, porque a família janta junta. O que se desloca
é a **rotina**, não o aparelho. Cada dia sorteia um atraso do domicílio e um
ruído por âncora; todos os equipamentos ancorados no jantar andam juntos, e um
jantar tarde empurra a hora de dormir — mas dormir tarde não puxa o jantar de
volta, porque a causalidade tem um sentido só.

**O regime de madrugada** é um dia inteiro de outra natureza, não uma cauda da
distribuição normal: quem vira a noite não janta 40 minutos mais tarde, dorme
quatro horas mais tarde e mantém escritório e iluminação acesos. Ele entra como
uma mistura — uma fração pequena dos dias sorteia esse regime — porque é assim
que ele aparece numa série real de medição.

O que **não** se desloca: carga contínua. Uma geladeira não sabe que dia é hoje,
e um alarme também não. Deslocá-las seria ruído sem significado físico.
"""
from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .cenario import Cenario

__all__ = [
    "ANCORAS",
    "Ancora",
    "PerfilDeRotina",
    "ROTINA_PADRAO",
    "ROTINA_ALTO_PADRAO",
    "Rotina",
    "ancora_da_janela",
    "deslocar",
    "simular_com_rotina",
    "sortear_rotinas",
]

_MIN_POR_DIA = 1440


# ----------------------------------------------------------------------------
# As âncoras
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Ancora:
    """
    Um momento da rotina, e quanto ele varia de um dia para o outro.

    ``faixa`` é a banda do relógio que a âncora governa, em horas. Uma janela é
    atribuída à âncora cujo **meio** cai na banda: é o meio que diz de que
    momento do dia a janela fala, e não o início, que pode ter sido declarado
    com folga pelo vistoriador.

    ``sigma_min`` é o desvio-padrão do deslocamento, em minutos. Os valores
    padrão vêm do que se observa em pesquisa de uso de tempo doméstico: a hora
    de acordar de quem trabalha é a mais regular da casa, e a de dormir é a mais
    solta, porque nada a força.
    """

    nome: str
    faixa: tuple[float, float]
    sigma_min: float
    #: Quanto do atraso do jantar chega até aqui. Um jantar tarde empurra a hora
    #: de dormir; dormir tarde não puxa o jantar de volta.
    arrasto_do_jantar: float = 0.0
    #: Limite do deslocamento, em minutos, para os dois lados. Sem ele a cauda
    #: da normal produz um jantar às 3 da manhã uma vez a cada mil dias.
    limite_min: float = 150.0


#: As âncoras de uma residência, na ordem do dia.
#:
#: Cinco, e não uma por refeição: o que desloca a carga é o momento do dia, e
#: café, almoço e jantar são momentos com regularidades diferentes. A manhã é
#: presa pelo horário de trabalho e varia pouco; a noite não é presa por nada.
ANCORAS: dict[str, Ancora] = {
    "manha": Ancora("manha", (4.5, 10.0), sigma_min=35.0, limite_min=105.0),
    "almoco": Ancora("almoco", (10.0, 15.5), sigma_min=50.0, limite_min=150.0),
    "tarde": Ancora("tarde", (15.5, 18.0), sigma_min=45.0, limite_min=135.0),
    "jantar": Ancora("jantar", (18.0, 22.0), sigma_min=55.0, limite_min=165.0),
    "noite": Ancora("noite", (22.0, 28.5), sigma_min=70.0,
                    arrasto_do_jantar=0.6, limite_min=210.0),
}


def ancora_da_janela(inicio_h: float, fim_h: float) -> str | None:
    """
    A qual momento do dia esta janela pertence.

    Devolve ``None`` para o que não tem momento: carga contínua, e qualquer
    janela larga o bastante para atravessar o dia inteiro. Deslocar uma
    geladeira não significa nada — ela não sabe que dia é hoje.
    """
    duracao = (fim_h - inicio_h) % 24.0 or 24.0
    if duracao >= 12.0:
        return None

    meio = (inicio_h + duracao / 2.0) % 24.0
    for ancora in ANCORAS.values():
        comeco, fim = ancora.faixa
        # A faixa da noite passa das 24 h de propósito: 23:30 e 01:00 são o
        # mesmo momento da rotina, e parti-la no meio da noite separaria o que
        # o morador vive como uma coisa só.
        if fim > 24.0:
            if meio >= comeco or meio < fim - 24.0:
                return ancora.nome
        elif comeco <= meio < fim:
            return ancora.nome
    return None


# ----------------------------------------------------------------------------
# O perfil e o sorteio
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class PerfilDeRotina:
    """
    Quanto a rotina de uma casa varia — e com que frequência ela vira do avesso.

    ``sigma_comum_min`` é o atraso do **domicílio** no dia: o componente que
    todas as âncoras compartilham. É ele que faz um dia atrasado ser atrasado em
    tudo, e é a diferença entre modelar uma casa e modelar aparelhos avulsos.

    ``prob_madrugada`` é a fração de dias em que alguém vira a noite. Não é
    cauda da normal: é regime, e por isso entra como mistura.
    """

    sigma_comum_min: float = 30.0
    prob_madrugada: float = 0.06
    #: Deslocamento da hora de dormir num dia de madrugada, em minutos.
    madrugada_min: tuple[float, float] = (120.0, 260.0)
    #: Multiplica o sigma de todas as âncoras. Uma casa com criança pequena e
    #: horário rígido varia menos; uma casa de dois adultos sem filhos varia
    #: mais, e um imóvel de alto padrão com serviço varia mais ainda.
    dispersao: float = 1.0

    def sigma(self, ancora: Ancora) -> float:
        return ancora.sigma_min * max(0.0, self.dispersao)


#: A casa de rotina comum: horário de trabalho, jantar em família, sono regular.
ROTINA_PADRAO = PerfilDeRotina()

#: Alto padrão: rotina mais solta e madrugada mais frequente.
#:
#: Não é estereótipo, é o que as janelas do levantamento mostram — home office,
#: jantar que começa tarde, recepção no fim de semana e ninguém preso ao ônibus
#: das seis. A dispersão maior não muda a energia; muda a coincidência, que é
#: justamente o que se dimensiona errado quando se supõe rotina de relógio.
ROTINA_ALTO_PADRAO = PerfilDeRotina(
    sigma_comum_min=42.0, prob_madrugada=0.10, dispersao=1.25,
)


@dataclass(frozen=True)
class Rotina:
    """O deslocamento de cada âncora num dia, em minutos."""

    deslocamentos: Mapping[str, float]
    madrugada: bool = False

    def de(self, ancora: str | None) -> float:
        if ancora is None:
            return 0.0
        return float(self.deslocamentos.get(ancora, 0.0))

    def as_dict(self) -> dict[str, Any]:
        return {"madrugada": self.madrugada,
                **{k: round(v, 1) for k, v in self.deslocamentos.items()}}


def sortear_rotinas(
    quantidade: int,
    perfil: PerfilDeRotina = ROTINA_PADRAO,
    semente: int | None = 42,
) -> list[Rotina]:
    """
    Sorteia ``quantidade`` dias de rotina.

    Cada dia é um atraso do domicílio mais um ruído por âncora, e a hora de
    dormir herda parte do atraso do jantar. Uma fração dos dias sai em regime de
    madrugada, com a noite empurrada horas adiante em vez de minutos.
    """
    rng = np.random.default_rng(semente)
    rotinas: list[Rotina] = []
    for _ in range(max(1, int(quantidade))):
        comum = float(rng.normal(0.0, perfil.sigma_comum_min))
        bruto: dict[str, float] = {}
        for nome, ancora in ANCORAS.items():
            ruido = float(rng.normal(0.0, perfil.sigma(ancora)))
            bruto[nome] = comum + ruido

        # A causalidade tem um sentido só: o jantar empurra a noite.
        for nome, ancora in ANCORAS.items():
            if ancora.arrasto_do_jantar:
                bruto[nome] += ancora.arrasto_do_jantar * bruto.get("jantar", 0.0)

        madrugada = bool(rng.random() < perfil.prob_madrugada)
        if madrugada:
            baixo, alto = perfil.madrugada_min
            bruto["noite"] = bruto.get("noite", 0.0) + float(rng.uniform(baixo, alto))

        deslocamentos = {}
        for nome, ancora in ANCORAS.items():
            limite = ancora.limite_min * (3.0 if (madrugada and nome == "noite") else 1.0)
            deslocamentos[nome] = float(np.clip(bruto[nome], -limite, limite))
        rotinas.append(Rotina(deslocamentos=deslocamentos, madrugada=madrugada))
    return rotinas


# ----------------------------------------------------------------------------
# Aplicar o deslocamento
# ----------------------------------------------------------------------------
_JANELA = re.compile(r"(\d{1,2}):(\d{2})\s*(?:as|às|a|-)\s*(\d{1,2}):(\d{2})", re.I)


def _hhmm(minutos: float) -> str:
    m = int(round(minutos)) % _MIN_POR_DIA
    return f"{m // 60:02d}:{m % 60:02d}"


def _deslocar_texto(intervalo: str, rotina: Rotina) -> str:
    """
    Desloca cada janela do texto pela âncora a que ela pertence.

    Um texto com duas janelas — almoço e jantar — recebe **dois** deslocamentos
    diferentes, que é o comportamento certo: são dois momentos do dia, e eles
    não atrasam juntos por acaso, atrasam juntos pelo componente comum.

    A duração é preservada por construção. O resultado pode atravessar a
    meia-noite, e isso é físico: uma rotina empurrada para depois das 24 h está
    na madrugada, e o motor já sabe tratar janela circular.
    """
    achados = list(_JANELA.finditer(intervalo or ""))
    if not achados:
        return intervalo

    partes: list[str] = []
    for achado in achados:
        h1, m1, h2, m2 = (int(g) for g in achado.groups())
        inicio, fim = h1 * 60 + m1, h2 * 60 + m2
        duracao = (fim - inicio) % _MIN_POR_DIA or _MIN_POR_DIA
        ancora = ancora_da_janela(inicio / 60.0, fim / 60.0)
        if ancora is None:
            partes.append(f"{_hhmm(inicio)} as {_hhmm(fim)}")
            continue
        novo = inicio + rotina.de(ancora)
        partes.append(f"{_hhmm(novo)} as {_hhmm(novo + duracao)}")
    return " e ".join(partes)


def deslocar(cenario: Cenario, rotina: Rotina) -> Cenario:
    """
    O cenário como ele fica **naquele dia**.

    Não altera o original. Carga contínua passa intacta, e a janela que não se
    encaixa em nenhuma âncora também: deslocar o que não tem hora seria inventar
    variação onde não há.
    """
    comodos: dict[str, pd.DataFrame] = {}
    for nome, tabela in cenario.comodos.items():
        nova = tabela.copy(deep=True)
        if "intervalo" in nova.columns:
            for indice, linha in nova.iterrows():
                texto = linha.get("intervalo")
                if isinstance(texto, str) and texto.strip():
                    nova.at[indice, "intervalo"] = _deslocar_texto(texto, rotina)
        comodos[nome] = nova

    novo = copy.copy(cenario)
    novo.comodos = comodos
    novo.instancias = dict(cenario.instancias)
    novo.essenciais = list(cenario.essenciais)
    return novo


# ----------------------------------------------------------------------------
# A simulação
# ----------------------------------------------------------------------------
def simular_com_rotina(
    cenario: Cenario,
    num_simulacoes: int = 300,
    perfil: PerfilDeRotina = ROTINA_PADRAO,
    ajustes_sazonais: Mapping[str, dict] | None = None,
    sorteios: int = 24,
    semente: int | None = 42,
    apenas_essenciais: bool = False,
    **kwargs: Any,
):
    """
    Monte Carlo com a rotina variando de um dia para o outro.

    ``sorteios`` é quantas rotinas distintas entram na mistura; as simulações
    são repartidas entre elas. Poucos sorteios deixam degraus visíveis no perfil
    médio; muitos custam tempo sem acrescentar informação. Duas dúzias já
    reproduzem uma distribuição contínua de horários.

    Devolve um :class:`~aurum.demanda.ensemble.EnsembleCarga` com o mesmo
    contrato do simulador de sempre — quem consome não precisa saber que a
    rotina variou.
    """
    from .ensemble import EnsembleCarga, simular_ensemble

    sorteios = max(1, int(sorteios))
    total = max(1, int(num_simulacoes))
    rotinas = sortear_rotinas(sorteios, perfil, semente)

    # Reparte as simulações entre os sorteios, distribuindo o resto para que a
    # soma seja exatamente `num_simulacoes` — um ensemble com menos dias que o
    # pedido faria a estatística mudar sem ninguém notar.
    base = total // sorteios
    sobra = total - base * sorteios
    fatias = [base + (1 if i < sobra else 0) for i in range(sorteios)]

    parciais = []
    for i, (rotina, fatia) in enumerate(zip(rotinas, fatias)):
        if fatia <= 0:
            continue
        do_dia = deslocar(cenario, rotina)
        do_dia.criticidades_essenciais = getattr(
            cenario, "criticidades_essenciais", ())
        parciais.append(simular_ensemble(
            do_dia.para_comodos(apenas_essenciais),
            do_dia.instancias_de(apenas_essenciais),
            fatia, ajustes_sazonais,
            semente=None if semente is None else semente + 1000 * i,
            **kwargs,
        ))

    if not parciais:
        raise ValueError("nenhuma simulação foi produzida")

    estacoes = parciais[0].estacoes
    perfis = {
        estacao: np.concatenate([p.perfis_w[estacao] for p in parciais], axis=0)
        for estacao in estacoes
    }
    metadados = dict(parciais[0].metadados)
    metadados.update({
        "num_simulacoes": total,
        "rotina_sorteios": sorteios,
        "rotina_perfil": {
            "sigma_comum_min": perfil.sigma_comum_min,
            "prob_madrugada": perfil.prob_madrugada,
            "dispersao": perfil.dispersao,
        },
        "rotina_madrugadas": sum(1 for r in rotinas if r.madrugada),
    })
    return EnsembleCarga(perfis_w=perfis, passo_min=parciais[0].passo_min,
                         metadados=metadados)


def resumo_do_deslocamento(
    cenario: Cenario, rotinas: Sequence[Rotina],
) -> pd.DataFrame:
    """
    Quantas janelas cada âncora governa, para o relatório mostrar o que mexeu.

    Sem isto, o deslocamento é uma caixa-preta que muda números — e uma caixa
    dessas num estudo de engenharia vale menos que não ter o modelo.
    """
    contagem: dict[str, int] = {nome: 0 for nome in ANCORAS}
    contagem["sem âncora"] = 0
    for tabela in cenario.comodos.values():
        for texto in tabela.get("intervalo", pd.Series(dtype=str)):
            if not isinstance(texto, str):
                continue
            for achado in _JANELA.finditer(texto):
                h1, m1, h2, m2 = (int(g) for g in achado.groups())
                nome = ancora_da_janela(h1 + m1 / 60.0, h2 + m2 / 60.0)
                contagem[nome or "sem âncora"] += 1

    linhas = []
    for nome, quantas in contagem.items():
        if nome == "sem âncora":
            linhas.append({"ancora": nome, "janelas": quantas,
                           "desvio_medio_min": 0.0, "maior_atraso_min": 0.0})
            continue
        valores = [r.de(nome) for r in rotinas]
        linhas.append({
            "ancora": nome,
            "janelas": quantas,
            "desvio_medio_min": float(np.std(valores)) if valores else 0.0,
            "maior_atraso_min": float(np.max(valores)) if valores else 0.0,
        })
    return pd.DataFrame(linhas)
