"""
Índice de oportunidade: de 0 a 100, quanto a casa (ou o negócio) tem a ganhar.

Um número só, para o cliente e para a fila do CRM, feito de quatro partes que
o pré-relatório já calculou. Cada parte tem teto, e o teto é a opinião
declarada de quanto ela pesa na decisão de compra:

=====================  =====  ===============================================
Parte                  Teto   O que mede
=====================  =====  ===============================================
Economia               35     quanto da conta o solar tira (fração) e quanto
                              isso vale em reais — R$ 300 de uma conta de
                              R$ 400 e R$ 3.000 de uma de R$ 4.000 não são
                              a mesma oportunidade
Falta de luz           30     frequência × duração das faltas declaradas,
                              pesadas pelo que é crítico (aparelho de saúde,
                              "nem um minuto", mercadoria que estraga)
Consumo sem sol        15     quanto do dia acontece sem sol: é a energia
                              que a bateria desloca, e onde o solar sozinho
                              não chega
Encaixe da solução     20     se a bateria recomendada segura a falta que o
                              cliente descreveu
=====================  =====  ===============================================

**O que o índice não é.** Não é probabilidade de fechar nem retorno do
investimento: é uma régua para comparar leads e para o cliente ver de onde vem
a recomendação. As curvas que convertem respostas em pontos (as tabelas
abaixo) são arbitradas e devem ser recalibradas quando houver vendas fechadas
para comparar — o código diz isso aqui para ninguém tratar o 78 como medição.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

#: Tetos das partes. Somam 100.
TETO_ECONOMIA, TETO_FALTA, TETO_NOITE, TETO_ENCAIXE = 35.0, 30.0, 15.0, 20.0

#: Frequência e duração da falta de luz, em 0..1. A raiz do produto entra nos
#: pontos: "todo mês, uma ou duas horas" é uma dor real e não pode valer metade
#: de "toda semana, meio dia".
PESO_FREQUENCIA = {"rara": 0.15, "algumas": 0.40, "mensal": 0.75, "semanal": 1.0}
PESO_DURACAO = {"minutos": 0.35, "horas": 0.65, "meio_dia": 0.90, "dia": 1.0}
#: Quanto pesa o que fica sem energia: algo que não pode cair, algo que faz
#: falta, ou nada declarado.
PESO_CRITICO = {"grave": 1.0, "critico": 0.75, "nenhum": 0.45}

FAIXAS = ((80, "muito alta"), (60, "alta"), (40, "moderada"), (0, "baixa"))


@dataclass(frozen=True)
class Parte:
    nome: str
    pontos: float
    teto: float
    detalhe: str = ""


@dataclass(frozen=True)
class IndiceOportunidade:
    valor: int
    faixa: str
    partes: tuple[Parte, ...]

    def as_dict(self) -> dict:
        return {"valor": self.valor, "faixa": self.faixa,
                "partes": [{"nome": p.nome, "pontos": round(p.pontos, 1), "teto": p.teto, "detalhe": p.detalhe}
                           for p in self.partes]}


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def faixa(valor: float) -> str:
    return next(nome for corte, nome in FAIXAS if valor >= corte)


def calcular(
    conta_atual_brl: float | None,
    economia_mensal_brl: float | None,
    frequencia: str | None,
    duracao: str | None,
    nivel_critico: str,
    fracao_sem_sol: float | None,
    cobertura_da_falta: float | None,
) -> IndiceOportunidade:
    """
    ``nivel_critico`` é "grave" (algo MC, aparelho de saúde ou prejuízo direto
    declarado), "critico" (só C) ou "nenhum". ``cobertura_da_falta`` é a chance
    (0..1) de a bateria recomendada segurar a falta que o cliente descreveu;
    ``None`` quando o estudo não recomendou banco.
    """
    # Economia: metade fração da conta, metade valor em reais (escala log de
    # R$ 100 a R$ 3.000 por mês — abaixo de R$ 100 a economia não move ninguém).
    if conta_atual_brl and economia_mensal_brl and economia_mensal_brl > 0:
        fracao = economia_mensal_brl / conta_atual_brl
        reais = _clip01(math.log(max(economia_mensal_brl, 1) / 100.0) / math.log(30.0))
        p_eco = 20.0 * _clip01(fracao / 0.85) + 15.0 * reais
        d_eco = f"solar tira ~{fracao:.0%} da conta"
    else:
        p_eco, d_eco = 0.0, "sem economia calculada"

    f = PESO_FREQUENCIA.get(frequencia or "", 0.0)
    d = PESO_DURACAO.get(duracao or "", 0.5)
    p_falta = TETO_FALTA * math.sqrt(f * d) * PESO_CRITICO.get(nivel_critico, PESO_CRITICO["nenhum"])
    d_falta = {"grave": "algo não pode ficar sem luz", "critico": "o que faz falta foi apontado",
               "nenhum": "nada marcado como crítico"}.get(nivel_critico, "")

    # Até 35% do consumo sem sol é o normal de qualquer casa: só o que passa
    # disso é energia que a bateria tem para deslocar.
    p_noite = TETO_NOITE * _clip01(((fracao_sem_sol or 0.0) - 0.35) / 0.45)
    d_noite = f"{fracao_sem_sol:.0%} do consumo sem sol" if fracao_sem_sol is not None else "sem curva"

    if cobertura_da_falta is None:
        p_enc, d_enc = 0.25 * TETO_ENCAIXE, "bateria a definir na conversa"
    else:
        p_enc = TETO_ENCAIXE * _clip01(cobertura_da_falta)
        d_enc = ("segura todas as faltas declaradas" if cobertura_da_falta >= 0.995
                 else f"segura ~{cobertura_da_falta:.0%} das faltas declaradas")

    partes = (
        Parte("Economia", p_eco, TETO_ECONOMIA, d_eco),
        Parte("Falta de luz", p_falta, TETO_FALTA, d_falta),
        Parte("Consumo sem sol", p_noite, TETO_NOITE, d_noite),
        Parte("Encaixe da solução", p_enc, TETO_ENCAIXE, d_enc),
    )
    valor = int(round(sum(p.pontos for p in partes)))
    return IndiceOportunidade(valor=valor, faixa=faixa(valor), partes=partes)
