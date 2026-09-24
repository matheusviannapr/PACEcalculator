"""
Ajuste da curva típica à conta de luz — o caminho de quem não tem levantamento.

O levantamento de equipamentos deriva a curva de carga do comportamento de
cada aparelho. Quando ele não existe, o que existe é a fatura: energia do mês,
às vezes separada em ponta e fora de ponta, às vezes com a demanda medida. Este
módulo pega a **forma** de uma curva típica do segmento e a ajusta a esses
números, de modo que a curva resultante, somada ao longo do ciclo de
faturamento, reproduza a conta.

O que o ajuste faz, em ordem:

1. **Horário de funcionamento.** Fora do horário declarado a curva típica cai
   para a carga de base do perfil — a geladeira, o servidor, a segurança que
   ficam ligados com o prédio vazio.
2. **Dias de operação.** Um comércio que fecha domingo tem dois dias: o de
   operação, com a forma do perfil, e o fechado, plano na carga de base. A
   energia do mês é a soma ponderada dos dois, e é ela que fecha com a fatura.
3. **Escala por posto tarifário.** Com a conta do Grupo A, ponta e fora de
   ponta recebem cada uma o seu fator, de modo que a energia de cada posto
   bata. Com a do Grupo B, um fator só.
4. **Forma pela demanda medida.** A demanda registrada na conta é o maior
   intervalo de 15 minutos do mês; o pico da curva de um dia típico fica
   abaixo dele. Quando a demanda vem informada, o que a curva tem acima da
   carga de base é elevado a um expoente — preservando a energia de cada
   posto — até o pico bater com a demanda medida descontada dessa diferença.
   Expoente maior que um afina o pico e esvazia as horas de rampa; menor que
   um achata.
5. **Sazonalidade.** Com o histórico de doze meses, cada estação recebe o seu
   fator; sem ele, as quatro estações saem iguais.

O resultado sabe o que é: uma curva média de dia, sem a variabilidade que o
levantamento produz. A variabilidade é **assumida** ao gerar o ensemble (ver
:meth:`AjusteCurva.ensemble`), e o relatório precisa dizer isso.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

import numpy as np

from .biblioteca import PerfilTipico, perfis_tipicos
from .ensemble import EnsembleCarga
from .nucleo import ESTACOES_ANO

__all__ = [
    "AjusteCurva",
    "ContaDeLuz",
    "FATOR_DEMANDA_MEDIDA",
    "MESES_DA_ESTACAO",
    "ajustar",
    "ajustar_melhor_perfil",
    "recortar_fracao",
]

#: Meses de cada estação, no hemisfério sul.
MESES_DA_ESTACAO: dict[str, tuple[int, ...]] = {
    "verão": (12, 1, 2),
    "outono": (3, 4, 5),
    "inverno": (6, 7, 8),
    "primavera": (9, 10, 11),
}

#: Razão entre a demanda medida na conta (o maior intervalo de 15 min do mês)
#: e o pico horário de um dia típico. A medição pega o pior quarto de hora do
#: pior dia; a curva média não tem nem um nem outro. A ordem de grandeza
#: (10%) vem da comparação entre curvas medidas e a sua média mensal em
#: cargas comerciais; é hipótese declarada, não constante física.
FATOR_DEMANDA_MEDIDA = 1.10

#: Erro acima do qual o ajuste avisa que não fechou com a conta.
TOLERANCIA_ERRO_PCT = 5.0

#: Limites do expoente de forma. Abaixo de 0,5 a curva vira um platô; acima
#: de 3 a demanda pedida não cabe na forma do perfil, e o aviso é mais útil
#: que uma curva com uma hora só de carga.
EXPOENTE_MIN, EXPOENTE_MAX = 0.5, 3.0

_PESOS_ERRO = {
    "energia_ponta_kwh": 0.35,
    "energia_fora_ponta_kwh": 0.25,
    "energia_mensal_kwh": 0.20,
    "demanda_ponta_kw": 0.10,
    "demanda_fora_ponta_kw": 0.10,
}


# ----------------------------------------------------------------------------
# A conta
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class ContaDeLuz:
    """
    O que a fatura diz. O mínimo é o consumo do mês; o resto refina.

    ``consumo_ponta_kwh`` e ``consumo_fora_ponta_kwh`` juntos caracterizam uma
    conta do Grupo A (ou da tarifa branca): o ajuste passa a ter dois fatores.
    ``demanda_*_kw`` é a demanda **medida**, não a contratada — a contratada é
    contrato, a medida é o que a instalação puxou.

    ``consumo_por_mes_kwh`` é o histórico de doze meses, indexado pelo número
    do mês (1 = janeiro). Com ele o ajuste sabe que julho não é janeiro.
    """

    consumo_mensal_kwh: float | None = None
    consumo_ponta_kwh: float | None = None
    consumo_fora_ponta_kwh: float | None = None
    demanda_ponta_kw: float | None = None
    demanda_fora_ponta_kw: float | None = None
    ponta_inicio_h: int = 18
    ponta_fim_h: int = 21
    dias_ciclo: int = 30
    dias_operacao_semana: int = 7
    #: Horário de funcionamento em horas cheias; ``None`` nos dois é 24 h.
    abertura_h: int | None = None
    fechamento_h: int | None = None
    consumo_por_mes_kwh: Mapping[int, float] | None = None

    def __post_init__(self) -> None:
        if not 1 <= int(self.dias_operacao_semana) <= 7:
            raise ValueError("dias de operação por semana deve estar entre 1 e 7")
        if not 1 <= int(self.dias_ciclo) <= 31:
            raise ValueError("dias do ciclo de faturamento deve estar entre 1 e 31")
        for nome in ("ponta_inicio_h", "ponta_fim_h"):
            if not 0 <= int(getattr(self, nome)) <= 24:
                raise ValueError(f"{nome} deve estar entre 0 e 24")
        if (self.abertura_h is None) != (self.fechamento_h is None):
            raise ValueError("abertura e fechamento vêm juntos, ou nenhum dos dois")
        for nome in ("consumo_mensal_kwh", "consumo_ponta_kwh", "consumo_fora_ponta_kwh",
                     "demanda_ponta_kw", "demanda_fora_ponta_kw"):
            valor = getattr(self, nome)
            if valor is not None and (not np.isfinite(valor) or valor < 0):
                raise ValueError(f"{nome} deve ser um número não negativo")
        if self.tem_postos:
            soma = float(self.consumo_ponta_kwh) + float(self.consumo_fora_ponta_kwh)
            if soma <= 0:
                raise ValueError("a conta precisa ter consumo em ao menos um posto")
            # A fatura do Grupo A traz os postos, não o total: o total é a soma.
            object.__setattr__(self, "consumo_mensal_kwh", soma)
        elif self.consumo_por_mes_kwh:
            valores = [float(v) for v in self.consumo_por_mes_kwh.values() if v is not None]
            if not valores or any(v < 0 for v in valores):
                raise ValueError("histórico mensal precisa de valores não negativos")
            if self.consumo_mensal_kwh is None:
                object.__setattr__(self, "consumo_mensal_kwh", float(np.mean(valores)))
        if not self.consumo_mensal_kwh or self.consumo_mensal_kwh <= 0:
            raise ValueError("a conta precisa de um consumo mensal positivo")

    # -- caracterização ----------------------------------------------------
    @property
    def tem_postos(self) -> bool:
        return self.consumo_ponta_kwh is not None and self.consumo_fora_ponta_kwh is not None

    @property
    def grupo(self) -> str:
        """``"A"`` quando os postos vêm separados; ``"B"`` quando só há o total."""
        return "A" if self.tem_postos else "B"

    @property
    def tem_demanda(self) -> bool:
        return bool(self.demanda_ponta_kw) or bool(self.demanda_fora_ponta_kw)

    @property
    def mascara_ponta(self) -> np.ndarray:
        """Horas do dia (0–23) que caem no posto de ponta, em dia útil."""
        horas = np.arange(24)
        inicio, fim = int(self.ponta_inicio_h), int(self.ponta_fim_h)
        if inicio == fim:
            return np.zeros(24, dtype=bool)
        if inicio < fim:
            return (horas >= inicio) & (horas < fim)
        return (horas >= inicio) | (horas < fim)

    @property
    def mascara_aberto(self) -> np.ndarray:
        """Horas em que o lugar funciona; tudo verdadeiro quando é 24 h."""
        if self.abertura_h is None or self.fechamento_h is None:
            return np.ones(24, dtype=bool)
        horas = np.arange(24)
        abre, fecha = int(self.abertura_h), int(self.fechamento_h)
        if abre == fecha:
            return np.ones(24, dtype=bool)
        if abre < fecha:
            return (horas >= abre) & (horas < fecha)
        return (horas >= abre) | (horas < fecha)

    # -- calendário do ciclo -----------------------------------------------
    @property
    def dias_uteis_operacao(self) -> float:
        """Dias de segunda a sexta em que opera, no ciclo — onde a ponta vale."""
        return float(self.dias_ciclo) * min(int(self.dias_operacao_semana), 5) / 7.0

    @property
    def dias_fim_de_semana_operacao(self) -> float:
        """Sábados e domingos em que opera: tudo fora de ponta pela tarifa."""
        return float(self.dias_ciclo) * max(int(self.dias_operacao_semana) - 5, 0) / 7.0

    @property
    def dias_fechado(self) -> float:
        return float(self.dias_ciclo) - self.dias_uteis_operacao - self.dias_fim_de_semana_operacao

    @property
    def fracao_dias_operacao(self) -> float:
        return int(self.dias_operacao_semana) / 7.0

    # -- sazonalidade ------------------------------------------------------
    def fatores_sazonais(self) -> dict[str, float]:
        """
        Quanto cada estação consome em relação à média do ano.

        Sem histórico, 1,0 para todas. Com histórico parcial, a estação sem
        mês informado fica em 1,0 — o ajuste não inventa o que a conta não
        trouxe.
        """
        if not self.consumo_por_mes_kwh:
            return {estacao: 1.0 for estacao in ESTACOES_ANO}
        historico = {int(m): float(v) for m, v in self.consumo_por_mes_kwh.items() if v is not None}
        media = float(np.mean(list(historico.values())))
        fatores: dict[str, float] = {}
        for estacao in ESTACOES_ANO:
            meses = [historico[m] for m in MESES_DA_ESTACAO[estacao] if m in historico]
            fatores[estacao] = float(np.mean(meses) / media) if meses and media > 0 else 1.0
        return fatores

    def resumo(self) -> dict[str, Any]:
        return {
            "grupo": self.grupo,
            "consumo_mensal_kwh": float(self.consumo_mensal_kwh),
            "consumo_ponta_kwh": self.consumo_ponta_kwh,
            "consumo_fora_ponta_kwh": self.consumo_fora_ponta_kwh,
            "demanda_ponta_kw": self.demanda_ponta_kw,
            "demanda_fora_ponta_kw": self.demanda_fora_ponta_kw,
            "ponta": (int(self.ponta_inicio_h), int(self.ponta_fim_h)),
            "dias_ciclo": int(self.dias_ciclo),
            "dias_operacao_semana": int(self.dias_operacao_semana),
            "horario": (
                None if self.abertura_h is None
                else (int(self.abertura_h), int(self.fechamento_h))
            ),
            "meses_no_historico": len(self.consumo_por_mes_kwh or {}),
        }


# ----------------------------------------------------------------------------
# O resultado
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class AjusteCurva:
    """
    A curva típica depois do ajuste, com a memória do que foi feito.

    ``curva_operacao_kw`` é o dia em que o lugar funciona; ``curva_fechado_kw``
    o dia em que não. As duas somadas no calendário do ciclo reproduzem a
    conta — ``obtidos`` contra ``alvos`` mostra o quanto.
    """

    perfil: PerfilTipico
    conta: ContaDeLuz
    curva_operacao_kw: np.ndarray
    curva_fechado_kw: np.ndarray
    fator_ponta: float
    fator_fora: float
    expoente: float
    fatores_sazonais: dict[str, float]
    alvos: dict[str, float | None]
    obtidos: dict[str, float]
    erros_pct: dict[str, float | None]
    avisos: list[str] = field(default_factory=list)
    pontuacao: float = 0.0
    fator_demanda_medida: float = FATOR_DEMANDA_MEDIDA
    #: A curva do dia de operação foi reescrita pelo operador sobre o ajuste.
    editada: bool = False

    # -- grandezas derivadas -----------------------------------------------
    @property
    def carga_base_kw(self) -> float:
        return float(self.curva_fechado_kw[0])

    @property
    def consumo_mensal_kwh(self) -> float:
        return float(self.obtidos["energia_mensal_kwh"])

    @property
    def consumo_anual_kwh(self) -> float:
        """A conta somada no ano, com a sazonalidade quando houve histórico."""
        media_fatores = float(np.mean(list(self.fatores_sazonais.values())))
        return self.consumo_mensal_kwh * 12.0 * media_fatores

    @property
    def consumo_diario_operacao_kwh(self) -> float:
        return float(self.curva_operacao_kw.sum())

    @property
    def consumo_diario_medio_kwh(self) -> float:
        """A média entre dias abertos e fechados: o que a conta vê por dia."""
        return self.consumo_mensal_kwh / float(self.conta.dias_ciclo)

    @property
    def demanda_maxima_kw(self) -> float:
        return float(self.curva_operacao_kw.max())

    @property
    def demanda_media_kw(self) -> float:
        return float(self.curva_operacao_kw.mean())

    @property
    def fator_de_carga(self) -> float:
        pico = self.demanda_maxima_kw
        return self.demanda_media_kw / pico if pico > 0 else 0.0

    @property
    def fechou(self) -> bool:
        """Nenhum alvo informado ficou fora da tolerância."""
        return all(
            erro is None or erro <= TOLERANCIA_ERRO_PCT for erro in self.erros_pct.values()
        )

    @property
    def sazonalidade_aplicada(self) -> bool:
        return any(abs(f - 1.0) > 1e-9 for f in self.fatores_sazonais.values())

    def com_curva_editada(
        self,
        curva_kw: Sequence[float],
        manter_energia: bool = True,
    ) -> "AjusteCurva":
        """
        O mesmo ajuste com a curva do dia de operação reescrita pelo operador.

        Quem conhece a instalação sabe o que a curva típica não sabe — o
        chuveiro das sete, a máquina que liga às cinco. A edição entra sobre a
        curva ajustada, hora a hora. O dia fechado acompanha: fica na hora mais
        vazia da curva editada.

        Com ``manter_energia`` a curva editada é reescalada para que o ciclo
        continue fechando com a energia da conta: o operador desenha a
        **forma**, a fatura dá o **tamanho**. Sem ele, o que foi digitado vale
        em kW, e a diferença para a conta aparece na conferência.
        """
        curva = np.asarray(curva_kw, dtype=float).ravel()
        if curva.size != 24 or not np.isfinite(curva).all() or (curva < 0).any():
            raise ValueError("a curva editada precisa de 24 valores não negativos, um por hora")
        if curva.sum() <= 0:
            raise ValueError("a curva editada é nula")
        fechado = np.full(24, float(curva.min()))
        if manter_energia:
            conta = self.conta
            dias_abertos = conta.dias_uteis_operacao + conta.dias_fim_de_semana_operacao
            energia = dias_abertos * float(curva.sum()) + conta.dias_fechado * float(fechado.sum())
            escala = float(conta.consumo_mensal_kwh) / energia if energia > 0 else 1.0
            curva = curva * escala
            fechado = fechado * escala
        obtidos, erros, pontuacao, avisos = _conferir(
            self.conta, curva, fechado, self.alvos, self.fator_demanda_medida,
        )
        avisos.insert(0, (
            "A curva do dia de operação foi editada pelo operador sobre a curva "
            "ajustada" + (
                "; a energia do ciclo foi mantida na da conta." if manter_energia
                else ", em kW — a conferência mostra o quanto ela se afasta da conta."
            )
        ))
        avisos.extend(a for a in self.avisos if "sintética" in a)
        return replace(
            self,
            curva_operacao_kw=curva,
            curva_fechado_kw=fechado,
            obtidos=obtidos,
            erros_pct=erros,
            pontuacao=float(pontuacao),
            avisos=avisos,
            editada=True,
        )

    def curva_do_dia(self, estacao: str | None = None, operacao: bool = True) -> np.ndarray:
        """A curva de 24 h em kW, de uma estação, em dia aberto ou fechado."""
        base = self.curva_operacao_kw if operacao else self.curva_fechado_kw
        fator = 1.0 if estacao is None else self.fatores_sazonais.get(estacao, 1.0)
        return base * fator

    # -- o ensemble --------------------------------------------------------
    def ensemble(
        self,
        num_simulacoes: int = 300,
        dispersao_diaria: float = 0.15,
        dispersao_horaria: float = 0.05,
        semente: int | None = 42,
        estacoes: Sequence[str] = tuple(ESTACOES_ANO),
    ) -> EnsembleCarga:
        """
        O ensemble que o estudo consome, com a variabilidade **assumida**.

        Cada dia simulado é a curva ajustada vezes dois ruídos log-normais de
        média um: um fator do dia (``dispersao_diaria``: o dia mais movimentado
        e o mais parado) e um fator por hora (``dispersao_horaria``: a hora em
        que tudo liga junto). O primeiro move a energia; o segundo forma a
        cauda de pico que decide o inversor. Dias abertos e fechados entram na
        proporção da semana declarada.

        Os dois desvios são hipóteses sobre a instalação, não medições dela —
        15% e 5% são a ordem de grandeza da carga comercial. O levantamento de
        equipamentos deriva essa distribuição; este atalho a arbitra, e o
        relatório diz isso.
        """
        n = int(num_simulacoes)
        if n < 1:
            raise ValueError("ao menos uma simulação")
        rng = np.random.default_rng(semente)
        n_abertos = int(round(n * self.conta.fracao_dias_operacao))
        n_fechados = n - n_abertos
        # Média um: lognormal(0, s) tem média exp(s²/2), e o 1% que isso põe
        # na energia viraria erro contra a conta que acabou de fechar.
        s_d, s_h = float(dispersao_diaria), float(dispersao_horaria)
        interpolador = _interpolador_minuto()

        perfis: dict[str, np.ndarray] = {}
        for estacao in estacoes:
            fator = self.fatores_sazonais.get(estacao, 1.0)
            blocos = []
            for curva, quantidade in (
                (self.curva_operacao_kw, n_abertos),
                (self.curva_fechado_kw, n_fechados),
            ):
                if quantidade <= 0:
                    continue
                dia = rng.lognormal(-s_d**2 / 2, s_d, size=(quantidade, 1)) if s_d > 0 else 1.0
                hora = rng.lognormal(-s_h**2 / 2, s_h, size=(quantidade, 24)) if s_h > 0 else 1.0
                blocos.append(curva[None, :] * fator * dia * hora)
            horario_kw = np.concatenate(blocos, axis=0)
            rng.shuffle(horario_kw, axis=0)
            perfis[estacao] = horario_kw @ interpolador * 1000.0

        return EnsembleCarga(
            perfis_w=perfis,
            passo_min=1,
            metadados={
                "origem": "curva_tipica_ajustada",
                "num_simulacoes": n,
                "dispersao_diaria": s_d,
                "dispersao_horaria": s_h,
                "sazonalidade_aplicada": self.sazonalidade_aplicada,
                "dias_operacao_semana": int(self.conta.dias_operacao_semana),
                "semente": semente,
                "ajuste": self.resumo(),
                "aviso": (
                    "Ensemble gerado de curva típica ajustada à conta de luz: a forma vem "
                    f"do perfil {self.perfil.nome} e o tamanho, da fatura. A variabilidade "
                    f"dia a dia ({s_d:.0%}) e hora a hora ({s_h:.0%}) é hipótese, não "
                    "medição — e é ela que forma a cauda que dimensiona o inversor."
                ),
            },
        )

    def resumo(self) -> dict[str, Any]:
        return {
            "perfil": self.perfil.id,
            "perfil_nome": self.perfil.nome,
            "perfil_confiavel": self.perfil.confiavel,
            "conta": self.conta.resumo(),
            "fator_ponta": float(self.fator_ponta),
            "fator_fora": float(self.fator_fora),
            "expoente": float(self.expoente),
            "fator_demanda_medida": float(self.fator_demanda_medida),
            "fatores_sazonais": dict(self.fatores_sazonais),
            "curva_operacao_kw": [float(v) for v in self.curva_operacao_kw],
            "curva_fechado_kw": [float(v) for v in self.curva_fechado_kw],
            "carga_base_kw": self.carga_base_kw,
            "consumo_mensal_kwh": self.consumo_mensal_kwh,
            "consumo_anual_kwh": self.consumo_anual_kwh,
            "consumo_diario_operacao_kwh": self.consumo_diario_operacao_kwh,
            "demanda_maxima_kw": self.demanda_maxima_kw,
            "demanda_media_kw": self.demanda_media_kw,
            "fator_de_carga": self.fator_de_carga,
            "alvos": dict(self.alvos),
            "obtidos": dict(self.obtidos),
            "erros_pct": dict(self.erros_pct),
            "fechou": self.fechou,
            "pontuacao": float(self.pontuacao),
            "editada": bool(self.editada),
            "avisos": list(self.avisos),
        }


# ----------------------------------------------------------------------------
# O ajuste
# ----------------------------------------------------------------------------
def ajustar(
    conta: ContaDeLuz,
    perfil: PerfilTipico,
    fator_demanda_medida: float = FATOR_DEMANDA_MEDIDA,
) -> AjusteCurva:
    """
    Ajusta a forma do perfil aos números da conta.

    Ver o cabeçalho do módulo para os passos. O resultado sempre existe: quando
    a conta pede algo que a forma não comporta — consumo na ponta num lugar
    que fecha antes dela, demanda medida três vezes o pico da forma — o ajuste
    faz o que dá, registra o erro e avisa, em vez de levantar exceção. Uma
    conta mal digitada tem que aparecer na tela como erro de ajuste, não como
    traceback.
    """
    avisos: list[str] = []
    pu = _forma_no_horario(perfil.curva_pu, conta.mascara_aberto)
    ponta = conta.mascara_ponta
    fora = ~ponta

    n_u = conta.dias_uteis_operacao
    n_s = conta.dias_fim_de_semana_operacao
    n_f = conta.dias_fechado

    # 3. escala por posto -------------------------------------------------
    alvo_total = float(conta.consumo_mensal_kwh)
    alvo_ponta = alvo_fora = None
    fator_ponta = fator_fora = None
    if conta.tem_postos:
        alvo_ponta = float(conta.consumo_ponta_kwh)
        alvo_fora = float(conta.consumo_fora_ponta_kwh)
        s_p, s_f = float(pu[ponta].sum()), float(pu[fora].sum())
        if alvo_ponta > 0 and (n_u <= 0 or s_p <= 0):
            avisos.append(
                "A conta registra consumo na ponta, mas o perfil não tem carga nesse "
                "horário com o funcionamento declarado. O ajuste usou só o total do mês."
            )
        elif alvo_ponta <= 0 and s_p > 0:
            # Ponta zerada é conta de quem desliga na ponta; a forma acompanha.
            fator_ponta = 0.0
        elif alvo_ponta > 0 and not (ponta & conta.mascara_aberto).any():
            avisos.append(
                "A conta registra consumo na ponta, mas o horário de funcionamento "
                "declarado fecha antes dela. O ajuste pôs essa energia sobre a carga de "
                "base das horas de ponta — confira o horário, ou a conta."
            )
        if fator_ponta is None and n_u > 0 and s_p > 0:
            fator_ponta = alvo_ponta / (n_u * s_p)
        if fator_ponta is not None:
            fator_fora = _fator_fora_de_ponta(
                alvo_fora, fator_ponta, pu, ponta, n_u, n_s, n_f,
            )
            if fator_fora <= 0:
                avisos.append(
                    "A energia fora de ponta da conta é menor do que a forma do perfil "
                    "exige para o consumo de ponta informado. Confira os dois valores; "
                    "o ajuste usou só o total do mês."
                )
                fator_ponta = fator_fora = None
    if fator_ponta is None:
        denominador = (n_u + n_s) + n_f * 24.0 * float(pu.min())
        escala = alvo_total / denominador
        fator_ponta = fator_fora = escala
    curva = np.where(ponta, pu * fator_ponta, pu * fator_fora)
    # A carga de base é a hora mais vazia do dia aberto — o que fica ligado
    # com o prédio vazio — e é fixada aqui, antes de mudar a forma: afinar o
    # pico muda a tarde, não o que a geladeira consome de madrugada.
    fechado = np.full(24, float(curva.min()))

    # 4. forma pela demanda medida ----------------------------------------
    expoente = 1.0
    if conta.tem_demanda:
        expoente, aviso = _expoente_de_forma(
            curva, ponta, fechado[0], conta.demanda_ponta_kw, conta.demanda_fora_ponta_kw,
            fator_demanda_medida,
        )
        if aviso:
            avisos.append(aviso)
        curva = _aplicar_expoente(curva, ponta, fechado[0], expoente)
    curva = np.maximum(curva, 0.0)

    # 5. sazonalidade -----------------------------------------------------
    fatores = conta.fatores_sazonais()
    if conta.consumo_por_mes_kwh and len(conta.consumo_por_mes_kwh) < 12:
        avisos.append(
            f"Histórico com {len(conta.consumo_por_mes_kwh)} meses: as estações sem mês "
            "informado ficaram na média do ano."
        )

    # conferência ---------------------------------------------------------
    alvos: dict[str, float | None] = {
        "energia_mensal_kwh": alvo_total,
        "energia_ponta_kwh": alvo_ponta,
        "energia_fora_ponta_kwh": alvo_fora,
        "demanda_ponta_kw": conta.demanda_ponta_kw or None,
        "demanda_fora_ponta_kw": conta.demanda_fora_ponta_kw or None,
    }
    obtidos, erros, pontuacao, avisos_conferencia = _conferir(
        conta, curva, fechado, alvos, fator_demanda_medida,
    )
    avisos.extend(avisos_conferencia)
    if not perfil.confiavel:
        avisos.append(
            f"A curva de referência ({perfil.nome}) está marcada como sintética na "
            "origem da base: serve de forma plausível, não de medição do setor."
        )

    return AjusteCurva(
        perfil=perfil,
        conta=conta,
        curva_operacao_kw=curva,
        curva_fechado_kw=fechado,
        fator_ponta=float(fator_ponta),
        fator_fora=float(fator_fora),
        expoente=float(expoente),
        fatores_sazonais=fatores,
        alvos=alvos,
        obtidos=obtidos,
        erros_pct=erros,
        avisos=avisos,
        pontuacao=float(pontuacao),
        fator_demanda_medida=float(fator_demanda_medida),
    )


def ajustar_melhor_perfil(
    conta: ContaDeLuz,
    candidatos: Sequence[PerfilTipico] | None = None,
    preferido: PerfilTipico | None = None,
    fator_demanda_medida: float = FATOR_DEMANDA_MEDIDA,
) -> AjusteCurva:
    """
    Ajusta todos os candidatos e devolve o que melhor explica a conta.

    Só faz sentido quando a conta traz mais do que a energia: com o total do
    mês apenas, toda forma fecha exatamente, e a escolha volta para o
    ``preferido`` (o perfil do segmento). Com ponta, fora de ponta e demanda
    medida, a forma que reproduz esses números com menos deformação e menor
    erro é a que se parece mais com a instalação.

    O empate é decidido pelo preferido, para que o segmento declarado não
    seja trocado por um vizinho que explica a conta igualmente bem.
    """
    lista = list(candidatos) if candidatos is not None else list(perfis_tipicos().values())
    if preferido is not None and all(p.id != preferido.id for p in lista):
        lista.insert(0, preferido)
    if not lista:
        raise ValueError("nenhum perfil típico disponível para ajustar")
    if not (conta.tem_postos or conta.tem_demanda):
        escolhido = preferido or lista[0]
        return ajustar(conta, escolhido, fator_demanda_medida)

    ajustes = [ajustar(conta, p, fator_demanda_medida) for p in lista]

    def chave(ajuste: AjusteCurva) -> tuple[float, int]:
        # A pontuação é o erro ponderado. O resto mede o quanto a forma teve
        # de ser forçada para fechar: o expoente longe de 1, e os fatores de
        # ponta e fora de ponta longe um do outro — um perfil cuja fatia
        # natural de ponta é a da conta não precisa de fatores diferentes.
        forcado = abs(np.log(ajuste.expoente)) * 10.0
        if ajuste.fator_ponta > 0 and ajuste.fator_fora > 0:
            forcado += abs(np.log(ajuste.fator_ponta / ajuste.fator_fora)) * 10.0
        empate = 0 if (preferido is not None and ajuste.perfil.id == preferido.id) else 1
        # Arredondado: ruído de ponto flutuante não pode decidir contra o
        # segmento declarado.
        return (round(ajuste.pontuacao + forcado, 6), empate)

    melhor = min(ajustes, key=chave)
    if preferido is not None and melhor.perfil.id != preferido.id:
        melhor.avisos.insert(0, (
            f"O perfil que melhor explica a conta é {melhor.perfil.nome}, não o do "
            f"segmento declarado ({preferido.nome}). O ajuste seguiu com o primeiro."
        ))
    return melhor


def recortar_fracao(ensemble: EnsembleCarga, fracao: float) -> EnsembleCarga:
    """
    Uma fração da carga total, para o quadro de backup sem lista de circuitos.

    É recorte grosseiro, e o metadado diz isso: sem levantamento não há como
    saber quais circuitos ficam. Com ``fracao == 1`` devolve o próprio
    ensemble.
    """
    f = float(fracao)
    if not 0.0 < f <= 1.0:
        raise ValueError("fração do backup deve estar em (0, 1]")
    if f >= 1.0:
        return ensemble
    return EnsembleCarga(
        perfis_w={e: ensemble.perfis_w[e] * f for e in ensemble.estacoes},
        picos_janela_w={e: ensemble.picos_janela_w[e] * f for e in ensemble.estacoes},
        passo_min=ensemble.passo_min,
        metadados=dict(ensemble.metadados, fracao_backup=f),
    )


# ----------------------------------------------------------------------------
# Peças
# ----------------------------------------------------------------------------
_ROTULOS = {
    "energia_mensal_kwh": "energia do mês",
    "energia_ponta_kwh": "energia na ponta",
    "energia_fora_ponta_kwh": "energia fora de ponta",
    "demanda_ponta_kw": "demanda na ponta",
    "demanda_fora_ponta_kw": "demanda fora de ponta",
}


def _conferir(
    conta: ContaDeLuz,
    curva: np.ndarray,
    fechado: np.ndarray,
    alvos: Mapping[str, float | None],
    fator_demanda_medida: float,
) -> tuple[dict[str, float], dict[str, float | None], float, list[str]]:
    """
    Soma a curva no calendário do ciclo e compara com a conta.

    Devolve o obtido, o erro percentual por grandeza informada, a pontuação
    ponderada e os avisos de conferência. É a mesma conta para a curva
    ajustada e para a curva editada — e é isso que faz a edição continuar
    auditável.
    """
    ponta = conta.mascara_ponta
    fora = ~ponta
    n_u, n_s, n_f = conta.dias_uteis_operacao, conta.dias_fim_de_semana_operacao, conta.dias_fechado
    e_mes = (n_u + n_s) * float(curva.sum()) + n_f * float(fechado.sum())
    e_ponta = n_u * float(curva[ponta].sum())
    obtidos = {
        "energia_mensal_kwh": e_mes,
        "energia_ponta_kwh": e_ponta,
        "energia_fora_ponta_kwh": e_mes - e_ponta,
        "demanda_ponta_kw": float(curva[ponta].max()) * fator_demanda_medida if ponta.any() else 0.0,
        "demanda_fora_ponta_kw": float(curva[fora].max()) * fator_demanda_medida if fora.any() else 0.0,
    }
    erros = {
        chave: (abs(obtidos[chave] - alvo) / alvo * 100.0 if alvo else None)
        for chave, alvo in alvos.items()
    }
    pontuacao = sum(_PESOS_ERRO[k] * e for k, e in erros.items() if e is not None)
    avisos: list[str] = []
    for chave, erro in erros.items():
        if erro is not None and erro > TOLERANCIA_ERRO_PCT:
            avisos.append(
                f"O ajuste não fechou em {_ROTULOS[chave]}: {erro:.0f}% de diferença "
                "para a conta."
            )
    fator_carga = float(curva.mean() / curva.max()) if curva.max() > 0 else 0.0
    if fator_carga < 0.15 or fator_carga > 0.95:
        avisos.append(
            f"Fator de carga de {fator_carga:.0%} está fora do que se vê em instalações "
            "reais (15% a 95%). Ou a demanda medida não corresponde ao consumo, ou o "
            "perfil não é o desse lugar."
        )
    return obtidos, erros, float(pontuacao), avisos


def _fator_fora_de_ponta(
    alvo_fora: float,
    fator_ponta: float,
    pu: np.ndarray,
    ponta: np.ndarray,
    n_u: float,
    n_s: float,
    n_f: float,
) -> float:
    """
    O fator fora de ponta que fecha a energia fora de ponta do ciclo.

    A energia fora de ponta soma três parcelas: as horas fora de ponta dos dias
    de operação, os fins de semana em operação (onde a ponta não vale e o dia
    inteiro é fora) e os dias fechados, na carga de base. A base é a hora mais
    vazia do dia aberto — que pode cair numa hora de ponta quando o fator de
    ponta é pequeno —, então ela depende do próprio fator que se procura. O
    ponto fixo converge em poucas voltas porque a base pesa pouco no total.
    """
    fora = ~ponta
    s_p, s_f = float(pu[ponta].sum()), float(pu[fora].sum())
    dias_operacao = n_u + n_s
    if dias_operacao * s_f <= 0:
        return 0.0
    fator_fora = alvo_fora / (dias_operacao * s_f)
    for _ in range(50):
        curva = np.where(ponta, pu * fator_ponta, pu * fator_fora)
        base = float(curva.min())
        novo = (alvo_fora - n_s * fator_ponta * s_p - n_f * 24.0 * base) / (dias_operacao * s_f)
        if abs(novo - fator_fora) <= 1e-12 * max(1.0, abs(novo)):
            fator_fora = novo
            break
        fator_fora = novo
    return fator_fora


def _forma_no_horario(curva_pu: np.ndarray, aberto: np.ndarray) -> np.ndarray:
    """A forma do perfil com as horas fechadas rebaixadas à carga de base."""
    pu = np.asarray(curva_pu, dtype=float).copy()
    if not aberto.all():
        base = float(pu.min())
        pu = np.where(aberto, pu, base)
    total = pu.sum()
    if total <= 0:
        raise ValueError("a forma do perfil ficou nula depois de aplicar o horário")
    return pu / total


def _com_expoente(curva: np.ndarray, mascara: np.ndarray, base: float, gama: float) -> np.ndarray:
    """
    O excesso acima da base elevado a ``gama`` nas horas da máscara, com a
    energia dessas horas preservada.
    """
    saida = curva.copy()
    excesso = np.maximum(curva[mascara] - base, 0.0)
    total = float(excesso.sum())
    if total <= 0:
        return saida
    potencia = excesso ** gama
    saida[mascara] = base + potencia * (total / float(potencia.sum()))
    return saida


def _expoente_de_forma(
    curva: np.ndarray,
    ponta: np.ndarray,
    base: float,
    demanda_ponta: float | None,
    demanda_fora: float | None,
    fator_demanda_medida: float,
) -> tuple[float, str | None]:
    """
    O expoente que leva o pico da forma à demanda medida.

    Em cada posto, o que a curva tem acima da carga de base vira
    ``base + A·(curva − base)^γ``, com ``A`` escolhido para que a energia do
    posto não mude. Um ``γ`` maior que um afina o pico à custa das horas de
    rampa; menor que um achata. A madrugada, que está na base, não se move —
    e nada fica negativo, seja qual for o ``γ``.

    Com duas demandas, ``γ`` minimiza o erro relativo quadrático das duas; com
    uma, fecha essa. A busca é por varredura fina, porque a função é monótona
    e barata.
    """
    alvos: list[tuple[np.ndarray, float]] = []
    for mascara, demanda in zip((ponta, ~ponta), (demanda_ponta, demanda_fora)):
        if demanda and mascara.any() and float(curva[mascara].max()) > base:
            alvos.append((mascara, float(demanda) / fator_demanda_medida))
    if not alvos:
        return 1.0, None

    def erro(gama: float) -> float:
        total = 0.0
        for mascara, alvo in alvos:
            pico = float(_com_expoente(curva, mascara, base, gama)[mascara].max())
            total += ((pico - alvo) / alvo) ** 2
        return total

    grade = np.linspace(EXPOENTE_MIN, EXPOENTE_MAX, 251)
    erros = np.array([erro(float(g)) for g in grade])
    gama = float(grade[int(np.argmin(erros))])
    aviso = None
    if gama <= EXPOENTE_MIN + 1e-9 or gama >= EXPOENTE_MAX - 1e-9:
        picos = ", ".join(
            f"{float(_com_expoente(curva, m, base, gama)[m].max()) * fator_demanda_medida:.0f} kW "
            f"contra {alvo * fator_demanda_medida:.0f} kW medidos"
            for m, alvo in alvos
        )
        aviso = (
            f"A demanda medida não cabe na forma do perfil: o ajuste chegou ao limite do "
            f"expoente ({gama:.1f}) e o pico ficou em {picos}. Ou a demanda da conta vem "
            "de um pico que a curva típica não modela, ou o perfil não é o desse lugar."
        )
    return gama, aviso


def _aplicar_expoente(curva: np.ndarray, ponta: np.ndarray, base: float, gama: float) -> np.ndarray:
    saida = curva
    for mascara in (ponta, ~ponta):
        if mascara.any():
            saida = _com_expoente(saida, mascara, base, gama)
    return saida


def _interpolador_minuto() -> np.ndarray:
    """
    Matriz ``(24, 1440)`` que leva valores horários a minutos, por reta.

    Cada valor horário vale no centro da hora, e o minuto entre dois centros
    é a reta entre eles — com a volta pela meia-noite, para que a curva seja
    contínua. A interpolação linear periódica preserva a soma exatamente (a
    energia do dia) e nunca passa do maior valor horário (o pico).
    """
    matriz = np.zeros((24, 1440))
    # O minuto 30 de cada hora carrega o valor horário inteiro; a soma de
    # cada segmento linear entre dois centros telescopa no ciclo, e a energia
    # do dia sai exata.
    posicao = np.arange(1440) / 60.0 - 0.5  # 0 no centro da hora 0
    esquerda = np.floor(posicao).astype(int) % 24
    direita = (esquerda + 1) % 24
    peso = posicao - np.floor(posicao)
    matriz[esquerda, np.arange(1440)] += 1.0 - peso
    matriz[direita, np.arange(1440)] += peso
    return matriz
