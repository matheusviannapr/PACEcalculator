"""
A curva híbrida: as cargas críticas que o cliente contou, dentro da curva
característica do segmento ajustada à conta de luz.

O questionário não é um levantamento. Ele sabe bem o que é **crítico** — o
cliente disse o que fez falta, e são poucos itens, de potência conhecida —, e
sabe mal o **resto da casa**: a lista de "o que mais tem" é incompleta e usa
potência típica. A conta de luz é o contrário: sabe o total, não sabe o que é
o quê. Este módulo junta as duas metades:

1. **Críticas, item a item.** Os equipamentos MC e C viram o mesmo Monte Carlo
   da vistoria. É daí que sai o quadro de backup, com a variabilidade real de
   quem liga junto.
2. **A forma do dia, das duas fontes.** A curva do cenário inteiro (tudo que o
   cliente marcou, com os horários da rotina dele) e a curva típica do
   segmento. Misturam-se pelo **quanto da conta as respostas explicam**: se o
   que ele listou soma 80% dos kWh da fatura, a forma dele manda; se soma 20%,
   ele esqueceu a maior parte da casa, e a forma típica manda.
3. **O tamanho, da conta.** O ajuste à fatura (:mod:`aurum.demanda.ajuste`)
   fecha a energia do mês. A conferência dele continua valendo.
4. **Críticas por dentro do total, sempre.** Em nenhuma hora a curva total fica
   abaixo do que as críticas puxam: o resto da casa é o que sobra da conta
   depois delas, redistribuído na forma misturada.

O ensemble total soma, dia a dia, as críticas simuladas e o resto com um fator
diário log-normal. Assim o pico da hora em que a geladeira e o ar ligam juntos
não some no meio de uma média.

**A tabela de cargas** é o que deixa o estudo sair "como se houvesse
levantamento": cada item marcado com a sua energia simulada e a criticidade
declarada, e uma linha final — *Demais cargas (pela conta)* — com o que a
fatura tem e a lista não explica. Se a lista passa da conta, os itens não
críticos encolhem na proporção (potência típica alta), e isso vira aviso.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from ..demanda.ajuste import AjusteCurva, ContaDeLuz, ajustar
from ..demanda.biblioteca import PerfilTipico, perfil_tipico
from ..demanda.cenario import Cenario
from ..demanda.ensemble import EnsembleCarga, simular_ensemble
from ..demanda.nucleo import ESTACOES_ANO

#: Quem conta como crítico para o quadro de backup. "P" (aguento algumas horas)
#: fica fora do quadro e entra no estudo de escopos, como na vistoria.
CRITICAS = ("MC", "C")

#: Limites do peso da forma das respostas. Nunca zero — o horário do jantar e do
#: ar a noite toda é informação que a curva típica não tem —, nunca um — a lista
#: do questionário nunca é a casa inteira.
PESO_MINIMO, PESO_MAXIMO = 0.25, 0.75

#: Rótulo da linha que a conta tem e a lista não explica.
DEMAIS = "Demais cargas (estimado pela conta)"


@dataclass
class CurvaHibrida:
    perfil: PerfilTipico
    #: O ajuste à conta com a curva do dia reescrita pela forma híbrida. É ele
    #: que guarda a conferência contra a fatura e os fatores sazonais.
    ajuste: AjusteCurva
    ensemble_total: EnsembleCarga
    ensemble_critico: EnsembleCarga | None
    #: Curvas médias de 24 h, em kW, do dia de operação.
    curva_total_kw: np.ndarray
    curva_critica_kw: np.ndarray
    curva_respostas_kw: np.ndarray
    curva_tipica_kw: np.ndarray
    #: Quanto a forma das respostas pesou na mistura (0..1).
    peso_respostas: float
    #: Quanto da conta a lista do questionário explica, antes de qualquer ajuste.
    cobertura_respostas: float
    #: Uma linha por equipamento marcado, mais a linha das demais cargas.
    cargas: pd.DataFrame
    avisos: list[str] = field(default_factory=list)

    @property
    def consumo_mensal_kwh(self) -> float:
        return float(self.ajuste.conta.consumo_mensal_kwh)

    def principais_consumidores(self, n: int = 3) -> pd.DataFrame:
        """Os ``n`` itens que mais pesam, mais a linha das demais cargas no fim."""
        itens = self.cargas[self.cargas["equipamento"] != DEMAIS]
        topo = itens.groupby("rotulo", as_index=False)["energia_kwh_mes"].sum()
        topo = topo.sort_values("energia_kwh_mes", ascending=False).head(n)
        resto = self.consumo_mensal_kwh - float(topo["energia_kwh_mes"].sum())
        linhas = [{"rotulo": r.rotulo, "energia_kwh_mes": float(r.energia_kwh_mes)} for r in topo.itertuples()]
        if resto > 0.005 * self.consumo_mensal_kwh:
            linhas.append({"rotulo": "Demais cargas", "energia_kwh_mes": resto})
        df = pd.DataFrame(linhas)
        df["fracao"] = df["energia_kwh_mes"] / self.consumo_mensal_kwh
        return df


# ----------------------------------------------------------------------------
# Peças
# ----------------------------------------------------------------------------
def _horaria_kw(ensemble: EnsembleCarga) -> np.ndarray:
    """Média de 24 h, em kW, sobre simulações e estações."""
    perfis = np.vstack([ensemble.perfil_medio_w(e) for e in ensemble.estacoes])
    por_minuto = perfis.mean(axis=0)
    passos_hora = ensemble.passos_por_dia // 24
    return por_minuto.reshape(24, passos_hora).mean(axis=1) / 1000.0


def _so_criticas(cenario: Cenario) -> Cenario | None:
    """O mesmo cenário só com as linhas MC e C, ou None se não houver nenhuma."""
    if not cenario.tem_criticidade:
        return None
    comodos = {}
    for nome, tabela in cenario.comodos.items():
        recorte = tabela[tabela["criticidade"].astype(str).str.upper().isin(CRITICAS)]
        if not recorte.empty:
            comodos[nome] = recorte.reset_index(drop=True)
    if not comodos:
        return None
    return replace(cenario, comodos=comodos, instancias={n: cenario.instancias.get(n, 1) for n in comodos})


def _energia_por_item(cenario: Cenario, simulacoes: int, semente: int) -> pd.DataFrame:
    """
    Energia diária média de cada equipamento, simulado sozinho.

    Simular item a item custa pouco (o D² vetoriza por equipamento) e evita uma
    fórmula paralela de "horas × fator" que divergiria do motor em silêncio.
    """
    linhas = []
    for comodo, tabela in cenario.comodos.items():
        instancias = int(cenario.instancias.get(comodo, 1))
        for i in range(len(tabela)):
            linha = tabela.iloc[[i]].reset_index(drop=True)
            sozinho = replace(cenario, comodos={comodo: linha}, instancias={comodo: instancias})
            try:
                ens = simular_ensemble(sozinho.para_comodos(), sozinho.instancias_de(),
                                       num_simulacoes=simulacoes, semente=semente + i)
                kwh_dia = float(ens.energia_diaria_kwh().mean())
            except ValueError:
                # Item que o cenário recusa sozinho (ex.: "outro" sem potência) não pesa.
                kwh_dia = 0.0
            r = linha.iloc[0]
            linhas.append({
                "comodo": comodo,
                "equipamento": str(r.get("Equipamento")),
                "criticidade": str(r.get("criticidade") or "NC").upper(),
                "quantidade": float(r.get("Quantidade") or 1) * instancias,
                "potencia_w": float(r.get("Potência") or 0),
                "energia_kwh_dia": kwh_dia,
            })
    return pd.DataFrame(linhas)


def _rotulo(nome: str) -> str:
    """'Ar-condicionado split 12.000 BTU/h — madrugada' -> 'Ar-condicionado'."""
    base = str(nome).split(" — ")[0]
    for corte in (" split ", " 4 portas", " (", " de embutir", " duplex", " 1-2 zonas", " 5.500 W", " até 100 L"):
        if corte in base:
            base = base.split(corte)[0]
    return base.replace("Luminária LED 2×20 W", "Iluminação").replace("Iluminação geral (por ponto)", "Iluminação") \
               .replace("Tomadas de uso geral (carga estimada)", "Tomadas e pequenos aparelhos").strip()


def _lognormal_media_um(rng: np.random.Generator, sigma: float, tamanho) -> np.ndarray:
    return rng.lognormal(mean=-0.5 * sigma ** 2, sigma=sigma, size=tamanho)


# ----------------------------------------------------------------------------
# A montagem
# ----------------------------------------------------------------------------
def montar_curva(
    cenario: Cenario,
    conta: ContaDeLuz,
    perfil_id: str,
    simulacoes: int = 200,
    semente: int = 42,
    dispersao_diaria: float = 0.15,
    rotulos: dict[str, str] | None = None,
) -> CurvaHibrida:
    """
    Junta críticas, respostas, curva típica e conta numa curva só.

    ``conta`` já traz o horário e os dias de funcionamento quando é negócio:
    o ajuste usa isso para pôr o dia fechado na carga de base. ``rotulos`` leva
    o nome do equipamento do catálogo ao nome do cartão que o cliente tocou.
    """
    avisos: list[str] = []
    # Perfil sintético na base: o próprio ajuste avisa, e o aviso vem junto abaixo.
    perfil = perfil_tipico(perfil_id)

    # 1. a conta, na forma típica — o tamanho do dia de operação vem daqui
    ajuste_tipico = ajustar(conta, perfil)
    curva_tipica = np.asarray(ajuste_tipico.curva_operacao_kw, dtype=float)
    dia_kwh = float(curva_tipica.sum())

    # 2. a forma das respostas e as críticas, pelo Monte Carlo do motor
    ens_respostas = simular_ensemble(cenario.para_comodos(), cenario.instancias_de(),
                                     num_simulacoes=simulacoes, semente=semente)
    curva_respostas = _horaria_kw(ens_respostas)
    criticas = _so_criticas(cenario)
    if criticas is not None:
        ens_critico = simular_ensemble(criticas.para_comodos(), criticas.instancias_de(),
                                       num_simulacoes=simulacoes, semente=semente + 1)
        curva_critica = _horaria_kw(ens_critico)
    else:
        ens_critico = None
        curva_critica = np.zeros(24)
        avisos.append("Nenhuma carga marcada como crítica (MC ou C): o quadro de backup "
                      "fica vazio até a conversa com o cliente.")

    # 3. a mistura das formas, pelo quanto a lista explica a conta
    cobertura = float(curva_respostas.sum()) / dia_kwh if dia_kwh > 0 else 0.0
    peso = float(np.clip(cobertura, PESO_MINIMO, PESO_MAXIMO))
    forma = (peso * curva_respostas / max(curva_respostas.sum(), 1e-9)
             + (1 - peso) * curva_tipica / max(curva_tipica.sum(), 1e-9))
    if cobertura > 1.15:
        avisos.append(f"Os equipamentos marcados somam {cobertura:.0%} do consumo da conta: "
                      "as potências típicas estão altas para esta casa, ou a conta é de um mês fraco. "
                      "Os itens não críticos foram reduzidos na proporção.")
    elif cobertura < 0.3:
        avisos.append(f"Os equipamentos marcados explicam só {cobertura:.0%} da conta: "
                      "a maior parte do consumo vem de cargas que o cliente não listou.")

    # 4. críticas por dentro do total: o resto é o que sobra da conta
    total = forma * dia_kwh
    critico_dia = float(curva_critica.sum())
    if critico_dia >= dia_kwh:
        avisos.append("As cargas críticas declaradas somam mais que a conta inteira: "
                      "confira as quantidades e as potências antes de dimensionar.")
        resto = np.full(24, 0.05 * dia_kwh / 24)
    else:
        resto = np.clip(total - curva_critica, 0.0, None)
        if resto.sum() <= 0:
            resto = np.ones(24)
        resto *= (dia_kwh - critico_dia) / resto.sum()
    curva_total = curva_critica + resto

    # O ajuste guarda a conferência contra a fatura com a forma nova.
    ajuste = ajuste_tipico.com_curva_editada(curva_total, manter_energia=True)
    avisos.extend(a for a in ajuste.avisos if a not in avisos and "editada pelo operador" not in a)
    escala = float(ajuste.curva_operacao_kw.sum()) / max(curva_total.sum(), 1e-9)
    resto_kw_min = np.repeat(resto * escala, 60) * 1000.0            # W por minuto

    # 5. o ensemble total: críticas simuladas + resto com o fator do dia
    rng = np.random.default_rng(semente + 7)
    perfis: dict[str, np.ndarray] = {}
    for estacao in ESTACOES_ANO:
        fator_sazonal = float(ajuste.fatores_sazonais.get(estacao, 1.0))
        if ens_critico is not None and estacao in ens_critico.perfis_w:
            crit = np.asarray(ens_critico.perfis(estacao), dtype=float) * escala
        else:
            crit = np.zeros((simulacoes, 1440))
        n = crit.shape[0]
        fatores = _lognormal_media_um(rng, dispersao_diaria, (n, 1))
        perfis[estacao] = crit + resto_kw_min[None, :] * fatores * fator_sazonal
    ensemble_total = EnsembleCarga(perfis_w=perfis, passo_min=1, metadados={
        "origem": "curva_hibrida",
        "aviso": ("Curva total híbrida: cargas críticas simuladas item a item, dentro da curva "
                  f"característica '{perfil.nome}' ajustada à conta, com a forma corrigida pelas "
                  f"respostas do cliente (peso {peso:.0%})."),
    })

    # 6. a tabela de cargas: tudo que foi marcado, e o que a conta tem a mais
    itens = _energia_por_item(cenario, max(40, simulacoes // 4), semente + 100)
    dias_mes = (conta.dias_uteis_operacao + conta.dias_fim_de_semana_operacao) or 30
    itens["energia_kwh_mes"] = itens["energia_kwh_dia"] * dias_mes
    alvo = float(conta.consumo_mensal_kwh)
    soma = float(itens["energia_kwh_mes"].sum())
    if soma > alvo and soma > 0:
        criticos = itens["criticidade"].isin(CRITICAS)
        e_crit = float(itens.loc[criticos, "energia_kwh_mes"].sum())
        e_resto = soma - e_crit
        if e_resto > 0:
            fator = max(alvo - e_crit, 0.0) / e_resto
            itens.loc[~criticos, "energia_kwh_mes"] *= fator
    demais = max(alvo - float(itens["energia_kwh_mes"].sum()), 0.0)
    itens["origem"] = "questionario"
    cargas = pd.concat([itens, pd.DataFrame([{
        "comodo": "Geral", "equipamento": DEMAIS, "criticidade": "NC", "quantidade": 1.0,
        "potencia_w": 0.0, "energia_kwh_dia": demais / dias_mes, "energia_kwh_mes": demais, "origem": "conta",
    }])], ignore_index=True)
    rotulos = rotulos or {}
    cargas["rotulo"] = cargas["equipamento"].map(lambda n: rotulos.get(n) or _rotulo(n))
    cargas.loc[cargas["equipamento"] == DEMAIS, "rotulo"] = "Demais cargas"
    cargas["fracao"] = cargas["energia_kwh_mes"] / alvo

    return CurvaHibrida(
        perfil=perfil, ajuste=ajuste, ensemble_total=ensemble_total, ensemble_critico=ens_critico,
        curva_total_kw=np.asarray(ajuste.curva_operacao_kw, dtype=float), curva_critica_kw=curva_critica * escala,
        curva_respostas_kw=curva_respostas, curva_tipica_kw=curva_tipica,
        peso_respostas=peso, cobertura_respostas=cobertura, cargas=cargas, avisos=avisos,
    )
