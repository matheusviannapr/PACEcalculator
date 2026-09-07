"""
Despacho do conjunto durante o apagão — o motor do estudo.

Simula, passo a passo e para milhares de cenários de uma vez, o que acontece
quando a rede cai: a carga pede, o sol dá o que tem, a bateria cobre a
diferença enquanto puder, e o inversor limita tudo isso por cima.

Três limites separados, que falham de jeitos diferentes e são registrados
como coisas diferentes:

* **Energia** — o banco esvazia. Falha previsível, cresce com a duração do
  apagão, resolve-se com mais kWh.
* **Potência contínua** — a carga instantânea passa do que o inversor entrega
  em regime. O banco pode estar cheio e a luz cair do mesmo jeito. Resolve-se
  com inversor maior ou com corte seletivo de carga, nunca com mais bateria.
* **Surto** — a partida de um motor pede, por segundos, mais que a sobrecarga
  que o inversor sustenta. É o modo de falha que o dimensionamento por energia
  média não enxerga, e é o que faz o compressor não partir num apagão com a
  bateria em 90%.

Distinguir os três é o ponto do módulo. Um relatório que só diz "a bateria não
aguentou" manda comprar mais kWh; se a falha era de potência, mais kWh não
resolve nada e o cliente descobre isso no apagão seguinte.

Convenção de estado: ``soc`` é energia **no lado CC** (kWh armazenados). A
conversão para o lado CA acontece na fronteira, pela eficiência de descarga —
misturar os dois lados é o erro que faz a autonomia sair 5% a 10% otimista.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..precos import PRECOS
from .catalogo import ConjuntoArmazenamento

__all__ = ["Gerador", "LimitesDespacho", "ResultadoDespacho", "simular_ilhamento"]

#: Códigos de causa da primeira falha.
CAUSA_OK = 0
CAUSA_POTENCIA = 1
CAUSA_ENERGIA = 2
NOME_CAUSA = {CAUSA_OK: "atendido", CAUSA_POTENCIA: "potência", CAUSA_ENERGIA: "energia"}

#: Abaixo desta fração da capacidade útil, uma falha é creditada à energia e
#: não à potência. Não é zero porque o último passo de descarga quase nunca
#: zera exatamente o banco.
_LIMIAR_BANCO_VAZIO = 0.02


@dataclass(frozen=True)
class Gerador:
    """
    Grupo gerador: potência limitada, energia praticamente ilimitada.

    É o oposto exato da bateria, e por isso as duas se complementam em vez de
    competir. A bateria tem energia contada em kWh e potência de sobra por
    alguns segundos; o gerador tem potência de placa e combustível enquanto
    houver tanque. Num apagão de 36 h a bateria acaba e o gerador não; na
    partida de um motor o gerador afunda a tensão e a bateria não.

    Dois parâmetros mandam no resultado:

    * ``potencia_kw`` é o teto duro. Carga acima disso não é atendida, por
      mais horas de combustível que haja -- é o mesmo modo de falha que
      reprova um inversor pequeno. :meth:`para_carga` dimensiona o grupo para
      que esse teto não morda, que é o caso normal: quem instala um gerador
      espera que ele supra, e a pergunta que sobra é quanto custa rodar.
    * ``atraso_partida_min`` é o tempo entre a queda da rede e o grupo assumir
      a carga. Num grupo com QTA são 10 a 30 segundos; o que acontece nesse
      intervalo depende de haver ou não bateria, e por isso é simulado em vez
      de ignorado.

    ``custo_energia_brl_kwh`` reúne combustível, lubrificante e manutenção por
    kWh gerado -- é o número que entra na conta, e o que separa um gerador de
    emergência (roda 20 h/ano, custo irrelevante) de um gerador que trabalha
    todo dia, onde o custo domina o payback.
    """

    #: Zero é o pedido de dimensionamento: o estudo substitui pelo grupo
    #: comercial que supre o pico da carga essencial (:meth:`para_carga`).
    potencia_kw: float
    #: Combustível, lubrificante e manutenção por kWh gerado. O padrão vem de
    #: :mod:`aurum.precos`, calculado do consumo específico e do preço do
    #: diesel -- não de um número lembrado de cor.
    custo_energia_brl_kwh: float = PRECOS["gerador_diesel_brl_kwh"].tipico
    capex_brl: float = 0.0
    opex_fixo_brl_ano: float = 0.0
    atraso_partida_min: float = 0.5
    modelo: str = "Grupo gerador"
    #: Permite ao grupo recarregar o banco com a folga de potência -- o que um
    #: inversor híbrido com entrada de gerador faz, e o que torna a dupla
    #: gerador + bateria capaz de atravessar apagões de dias.
    recarrega_bateria: bool = True

    #: Potências comerciais de grupo gerador em kVA, a 0,8 de fator de
    #: potência. Arredondar para cima até uma delas evita recomendar um grupo
    #: de 137 kW, que ninguém fabrica.
    _COMERCIAIS_KVA = (
        # A faixa pequena existe e é a que atende residência: um quadro de
        # backup de 2 kW não pede um grupo de 30 kVA. Sem estas linhas, o
        # dimensionamento automático saltava para 24 kW num pico de 0,7 kW.
        4, 6, 8, 10, 12, 15, 18, 20, 25,
        30, 40, 50, 60, 75, 90, 110, 130, 150, 180, 200, 230, 260, 300, 350,
        400, 450, 500, 600, 700, 800, 900, 1000, 1250, 1500, 1800, 2000,
    )

    @classmethod
    def para_carga(
        cls,
        potencia_exigida_kw: float,
        folga: float = 1.15,
        **campos: Any,
    ) -> "Gerador":
        """
        Um grupo dimensionado para suprir a carga, e não para ser um teto.

        Grupo gerador não é bateria: quem instala um espera que a luz fique
        acesa, não que ela fique acesa até acabar alguma coisa. Dimensioná-lo
        pelo pico da carga essencial tira a potência da discussão e deixa a
        pergunta que interessa -- quanto custa rodar -- sozinha em cena.

        A folga cobre a partida de motores e a perda de potência com altitude
        e temperatura; 15% é conservador para carga predial e insuficiente
        para carga com muitos motores de partida direta, que pede o dobro.
        """
        alvo_kva = float(potencia_exigida_kw) * float(folga) / 0.8
        escolhida = next(
            (kva for kva in cls._COMERCIAIS_KVA if kva >= alvo_kva),
            cls._COMERCIAIS_KVA[-1],
        )
        return cls(potencia_kw=escolhida * 0.8, **campos)

    def custo_da_energia_brl(self, energia_kwh: float) -> float:
        """O que custa gerar essa energia — combustível, óleo e manutenção."""
        return max(0.0, float(energia_kwh)) * float(self.custo_energia_brl_kwh)

    def descricao(self) -> str:
        return f"{self.modelo} de {self.potencia_kw:g} kW"

    def as_dict(self) -> dict[str, Any]:
        return {
            "modelo": self.modelo,
            "potencia_kw": self.potencia_kw,
            "custo_energia_brl_kwh": self.custo_energia_brl_kwh,
            "capex_brl": self.capex_brl,
            "opex_fixo_brl_ano": self.opex_fixo_brl_ano,
            "atraso_partida_min": self.atraso_partida_min,
        }


@dataclass(frozen=True)
class LimitesDespacho:
    """Os limites efetivos do conjunto, já resolvidos entre bateria e inversor."""

    energia_util_dc_kwh: float
    potencia_descarga_kw: float
    potencia_pico_kw: float
    duracao_pico_s: float
    potencia_carga_kw: float
    potencia_fv_kw: float
    eficiencia_descarga: float
    eficiencia_carga: float

    @classmethod
    def do_conjunto(cls, conjunto: ConjuntoArmazenamento) -> "LimitesDespacho":
        bateria = conjunto.bateria
        return cls(
            energia_util_dc_kwh=(
                bateria.capacidade_nominal_kwh
                * (bateria.profundidade_descarga_percent / 100.0)
                * conjunto.modulos
            ),
            potencia_descarga_kw=conjunto.potencia_descarga_kw,
            potencia_pico_kw=conjunto.potencia_pico_kw,
            duracao_pico_s=conjunto.inversor.duracao_pico_s,
            potencia_carga_kw=conjunto.potencia_carga_kw,
            potencia_fv_kw=conjunto.potencia_fv_aproveitavel_kw,
            eficiencia_descarga=bateria.eficiencia_descarga,
            eficiencia_carga=bateria.eficiencia_carga,
        )

    @classmethod
    def sem_bateria(cls, potencia_fv_kw: float = 0.0) -> "LimitesDespacho":
        """
        Os limites de uma instalação que não tem banco nenhum.

        Serve aos cenários sem bateria da comparação de fontes. O FV entra
        zerado de propósito: inversor conectado à rede desliga quando a rede
        cai (anti-ilhamento, ABNT NBR 16149), então sol sem bateria não é
        fonte de backup nenhuma. Quem quiser representar um inversor híbrido
        ilhando com FV precisa do banco, que é o que dá a referência de tensão.
        """
        return cls(
            energia_util_dc_kwh=0.0,
            potencia_descarga_kw=0.0,
            potencia_pico_kw=0.0,
            duracao_pico_s=0.0,
            potencia_carga_kw=0.0,
            potencia_fv_kw=float(potencia_fv_kw),
            eficiencia_descarga=1.0,
            eficiencia_carga=1.0,
        )

    @property
    def energia_util_ca_kwh(self) -> float:
        """Quanto o banco cheio entrega na tomada, já descontada a conversão."""
        return self.energia_util_dc_kwh * self.eficiencia_descarga


@dataclass(frozen=True)
class ResultadoDespacho:
    """
    Um registro por cenário simulado. Todos os vetores têm comprimento ``S``.

    ``ens_kwh`` é a *energia não suprida*: o que a carga pediu e não recebeu.
    É a medida certa de severidade — "falhou" é binário e trata igual o apagão
    que perdeu 0,2 kWh no último minuto e o que perdeu 40 kWh desde a segunda
    hora.
    """

    atendido: np.ndarray
    causa: np.ndarray
    minutos_ate_falha: np.ndarray
    ens_kwh: np.ndarray
    energia_demandada_kwh: np.ndarray
    energia_bateria_kwh: np.ndarray
    energia_pv_kwh: np.ndarray
    energia_gerador_kwh: np.ndarray
    #: Parte do ``ens_kwh`` gasta no intervalo entre a queda da rede e o grupo
    #: gerador assumir a carga. É energia que faltou de verdade, mas não é
    #: falha do arranjo: é a característica declarada de um grupo com QTA.
    ens_transferencia_kwh: np.ndarray
    energia_recarregada_kwh: np.ndarray
    energia_desperdicada_kwh: np.ndarray
    soc_final_frac: np.ndarray
    soc_min_frac: np.ndarray
    eventos_desarme: np.ndarray
    passo_min: int
    limites: LimitesDespacho
    trajetoria_soc_frac: np.ndarray | None = None

    @property
    def n(self) -> int:
        return int(self.atendido.size)

    @property
    def prob_atendimento(self) -> float:
        """Fração dos cenários atravessados sem nenhum minuto de interrupção."""
        return float(self.atendido.mean())

    @property
    def fracao_energia_atendida(self) -> float:
        total = float(self.energia_demandada_kwh.sum())
        return 1.0 - float(self.ens_kwh.sum()) / total if total > 0 else 1.0

    def resumo(self) -> dict[str, Any]:
        falhas = ~self.atendido
        return {
            "cenarios": self.n,
            "prob_atendimento": self.prob_atendimento,
            "fracao_energia_atendida": self.fracao_energia_atendida,
            "ens_medio_kwh": float(self.ens_kwh.mean()),
            "ens_p95_kwh": float(np.percentile(self.ens_kwh, 95)),
            "ens_max_kwh": float(self.ens_kwh.max()),
            "minutos_ate_falha_mediano": (
                float(np.median(self.minutos_ate_falha[falhas])) if falhas.any() else float("nan")
            ),
            "minutos_ate_falha_p05": (
                float(np.percentile(self.minutos_ate_falha[falhas], 5)) if falhas.any() else float("nan")
            ),
            "falhas_por_potencia": float((self.causa == CAUSA_POTENCIA).mean()),
            "falhas_por_energia": float((self.causa == CAUSA_ENERGIA).mean()),
            "desarmes_por_surto_medio": float(self.eventos_desarme.mean()),
            "cenarios_com_desarme": float((self.eventos_desarme > 0).mean()),
            "soc_final_medio": float(self.soc_final_frac.mean()),
            "soc_min_medio": float(self.soc_min_frac.mean()),
            "energia_pv_media_kwh": float(self.energia_pv_kwh.mean()),
            "energia_gerador_media_kwh": float(self.energia_gerador_kwh.mean()),
            "ens_transferencia_media_kwh": float(self.ens_transferencia_kwh.mean()),
            "energia_recarregada_media_kwh": float(self.energia_recarregada_kwh.mean()),
        }


def simular_ilhamento(
    carga_kw: np.ndarray,
    carga_pico_kw: np.ndarray,
    geracao_kw: np.ndarray,
    limites: LimitesDespacho,
    passo_min: int,
    soc_inicial_frac: float = 1.0,
    guardar_trajetoria: bool = False,
    gerador: Gerador | None = None,
) -> ResultadoDespacho:
    """
    Roda o ilhamento para ``S`` cenários de ``T`` passos, todos de uma vez.

    ``carga_kw`` é a média do passo (fecha a energia) e ``carga_pico_kw`` é a
    máxima dentro do passo (dispara o teste de potência). Passar a mesma matriz
    nos dois é legítimo quando o passo é de um minuto.

    A prioridade de despacho é a do inversor híbrido real: o FV atende a carga
    primeiro, a bateria cobre o que faltar, e só o excedente sobre a carga vai
    para o banco. Não há exportação — a rede caiu —, então o que sobra depois
    de encher a bateria é cortado, e esse corte é contabilizado: num apagão
    longo de fim de semana ele costuma ser grande e é o argumento para um banco
    maior mesmo quando a autonomia já fecha.

    Com ``gerador``, o grupo entra em paralelo ao inversor e cobre o que sol e
    bateria não deram: os picos acima do que o inversor sustenta e, num apagão
    longo, tudo depois que o banco esvazia. É a ordem de um inversor híbrido
    com entrada de gerador -- energia grátis primeiro, combustível por último
    -- e não a de um grupo que assume o quadro inteiro do primeiro minuto. O
    grupo só existe a partir do atraso de partida; antes disso, o que segura o
    quadro é a bateria, se houver — e a energia perdida nessa transferência
    entra no ENS, separada em ``ens_transferencia_kwh``, sem reprovar o
    cenário. Um grupo com QTA leva de 10 a 30 s para assumir: isso é dado de
    catálogo, não defeito do arranjo.
    """
    carga = np.atleast_2d(np.asarray(carga_kw, dtype=float))
    pico = np.atleast_2d(np.asarray(carga_pico_kw, dtype=float))
    sol = np.atleast_2d(np.asarray(geracao_kw, dtype=float))
    if not (carga.shape == pico.shape == sol.shape):
        raise ValueError("carga, pico e geração precisam ter a mesma forma (S, T)")

    n_cenarios, n_passos = carga.shape
    dt_h = passo_min / 60.0
    dt_s = passo_min * 60.0

    soc = np.full(n_cenarios, float(soc_inicial_frac) * limites.energia_util_dc_kwh)
    credito_s = np.full(n_cenarios, float(limites.duracao_pico_s))

    ens = np.zeros(n_cenarios)
    energia_bateria = np.zeros(n_cenarios)
    energia_pv_usada = np.zeros(n_cenarios)
    energia_gerador = np.zeros(n_cenarios)
    ens_transferencia = np.zeros(n_cenarios)
    energia_recarga = np.zeros(n_cenarios)
    energia_cortada = np.zeros(n_cenarios)
    desarmes = np.zeros(n_cenarios)
    passo_falha = np.full(n_cenarios, -1, dtype=int)
    causa = np.zeros(n_cenarios, dtype=int)
    soc_min = soc.copy()
    trajetoria = np.empty((n_cenarios, n_passos)) if guardar_trajetoria else None

    capacidade = max(1e-9, limites.energia_util_dc_kwh)
    fv_teto = limites.potencia_fv_kw
    ger_kw = float(gerador.potencia_kw) if gerador is not None else 0.0
    ger_atraso_min = float(gerador.atraso_partida_min) if gerador is not None else 0.0
    ger_recarrega = gerador is not None and gerador.recarrega_bateria

    for t in range(n_passos):
        pv = np.minimum(sol[:, t], fv_teto)
        demanda = carga[:, t]
        demanda_pico = pico[:, t]
        # Fração do passo em que o grupo já assumiu a carga. Arredondar a
        # partida para o passo inteiro transformava 30 s de transferência em
        # 5 minutos de apagão — e, como qualquer déficit reprova o cenário,
        # zerava a autonomia de um grupo capaz de suprir a carga inteira.
        coberto = (
            float(np.clip(((t + 1) * passo_min - ger_atraso_min) / passo_min, 0.0, 1.0))
            if gerador is not None
            else 0.0
        )
        ger_disponivel = ger_kw * coberto
        transferindo = gerador is not None and coberto < 1.0

        # -- teto de potência do inversor neste passo -----------------------
        acima_do_continuo = demanda_pico > limites.potencia_descarga_kw
        com_credito = credito_s > 0.0
        teto = np.where(
            acima_do_continuo & com_credito, limites.potencia_pico_kw, limites.potencia_descarga_kw
        )
        # O crédito de surto se gasta enquanto a solicitação está acima do
        # contínuo e se restaura assim que ela volta para baixo — é assim que a
        # proteção térmica de um inversor real se comporta.
        credito_s = np.where(acima_do_continuo, np.maximum(0.0, credito_s - dt_s), limites.duracao_pico_s)
        # Com o grupo em paralelo, o que pode desarmar o inversor é o pico que
        # sobra depois dele.
        desarmes += (demanda_pico - ger_disponivel > limites.potencia_pico_kw).astype(float)

        # -- quanto o conjunto consegue entregar ----------------------------
        # O teto já é o limite de potência do passo: contínuo, ou de surto
        # quando há crédito. A bateria acompanha esse teto, e não o contínuo —
        # limitá-la ao contínuo aqui anularia o surto, que é justamente o que
        # faz o compressor partir. O quanto o banco tolera acima do contínuo já
        # está embutido em `potencia_pico_kw` pelo catálogo.
        descarga_por_energia = soc * limites.eficiencia_descarga / dt_h
        descarga_max = np.minimum(teto, descarga_por_energia)
        entrega_max = np.minimum(teto, pv + descarga_max)

        do_inversor = np.minimum(demanda, entrega_max)
        do_gerador = np.minimum(np.maximum(0.0, demanda - do_inversor), ger_disponivel)
        atendida = do_inversor + do_gerador
        deficit = demanda - atendida

        # -- movimenta o banco ---------------------------------------------
        pv_para_carga = np.minimum(pv, do_inversor)
        da_bateria = np.maximum(0.0, do_inversor - pv)
        soc -= da_bateria * dt_h / limites.eficiencia_descarga
        energia_bateria += da_bateria * dt_h
        energia_pv_usada += pv_para_carga * dt_h
        energia_gerador += do_gerador * dt_h

        sobra_pv = np.maximum(0.0, pv - pv_para_carga)
        espaco_kw = np.maximum(0.0, capacidade - soc) / dt_h / limites.eficiencia_carga
        recarga = np.minimum(np.minimum(sobra_pv, limites.potencia_carga_kw), espaco_kw)
        recarga_ger = np.zeros_like(recarga)
        if ger_recarrega:
            # A folga de potência do grupo enche o banco: é o que permite ao
            # conjunto atravessar um apagão de dias sem que a bateria seja
            # dimensionada para a energia toda.
            folga_grupo = np.maximum(0.0, ger_disponivel - do_gerador)
            folga_carga = np.maximum(0.0, limites.potencia_carga_kw - recarga)
            recarga_ger = np.minimum(
                np.minimum(folga_grupo, folga_carga), np.maximum(0.0, espaco_kw - recarga)
            )
            energia_gerador += recarga_ger * dt_h
        soc += (recarga + recarga_ger) * dt_h * limites.eficiencia_carga
        energia_recarga += (recarga + recarga_ger) * dt_h
        energia_cortada += (sobra_pv - recarga) * dt_h

        soc = np.clip(soc, 0.0, capacidade)
        soc_min = np.minimum(soc_min, soc)
        if trajetoria is not None:
            trajetoria[:, t] = soc / capacidade

        # -- registra a primeira falha -------------------------------------
        houve_falha = deficit > 1e-9
        ens += deficit * dt_h
        if transferindo:
            # A transferência conta como energia não suprida — ela faltou — mas
            # não marca falha do arranjo: um grupo com QTA leva de 10 a 30 s
            # para assumir, isso é dado de catálogo, e reprovar o conjunto por
            # causa disso responderia "o gerador não serve" a quem perguntou
            # "quanto custa rodar o gerador".
            ens_transferencia += deficit * dt_h
        primeira = houve_falha & (passo_falha < 0) & (not transferindo)
        if primeira.any():
            passo_falha = np.where(primeira, t, passo_falha)
            # Sem banco nenhum, a falha nunca é de energia: o que faltou foi
            # potência de gerador ou de sol, e chamar isso de "bateria vazia"
            # mandaria comprar kWh para um problema de kW.
            banco_vazio = (soc <= _LIMIAR_BANCO_VAZIO * capacidade) & (
                limites.energia_util_dc_kwh > 0
            )
            causa = np.where(primeira, np.where(banco_vazio, CAUSA_ENERGIA, CAUSA_POTENCIA), causa)

    atendido = passo_falha < 0
    minutos = np.where(atendido, n_passos * passo_min, passo_falha * passo_min).astype(float)

    return ResultadoDespacho(
        atendido=atendido,
        causa=causa,
        minutos_ate_falha=minutos,
        ens_kwh=ens,
        energia_demandada_kwh=carga.sum(axis=1) * dt_h,
        energia_bateria_kwh=energia_bateria,
        energia_pv_kwh=energia_pv_usada,
        energia_gerador_kwh=energia_gerador,
        ens_transferencia_kwh=ens_transferencia,
        energia_recarregada_kwh=energia_recarga,
        energia_desperdicada_kwh=energia_cortada,
        soc_final_frac=soc / capacidade,
        soc_min_frac=soc_min / capacidade,
        eventos_desarme=desarmes,
        passo_min=int(passo_min),
        limites=limites,
        trajetoria_soc_frac=trajetoria,
    )
