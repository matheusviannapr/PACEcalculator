"""
Catálogo de baterias e inversores híbridos (``BDBaterias.xlsx``).

Segue o mesmo contrato do :mod:`aurum.pv.equipment`: a planilha é a fonte da
verdade, o código lê e ignora linha incompleta em vez de estourar. Duas
diferenças, e as duas importam para o estudo:

**Potência de pico não é um detalhe de catálogo — é a restrição que decide.**
Um inversor de 5 kW não aciona a partida de um motor de 3 kW pelo valor
nominal; aciona (ou não) pela sobrecarga que sustenta por alguns segundos. Por
isso ``potencia_ca_pico_kw`` e ``duracao_pico_s`` são campos de primeira
classe, e é contra eles que a curva de excedência instantânea é comparada em
:mod:`aurum.bateria.excedencia`.

**Bateria tem quatro limites, não um.** Energia nominal, profundidade de
descarga, taxa de descarga (C-rate) e eficiência de ida e volta. Dimensionar
por energia nominal — o erro comum — superestima a autonomia em 20% a 40%
dependendo da química, porque a energia utilizável é o produto dos quatro.

A semente do catálogo vem em :data:`SEMENTE_CATALOGO`, com os campos marcados
``fonte_dado="a conferir"``: são ordens de grandeza de mercado para o estudo
rodar hoje, **não** datasheets conferidos. Corrija com os documentos dos seus
fornecedores antes de assinar qualquer coisa.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from ..config import get_settings

LOGGER = logging.getLogger(__name__)

ABA_BATERIAS = "baterias"
ABA_INVERSORES = "inversores_hibridos"
ABA_LEIAME = "leia-me"

#: Nome padrão da planilha, ao lado da BDFotovoltaica.
NOME_PADRAO = "BDBaterias.xlsx"

__all__ = [
    "Bateria",
    "BaseBaterias",
    "ConjuntoArmazenamento",
    "InversorHibrido",
    "CatalogoError",
    "carregar_catalogo",
    "criar_planilha_modelo",
    "semear_planilha",
]


class CatalogoError(RuntimeError):
    """Planilha de baterias ausente, vazia ou sem as colunas obrigatórias."""


def _num(valor: Any) -> float | None:
    """Converte para float aceitando vírgula decimal; None quando não numérico."""
    if valor is None:
        return None
    if isinstance(valor, (int, float)):
        return None if (isinstance(valor, float) and math.isnan(valor)) else float(valor)
    texto = str(valor).strip().replace(",", ".")
    if not texto:
        return None
    try:
        return float(texto)
    except ValueError:
        return None


# ----------------------------------------------------------------------------
# Bateria
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Bateria:
    """Um módulo de bateria do catálogo."""

    modelo: str
    fabricante: str
    quimica: str
    capacidade_nominal_kwh: float
    tensao_nominal_v: float
    profundidade_descarga_percent: float
    eficiencia_roundtrip_percent: float
    potencia_carga_max_kw: float
    potencia_descarga_max_kw: float
    ciclos_vida: int
    retencao_fim_vida_percent: float = 80.0
    max_modulos_paralelo: int = 16
    preco_brl: float | None = None
    fonte_dado: str = "a conferir"

    @property
    def energia_util_kwh(self) -> float:
        """
        O que a bateria de fato entrega num ciclo completo.

        Nominal × profundidade de descarga × rendimento de descarga. O
        rendimento entra pela raiz do valor de ida e volta, porque metade da
        perda acontece na carga e metade na descarga; atribuir a perda inteira
        à descarga puniria o banco duas vezes no balanço do apagão.
        """
        return (
            self.capacidade_nominal_kwh
            * (self.profundidade_descarga_percent / 100.0)
            * self.eficiencia_descarga
        )

    @property
    def eficiencia_descarga(self) -> float:
        return math.sqrt(max(0.01, self.eficiencia_roundtrip_percent / 100.0))

    @property
    def eficiencia_carga(self) -> float:
        return self.eficiencia_descarga

    @property
    def taxa_c_descarga(self) -> float:
        """C-rate de descarga: potência máxima dividida pela energia nominal."""
        return self.potencia_descarga_max_kw / self.capacidade_nominal_kwh

    @property
    def energia_vitalicia_kwh(self) -> float:
        """Throughput até o fim de vida, medido na energia útil por ciclo."""
        return self.energia_util_kwh * self.ciclos_vida

    def __str__(self) -> str:
        return f"{self.fabricante} {self.modelo} ({self.capacidade_nominal_kwh:.2f} kWh {self.quimica})"


# ----------------------------------------------------------------------------
# Inversor híbrido
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class InversorHibrido:
    """Um inversor híbrido (rede + FV + bateria, com saída de backup)."""

    modelo: str
    fabricante: str
    potencia_ca_nominal_kw: float
    potencia_ca_pico_kw: float
    duracao_pico_s: float
    potencia_fv_max_kw: float
    potencia_carga_bateria_kw: float
    tensao_bateria_min_v: float
    tensao_bateria_max_v: float
    eficiencia_percent: float = 96.0
    fases: int = 1
    tempo_transferencia_ms: float = 10.0
    #: Tensões de linha que o inversor atende, como o datasheet as declara --
    #: "127/220", "220/380", "400/380", ou várias separadas por vírgula.
    #:
    #: Sem este campo o catálogo tratava todo inversor como se servisse a
    #: qualquer rede, e a maior parte dos híbridos trifásicos do mercado é de
    #: 380/400 V. Num prédio de 127/220 -- que é a rede de boa parte do país --
    #: o estudo recomendava equipamento que não liga.
    tensao_ca_v: str = "220/380"
    preco_brl: float | None = None
    fonte_dado: str = "a conferir"

    @property
    def razao_de_pico(self) -> float:
        """Quantas vezes o nominal o inversor sustenta no surto."""
        return self.potencia_ca_pico_kw / self.potencia_ca_nominal_kw

    @property
    def tensoes_de_linha_v(self) -> tuple[float, ...]:
        """
        As tensões **de linha** (fase-fase) que o modelo atende.

        O datasheet escreve "127/220" para dizer 127 V de fase e 220 V de
        linha. Quem dimensiona pensa em linha -- é o número da conta de luz e
        do projeto elétrico --, então é o maior de cada par que interessa.
        """
        tensoes: list[float] = []
        for grupo in str(self.tensao_ca_v).split(","):
            valores = [
                float(v) for v in grupo.replace("V", "").split("/")
                if v.strip().replace(".", "").isdigit()
            ]
            if valores:
                tensoes.append(max(valores))
        return tuple(sorted(set(tensoes)))

    def atende_rede(self, tensao_linha_v: float, tolerancia: float = 0.06) -> bool:
        """
        O inversor liga nessa rede?

        A tolerância de 6% cobre a diferença entre nominais vizinhos que são a
        mesma rede na prática -- 380, 400 e 415 V --, e não estica o suficiente
        para deixar um inversor de 380 passar por um de 220.
        """
        alvo = float(tensao_linha_v)
        return any(abs(v - alvo) <= tolerancia * alvo for v in self.tensoes_de_linha_v)

    def aceita_banco(self, bateria: Bateria, modulos: int = 1) -> bool:
        """
        A tensão do banco cabe na janela de bateria do inversor?

        Módulos em série somam tensão; a maioria dos bancos comerciais de 48 V
        é ligada em paralelo (tensão constante), e é isso que assumimos aqui —
        o campo ``max_modulos_paralelo`` da bateria é que limita a quantidade.
        """
        return self.tensao_bateria_min_v <= bateria.tensao_nominal_v <= self.tensao_bateria_max_v

    def __str__(self) -> str:
        return (
            f"{self.fabricante} {self.modelo} "
            f"({self.potencia_ca_nominal_kw:.1f} kW / {self.potencia_ca_pico_kw:.1f} kW pico)"
        )


# ----------------------------------------------------------------------------
# Conjunto: inversor + banco
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class ConjuntoArmazenamento:
    """
    Um inversor híbrido com um banco de N módulos iguais — o objeto que o
    despacho simula.

    Os limites efetivos são o mínimo entre o que a bateria aguenta e o que o
    inversor entrega, e é comum que o gargalo mude com o tamanho do banco: um
    módulo só costuma limitar pela bateria, quatro módulos passam a limitar
    pelo inversor. O estudo tem que enxergar essa troca, e por isso ela é
    calculada aqui em vez de assumida em qualquer dos dois lados.
    """

    inversor: InversorHibrido
    bateria: Bateria
    modulos: int = 1
    potencia_fv_kwp: float = 0.0

    def __post_init__(self) -> None:
        if self.modulos < 1:
            raise ValueError("o banco precisa de ao menos um módulo")
        if self.modulos > self.bateria.max_modulos_paralelo:
            raise ValueError(
                f"{self.bateria.modelo} admite no máximo "
                f"{self.bateria.max_modulos_paralelo} módulos em paralelo"
            )

    # -- energia -----------------------------------------------------------
    @property
    def capacidade_nominal_kwh(self) -> float:
        return self.bateria.capacidade_nominal_kwh * self.modulos

    @property
    def energia_util_kwh(self) -> float:
        return self.bateria.energia_util_kwh * self.modulos

    # -- potência ----------------------------------------------------------
    @property
    def potencia_descarga_kw(self) -> float:
        """Descarga contínua: o menor entre o C-rate do banco e o nominal do inversor."""
        return min(
            self.bateria.potencia_descarga_max_kw * self.modulos,
            self.inversor.potencia_ca_nominal_kw,
        )

    @property
    def potencia_pico_kw(self) -> float:
        """
        Surto: limitado pelo inversor, mas nunca acima do que o banco entrega.

        Baterias de lítio toleram sobrecarga curta bem acima do contínuo; a
        margem de 1,5× abaixo é conservadora e explícita para não esconder uma
        hipótese generosa dentro de um número redondo.
        """
        return min(
            self.inversor.potencia_ca_pico_kw,
            self.bateria.potencia_descarga_max_kw * self.modulos * 1.5,
        )

    @property
    def potencia_carga_kw(self) -> float:
        return min(
            self.bateria.potencia_carga_max_kw * self.modulos,
            self.inversor.potencia_carga_bateria_kw,
        )

    @property
    def potencia_fv_aproveitavel_kw(self) -> float:
        """FV que o inversor consegue processar, mesmo que haja mais no telhado."""
        return min(float(self.potencia_fv_kwp), self.inversor.potencia_fv_max_kw)

    @property
    def compativel(self) -> bool:
        return self.inversor.aceita_banco(self.bateria, self.modulos)

    @property
    def autonomia_nominal_h(self) -> float:
        """Horas à potência contínua — o número de folheto, útil só como âncora."""
        return self.energia_util_kwh / max(1e-6, self.potencia_descarga_kw)

    def descricao(self) -> str:
        return (
            f"{self.inversor.fabricante} {self.inversor.modelo} + {self.modulos}× "
            f"{self.bateria.fabricante} {self.bateria.modelo} "
            f"({self.energia_util_kwh:.1f} kWh úteis, {self.potencia_descarga_kw:.1f} kW)"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "inversor": str(self.inversor),
            "bateria": str(self.bateria),
            "modulos": self.modulos,
            "capacidade_nominal_kwh": self.capacidade_nominal_kwh,
            "energia_util_kwh": self.energia_util_kwh,
            "potencia_descarga_kw": self.potencia_descarga_kw,
            "potencia_pico_kw": self.potencia_pico_kw,
            "duracao_pico_s": self.inversor.duracao_pico_s,
            "potencia_carga_kw": self.potencia_carga_kw,
            "potencia_fv_kwp": self.potencia_fv_kwp,
            "autonomia_nominal_h": self.autonomia_nominal_h,
            "compativel": self.compativel,
        }


# ----------------------------------------------------------------------------
# Base
# ----------------------------------------------------------------------------
@dataclass
class BaseBaterias:
    """Catálogo carregado, com acesso por modelo."""

    baterias: list[Bateria]
    inversores: list[InversorHibrido]
    caminho: Path | None = None

    @property
    def vazia(self) -> bool:
        return not self.baterias or not self.inversores

    def bateria_por_modelo(self, modelo: str) -> Bateria | None:
        alvo = str(modelo).strip().lower()
        return next((b for b in self.baterias if b.modelo.lower() == alvo), None)

    def inversor_por_modelo(self, modelo: str) -> InversorHibrido | None:
        alvo = str(modelo).strip().lower()
        return next((i for i in self.inversores if i.modelo.lower() == alvo), None)

    def inversores_ordenados(self) -> list[InversorHibrido]:
        return sorted(self.inversores, key=lambda i: i.potencia_ca_nominal_kw)

    def combinacoes(
        self,
        potencia_fv_kwp: float = 0.0,
        modulos: Sequence[int] = (1, 2, 3, 4, 6, 8),
        energia_util_min_kwh: float = 0.0,
        energia_util_max_kwh: float = float("inf"),
    ) -> list[ConjuntoArmazenamento]:
        """
        Todos os conjuntos inversor × bateria × nº de módulos que fecham.

        Descarta o que a tensão não permite e o que cai fora da faixa de energia
        pedida — sem isso a varredura devolve centenas de arranjos que ninguém
        cotaria, e a tabela de resultado deixa de ser legível.
        """
        conjuntos: list[ConjuntoArmazenamento] = []
        for inversor in self.inversores:
            for bateria in self.baterias:
                if not inversor.aceita_banco(bateria):
                    continue
                for n in modulos:
                    if n > bateria.max_modulos_paralelo:
                        continue
                    conjunto = ConjuntoArmazenamento(inversor, bateria, int(n), potencia_fv_kwp)
                    if energia_util_min_kwh <= conjunto.energia_util_kwh <= energia_util_max_kwh:
                        conjuntos.append(conjunto)
        return conjuntos

    def resumo(self) -> dict[str, Any]:
        return {
            "baterias": len(self.baterias),
            "inversores": len(self.inversores),
            "a_conferir": sum(
                1 for x in [*self.baterias, *self.inversores] if "conferir" in str(x.fonte_dado).lower()
            ),
            "energia_modulo_min_kwh": min((b.capacidade_nominal_kwh for b in self.baterias), default=0.0),
            "energia_modulo_max_kwh": max((b.capacidade_nominal_kwh for b in self.baterias), default=0.0),
            "inversor_min_kw": min((i.potencia_ca_nominal_kw for i in self.inversores), default=0.0),
            "inversor_max_kw": max((i.potencia_ca_nominal_kw for i in self.inversores), default=0.0),
        }


# ----------------------------------------------------------------------------
# Leitura
# ----------------------------------------------------------------------------
def _ler_baterias(df: pd.DataFrame) -> list[Bateria]:
    itens: list[Bateria] = []
    for _, linha in df.iterrows():
        capacidade = _num(linha.get("capacidade_nominal_kwh"))
        tensao = _num(linha.get("tensao_nominal_v"))
        if not (capacidade and capacidade > 0 and tensao and tensao > 0):
            LOGGER.debug("Bateria ignorada por dados incompletos: %s", linha.get("modelo"))
            continue
        dod = _num(linha.get("profundidade_descarga_percent")) or 90.0
        eficiencia = _num(linha.get("eficiencia_roundtrip_percent")) or 90.0
        # Sem C-rate declarado, 0,5C é o padrão conservador de banco estacionário.
        descarga = _num(linha.get("potencia_descarga_max_kw")) or capacidade * 0.5
        itens.append(
            Bateria(
                modelo=str(linha.get("modelo") or "sem modelo").strip(),
                fabricante=str(linha.get("fabricante") or "sem fabricante").strip(),
                quimica=str(linha.get("quimica") or "LFP").strip(),
                capacidade_nominal_kwh=capacidade,
                tensao_nominal_v=tensao,
                profundidade_descarga_percent=min(100.0, max(10.0, dod)),
                eficiencia_roundtrip_percent=min(100.0, max(50.0, eficiencia)),
                potencia_carga_max_kw=_num(linha.get("potencia_carga_max_kw")) or descarga,
                potencia_descarga_max_kw=descarga,
                ciclos_vida=int(_num(linha.get("ciclos_vida")) or 4000),
                retencao_fim_vida_percent=_num(linha.get("retencao_fim_vida_percent")) or 80.0,
                max_modulos_paralelo=int(_num(linha.get("max_modulos_paralelo")) or 16),
                preco_brl=_num(linha.get("preco_brl")),
                fonte_dado=str(linha.get("fonte_dado") or "a conferir").strip(),
            )
        )
    return itens


def _ler_inversores(df: pd.DataFrame) -> list[InversorHibrido]:
    itens: list[InversorHibrido] = []
    for _, linha in df.iterrows():
        nominal = _num(linha.get("potencia_ca_nominal_kw"))
        if not (nominal and nominal > 0):
            LOGGER.debug("Inversor híbrido ignorado por dados incompletos: %s", linha.get("modelo"))
            continue
        # Sem pico declarado, 1,5× o nominal por 10 s é a hipótese mais comum
        # entre híbridos de lítio -- e fica registrada como hipótese.
        pico = _num(linha.get("potencia_ca_pico_kw")) or nominal * 1.5
        # Zero é um valor legítimo aqui: inversor sem MPPT próprio (Victron
        # MultiPlus, por exemplo) não processa FV nenhum. `or` engoliria o zero.
        fv_max = _num(linha.get("potencia_fv_max_kw"))
        itens.append(
            InversorHibrido(
                modelo=str(linha.get("modelo") or "sem modelo").strip(),
                fabricante=str(linha.get("fabricante") or "sem fabricante").strip(),
                potencia_ca_nominal_kw=nominal,
                potencia_ca_pico_kw=max(pico, nominal),
                duracao_pico_s=_num(linha.get("duracao_pico_s")) or 10.0,
                potencia_fv_max_kw=nominal * 1.3 if fv_max is None else fv_max,
                potencia_carga_bateria_kw=_num(linha.get("potencia_carga_bateria_kw")) or nominal,
                tensao_bateria_min_v=_num(linha.get("tensao_bateria_min_v")) or 40.0,
                tensao_bateria_max_v=_num(linha.get("tensao_bateria_max_v")) or 60.0,
                eficiencia_percent=_num(linha.get("eficiencia_percent")) or 96.0,
                fases=int(_num(linha.get("fases")) or 1),
                tempo_transferencia_ms=_num(linha.get("tempo_transferencia_ms")) or 10.0,
                # Planilha antiga não tem a coluna. O padrão é 220/380, que é
                # o que a maioria dos híbridos trifásicos atende -- mas quem
                # tem rede de 127/220 precisa preencher, senão o estudo
                # recomenda equipamento que não liga na instalação.
                tensao_ca_v=str(linha.get("tensao_ca_v") or "220/380").strip(),
                preco_brl=_num(linha.get("preco_brl")),
                fonte_dado=str(linha.get("fonte_dado") or "a conferir").strip(),
            )
        )
    return itens


def caminho_padrao() -> Path:
    return Path(get_settings().equipment_xlsx).parent / NOME_PADRAO


def carregar_catalogo(
    caminho: str | Path | None = None,
    criar_se_ausente: bool = True,
) -> BaseBaterias:
    """
    Lê ``BDBaterias.xlsx``; cria a planilha semente na primeira execução.

    Criar em vez de falhar é deliberado: sem catálogo o estudo não roda, e
    exigir que o usuário monte uma planilha do zero antes de ver o programa
    funcionar uma vez é o tipo de barreira que faz a ferramenta não ser usada.
    O arquivo gerado vem com todos os campos marcados ``a conferir``.
    """
    destino = Path(caminho) if caminho else caminho_padrao()
    if not destino.exists():
        if not criar_se_ausente:
            raise CatalogoError(f"Catálogo de baterias não encontrado: {destino}")
        LOGGER.info("Catálogo ausente; gerando planilha semente em %s", destino)
        criar_planilha_modelo(destino)

    try:
        planilhas = pd.read_excel(destino, sheet_name=None)
    except Exception as exc:  # noqa: BLE001 - qualquer falha de leitura vira erro de catálogo
        raise CatalogoError(f"Não foi possível ler {destino}: {exc}") from exc

    if ABA_BATERIAS not in planilhas or ABA_INVERSORES not in planilhas:
        raise CatalogoError(
            f"{destino} precisa das abas '{ABA_BATERIAS}' e '{ABA_INVERSORES}'"
        )

    base = BaseBaterias(
        baterias=_ler_baterias(planilhas[ABA_BATERIAS]),
        inversores=_ler_inversores(planilhas[ABA_INVERSORES]),
        caminho=destino,
    )
    if base.vazia:
        raise CatalogoError(f"{destino} não tem nenhuma bateria ou inversor válido")
    return base


# ----------------------------------------------------------------------------
# Semente do catálogo
# ----------------------------------------------------------------------------
#: Ordens de grandeza de mercado, **não** datasheets conferidos. Cada linha sai
#: marcada "a conferir" na planilha para que ninguém confunda as duas coisas.
#: Fontes consultadas para as linhas conferidas.
FONTE_LYNX_G2 = ("GoodWe Lynx F G2 — en.goodwe.com/Ftp/EN/Downloads/Datasheet/"
                 "GW_Lynx-F-G2_Datasheet-EN.pdf")
FONTE_LYNX_FPLUS = ("GoodWe Lynx Home F Plus+ — en.goodwe.com/Ftp/EN/Downloads/Datasheet/"
                    "GW_Lynx Home F PLUS+ Series (HV)_Datasheet-EN.pdf")
FONTE_ET = ("GoodWe ET 15-30kW — en.goodwe.com/Ftp/EN/Downloads/Datasheet/"
            "GW_ET 15-30kW_Datasheet-EN.pdf")
FONTE_ES_LD = ("GoodWe ES LD 5-10kW — br.goodwe.com/Ftp/Downloads/Datasheet/PT/"
               "GW_ES-LD_Datasheet-PT.pdf")
FONTE_ETR = ("GoodWe ETR 50-125kW — br.goodwe.com/Ftp/Downloads/Datasheet/PT/"
             "GW_ETR_Datasheet-PT.pdf")
FONTE_ETC = ("GoodWe ETC 50kW — br.goodwe.com/Ftp/Downloads/Datasheet/PT/"
             "GW_ETC_Datasheet-PT.pdf")
FONTE_ET_LV = ("GoodWe ET LV 12-20kW — goodwe.com.au/Ftp/Downloads/Datasheet/PT/"
               "GW_ET-LV_Datasheet-PT.pdf")
FONTE_LYNX_U = ("GoodWe Lynx Home U 5.4-32.4kWh — br.goodwe.com/Ftp/Downloads/Datasheet/PT/"
                "GW_Lynx Home U Series (LV)_5.4-20_Datasheet-PT.pdf")
FONTE_ET_PLUS = ("GoodWe ET PLUS+ 5-10kW — en.goodwe.com/Ftp/EN/Downloads/Datasheet/"
                 "GW_ET PLUS+_Datasheet-EN.pdf")

#: Ressalva comum às baterias GoodWe: o datasheet publica **energia utilizável**
#: medida a 100% de profundidade de descarga, então a nominal e a útil coincidem
#: na planilha. Vida em ciclos e rendimento de ida e volta não estão nesse
#: documento — ficam declarados como a conferir no campo de cada linha.
_LYNX_RESSALVA = " | energia utilizável a 100% DOD; ciclos e ηRT a conferir"


def _lynx_g2(modelo: str, kwh: float, tensao: float, kw: float, nota: str = "") -> dict[str, Any]:
    """
    Uma configuração da torre Lynx F G2.

    Cada linha é uma **torre montada**, não um módulo solto: os módulos de
    3,2 kWh se empilham em **série**, e é a quantidade deles que fixa a tensão
    do banco. O modelo de conjunto deste pacote trata `modulos` como paralelo,
    de tensão constante — então representar cada torre como um item próprio é o
    que mantém a tensão certa na verificação de compatibilidade com o inversor.
    """
    return dict(
        fabricante="GoodWe", modelo=modelo, quimica="LFP",
        capacidade_nominal_kwh=kwh, tensao_nominal_v=tensao,
        profundidade_descarga_percent=100.0, eficiencia_roundtrip_percent=95.0,
        potencia_carga_max_kw=kw, potencia_descarga_max_kw=kw,
        ciclos_vida=6000, retencao_fim_vida_percent=80.0,
        # Torres em paralelo: o ET admite mais de uma, mas o número exato
        # depende do modelo de inversor. Dois é o piso seguro.
        max_modulos_paralelo=2, preco_brl=None,
        fonte_dado=FONTE_LYNX_G2 + _LYNX_RESSALVA + nota,
    )


def _lynx_u(modelo: str, kwh: float, kw: float, nota: str = "") -> dict[str, Any]:
    """
    Lynx Home U: a torre de **48 V**, par das linhas ES/EM/SBP e ET LV.

    Cada módulo é uma unidade autônoma com BMS próprio -- não há unidade de
    controle separada, o que muda a conta de um banco pequeno: o primeiro
    módulo já é um sistema completo.

    A corrente nominal salta de 50 A num módulo para 100 A a partir de dois,
    então a potência do banco **não** cresce linearmente com a energia: 5,4 kWh
    entregam 2,56 kW e 10,8 kWh entregam 5,12 kW, mas 32,4 kWh também entregam
    5,12 kW. É o caso clássico em que mais kWh não compra mais kW, e o estudo
    precisa saber disso para não prometer partida de motor que o banco não dá.
    """
    return dict(
        fabricante="GoodWe", modelo=modelo, quimica="LFP",
        capacidade_nominal_kwh=kwh, tensao_nominal_v=51.2,
        # Energia utilizável é a declarada; o datasheet mede a 100% de DoD.
        profundidade_descarga_percent=100.0, eficiencia_roundtrip_percent=95.0,
        potencia_carga_max_kw=kw, potencia_descarga_max_kw=kw,
        ciclos_vida=6000, retencao_fim_vida_percent=80.0, max_modulos_paralelo=6,
        preco_brl=None,
        fonte_dado=FONTE_LYNX_U + _LYNX_RESSALVA + nota,
    )


def _lynx_fplus(modelo: str, kwh: float, tensao: float, kw: float) -> dict[str, Any]:
    return dict(
        fabricante="GoodWe", modelo=modelo, quimica="LFP",
        capacidade_nominal_kwh=kwh, tensao_nominal_v=tensao,
        profundidade_descarga_percent=100.0, eficiencia_roundtrip_percent=95.0,
        potencia_carga_max_kw=kw, potencia_descarga_max_kw=kw,
        ciclos_vida=6000, retencao_fim_vida_percent=80.0,
        max_modulos_paralelo=2, preco_brl=None,
        fonte_dado=FONTE_LYNX_FPLUS + _LYNX_RESSALVA,
    )


SEMENTE_BATERIAS: list[dict[str, Any]] = [
    # -- GoodWe Lynx F G2 (módulo LX F3.2-20: 64 V, 3,2 kWh) ---------------
    _lynx_g2("LX F6.4-H-20", 6.4, 128.0, 4.48,
             " | o datasheet avisa que a tensão de partida desta torre não "
             "acende os inversores ET de 15-30 kW"),
    _lynx_g2("LX F9.6-H-20", 9.6, 192.0, 6.72),
    _lynx_g2("LX F12.8-H-20", 12.8, 256.0, 8.96),
    _lynx_g2("LX F16.0-H-20", 16.0, 320.0, 11.20),
    _lynx_g2("LX F19.2-H-20", 19.2, 384.0, 13.44),
    _lynx_g2("LX F22.4-H-20", 22.4, 448.0, 15.68),
    _lynx_g2("LX F25.6-H-20", 25.6, 512.0, 17.92),
    _lynx_g2("LX F28.8-H-20", 28.8, 576.0, 20.16),
    # -- GoodWe Lynx Home F Plus+ (módulo LX F3.3-H: 102,4 V, 3,27 kWh) ----
    _lynx_fplus("LX F6.6-H", 6.55, 204.8, 5.12),
    _lynx_fplus("LX F9.8-H", 9.83, 307.2, 7.68),
    _lynx_fplus("LX F13.1-H", 13.10, 409.6, 10.24),
    _lynx_fplus("LX F16.4-H", 16.38, 512.0, 12.80),
    # -- GoodWe Lynx Home U, 48 V ------------------------------------------
    # O par das linhas ES/EM/SBP e ET LV. Note a potência: ela para de crescer
    # no segundo módulo, e é isso que o estudo precisa enxergar.
    _lynx_u("LX U5.4-20", 5.40, 2.56, " | módulo único: 50 A nominais"),
    _lynx_u("LX U5.4-20 ×2", 10.80, 5.12),
    _lynx_u("LX U5.4-20 ×3", 16.20, 5.12),
    _lynx_u("LX U5.4-20 ×4", 21.60, 5.12),
    _lynx_u("LX U5.4-20 ×5", 27.00, 5.12),
    _lynx_u("LX U5.4-20 ×6", 32.40, 5.12,
            " | máximo da linha: 6 módulos em paralelo"),
    # -- Demais marcas: ordens de grandeza de mercado ----------------------
    dict(fabricante="Pylontech", modelo="US3000C", quimica="LFP", capacidade_nominal_kwh=3.55,
         tensao_nominal_v=48.0, profundidade_descarga_percent=95.0, eficiencia_roundtrip_percent=95.0,
         potencia_carga_max_kw=1.78, potencia_descarga_max_kw=1.78, ciclos_vida=6000,
         retencao_fim_vida_percent=80.0, max_modulos_paralelo=16, preco_brl=9500.0),
    dict(fabricante="Pylontech", modelo="US5000", quimica="LFP", capacidade_nominal_kwh=4.80,
         tensao_nominal_v=48.0, profundidade_descarga_percent=95.0, eficiencia_roundtrip_percent=95.0,
         potencia_carga_max_kw=2.40, potencia_descarga_max_kw=2.40, ciclos_vida=6000,
         retencao_fim_vida_percent=80.0, max_modulos_paralelo=16, preco_brl=12500.0),
    dict(fabricante="Deye", modelo="SE-G5.1Pro", quimica="LFP", capacidade_nominal_kwh=5.12,
         tensao_nominal_v=51.2, profundidade_descarga_percent=95.0, eficiencia_roundtrip_percent=95.0,
         potencia_carga_max_kw=5.12, potencia_descarga_max_kw=5.12, ciclos_vida=6000,
         retencao_fim_vida_percent=80.0, max_modulos_paralelo=16, preco_brl=13500.0),
    dict(fabricante="Growatt", modelo="ARK 2.5H-A1", quimica="LFP", capacidade_nominal_kwh=2.56,
         tensao_nominal_v=51.2, profundidade_descarga_percent=90.0, eficiencia_roundtrip_percent=94.0,
         potencia_carga_max_kw=1.28, potencia_descarga_max_kw=1.28, ciclos_vida=6000,
         retencao_fim_vida_percent=80.0, max_modulos_paralelo=12, preco_brl=8200.0),
    dict(fabricante="Dyness", modelo="Tower T10", quimica="LFP", capacidade_nominal_kwh=10.66,
         tensao_nominal_v=51.2, profundidade_descarga_percent=95.0, eficiencia_roundtrip_percent=95.0,
         potencia_carga_max_kw=5.33, potencia_descarga_max_kw=5.33, ciclos_vida=6000,
         retencao_fim_vida_percent=80.0, max_modulos_paralelo=8, preco_brl=26000.0),
    dict(fabricante="Livoltek", modelo="BLF-B512", quimica="LFP", capacidade_nominal_kwh=5.12,
         tensao_nominal_v=51.2, profundidade_descarga_percent=95.0, eficiencia_roundtrip_percent=95.0,
         potencia_carga_max_kw=2.56, potencia_descarga_max_kw=2.56, ciclos_vida=6000,
         retencao_fim_vida_percent=80.0, max_modulos_paralelo=10, preco_brl=13000.0),
    dict(fabricante="BYD", modelo="Battery-Box Premium HVS 5.1", quimica="LFP",
         capacidade_nominal_kwh=5.12, tensao_nominal_v=204.0, profundidade_descarga_percent=96.0,
         eficiencia_roundtrip_percent=96.0, potencia_carga_max_kw=5.12, potencia_descarga_max_kw=5.12,
         ciclos_vida=6000, retencao_fim_vida_percent=80.0, max_modulos_paralelo=5, preco_brl=17000.0),
    dict(fabricante="BYD", modelo="Battery-Box Premium HVM 8.3", quimica="LFP",
         capacidade_nominal_kwh=8.28, tensao_nominal_v=256.0, profundidade_descarga_percent=96.0,
         eficiencia_roundtrip_percent=96.0, potencia_carga_max_kw=8.28, potencia_descarga_max_kw=8.28,
         ciclos_vida=6000, retencao_fim_vida_percent=80.0, max_modulos_paralelo=5, preco_brl=26000.0),
    # Chumbo-ácido entra de propósito: é o contraste que mostra por que a
    # energia útil, e não a nominal, é a grandeza que dimensiona.
    dict(fabricante="Moura", modelo="Clean Nano 220Ah (banco 48V)", quimica="Chumbo-ácido",
         capacidade_nominal_kwh=10.56, tensao_nominal_v=48.0, profundidade_descarga_percent=50.0,
         eficiencia_roundtrip_percent=80.0, potencia_carga_max_kw=1.06, potencia_descarga_max_kw=2.11,
         ciclos_vida=1200, retencao_fim_vida_percent=80.0, max_modulos_paralelo=4, preco_brl=14000.0),
]

def _et(modelo: str, kw: float, pico_kw: float, dur_s: float, fv_kw: float,
        carga_kw: float, nota: str = "", v_min: float = 200.0, v_max: float = 800.0,
        fonte: str | None = None) -> dict[str, Any]:
    """Um híbrido da série ET. Pico e duração vêm tabelados — coisa rara."""
    return dict(
        fabricante="GoodWe", modelo=modelo,
        potencia_ca_nominal_kw=kw, potencia_ca_pico_kw=pico_kw, duracao_pico_s=dur_s,
        potencia_fv_max_kw=fv_kw, potencia_carga_bateria_kw=carga_kw,
        tensao_bateria_min_v=v_min, tensao_bateria_max_v=v_max,
        eficiencia_percent=98.0, fases=3, tempo_transferencia_ms=10.0,
        tensao_ca_v="220/380, 230/400", preco_brl=None,
        fonte_dado=(fonte or FONTE_ET) + nota,
    )


def _et_plus(modelo: str, kw: float, pico_kw: float, fv_kw: float, carga_kw: float) -> dict[str, Any]:
    """
    ET PLUS+ 5-10 kW: a faixa que cobre a maioria das instalações.

    A sobrecarga de backup aqui é de **60 segundos**, e não de 10 como na faixa
    grande — o que muda o veredito para carga de partida longa. E a janela de
    bateria começa em 180 V, mais baixa que a dos ET de 15-30 kW: torres Lynx
    que não acendem o inversor grande acendem este.
    """
    return _et(modelo, kw, pico_kw, 60.0, fv_kw, carga_kw,
               v_min=180.0, v_max=600.0, fonte=FONTE_ET_PLUS)


def _et_lv(modelo: str, kw: float, pico_kva: float, fv_kw: float, carga_kw: float,
           mppts: int, tensao_ca: str = "220/380, 230/400", fases: int = 3,
           nota: str = "") -> dict[str, Any]:
    """
    Linha ET LV: híbrido de **bateria de 48 V**, que é outra família.

    Os ET e ET PLUS+ já no catálogo trabalham com banco de alta tensão (180 a
    800 V); estes trabalham com 48 V nominais (40 a 60 V), e por isso aceitam
    as torres Lynx U / Lynx A e as baterias de 48 V de qualquer marca — que é
    o parque instalado brasileiro. Misturar as duas famílias no dimensionamento
    produz um conjunto que não liga.

    Dois deles merecem atenção:

    * ``GW12K-ET-LL-G10`` é **220 V trifásico**, a tensão da maior parte da
      rede de distribuição brasileira fora do Sudeste. Sem ele, o catálogo só
      oferecia 380/400 V para quem tem 220.
    * a sobrecarga de backup é de **2x a nominal por 10 s** — 24 kVA num
      inversor de 12 kW. É o dobro do que a linha ET de alta tensão entrega, e
      muda o veredito de partida de motor.
    """
    return dict(
        fabricante="GoodWe", modelo=modelo,
        potencia_ca_nominal_kw=kw, potencia_ca_pico_kw=pico_kva, duracao_pico_s=10.0,
        potencia_fv_max_kw=fv_kw, potencia_carga_bateria_kw=carga_kw,
        # A janela de 48 V é o que separa esta linha das outras duas.
        tensao_bateria_min_v=40.0, tensao_bateria_max_v=60.0,
        eficiencia_percent=97.8, fases=fases, tempo_transferencia_ms=4.0,
        tensao_ca_v=tensao_ca, preco_brl=None,
        fonte_dado=FONTE_ET_LV + f" | {mppts} MPPT" + nota,
    )


def _es_ld(modelo: str, kw: float, pico_kva: float, fv_kw: float,
           carga_kw: float, desc_kw: float) -> dict[str, Any]:
    """
    Linha ES LD: bifásico de **127/220 V**, bateria de 48 V.

    É a linha que resolve o caso mais comum do país e que faltava por inteiro
    no catálogo: rede 127/220, duas fases e neutro, banco de 48 V. Os híbridos
    trifásicos de 380/400 V que dominavam o catálogo simplesmente não ligam
    nessa rede.

    A sobrecarga de backup é de **2x a nominal por 10 s** fora da rede — e,
    conectado à rede, o barramento de backup passa 16 kVA em qualquer um dos
    três modelos, porque aí quem entrega é a rede. É a distinção que o
    datasheet faz e que muda o veredito de partida de motor: com a rede
    presente, o quadro aguenta; no apagão, aguenta o dobro do nominal por
    10 segundos.
    """
    return dict(
        fabricante="GoodWe", modelo=modelo,
        potencia_ca_nominal_kw=kw, potencia_ca_pico_kw=pico_kva, duracao_pico_s=10.0,
        potencia_fv_max_kw=fv_kw, potencia_carga_bateria_kw=carga_kw,
        tensao_bateria_min_v=40.0, tensao_bateria_max_v=60.0,
        eficiencia_percent=97.6, fases=2, tempo_transferencia_ms=10.0,
        tensao_ca_v="120/208, 127/220, 120/240, 127/254", preco_brl=None,
        fonte_dado=FONTE_ES_LD + f" | descarga máx. {desc_kw:g} kW",
    )


def _etr(modelo: str, kw: float, fv_kw: float, v_bat_min: float, v_bat_max: float,
         tensao_ca: str, mppts: int) -> dict[str, Any]:
    """
    Linha ETR: o C&I de 50 a 125 kW, e o único que atende 127/220 nessa faixa.

    A sobrecarga aqui é modesta -- **120% por 60 s** -- e é assim mesmo: num
    inversor de 125 kW, 20% de folga já são 25 kW, e o que dimensiona esta
    faixa é energia e demanda contratada, não partida de motor.

    Os dois modelos com sufixo ``-L-`` são de rede **127/220 V**; os demais,
    de 220/380 a 240/415. Misturar os dois grupos numa proposta é recomendar
    equipamento que não liga na instalação.
    """
    return dict(
        fabricante="GoodWe", modelo=modelo,
        potencia_ca_nominal_kw=kw, potencia_ca_pico_kw=kw * 1.2, duracao_pico_s=60.0,
        potencia_fv_max_kw=fv_kw,
        potencia_carga_bateria_kw=kw,
        tensao_bateria_min_v=v_bat_min, tensao_bateria_max_v=v_bat_max,
        eficiencia_percent=98.5, fases=3, tempo_transferencia_ms=10.0,
        tensao_ca_v=tensao_ca, preco_brl=None,
        fonte_dado=FONTE_ETR + f" | {mppts} MPPT | sobrecarga 120% por 60 s",
    )


SEMENTE_INVERSORES: list[dict[str, Any]] = [
    # -- GoodWe ET 15-30 kW, trifásico -------------------------------------
    # A sobrecarga de backup destes vem **tabelada com a duração**, o que é
    # incomum: quase todo datasheet publica só um número de pico sem dizer por
    # quanto tempo. Sem a duração, não dá para saber se o inversor cobre a
    # partida de um motor ou só o instante inicial dela.
    _et("GW15K-ET", 15.0, 18.0, 10.0, 22.5, 15.0,
        " | o datasheet também traz 24 kVA por 3 s; aqui usamos o valor de 10 s,"
        " que é o que sustenta uma partida de motor"),
    _et("GW20K-ET", 20.0, 24.0, 10.0, 30.0, 20.0,
        " | também 32 kVA por 3 s"),
    _et("GW25K-ET", 25.0, 30.0, 60.0, 37.5, 25.0),
    _et("GW29.9K-ET", 29.9, 36.0, 60.0, 45.0, 30.0),
    _et("GW30K-ET", 30.0, 36.0, 60.0, 45.0, 30.0),
    # -- GoodWe ES LD 5-10 kW, BIFÁSICO 127/220 V --------------------------
    # A linha que faltava para a rede mais comum do país. Bateria de 48 V,
    # duas fases e neutro, e o dobro do nominal por 10 s no apagão.
    _es_ld("GW5K-ES-LD-G10", 5.0, 10.0, 10.0, 5.0, 5.5),
    _es_ld("GW7.5K-ES-LD-G10", 7.5, 15.0, 15.0, 7.5, 8.2),
    _es_ld("GW10K-ES-LD-G10", 10.0, 20.0, 20.0, 10.0, 11.0),
    # -- GoodWe ETR 50-125 kW, comercial e industrial ----------------------
    # Os dois primeiros são 127/220 V: é a única opção do catálogo inteiro
    # para prédio grande em rede de baixa tensão.
    _etr("GW50K-ETR-L-G10", 50.0, 100.0, 440.0, 700.0, "127/220", 8),
    _etr("GW75K-ETR-L-G10", 75.0, 150.0, 440.0, 700.0, "127/220", 10),
    _etr("GW75K-ETR-G10", 75.0, 150.0, 700.0, 1000.0, "220/380, 230/400, 240/415", 8),
    _etr("GW100K-ETR-G10", 100.0, 200.0, 700.0, 1000.0, "220/380, 230/400, 240/415", 10),
    _etr("GW110K-ETR-G10", 110.0, 220.0, 700.0, 1000.0, "220/380, 230/400, 240/415", 10),
    _etr("GW125K-ETR-G10", 125.0, 250.0, 700.0, 1000.0, "220/380, 230/400, 240/415", 10),
    # -- GoodWe ETC 50 kW, MPPT modular ------------------------------------
    dict(fabricante="GoodWe", modelo="GW50K-ETC",
         potencia_ca_nominal_kw=50.0, potencia_ca_pico_kw=55.0, duracao_pico_s=60.0,
         potencia_fv_max_kw=65.0, potencia_carga_bateria_kw=50.0,
         tensao_bateria_min_v=200.0, tensao_bateria_max_v=865.0,
         eficiencia_percent=98.0, fases=3, tempo_transferencia_ms=10.0,
         tensao_ca_v="230/400", preco_brl=None,
         fonte_dado=FONTE_ETC + " | faixa de bateria de 200 a 865 V, a mais larga do catálogo"),
    # -- GoodWe ET LV 12-20 kW, bateria de 48 V ----------------------------
    # Entram porque o catálogo só tinha híbrido de alta tensão, e o parque
    # brasileiro de baterias de 48 V é grande. O de 220 V trifásico é o único
    # do catálogo inteiro que atende essa rede.
    _et_lv("GW12K-ET-L-G10", 12.0, 24.0, 24.0, 12.0, 3),
    _et_lv("GW15K-ET-L-G10", 15.0, 30.0, 30.0, 15.0, 4),
    _et_lv("GW20K-ET-L-G10", 20.0, 40.0, 40.0, 20.0, 4),
    _et_lv("GW12K-ET-LL-G10", 12.0, 24.0, 24.0, 12.0, 3, tensao_ca="127/220",
           nota=" | 127/220 V trifásico (3L/N/PE), a rede da maior parte do país"),
    # -- GoodWe ET PLUS+ 5-10 kW, trifásico --------------------------------
    _et_plus("GW5K-ET", 5.0, 10.0, 7.5, 7.5),
    _et_plus("GW6.5K-ET", 6.5, 13.0, 9.7, 8.45),
    _et_plus("GW8K-ET", 8.0, 16.0, 12.0, 9.6),
    _et_plus("GW10K-ET", 10.0, 16.5, 15.0, 10.0),
    # -- Demais marcas: ordens de grandeza de mercado ----------------------
    dict(fabricante="Deye", modelo="SUN-5K-SG03LP1", potencia_ca_nominal_kw=5.0,
         potencia_ca_pico_kw=10.0, duracao_pico_s=10.0, potencia_fv_max_kw=6.5,
         potencia_carga_bateria_kw=5.0, tensao_bateria_min_v=40.0, tensao_bateria_max_v=60.0,
         eficiencia_percent=97.0, fases=1, tempo_transferencia_ms=10.0, preco_brl=9500.0),
    dict(fabricante="Deye", modelo="SUN-8K-SG04LP3", potencia_ca_nominal_kw=8.0,
         potencia_ca_pico_kw=12.0, duracao_pico_s=10.0, potencia_fv_max_kw=10.4,
         potencia_carga_bateria_kw=8.0, tensao_bateria_min_v=40.0, tensao_bateria_max_v=60.0,
         eficiencia_percent=97.5, fases=3, tempo_transferencia_ms=10.0, preco_brl=15500.0),
    dict(fabricante="Deye", modelo="SUN-12K-SG04LP3", potencia_ca_nominal_kw=12.0,
         potencia_ca_pico_kw=18.0, duracao_pico_s=10.0, potencia_fv_max_kw=15.6,
         potencia_carga_bateria_kw=12.0, tensao_bateria_min_v=40.0, tensao_bateria_max_v=60.0,
         eficiencia_percent=97.5, fases=3, tempo_transferencia_ms=10.0, preco_brl=21000.0),
    dict(fabricante="Growatt", modelo="SPH 5000TL BL-UP", potencia_ca_nominal_kw=5.0,
         potencia_ca_pico_kw=7.5, duracao_pico_s=10.0, potencia_fv_max_kw=6.5,
         potencia_carga_bateria_kw=5.0, tensao_bateria_min_v=40.0, tensao_bateria_max_v=60.0,
         eficiencia_percent=97.0, fases=1, tempo_transferencia_ms=20.0, preco_brl=11000.0),
    dict(fabricante="Victron", modelo="MultiPlus-II 48/5000/70", potencia_ca_nominal_kw=4.0,
         potencia_ca_pico_kw=9.0, duracao_pico_s=3.0, potencia_fv_max_kw=0.0,
         potencia_carga_bateria_kw=3.4, tensao_bateria_min_v=38.0, tensao_bateria_max_v=66.0,
         eficiencia_percent=95.0, fases=1, tempo_transferencia_ms=20.0, preco_brl=18000.0),
    dict(fabricante="Victron", modelo="Quattro 48/10000/140", potencia_ca_nominal_kw=8.0,
         potencia_ca_pico_kw=20.0, duracao_pico_s=3.0, potencia_fv_max_kw=0.0,
         potencia_carga_bateria_kw=6.7, tensao_bateria_min_v=38.0, tensao_bateria_max_v=66.0,
         eficiencia_percent=96.0, fases=1, tempo_transferencia_ms=20.0, preco_brl=34000.0),
    dict(fabricante="Sungrow", modelo="SH10RT", potencia_ca_nominal_kw=10.0,
         potencia_ca_pico_kw=11.0, duracao_pico_s=60.0, potencia_fv_max_kw=15.0,
         potencia_carga_bateria_kw=10.0, tensao_bateria_min_v=150.0, tensao_bateria_max_v=560.0,
         eficiencia_percent=98.0, fases=3, tempo_transferencia_ms=20.0, preco_brl=19000.0),
    dict(fabricante="Solis", modelo="RHI-5K-48ES-5G", potencia_ca_nominal_kw=5.0,
         potencia_ca_pico_kw=6.0, duracao_pico_s=10.0, potencia_fv_max_kw=6.5,
         potencia_carga_bateria_kw=5.0, tensao_bateria_min_v=42.0, tensao_bateria_max_v=58.0,
         eficiencia_percent=97.0, fases=1, tempo_transferencia_ms=20.0, preco_brl=10500.0),
]

_LEIAME = [
    ("Aba", "Para que serve"),
    ("baterias", "Um módulo por linha. Vários módulos iguais em paralelo formam o banco."),
    ("inversores_hibridos", "Inversor com entrada FV, entrada de bateria e saída de backup."),
    ("", ""),
    ("Campo", "O que significa e por que importa"),
    ("capacidade_nominal_kwh", "Energia de placa. NÃO é a energia utilizável."),
    ("profundidade_descarga_percent", "Fração da nominal que o BMS libera. LFP ~95%, chumbo ~50%."),
    ("eficiencia_roundtrip_percent", "Ida e volta. O código toma a raiz para separar carga de descarga."),
    ("potencia_descarga_max_kw", "C-rate × kWh. Limita a potência mesmo com banco cheio."),
    ("potencia_carga_max_kw", "Quanto o módulo aceita por vez — decide a recarga durante o apagão."),
    ("ciclos_vida", "Ciclos até a retenção de fim de vida. Alimenta o modelo de degradação."),
    ("retencao_fim_vida_percent", "Capacidade restante no fim da vida útil declarada. Tipicamente 80%."),
    ("max_modulos_paralelo", "Limite do fabricante para módulos em paralelo no mesmo banco."),
    ("potencia_ca_nominal_kw", "Potência contínua da saída de backup."),
    ("potencia_ca_pico_kw", "Sobrecarga sustentada por duracao_pico_s. É o que decide partida de motor."),
    ("duracao_pico_s", "Por quanto tempo o pico é sustentado antes do desarme."),
    ("potencia_fv_max_kw", "FV que o inversor processa. Zero = inversor sem MPPT (usa carregador externo)."),
    ("potencia_carga_bateria_kw", "Teto do carregador interno."),
    ("tensao_bateria_min_v / max_v", "Janela de tensão do banco. Filtra combinações impossíveis."),
    ("preco_brl", "Preço de referência. Só entra na análise econômica; pode ficar em branco."),
    ("fonte_dado", "Deixe 'a conferir' até checar o datasheet. O relatório sinaliza o que não foi conferido."),
    ("", ""),
    ("AVISO", "Os valores que vieram na planilha semente são ordens de grandeza de mercado,"),
    ("", "não datasheets conferidos. Substitua pelos dados dos seus fornecedores"),
    ("", "antes de usar este estudo em proposta comercial."),
]


def _com_fonte(linha: dict[str, Any]) -> dict[str, Any]:
    """Linha sem procedência declarada vira 'a conferir' — nunca o contrário."""
    return dict(linha, fonte_dado=linha.get("fonte_dado") or "a conferir")


def semear_planilha(
    caminho: str | Path | None = None,
    substituir: bool = False,
) -> tuple[Path, dict[str, int]]:
    """
    Acrescenta os modelos da semente a um ``BDBaterias.xlsx`` já existente.

    Acrescenta, não sobrescreve: o que está na planilha pode ter sido corrigido
    à mão com o datasheet do lote comprado, e apagar isso destruiria trabalho.
    ``substituir=True`` inverte a regra.
    """
    destino = Path(caminho) if caminho else caminho_padrao()
    if not destino.exists():
        return criar_planilha_modelo(destino), {ABA_BATERIAS: len(SEMENTE_BATERIAS),
                                                ABA_INVERSORES: len(SEMENTE_INVERSORES)}

    planilhas = pd.read_excel(destino, sheet_name=None)
    resumo: dict[str, int] = {}
    saida = dict(planilhas)
    for aba, semente in ((ABA_BATERIAS, SEMENTE_BATERIAS), (ABA_INVERSORES, SEMENTE_INVERSORES)):
        atual = planilhas.get(aba, pd.DataFrame())
        novos = pd.DataFrame([_com_fonte(linha) for linha in semente])
        if atual.empty:
            saida[aba], resumo[aba] = novos, len(novos)
            continue
        chaves = novos["modelo"].astype(str).str.strip().str.lower()
        conhecidos = set(atual["modelo"].astype(str).str.strip().str.lower())
        if substituir:
            mantidos = atual[~atual["modelo"].astype(str).str.strip().str.lower().isin(set(chaves))]
            saida[aba], resumo[aba] = pd.concat([mantidos, novos], ignore_index=True), len(novos)
        else:
            faltantes = novos[~chaves.isin(conhecidos)]
            saida[aba] = pd.concat([atual, faltantes], ignore_index=True)
            resumo[aba] = len(faltantes)
        saida[aba]["fonte_dado"] = saida[aba].get("fonte_dado", pd.Series(dtype=str)).fillna("a conferir")

    with pd.ExcelWriter(destino, engine="openpyxl") as escritor:
        for aba, tabela in saida.items():
            tabela.to_excel(escritor, sheet_name=str(aba)[:31], index=False)
    return destino, resumo


def criar_planilha_modelo(caminho: str | Path | None = None) -> Path:
    """Grava ``BDBaterias.xlsx`` com a semente do catálogo e a aba de instruções."""
    destino = Path(caminho) if caminho else caminho_padrao()
    destino.parent.mkdir(parents=True, exist_ok=True)

    baterias = pd.DataFrame([_com_fonte(linha) for linha in SEMENTE_BATERIAS])
    inversores = pd.DataFrame([_com_fonte(linha) for linha in SEMENTE_INVERSORES])
    leiame = pd.DataFrame(_LEIAME[1:], columns=list(_LEIAME[0]))

    with pd.ExcelWriter(destino, engine="openpyxl") as escritor:
        leiame.to_excel(escritor, sheet_name=ABA_LEIAME, index=False)
        baterias.to_excel(escritor, sheet_name=ABA_BATERIAS, index=False)
        inversores.to_excel(escritor, sheet_name=ABA_INVERSORES, index=False)
    return destino
