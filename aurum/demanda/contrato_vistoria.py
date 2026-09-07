"""
O contrato entre a vistoria técnica e o estudo de energia.

Dois programas trocam um arquivo, e a qualidade do estudo é limitada pelo que
esse arquivo carrega. Este módulo escreve o contrato de forma executável: o
que é **obrigatório** para o estudo rodar, o que é **desejável** e melhora o
resultado quando existe, e o que a vistoria já coleta e o estudo ainda joga
fora.

Existe para que a conversa entre as duas equipes seja sobre campos, e não
sobre impressões. Rodar :func:`conferir` num backup responde em segundos se
ele serve, o que falta, e quanto cada falta custa em precisão.

**A regra de ouro:** o backup em JSON é a fonte. As duas planilhas exportadas
ao lado dele são derivadas e perdem informação — a do simulador descarta
cômodo e criticidade, que é o que define o quadro de backup, e o inventário
sai com o texto corrompido porque é escrito em UTF-8 e reaberto em Latin-1.
Quem consome deve ler o JSON; quem produz deve garantir que ele saia completo.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable

__all__ = [
    "CAMPOS",
    "Campo",
    "Gravidade",
    "Achado",
    "conferir",
    "contrato_markdown",
]


class Gravidade(Enum):
    """Quanto a ausência de um campo custa."""

    #: Sem isto o estudo não roda.
    OBRIGATORIO = "obrigatório"
    #: Sem isto o estudo roda com hipótese no lugar, e a hipótese vai no
    #: relatório como ressalva. É onde mais se perde precisão de graça.
    IMPORTANTE = "importante"
    #: Melhora o resultado, mas há substituto razoável.
    DESEJAVEL = "desejável"


@dataclass(frozen=True)
class Campo:
    """Um campo do backup, o que ele resolve, e o que acontece sem ele."""

    caminho: str
    onde: str
    gravidade: Gravidade
    para_que: str
    sem_ele: str
    #: Quando dado, decide se o valor presente é aceitável — não basta existir.
    valido: Callable[[Any], bool] | None = None
    #: Quando dado, o campo só é exigido nos itens em que isto é verdade.
    #:
    #: ``durMin`` só faz sentido no intervalo dinâmico, e ``modo`` só no fixo.
    #: Cobrar os dois de todo item produzia uma lista de faltas que não faltam
    #: — e uma lista assim é ignorada inteira, inclusive nas linhas certas.
    exigido_se: Callable[[dict[str, Any]], bool] | None = None


def _preenchido(valor: Any) -> bool:
    return valor not in (None, "", [], {})


def _e_dinamico(item: dict[str, Any]) -> bool:
    """Intervalo dinâmico: o acionamento é sorteado dentro da janela."""
    return str(item.get("tipoInt", "")).lower().startswith(("din", "dinâ", "dinÃ"))


def _e_fixo(item: dict[str, Any]) -> bool:
    return not _e_dinamico(item)


def _numero_positivo(valor: Any) -> bool:
    try:
        return float(valor) > 0
    except (TypeError, ValueError):
        return False


#: O contrato. Cada linha é uma conversa que não precisa mais acontecer.
CAMPOS: tuple[Campo, ...] = (
    # -- identificação -----------------------------------------------------
    Campo("vistoria.cliente", "vistoria", Gravidade.OBRIGATORIO,
          "nomear o estudo e a capa do dossiê",
          "o documento sai como 'instalação'", _preenchido),
    Campo("vistoria.sistema", "vistoria", Gravidade.OBRIGATORIO,
          "escolher os inversores que ligam nessa rede",
          "o catálogo é filtrado por 380 V por padrão, e num prédio de 127/220 "
          "isso recomenda equipamento que o eletricista devolve na entrega",
          _preenchido),
    # -- localização -------------------------------------------------------
    Campo("vistoria.cidade", "vistoria", Gravidade.IMPORTANTE,
          "buscar a série horária de geração no local certo",
          "a irradiação vem do centro da UF, e ela varia mais de 15% dentro de "
          "um mesmo estado", _preenchido),
    Campo("vistoria.endereco", "vistoria", Gravidade.DESEJAVEL,
          "marcar o telhado na imagem de satélite",
          "não dá para medir a cobertura nem contar quantos módulos cabem",
          _preenchido),
    # -- o que já existe na instalação ------------------------------------
    Campo("vistoria.fv", "vistoria", Gravidade.IMPORTANTE,
          "saber se o estudo é de ampliação ou de sistema novo",
          "o estudo assume que não há geração, e superestima a economia se houver"),
    Campo("vistoria.gerDiesel", "vistoria", Gravidade.DESEJAVEL,
          "considerar o grupo que já está instalado",
          "o cenário com gerador é orçado como compra nova"),
    Campo("vistoria.ups", "vistoria", Gravidade.DESEJAVEL,
          "descontar o que o nobreak já atende",
          "cargas já protegidas entram no dimensionamento do banco"),
    # -- consumo real ------------------------------------------------------
    Campo("fatura.consumoMensalKwh", "fatura", Gravidade.IMPORTANTE,
          "ancorar a simulação no consumo medido",
          "a conta e a economia saem do levantamento de cargas, que é "
          "estimativa; a fatura é medição"),
    Campo("fatura.tarifaBrlKwh", "fatura", Gravidade.IMPORTANTE,
          "calcular economia e payback com a tarifa do cliente",
          "usa-se uma tarifa de referência, e o payback erra na mesma proporção"),
    # -- por item ----------------------------------------------------------
    Campo("itens[].equipamento", "item", Gravidade.OBRIGATORIO,
          "nomear a carga no anexo e na composição do pico",
          "a linha aparece como 'sem nome'", _preenchido),
    Campo("itens[].potTipica", "item", Gravidade.OBRIGATORIO,
          "a potência da carga, que é o insumo do Monte Carlo",
          "o item é descartado da simulação", _numero_positivo),
    Campo("itens[].comodo", "item", Gravidade.OBRIGATORIO,
          "agrupar a carga por ambiente na análise",
          "tudo cai em 'Sem ambiente' e a composição por ambiente perde sentido",
          _preenchido),
    Campo("itens[].criticidade", "item", Gravidade.OBRIGATORIO,
          "montar o quadro de backup equipamento a equipamento",
          "o recorte volta a ser por cômodo, e a geladeira crítica arrasta o "
          "forno de 4 kW junto — foi o que mudou o inversor de 12 kW para 5 kW "
          "no primeiro caso real", _preenchido),
    Campo("itens[].intervalo", "item", Gravidade.OBRIGATORIO,
          "a janela em que o equipamento pode ligar",
          "o item é tratado como ligado o dia inteiro", _preenchido),
    Campo("itens[].tipoInt", "item", Gravidade.OBRIGATORIO,
          "distinguir carga contínua de acionamento sorteado",
          "assume-se fixo, e uma geladeira vira consumo de 24 h", _preenchido),
    Campo("itens[].prob", "item", Gravidade.IMPORTANTE,
          "a chance de o equipamento ser usado num dia qualquer",
          "assume-se 100%, e o pico sai alto demais", _numero_positivo),
    Campo("itens[].fd", "item", Gravidade.IMPORTANTE,
          "o fator de demanda: quanto da placa é realmente puxado",
          "assume-se 1,0, e a potência sai superestimada", _numero_positivo),
    Campo("itens[].durMin", "item", Gravidade.IMPORTANTE,
          "quanto tempo dura cada acionamento do intervalo dinâmico",
          "o equipamento fica ligado a janela inteira — numa geladeira, "
          "24 h em vez de 40 minutos por hora",
          valido=_numero_positivo, exigido_se=_e_dinamico),
    Campo("itens[].durMax", "item", Gravidade.IMPORTANTE,
          "o limite superior da duração sorteada",
          "mesmo efeito do anterior",
          valido=_numero_positivo, exigido_se=_e_dinamico),
    Campo("itens[].modo", "item", Gravidade.IMPORTANTE,
          "no intervalo fixo, se ocupa a janela toda ou um trecho dela",
          "FIXO_DURACAO_INTERVALAR sem duração se comporta como FIXO_100%, e o "
          "consumo do item sai inflado",
          exigido_se=_e_fixo),
    Campo("itens[].fp", "item", Gravidade.DESEJAVEL,
          "o fator de partida, que decide se o inversor aguenta o motor",
          "o surto do compressor não é simulado, e a triagem por potência "
          "aprova inversor que desarma na partida"),
    Campo("itens[].tensao", "item", Gravidade.DESEJAVEL,
          "conferir o balanceamento entre fases",
          "o estudo trata a carga como equilibrada"),
    Campo("itens[].tempoMax", "item", Gravidade.DESEJAVEL,
          "graduar a criticidade dentro do mesmo nível",
          "todos os itens críticos são tratados como igualmente urgentes"),
)


@dataclass
class Achado:
    """Um campo que falta, ou que veio com valor inutilizável."""

    campo: Campo
    quantos: int = 0
    de_quantos: int = 0

    @property
    def fracao(self) -> float:
        return self.quantos / self.de_quantos if self.de_quantos else 1.0

    def __str__(self) -> str:
        onde = (
            f"{self.quantos} de {self.de_quantos} itens"
            if self.campo.onde == "item"
            else "ausente"
        )
        return f"[{self.campo.gravidade.value}] {self.campo.caminho} — {onde}"


@dataclass
class Conferencia:
    """O veredito sobre um backup."""

    achados: list[Achado] = field(default_factory=list)
    total_de_itens: int = 0

    @property
    def bloqueios(self) -> list[Achado]:
        return [a for a in self.achados if a.campo.gravidade is Gravidade.OBRIGATORIO]

    @property
    def serve(self) -> bool:
        return not self.bloqueios

    def por_gravidade(self, gravidade: Gravidade) -> list[Achado]:
        return [a for a in self.achados if a.campo.gravidade is gravidade]

    def resumo(self) -> str:
        if not self.achados:
            return f"Backup completo: {self.total_de_itens} itens, nada faltando."
        partes = [
            f"{len(self.por_gravidade(g))} {g.value}(is)"
            for g in Gravidade
            if self.por_gravidade(g)
        ]
        veredito = "serve, com ressalvas" if self.serve else "NÃO serve"
        return f"Backup {veredito}: {', '.join(partes)}."


def _valor(dados: dict[str, Any], caminho: str) -> Any:
    no: Any = dados
    for parte in caminho.split("."):
        if not isinstance(no, dict):
            return None
        no = no.get(parte)
    return no


def conferir(dados: dict[str, Any]) -> Conferencia:
    """
    Confere um backup contra o contrato e devolve o que falta.

    Responde três perguntas de uma vez: o estudo roda? o que se perde? e
    quanto? É o que transforma "o arquivo veio incompleto" numa lista de
    campos que a outra equipe pode implementar.
    """
    itens = dados.get("itens") or []
    conferencia = Conferencia(total_de_itens=len(itens))

    for campo in CAMPOS:
        if campo.onde == "item":
            chave = campo.caminho.split("[].", 1)[1]
            faltando = 0
            for item in itens:
                if not item.get("presente", True) or not item.get("simular", True):
                    continue
                if campo.exigido_se is not None and not campo.exigido_se(item):
                    continue
                valor = item.get(chave)
                if campo.valido is not None:
                    if not campo.valido(valor):
                        faltando += 1
                elif not _preenchido(valor):
                    faltando += 1
            aplicaveis = sum(
                1 for i in itens
                if i.get("presente", True) and i.get("simular", True)
                and (campo.exigido_se is None or campo.exigido_se(i))
            )
            if faltando:
                conferencia.achados.append(Achado(campo, faltando, aplicaveis))
            continue

        valor = _valor(dados, campo.caminho)
        if campo.valido is not None:
            ausente = not campo.valido(valor)
        else:
            # Booleano ausente é diferente de booleano falso: o primeiro é
            # informação que ninguém deu, o segundo é uma resposta.
            ausente = valor is None
        if ausente:
            conferencia.achados.append(Achado(campo, 1, 1))

    return conferencia


def contrato_markdown(gravidades: Iterable[Gravidade] | None = None) -> str:
    """O contrato em Markdown, para colar no chamado da outra equipe."""
    escolhidas = list(gravidades or list(Gravidade))
    linhas = [
        "| Campo | Onde | Gravidade | Para quê | Sem ele |",
        "| --- | --- | --- | --- | --- |",
    ]
    for campo in CAMPOS:
        if campo.gravidade not in escolhidas:
            continue
        linhas.append(
            f"| `{campo.caminho}` | {campo.onde} | {campo.gravidade.value} | "
            f"{campo.para_que} | {campo.sem_ele} |"
        )
    return "\n".join(linhas)
