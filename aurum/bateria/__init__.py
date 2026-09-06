"""
Estudo de baterias — dimensionamento por resiliência, potência e economia.

Cruza três coisas que, separadas, não dimensionam nada:

* a **demanda probabilística** do D² (:mod:`aurum.demanda`) — não uma curva de
  carga, mas a distribuição de curvas possíveis;
* a **geração solar horária** de anos reais no local (:mod:`.geracao`);
* os **limites do equipamento** — energia útil da bateria, potência contínua e
  de surto do inversor híbrido (:mod:`.catalogo`).

E responde, para faltas de 1, 2, 3, 6, 12, 24 e 36 h começando em qualquer hora
de qualquer estação: com que probabilidade o conjunto atravessa, quanta energia
falta quando não atravessa, quanto tempo até a primeira falha, e se a falha foi
de energia (banco vazio) ou de potência (inversor pequeno) — que exigem
soluções opostas.

Antes disso tudo há uma pergunta que o estudo também responde: **vale a pena
comprar alguma coisa, e o quê**. :mod:`.fontes` monta um cenário para cada
combinação de solar, bateria e gerador e mede todos contra a conta que o
cliente paga hoje -- inclusive o cenário de não fazer nada, que é o único
referencial honesto.

Ponto de entrada: :func:`aurum.bateria.executar_estudo`.
"""
from .apagao import (
    MalhaApagao,
    ResultadoResiliencia,
    avaliar_conjunto,
    avaliar_limites,
    varrer_conjuntos,
)
from .catalogo import (
    Bateria,
    BaseBaterias,
    ConjuntoArmazenamento,
    InversorHibrido,
    carregar_catalogo,
    criar_planilha_modelo,
)
from .degradacao import ModeloDegradacao, trajetoria_de_vida
from .despacho import Gerador, LimitesDespacho, ResultadoDespacho, simular_ilhamento
from .economia import PremissasBateria, avaliar_economia, simular_operacao_anual
from .estudo import ConfiguracaoEstudo, ResultadoEstudo, executar_estudo
from .fontes import (
    ComparacaoFontes,
    Composicao,
    Fatura,
    ResultadoCenario,
    comparar_fontes,
)
from .excedencia import (
    DiagnosticoInversor,
    avaliar_inversores,
    diagnosticar_inversor,
    potencia_para_excedencia,
    tabela_por_faixa_horaria,
)
from .geracao import SerieGeracao, obter_serie_horaria, serie_sintetica

__all__ = [
    "BaseBaterias",
    "Bateria",
    "ComparacaoFontes",
    "Composicao",
    "ConfiguracaoEstudo",
    "ConjuntoArmazenamento",
    "DiagnosticoInversor",
    "Fatura",
    "Gerador",
    "InversorHibrido",
    "LimitesDespacho",
    "MalhaApagao",
    "ModeloDegradacao",
    "PremissasBateria",
    "ResultadoCenario",
    "ResultadoDespacho",
    "ResultadoEstudo",
    "ResultadoResiliencia",
    "SerieGeracao",
    "avaliar_conjunto",
    "avaliar_economia",
    "avaliar_inversores",
    "avaliar_limites",
    "carregar_catalogo",
    "comparar_fontes",
    "criar_planilha_modelo",
    "diagnosticar_inversor",
    "executar_estudo",
    "obter_serie_horaria",
    "potencia_para_excedencia",
    "serie_sintetica",
    "simular_ilhamento",
    "simular_operacao_anual",
    "tabela_por_faixa_horaria",
    "trajetoria_de_vida",
    "varrer_conjuntos",
]
