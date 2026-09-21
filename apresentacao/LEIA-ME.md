# Apresentação comercial da PACE (beamer)

A proposta de 19 slides que acompanha o estudo, refeita em LaTeX a partir do
modelo em PowerPoint da Solar21 (`10.09.2026_Solar21_Shaped_PROPOSTA.pptx`),
com a identidade da PACE — preto e âmbar de `aurum/marca.py`, Montserrat,
logo da PACE em cada slide.

```
apresentacao/
├── apresentacao.tex        o esqueleto: a ordem dos slides e o discurso da casa
├── pace-apresentacao.sty   o tema visual (cores, cabeçalho, cartões, tabelas, gráfico)
├── dados.tex               TUDO que muda de proposta para proposta
├── figuras/                a foto aérea do telhado com os módulos (uma por proposta)
└── estaticas/              fotos de cases, logos de certificação e de bancos,
                            fotos de equipamento, foto da capa — não mudam
```

## Compilar

Duas passadas, porque a capa, as notas de rodapé e os próximos passos usam
`remember picture` e só acertam a posição na segunda:

```bash
pdflatex apresentacao.tex && pdflatex apresentacao.tex
```

Precisa de MiKTeX ou TeX Live com `beamer`, `montserrat`, `fontawesome5`,
`tcolorbox` e `pgfplots` — todos padrão. O `../assets/` com as logos da PACE
precisa estar ao lado da pasta.

## Os dois modos

Cada slide é escuro (a proposta, slides 1–12 e 17 e 19) ou claro (o anexo
técnico, 13–16 e 18). Chame `\modoescuro` ou `\modoclaro` antes do frame; a
cor de cartão, de tabela, de ícone e a versão da logo trocam juntas.

## Como a PACEcalculator gera a apresentação

`aurum/bateria/apresentacao.py` escreve o `dados.tex` a partir do
`ResultadoEstudo` e copia este esqueleto (com as imagens fixas e as logos)
para `estudo/apresentacao/` dentro do pacote do dossiê. Entra pelo mesmo
caminho do dossiê: `escrever_dossie(..., comercial=OpcoesComerciais())` ou
`zip_do_dossie(..., comercial=...)`. A tela de entrega e o
`estudo_coleta` já passam isso.

O que vem **do estudo**: cliente, potência, módulo e quantidade (memória de
cálculo ou layout), inversor de string (memória), inversor híbrido e banco
(conjunto recomendado), área ocupada (layout), geração anual, investimento,
economia do 1º ano, payback, TIR e o fluxo ano a ano — tudo do cenário de
fontes apresentado (`solar+bateria` quando há banco recomendado, senão
`solar`), mais a foto aérea do telhado.

O que vem **do operador** (`DadosCapa` + `OpcoesComerciais`), com padrão
declarado: responsável, CREA e credenciais; juros, prazo e carência do
financiamento; fração da economia que vira mensalidade do leasing e seu
prazo; seguro e gerenciamento mensais; garantias; estrutura; os números
institucionais do slide de monitoramento. São premissas de venda — o
estudo não tem como calculá-las, e os padrões são os da proposta modelo.

## O que vem de `dados.tex`

Só valores já formatados em pt-BR, com `R\$` e `\%` escapados. Nenhuma
lógica. O `dados.tex` desta pasta é o exemplo da proposta Shaped, para o
esqueleto compilar sozinho; o gerado pela PACEcalculator segue o mesmo
contrato:

| Grupo | Comandos |
|---|---|
| Identificação | `\cliente`, `\dataproposta`, `\tituloproposta`, `\responsavel`, `\responsavelregistro`, `\responsavelcredenciais`; `\renewcommand{\pastalogos}{estaticas/}` quando as logos foram copiadas |
| Sistema | `\potenciaprojeto`, `\potenciamodulo`, `\qtdmodulos`, `\potenciainversor`, `\qtdinversores`, `\areaocupada`, `\producaoanual`, `\bess`, `\fototelhado` |
| Compra à vista | `\solucao`, `\geracaoanual`, `\investimento`, `\economiaanual`, `\rotulopayback`, `\payback`, `\gerenciamentomensal`, `\seguromensal`, `\rotulotir`, `\tir` |
| Financiado | `\investimentocurto`, `\economiamensal`, `\mensalidadefinanciada`, `\segurofinanciado`, `\gerenciamentofinanciado`, `\prazofinanciamento`, `\carenciafinanciamento`, `\validadeproposta` |
| Leasing | `\mensalidadeleasing`, `\prazoleasing`, `\carencialeasing` |
| Desembolso mensal | `\linhasdesembolso` — uma `\linhadesembolso{cenário}{investimento}{mensalidade}{custo}{economia}{saldo}` por cenário |
| Fluxo de caixa | `\linhasfluxo` (tabela, já formatado); `\fluxoanos`, `\fluxozero`, `\fluxoavista`, `\fluxofinanciado`, `\fluxoservico` (gráfico, valor cru); `\notafluxo` (a ressalva) |
| Lista de equipamentos | `\linhasequipamentos`, `\fotosequipamentos` |
| Institucional | `\clientesgeridos`, `\potenciagerida` |

Valores positivos e negativos nas tabelas financeiras vão dentro de
`\positivo{}` e `\negativo{}`, que pintam a célula de verde ou vermelho.

## O que ficou diferente do PowerPoint

- A marca é a PACE, não a Solar21: logo, cores e o nome nos títulos. Os
  logos de terceiros (CARINA, EQI, Solar21, BTG, certificadoras, bancos,
  GoodWe) ficaram como imagens.
- O gráfico do fluxo de caixa é desenhado em `pgfplots` a partir dos dados,
  não colado como imagem.
- Os ícones vêm do FontAwesome, não de PNGs pequenos.
