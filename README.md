# Aurum — Prospecção e proposta automática de geração solar

Ferramenta de prospecção comercial para sistemas fotovoltaicos. Varre uma
região no OpenStreetMap, identifica os maiores telhados aproveitáveis,
descobre qual empresa ocupa cada um e gera a proposta técnico-comercial
completa em LaTeX.

Unifica dois programas que antes eram separados: o **Aurumcalc** (calculadora
e proposta) e o **Aurum Lead Mapper** (identificação de telhados no OSM).

---

## O fluxo

```
região (nome ou bbox)
   ↓  OpenStreetMap / Overpass
todas as edificações da área
   ↓  filtro geométrico + pontuação
telhados aproveitáveis, do maior para o menor
   ↓  enriquecimento (nome, telefone, site, endereço)
lista de leads acionáveis
═══════════ VOCÊ ESCOLHE QUAIS QUER ═══════════
   ↓  PVGIS + empacotamento de módulos + arranjo elétrico + economia
proposta em LaTeX + PDF, uma por telhado
```

**Nenhum estudo é calculado antes da sua escolha.** A linha dupla acima é uma
fronteira real no código, não só no desenho: prospectar é barato e olha a
região inteira; dimensionar é caro e só roda no que você marcou. A regra vale
nos três caminhos — interface, linha de comando e biblioteca — e tem teste
automatizado que impede uma refatoração futura de furá-la.

---

## Deploy no Streamlit Community Cloud

O ponto de entrada é `app.py`, na raiz — é o que o Streamlit Cloud pede. As
dependências da aplicação estão em `requirements.txt`; as de desenvolvimento,
em `requirements-dev.txt`, que o deploy não instala.

1. No [share.streamlit.io](https://share.streamlit.io), **New app**.
2. Repositório `PACEcalculator`, branch `main`, arquivo `app.py`.
3. Em *Advanced settings*, Python **3.12**.

Não há segredo a configurar: o software não usa chave de API. O PVGIS e o
OpenStreetMap são abertos, e o catálogo de equipamento vem nas planilhas do
próprio repositório.

### O que muda no Cloud

**O PDF não compila.** O Streamlit Cloud não tem distribuição LaTeX, e instalar
uma passa de 1 GB. O software já trata isso: o dossiê sai com o `.tex`, as
figuras em PNG e as tabelas em CSV, que compilam no Overleaf sem ajuste — e a
tela avisa antes de você clicar. Rodando na sua máquina, com o MiKTeX
instalado, o PDF sai junto.

**O disco é efêmero.** Cada reinício do app zera o cache do PVGIS e as
planilhas de catálogo voltam ao que está no repositório. Na prática isso quer
dizer duas coisas: a primeira consulta de cada local demora alguns segundos a
mais, e **correção feita na planilha de equipamento pela interface se perde**.
Cadastro de equipamento que precisa durar vai por commit, não pela tela.

**Dados de cliente não sobem.** `propostas/`, `leads/`, `outputs/` e `.cache/`
estão no `.gitignore` desde o começo. Confira antes de tornar o repositório
público.

---

## Instalação

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Para gerar PDF é preciso uma distribuição LaTeX (MiKTeX ou TeX Live). Sem ela
o programa entrega os arquivos `.tex`, que compilam no
[Overleaf](https://overleaf.com) sem ajuste. A busca pelo compilador cobre o
PATH e os diretórios de instalação padrão do Windows.

---

## Uso

### Interface gráfica

```bash
streamlit run app.py
```

Dois modos, escolhidos no alto da barra lateral.

**Estudo de energia** — um cliente conhecido, do levantamento de cargas ao
dimensionamento de solar e bateria, em **duas fases**:

| Fase | Passo | A pergunta na tela |
| --- | --- | --- |
| **1 · Dimensionamento** | Cliente | Para quem é o estudo, e onde fica? |
| | Cargas | O que consome energia nesse lugar? |
| | Consumo | Como esse lugar consome energia? |
| | Telhado | Onde os módulos vão ficar? |
| | Equipamento | Que módulo e que inversor? |
| **2 · Análise com bateria** | Backup | O que não pode faltar quando a luz cai? |
| | Meta | Quanto tempo sem luz precisa aguentar, com que fontes, e qual é a conta de hoje? |
| | Resultado | O que comprar — e o dossiê para baixar |

A fronteira entre as fases é real, não decorativa. A primeira fase termina num
projeto fechado: equipamento escolhido do catálogo e memória de cálculo do
arranjo. Só depois disso a segunda pergunta o que acontece quando a rede cai.

Misturar as duas era o defeito do fluxo anterior — quem só queria dimensionar
atravessava perguntas sobre apagão, e quem queria a análise de bateria nunca
via o dimensionamento fechar. Um teste automatizado percorre a fase 1 inteira e
falha se a palavra "apagão", "autonomia" ou "bateria" aparecer em qualquer
controle antes da hora.

O desenho segue quatro regras: **uma pergunta por tela em português**, o nome
do parâmetro fica escondido; **nada de folha em branco**, o passo das cargas já
vem montado a partir do tipo de instalação; **padrão em tudo**, dá para
atravessar o fluxo só apertando "Continuar" e ainda sair com um estudo
defensável; e **nunca um beco sem saída** — botão que não funciona diz o que
falta, e estudo que não acha solução diz por quê.

**Prospecção de telhados** — o fluxo antigo, para quando o cliente ainda não
existe: varre uma região no OpenStreetMap, você escolhe os telhados, saem as
propostas.

### Linha de comando

O fluxo tem duas etapas, com a sua escolha no meio.

**1. Varrer a região.** Não calcula estudo nenhum; só lista e grava.

```bash
python -m aurum.cli prospectar --regiao "Cidade Industrial de Curitiba" --min-area 2000 --top 40 --saida leads/cic
```

Para procurar um tipo de local específico, use `--alvo` (vários separados por
vírgula):

```bash
python -m aurum.cli prospectar --regiao "Castro, Paraná" --alvo agro --saida leads/castro
python -m aurum.cli prospectar --regiao "Água Verde, Curitiba" --alvo condominio --saida leads/agua-verde
python -m aurum.cli prospectar --regiao "Londrina" --alvo educacao,saude --saida leads/londrina
```

`python -m aurum.cli segmentos` lista os tipos disponíveis com a descrição de
cada um.

Isso grava `leads/cic/leads.csv` com a coluna `selecionar` em branco na
primeira posição, além de `leads.geojson` (geometrias) e `prospeccao.json`
(estatísticas da varredura).

**2. Escolher.** Abra `leads/cic/leads.csv` no Excel ou LibreOffice, escreva
`x` na coluna `selecionar` das linhas que interessam, salve. Valem também
`sim`, `1`, `ok` e afins.

**3. Gerar as propostas** — só dos marcados:

```bash
python -m aurum.cli propor --da-lista leads/cic --saida propostas/cic
```

Antes de calcular, o comando mostra o que vai processar:

```
Serão calculados: 2 de 10 telhados marcados na planilha
  ·    24,587 m²  Edificação sem nome (yes)
  ·    10,338 m²  Osten Ferragens
```

Se preferir indicar pela linha de comando em vez da planilha:

```bash
python -m aurum.cli propor --da-lista leads/cic --ids way/447998570,way/357014707
```

Para pular a escolha manual e aceitar os N melhores, é preciso dizer isso com
todas as letras — não existe caminho implícito:

```bash
python -m aurum.cli propor --regiao "Cidade Industrial de Curitiba" --auto-top 10
```

Conferir o catálogo de equipamentos carregado:

```bash
python -m aurum.cli equipamentos
```

### Como biblioteca

```python
from aurum.pipeline import (
    carregar_leads, dimensionar_lote, escrever_geojson,
    filtrar_selecionados, ler_selecao_csv, prospectar,
)
from aurum.proposal.render import escrever_proposta

# 1. varre e grava — nada é dimensionado aqui
resultado = prospectar(regiao="Distrito Industrial de Cariacica", min_area_m2=2000)
escrever_geojson(resultado.leads, "leads/cariacica/leads.geojson")

# 2. depois de escolher (na planilha, na sua interface, como preferir)
leads = carregar_leads("leads/cariacica")
escolhidos = filtrar_selecionados(leads, ler_selecao_csv("leads/cariacica"))

# 3. só agora o cálculo roda
contextos, falhas = dimensionar_lote(escolhidos)
for ctx in contextos:
    escrever_proposta(ctx, f"propostas/{ctx.referencia}")
```

`executar()` faz tudo de uma vez, mas exige `selecao=[...]` ou `top=N`
explícito — chamada sem nenhum dos dois levanta erro em vez de dimensionar a
região inteira.

---

## O levantamento de cargas

O passo 2 é onde o estudo ganha ou perde. Três caminhos, do melhor para o pior:

**Rascunho do modelo (padrão).** Escolhido o tipo de instalação no passo 1, o
cenário nasce montado: um hotel vem com apartamento, áreas comuns, cozinha e
lavanderia, cada um com seus equipamentos, potências, janelas de uso e fatores
de demanda. Você ajusta a quantidade de cada ambiente e corrige o que for
diferente. Corrigir um rascunho é muito mais barato que partir do zero — e é o
único jeito de alguém terminar o levantamento na primeira tentativa.

São dez segmentos (hotel, escritório, supermercado, restaurante, escola,
hospital, indústria, frigorífico, varejo, academia) e um catálogo de **69
equipamentos** em 12 categorias, de lâmpada LED a ponte rolante, cada um com a
janela e o fator de demanda típicos. Adicionar um equipamento é escolher
categoria, modelo e quantidade.

**Planilha do D².** O mesmo formato que o `DemandaDados` já lê — uma aba por
cômodo. Importa e edita na tela; exporta de volta a qualquer momento.

**Só a conta de luz.** Sem levantamento, a curva típica do segmento é calibrada
pelo consumo mensal da fatura. A tela diz em voz alta o que se perde: a forma
vem do padrão do setor e o tamanho da conta, mas **falta a variabilidade dia a
dia — e é ela que produz o pico raro que decide o inversor**. O estudo assume
uma dispersão de 15%, que é hipótese, não medição.

### O que a validação pega antes de simular

O editor confere cada linha e aponta cômodo, linha e campo. Três dos achados
não são erros de digitação óbvios — são armadilhas do D² que produzem um número
plausível e errado:

| Situação | O que o D² faz | O que a tela diz |
| --- | --- | --- |
| Tipo `dinâmico` sem duração | Cai no parser legado, não reconhece `"08:00 as 18:00"` e assume **1 hora à meia-noite** | Erro, com o texto do formato esperado |
| Janela que vira a meia-noite | Perde a carga inteira (`carga[1080:360]` é vazio) | Aviso — aqui é tratada como dia cíclico |
| Duração maior que a janela | Encurta em silêncio para caber | Aviso do encurtamento |

### A conferência contra a curva do setor

Depois de simular, a aba *Bate com o padrão do setor?* compara a curva do Monte
Carlo com a curva de referência do segmento — as curvas normalizadas de 24 h
que vieram do Aurumcalc, em `data/load_profiles/`.

O veredito sai da **energia por período de 6 h** (madrugada, manhã, tarde,
noite), não da correlação hora a hora. A correlação sobre 24 pontos é frágil:
um pico legítimo que a curva de referência não modela — o chuveiro elétrico de
um hotel às 7 h — derruba o coeficiente sem que nada esteja errado, e um alarme
que dispara no caminho padrão é pior que alarme nenhum. Já a distribuição entre
períodos só se desloca quando a energia foi mesmo parar na hora errada.

Essa conferência tem teste automatizado sobre os dez modelos semeados, e foi
ela que pegou dois defeitos nos próprios modelos deste repositório: o quarto de
hotel com pico ao meio-dia (hora em que o quarto está vazio, porque a janela
padrão do ar-condicionado era 08:00–18:00) e a academia com pico às 7 h, puxado
pelos chuveiros do vestiário.

---

## Estrutura

```
aurum/
  marca.py           a identidade da PACE: nome, cores e logos, num lugar só
  config.py          parâmetros, chaves e caminhos
  precos.py          faixas de preço de mercado, datadas e com origem
  pipeline.py        orquestração: prospectar / escolher / dimensionar
  cli.py             linha de comando
  geo/
    overpass.py      cliente Overpass com cache, rate limit e failover
    nominatim.py     geocodificação de região e endereço
    geometry.py      projeção métrica e métricas de forma
    segments.py      taxonomia de tipos de local (agro, escola, condomínio…)
    roofs.py         parsing, filtro, pontuação e deduplicação
    enrich.py        nome, telefone, site e endereço da empresa
  pv/
    solar.py         recurso solar (PVGIS → PVWatts → estimativa)
    equipment.py     catálogo de módulos e inversores
    semente.py       datasheets de módulo e inversor, com procedência por linha
    telhado.py       do polígono desenhado ao arranjo de módulos
    imagem.py        recorte de satélite para a foto do telhado
    memoria.py       memória de cálculo do arranjo, passo a passo
    edicao.py        edição manual do arranjo: tirar, pôr, deslocar, girar
    layout.py        empacotamento de módulos no polígono real
    sizing.py        inversores e arranjo de strings
    consumption.py   consumo estimado por segmento
    financials.py    fluxo de caixa, VPL, TIR, Lei 14.300
  proposal/
    latex.py         escape e primitivas de LaTeX
    sections.py      seções do documento
    context.py       pacote de dados da proposta
    render.py        escrita, compilação e empacotamento
  demanda/
    nucleo.py        núcleo Monte Carlo do D², extraído sem Streamlit
    correcoes.py     correções ao núcleo, ligadas por contexto explícito
    ensemble.py      curvas diárias por estação e curvas de excedência
    biblioteca.py    catálogo de equipamentos, modelos de segmento, curvas típicas
    cenario.py       cenário editável, com validação antes de simular
    analise.py       a análise estatística completa do consumo
  bateria/
    geracao.py       série horária de 8.760 h via PVGIS seriescalc
    catalogo.py      BDBaterias.xlsx: baterias e inversores híbridos
    despacho.py      simulação do ilhamento, vetorizada
    apagao.py        malha duração × hora de início × estação
    excedencia.py    curva de excedência contra o limite do inversor
    degradacao.py    envelhecimento por calendário e por ciclagem
    economia.py      operação anual, valor da resiliência, fluxo de caixa
    fontes.py        solar, bateria e gerador ligados e desligados: os cenários
    estudo.py        orquestração do estudo completo
    relatorio.py     figuras, planilhas e resumo em Markdown
    documento.py     o dossiê: LaTeX completo, PDF e ZIP
    cli.py           subcomandos `aurum bateria`
  pagina_estudo.py   interface do estudo de energia (duas fases, 8 passos)
app.py               interface Streamlit (dois modos)
tests/               462 testes automatizados
_legacy/             os dois repositórios originais, para referência
```

---

## A análise do consumo

O passo 3 é a análise que o **D²** faz, refeita sobre o ensemble sazonal e
estendida. Ela vem antes de qualquer decisão de equipamento porque o número que
sai dela — o pico, quem o causa e a que hora — é o que justifica cada escolha
adiante.

**O que veio do D²:** distribuição dos picos com média e P95, curva de
probabilidade de excedência, curva de duração de carga, perfil médio do dia,
fator de carga hora a hora, e a composição por ambiente empilhada.

**O que foi acrescentado, e por quê:**

| Acréscimo | Por que importa |
| --- | --- |
| Cada grandeza por estação | O mesmo prédio tem picos diferentes em janeiro e em julho, e é o pior deles que dimensiona |
| Composição do pico por equipamento | Saber que o pico é 60 kW não diz o que fazer; saber que 63% dele são chuveiros às 8 h diz |
| Fator de coincidência | Mede quanto a operação real economiza sobre a potência de placa — num hotel de 40 apartamentos, um fator de cinco |
| Incerteza do próprio Monte Carlo | Com poucas simulações o P95 tem margem, e reportá-lo sem ela sugere precisão que o número não tem |

A composição do pico é o acréscimo que mais muda a conversa. O D² já a
calculava — `simula_carga_total(coletar_detalhes_pico=True)` — e não a levava
ao relatório. Num hotel de 40 apartamentos, ela mostra:

```
Chuveiro elétrico 5.500 W    Apartamento    30,66 kW    presença 99%    63,2% do pico
Ar-condicionado 9.000 BTU    Apartamento     5,90 kW    presença 100%   12,2% do pico
Câmara fria                  Cozinha         1,65 kW    presença 100%    3,4% do pico
```

Com essa tabela, a conversa deixa de ser sobre comprar um inversor maior e
passa a ser sobre aquecimento solar ou deslocamento de carga — quase sempre
mais barato.

---

## A memória de cálculo

O passo 5 é onde o equipamento é escolhido do catálogo e a conta aparece. Cada
passo traz a **fórmula**, os **números que entraram** e o **resultado com a
unidade** — que é o que um responsável técnico assina, em vez de pedir fé.

A ordem é a ordem em que a decisão acontece:

1. **Tensão no frio.** A Voc sobe quando esfria, e é ela contra a tensão máxima
   CC que fixa o máximo de módulos em série. Errar aqui não é perder
   rendimento: é queimar o inversor no primeiro amanhecer frio do ano.
2. **Tensão no calor.** A Vmp cai quando esquenta, e é ela contra a tensão de
   partida que fixa o mínimo da string.
3. **Corrente por MPPT.** O datasheet declara dois limites diferentes —
   corrente de operação e corrente de curto — comparados com Imp e Isc. O menor
   manda.
4. **Potência FV do inversor.** Com módulo de alta potência este costuma ser o
   limite que morde antes da corrente.
5. **Razão CC/CA de projeto.** O fabricante e o projeto têm tetos diferentes: a
   série MT admite 150% de sobredimensionamento, e a regra de projeto adotada é
   1,35. O primeiro protege o equipamento; o segundo protege o rendimento do
   investimento. A memória diz qual dos cinco limites governou o arranjo.
6. **Quantidade de inversores.** Quando o telhado comporta mais do que um
   inversor absorve, o último recebe as strings que sobraram — e por isso o
   total do sistema não é o produto simples de inversores por strings.

Trocando o inversor no seletor, a memória inteira é recalculada. Para um
telhado de 288 módulos LONGi de 620 Wp:

| Inversor | Arranjo | Sistema | Governa |
| --- | --- | --- | --- |
| GW50KN-MT | 19/string × 15 strings × 3 inv | 285 mód · 176,7 kWp | razão CC/CA de projeto |
| GW50KS-MT | 19/string × 15 strings × 3 inv | 285 mód · 176,7 kWp | potência FV do inversor |
| GW80K-MT | 19/string × 15 strings × 2 inv | 285 mód · 176,7 kWp | razão CC/CA de projeto |

Tudo isso vai para o dossiê, na seção do sistema fotovoltaico.

---

## O telhado, marcado no mapa

No passo 4 você desenha a água do telhado sobre a imagem de satélite. Dali sai
tudo o que o dimensionamento precisa, medido em vez de estimado:

* **A área**, projetada em coordenadas métricas — e também a superfície real da
  cobertura, que num telhado de 20° é 6% maior que o contorno visto de cima. É
  a superfície que recebe módulo, não a projeção.
* **A orientação**, deduzida da cumeeira: a água é perpendicular ao lado mais
  longo. O contorno em planta **não distingue para qual dos dois lados a água
  cai** — as duas opções são geometricamente idênticas. A tela oferece as duas,
  adota a mais próxima do ótimo da latitude e diz que fez isso. Em laje plana o
  problema não existe: a estrutura aponta para onde se quiser.
* **Quantos módulos cabem**, pelo empacotamento real no polígono (`aurum/pv/layout.py`),
  com recuo de borda, espaçamento entre fileiras contra sombreamento e desconto
  de obstáculos que a imagem não mostra.

Um galpão de 40 × 25 m dá 178,6 kWp em telhado coplanar e 90,9 kWp em laje
plana com estrutura inclinada — a diferença é o espaçamento entre fileiras, e é
por isso que a regra de bolso de "kWp por 100 m²" erra por um fator de dois
dependendo da cobertura.

O croqui volta para o mapa com cada módulo no lugar, e vai para o relatório
junto com a **foto aérea do telhado com os painéis sobrepostos** — a figura que
o cliente entende sem explicação, porque reconhece o próprio prédio.

### O arranjo é editável

O empacotamento em grade é um bom ponto de partida e um péssimo ponto final.
Ele não sabe onde fica a caixa d'água, não vê a sombra da casa de máquinas do
elevador, não sabe que aquele canto é o único acesso de manutenção e não sabe
que o cliente quer os módulos alinhados com a fachada por razões que não são de
engenharia. Quem sabe está olhando a imagem — e a tela dá quatro jeitos de
dizer:

| Operação | Como | O que acontece |
| --- | --- | --- |
| **Tirar um módulo** | clique nele no mapa | fica cinza tracejado, e volta com outro clique |
| **Pôr um módulo** | clique num vazio | entra se couber sem invadir a borda nem outro painel |
| **Deslocar a grade** | dois controles em metros | alinha as fileiras com o telhado real; o que sai do contorno é retirado |
| **Girar / inclinar** | ângulo das fileiras e inclinação | reempacota do zero |

O módulo removido **não some do mapa** — fica em cinza tracejado. Um painel que
desaparecesse ao ser clicado não teria como voltar.

Duas invariantes são garantidas pelo código, e testadas, porque são o que separa
um croqui de um desenho bonito: nenhum módulo fora do contorno (nem depois de
arrastar a grade), e nenhum módulo em cima de outro. Sobreposição some do papel
e reaparece na obra, quando o instalador descobre que faltam trilhos.

A edição guarda **intenção, não geometria**: os índices desligados valem sobre a
numeração estável do empacotamento, e os módulos manuais são pontos de centro em
UTM. Trocar o módulo do catálogo não apaga o trabalho de tirar os seis painéis
de cima da claraboia. Girar a grade, sim — e a tela avisa antes, porque a
numeração deixa de valer.

---

## A tensão da rede não é detalhe de instalação

A maior parte dos híbridos trifásicos do mercado é de **220/380 V**. Num prédio
de **127/220 V** — que é a rede de boa parte do país — nenhum deles liga. É um
erro que não aparece em simulação nenhuma: aparece na entrega, quando o
eletricista devolve o equipamento.

O passo **Cliente** pergunta a tensão de linha antes de qualquer outra coisa, e
ela filtra o catálogo já na triagem — antes de simular apagão, porque é o filtro
mais barato e o mais definitivo: um inversor de 380 V não vira 220 V com corte
de carga nem com banco maior.

| Rede | O que o catálogo oferece |
| --- | --- |
| **127/220 V** | GoodWe ES LD (5 a 10 kW, bifásico), ET LV `-LL-` (12 kW trifásico), ETR `-L-` (50 e 75 kW) |
| **220/380 V** | o resto do catálogo — ET, ET PLUS+, ET LV, ETR, ETC, Deye, Growatt, Sungrow, Solis, Victron |

Quando o filtro esvaziaria o catálogo, ele **não** é aplicado: o estudo segue
com tudo e diz por quê. Bloquear deixaria o usuário sem resposta nenhuma;
avisar deixa a decisão com quem sabe se o projeto prevê transformador — que é
uma solução real, e não cabe ao software descartá-la.

Um teste garante que nenhum modelo aparece nas duas listas: 220 e 380 exigem
enrolamentos diferentes, e um cadastro nas duas seria tensão mal preenchida.

---

## Preço tem data; datasheet não

A Voc de um módulo não muda porque o dólar subiu. Preço muda toda semana, varia
entre distribuidores e depende de volume. Num documento que mistura os dois, o
cliente não distingue *"o inversor entrega 24 kVA por 10 s"* (fato de catálogo)
de *"o inversor custa R$ 18.000"* (chute de dois meses atrás).

Por isso as faixas de mercado moram num módulo próprio, `aurum/precos.py`, com
**data** e com a origem de cada uma:

| Item | Faixa | Adotado |
| --- | --- | --- |
| Módulo fotovoltaico | R$ 1,36 – 2,18/Wp | R$ 1,75/Wp |
| Inversor híbrido | R$ 1,60 – 2,60/W | R$ 2,00/W |
| Bateria de lítio (LFP) | R$ 2.500 – 5.000/kWh | R$ 3.200/kWh |
| Energia de grupo gerador | R$ 1,80 – 3,20/kWh | R$ 2,30/kWh |

A faixa é publicada inteira, e não só a média: **a distância entre o mínimo e o
máximo é a informação mais honesta da tabela**. Onde ela é larga — a bateria
dobra de ponta a ponta —, o valor do meio não merece a confiança que um número
único aparenta ter. O dossiê traz a tabela e a ressalva na seção de procedência.

Cotação real vence tudo isso: preencha a coluna de preço na planilha do
catálogo e as faixas deixam de ser consultadas. Elas existem para o estudo não
travar por falta de preço, não para substituir orçamento.

Um registro que vale guardar: em 2026 os preços **subiram**, contrariando a
série histórica. Módulos acumularam cerca de 30% de alta entre dezembro de 2025
e março de 2026, com o imposto de importação sobre painel chinês escalonando
até 35% em julho e o fim do reembolso de VAT na China. Premissa calibrada em
2024 está errando para baixo.

---

## Os catálogos de equipamento

Duas planilhas, ambas regeneráveis a partir do código e ambas com uma coluna
`fonte_dado` que diz de onde veio cada linha:

```bash
python -m aurum.cli bateria catalogo --criar
```

**`BDFotovoltaica.xlsx`** — módulos e inversores de rede. Traz a linha
**GoodWe MT (50 a 80 kW)** e **SMT (50 e 60 kW)**, lidas do datasheet do
fabricante, e os módulos de 620 Wp da **LONGi Hi-MO 7** e da **DAH Solar**, com
as dimensões reais. As dimensões importam tanto quanto a potência: o código
antes assumia 2,279 × 1,134 m para qualquer modelo, o que faz a restrição de
área mentir para um painel de 2,465 m de comprimento.

**`BDBaterias.xlsx`** — baterias e inversores híbridos. Traz a linha **GoodWe
ET PLUS+ (5 a 10 kW)** e **ET (15 a 30 kW)**, e as torres **Lynx F G2** e
**Lynx Home F Plus+**.

O que o datasheet da GoodWe dá e quase nenhum outro dá: **a sobrecarga de
backup com a duração**. O ET PLUS+ de 10 kW sustenta 16,5 kVA por 60 s; o ET de
15 kW sustenta 18 kVA por 10 s. Sem a duração não dá para saber se o inversor
cobre a partida de um motor ou apenas o instante inicial dela — e era esse o
campo que a semente inicial do catálogo tinha de arbitrar.

Cada linha declara a natureza do dado em três estados, e o relatório reproduz
essa tabela:

| Estado | O que significa |
| --- | --- |
| Datasheet do fabricante | Todos os campos lidos do documento |
| Datasheet, com ressalva | Documento consultado, mas algum campo não consta dele |
| Ordem de grandeza de mercado | Valor plausível, **não** conferido |

As baterias GoodWe caem no segundo estado: o datasheet publica a energia
utilizável a 100% de profundidade de descarga, mas não publica vida em ciclos
nem rendimento de ida e volta, que ficam declarados como pendentes.

`semear_planilha()` **acrescenta** sem sobrescrever: o que você corrigiu à mão
com o datasheet do lote comprado continua lá depois de reexecutar a semente.

---

## Cenários: com e sem cada fonte

Antes de "qual bateria comprar" existe uma pergunta que a maioria dos estudos
pula: **vale a pena comprar alguma coisa, e o quê**. Entre não fazer nada e
comprar tudo existem seis arranjos intermediários, e cada um resolve uma parte
diferente do problema por um preço diferente.

O passo **Meta** deixa ligar e desligar cada fonte, informar o grupo gerador e
digitar a conta de luz do cliente. O estudo então monta um cenário por
combinação e mede todos contra o mesmo referencial — **só a rede**, com a conta
que o cliente paga hoje.

| Fonte | Mexe na conta | Mexe no apagão |
| --- | --- | --- |
| **Solar** | Muito | **Nada** |
| **Bateria** | Quase nada, na tarifa simples | Muito |
| **Gerador** | Nada | Muito, e sem limite de horas |

As duas linhas contraintuitivas são as que mais mudam a conversa de venda:

**Solar sozinho não dá backup nenhum.** Inversor conectado à rede desliga
quando a rede cai — é exigência de anti-ilhamento da ABNT NBR 16149, não
limitação de marca. Vender solar como segurança energética é o engano mais
comum do setor, e o quadro o desfaz sozinho: a coluna de autonomia do cenário
"só solar" é zero, e a energia que falta no apagão é a mesma de quem não
comprou nada.

**Bateria na tarifa simples quase não economiza.** Sem diferença de tarifa
entre horários não existe arbitragem; o que sobra é recuperar o pedaço que o
Fio B retém da energia injetada, guardando o excedente do meio-dia para a
noite. Num caso real do repositório, a bateria acrescentou R$ 208/ano sobre um
sistema solar que já economizava R$ 34.316/ano. Quem paga a bateria, na tarifa
simples, é a resiliência — e por isso o estudo avisa em vermelho quando o custo
da interrupção fica em zero.

### O gerador supre; o que se decide é o custo

Grupo gerador não se dimensiona por autonomia — combustível se compra. Pedindo
um grupo sem potência, o estudo o dimensiona pela placa comercial (kVA a 0,8 de
fator de potência) que cobre o pico exigido pela carga essencial em 99% dos
dias, com folga. Com a potência resolvida, sobra a única pergunta que o grupo
tem de específico: **quanto custa cada hora ligada**. O dossiê traz a resposta
por duração de falta, em kWh queimados, reais por evento e reais por ano.

O grupo é o oposto exato da bateria: energia praticamente ilimitada, potência
de placa, e um custo por kWh que só aparece quando ele roda. No ilhamento ele
entra em paralelo ao inversor e cobre o que sol e bateria não deram — os picos
acima do que o inversor sustenta e, num apagão longo, tudo depois que o banco
esvazia. Energia grátis primeiro, combustível por último. E é por isso que o
que poupa combustível é o **sol**, não a bateria: cada kWh que o sol entrega ao
quadro ilhado é um kWh que o grupo não queima, enquanto a bateria, que o grupo
recarrega, pode até aumentar o consumo. O que ela compra é a transferência
instantânea e o pico que o grupo não aguenta.

Três detalhes que o modelo trata e que costumam ficar de fora:

* **O atraso de partida não é gratuito.** Entre a queda da rede e o grupo
  assumir a carga passam 10 a 30 segundos num QTA. Sem bateria, essa energia
  falta; com bateria, não. É a diferença entre um grupo e um nobreak, e ela
  aparece na simulação.
* **O grupo falha por potência, nunca por energia.** Carga acima da placa não
  é atendida por mais tanque que haja, e o estudo classifica a falha como
  potência — porque comprar combustível não resolve.
* **A folga de potência recarrega o banco.** É o que faz a dupla gerador +
  bateria atravessar dias sem que a bateria seja dimensionada para a energia
  toda: o grupo enche o banco na folga, e o banco devolve nos picos que o
  grupo não aguenta.

Em paralelo à rede, o grupo só é acionado se o kWh dele for mais barato que o
da distribuidora — o que quase nunca acontece com diesel. A regra é explícita
no código em vez de suposta.

### O preço do kWp não é um número só

O mesmo kWp custa coisas muito diferentes conforme a obra, e o que muda não é o
módulo: é estrutura, acabamento, prazo em prédio ocupado, ART, seguro e a
exigência de projeto executivo que um condomínio de alto padrão faz e um galpão
não faz. Um estudo que trata os dois pela mesma curva erra o investimento em
quase 50%, e o erro atravessa inteiro para o payback.

| Padrão | Referência em 100 kWp | Quando usar |
| --- | --- | --- |
| `basico` | R$ 2.700/kWp | telhado metálico, acesso livre, estrutura padrão |
| `padrao` | R$ 3.200/kWp | o caso médio do mercado de geração distribuída |
| `alto` | R$ 4.705/kWp | prédio ocupado, projeto executivo, prazo curto |

A curva tem ganho de escala — expoente −0,12 sobre a razão de potência —, então
o R$/kWp cai com o tamanho: os R$ 4.705/kWp de referência viram R$ 4.104/kWp
num sistema de 312 kWp.

**O nível `alto` é calibrado, não chutado.** O ponto de calibração é uma obra
real: 312 kWp fechados a R$ 1.280.400, ou R$ 4.104/kWp. Um teste automatizado
verifica que a curva continua passando por esse ponto, porque é dele que o
número tira a procedência que o documento afirma ter.

Cotação de verdade passa na frente de qualquer curva: informando o
investimento, o padrão é ignorado. E o dossiê **sempre declara** de onde veio o
R$/kWp — apresentar payback sem dizer a que preço ele foi calculado é pedir
confiança em vez de dar informação.

### O piso da conta

Nenhum sistema leva a fatura a zero: o custo de disponibilidade (30 kWh
monofásico, 50 bifásico, 100 trifásico) é cobrado todo mês, e o piso é aplicado
mês a mês, não sobre o total do ano. Um sistema que zera o consumo de dez meses
e deixa dois acima do piso paga doze mínimos, não a média deles.

### Autonomia zero não é o mesmo que inútil

O quadro traz duas colunas de resiliência de propósito. **Autonomia garantida**
exige atravessar a falta inteira em 95% dos casos, no pior par de estação e
hora — é a promessa que se pode assinar, e ela empata em zero para todo mundo
sempre que a carga essencial é pesada. **Energia que falta** mede o estrago
real numa interrupção média, e separa os arranjos em qualquer situação: um
grupo gerador que cobre 95% da energia mas afunda na partida de um motor
aparece com autonomia zero e com "sem energia" perto de zero.

Tarifa simples apenas, por enquanto. Tarifa branca e Grupo A mudam a conta da
bateria por completo — a arbitragem passa a existir — e exigem posto tarifário
e demanda contratada, que `aurum/bateria/economia.py` já modela para o conjunto
recomendado.

---

## O dossiê

O botão do passo 6 gera um ZIP com o estudo inteiro em LaTeX, compilado em PDF
quando há uma distribuição LaTeX na máquina:

```
estudo/estudo.tex     documento completo, com capa e sumário
estudo/estudo.pdf     o mesmo, compilado
estudo/resumo.md      resumo em Markdown, para colar em e-mail
estudo/figuras/       todas as figuras em PNG
estudo/tabelas/       todas as tabelas em CSV
```

Sem LaTeX instalado, o pacote sai com o `.tex` e as figuras, que compilam no
Overleaf sem ajuste nenhum.

As seções, na ordem em que uma pergunta depende da anterior:

1. **O que se conclui** — a resposta antes de qualquer tabela.
2. **O telhado** — foto aérea com os módulos, croqui cotado, medidas.
3. **O recurso solar** — série horária, produtividade, geração por estação.
4. **A demanda elétrica** — o método de Monte Carlo e a análise completa:
   distribuição dos picos com a incerteza do método, composição do pico por
   equipamento, os três fatores, curva de duração, composição por ambiente,
   comportamento sazonal e excedência de pico.
5. **O sistema fotovoltaico** — módulo, características elétricas e a **memória
   de cálculo do arranjo**, passo por passo, para o equipamento escolhido.
6. **O armazenamento** — triagem por potência, resiliência, alternativas.
7. **Vida útil e degradação** — a autonomia com data.
8. **Cenários** — solar, bateria e gerador ligados e desligados, cada arranjo
   contra a conta de hoje, com payback e valor presente.
9. **Análise econômica** — fluxo de caixa e premissas do conjunto recomendado.
10. **Premissas, limitações e procedência** — o que é datasheet, o que é
    estimativa, e o que o estudo **não** afirma.
11. **Anexo — o levantamento de cargas** — todo equipamento que entrou na
    simulação, ambiente por ambiente, com potência, quantidade, probabilidade,
    fator de demanda e janela de uso em horário de relógio. É o que torna o
    estudo auditável: um pico de 60 kW não se discute, mas "chuveiro elétrico,
    5.500 W, 40 unidades, 90%, das 06:00 às 09:00" se discute com o cliente.

Num estudo **sem energia solar** as seções 2, 3 e 5 não existem. Deixar a curva
de geração num estudo de cliente que não vai instalar painel faz quem lê
entender que ela faz parte da proposta.

A última seção não é formalidade. Um estudo que mistura número de datasheet com
ordem de grandeza de mercado e não diz qual é qual transfere ao leitor um risco
que ele não tem como avaliar.

---

## Estudo de baterias

Módulo separado, construído sobre o **D² — Demanda e Dados**
(`matheusviannapr/DemandaDados`). Responde a pergunta que o dimensionamento
fotovoltaico não responde: *quando a rede cair, essa instalação atravessa?*

O ponto é que "atravessa" não é uma pergunta só. Depende da **duração** da
falta, da **hora** em que ela começa (um apagão às 6 h tem o dia inteiro de sol
pela frente; um às 18 h tem a noite inteira antes), e da **estação** — que mexe
no sol e na carga ao mesmo tempo. O estudo varre as três dimensões:

```
        carga (D²)                    geração (PVGIS horário)
  Monte Carlo por estação        8.760 h de anos reais no local
   distribuição de curvas         com persistência de nebulosidade
            └──────────────┬──────────────┘
                    despacho ilhado
        (energia útil, potência contínua, surto)
                           ↓
   1, 2, 3, 6, 12, 24, 36 h × 24 horas de início × 4 estações
                           ↓
    probabilidade de atravessar · energia não suprida ·
    tempo até a primeira falha · causa (energia ou potência)
```

### As três grandezas que costumam virar uma só

O caso "não consigo, com um inversor de 5 kW, alimentar uma carga de 8 kW" tem
três respostas diferentes, e o estudo dá as três separadas:

| Grandeza | Pergunta que responde | Onde aparece |
| --- | --- | --- |
| P(pico diário > limite) | com que **frequência** acontece | `prob_pico_diario_acima_nominal` |
| Fração do tempo acima | se é pico afiado ou **patamar** | `fracao_do_tempo_acima_nominal` |
| Duração do evento | **quantos minutos** seguidos dura | `duracao_media_min`, `duracao_p95_min` |

A terceira é a que se compara contra `duracao_pico_s` do inversor. Um inversor
de 5 kW com sobrecarga de 10 kW por 10 s atende uma partida de motor de 8 kW
por 2 s e **não** atende um chuveiro de 5,5 kW ligado por 8 minutos — potência
menor, e é o segundo que derruba. Nenhum número único de "potência" separa os
dois casos.

Do mesmo modo, a falha durante o apagão é classificada em **energia** (o banco
esvaziou) ou **potência** (o inversor não deu conta, com o banco ainda cheio).
As soluções são opostas: a primeira pede mais kWh, a segunda pede inversor
maior ou corte seletivo de carga. Um relatório que só diz "não aguentou" manda
comprar a coisa errada metade das vezes.

### Uso

```bash
python -m aurum.cli bateria catalogo --criar
```

```bash
python -m aurum.cli bateria excedencia --cargas cargas.xlsx --horaria
```

```bash
python -m aurum.cli bateria estudo --lat -25.43 --lon -49.27 --nome "Hotel Central" --cargas cargas.xlsx --essenciais "Área Comum,Cozinha" --kwp 40 --autonomia 12 --confiabilidade 0.95 --custo-interrupcao 30 --saida estudos/hotel-central
```

O primeiro comando cria `BDBaterias.xlsx` com a semente do catálogo. O segundo
é a triagem barata: não simula apagão nenhum, só cruza a distribuição de picos
com o nominal e a sobrecarga de cada inversor. O terceiro roda o estudo inteiro.

`--cargas` recebe a planilha de cenário do D² (uma aba por cômodo). Sem ela, o
estudo roda no cenário de demonstração — útil para ver o programa funcionando,
inútil para proposta.

`--essenciais` é o que separa um estudo realista de um exercício: são os
cômodos ligados ao **quadro de backup**. Um hotel que põe só elevador, bombas e
circulação no backup precisa de um terço do inversor que a carga total pediria.
Sem esse recorte, o estudo dimensiona o prédio inteiro em ilha, que ninguém
compra.

`--custo-interrupcao` é quanto vale, em reais, 1 kWh que faltou. É o parâmetro
mais sensível da análise econômica e o único que não se estima de fora:
pergunte ao cliente quanto custa uma hora de câmara fria parada. Deixado em
zero, a resiliência entra no relatório mas não entra no fluxo de caixa.

Como biblioteca:

```python
from aurum.bateria import ConfiguracaoEstudo, MalhaApagao, executar_estudo
from aurum.bateria.relatorio import escrever_relatorio
from aurum.demanda import carregar_cenario_excel

comodos = carregar_cenario_excel("cargas.xlsx")
estudo = executar_estudo(ConfiguracaoEstudo(
    latitude=-25.43, longitude=-49.27, nome="Hotel Central",
    comodos=comodos,
    instancias_por_comodo={"Quarto Standard": 40, "Área Comum": 1},
    comodos_essenciais=["Área Comum"],
    potencia_fv_kwp=40.0, autonomia_alvo_h=12.0,
    malha=MalhaApagao(duracoes_h=(1, 2, 3, 6, 12, 24, 36), amostras=200),
))
escrever_relatorio(estudo, "estudos/hotel-central")
```

### O que sai

Um pacote com `estudo.md`, `estudo.tex` (seção pronta para entrar na proposta
que o `aurum.proposal` já gera), 16 planilhas CSV e seis figuras — entre elas o
**mapa de atendimento** (hora de início × duração, uma matriz por estação) e a
**trajetória do estado de carga** num apagão de 36 h com faixa P5–P95, que
mostra o banco drenando pela noite, o sol recarregando no dia seguinte e a
segunda noite decidindo o resultado.

### Critério da recomendação

O conjunto de **menor CAPEX que cumpre a autonomia alvo com a confiabilidade
pedida, no pior par (estação, hora)**. Quando nenhum cumpre, o estudo diz isso
e mostra o que faltou, em vez de eleger o menos ruim em silêncio. O critério é
declarado; não é um índice composto inventado.

### Premissas e limites

- **A malha padrão são 100.800 cenários por conjunto** e roda em segundos: o
  despacho é vetorizado e as 24 horas de início de cada (estação, duração) são
  simuladas num lote só.
- **O catálogo semente é ordem de grandeza, não datasheet.** Todo item de
  `BDBaterias.xlsx` sai marcado `a conferir`, e o relatório conta quantos ainda
  estão assim. Corrija com os documentos dos seus fornecedores antes de usar em
  proposta comercial.
- **A degradação não modela temperatura.** A dependência é forte — a perda de
  calendário praticamente dobra a cada 10 °C acima de 25 °C — e modelá-la sem
  saber onde o banco será instalado daria precisão falsa. Banco em casa de
  máquinas sem ventilação: dobre `fade_calendario_ano`.
- **Sem mercado livre nem serviços ancilares** na economia. Nenhum deles está
  acessível ao cliente típico de geração distribuída no Brasil hoje, e
  incluí-los inflaria o retorno com receita que não existe.
- **Sem rede, a série horária é sintética** — forma por geometria solar, energia
  ancorada no total mensal — e o relatório carrega o aviso. Use `--exigir-pvgis`
  para falhar em vez de cair para ela.

---

## O que foi corrigido

Os bugs abaixo estavam nos programas originais e foram encontrados durante a
unificação. Cada um tem teste automatizado que impede o retorno.

### Demanda (D² — Demanda e Dados)

**Janela de uso que atravessa a meia-noite some do perfil.** No D²,
`Equipamento.simula_carga` faz `carga[inicio:fim] += potência`. Uma janela como
`"18:00 as 06:00"` é parseada como `(1080, 360)`, e `carga[1080:360]` é uma
fatia **vazia** em NumPy: o equipamento desaparece do dia inteiro, sem erro e
sem aviso. Iluminação de área comum, bombas noturnas, câmaras frias e sistemas
de segurança — exatamente as cargas que um sistema de backup existe para
atender — somam **zero**. No cenário de demonstração do próprio D², a área
comum de um hotel passa de 2,25 para **16,65 kWh/dia** com a correção (7,4×).

A correção trata o dia como cíclico, que é o que um perfil diário é, e vive em
`aurum/demanda/correcoes.py` — fora da extração literal do núcleo, ligada por
um gerenciador de contexto para que o número original continue reproduzível.
**Vale corrigir também no repositório do D²:** qualquer perfil já gerado lá com
equipamento noturno está subestimado.

**Janela noturna com duração sorteada levanta exceção.** O mesmo problema pela
outra porta. Equipamento de duração variável dentro de uma janela que vira a
meia-noite — um degelo de câmara fria que roda 2 h entre 22 h e 2 h — passa por
`gerar_intervalo_uso`, que levanta `ValueError` quando o fim vem antes do
início. Não some em silêncio: quebra a simulação inteira, e o usuário fica sem
saber qual dos dezenas de equipamentos causou. A correção estende a janela em
um dia; o índice resultante passa de 1.440 e o dia cíclico dá conta.

**Tipo `dinâmico` sem duração vira uma hora à meia-noite.** Não é corrigido —
é *detectado*. Nesse caso o D² cai em `parse_intervalo_dinamico_split`, que
espera o formato legado `"Início entre 10:30-14:00, duração 2"`. Recebendo
`"08:00 as 18:00"`, a expressão regular não casa e a função devolve o padrão
`[(0, 60)]`: uma hora de operação à meia-noite, em silêncio, para um
equipamento que devia rodar o dia inteiro. É o pior dos três, porque o número
resultante parece plausível. `Cenario.validar` bloqueia antes de simular.

### Dimensionamento

**Azimute invertido no hemisfério sul.** O padrão era 180° (Sul) na convenção
do PVWatts, com o texto de ajuda dizendo que 180° era o Norte. No Brasil o
ótimo é 0° (Norte). Medido contra a base do PVGIS em Curitiba: 967 kWh/kWp/ano
voltado ao Sul contra 1.304 voltado ao Norte — **26% de geração a menos**, o
que inflava o sistema dimensionado e o investimento na mesma proporção.

**Tensão de string sem correção de temperatura.** A janela de MPPT era
verificada com a Voc de catálogo, medida a 25 °C. A Voc sobe cerca de 7% a
0 °C. Todos os quatro inversores do catálogo aceitavam uma string a mais do
que suportam: no MAX 75KTL3, 27 módulos dariam 1.164 V numa manhã fria, contra
o limite de 1.100 V. Isso queima o inversor.

**Geração calculada para o sistema errado.** A energia anual era calculada
para a potência *necessária*, não para a potência de cada alternativa
dimensionada, então todas as linhas do comparativo mostravam o mesmo número.

**Quantidade de módulos por área bruta.** A conta era
`área × 0,70 ÷ área do módulo`, que dá o mesmo resultado para um retângulo e
para um telhado em L de mesma área. O empacotamento geométrico mostra 22% de
diferença entre os dois. Além disso, as dimensões do módulo eram fixas em
2,279 × 1,134 m para qualquer modelo.

### Proposta em LaTeX

**Escape com barra dupla.** As substituições eram declaradas como `r"\\&"`,
que em Python é a string `\\` seguida de `&` — uma quebra de linha do LaTeX
colada num E comercial. Os dez caracteres especiais estavam assim.

**Escape duplo de conteúdo formatado.** Células eram montadas como
`f"R\\$ {v:,.2f}"` e depois passavam pelo escapador, que convertia a barra em
`\textbackslash{}`. Toda tabela de valores saía corrompida.

**f-string esquecida.** A linha do zero do gráfico de VPL era escrita numa
string comum que parecia f-string, e o texto `{len(sym_coords)}` ia
literalmente para o arquivo `.tex`, impedindo a compilação.

**Tarifa média invertida.** `consumo ÷ custo` em vez de `custo ÷ consumo`.

**Números em formato americano** num documento em português (`1,234.56`).

### Economia

**Lei 14.300/2022 ignorada.** O modelo valorava 100% da energia gerada pela
tarifa cheia. A lei institui cobrança gradual do Fio B sobre a energia
injetada. No caso testado, ignorar isso **superestimava o VPL em 15%**.

**Sensibilidade que não sensibilizava.** Os cenários de tarifa apenas
multiplicavam a economia do ano 1; payback, VPL e TIR não se moviam. Agora
todo o fluxo é recalculado.

**Economia acima do consumo.** Nada impedia que a economia projetada
excedesse a conta de luz do cliente.

### Série horária de geração

**A série do PVGIS vinha em UTC.** O parâmetro `localtime=1` é enviado e o
serviço o ignora: em Curitiba, o pico de geração caía às 15 h, três horas
depois do meio-dia solar. Três horas de erro não são detalhe — deslocam toda a
sobreposição entre geração e carga, e com ela o autoconsumo, o estado de carga
no início do apagão e a economia calculada. A série passou a ser reindexada
pelo fuso derivado da longitude, com o resultado conferido contra o próprio
dado: se o centro de massa da geração não cair perto do meio-dia, o estudo
avisa em vez de seguir calado.

**Bateria com carga de graça todo dia.** No balanço anual de cada cenário, o
banco começava cada dia simulado com metade da energia útil. Sem sol e sem
carga pela rede, isso virava economia fantasma: o cenário "bateria sem solar"
aparecia economizando o que não economiza. Agora cada estação roda uma passada
de aquecimento antes de contar, e a segunda começa onde a primeira terminou —
que é o regime permanente de um banco em uso diário.

### Despacho do gerador

**A transferência do QTA reprovava o arranjo inteiro.** Os 30 segundos entre a
queda da rede e o grupo assumir a carga eram arredondados para o passo inteiro
da simulação — cinco minutos —, e como qualquer déficit reprova o cenário, todo
grupo gerador saía com **autonomia zero**, por maior que fosse. O intervalo
agora é modelado como fração do passo, a energia perdida entra no ENS separada
em `ens_transferencia_kwh`, e o cenário não é reprovado: um grupo com QTA leva
de 10 a 30 s para assumir, isso é dado de catálogo, não defeito do arranjo.

### Interface

**O multiselect de ambientes essenciais voltava sozinho.** O widget usava
`default=`, que reimpõe o padrão a cada re-execução; como lista vazia é falsa em
Python, `[] or cenario.essenciais or todos` caía no terceiro termo, e desmarcar
o último ambiente devolvia a lista inteira. O sintoma era ter de clicar duas
vezes para tirar um ambiente. Com `key=`, o estado é do widget.

**O mapa abria no lugar errado.** O `st_folium` guarda centro e zoom por chave,
então um mapa criado antes de o usuário localizar o endereço continuava
mostrando o lugar antigo, por mais que `folium.Map(location=...)` pedisse outro.
A chave passou a carregar as coordenadas: coordenada nova, mapa novo.

**O valor da resiliência era sempre zero.** O campo "quanto custa 1 kWh que
faltou" nascia em zero — e zero não é ausência de dado, é a afirmação de que
ficar sem energia não custa nada, o que nenhuma instalação com quadro de backup
acredita. Com ele zerado, a bateria aparecia em todo estudo como investimento
sem retorno. Agora cada segmento traz uma ordem de grandeza (R$ 8/kWh numa
escola, R$ 120 num frigorífico, R$ 150 num hospital), marcada como sugestão: a
pergunta na tela passou de *"que valor é esse?"* para *"esse valor está certo?"*.

### Documento

**Uma letra grega derrubava o PDF inteiro.** O `pdflatex` com `inputenc utf8`
estoura com "Unicode character not set up for use with LaTeX" e não gera saída
nenhuma — e a mensagem não diz de onde veio o caractere. Aconteceu duas vezes:
com um `β` na memória de cálculo, e com o `ηRT` da ressalva das baterias Lynx,
que entrou pela coluna de procedência. A tradução passou a viver no `esc`, que é
a fronteira por onde todo dado externo passa, em vez de em cada lugar que monta
texto: assim a garantia vale também para o que ainda não foi escrito.

**O `%` da URL comentava a linha da tabela.** A coluna de procedência
percent-codificava os espaços do nome do arquivo (`GW_ET%20PLUS+...`), e `%` em
LaTeX comenta o resto da linha — levando junto o `\\` que fecha a linha da
tabela. Agora sai escapado, e a URL para antes da nota que o catálogo acrescenta
depois de `|`.

### Prospecção

**Três implementações divergentes** do mesmo pipeline (`main.py`, `src/` e
`aurum_lead_mapper_app.py`), com filtros e pontuações diferentes.

**Deduplicação O(n²)** comparando todos os telhados contra todos — minutos numa
cidade. Agora usa índice espacial.

**Azimute inferido pelo ângulo matemático** (`atan2(dy, dx)`, medido do Leste)
em vez do azimute geográfico (`atan2(dx, dy)`, medido do Norte), e depois
reduzido a 0–180°, o que tornava impossível detectar orientação Norte.

**Exportação GeoJSON quebrada:** o dicionário de tags do OSM ia como
propriedade e o driver não conseguia serializá-lo.

**Contatos descartados.** O saneamento de tags mantinha só nove chaves e
jogava fora `website`, `phone`, `operator` e `contact:*` — justamente os
dados comerciais que a prospecção precisa.

---

## Filtro por tipo de local

Quatorze segmentos, escolhidos por como a prospecção de energia solar é
realmente organizada — e não pela taxonomia do OpenStreetMap. Na interface é
uma seleção múltipla; na linha de comando, `--alvo agro,educacao`.

| Chave | Segmento | Peso | O que entra |
| --- | --- | ---: | --- |
| `agro` | Agro e agroindústria | 20 | Granjas, aviários, silos, armazéns rurais, laticínios |
| `frigorifico` | Frigorífico e câmara fria | 20 | Frigoríficos e centrais de resfriamento |
| `industrial` | Indústria | 20 | Fábricas, metalúrgicas, plantas de processo |
| `supermercado` | Supermercado e atacado | 19 | Supermercados, atacarejos, centrais de abastecimento |
| `saude` | Saúde | 18 | Hospitais, clínicas, laboratórios |
| `comercio` | Comércio e varejo | 17 | Lojas, shoppings, concessionárias |
| `educacao` | Escolas e universidades | 16 | Escolas, creches, faculdades |
| `logistica` | Logística e armazenagem | 16 | Galpões, centros de distribuição |
| `escritorio` | Escritórios e corporativo | 15 | Sedes administrativas, coworkings |
| `publico` | Público e institucional | 14 | Prefeituras, órgãos públicos, quartéis |
| `hotelaria` | Hotelaria | 13 | Hotéis, pousadas, resorts |
| `esporte_lazer` | Esporte e lazer | 12 | Ginásios, clubes, estádios, academias |
| `condominio` | Condomínios residenciais | 11 | Prédios de apartamentos com área comum |
| `religioso` | Templos e igrejas | 8 | Igrejas, templos, salões paroquiais |

O peso é a contribuição do segmento na pontuação do lead. Reflete três coisas:
consumo por metro quadrado, alinhamento entre consumo e horário de geração, e
facilidade de decisão. Agro pontua no topo porque irrigação, ventilação de
aviário e câmara fria consomem muito, consomem de dia e têm um dono só; igreja
fica no fim porque o uso é à noite e no fim de semana.

Cada segmento também tem faixa de consumo (kWh/m²/mês), fração de autoconsumo
e fator de obstáculos de cobertura próprios. Uma estufa, por exemplo, recebe
fator 0,35 — a cobertura precisa passar luz para a lavoura.

### Por que o uso do solo importa

No mapeamento brasileiro a maioria das edificações é `building=yes`, sem dizer
o que é. Filtrar só pelas tags do prédio devolveria quase nada. Por isso a
prospecção baixa também as áreas de uso do solo da região e classifica pelo
que contém o prédio.

Medido em campo:

- **Cidade Industrial de Curitiba** — não identificados caem de 112 para 81, e
  a indústria sobe de 24 para 55 telhados.
- **Bacia leiteira de Castro/Carambeí (PR)**, filtro `agro` — **0 telhados sem
  o contexto, 24 com ele** (49.419 m²). Todos os vinte e quatro são
  `building=yes` dentro de um `landuse=farmyard`. Sem essa consulta extra, o
  filtro de agro simplesmente não funcionaria no interior.

Uma tag no próprio prédio sempre vence o contexto: uma escola dentro de área
industrial continua sendo escola.

### Freio de área

Filtrar por tipo convida a varrer o município inteiro — "quero todo o agro de
Carambeí". Carambeí tem 1.262 km²: 308 consultas ao OpenStreetMap e dezenas de
minutos. Acima de 400 km² a varredura é recusada, com a conta na mensagem:

```
A região tem 1.262 km², o que exigiria cerca de 308 consultas ao
OpenStreetMap e muitos minutos de espera.
Recorte em bairros ou distritos (o distrito industrial costuma ser o que
interessa), ou informe um bbox menor.
```

A recusa vem **antes** de qualquer consulta de rede. Para varrer assim mesmo:
`--area-grande` na linha de comando, ou a caixa "Permitir região muito grande"
na interface. O botão *Conferir localização e tamanho* mostra área, número de
consultas e tempo estimado antes de você começar.

### O segmento muda o cálculo, não só a lista

Classificar por segmento só vale se o dimensionamento mudar junto. Cada
segmento tem um tipo de edificação de referência, usado quando o prédio não
declara o próprio tipo — o caso da maioria. Um galpão `building=yes` dentro de
um `landuse=farmyard` é calculado como galpão rural:

| | genérico | reconhecido como agro |
| --- | ---: | ---: |
| Consumo | 9,0 kWh/m²/mês | 5,0 kWh/m²/mês |
| Autoconsumo | 60% | 65% |
| Obstáculos de cobertura | 82% | 88% |

O segmento e o uso do solo são gravados no `leads.geojson` e restaurados na
segunda etapa. Sem isso a proposta voltaria a ser "não identificado" na hora
de calcular, mesmo tendo sido corretamente classificada na prospecção — foi
exatamente o que aconteceu antes de os testes cobrirem esse caminho.

### Interações que o filtro trata

- **Condomínio** ignora o piso de área residencial que existe para descartar
  casa isolada — senão o filtro esvaziaria justamente o alvo.
- **Casa isolada** nunca entra em `condominio`: não tem área comum nem síndico.
  Se aparecer numa busca sem filtro, pontua no máximo 5 em uso.
- **Não identificado** é um segmento selecionável. Filtrar por tipo descarta
  essas edificações, e a interface avisa quando isso acontece.

---

## Decisões técnicas

**PVGIS como fonte primária de irradiação.** O PVWatts exige chave de API e a
`DEMO_KEY` tolera poucas dezenas de requisições por hora, o que não sustenta
prospecção em lote. O PVGIS (Comissão Europeia) usa a base SARAH3 derivada de
satélite, cobre o Brasil e não tem chave nem limite. O PVWatts continua como
segunda opção e há estimativa offline por latitude como último recurso, sempre
declarada na proposta.

**Uma consulta de irradiação por região, não por telhado.** A geração é linear
na potência instalada, então basta simular 1 kWp e escalar. Com lat/lon
arredondados para 1,1 km, uma região inteira consome poucas chamadas.

**Fator de obstáculos de cobertura.** O polígono do OSM é a projeção do
prédio, não uma planta de cobertura: não mostra casa de máquinas,
climatização, claraboias nem corredores. Um fator por segmento (0,70 a 0,88)
desconta isso, e o número aparece na proposta para ser contestado.

**Escolha do módulo por custo, não por potência.** No galpão de teste, o
arranjo de maior kWp exigia 975 kW de inversor para 866 kWp (razão CC/CA de
0,89), enquanto a alternativa de 820 kWp precisava de 635 kW (razão 1,29) —
5% menos geração por 35% menos investimento em inversor.

**Cinco espelhos do Overpass, com desvio automático.** Durante os testes,
três dos oito espelhos públicos conhecidos estavam simultaneamente fora do
ar. Um espelho que falha sai da rotação por alguns minutos, e o timeout de
conexão é curto (10 s) enquanto o de leitura permanece longo (180 s) — host
morto cede a vez em segundos, consulta em andamento tem o tempo que precisar.

**Detecção de espelho dessincronizado.** Um espelho com o banco quebrado
responde HTTP 200, envelope bem formado e **zero elementos** — indistinguível
de uma região sem edificação. O sinal está em `osm3s.timestamp_osm_base`:
numa instância sadia é ISO-8601, e na degradada vinha como `"116621"`. Sem
essa checagem o vazio entrava no cache e a região ficava "sem telhados" pelos
30 dias de validade do cache. Foi exatamente o que aconteceu durante o
desenvolvimento, e por isso o teste existe.

**Empacotamento vetorizado.** O teste de encaixe posição a posição levava
mais de 4 s por galpão grande, multiplicado por quatro modelos de módulo e por
todos os telhados do lote. A versão em lote com `shapely.contains` sobre um
array de candidatos faz o mesmo em 0,4 s — o lote de 30 telhados caiu de
cerca de 90 s para 4,4 s.

---

## Limitações

Nenhuma proposta gerada aqui deve ir para o cliente sem verificação. Em
particular:

- **A geometria vem do OpenStreetMap**, mapeamento colaborativo. A cobertura
  varia muito por região e a área é a projeção horizontal do prédio.
- **O consumo é estimado por segmento**, não lido de fatura. A faixa dentro do
  mesmo segmento é larga; o número serve para dimensionar a conversa, não o
  contrato.
- **Sombreamento do entorno não é avaliado.** Prédios vizinhos, torres e
  vegetação não entram no cálculo.
- **A estrutura da cobertura não é avaliada.** Capacidade de carga, estado da
  telha e fixação exigem laudo.
- **Clientes do Grupo A** têm cobrança de demanda contratada, que o modelo
  simplificado não trata.
- **Os dados de contato vêm do OSM** e podem estar desatualizados ou pertencer
  a um vizinho, quando obtidos por proximidade. A proposta declara a fonte de
  cada campo.

Toda proposta traz uma seção de ressalvas com esses pontos e os avisos
específicos do caso.

---

## O que ficou de fora

O Aurumcalc original tinha um segundo fluxo, para cliente **já conhecido**,
com a conta de luz em mãos. Ele não foi transportado, porque depende de dados
que a prospecção não tem:

| Recurso | Por que ficou fora |
| --- | --- |
| Análise horária (8.760 h) | Exige curva de carga; na prospecção não há fatura |
| Calibração de perfil de carga | Idem — precisa da conta para ajustar a curva |
| Peak shaving e load shifting | Dependem da curva horária |
| Grupo A com ponta/fora ponta e demanda contratada | Exige a fatura e a demanda contratada real |
| Banco de projetos (salvar/carregar cliente) | Fluxo de estudo detalhado, não de triagem |
| Exportação em Excel | Substituída pelo CSV do índice e pelo JSON por proposta |

O código original está preservado em `_legacy/Aurumcalc/` e a base de perfis
de carga em `data/load_profiles/` (hoje sem uso), de modo que reincorporar
esse fluxo é uma continuação natural: o caminho seria uma quinta etapa
"Estudo detalhado", que parte de uma proposta já gerada e aceita a fatura do
cliente para refinar consumo, tarifa e autoconsumo com dados reais.

---

## Testes

```bash
pytest tests/ -q              # 462 testes
pytest tests/ -q --rede       # inclui OpenStreetMap e PVGIS reais
```

O teste `test_proposta_compila_em_pdf` gera um documento completo e o compila
com pdflatex — é o que garante que os `.tex` entregues funcionam.

---

## Configuração

Ajustável por variável de ambiente:

| Variável | Efeito |
| --- | --- |
| `PVWATTS_API_KEY` | Chave do PVWatts (opcional; o PVGIS é a fonte primária) |
| `AURUM_CACHE_DIR` | Onde guardar o cache de Overpass, Nominatim e irradiação |
| `AURUM_OUTPUT_DIR` | Destino padrão das propostas |
| `AURUM_EQUIPMENT_XLSX` | Caminho da planilha de equipamentos |
| `AURUM_LATEX_PREAMBLE` | Arquivo `.tex` com o preâmbulo próprio da proposta |

### Identidade visual própria na proposta

Para trocar a aparência do documento sem mexer no código, aponte
`AURUM_LATEX_PREAMBLE` para um arquivo `.tex`:

```bash
set AURUM_LATEX_PREAMBLE=modelos\meu-preambulo.tex
python -m aurum.cli propor --regiao "..." --top 10
```

O arquivo vai de `\documentclass` até logo antes de `\begin{document}`. Ele
precisa definir as quatro cores usadas pelas seções (`azulaurum`,
`verdeaurum`, `vermelhoaviso`, `cinzaclaro`) e carregar `booktabs`,
`longtable`, `float`, `pgfplots`, `hyperref` e `textcomp`. Os comandos
`\cabecalhoempresa` e `\cabecalhoreferencia` são injetados depois e podem ser
usados no cabeçalho ou rodapé.

Para mudar o conteúdo e não só a forma, cada seção é uma função independente
em `aurum/proposal/sections.py`, montadas na ordem definida por
`montar_documento`.

Premissas técnicas e econômicas ficam em `aurum/config.py`, na classe
`SolarDefaults`.

### Catálogo de equipamentos

`BDFotovoltaica.xlsx`, com as abas `paineis_solares` e `Inversores`. Se as
colunas `comprimento_m` e `largura_m` não existirem, as dimensões são
derivadas da eficiência declarada — e a proposta avisa que foi assim.

O catálogo atual tem 4 módulos e 4 inversores. **Ampliá-lo melhora
diretamente o aproveitamento dos telhados**: nos testes, faltou inversor
compatível para até 25% das posições em alguns casos.
