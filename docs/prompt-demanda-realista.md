# Prompt — levar o modelo de demanda ao nível de dimensionamento robusto

Use este prompt quando a curva de carga que o PACE Calculator produz não
descrever a casa que se visitou: consumo alto demais, pico pontudo demais,
madrugada cheia demais, ou um resultado que "parece certo" sem que ninguém
consiga dizer de onde saiu.

Ele não é um checklist de estilo. É a lista dos erros que já custaram caro neste
software, escrita para que o próximo os encontre em minutos em vez de semanas.

---

## Como usar

Cole o bloco abaixo, junto com o backup da vistoria (`.json`), o PDF do estudo
gerado e o número que incomodou.

> Analise o modelo de demanda deste estudo. O consumo/pico está em **[valor]** e
> eu esperava algo em torno de **[valor]**. Percorra as sete frentes abaixo, na
> ordem. Em cada uma, **meça antes de opinar**: rode o ranking de energia por
> equipamento e traga números, não impressões. Ao final, separe explicitamente
> o que é **erro** (o modelo diz algo falso) do que é **premissa** (o modelo diz
> algo defensável de que eu posso discordar). Não ajuste nenhum fator para
> chegar ao número que eu disse esperar: se o modelo estiver certo e minha
> expectativa errada, diga isso.

---

## Frente 1 — Placa não é consumo

**A pergunta:** cada equipamento puxa mesmo a potência que está no catálogo?

O catálogo traz **potência de placa** e um fator de demanda padrão (0,8 para
quase tudo, 1,0 para o que fica ligado). Isso descreve carga resistiva que liga
e desliga. Descreve mal:

| Classe | Placa | Média real | Por quê |
|---|---|---|---|
| Split inverter | nominal | 0,45–0,55 | nominal por ~20 min, depois modula em 30–45% |
| Computador | fonte | 0,35–0,45 | a placa é da fonte, não do consumo |
| Cooktop de indução | soma das zonas | 0,30–0,35 | cozinha-se com uma ou duas zonas |
| Forno, ferro, toalheiro | nominal | 0,50–0,60 | ciclam no termostato depois de aquecer |
| Lava-louças, lavadora | nominal | 0,35–0,45 | só aquecem em parte do ciclo |
| Câmera PoE | 12 W | 0,45–0,55 | 12 W é o teto do padrão de alimentação |
| Switch PoE | orçamento PoE | 0,35–0,45 | a placa é o que ele **pode** entregar |
| Áudio | potência de pico | 0,30–0,40 | música não é onda quadrada |
| Resistência pura (torradeira, chuveiro) | nominal | 0,85–1,00 | aí a placa vale |

**O sintoma:** o erro é sempre para cima, é pequeno por item e some na
soma — dezenas de itens 30 a 60% acima do que puxam. Num caso real este item
sozinho respondeu por **35% do consumo mensal**.

**O teste:** para os cinco maiores itens do ranking, calcule
`potência × fd × probabilidade × duração média` e compare com o consumo típico
publicado do aparelho. Um split de 24.000 BTU rodando 5,5 h por noite consome
4 a 6 kWh; se o modelo disser 9, o fator está errado.

**A entrega:** todo fator declarado vem com o **motivo escrito ao lado**. Fator
sem motivo é chute com aparência de dado, e daqui a três meses ninguém sabe se
foi medido ou inventado.

---

## Frente 2 — O que nunca desliga

**A pergunta:** quanto da casa é carga permanente, e ela está certa?

Some tudo que roda em `00:00 as 23:59` com `FIXO_100%`. Numa casa com automação
isso costuma passar de **um terço do consumo**, e ninguém olha para lá porque
cada item é pequeno.

Dois erros específicos a procurar:

- **Contagem dupla de conjunto e conteúdo.** "Rack de automação / rede" é o
  armário com o que há dentro; listar também o switch, o roteador, o mesh e a
  central de automação conta os mesmos equipamentos duas vezes. Escolha a lista
  item a item — ela é o que a vistoria levanta e o que permite discutir
  criticidade — e descarte o agregado.
- **Equipamento que só consome quando age, modelado como contínuo.** A sirene do
  alarme está no catálogo como `FIXO_100%` em 24 h: 0,72 kWh/dia para um alarme
  que nunca disparou. A iluminação de emergência tem o mesmo vício. Estes são
  defeitos **do catálogo**, e valem para todo estudo que os traga.

---

## Frente 3 — Sazonalidade é frequência, não intensidade

**A pergunta:** no inverno o ar-condicionado liga mais fraco, ou não liga?

Aplicar o fator sazonal sobre o `fator_demanda` transforma −45% de inverno em
"roda as mesmas 5,5 horas todas as noites, a 55% da potência". O modelo passa a
pôr um platô de ar-condicionado em todas as madrugadas de julho.

As duas leituras dão **energia parecida** — multiplicação é comutativa — e
**curvas completamente diferentes**. Quem dimensiona banco de bateria pela
madrugada recebe uma carga de inverno que não existe.

Use `modo: "probabilidade"` em `aplicar_ajuste_sazonal`. E verifique que ele
**chega**: item de intervalo dinâmico com duração guarda a probabilidade dentro
do gerador de janelas, não num campo, e mexer no campo depois de construído não
faz nada. O sintoma desse bug é o mais traiçoeiro que há: as quatro estações
saem com energia idêntica com e sem a correção.

---

## Frente 4 — A rotina não é a mesma todo dia

**A pergunta:** o modelo sorteia **quando** a rotina acontece?

Sem isso, toda simulação de todo dia põe o almoço na mesma janela e a hora de
dormir na mesma hora. O perfil médio sai mais pontudo que o de qualquer casa
real, e quem o lê dimensiona transformador, cabo e inversor para uma sincronia
que não existe.

Ver `aurum.demanda.rotina`. Quatro decisões que não podem ser trocadas:

1. **O que se desloca é a rotina, não o aparelho.** Jitter independente por
   equipamento é pior que jitter nenhum: destrói a coincidência que de fato
   existe, porque a família janta junta.
2. **O atraso é do domicílio.** Cada dia sorteia um componente comum a todas as
   âncoras — é ele que faz um dia atrasado ser atrasado em tudo.
3. **A causalidade tem um sentido só.** Um jantar tarde empurra a hora de
   dormir; dormir tarde não adianta o jantar.
4. **Madrugada é regime, não cauda.** Quem vira a noite não janta 40 minutos
   mais tarde: dorme quatro horas mais tarde. Entra como mistura, com uma fração
   pequena dos dias.

**O que esperar, e o que desconfiar:** energia praticamente inalterada (deslocar
não cria nem destrói consumo — se cair, alguma janela saiu do dia, e isso é bug)
e cume do perfil médio menor. O pico do **dia** pode **subir**, e isso é
resultado, não defeito: com janelas fixas, jantar e ar-condicionado nunca se
sobrepunham porque o modelo os prendia em horários que não se cruzam.

---

## Frente 5 — Coincidência, e o que se dimensiona com ela

**A pergunta:** qual número dimensiona o quê?

- **Pico do dia (P95/P99)** dimensiona inversor e proteção. Sai do pior dia.
- **Perfil médio** descreve a conta de luz e a forma da curva. Sai da média
  ponderada dos cenários de uso.
- **Fator de coincidência** = cume do perfil médio ÷ pico do dia. Quanto menor,
  mais a rotina se espalha. Uma residência fica tipicamente entre 0,3 e 0,5;
  acima de 0,6 o modelo está sincronizando demais.

Trocar um pelo outro é o erro clássico e ele tem duas caras: solar dimensionado
pelo fim de semana fica grande demais, e banco dimensionado pela média não
atravessa o sábado.

**Verificar sempre:** as seções que escolhem **equipamento** têm de medir contra
o **mesmo dia**. Já aconteceu de a recomendação vir com 9,3 kWh e a seção de
escopos, medindo o levantamento cru em vez do dia que dimensiona, dizer que o
mesmo recorte cabe em 4,6 kWh.

---

## Frente 6 — A conferência contra o mundo

**A pergunta:** o que o levantamento diz bate com a conta de luz?

O estudo já avisa quando a divergência passa de 15%. Não silencie o aviso
mexendo em fatores até fechar: uma divergência de 20% entre carga levantada e
fatura pode ser levantamento inflado, pode ser fatura de um mês atípico, e pode
ser que a casa tenha mudado de rotina. Cada uma pede coisa diferente.

Âncoras úteis para residência brasileira, por mês:

| Perfil | kWh/mês | Observação |
|---|---|---|
| Apartamento sem ar, aquecimento a gás | 150–300 | |
| Casa média, chuveiro elétrico | 300–600 | o chuveiro sozinho pesa 100–200 |
| Alto padrão, água a gás, 3–4 splits | 900–1.500 | |
| Alto padrão, água elétrica, 3–4 splits | 1.400–2.000 | |
| Alto padrão com piscina aquecida ou sauna | 2.000+ | a piscina domina |

Se o estudo cair fora da faixa do perfil, **ache o item**, não o fator global.

---

## Frente 7 — Auditabilidade

**A pergunta:** um terceiro consegue refazer este número?

Para cada correção aplicada, o estudo tem de conseguir responder:

- qual item mudou, de quanto para quanto;
- se é erro ou premissa;
- quem decide (o motor, o vistoriador ou o cliente);
- o que aconteceria se a premissa fosse a outra.

Uma correção que muda o total e não aparece em lugar nenhum do documento é pior
que o erro que ela corrige: o erro pelo menos era reproduzível.

---

## O que **não** fazer

- **Não ajuste fatores para chegar a um número esperado.** Se o modelo e a
  expectativa discordarem, uma das duas está errada, e descobrir qual é o
  trabalho. Ajustar até fechar destrói a única coisa que o modelo tinha de bom.
- **Não misture correções numa medição só.** Cada peça se mede sozinha, ou
  ninguém sabe qual resolveu o quê — e a próxima pessoa terá de desfazer todas
  para descobrir.
- **Não encolha carga real.** Ar-condicionado, cozinha e computador de uma casa
  de alto padrão consomem mesmo. Trocar levantamento por expectativa entrega um
  sistema subdimensionado, e o cliente descobre no primeiro apagão.
- **Não deixe correção sem teste.** Especialmente as de fio: nesta base, duas
  vezes um argumento novo caiu na chamada errada por âncora curta de edição, e
  numa delas a peça inteira ficou desligada em silêncio.

---

## Estado atual e o que falta

Implementado:

- deslocamento de rotina com âncoras correlacionadas e regime de madrugada
  (`aurum.demanda.rotina`);
- sazonalidade por frequência de uso (`modo: "probabilidade"`), alcançando
  também o intervalo dinâmico;
- fator de utilização declarado item a item, com o motivo em cada linha.

Falta, e é o próximo nível:

1. **Fator de utilização no motor, por classe de tecnologia.** Hoje ele é
   declarado por estudo. O lugar dele é o catálogo: cada equipamento carrega sua
   classe (inverter, resistivo, fonte chaveada, motor), e a classe carrega a
   faixa de utilização com procedência. Enquanto isso não existir, cada estudo
   vai redescobrir os mesmos fatores.
2. **Correção do catálogo do aplicativo de vistoria.** Sirene e iluminação de
   emergência como carga contínua estão em `vistoria/dados.js` e contaminam todo
   levantamento novo.
3. **Perfil de rotina a partir da vistoria.** Hoje se escolhe entre dois perfis
   fixos. As perguntas que separariam um do outro — horário de trabalho, quem
   fica em casa, se alguém trabalha à noite — cabem em três campos do
   aplicativo.
4. **Calibração contra medição real.** Nada aqui foi conferido contra um
   medidor. Uma única casa instrumentada por um mês transformaria metade destas
   premissas em dado.
