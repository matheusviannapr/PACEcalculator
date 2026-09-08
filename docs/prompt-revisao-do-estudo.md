# Prompt de revisão do estudo de energia da PACE

Cole o texto abaixo junto com o PDF do estudo. Ele foi calibrado pelos defeitos
que apareceram de verdade na revisão dos primeiros dossiês — cada item da lista
existe porque um deles passou despercebido pelo menos uma vez.

---

## O prompt

> Você vai revisar um estudo técnico-comercial de energia — dimensionamento de
> sistema fotovoltaico e de armazenamento para uma instalação real. O documento
> é gerado por um software a partir de uma vistoria em campo, e vai às mãos de
> um cliente que **não é engenheiro** e que vai decidir uma compra de dezenas de
> milhares de reais com base nele.
>
> O que está em jogo: um número plausível e errado é pior que um número
> ausente, porque não se anuncia. Um cliente não tem como distinguir ruído de
> simulação de erro de cálculo lendo um relatório, e não deveria precisar.
>
> Faça uma análise completa e devolva achados acionáveis. Trabalhe nesta ordem,
> que é a ordem da gravidade.
>
> ### 1. Coerência numérica entre seções
>
> Este é o item mais importante. Percorra o documento inteiro e monte a lista
> de todo número que aparece **mais de uma vez** ou que é **derivado** de
> outro. Para cada um, confira se bate.
>
> - O consumo diário do resumo é o mesmo da tabela de cenários e o mesmo que
>   sustenta a conta de economia?
> - O investimento do resumo é o mesmo do quadro de arranjos e o mesmo da
>   tabela de cenários?
> - A geração anual bate com potência × produtividade declarada?
> - O payback bate com investimento ÷ economia?
> - A soma das parcelas da composição bate com o total?
> - As porcentagens fecham em 100%?
>
> Diferenças de arredondamento (até ~0,5%) são aceitáveis; diga que conferiu e
> que fecham. Qualquer coisa acima disso é achado, mesmo que pequena — foi uma
> diferença de 1,6% entre duas páginas que revelou que uma tabela inteira
> estava sendo calculada sem os ajustes sazonais que o resto do documento
> aplicava.
>
> ### 2. Rótulos que conflatam grandezas diferentes
>
> Procure o mesmo rótulo usado para coisas diferentes, e grandezas diferentes
> sob o mesmo nome. Exemplos reais que passaram:
>
> - "Retorno do investimento" aparecendo duas vezes com valores diferentes —
>   uma para o sistema inteiro, outra só para o banco de baterias.
> - "Consumo diário" ora significando o dia que dimensiona o equipamento (o
>   pior), ora o consumo médio que a conta de luz vê.
> - "Investimento" ora sendo só o armazenamento, ora o sistema completo.
>
> Para cada caso: o rótulo tem de dizer **de que** ele está falando, ou os dois
> números têm de virar um.
>
> ### 3. Premissa arbitrada apresentada como medida
>
> Separe o documento em três origens de dado e diga a qual cada número
> pertence:
>
> - **medido ou cotado** — datasheet, tabela de preço com data, série do PVGIS,
>   levantamento em campo;
> - **calculado** — sai dos anteriores por conta declarada;
> - **arbitrado** — alguém escolheu, e outra pessoa escolheria diferente.
>
> Todo número arbitrado tem de estar identificado como tal **onde ele
> aparece**, e não só num anexo de premissas. Aponte os que não estão. Aponte
> também os que estão identificados mas sem dizer o quanto importam: um
> parâmetro arbitrado que move o payback em dois anos merece frase diferente de
> um que move em duas semanas.
>
> ### 4. Artefato do modelo apresentado como propriedade do imóvel
>
> Este é o mais difícil de ver e o que mais compromete o estudo. Procure
> resultados que descrevem o **método** e não a instalação:
>
> - curva de carga com pico ao meio-dia numa residência (o jantar é o pico de
>   qualquer casa brasileira);
> - madrugada com quase zero de consumo numa casa com ar-condicionado;
> - as quatro estações com o mesmo consumo;
> - um cenário de menor uso saindo com curva idêntica à de outro;
> - autonomia "de 24 h" que é o teto da malha simulada e não um limite físico.
>
> Para cada um, diga se é propriedade do imóvel ou artefato, e como distinguir.
>
> ### 5. Nome de equipamento no corpo do documento
>
> O estudo especifica **característica nominal** — potência, energia útil,
> tensão, autonomia — e nunca fabricante ou modelo, exceto nos anexos de
> referência e de procedência, no fim. Aponte qualquer marca que apareça antes
> disso, inclusive em título de gráfico, legenda de figura, rótulo de eixo e
> tabela.
>
> ### 6. O que falta para decidir, e o que sobra
>
> Liste o que um cliente precisa para decidir e não está no documento. Liste
> também o que está e não serve à decisão — figura que exige esforço maior que
> o argumento que ela sustenta, tabela que repete o que o texto já disse,
> seção que responde pergunta que ninguém fez.
>
> ### 7. Ordem e altitude
>
> A conclusão vem antes da justificativa? Cada seção depende só do que veio
> antes? Há alguma seção que só faz sentido depois de outra que vem depois
> dela? O leitor consegue parar na página 3 e já ter a resposta?
>
> ### 8. Linguagem
>
> Frase que promete mais do que o cálculo sustenta. Voz passiva que esconde
> quem decidiu. Jargão sem tradução na primeira aparição. Número sem unidade.
> Precisão falsa — três casas decimais num valor arbitrado.
>
> ---
>
> ### Como responder
>
> Devolva os achados **ordenados por impacto na decisão do cliente**, não pela
> ordem das páginas. Para cada um:
>
> 1. **onde** — seção e página;
> 2. **o que está escrito** — cite o trecho ou o número;
> 3. **por que é problema** — o que o leitor conclui de errado;
> 4. **a correção** — o texto ou o número que deveria estar ali.
>
> Regras:
>
> - **Não invente número.** Se a correção exige recalcular algo que o PDF não
>   contém, diga qual conta é preciso refazer e com que dado.
> - **Diga quando não dá para verificar** pelo documento. "Não confere" e "não
>   consigo conferir" são achados diferentes e igualmente úteis.
> - **Separe erro de preferência.** Marque cada achado como `erro`,
>   `inconsistência`, `omissão` ou `estilo`. Não misture os quatro numa lista
>   só.
> - Se algo estiver **certo e bem resolvido**, diga em uma linha. Serve para eu
>   saber que foi conferido e não esquecido.
> - Se você encontrar **zero achados** numa das oito frentes, diga isso
>   explicitamente em vez de omitir a frente.

---

## Como usar

1. Anexe o PDF do estudo.
2. Cole o prompt.
3. Trabalhe os achados de cima para baixo — eles vêm ordenados por impacto.

Os achados de coerência numérica (item 1) quase sempre apontam defeito de
software, não de redação: quando o mesmo número sai diferente em duas páginas,
há dois caminhos de cálculo onde deveria haver um.
