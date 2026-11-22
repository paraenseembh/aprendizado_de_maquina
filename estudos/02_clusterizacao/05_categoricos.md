# 02.5 — Clusterização de Dados Categóricos e Mistos

> **Resumo:** quando os atributos são categorias (como quase toda a PNS), média e
> distância euclidiana perdem sentido; usamos moda, distância de Hamming ou Gower.

## O problema
- "Média" de `Parda`, `Branca` e `Preta` não existe.
- Codificar como inteiros cria ordem falsa (ver [01_padronizacao](../01_padronizacao/README.md)).
- One-hot + euclidiana funciona, mas: a distância entre dois registros vira
  $\sqrt{2 \times \text{nº de atributos diferentes}}$, os centróides viram proporções
  (não categorias reais) e atributos com muitas categorias ganham mais colunas.

## Distância de Hamming (simple matching)
$$d(x,y) = \sum_j \mathbb{1}[x_j \neq y_j]$$
Conta em quantos atributos dois registros diferem.

## K-Modes (Huang, 1998)
Igual ao K-Means, trocando:
- distância euclidiana → **Hamming**;
- média → **moda** de cada atributo (o centro é uma combinação de categorias reais, interpretável!).

Inicializações: `"Huang"` ou `"Cao"` (baseada em densidade, mais estável).

## K-Prototypes
Para dados **mistos** (numéricos + categóricos):
$$d(x,y) = \underbrace{\sum_{num}(x_j-y_j)^2}_{\text{euclidiana}} + \gamma \underbrace{\sum_{cat}\mathbb{1}[x_j\neq y_j]}_{\text{Hamming}}$$
O peso $\gamma$ equilibra as duas partes (padrão: metade do desvio padrão médio dos
numéricos). **Padronize os numéricos** antes, senão $\gamma$ fica sem referência.

## Distância de Gower
Média das dissimilaridades por atributo, cada uma em [0, 1]:
- numérico: $|x_j-y_j| / \text{amplitude}_j$;
- categórico: 0 se igual, 1 se diferente;
- ordinal: posição normalizada pelo nº de níveis.

Gera uma matriz de distâncias que pode alimentar **hierárquico**, **K-Medoids** ou
**DBSCAN** com `metric="precomputed"` (custo O(n²) → use amostra).

## Atributos ordinais
Faixa etária, escolaridade e renda (`C008`, `VDD004A`, `VDF004`) têm ordem.
Tratar como nominal perde informação; uma boa opção é `OrdinalEncoder` com ordem
explícita + escalonar e usar como numérico no K-Prototypes/Gower.

## Como o Weka (Projeto 2) lida com isso
A `EuclideanDistance` do Weka, usada no nó SimpleKMeans do KNIME:
- normaliza numéricos para [0, 1] (a menos que `-D` desative);
- para nominais, usa diferença 0 (igual) / 1 (diferente);
- o centróide de atributos nominais é a **moda**.

Na prática isso se aproxima de um K-Prototypes com Hamming — razoável para a PNS,
mas ordinais tratados como nominais perdem a noção de "vizinhança" entre faixas.

## Em código
```python
from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes

km = KModes(n_clusters=5, init="Cao", n_init=5, random_state=42)
rotulos = km.fit_predict(df_categorico)
km.cluster_centroids_     # modas — diretamente interpretáveis
km.cost_                  # soma das distâncias de Hamming (use no "cotovelo")

kp = KPrototypes(n_clusters=5, init="Cao", random_state=42)
rotulos = kp.fit_predict(X_misto, categorical=[0, 2, 3])  # índices das colunas categóricas
```

## Para revisar
1. Por que o K-Means "puro" não serve para dados categóricos?
2. O que muda do K-Means para o K-Modes?
3. Para que serve o parâmetro γ do K-Prototypes?
4. Como a distância de Gower trata cada tipo de atributo?

## Referências
- Huang, Z. (1998). *Extensions to the k-Means Algorithm for Clustering Large Data Sets with Categorical Values*.
- Gower, J. C. (1971). *A general coefficient of similarity and some of its properties*.
- Pacote `kmodes`: https://github.com/nicodv/kmodes
