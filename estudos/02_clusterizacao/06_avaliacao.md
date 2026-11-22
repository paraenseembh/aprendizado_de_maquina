# 02.6 — Avaliação de Clusters

> **Resumo:** sem rótulos, avaliamos se os grupos são **compactos** e **separados**
> (métricas internas); com rótulos de referência, medimos a **concordância** (externas).

## Métricas internas (sem rótulos)

| Métrica | Ideia | Faixa | Melhor | sklearn |
|---|---|---|---|---|
| **Inércia / SSE** | soma das distâncias² ao centróide | ≥ 0 | menor (mas sempre cai com *k*) | `KMeans.inertia_` |
| **Silhueta** | $s=\frac{b-a}{\max(a,b)}$; $a$ = dist. média ao próprio cluster, $b$ = ao cluster vizinho mais próximo | [-1, 1] | maior | `silhouette_score` |
| **Calinski-Harabasz** | razão variância entre / dentro | ≥ 0 | maior | `calinski_harabasz_score` |
| **Davies-Bouldin** | similaridade média de cada cluster com o mais parecido | ≥ 0 | menor | `davies_bouldin_score` |
| **BIC / AIC** (GMM) | verossimilhança penalizada | — | menor | `gmm.bic(X)` |

Interpretação aproximada da silhueta média: > 0,7 forte · 0,5–0,7 razoável ·
0,25–0,5 fraca · < 0,25 sem estrutura substancial.

> ⚠️ Silhueta, CH e DB favorecem clusters convexos — tendem a "punir" bons
> resultados do DBSCAN em formas de lua. Use mais de uma métrica e **olhe os dados**.

A silhueta aceita qualquer métrica: `silhouette_score(X, rotulos, metric="hamming")`
para dados categóricos codificados. Custa O(n²): use `sample_size=` em bases grandes.

## Métricas externas (com rótulos de referência)

Útil em dados sintéticos ou para checar se os clusters se relacionam a uma variável
conhecida (ex.: `Q00201` na PNS) — lembrando que clusters **não** precisam coincidir com classes.

| Métrica | Faixa | Observação |
|---|---|---|
| **ARI** (Adjusted Rand Index) | [-0,5, 1]; 0 = acaso | conta pares concordantes, ajustado ao acaso |
| **NMI / AMI** | [0, 1] | informação mútua normalizada |
| **Homogeneidade / Completude / V-measure** | [0, 1] | cada cluster tem uma só classe? / cada classe está em um só cluster? |
| **Pureza** | [0, 1] | simples, mas sobe artificialmente com *k* |
| **Tabela de contingência** | — | `pd.crosstab(cluster, classe)` — sempre olhe! |

## Estabilidade
Um bom agrupamento deve se repetir:
- rode com várias sementes e compare os rótulos com ARI;
- rode em subamostras (bootstrap) e veja se os mesmos grupos reaparecem.

## Escolha de *k* — receita prática
1. Varie *k* (ex.: 2 a 10).
2. Plote inércia (cotovelo) **e** silhueta (ou CH/DB).
3. Prefira o *k* que equilibre as métricas **e** gere grupos interpretáveis.
4. Verifique a estabilidade do *k* escolhido.
5. Justifique no relatório (gráfico + argumento).

## Para revisar
1. Por que a inércia não serve sozinha para escolher *k*?
2. Calcule a silhueta de um ponto com a = 2 e b = 5.
3. Diferença entre métricas internas e externas.
4. O que significa ARI = 0?
