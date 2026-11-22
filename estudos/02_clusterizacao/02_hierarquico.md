# 02.2 — Clusterização Hierárquica

> **Resumo:** constrói uma hierarquia de agrupamentos (dendrograma); o número de
> clusters é escolhido depois, "cortando" a árvore em uma altura.

## Duas direções
- **Aglomerativa (bottom-up)** — começa com cada ponto sozinho e junta os pares mais
  próximos até sobrar um grupo. É a mais usada.
- **Divisiva (top-down)** — começa com tudo junto e divide (ex.: Bisecting K-Means, DIANA).

## Algoritmo aglomerativo
1. Calcule a matriz de distâncias entre todos os pontos.
2. Junte os dois clusters mais próximos.
3. Atualize as distâncias do novo cluster aos demais (segundo o **critério de ligação**).
4. Repita até restar um cluster. Registre cada fusão e sua altura → dendrograma.

## Critérios de ligação (*linkage*)
| Ligação | Distância entre clusters A e B | Comportamento |
|---|---|---|
| **Single** | menor distância entre pontos de A e B | encadeamento; acha formas alongadas; sensível a ruído |
| **Complete** | maior distância | clusters compactos, de diâmetro parecido |
| **Average** (UPGMA) | média de todas as distâncias | meio-termo |
| **Ward** | aumento da variância intra-cluster ao fundir | parecido com K-Means; só euclidiana |

## Lendo o dendrograma
- Eixo vertical = distância em que os grupos foram fundidos.
- Saltos grandes na altura indicam grupos bem separados → bom lugar para cortar.
- Cortar com uma linha horizontal = escolher o número de clusters.

## Prós e contras
| Prós | Contras |
|---|---|
| Não precisa de *k* antes | O(n²) memória, O(n²·log n)–O(n³) tempo → inviável para ~50 mil linhas sem amostrar |
| Dendrograma é ótimo para interpretação | Fusões são definitivas (não corrige erros) |
| Aceita **qualquer** matriz de distância (ex.: Gower para dados mistos) | Sensível à escolha da ligação |
| Determinístico | |

## Em código
```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

Z = linkage(X_padronizado, method="ward")       # SciPy para o dendrograma
dendrogram(Z, truncate_mode="lastp", p=30)
rotulos = fcluster(Z, t=4, criterion="maxclust")

# ou, direto no scikit-learn:
rotulos = AgglomerativeClustering(n_clusters=4, linkage="ward").fit_predict(X_padronizado)

# com matriz de distância pré-calculada (ex.: Gower):
AgglomerativeClustering(n_clusters=4, metric="precomputed", linkage="average").fit_predict(D)
```

## Na PNS
Com 53.630 linhas, a matriz de distâncias teria ~1,4 bilhão de pares — use uma
**amostra** (ex.: 2.000 linhas) para explorar o dendrograma e escolher *k*, depois
aplique K-Means/K-Modes ao conjunto completo.

## Para revisar
1. Diferença entre ligação single e complete; qual sofre "encadeamento"?
2. Por que Ward só funciona com distância euclidiana?
3. Como o dendrograma ajuda a escolher o número de clusters?
