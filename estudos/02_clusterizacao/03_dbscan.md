# 02.3 — DBSCAN e HDBSCAN

> **Resumo:** clusters são regiões densas de pontos separadas por regiões esparsas;
> pontos em regiões esparsas são marcados como **ruído** (rótulo `-1`).

## Conceitos (DBSCAN)
Dois parâmetros: raio `eps` (ε) e `min_samples` (MinPts).

| Tipo de ponto | Definição |
|---|---|
| **Núcleo** (*core*) | tem ≥ MinPts vizinhos dentro do raio ε (contando ele mesmo) |
| **Borda** (*border*) | não é núcleo, mas está a ≤ ε de um núcleo |
| **Ruído** (*noise*) | nem núcleo nem borda |

Um cluster é o conjunto de núcleos **alcançáveis por densidade** entre si (cadeia de
núcleos a ≤ ε) mais suas bordas.

## Algoritmo
1. Para cada ponto não visitado, conte os vizinhos em ε.
2. Se for núcleo, crie um cluster e expanda: adicione vizinhos; vizinhos que também
   são núcleos trazem os vizinhos deles.
3. Pontos que não entram em nenhum cluster ficam como ruído.

## Escolhendo os parâmetros
- `min_samples`: regra prática ≥ dimensão + 1; frequentemente `2 × dimensões`.
- `eps`: **gráfico k-distância** — para cada ponto calcule a distância ao
  k-ésimo vizinho (k = min_samples), ordene e plote; o "joelho" da curva é um bom ε.
- Padronize antes: ε é uma distância, então depende da escala!

## Prós e contras
| Prós | Contras |
|---|---|
| Não precisa de *k* | Um único ε não serve para clusters de **densidades diferentes** |
| Encontra formas arbitrárias (luas, anéis) | Sensível a ε |
| Detecta outliers naturalmente | Sofre em alta dimensão (distâncias ficam parecidas) |

## HDBSCAN (evolução)
Roda DBSCAN para **todos** os ε, monta uma hierarquia e extrai os clusters mais
estáveis. Só precisa de `min_cluster_size`. Lida bem com densidades variáveis e
costuma ser a melhor escolha "padrão" baseada em densidade.
(`sklearn.cluster.HDBSCAN` a partir da versão 1.3.)

OPTICS é outra alternativa que ordena os pontos por alcançabilidade e permite extrair
clusters em vários níveis de densidade.

## Em código
```python
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.neighbors import NearestNeighbors
import numpy as np

# gráfico k-distância para escolher eps
k = 5
dist, _ = NearestNeighbors(n_neighbors=k).fit(Xs).kneighbors(Xs)
plt.plot(np.sort(dist[:, -1]))

rotulos = DBSCAN(eps=0.3, min_samples=k).fit_predict(Xs)
n_ruido = (rotulos == -1).sum()

rotulos_h = HDBSCAN(min_cluster_size=15).fit_predict(Xs)
```

## Para revisar
1. Defina ponto núcleo, de borda e ruído.
2. Por que o DBSCAN tem dificuldade com clusters de densidades diferentes? Como o HDBSCAN resolve?
3. Como usar o gráfico k-distância?
4. Em que situação o DBSCAN supera o K-Means? E o contrário?

## Referências
- Ester et al. (1996), *A density-based algorithm for discovering clusters…* (KDD).
- Campello, Moulavi & Sander (2013), *Density-based clustering based on hierarchical density estimates*.
