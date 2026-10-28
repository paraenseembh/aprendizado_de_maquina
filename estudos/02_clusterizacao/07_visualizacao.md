# 02.7 — Visualizando Clusters: PCA e t-SNE

> **Resumo:** projetamos dados de muitas dimensões em 2D para enxergar os clusters;
> PCA preserva a estrutura **global** (linear), t-SNE preserva vizinhanças **locais**.

## PCA (Análise de Componentes Principais)
- Encontra direções ortogonais de **máxima variância**; projeta nos 2 primeiros componentes.
- Linear, determinístico, rápido; distâncias e posições relativas são interpretáveis.
- `explained_variance_ratio_` diz quanto da variância cabe no gráfico.
- **Precisa de padronização** (variância depende da escala).
- Também serve como pré-processamento para clusterizar (reduz ruído e dimensão).

## t-SNE
- Converte distâncias em probabilidades de vizinhança e busca um mapa 2D que as preserve.
- Excelente para revelar grupos locais; **não linear** e **estocástico**.
- Hiperparâmetro principal: `perplexity` (≈ nº efetivo de vizinhos; típico 5–50).

### O que o t-SNE NÃO mostra (cuidado no relatório!)
- **Tamanho** dos clusters no mapa não significa nada.
- **Distância entre** clusters no mapa não significa nada.
- Pode criar "grupos" em dados sem estrutura, especialmente com perplexity baixa.
- Resultados mudam com a semente.
- Não use as coordenadas do t-SNE como entrada para clusterizar (no máximo para explorar).

UMAP (pacote `umap-learn`) é uma alternativa mais rápida que preserva um pouco melhor a estrutura global.

## No Projeto 2 (KNIME)
O workflow usa **t-SNE (L. Jonsson)** → **Scatter Plot** com as dimensões 0 e 1,
colorindo por `Q00201`. Ao interpretar:
- Colorir pelo **cluster** do K-Means mostra se os grupos são separáveis localmente.
- Colorir pela **classe** (`Q00201`) mostra se a classe se relaciona com a estrutura — são perguntas diferentes.
- Complemente com PCA para ter uma visão global e com métricas da [avaliação](06_avaliacao.md).

## Em código
```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

pca = PCA(n_components=2).fit(Xs)
X2 = pca.transform(Xs); print(pca.explained_variance_ratio_.sum())

X_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(Xs_amostra)
```

## Para revisar
1. Por que padronizar antes do PCA?
2. Cite duas coisas que não podem ser concluídas de um gráfico t-SNE.
3. Quando preferir PCA a t-SNE para visualizar clusters?
