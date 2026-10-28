# Glossário

Termos em ordem alfabética. Adicione novos termos ao estudar cada tópico.

| Termo | Definição | Tópico |
|---|---|---|
| **AIC / BIC** | Critérios de informação: verossimilhança penalizada pelo nº de parâmetros; menor é melhor. | [GMM](02_clusterizacao/04_gmm.md) |
| **ARI** | *Adjusted Rand Index*: concordância entre dois agrupamentos ajustada ao acaso. | [Avaliação](02_clusterizacao/06_avaliacao.md) |
| **Centróide** | Ponto médio de um cluster (K-Means). | [K-Means](02_clusterizacao/01_kmeans.md) |
| **Codificação one-hot** | Uma coluna 0/1 por categoria de um atributo nominal. | [Padronização](01_padronizacao/README.md) |
| **Dendrograma** | Árvore que registra as fusões da clusterização hierárquica e suas alturas. | [Hierárquico](02_clusterizacao/02_hierarquico.md) |
| **Distância de Gower** | Média de dissimilaridades por atributo em [0,1]; serve para dados mistos. | [Categóricos](02_clusterizacao/05_categoricos.md) |
| **Distância de Hamming** | Nº de atributos em que dois registros diferem. | [Categóricos](02_clusterizacao/05_categoricos.md) |
| **EM** | *Expectation–Maximization*: algoritmo iterativo para estimar modelos com variáveis latentes. | [GMM](02_clusterizacao/04_gmm.md) |
| **eps (ε)** | Raio de vizinhança do DBSCAN. | [DBSCAN](02_clusterizacao/03_dbscan.md) |
| **Inércia (SSE/WCSS)** | Soma das distâncias quadráticas de cada ponto ao centróide do seu cluster. | [K-Means](02_clusterizacao/01_kmeans.md) |
| **IQR** | Intervalo interquartil, Q3 − Q1. | [Padronização](01_padronizacao/README.md) |
| **k-means++** | Inicialização que espalha os centróides iniciais com probabilidade ∝ D². | [K-Means](02_clusterizacao/01_kmeans.md) |
| **Ligação (*linkage*)** | Regra de distância entre clusters no hierárquico (single, complete, average, Ward). | [Hierárquico](02_clusterizacao/02_hierarquico.md) |
| **Medóide** | Ponto real do cluster que minimiza a distância aos demais. | [K-Means](02_clusterizacao/01_kmeans.md) |
| **Moda** | Valor mais frequente; "centro" do K-Modes. | [Categóricos](02_clusterizacao/05_categoricos.md) |
| **Padronização (z-score)** | $(x-\mu)/\sigma$: média 0 e desvio 1. | [Padronização](01_padronizacao/README.md) |
| **Normalização min-max** | $(x-min)/(max-min)$: intervalo [0,1]. | [Padronização](01_padronizacao/README.md) |
| **Perplexity** | Hiperparâmetro do t-SNE ≈ número efetivo de vizinhos. | [Visualização](02_clusterizacao/07_visualizacao.md) |
| **Ponto núcleo / borda / ruído** | Classificação dos pontos no DBSCAN. | [DBSCAN](02_clusterizacao/03_dbscan.md) |
| **Silhueta** | $(b-a)/\max(a,b)$: quão bem um ponto se encaixa no próprio cluster vs. o vizinho. | [Avaliação](02_clusterizacao/06_avaliacao.md) |
| **Vazamento de dados** | Uso de informação do teste durante o treino (ex.: ajustar scaler em tudo). | [Padronização](01_padronizacao/README.md) |
