# 02 — Clusterização (visão geral)

> **Resumo em uma frase:** encontrar grupos de objetos semelhantes entre si e
> diferentes dos outros grupos, **sem** usar rótulos (aprendizado não supervisionado).

## Pré-requisitos
- [01 — Padronização](../01_padronizacao/README.md) (quase todo algoritmo aqui depende de distância).

## As quatro grandes famílias

| Família | Ideia | Algoritmo representativo | Formato dos clusters |
|---|---|---|---|
| **Particionamento** | divide em *k* grupos minimizando um custo | K-Means, K-Medoids, K-Modes | convexos, "esféricos" |
| **Hierárquica** | constrói uma árvore de fusões (dendrograma) | Aglomerativo (Ward, average…) | depende da ligação |
| **Densidade** | cluster = região densa separada por regiões vazias | DBSCAN, HDBSCAN, OPTICS | formas arbitrárias, detecta ruído |
| **Modelo / probabilística** | dados vêm de uma mistura de distribuições | GMM (EM) | elípticos, atribuição "suave" |

## Material por algoritmo

1. [K-Means](01_kmeans.md) — o mais usado; base de comparação.
2. [Hierárquico](02_hierarquico.md) — dendrograma, não precisa fixar *k* antes.
3. [DBSCAN / HDBSCAN](03_dbscan.md) — formas arbitrárias e ruído.
4. [GMM](04_gmm.md) — probabilidades de pertinência, clusters elípticos.
5. [Dados categóricos e mistos](05_categoricos.md) — K-Modes, K-Prototypes, Gower (**o caso da PNS**).
6. [Avaliação](06_avaliacao.md) — cotovelo, silhueta, Davies-Bouldin, ARI…
7. [Visualização](07_visualizacao.md) — PCA e t-SNE.

## Tabela comparativa

| | K-Means | Hierárquico | DBSCAN | GMM | K-Modes/Prototypes |
|---|---|---|---|---|---|
| Precisa de *k*? | sim | não (corta depois) | não | sim | sim |
| Formato | esférico | depende da ligação | arbitrário | elíptico | — |
| Detecta ruído/outlier | não | não | **sim** | parcialmente (baixa prob.) | não |
| Escala (n grande) | ótima — O(n·k·i) | ruim — O(n²) memória | boa com índice espacial | média | boa |
| Tipo de dado | numérico | qualquer (com matriz de distância) | numérico (ou distância) | numérico | categórico / misto |
| Determinístico? | não (inicialização) | sim | quase (pontos de borda) | não | não |
| Atribuição | rígida | rígida | rígida | **suave** | rígida |

## Fluxo de trabalho recomendado

```
1. Entender os dados (tipos, faltantes, outliers)
2. Pré-processar: remover constantes, imputar, codificar, ESCALONAR
3. (Opcional) reduzir dimensionalidade (PCA)
4. Escolher algoritmo(s) pela forma esperada dos grupos e pelo tipo de dado
5. Escolher hiperparâmetros (k, eps…) com métricas internas + conhecimento do domínio
6. Avaliar estabilidade (rodar com sementes/amostras diferentes)
7. INTERPRETAR: perfilar cada cluster (médias, modas) — clusters precisam fazer sentido
```

O passo 7 é o mais esquecido e o mais importante: um agrupamento só tem valor se
você consegue **descrever** cada grupo.

## Notebooks
- [`comparacao_algoritmos.ipynb`](comparacao_algoritmos.ipynb) — todos os algoritmos em dados sintéticos (bolhas, luas, anisotrópicos, ruído) + efeito da padronização + métricas.
- [`aplicacao_pns.ipynb`](aplicacao_pns.ipynb) — reproduz e critica o K-Means k=7 do Projeto 2 (KNIME) e compara com K-Modes.

## Referências gerais
- scikit-learn — *Clustering*: https://scikit-learn.org/stable/modules/clustering.html
- Tan, Steinbach & Kumar. *Introduction to Data Mining*, caps. 7–8.
- Han, Kamber & Pei. *Data Mining: Concepts and Techniques*, caps. 10–11.
