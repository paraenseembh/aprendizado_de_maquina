# Repositório de Estudos — Aprendizado de Máquina

Material de estudo pessoal, construído a partir dos trabalhos da disciplina
(listas, C4.5 e o **Projeto 2 em KNIME** em [`../mineracao_de_dados`](../mineracao_de_dados)).
A ideia é que esta pasta **cresça ao longo do curso**: cada tópico novo ganha uma
pasta numerada com teoria (`.md`) e prática (`.ipynb`).

## Como estudar

1. Leia o `README.md` do tópico (teoria, intuição e fórmulas).
2. Abra o notebook do tópico e execute célula a célula — altere parâmetros e observe.
3. Faça os exercícios no fim de cada notebook.
4. Marque o progresso no [roteiro](#roteiro) abaixo.

```bash
pip install -r estudos/requirements.txt
jupyter lab estudos/
```

## Roteiro

| # | Tópico | Teoria | Prática | Status |
|---|--------|--------|---------|--------|
| 01 | Padronização e escalonamento de dados | [README](01_padronizacao/README.md) | [notebook](01_padronizacao/padronizacao.ipynb) | ☐ |
| 02 | Algoritmos de clusterização (visão geral) | [README](02_clusterizacao/README.md) | [comparação](02_clusterizacao/comparacao_algoritmos.ipynb) | ☐ |
| 02.1 | K-Means | [01_kmeans.md](02_clusterizacao/01_kmeans.md) | ↑ | ☐ |
| 02.2 | Hierárquico (aglomerativo) | [02_hierarquico.md](02_clusterizacao/02_hierarquico.md) | ↑ | ☐ |
| 02.3 | DBSCAN / HDBSCAN | [03_dbscan.md](02_clusterizacao/03_dbscan.md) | ↑ | ☐ |
| 02.4 | Mistura de Gaussianas (GMM) | [04_gmm.md](02_clusterizacao/04_gmm.md) | ↑ | ☐ |
| 02.5 | Dados categóricos: K-Modes, K-Prototypes, Gower | [05_categoricos.md](02_clusterizacao/05_categoricos.md) | [aplicação PNS](02_clusterizacao/aplicacao_pns.ipynb) | ☐ |
| 02.6 | Avaliação de clusters | [06_avaliacao.md](02_clusterizacao/06_avaliacao.md) | ↑ | ☐ |
| 02.7 | Redução de dimensionalidade para visualização (PCA, t-SNE) | [07_visualizacao.md](02_clusterizacao/07_visualizacao.md) | ↑ | ☐ |
| 03 | *(próximo tópico — ver "Como evoluir")* | | | |

Outros materiais: [glossário](glossario.md) · [registro de uso de IA](REGISTRO_USO_IA.md)

## Ligação com o Projeto 2 (KNIME)

O workflow `mineracao_de_dados/workflow.knime` usa os dados da **PNS (Pesquisa
Nacional de Saúde)** em `c45/dados_c45_sembalanceamento_originais.csv`
(53.630 linhas, 26 colunas, quase todas **categóricas**) e contém, entre outros, os nós:

`CSV Reader → Missing Value → … → SimpleKMeans (k=7, distância euclidiana) → Weka Cluster Assigner → t-SNE → Scatter Plot`

Perguntas que este material ajuda a responder sobre o projeto:

- A distância euclidiana faz sentido para variáveis majoritariamente categóricas? → [05_categoricos.md](02_clusterizacao/05_categoricos.md)
- Os atributos deveriam ter sido padronizados antes do K-Means? → [01_padronizacao](01_padronizacao/README.md)
- k=7 foi uma boa escolha? Como justificar? → [06_avaliacao.md](02_clusterizacao/06_avaliacao.md)
- O que o gráfico do t-SNE mostra (e o que **não** mostra)? → [07_visualizacao.md](02_clusterizacao/07_visualizacao.md)

O notebook [`aplicacao_pns.ipynb`](02_clusterizacao/aplicacao_pns.ipynb) reproduz
o pipeline em Python e compara alternativas.

## Como evoluir este repositório

1. Crie uma pasta `NN_nome_do_topico/` (próximo número livre).
2. Copie [`_modelos/TOPICO_README.md`](_modelos/TOPICO_README.md) como `README.md` da pasta e preencha.
3. Crie o notebook prático; funções reutilizáveis vão em [`utils/`](utils/).
4. Adicione termos novos ao [glossário](glossario.md).
5. Atualize a tabela do [roteiro](#roteiro).
6. Registre o uso de IA em [`REGISTRO_USO_IA.md`](REGISTRO_USO_IA.md).

Sugestões de próximos tópicos: PCA em profundidade · seleção de atributos ·
regras de associação (Apriori/FP-Growth) · validação cruzada e busca de hiperparâmetros ·
detecção de anomalias (Isolation Forest, LOF) · clusterização de séries temporais.

## Estrutura

```
estudos/
├── README.md                 ← você está aqui
├── requirements.txt
├── glossario.md
├── REGISTRO_USO_IA.md
├── _modelos/                 ← templates para novos tópicos
├── utils/                    ← funções compartilhadas pelos notebooks
├── 01_padronizacao/
└── 02_clusterizacao/
```
