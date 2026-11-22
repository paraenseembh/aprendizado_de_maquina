# 01 — Padronização e Escalonamento de Dados

> **Resumo em uma frase:** transformar os atributos para que fiquem em escalas
> comparáveis, de modo que nenhum domine o algoritmo apenas por causa da sua unidade de medida.

## Pré-requisitos
- Média, desvio padrão, mediana, quartis.
- Noção de distância euclidiana: $d(x,y)=\sqrt{\sum_j (x_j-y_j)^2}$.

## Intuição

Imagine agrupar pessoas por **idade** (20 a 80 anos) e **renda** (R$ 1.000 a R$ 20.000).
Na distância euclidiana, uma diferença de R$ 1.000 pesa o mesmo que uma diferença de
1.000 anos de idade. Resultado: a renda "manda" sozinha no agrupamento e a idade é ignorada.

Escalonar é colocar todos os atributos "na mesma régua" antes de calcular distâncias,
gradientes ou variâncias.

## Terminologia (os nomes variam muito entre autores!)

| Termo | Normalmente significa | No scikit-learn |
|---|---|---|
| **Padronização** (*standardization*, z-score) | média 0, desvio 1 | `StandardScaler` |
| **Normalização min-max** | intervalo [0, 1] | `MinMaxScaler` |
| **Normalização de vetor** | cada **linha** com norma 1 | `Normalizer` |
| **Escalonamento** (*feature scaling*) | termo guarda-chuva | — |

> ⚠️ Em português "normalização" é usado para coisas diferentes. Sempre diga **qual** fórmula usou.

## Principais técnicas

Notação: $x$ é o valor original de um atributo, $x'$ o valor transformado.

### 1. Padronização z-score — `StandardScaler`
$$x' = \frac{x - \mu}{\sigma}$$
- Resultado: média 0, desvio padrão 1 (o formato da distribuição **não** muda).
- Padrão para K-Means, PCA, SVM, regressão logística/regularizada, redes neurais.
- Sensível a outliers (eles inflam $\sigma$).

### 2. Min-Max — `MinMaxScaler`
$$x' = \frac{x - x_{min}}{x_{max} - x_{min}}$$
- Resultado: intervalo [0, 1] (ou outro definido).
- Útil quando o algoritmo espera faixa limitada (redes neurais com sigmoid, imagens).
- **Muito** sensível a outliers: um único valor extremo comprime todos os outros.

### 3. Escalonamento robusto — `RobustScaler`
$$x' = \frac{x - \text{mediana}}{Q_3 - Q_1}$$
- Usa mediana e intervalo interquartil (IQR), que ignoram os extremos.
- Boa escolha quando há outliers que você **não** quer remover.

### 4. MaxAbs — `MaxAbsScaler`
$$x' = \frac{x}{\max|x|}$$
- Intervalo [-1, 1], **preserva zeros** → adequado para matrizes esparsas (ex.: texto TF-IDF).

### 5. Normalização por linha — `Normalizer`
$$\vec{x}' = \frac{\vec{x}}{\lVert\vec{x}\rVert_2}$$
- Atua em cada **amostra**, não em cada atributo. Importa a direção, não a magnitude.
- Usado com similaridade do cosseno (textos, perfis de consumo).

### 6. Transformações que mudam a forma da distribuição
| Técnica | Quando | scikit-learn |
|---|---|---|
| Log, $\log(1+x)$ | assimetria à direita (renda, contagens) | `FunctionTransformer(np.log1p)` |
| Box-Cox | só valores > 0 | `PowerTransformer(method="box-cox")` |
| Yeo-Johnson | aceita zero e negativos | `PowerTransformer()` |
| Quantílica | forçar distribuição uniforme/normal | `QuantileTransformer` |

## Quem precisa de escalonamento?

| Precisa (sensível à escala) | Não precisa (invariante à escala) |
|---|---|
| K-Means, hierárquico, DBSCAN, GMM | Árvores de decisão (ID3, C4.5, CART) |
| KNN, KNNImputer | Random Forest, Gradient Boosting, XGBoost |
| SVM, regressão com regularização | Naive Bayes (em geral) |
| PCA, t-SNE, UMAP | Regras de associação |
| Redes neurais / gradiente descendente | |

> Por isso, nas Listas 3 e 4 (árvores, Random Forest) a padronização não alterava o
> resultado — mas no K-Means do Projeto 2 ela é decisiva.

## E os atributos categóricos?

Escalonamento só faz sentido para números. Para categorias é preciso **codificar** antes:

| Tipo | Exemplo (PNS) | Codificação recomendada |
|---|---|---|
| Nominal (sem ordem) | `C009` cor/raça, `A001` tipo de domicílio | One-hot (`OneHotEncoder`) |
| Ordinal (com ordem) | `C008` faixa etária, `VDD004A` escolaridade, `VDF004` renda | `OrdinalEncoder` com a ordem explícita |
| Binário | `C006` sexo, `Q00201` (Sim/Não) | 0/1 |

Cuidados:
- **Codificar nominal como 0,1,2,3…** inventa uma ordem e distâncias que não existem
  (ex.: "Branca=0, Preta=1, Parda=2" diria que Branca está mais perto de Preta que de Parda).
- **One-hot + z-score**: padronizar colunas dummy dá peso maior a categorias **raras**
  (desvio pequeno → divisão por número pequeno). Às vezes é desejável, às vezes não; seja consciente.
- Atributo com **um único valor** (ex.: `VDM001`, `Status_N017` na PNS) tem $\sigma=0$ e
  não carrega informação → remova antes.
- Para dados mistos considere distâncias próprias (Gower) ou algoritmos como
  K-Prototypes — ver [05_categoricos.md](../02_clusterizacao/05_categoricos.md).

## A regra de ouro: evite vazamento de dados (*data leakage*)

Os parâmetros do escalonador ($\mu,\sigma$, min, max…) devem ser **aprendidos só no treino**
e depois aplicados ao teste:

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)
scaler = StandardScaler().fit(X_train)   # aprende μ e σ SÓ do treino
X_train_s = scaler.transform(X_train)
X_test_s  = scaler.transform(X_test)     # NUNCA chame fit no teste
```

O jeito mais seguro é um `Pipeline`, que garante isso também dentro da validação cruzada:

```python
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
modelo = make_pipeline(StandardScaler(), KNeighborsClassifier())
```

Em clusterização (não supervisionado) normalmente usamos todos os dados para ajustar,
mas a mesma regra vale se você for atribuir clusters a dados novos depois.

### Pipeline com colunas mistas

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

pre = ColumnTransformer([
    ("num", StandardScaler(), colunas_numericas),
    ("ord", make_pipeline(OrdinalEncoder(categories=ordens), StandardScaler()), colunas_ordinais),
    ("nom", OneHotEncoder(handle_unknown="ignore"), colunas_nominais),
])
```

## Como escolher (fluxograma resumido)

1. O algoritmo é sensível à escala? Se não (árvores), pode pular.
2. Há outliers relevantes? Sim → `RobustScaler` (ou trate os outliers antes).
3. Distribuição muito assimétrica? Sim → log / `PowerTransformer` e depois z-score.
4. Precisa de faixa fixa ou dados esparsos? → `MinMaxScaler` / `MaxAbsScaler`.
5. Caso contrário → `StandardScaler`.
6. Categóricas → codificar primeiro (seção acima).

## No KNIME

| Operação | Nó KNIME |
|---|---|
| z-score, min-max, decimal scaling | **Normalizer** (opções no diálogo) |
| Aplicar a mesma transformação ao teste | **Normalizer (Apply)** usando a porta de modelo |
| Desfazer | **Denormalizer** |
| One-hot | **One to Many** |

## Armadilhas comuns
- Ajustar o scaler em todo o dataset antes do `train_test_split` (vazamento).
- Escalonar o **alvo** de classificação.
- Achar que padronizar torna a distribuição normal (não torna; só desloca e reescala).
- Esquecer de escalonar dados novos com o **mesmo** scaler do treino.
- Interpretar centróides do K-Means na escala transformada — use `scaler.inverse_transform`.

## Para revisar (perguntas de prova)
1. Qual a diferença entre `StandardScaler` e `Normalizer`?
2. Por que árvores de decisão não precisam de padronização?
3. Um atributo tem valores entre 0 e 10, com um outlier em 1000. Qual escalonador escolher e por quê?
4. O que é vazamento de dados no contexto de escalonamento? Como o `Pipeline` evita?
5. Por que codificar cor/raça como 0,1,2,3,4 é problemático para o K-Means?

## Referências
- scikit-learn User Guide — *Preprocessing data*: https://scikit-learn.org/stable/modules/preprocessing.html
- scikit-learn — *Compare the effect of different scalers on data with outliers*.
- Han, Kamber & Pei. *Data Mining: Concepts and Techniques*, cap. 3 (Data Preprocessing).
- Géron, A. *Mãos à Obra: Aprendizado de Máquina com Scikit-Learn, Keras & TensorFlow*, cap. 2.
