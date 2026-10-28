# 02.1 — K-Means

> **Resumo:** divide os dados em *k* grupos, cada um representado pela média
> (centróide) de seus pontos, minimizando a soma das distâncias quadráticas ao centróide.

## Intuição
Coloque *k* "ímãs" no espaço. Cada ponto gruda no ímã mais próximo. Mova cada ímã
para o centro dos pontos que grudaram nele. Repita até os ímãs pararem de se mover.

## Função objetivo (inércia / WCSS / SSE)
$$J = \sum_{c=1}^{k}\sum_{x_i \in C_c}\lVert x_i - \mu_c \rVert^2$$

## Algoritmo de Lloyd
1. Escolha *k* centróides iniciais.
2. **Atribuição:** cada ponto vai para o centróide mais próximo.
3. **Atualização:** cada centróide vira a média dos pontos do seu cluster.
4. Repita 2–3 até convergir (atribuições não mudam ou $J$ varia menos que uma tolerância).

Cada passo nunca aumenta $J$ → converge sempre, mas para um **mínimo local**.

### Inicialização k-means++
Escolhe o 1º centróide ao acaso e os seguintes com probabilidade proporcional a $D(x)^2$
(distância ao centróide mais próximo já escolhido). Espalha os centróides e melhora
muito o resultado. É o padrão no scikit-learn. Mesmo assim, rode várias vezes (`n_init`).

## Suposições implícitas (e quando falha)
- Clusters **convexos e aproximadamente esféricos**.
- Tamanhos e variâncias **parecidos**.
- Todos os atributos com **escala comparável** → por isso padronizar é obrigatório.
- Falha com: luas/anéis, clusters alongados, densidades muito diferentes, outliers
  (a média é puxada por eles).

## Hiperparâmetros (scikit-learn `KMeans`)
| Parâmetro | Efeito | Típico |
|---|---|---|
| `n_clusters` | número de grupos *k* | escolher com cotovelo/silhueta |
| `init` | `"k-means++"` ou `"random"` | k-means++ |
| `n_init` | reinicializações (fica a melhor $J$) | 10 |
| `max_iter` | limite de iterações | 300 |
| `random_state` | reprodutibilidade | fixe sempre |

## Como escolher *k*
- **Cotovelo:** plote $J$ × *k* e procure a "dobra". Subjetivo.
- **Silhueta média:** escolha o *k* com maior valor. Mais objetivo.
- **Gap statistic, Calinski-Harabasz, Davies-Bouldin:** ver [06_avaliacao.md](06_avaliacao.md).
- **Domínio:** às vezes *k* é definido pelo problema (ex.: 3 faixas de risco).

## Variantes
| Variante | Diferença | Quando usar |
|---|---|---|
| **Mini-Batch K-Means** | atualiza com amostras pequenas | milhões de linhas |
| **K-Medoids (PAM)** | centro é um ponto real; qualquer distância | outliers, distâncias não euclidianas |
| **K-Modes** | moda e distância de Hamming | dados categóricos |
| **K-Prototypes** | K-Means + K-Modes | dados mistos |
| **Bisecting K-Means** | divide recursivamente | hierarquia + eficiência |

## Em código
```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

modelo = make_pipeline(StandardScaler(), KMeans(n_clusters=4, n_init=10, random_state=42))
rotulos = modelo.fit_predict(X)
centroides_originais = modelo[0].inverse_transform(modelo[1].cluster_centers_)
```

## No Projeto 2 (KNIME)
O nó **SimpleKMeans (Weka 3.7)** foi configurado com `-N 7` (k=7), `EuclideanDistance`,
`-I 500` iterações e semente `-S 10`. Pontos de atenção:
- O Weka normaliza atributos numéricos para [0,1] internamente na `EuclideanDistance`
  (por padrão) e trata nominais com distância 0/1 — ou seja, já faz uma forma de
  escalonamento + Hamming. Confira o comportamento no diálogo do nó antes de concluir.
- A escolha de k=7 precisa de justificativa (cotovelo/silhueta) no relatório.
- Rodar com outras sementes ajuda a verificar se os clusters são estáveis.

## Armadilhas
- Não padronizar.
- Interpretar clusters como "reais" só porque o algoritmo devolveu *k* grupos — ele
  **sempre** devolve *k* grupos, mesmo em dados uniformes.
- Usar K-Means em one-hot de muitas categorias sem pensar na distância resultante.

## Para revisar
1. Por que o K-Means sempre converge, mas não necessariamente para o ótimo global?
2. O que o k-means++ melhora?
3. Dê um exemplo de dados em que o K-Means falha e um algoritmo que resolveria.
4. Por que a inércia sempre diminui quando *k* aumenta? Por que isso torna ela ruim para escolher *k* sozinha?

## Referências
- Lloyd (1982), *Least squares quantization in PCM*.
- Arthur & Vassilvitskii (2007), *k-means++: The advantages of careful seeding*.
