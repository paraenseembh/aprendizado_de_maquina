# 02.4 — Modelos de Mistura Gaussiana (GMM)

> **Resumo:** supõe que os dados foram gerados por *k* distribuições normais
> (gaussianas) e estima, para cada ponto, a **probabilidade** de pertencer a cada uma.

## Modelo
$$p(x) = \sum_{c=1}^{k} \pi_c \,\mathcal{N}(x \mid \mu_c, \Sigma_c)$$
- $\pi_c$: peso (proporção) do componente $c$, com $\sum \pi_c = 1$.
- $\mu_c$: média; $\Sigma_c$: matriz de covariância (define forma e orientação da elipse).

## Algoritmo EM (Expectation–Maximization)
1. Inicialize $\pi, \mu, \Sigma$ (geralmente com K-Means).
2. **E-step:** calcule a responsabilidade $\gamma_{ic} = P(c \mid x_i)$ de cada componente por cada ponto.
3. **M-step:** reestime $\pi_c, \mu_c, \Sigma_c$ como médias ponderadas por $\gamma_{ic}$.
4. Repita até a log-verossimilhança estabilizar.

> O K-Means é um caso particular "rígido" do GMM com covariâncias esféricas e iguais.

## Tipos de covariância (`covariance_type`)
| Tipo | Forma dos clusters | Nº de parâmetros |
|---|---|---|
| `spherical` | círculos, cada um com seu raio | mínimo |
| `diag` | elipses alinhadas aos eixos | médio |
| `tied` | todas as elipses com mesma forma | médio |
| `full` | elipses com qualquer orientação | máximo |

## Escolhendo *k* e a covariância
Use **BIC** ou **AIC** (menor é melhor): penalizam a verossimilhança pelo número de
parâmetros, evitando sobreajuste.

## Prós e contras
| Prós | Contras |
|---|---|
| Atribuição suave (probabilidades) | Supõe gaussianas |
| Clusters elípticos e de tamanhos diferentes | Mais lento e mais parâmetros que K-Means |
| Critérios formais (BIC/AIC) para escolher *k* | Pode degenerar (covariância singular) — use `reg_covar` |
| Serve também para estimar densidade / detectar anomalias | Precisa de padronização para estabilidade numérica |

## Em código
```python
from sklearn.mixture import GaussianMixture

bics = [GaussianMixture(k, covariance_type="full", random_state=0).fit(Xs).bic(Xs)
        for k in range(1, 10)]
gmm = GaussianMixture(n_components=4, covariance_type="full", random_state=0).fit(Xs)
rotulos = gmm.predict(Xs)
probabilidades = gmm.predict_proba(Xs)   # matriz n × k
```

## Para revisar
1. Qual a diferença entre atribuição rígida e suave?
2. Explique os passos E e M.
3. Por que o K-Means pode ser visto como caso particular do GMM?
4. Para que servem BIC e AIC?
