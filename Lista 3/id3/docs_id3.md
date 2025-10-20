## Documentação do Algoritmo ID3

### Fundamentos Teóricos

#### 1. **Entropia de Shannon**
$$Entropy(S) = -\sum_{i=1}^{c} p_i \log_2(p_i)$$

Onde:
- $S$ é o conjunto de exemplos
- $c$ é o número de classes
- $p_i$ é a proporção de exemplos da classe $i$

**Interpretação**: Mede a impureza/desordem do conjunto. Entropia = 0 indica conjunto puro (uma só classe).

#### 2. **Ganho de Informação (Information Gain)**
$$IG(S, A) = Entropy(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} \cdot Entropy(S_v)$$

Onde:
- $A$ é um atributo
- $Values(A)$ são os possíveis valores de $A$
- $S_v$ é o subconjunto de $S$ onde $A = v$

**Interpretação**: Mede quanto a divisão pelo atributo $A$ reduz a incerteza sobre as classes.

### Características Distintivas do ID3

#### **Diferenças em relação ao CART:**

| Característica | ID3 | CART |
|----------------|-----|------|
| **Critério** | Information Gain (entropia) | Gini Impurity |
| **Divisões** | Multi-way (n filhos) | Binárias (2 filhos) |
| **Atributos** | Apenas categóricos | Categóricos + contínuos |
| **Reutilização** | Não reutiliza atributos | Pode reutilizar |
| **Poda** | Não implementa (original) | Poda de complexidade |

### Justificativas das Decisões Técnicas

#### 1. **Por que Entropia ao invés de Gini?**
- **Fundamento teórico**: Entropia vem da teoria da informação de Shannon
- **Interpretação**: Mede "surpresa média" ou quantidade de informação
- **ID3 original**: Quinlan (1986) escolheu entropia por conexão com teoria da informação

#### 2. **Por que Divisões Multi-way?**
- **Intuição natural**: Para um atributo com k valores, cria k filhos
- **Exemplo**: Atributo "Cor" com valores {Vermelho, Azul, Verde} → 3 filhos
- **Limitação**: Pode criar árvores desbalanceadas se alguns valores são raros

#### 3. **Por que Apenas Atributos Categóricos?**
- **Simplicidade**: ID3 foi projetado para dados categóricos (survey data, etc.)
- **Solução**: Para contínuos, discretizar antes (como feito no exemplo)
- **Evolução**: C4.5 (sucessor do ID3) resolve essa limitação

#### 4. **Por que Não Reutilizar Atributos?**
- **Versão clássica**: Cada atributo usado uma vez por caminho
- **Justificativa**: Evita redundância em atributos categóricos
- **Consequência**: Árvore pode ser limitada em profundidade

### Vantagens e Limitações

**Vantagens:**
- Simples e intuitivo
- Regras facilmente interpretáveis
- Rápido para datasets pequenos/médios
- Base para C4.5 e outros algoritmos

**Limitações:**
- Não trabalha com atributos contínuos nativamente
- Sensível a atributos com muitos valores
- Não implementa poda (tende a overfitting)
- Não reutiliza atributos

### Uso Prático

```python
# 1. Preparar dados (discretizar se necessário)
from sklearn.preprocessing import KBinsDiscretizer
discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal')
X_discrete = discretizer.fit_transform(X)

# 2. Treinar ID3
id3 = ID3Classifier(max_depth=5, min_information_gain=0.01)
id3.fit(X_discrete, y)

# 3. Predizer
y_pred = id3.predict(X_test_discrete)

# 4. Extrair regras
rules = id3.get_rules()
```

### Próximos Passos

Agora posso:
1. Implementar o algoritmo **C4.5** (evolução do ID3 com gain ratio e atributos contínuos)
2. Criar **sistema de comparação** entre ID3, CART e C4.5
3. Adicionar **métricas específicas** para ID3
4. Implementar **visualização gráfica** das três árvores

Qual prefere?