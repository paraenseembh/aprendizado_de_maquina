# C4.5 Decision Tree - Implementação Python

Implementação do algoritmo C4.5 de árvore de decisão em Python puro com NumPy.

## Sobre o C4.5

O C4.5 é um algoritmo de aprendizado de máquina desenvolvido por Ross Quinlan como sucessor do ID3. É amplamente utilizado para classificação e possui as seguintes características:

### Vantagens sobre ID3

1. **Gain Ratio** ao invés de Information Gain - corrige viés de atributos com muitos valores
2. **Suporte nativo a atributos contínuos** - não requer discretização prévia
3. **Divisão binária para contínuos** - encontra threshold ótimo automaticamente
4. **Divisão multi-way para categóricos** - cria um filho por valor
5. **Reutilização de atributos** - contínuos podem ser usados múltiplas vezes

## Instalação

### Dependências

```bash
pip install numpy
```

Opcional para análise:
```bash
pip install scikit-learn pandas matplotlib
```

## Uso Básico

### 1. Importar e Preparar Dados

```python
from c45 import C45Classifier
import numpy as np

# Exemplo: seus dados
X = np.array([
    [5.1, 3.5, 1.4, 0.2],
    [4.9, 3.0, 1.4, 0.2],
    [6.2, 3.4, 5.4, 2.3],
    # ... mais amostras
])

y = np.array([0, 0, 1, ...])  # Labels
```

### 2. Criar e Treinar

```python
# Criar classificador
clf = C45Classifier(
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    min_gain_ratio=0.01
)

# Treinar
clf.fit(X, y)
```

### 3. Fazer Predições

```python
# Predizer novas amostras
predictions = clf.predict(X_test)
```

### 4. Visualizar Árvore

```python
# Imprimir estrutura da árvore
clf.print_tree()
```

## Parâmetros do Classificador

| Parâmetro | Tipo | Padrão | Descrição |
|-----------|------|--------|-----------|
| `max_depth` | int ou None | None | Profundidade máxima da árvore |
| `min_samples_split` | int | 2 | Mínimo de amostras para dividir um nó |
| `min_samples_leaf` | int | 1 | Mínimo de amostras por folha |
| `min_gain_ratio` | float | 0.0 | Gain Ratio mínimo para divisão |

### Quando usar cada parâmetro

**max_depth**: Use para controlar overfitting
- Datasets pequenos: `None` (sem limite)
- Datasets grandes: `10-20`

**min_samples_split**: Evita divisões com poucas amostras
- Datasets pequenos: `2-5`
- Datasets grandes: `20-50`

**min_samples_leaf**: Garante folhas representativas
- Classificação balanceada: `1-5`
- Classes desbalanceadas: `10-20`

**min_gain_ratio**: Exige ganho mínimo de informação
- Exploratório: `0.0`
- Produção: `0.01-0.05`

## Exemplo Completo com Avaliação

```python
from c45 import C45Classifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 1. Carregar seus dados
X = np.load('seus_dados_X.npy')
y = np.load('seus_dados_y.npy')

# 2. Dividir em treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3. Treinar
clf = C45Classifier(max_depth=15, min_samples_split=10)
clf.fit(X_train, y_train)

# 4. Avaliar
y_pred = clf.predict(X_test)
print(f"Acurácia: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

# 5. Visualizar
clf.print_tree()
```

## Carregando Dados de Diferentes Fontes

### CSV com Pandas

```python
import pandas as pd
import numpy as np

# Carregar CSV
df = pd.read_csv('seus_dados.csv')

# Separar features e target
X = df.drop('target_column', axis=1).values
y = df['target_column'].values

# Se tiver categorias textuais, codificar
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
```

### NumPy Arrays

```python
import numpy as np

# Carregar de arquivo
X = np.load('features.npy')
y = np.load('labels.npy')

# Ou criar manualmente
X = np.array([[...], [...]])
y = np.array([...])
```

### Sklearn Datasets

```python
from sklearn.datasets import load_iris, load_wine, load_breast_cancer

# Exemplo com Iris
data = load_iris()
X = data.data
y = data.target
```

## Validação Cruzada

```python
from sklearn.model_selection import cross_val_score

clf = C45Classifier(max_depth=10)

# 5-fold cross-validation
scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
print(f"Acurácia média: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

## Comparação com Sklearn

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Sklearn (usa CART, não C4.5, mas similar)
sklearn_clf = DecisionTreeClassifier(
    criterion='entropy',  # Usar entropia como C4.5
    max_depth=10
)
sklearn_clf.fit(X_train, y_train)

# Nossa implementação
c45_clf = C45Classifier(max_depth=10)
c45_clf.fit(X_train, y_train)

# Comparar
print(f"Sklearn: {accuracy_score(y_test, sklearn_clf.predict(X_test)):.4f}")
print(f"C4.5:    {accuracy_score(y_test, c45_clf.predict(X_test)):.4f}")
```

## Interpretando a Saída

### Exemplo de print_tree()

```
======================================================================
ESTRUTURA DA ÁRVORE C4.5
======================================================================

Root: n=150, entropy=1.585, class=0
  ├─ Divisão: X[2] <= 2.450
  [<=]: n=50, entropy=0.000, class=0 [FOLHA]
  [>]: n=100, entropy=1.000, class=1, GR=0.834
    ├─ Divisão: X[3] <= 1.750
    [<=]: n=54, entropy=0.445, class=1, GR=0.567
      ├─ Divisão: X[2] <= 4.950
      [<=]: n=48, entropy=0.146, class=1 [FOLHA]
      [>]: n=6, entropy=0.918, class=2, GR=0.248
    [>]: n=46, entropy=0.191, class=2 [FOLHA]
```

**Legenda**:
- `n`: número de amostras no nó
- `entropy`: impureza do nó (0 = puro, alto = misturado)
- `class`: classe majoritária naquele nó
- `GR`: Gain Ratio da divisão escolhida
- `[FOLHA]`: nó terminal (não divide mais)
- `X[i]`: índice da feature utilizada

## Estrutura de Arquivos

```
c45/
├── c45.py          # Implementação principal
├── README.md       # Este arquivo
└── claude.md       # Documentação de uso de IA
```

## Limitações

1. **Sem pós-poda**: Pode gerar árvores grandes (use max_depth)
2. **Missing values**: Não suportado (remova ou impute previamente)
3. **Performance**: Para datasets muito grandes (>100k), considere sklearn
4. **Apenas classificação**: Não suporta regressão

## Pré-processamento Recomendado

### Lidar com Valores Ausentes

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')  # ou 'median', 'most_frequent'
X = imputer.fit_transform(X)
```

### Normalização (opcional)

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(X)
```

### Codificar Variáveis Categóricas

```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Para target
le = LabelEncoder()
y = le.fit_transform(y)

# Para features categóricas (se necessário)
# Nota: C4.5 detecta automaticamente se é categórico ou contínuo
```

## Troubleshooting

### Erro: "index out of bounds"
- Verifique se X e y têm o mesmo número de amostras
- Certifique-se de que X é 2D: `X.shape = (n_samples, n_features)`

### Árvore muito profunda
- Reduza `max_depth`
- Aumente `min_samples_split` e `min_samples_leaf`
- Aumente `min_gain_ratio`

### Acurácia muito baixa
- Verifique se os dados estão corretos
- Tente remover `max_depth` temporariamente
- Verifique se há vazamento de dados (data leakage)
- Normalize features se tiverem escalas muito diferentes

### Overfitting (treino >> teste)
- Reduza `max_depth`
- Aumente `min_samples_leaf`
- Use validação cruzada
- Colete mais dados se possível

## Recursos Adicionais

### Teoria
- `claude.md` - Documentação detalhada sobre o algoritmo e uso de IA
- Quinlan, J. R. (1993). "C4.5: Programs for Machine Learning"

### Ferramentas Complementares
- scikit-learn: Implementações otimizadas de ML
- pandas: Manipulação de dados
- matplotlib/seaborn: Visualização

### Próximos Passos
- Implementar pós-poda (pruning)
- Adicionar suporte a missing values
- Otimizar para datasets grandes
- Adicionar visualização gráfica da árvore

## Licença

Código gerado com assistência de IA para fins educacionais.
