

## Principais Características

### 1. **Gain Ratio** (Inovação Principal)
```python
Gain Ratio = Information Gain / Split Information
```
- Corrige o viés do Information Gain que favorece atributos com muitos valores
- Normaliza pelo Split Information (entropia da divisão)

### 2. **Suporte a Atributos Contínuos e Categóricos**
- **Contínuos**: divisão binária (≤ threshold)
- **Categóricos**: divisão multi-way (um filho por valor)

### 3. **Detecção Automática de Tipo**
Heurística simples:
- Numérico + muitos valores únicos → contínuo
- Caso contrário → categórico

### 4. **Reutilização Inteligente**
- Atributos contínuos PODEM ser reusados
- Atributos categóricos NÃO são reusados (após divisão multi-way)

## Diferenças em Relação aos Outros

| Característica | ID3 | CART | C4.5 |
|----------------|-----|------|------|
| **Critério** | Info Gain | Gini | Gain Ratio |
| **Divisões** | Multi-way | Binária | Híbrida |
| **Contínuos** | ✗ | ✓ | ✓ |
| **Reutilização** | ✗ | ✓ | ✓ (só contínuos) |

Você gostaria que eu:
1. Adicione métricas específicas para o C4.5?
2. Crie um sistema de comparação visual entre os 3 algoritmos?
3. Implemente a poda (pruning) no C4.5?