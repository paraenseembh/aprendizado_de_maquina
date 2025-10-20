"""
Exemplo completo de uso da árvore de decisão com visualizações

Este script demonstra como usar todos os métodos de visualização
da árvore de decisão.
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from arvore_de_decisao_v1 import DecisionTree  

# Importar a classe DecisionTree (assumindo que está em decision_tree.py)
# from decision_tree import DecisionTree

# ==============================================================================
# 1. PREPARAR OS DADOS
# ==============================================================================

# Carregar dataset Iris
iris = load_iris()
X = iris.data
y = iris.target

# Usar apenas 2 features para simplificar a visualização
X = X[:, :2]  # Apenas sepal length e sepal width

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Nomes das características e classes
feature_names = ['sepal_length', 'sepal_width']
class_names = ['setosa', 'versicolor', 'virginica']

# ==============================================================================
# 2. TREINAR A ÁRVORE
# ==============================================================================

tree = DecisionTree(max_depth=3)
tree.fit(X_train, y_train)

# ==============================================================================
# 3. INFORMAÇÕES BÁSICAS DA ÁRVORE
# ==============================================================================

print("INFORMAÇÕES DA ÁRVORE:")
print(f"Profundidade: {tree.get_depth()}")
print(f"Número de folhas: {tree.get_n_leaves()}")
print(f"Número total de nós: {tree.get_n_nodes()}")

# ==============================================================================
# 4. VISUALIZAÇÃO EM TEXTO SIMPLES
# ==============================================================================

tree.print_tree(feature_names=feature_names, class_names=class_names)

# ==============================================================================
# 5. VISUALIZAÇÃO DETALHADA
# ==============================================================================

tree.visualize_text(feature_names=feature_names, max_depth=3)

# ==============================================================================
# 6. EXTRAIR E IMPRIMIR REGRAS
# ==============================================================================

tree.print_rules(feature_names=feature_names, class_names=class_names)

# Ou obter as regras como lista
rules = tree.get_rules(feature_names=feature_names, class_names=class_names)
print(f"\nTotal de regras: {len(rules)}")

# ==============================================================================
# 7. IMPORTÂNCIA DAS CARACTERÍSTICAS
# ==============================================================================

tree.print_feature_importance(feature_names=feature_names, n_features=2)

'''
# ==============================================================================
# 8. CONVERTER PARA DICIONÁRIO/JSON
# ==============================================================================

import json

tree_dict = tree.to_dict()
print("\nÁrvore em formato JSON:")
print(json.dumps(tree_dict, indent=2))

# ==============================================================================
# 9. VISUALIZAÇÃO GRÁFICA (requer matplotlib)
# ==============================================================================

try:
    tree.plot_tree(
        feature_names=feature_names,
        class_names=class_names,
        filename='iris_tree.png',
        dpi=150
    )
except ImportError:
    print("\nPara visualização gráfica, instale matplotlib:")
    print("pip install matplotlib")

# ==============================================================================
# 10. FAZER PREDIÇÕES
# ==============================================================================

predictions = tree.predict(X_test)

# Calcular acurácia
accuracy = np.mean(predictions == y_test)
print(f"\nAcurácia no conjunto de teste: {accuracy:.4f}")

# ==============================================================================
# 11. EXEMPLO DE PREDIÇÃO INDIVIDUAL COM EXPLICAÇÃO
# ==============================================================================

def explain_prediction(tree, sample, feature_names, class_names):
    """Explica o caminho de uma predição na árvore."""
    node = tree.root
    path = []
    
    while node.value is None:
        feature_name = feature_names[node.feature]
        threshold = node.threshold
        feature_value = sample[node.feature]
        
        if feature_value < threshold:
            decision = f"{feature_name} = {feature_value:.2f} < {threshold:.2f}"
            path.append(decision + " → IR PARA ESQUERDA")
            node = node.left
        else:
            decision = f"{feature_name} = {feature_value:.2f} >= {threshold:.2f}"
            path.append(decision + " → IR PARA DIREITA")
            node = node.right
    
    predicted_class = class_names[node.value] if class_names else node.value
    
    print("\n" + "="*60)
    print("EXPLICAÇÃO DA PREDIÇÃO")
    print("="*60)
    print(f"Amostra: {sample}")
    print("\nCaminho na árvore:")
    for i, step in enumerate(path, 1):
        print(f"  {i}. {step}")
    print(f"\nClasse predita: {predicted_class}")
    print("="*60 + "\n")
    
    return predicted_class

# Explicar uma predição específica
sample = X_test[0]
explain_prediction(tree, sample, feature_names, class_names)

# ==============================================================================
# 12. COMPARAÇÃO DE ÁRVORES COM DIFERENTES PROFUNDIDADES
# ==============================================================================

print("\n" + "="*60)
print("COMPARAÇÃO DE ÁRVORES COM DIFERENTES PROFUNDIDADES")
print("="*60)

for depth in [2, 3, 5, None]:
    tree_temp = DecisionTree(max_depth=depth)
    tree_temp.fit(X_train, y_train)
    
    predictions_temp = tree_temp.predict(X_test)
    accuracy_temp = np.mean(predictions_temp == y_test)
    
    depth_str = depth if depth else "Ilimitada"
    print(f"\nProfundidade máxima: {depth_str}")
    print(f"  Profundidade real: {tree_temp.get_depth()}")
    print(f"  Nº de folhas: {tree_temp.get_n_leaves()}")
    print(f"  Nº de nós: {tree_temp.get_n_nodes()}")
    print(f"  Acurácia: {accuracy_temp:.4f}")

print("="*60 + "\n")

# ==============================================================================
# 13. SALVAR ÁRVORE EM ARQUIVO DE TEXTO
# ==============================================================================

def save_tree_to_file(tree, filename, feature_names=None, class_names=None):
    """Salva a representação da árvore em um arquivo de texto."""
    import sys
    from io import StringIO
    
    # Capturar output
    old_stdout = sys.stdout
    sys.stdout = buffer = StringIO()
    
    tree.print_tree(feature_names=feature_names, class_names=class_names)
    tree.print_rules(feature_names=feature_names, class_names=class_names)
    tree.print_feature_importance(feature_names=feature_names, n_features=len(feature_names))
    
    # Restaurar stdout
    sys.stdout = old_stdout
    
    # Salvar em arquivo
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(buffer.getvalue())
    
    print(f"Árvore salva em: {filename}")

save_tree_to_file(tree, 'arvore_decisao.txt', feature_names, class_names)

print("\n✅ Todos os métodos de visualização foram executados com sucesso!")

'''