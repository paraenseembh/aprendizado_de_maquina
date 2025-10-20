import numpy as np
from collections import Counter
from typing import Optional, Union, Tuple, List

class Node:
    """
    Representa um nó na árvore de decisão CART.
    
    Attributes:
        feature_idx: índice do atributo usado para divisão
        threshold: valor de limiar para divisão (contínuos) ou valor categórico
        left: subárvore esquerda
        right: subárvore direita
        value: valor de predição (para nós folha)
        gini: índice de Gini do nó
        n_samples: número de amostras no nó
        class_distribution: distribuição de classes no nó
    """
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, 
                 value=None, gini=None, n_samples=None, class_distribution=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.gini = gini
        self.n_samples = n_samples
        self.class_distribution = class_distribution


class CARTClassifier:
    """
    Implementação do algoritmo CART (Classification and Regression Trees).
    
    Características principais:
    - Usa índice de Gini como critério de divisão
    - Realiza divisões binárias
    - Suporta atributos contínuos e categóricos
    
    Parameters:
        max_depth: profundidade máxima da árvore (None = sem limite)
        min_samples_split: número mínimo de amostras para dividir um nó
        min_samples_leaf: número mínimo de amostras em um nó folha
        min_impurity_decrease: diminuição mínima de impureza para realizar divisão
    """
    
    def __init__(self, max_depth: Optional[int] = None, 
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 min_impurity_decrease: float = 0.0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.root = None
        self.n_classes_ = None
        self.feature_importances_ = None
        
    def _gini_impurity(self, y: np.ndarray) -> float:
        """
        Calcula o índice de Gini para um conjunto de labels.
        
        Gini = 1 - Σ(p_i²), onde p_i é a proporção da classe i
        
        Args:
            y: array de labels
            
        Returns:
            float: índice de Gini (0 = puro, máximo quando uniforme)
        """
        if len(y) == 0:
            return 0.0
        
        proportions = np.bincount(y) / len(y)
        gini = 1.0 - np.sum(proportions ** 2)
        return gini
    
    def _split_data(self, X: np.ndarray, y: np.ndarray, 
                    feature_idx: int, threshold: Union[float, str]) -> Tuple:
        """
        Divide os dados baseado em um atributo e threshold.
        
        Para atributos contínuos: left = valores <= threshold
        Para atributos categóricos: left = valores == threshold
        
        Args:
            X: matriz de features
            y: array de labels
            feature_idx: índice do atributo
            threshold: valor de corte
            
        Returns:
            Tuple com (X_left, y_left, X_right, y_right)
        """
        feature_values = X[:, feature_idx]
        
        # Verifica se o atributo é numérico ou categórico
        if isinstance(threshold, (int, float, np.integer, np.floating)):
            # Atributo contínuo
            left_mask = feature_values <= threshold
        else:
            # Atributo categórico
            left_mask = feature_values == threshold
        
        right_mask = ~left_mask
        
        return (X[left_mask], y[left_mask], X[right_mask], y[right_mask])
    
    def _information_gain(self, y_parent: np.ndarray, y_left: np.ndarray, 
                         y_right: np.ndarray) -> float:
        """
        Calcula a redução de impureza (ganho de informação) de uma divisão.
        
        Gain = Gini(parent) - [peso_left * Gini(left) + peso_right * Gini(right)]
        
        Args:
            y_parent: labels do nó pai
            y_left: labels do filho esquerdo
            y_right: labels do filho direito
            
        Returns:
            float: ganho de informação
        """
        n_parent = len(y_parent)
        n_left = len(y_left)
        n_right = len(y_right)
        
        if n_left == 0 or n_right == 0:
            return 0.0
        
        gini_parent = self._gini_impurity(y_parent)
        gini_left = self._gini_impurity(y_left)
        gini_right = self._gini_impurity(y_right)
        
        # Média ponderada das impurezas dos filhos
        weighted_gini = (n_left / n_parent) * gini_left + \
                       (n_right / n_parent) * gini_right
        
        return gini_parent - weighted_gini
    
    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple:
        """
        Encontra a melhor divisão binária para os dados.
        
        Testa todas as features e todos os possíveis thresholds,
        selecionando aquele que maximiza o ganho de informação.
        
        Args:
            X: matriz de features
            y: array de labels
            
        Returns:
            Tuple com (melhor_feature, melhor_threshold, melhor_ganho)
        """
        n_samples, n_features = X.shape
        
        if n_samples <= 1:
            return None, None, 0.0
        
        best_gain = 0.0
        best_feature = None
        best_threshold = None
        
        # Testa cada feature
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            
            # Determina possíveis thresholds
            if np.issubdtype(feature_values.dtype, np.number):
                # Atributo contínuo: usa pontos médios entre valores únicos
                unique_values = np.unique(feature_values)
                if len(unique_values) <= 1:
                    continue
                thresholds = (unique_values[:-1] + unique_values[1:]) / 2
            else:
                # Atributo categórico: usa cada valor único
                thresholds = np.unique(feature_values)
            
            # Testa cada threshold
            for threshold in thresholds:
                X_left, y_left, X_right, y_right = self._split_data(
                    X, y, feature_idx, threshold
                )
                
                # Verifica restrições de tamanho mínimo
                if len(y_left) < self.min_samples_leaf or \
                   len(y_right) < self.min_samples_leaf:
                    continue
                
                # Calcula ganho
                gain = self._information_gain(y, y_left, y_right)
                
                # Atualiza melhor divisão
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """
        Constrói a árvore de decisão recursivamente.
        
        Args:
            X: matriz de features
            y: array de labels
            depth: profundidade atual
            
        Returns:
            Node: nó raiz da (sub)árvore
        """
        n_samples = len(y)
        n_classes = len(np.unique(y))
        
        # Distribuição de classes
        class_counts = Counter(y)
        predicted_class = max(class_counts, key=class_counts.get)
        gini = self._gini_impurity(y)
        
        # Cria nó
        node = Node(
            gini=gini,
            n_samples=n_samples,
            class_distribution=dict(class_counts),
            value=predicted_class
        )
        
        # Critérios de parada
        if depth == self.max_depth:
            return node
        
        if n_samples < self.min_samples_split:
            return node
        
        if n_classes == 1:  # Nó puro
            return node
        
        # Encontra melhor divisão
        feature_idx, threshold, gain = self._find_best_split(X, y)
        
        if feature_idx is None or gain < self.min_impurity_decrease:
            return node
        
        # Realiza divisão
        X_left, y_left, X_right, y_right = self._split_data(
            X, y, feature_idx, threshold
        )
        
        # Atualiza nó com informações da divisão
        node.feature_idx = feature_idx
        node.threshold = threshold
        
        # Constrói subárvores recursivamente
        node.left = self._build_tree(X_left, y_left, depth + 1)
        node.right = self._build_tree(X_right, y_right, depth + 1)
        
        return node
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Treina o classificador CART.
        
        Args:
            X: matriz de features (n_samples, n_features)
            y: array de labels (n_samples,)
            
        Returns:
            self
        """
        # Converte para numpy array se necessário
        X = np.array(X)
        y = np.array(y)
        
        # Armazena número de classes
        self.n_classes_ = len(np.unique(y))
        
        # Constrói a árvore
        self.root = self._build_tree(X, y)
        
        return self
    
    def _predict_sample(self, x: np.ndarray, node: Node) -> int:
        """
        Prediz a classe para uma única amostra.
        
        Args:
            x: vetor de features
            node: nó atual
            
        Returns:
            int: classe predita
        """
        # Se é folha, retorna valor
        if node.left is None and node.right is None:
            return node.value
        
        # Decide qual subárvore seguir
        feature_value = x[node.feature_idx]
        
        if isinstance(node.threshold, (int, float, np.integer, np.floating)):
            # Atributo contínuo
            if feature_value <= node.threshold:
                return self._predict_sample(x, node.left)
            else:
                return self._predict_sample(x, node.right)
        else:
            # Atributo categórico
            if feature_value == node.threshold:
                return self._predict_sample(x, node.left)
            else:
                return self._predict_sample(x, node.right)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Prediz classes para múltiplas amostras.
        
        Args:
            X: matriz de features (n_samples, n_features)
            
        Returns:
            np.ndarray: array de predições
        """
        X = np.array(X)
        return np.array([self._predict_sample(x, self.root) for x in X])
    
    def _print_tree(self, node: Node, depth: int = 0, prefix: str = "Root: "):
        """
        Imprime a estrutura da árvore de forma legível.
        
        Args:
            node: nó atual
            depth: profundidade atual
            prefix: prefixo para impressão
        """
        if node is None:
            return
        
        indent = "  " * depth
        
        # Informações do nó
        info = f"{prefix}"
        info += f"Gini={node.gini:.3f}, samples={node.n_samples}, "
        info += f"value={node.value}, dist={node.class_distribution}"
        print(indent + info)
        
        # Se não é folha, imprime filhos
        if node.left is not None or node.right is not None:
            feature_info = f"[X{node.feature_idx} <= {node.threshold}]"
            print(indent + f"  Split: {feature_info}")
            
            if node.left is not None:
                self._print_tree(node.left, depth + 1, "Left: ")
            
            if node.right is not None:
                self._print_tree(node.right, depth + 1, "Right: ")
    
    def print_tree(self):
        """Imprime a árvore completa."""
        print("\n=== Estrutura da Árvore CART ===\n")
        self._print_tree(self.root)


# ============================================================================
# Exemplo de uso e teste
# ============================================================================

if __name__ == "__main__":
    from sklearn.datasets import load_iris, load_wine
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.tree import DecisionTreeClassifier
    
    print("=" * 70)
    print("DEMONSTRAÇÃO: Algoritmo CART Implementado do Zero")
    print("=" * 70)
    
    # ========================================================================
    # Dataset 1: Iris (clássico)
    # ========================================================================
    print("\n\n### DATASET 1: IRIS ###\n")
    
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.3, random_state=42
    )
    
    # Nossa implementação
    print("Treinando CART customizado...")
    cart_custom = CARTClassifier(max_depth=5, min_samples_split=5, min_samples_leaf=2)
    cart_custom.fit(X_train, y_train)
    y_pred_custom = cart_custom.predict(X_test)
    
    # Sklearn como baseline
    print("Treinando CART sklearn (baseline)...")
    cart_sklearn = DecisionTreeClassifier(
        criterion='gini', max_depth=5, min_samples_split=5, 
        min_samples_leaf=2, random_state=42
    )
    cart_sklearn.fit(X_train, y_train)
    y_pred_sklearn = cart_sklearn.predict(X_test)
    
    # Resultados
    print("\n--- Resultados Iris ---")
    print(f"Acurácia CART customizado: {accuracy_score(y_test, y_pred_custom):.4f}")
    print(f"Acurácia CART sklearn:     {accuracy_score(y_test, y_pred_sklearn):.4f}")
    
    print("\nClassification Report (CART customizado):")
    print(classification_report(y_test, y_pred_custom, target_names=iris.target_names))
    
    # Estrutura da árvore
    cart_custom.print_tree()
    
    # ========================================================================
    # Dataset 2: Wine
    # ========================================================================
    print("\n\n### DATASET 2: WINE ###\n")
    
    wine = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(
        wine.data, wine.target, test_size=0.3, random_state=42
    )
    
    # Nossa implementação
    print("Treinando CART customizado...")
    cart_wine = CARTClassifier(max_depth=4, min_samples_split=10)
    cart_wine.fit(X_train, y_train)
    y_pred_wine = cart_wine.predict(X_test)
    
    # Sklearn como baseline
    cart_sklearn_wine = DecisionTreeClassifier(
        criterion='gini', max_depth=4, min_samples_split=10, random_state=42
    )
    cart_sklearn_wine.fit(X_train, y_train)
    y_pred_sklearn_wine = cart_sklearn_wine.predict(X_test)
    
    # Resultados
    print("\n--- Resultados Wine ---")
    print(f"Acurácia CART customizado: {accuracy_score(y_test, y_pred_wine):.4f}")
    print(f"Acurácia CART sklearn:     {accuracy_score(y_test, y_pred_sklearn_wine):.4f}")
    
    print("\nClassification Report (CART customizado):")
    print(classification_report(y_test, y_pred_wine, target_names=wine.target_names))