import numpy as np
from collections import Counter
from typing import Optional, Dict, Tuple


class NodeC45:
    """
    Nó da árvore C4.5.
    
    Attributes:
        feature_idx: índice do atributo para divisão
        threshold: valor de corte (contínuos) ou None (categóricos)
        is_continuous: True se atributo é contínuo
        children: dicionário de filhos
        value: classe predita (folhas)
        entropy: entropia do nó
        gain_ratio: razão de ganho
        n_samples: número de amostras
        is_leaf: indica se é folha
    """
    def __init__(self, feature_idx=None, threshold=None, is_continuous=False,
                 children=None, value=None, entropy=None, gain_ratio=None,
                 n_samples=None, is_leaf=False):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.is_continuous = is_continuous
        self.children = children if children else {}
        self.value = value
        self.entropy = entropy
        self.gain_ratio = gain_ratio
        self.n_samples = n_samples
        self.is_leaf = is_leaf


class C45Classifier:
    """
    Implementação simplificada do algoritmo C4.5.
    
    CARACTERÍSTICAS PRINCIPAIS:
    - Usa Gain Ratio (corrige viés do Information Gain)
    - Suporta atributos contínuos e categóricos
    - Divisão binária para contínuos, multi-way para categóricos
    
    Parameters:
        max_depth: profundidade máxima
        min_samples_split: mínimo de amostras para dividir
        min_samples_leaf: mínimo de amostras por folha
        min_gain_ratio: razão de ganho mínima
    """
    
    def __init__(self, max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 min_gain_ratio: float = 0.0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_gain_ratio = min_gain_ratio
        self.root = None
        
    def _entropy(self, y: np.ndarray) -> float:
        """
        Calcula entropia de Shannon.
        Entropy = -Σ(p_i * log2(p_i))
        """
        if len(y) == 0:
            return 0.0
        
        _, counts = np.unique(y, return_counts=True)
        proportions = counts / len(y)
        return -np.sum([p * np.log2(p) for p in proportions if p > 0])
    
    def _split_information(self, splits: Dict) -> float:
        """
        Calcula Split Information (entropia da divisão).
        Penaliza divisões com muitas partições.
        """
        total = sum(len(y) for y in splits.values())
        if total == 0:
            return 0.0
        
        split_info = 0.0
        for y_subset in splits.values():
            if len(y_subset) > 0:
                p = len(y_subset) / total
                split_info -= p * np.log2(p)
        
        return split_info
    
    def _information_gain(self, y_parent: np.ndarray, splits: Dict) -> float:
        """
        Calcula Information Gain.
        IG = Entropy(parent) - Σ[peso_i * Entropy(child_i)]
        """
        n_parent = len(y_parent)
        if n_parent == 0:
            return 0.0
        
        parent_entropy = self._entropy(y_parent)
        weighted_child_entropy = sum(
            (len(y_child) / n_parent) * self._entropy(y_child)
            for y_child in splits.values() if len(y_child) > 0
        )
        
        return parent_entropy - weighted_child_entropy
    
    def _gain_ratio(self, y_parent: np.ndarray, splits: Dict) -> float:
        """
        Calcula Gain Ratio (inovação principal do C4.5).
        
        Gain Ratio = Information Gain / Split Information
        
        Corrige o viés do IG que favorece atributos com muitos valores.
        """
        info_gain = self._information_gain(y_parent, splits)
        split_info = self._split_information(splits)
        
        if split_info == 0.0:
            return 0.0
        
        return info_gain / split_info
    
    def _is_continuous(self, feature_values: np.ndarray) -> bool:
        """
        Detecta se atributo é contínuo ou categórico.
        Heurística: numérico + muitos valores únicos = contínuo
        """
        if not np.issubdtype(feature_values.dtype, np.number):
            return False
        
        n_unique = len(np.unique(feature_values))
        n_total = len(feature_values)
        
        return n_unique > 10 or n_unique / n_total > 0.5
    
    def _split_continuous(self, X: np.ndarray, y: np.ndarray, 
                         feature_idx: int) -> Tuple:
        """
        Encontra melhor threshold para atributo contínuo.
        Testa pontos médios entre valores consecutivos.
        """
        feature_values = X[:, feature_idx]
        unique_values = np.sort(np.unique(feature_values))
        
        if len(unique_values) <= 1:
            return None, {}
        
        best_gr = 0.0
        best_threshold = None
        best_splits = {}
        
        for i in range(len(unique_values) - 1):
            threshold = (unique_values[i] + unique_values[i + 1]) / 2
            
            left_mask = feature_values <= threshold
            y_left = y[left_mask]
            y_right = y[~left_mask]
            
            if len(y_left) < self.min_samples_leaf or \
               len(y_right) < self.min_samples_leaf:
                continue
            
            splits = {'<=': y_left, '>': y_right}
            gr = self._gain_ratio(y, splits)
            
            if gr > best_gr:
                best_gr = gr
                best_threshold = threshold
                best_splits = splits
        
        return best_threshold, best_splits
    
    def _split_categorical(self, X: np.ndarray, y: np.ndarray, 
                          feature_idx: int) -> Dict:
        """
        Divide por atributo categórico (multi-way).
        Um filho para cada valor único.
        """
        feature_values = X[:, feature_idx]
        unique_values = np.unique(feature_values)
        
        splits = {}
        for value in unique_values:
            y_subset = y[feature_values == value]
            if len(y_subset) >= self.min_samples_leaf:
                splits[value] = y_subset
        
        return splits
    
    def _find_best_split(self, X: np.ndarray, y: np.ndarray, 
                        used_categorical: list) -> Tuple:
        """
        Encontra melhor divisão testando todos atributos.
        Retorna: (feature_idx, threshold, is_continuous, splits, gain_ratio)
        """
        n_features = X.shape[1]
        
        best_gr = 0.0
        best_feature = None
        best_threshold = None
        best_is_cont = False
        best_splits = {}
        
        for feat_idx in range(n_features):
            feature_values = X[:, feat_idx]
            is_cont = self._is_continuous(feature_values)
            
            # Categóricos não são reutilizados
            if not is_cont and feat_idx in used_categorical:
                continue
            
            if is_cont:
                threshold, splits = self._split_continuous(X, y, feat_idx)
                if not splits:
                    continue
                gr = self._gain_ratio(y, splits)
            else:
                splits = self._split_categorical(X, y, feat_idx)
                if len(splits) < 2:
                    continue
                threshold = None
                gr = self._gain_ratio(y, splits)
            
            if gr > best_gr:
                best_gr = gr
                best_feature = feat_idx
                best_threshold = threshold
                best_is_cont = is_cont
                best_splits = splits
        
        return best_feature, best_threshold, best_is_cont, best_splits, best_gr
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray,
                   used_categorical: list, depth: int = 0) -> NodeC45:
        """
        Constrói árvore recursivamente.
        """
        n_samples = len(y)
        class_counts = Counter(y)
        predicted_class = max(class_counts, key=class_counts.get)
        entropy = self._entropy(y)
        
        node = NodeC45(
            entropy=entropy,
            n_samples=n_samples,
            value=predicted_class
        )
        
        # Critérios de parada
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           len(np.unique(y)) == 1:
            node.is_leaf = True
            return node
        
        # Encontra melhor divisão
        feat_idx, threshold, is_cont, splits, gr = \
            self._find_best_split(X, y, used_categorical)
        
        if feat_idx is None or gr < self.min_gain_ratio:
            node.is_leaf = True
            return node
        
        # Atualiza nó
        node.feature_idx = feat_idx
        node.threshold = threshold
        node.is_continuous = is_cont
        node.gain_ratio = gr
        
        if not is_cont:
            used_categorical = used_categorical + [feat_idx]
        
        # Cria filhos
        if is_cont:
            # Binária
            left_mask = X[:, feat_idx] <= threshold
            node.children['<='] = self._build_tree(
                X[left_mask], y[left_mask], used_categorical, depth + 1
            )
            node.children['>'] = self._build_tree(
                X[~left_mask], y[~left_mask], used_categorical, depth + 1
            )
        else:
            # Multi-way
            for value, y_subset in splits.items():
                mask = X[:, feat_idx] == value
                node.children[value] = self._build_tree(
                    X[mask], y_subset, used_categorical, depth + 1
                )
        
        return node
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Treina o classificador C4.5."""
        X = np.array(X)
        y = np.array(y)
        self.root = self._build_tree(X, y, [])
        return self
    
    def _predict_sample(self, x: np.ndarray, node: NodeC45) -> int:
        """Prediz classe de uma amostra."""
        if node.is_leaf or len(node.children) == 0:
            return node.value
        
        feature_value = x[node.feature_idx]
        
        if node.is_continuous:
            key = '<=' if feature_value <= node.threshold else '>'
            return self._predict_sample(x, node.children[key])
        else:
            if feature_value in node.children:
                return self._predict_sample(x, node.children[feature_value])
            return node.value
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Prediz classes para múltiplas amostras."""
        X = np.array(X)
        return np.array([self._predict_sample(x, self.root) for x in X])
    
    def print_tree(self, node=None, depth=0, prefix="Root"):
        """Imprime estrutura da árvore."""
        if node is None:
            node = self.root
            print("\n=== Árvore C4.5 ===\n")
        
        indent = "  " * depth
        info = f"{prefix}: n={node.n_samples}, entropy={node.entropy:.3f}, class={node.value}"
        
        if node.gain_ratio:
            info += f", GR={node.gain_ratio:.3f}"
        if node.is_leaf:
            info += " [LEAF]"
        
        print(indent + info)
        
        if not node.is_leaf and node.children:
            if node.is_continuous:
                print(indent + f"  Split: X{node.feature_idx} <= {node.threshold:.3f}")
            else:
                print(indent + f"  Split: X{node.feature_idx} (categorical)")
            
            for key, child in sorted(node.children.items(), key=str):
                self.print_tree(child, depth + 1, str(key))


# ============================================================================
# Exemplo de uso
# ============================================================================

if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    
    print("=" * 70)
    print("C4.5 Simplificado - Demonstração")
    print("=" * 70)
    
    # Dataset Iris
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.3, random_state=42
    )
    
    # Treina C4.5
    c45 = C45Classifier(
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        min_gain_ratio=0.01
    )
    c45.fit(X_train, y_train)
    
    # Avalia
    y_pred_train = c45.predict(X_train)
    y_pred_test = c45.predict(X_test)
    
    print(f"\nAcurácia Treino: {accuracy_score(y_train, y_pred_train):.4f}")
    print(f"Acurácia Teste:  {accuracy_score(y_test, y_pred_test):.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test, 
                               target_names=iris.target_names))
    
    # Estrutura da árvore
    c45.print_tree()
    
    print("\n" + "=" * 70)
    print("CARACTERÍSTICAS DO C4.5:")
    print("=" * 70)
    print("""
    ✓ Gain Ratio: corrige viés do Information Gain
    ✓ Atributos contínuos: suporte nativo com divisão binária
    ✓ Atributos categóricos: divisão multi-way
    ✓ Reutilização: atributos contínuos podem ser reusados
    """)