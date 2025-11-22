"""
Implementação do algoritmo C4.5 de Árvore de Decisão.

O C4.5 é uma evolução do ID3 que introduz várias melhorias:
- Usa Gain Ratio ao invés de Information Gain
- Suporta atributos contínuos nativamente
- Suporta atributos categóricos com divisão multi-way
- Permite reutilização de atributos contínuos

Autor: Gerado com assistência de IA
Data: 2025-11-14
"""

import numpy as np
from collections import Counter
from typing import Optional, Dict, Tuple, Any


class NodeC45:
    """
    Representa um nó da árvore de decisão C4.5.

    Attributes:
        feature_idx (int): Índice do atributo usado para divisão neste nó
        threshold (float): Valor de corte para atributos contínuos (None para categóricos)
        is_continuous (bool): True se o atributo é contínuo, False se categórico
        children (dict): Dicionário mapeando valores/condições para nós filhos
        value (Any): Classe predita (usado em folhas)
        entropy (float): Entropia das amostras neste nó
        gain_ratio (float): Razão de ganho da divisão escolhida
        n_samples (int): Número de amostras que chegaram a este nó
        is_leaf (bool): Indica se este nó é uma folha
    """

    def __init__(
        self,
        feature_idx: Optional[int] = None,
        threshold: Optional[float] = None,
        is_continuous: bool = False,
        children: Optional[Dict] = None,
        value: Any = None,
        entropy: Optional[float] = None,
        gain_ratio: Optional[float] = None,
        n_samples: Optional[int] = None,
        is_leaf: bool = False
    ):
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
    Classificador baseado no algoritmo C4.5.
    """

    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_gain_ratio: float = 0.0
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_gain_ratio = min_gain_ratio
        self.root = None

    def _entropy(self, y: np.ndarray) -> float:
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        proportions = counts / len(y)
        return -np.sum([p * np.log2(p) for p in proportions if p > 0])

    def _split_information(self, splits: Dict) -> float:
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
        info_gain = self._information_gain(y_parent, splits)
        split_info = self._split_information(splits)
        if split_info == 0.0:
            return 0.0
        return info_gain / split_info

    def _is_continuous(self, feature_values: np.ndarray) -> bool:
        if not np.issubdtype(feature_values.dtype, np.number):
            return False
        n_unique = len(np.unique(feature_values))
        n_total = len(feature_values)
        return n_unique > 10 or n_unique / n_total > 0.5

    def _split_continuous(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_idx: int
    ) -> Tuple[Optional[float], Dict]:
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

    def _split_categorical(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_idx: int
    ) -> Dict:
        feature_values = X[:, feature_idx]
        unique_values = np.unique(feature_values)
        splits = {}
        for value in unique_values:
            y_subset = y[feature_values == value]
            if len(y_subset) >= self.min_samples_leaf:
                splits[value] = y_subset
        return splits

    def _find_best_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        used_categorical: list
    ) -> Tuple[Optional[int], Optional[float], bool, Dict, float]:
        n_features = X.shape[1]
        best_gr = 0.0
        best_feature = None
        best_threshold = None
        best_is_cont = False
        best_splits = {}
        for feat_idx in range(n_features):
            feature_values = X[:, feat_idx]
            is_cont = self._is_continuous(feature_values)
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

    def _build_tree(
        self,
        X: np.ndarray,
        y: np.ndarray,
        used_categorical: list,
        depth: int = 0
    ) -> NodeC45:
        n_samples = len(y)
        class_counts = Counter(y)
        predicted_class = max(class_counts, key=class_counts.get)
        entropy = self._entropy(y)
        node = NodeC45(
            entropy=entropy,
            n_samples=n_samples,
            value=predicted_class
        )
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           len(np.unique(y)) == 1:
            node.is_leaf = True
            return node
        feat_idx, threshold, is_cont, splits, gr = \
            self._find_best_split(X, y, used_categorical)
        if feat_idx is None or gr < self.min_gain_ratio:
            node.is_leaf = True
            return node
        node.feature_idx = feat_idx
        node.threshold = threshold
        node.is_continuous = is_cont
        node.gain_ratio = gr
        if not is_cont:
            used_categorical = used_categorical + [feat_idx]
        if is_cont:
            left_mask = X[:, feat_idx] <= threshold
            node.children['<='] = self._build_tree(
                X[left_mask], y[left_mask], used_categorical, depth + 1
            )
            node.children['>'] = self._build_tree(
                X[~left_mask], y[~left_mask], used_categorical, depth + 1
            )
        else:
            for value, y_subset in splits.items():
                mask = X[:, feat_idx] == value
                node.children[value] = self._build_tree(
                    X[mask], y_subset, used_categorical, depth + 1
                )
        return node

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'C45Classifier':
        X = np.array(X)
        y = np.array(y)
        self.root = self._build_tree(X, y, [])
        return self

    def _predict_sample(self, x: np.ndarray, node: NodeC45) -> Any:
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
        X = np.array(X)
        return np.array([self._predict_sample(x, self.root) for x in X])

    def print_tree(self, node: Optional[NodeC45] = None, depth: int = 0, prefix: str = "Root") -> None:
        if node is None:
            node = self.root
            print("\n" + "=" * 70)
            print("ESTRUTURA DA ÁRVORE C4.5")
            print("=" * 70 + "\n")
        indent = "  " * depth
        info = f"{prefix}: n={node.n_samples}, entropy={node.entropy:.3f}, class={node.value}"
        if node.gain_ratio:
            info += f", GR={node.gain_ratio:.3f}"
        if node.is_leaf:
            info += " [FOLHA]"
        print(indent + info)
        if not node.is_leaf and node.children:
            if node.is_continuous:
                print(indent + f"  ├─ Divisão: X[{node.feature_idx}] <= {node.threshold:.3f}")
            else:
                print(indent + f"  ├─ Divisão: X[{node.feature_idx}] (categórico)")
            for key, child in sorted(node.children.items(), key=str):
                self.print_tree(child, depth + 1, f"[{key}]")