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

    O C4.5 é um dos algoritmos mais populares de árvores de decisão, desenvolvido
    por Ross Quinlan como sucessor do ID3. Principais características:

    1. GAIN RATIO: Usa a razão de ganho (Gain Ratio) ao invés do ganho de informação
       puro, corrigindo o viés que favorece atributos com muitos valores únicos.

    2. ATRIBUTOS CONTÍNUOS: Suporte nativo para atributos numéricos contínuos através
       de divisões binárias baseadas em thresholds.

    3. DIVISÃO MULTI-WAY: Para atributos categóricos, cria múltiplos filhos (um para
       cada valor único do atributo).

    4. REUTILIZAÇÃO: Atributos contínuos podem ser reutilizados em diferentes níveis
       da árvore com thresholds diferentes.

    Parameters:
        max_depth (int, optional): Profundidade máxima da árvore. None para sem limite.
        min_samples_split (int): Número mínimo de amostras necessárias para dividir um nó.
        min_samples_leaf (int): Número mínimo de amostras necessárias em cada folha.
        min_gain_ratio (float): Razão de ganho mínima necessária para realizar uma divisão.

    Attributes:
        root (NodeC45): Nó raiz da árvore após o treinamento.

    Example:
        >>> # Preparar seus dados
        >>> X = np.array([[...], [...]])  # Suas features
        >>> y = np.array([...])  # Seus labels
        >>>
        >>> # Criar e treinar o classificador
        >>> clf = C45Classifier(max_depth=10, min_samples_split=5)
        >>> clf.fit(X, y)
        >>>
        >>> # Fazer predições
        >>> predictions = clf.predict(X_test)
        >>>
        >>> # Visualizar a árvore
        >>> clf.print_tree()
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
        """
        Calcula a entropia de Shannon para um conjunto de labels.

        A entropia mede a impureza/incerteza de um conjunto de dados:
        - Entropia = 0: todos os elementos pertencem à mesma classe (puro)
        - Entropia máxima: distribuição uniforme entre classes

        Fórmula: Entropy = -Σ(p_i * log2(p_i))
        onde p_i é a proporção da classe i

        Args:
            y (np.ndarray): Array de labels

        Returns:
            float: Valor da entropia
        """
        if len(y) == 0:
            return 0.0

        _, counts = np.unique(y, return_counts=True)
        proportions = counts / len(y)
        return -np.sum([p * np.log2(p) for p in proportions if p > 0])

    def _split_information(self, splits: Dict) -> float:
        """
        Calcula o Split Information (entropia da divisão).

        O Split Information penaliza divisões que criam muitas partições pequenas,
        ajudando a corrigir o viés do Information Gain.

        Fórmula: SplitInfo = -Σ(|S_i|/|S| * log2(|S_i|/|S|))
        onde S_i são as partições criadas pela divisão

        Args:
            splits (Dict): Dicionário mapeando valores para subconjuntos de y

        Returns:
            float: Valor do split information
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
        Calcula o Information Gain (ganho de informação).

        O IG mede a redução na entropia obtida ao dividir os dados usando
        um determinado atributo.

        Fórmula: IG = Entropy(parent) - Σ[peso_i * Entropy(child_i)]

        Args:
            y_parent (np.ndarray): Labels do nó pai
            splits (Dict): Dicionário mapeando valores para subconjuntos de y

        Returns:
            float: Valor do information gain
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
        Calcula o Gain Ratio (razão de ganho) - a inovação principal do C4.5.

        O Gain Ratio normaliza o Information Gain pelo Split Information,
        corrigindo o viés que favorece atributos com muitos valores únicos.

        Fórmula: Gain Ratio = Information Gain / Split Information

        Por exemplo, um atributo "ID único" teria IG alto mas SplitInfo também
        alto, resultando em GR baixo (desejável).

        Args:
            y_parent (np.ndarray): Labels do nó pai
            splits (Dict): Dicionário mapeando valores para subconjuntos de y

        Returns:
            float: Valor do gain ratio
        """
        info_gain = self._information_gain(y_parent, splits)
        split_info = self._split_information(splits)

        if split_info == 0.0:
            return 0.0

        return info_gain / split_info

    def _is_continuous(self, feature_values: np.ndarray) -> bool:
        """
        Detecta automaticamente se um atributo é contínuo ou categórico.

        Heurística utilizada:
        - Se não for numérico: categórico
        - Se tiver muitos valores únicos (>10 ou >50% do total): contínuo
        - Caso contrário: categórico

        Args:
            feature_values (np.ndarray): Valores do atributo

        Returns:
            bool: True se contínuo, False se categórico
        """
        if not np.issubdtype(feature_values.dtype, np.number):
            return False

        n_unique = len(np.unique(feature_values))
        n_total = len(feature_values)

        # Heurística: muitos valores únicos indica variável contínua
        return n_unique > 10 or n_unique / n_total > 0.5

    def _split_continuous(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_idx: int
    ) -> Tuple[Optional[float], Dict]:
        """
        Encontra o melhor threshold para dividir um atributo contínuo.

        Estratégia:
        1. Ordena os valores únicos do atributo
        2. Testa todos os pontos médios entre valores consecutivos como thresholds
        3. Para cada threshold, cria divisão binária (<=threshold, >threshold)
        4. Calcula o Gain Ratio de cada divisão
        5. Retorna o threshold com maior Gain Ratio

        Args:
            X (np.ndarray): Matriz de features
            y (np.ndarray): Array de labels
            feature_idx (int): Índice do atributo a ser testado

        Returns:
            Tuple[Optional[float], Dict]: (melhor_threshold, splits) ou (None, {})
        """
        feature_values = X[:, feature_idx]
        unique_values = np.sort(np.unique(feature_values))

        if len(unique_values) <= 1:
            return None, {}

        best_gr = 0.0
        best_threshold = None
        best_splits = {}

        # Testa cada ponto médio como threshold
        for i in range(len(unique_values) - 1):
            threshold = (unique_values[i] + unique_values[i + 1]) / 2

            left_mask = feature_values <= threshold
            y_left = y[left_mask]
            y_right = y[~left_mask]

            # Verifica restrições de tamanho mínimo
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
        """
        Cria divisão multi-way para atributo categórico.

        Para atributos categóricos, o C4.5 cria um filho para cada valor
        único do atributo (divisão multi-way), diferente da divisão binária
        usada para atributos contínuos.

        Args:
            X (np.ndarray): Matriz de features
            y (np.ndarray): Array de labels
            feature_idx (int): Índice do atributo a ser testado

        Returns:
            Dict: Dicionário mapeando cada valor único para seu subconjunto de y
        """
        feature_values = X[:, feature_idx]
        unique_values = np.unique(feature_values)

        splits = {}
        for value in unique_values:
            y_subset = y[feature_values == value]
            # Respeita restrição de tamanho mínimo
            if len(y_subset) >= self.min_samples_leaf:
                splits[value] = y_subset

        return splits

    def _find_best_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        used_categorical: list
    ) -> Tuple[Optional[int], Optional[float], bool, Dict, float]:
        """
        Encontra a melhor divisão testando todos os atributos disponíveis.

        Para cada atributo:
        - Detecta se é contínuo ou categórico
        - Encontra a melhor divisão para aquele atributo
        - Calcula o Gain Ratio

        Retorna a divisão com maior Gain Ratio.

        Args:
            X (np.ndarray): Matriz de features
            y (np.ndarray): Array de labels
            used_categorical (list): Lista de índices de atributos categóricos já usados

        Returns:
            Tuple contendo:
                - feature_idx (int): Índice do melhor atributo
                - threshold (float): Threshold para contínuos, None para categóricos
                - is_continuous (bool): Se o atributo é contínuo
                - splits (Dict): Dicionário com as divisões
                - gain_ratio (float): Valor do gain ratio
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

            # Atributos categóricos não podem ser reutilizados
            if not is_cont and feat_idx in used_categorical:
                continue

            if is_cont:
                threshold, splits = self._split_continuous(X, y, feat_idx)
                if not splits:
                    continue
                gr = self._gain_ratio(y, splits)
            else:
                splits = self._split_categorical(X, y, feat_idx)
                if len(splits) < 2:  # Precisa de pelo menos 2 partições
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
        """
        Constrói a árvore de decisão recursivamente.

        Algoritmo:
        1. Calcula estatísticas do nó atual (entropia, classe majoritária)
        2. Verifica critérios de parada (profundidade, tamanho, pureza)
        3. Se deve parar: retorna folha
        4. Se não: encontra melhor divisão
        5. Cria nós filhos recursivamente para cada partição

        Args:
            X (np.ndarray): Matriz de features das amostras neste nó
            y (np.ndarray): Array de labels das amostras neste nó
            used_categorical (list): Atributos categóricos já usados no caminho
            depth (int): Profundidade atual na árvore

        Returns:
            NodeC45: Nó construído (pode ser folha ou nó interno)
        """
        n_samples = len(y)
        class_counts = Counter(y)
        predicted_class = max(class_counts, key=class_counts.get)
        entropy = self._entropy(y)

        # Cria o nó com informações básicas
        node = NodeC45(
            entropy=entropy,
            n_samples=n_samples,
            value=predicted_class
        )

        # Critérios de parada
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           len(np.unique(y)) == 1:  # Nó puro
            node.is_leaf = True
            return node

        # Encontra melhor divisão
        feat_idx, threshold, is_cont, splits, gr = \
            self._find_best_split(X, y, used_categorical)

        # Se não encontrou divisão válida ou Gain Ratio muito baixo
        if feat_idx is None or gr < self.min_gain_ratio:
            node.is_leaf = True
            return node

        # Atualiza informações do nó interno
        node.feature_idx = feat_idx
        node.threshold = threshold
        node.is_continuous = is_cont
        node.gain_ratio = gr

        # Marca atributo categórico como usado
        if not is_cont:
            used_categorical = used_categorical + [feat_idx]

        # Cria nós filhos recursivamente
        if is_cont:
            # Divisão binária para atributos contínuos
            left_mask = X[:, feat_idx] <= threshold
            node.children['<='] = self._build_tree(
                X[left_mask], y[left_mask], used_categorical, depth + 1
            )
            node.children['>'] = self._build_tree(
                X[~left_mask], y[~left_mask], used_categorical, depth + 1
            )
        else:
            # Divisão multi-way para atributos categóricos
            for value, y_subset in splits.items():
                mask = X[:, feat_idx] == value
                node.children[value] = self._build_tree(
                    X[mask], y_subset, used_categorical, depth + 1
                )

        return node

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'C45Classifier':
        """
        Treina o classificador C4.5 com os dados fornecidos.

        Args:
            X (np.ndarray): Matriz de features, shape (n_samples, n_features)
            y (np.ndarray): Array de labels, shape (n_samples,)

        Returns:
            self: Retorna a própria instância para permitir method chaining
        """
        X = np.array(X)
        y = np.array(y)
        self.root = self._build_tree(X, y, [])
        return self

    def _predict_sample(self, x: np.ndarray, node: NodeC45) -> Any:
        """
        Prediz a classe de uma única amostra navegando pela árvore.

        Processo:
        1. Se nó é folha: retorna classe predita
        2. Se não: avalia condição do atributo
        3. Segue para o filho apropriado
        4. Repete recursivamente

        Args:
            x (np.ndarray): Vetor de features da amostra
            node (NodeC45): Nó atual na navegação

        Returns:
            Classe predita para a amostra
        """
        # Caso base: folha ou nó sem filhos
        if node.is_leaf or len(node.children) == 0:
            return node.value

        feature_value = x[node.feature_idx]

        if node.is_continuous:
            # Atributo contínuo: usa threshold
            key = '<=' if feature_value <= node.threshold else '>'
            return self._predict_sample(x, node.children[key])
        else:
            # Atributo categórico: busca valor exato
            if feature_value in node.children:
                return self._predict_sample(x, node.children[feature_value])
            # Valor não visto no treino: retorna classe majoritária do nó
            return node.value

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Prediz classes para múltiplas amostras.

        Args:
            X (np.ndarray): Matriz de features, shape (n_samples, n_features)

        Returns:
            np.ndarray: Array de predições, shape (n_samples,)
        """
        X = np.array(X)
        return np.array([self._predict_sample(x, self.root) for x in X])

    def print_tree(self, node: Optional[NodeC45] = None, depth: int = 0, prefix: str = "Root") -> None:
        """
        Imprime a estrutura da árvore de forma hierárquica.

        Útil para visualizar e debugar a árvore construída.

        Args:
            node (NodeC45, optional): Nó a imprimir (None = raiz)
            depth (int): Profundidade atual (para indentação)
            prefix (str): Prefixo descritivo do nó
        """
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


if __name__ == "__main__":
    print(__doc__)
    print("\nPara usar este classificador, importe e utilize conforme exemplo abaixo:\n")
    print("from c45 import C45Classifier")
    print("import numpy as np")
    print()
    print("# Carregue seus dados")
    print("X = np.array([...])")
    print("y = np.array([...])")
    print()
    print("# Crie e treine o classificador")
    print("clf = C45Classifier(max_depth=10, min_samples_split=5)")
    print("clf.fit(X, y)")
    print()
    print("# Faça predições")
    print("predictions = clf.predict(X_test)")
    print()
    print("# Visualize a árvore")
    print("clf.print_tree()")
