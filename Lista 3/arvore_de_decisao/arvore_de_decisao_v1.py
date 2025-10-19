"""""
Árvore de Decisão - Versão 1

Autor: Lucas Rafael P. do Nascimento
Data: 27/06/2024

Descrição: Implementação de uma árvore de decisão simples em Python. Código gerado inicialmente por ChatGPT, corrigido pelo Claude e editado por mim.


Este módulo implementa uma árvore de decisão para classificação usando
o critério de ganho de informação baseado em entropia.


O _ no início indica que aquele método ou atributo é "privado" ou "interno" à classe, ou seja, não deveria ser acessado diretamente de fora da classe.

Python não impede realmente o acesso a esses membros. É uma convenção social entre programadores que diz: "Ei, esse método é interno, use por sua conta e risco!"
"""

from math import log2
from collections import Counter

class Node:
    """Representa um nó na árvore de decisão.
    
    Attributes:
        feature (int): Índice da característica usada para divisão.
        threshold (float): Valor do limiar para a divisão.
        left (Node): Nó filho à esquerda (valores < threshold).
        right (Node): Nó filho à direita (valores >= threshold).
        value: Classe prevista para nós folha.
    """
    
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        """Inicializa um nó da árvore.
        
        Args:
            feature (int, optional): Índice da característica para divisão.
            threshold (float, optional): Valor do limiar para divisão.
            left (Node, optional): Nó filho esquerdo.
            right (Node, optional): Nó filho direito.
            value (optional): Valor da classe para nós folha.
        """
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTree:
    """Implementação de uma Árvore de Decisão para classificação.
    
    Esta classe implementa um classificador baseado em árvore de decisão
    que usa entropia e ganho de informação para selecionar as melhores
    divisões em cada nó.
    
    Attributes:
        root (Node): Nó raiz da árvore após o treinamento.
        max_depth (int): Profundidade máxima permitida para a árvore.
    
    Examples:
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.model_selection import train_test_split
        >>> 
        >>> X, y = load_iris(return_X_y=True)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
        >>> 
        >>> tree = DecisionTree(max_depth=5)
        >>> tree.fit(X_train, y_train)
        >>> predictions = tree.predict(X_test)
    """
    
    def __init__(self, max_depth=None):
        """Inicializa a árvore de decisão.
        
        Args:
            max_depth (int, optional): Profundidade máxima da árvore.
                Se None, a árvore crescerá até que todos os nós sejam puros
                ou não haja mais divisões possíveis. Padrão é None.
        """
        self.root = None
        self.max_depth = max_depth

    def fit(self, X, y):
        """Treina a árvore de decisão com os dados fornecidos.
        
        Args:
            X (array-like): Matriz de características de forma (n_samples, n_features).
            y (array-like): Vetor de rótulos de forma (n_samples,).
        
        Returns:
            self: Retorna a própria instância para permitir encadeamento.
        """
        self.root = self._build_tree(X, y)
        return self

    def _build_tree(self, X, y, depth=0):
        """Constrói a árvore recursivamente.
        
        Args:
            X (array-like): Matriz de características.
            y (array-like): Vetor de rótulos.
            depth (int): Profundidade atual na árvore.
        
        Returns:
            Node: Nó raiz da subárvore construída.
        """
        num_samples, num_features = X.shape
        unique_classes = set(y)

        # Critério de parada
        if len(unique_classes) == 1 or (self.max_depth and depth >= self.max_depth):
            most_common_class = max(unique_classes, key=list(y).count)
            return Node(value=most_common_class)

        best_feature, best_threshold = self._best_split(X, y, num_features)
        if best_feature is None:
            most_common_class = max(unique_classes, key=list(y).count)
            return Node(value=most_common_class)

        left_indices = X[:, best_feature] < best_threshold
        right_indices = X[:, best_feature] >= best_threshold

        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return Node(feature=best_feature, threshold=best_threshold, 
                   left=left_subtree, right=right_subtree)

    def _best_split(self, X, y, num_features):
        """Encontra a melhor divisão para os dados.
        
        Itera sobre todas as características e possíveis thresholds para
        encontrar a divisão que maximiza o ganho de informação.
        
        Args:
            X (array-like): Matriz de características.
            y (array-like): Vetor de rótulos.
            num_features (int): Número de características.
        
        Returns:
            tuple: (best_feature, best_threshold) ou (None, None) se não
                houver divisão válida.
        """
        best_gain = -1
        best_feature = None
        best_threshold = None

        for feature in range(num_features):
            thresholds = set(X[:, feature])
            for threshold in thresholds:
                gain = self._information_gain(X, y, feature, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _information_gain(self, X, y, feature, threshold):
        """Calcula o ganho de informação de uma divisão.
        
        Args:
            X (array-like): Matriz de características.
            y (array-like): Vetor de rótulos.
            feature (int): Índice da característica.
            threshold (float): Valor do limiar.
        
        Returns:
            float: Ganho de informação da divisão proposta.
        """
        parent_entropy = self._entropy(y)

        left_indices = X[:, feature] < threshold
        right_indices = X[:, feature] >= threshold

        if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
            return 0

        n = len(y)
        n_left, n_right = len(y[left_indices]), len(y[right_indices])
        child_entropy = (n_left / n) * self._entropy(y[left_indices]) + \
                       (n_right / n) * self._entropy(y[right_indices])
        return parent_entropy - child_entropy
    
    def _entropy(self, y):
        """Calcula a entropia de um conjunto de rótulos.
        
        A entropia mede a impureza ou incerteza dos dados. Uma entropia
        de 0 indica que todos os elementos pertencem à mesma classe.
        
        Args:
            y (array-like): Vetor de rótulos.
        
        Returns:
            float: Valor da entropia.
        """
        class_counts = Counter(y)
        n = len(y)
        entropy = 0.0

        for count in class_counts.values():
            probability = count / n
            entropy -= probability * log2(probability)

        return entropy
    
    def predict(self, X):
        """Faz predições para um conjunto de amostras.
        
        Args:
            X (array-like): Matriz de características de forma 
                (n_samples, n_features).
        
        Returns:
            list: Lista de predições para cada amostra.
        """
        return [self._traverse_tree(x, self.root) for x in X]
    
    def _traverse_tree(self, x, node):
        """Percorre a árvore para fazer uma predição individual.
        
        Args:
            x (array-like): Vetor de características de uma amostra.
            node (Node): Nó atual na travessia.
        
        Returns:
            Classe prevista para a amostra.
        """
        if node.value is not None:
            return node.value
        
        if x[node.feature] < node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)
        
    def print_tree(self, feature_names=None, class_names=None, decimals=2):
        """Imprime a árvore de decisão em formato de texto hierárquico.
        
        Args:
            feature_names (list, optional): Lista com nomes das características.
                Se None, usa índices (feature_0, feature_1, etc.).
            class_names (list, optional): Lista com nomes das classes.
                Se None, usa os valores originais.
            decimals (int): Número de casas decimais para thresholds.
        
        Examples:
            >>> tree.print_tree(feature_names=['idade', 'salário'], 
            ...                 class_names=['não', 'sim'])
        """
        
        def _print_node(node, depth=0, prefix=""):
            indent = "│   " * depth
            
            if node.value is not None:
                # Nó folha
                class_label = class_names[node.value] if class_names else node.value
                print(f"{indent}└── Classe: {class_label}")
            else:
                # Nó de decisão
                feature_name = feature_names[node.feature] if feature_names else f"X[{node.feature}]"
                threshold = round(node.threshold, decimals)
                
                print(f"{indent}├── {feature_name} < {threshold}?")
                _print_node(node.left, depth + 1, "Sim")
                
                print(f"{indent}├── {feature_name} >= {threshold}?")
                _print_node(node.right, depth + 1, "Não")
        
        print("\n=== Árvore de Decisão ===")
        if self.root is None:
            print("Árvore não treinada!")
            return
        
        _print_node(self.root)
        print("=" * 25 + "\n")


    def get_depth(self):
            """Retorna a profundidade atual da árvore.
            
            Returns:
                int: Profundidade da árvore (número de níveis).
            """
            def _get_depth(node):
                if node is None or node.value is not None:
                    return 0
                return 1 + max(_get_depth(node.left), _get_depth(node.right))
            
            return _get_depth(self.root)


    def get_n_leaves(self):
            """Retorna o número de nós folha na árvore.
            
            Returns:
                int: Número de folhas.
            """
            def _count_leaves(node):
                if node is None:
                    return 0
                if node.value is not None:
                    return 1
                return _count_leaves(node.left) + _count_leaves(node.right)
            
            return _count_leaves(self.root)


    def get_n_nodes(self):
            """Retorna o número total de nós na árvore.
            
            Returns:
                int: Número total de nós (internos + folhas).
            """
            def _count_nodes(node):
                if node is None:
                    return 0
                return 1 + _count_nodes(node.left) + _count_nodes(node.right)
            
            return _count_nodes(self.root)


    def to_dict(self, node=None):
            """Converte a árvore para um dicionário (útil para JSON).
            
            Args:
                node (Node, optional): Nó inicial. Se None, usa a raiz.
            
            Returns:
                dict: Representação da árvore em dicionário.
            
            Examples:
                >>> import json
                >>> tree_dict = tree.to_dict()
                >>> print(json.dumps(tree_dict, indent=2))
            """
            if node is None:
                node = self.root
            
            if node is None:
                return None
            
            if node.value is not None:
                return {
                    'type': 'leaf',
                    'value': node.value
                }
            
            return {
                'type': 'split',
                'feature': node.feature,
                'threshold': node.threshold,
                'left': self.to_dict(node.left),
                'right': self.to_dict(node.right)
            }


    def visualize_text(self, feature_names=None, max_depth=None):
            """Visualização em texto mais detalhada com estatísticas.
            
            Args:
                feature_names (list, optional): Nomes das características.
                max_depth (int, optional): Profundidade máxima a exibir.
            """
            def _visualize(node, depth=0, side="ROOT"):
                if max_depth and depth > max_depth:
                    return
                
                indent = "  " * depth
                
                if node.value is not None:
                    print(f"{indent}[{side}] LEAF → Class: {node.value}")
                else:
                    feature_name = feature_names[node.feature] if feature_names else f"feature_{node.feature}"
                    print(f"{indent}[{side}] {feature_name} < {node.threshold:.4f}")
                    
                    if node.left:
                        _visualize(node.left, depth + 1, "LEFT")
                    if node.right:
                        _visualize(node.right, depth + 1, "RIGHT")
            
            print("\n" + "="*50)
            print(f"Árvore de Decisão - Profundidade: {self.get_depth()}")
            print(f"Folhas: {self.get_n_leaves()} | Nós totais: {self.get_n_nodes()}")
            print("="*50)
            
            if self.root:
                _visualize(self.root)
            else:
                print("Árvore não treinada!")
            
            print("="*50 + "\n")


    def plot_tree(self, feature_names=None, class_names=None, 
                    filename='decision_tree.png', dpi=100):
            """Cria uma visualização gráfica da árvore usando matplotlib.
            
            Requer: matplotlib
            
            Args:
                feature_names (list, optional): Nomes das características.
                class_names (list, optional): Nomes das classes.
                filename (str): Nome do arquivo para salvar a imagem.
                dpi (int): Resolução da imagem.
            
            Examples:
                >>> tree.plot_tree(feature_names=['idade', 'salário'], 
                ...                class_names=['não', 'sim'])
            """
            try:
                import matplotlib.pyplot as plt
                import matplotlib.patches as mpatches
            except ImportError:
                print("Erro: matplotlib não está instalado.")
                print("Instale com: pip install matplotlib")
                return
            
            fig, ax = plt.subplots(figsize=(15, 10))
            ax.axis('off')
            
            def _get_tree_dimensions(node):
                if node is None or node.value is not None:
                    return 1
                return _get_tree_dimensions(node.left) + _get_tree_dimensions(node.right)
            
            total_width = _get_tree_dimensions(self.root)
            
            def _plot_node(node, x, y, width, depth=0):
                if node is None:
                    return
                
                # Cores
                if node.value is not None:
                    color = '#90EE90'  # Verde para folhas
                    label = class_names[node.value] if class_names else str(node.value)
                    text = f"Class: {label}"
                else:
                    color = '#87CEEB'  # Azul para nós de decisão
                    feature_name = feature_names[node.feature] if feature_names else f"X[{node.feature}]"
                    text = f"{feature_name}\n< {node.threshold:.2f}"
                
                # Desenhar caixa
                box = mpatches.FancyBboxPatch((x - 0.15, y - 0.08), 0.3, 0.16,
                                            boxstyle="round,pad=0.01",
                                            edgecolor='black', facecolor=color,
                                            linewidth=2)
                ax.add_patch(box)
                ax.text(x, y, text, ha='center', va='center', fontsize=9, weight='bold')
                
                # Desenhar filhos
                if node.value is None:
                    y_child = y - 0.3
                    
                    # Filho esquerdo
                    if node.left:
                        x_left = x - width / 4
                        ax.plot([x, x_left], [y - 0.08, y_child + 0.08], 'k-', linewidth=1.5)
                        ax.text((x + x_left) / 2 - 0.05, (y + y_child) / 2, 
                            'True', fontsize=8, style='italic')
                        _plot_node(node.left, x_left, y_child, width / 2, depth + 1)
                    
                    # Filho direito
                    if node.right:
                        x_right = x + width / 4
                        ax.plot([x, x_right], [y - 0.08, y_child + 0.08], 'k-', linewidth=1.5)
                        ax.text((x + x_right) / 2 + 0.05, (y + y_child) / 2, 
                            'False', fontsize=8, style='italic')
                        _plot_node(node.right, x_right, y_child, width / 2, depth + 1)
            
            if self.root:
                _plot_node(self.root, 0.5, 0.9, 1.0)
                plt.title('Árvore de Decisão', fontsize=16, weight='bold', pad=20)
                plt.tight_layout()
                plt.savefig(filename, dpi=dpi, bbox_inches='tight')
                print(f"Árvore salva em: {filename}")
                plt.show()
            else:
                print("Árvore não treinada!")


    def get_rules(self, feature_names=None, class_names=None):
            """Extrai as regras de decisão em formato legível.
            
            Args:
                feature_names (list, optional): Nomes das características.
                class_names (list, optional): Nomes das classes.
            
            Returns:
                list: Lista de strings com as regras de decisão.
            
            Examples:
                >>> rules = tree.get_rules(feature_names=['idade', 'salário'])
                >>> for rule in rules:
                ...     print(rule)
            """
            rules = []
            
            def _extract_rules(node, conditions=[]):
                if node is None:
                    return
                
                if node.value is not None:
                    # Nó folha - adicionar regra
                    class_label = class_names[node.value] if class_names else node.value
                    if conditions:
                        rule = "IF " + " AND ".join(conditions) + f" THEN class = {class_label}"
                    else:
                        rule = f"class = {class_label}"
                    rules.append(rule)
                else:
                    feature_name = feature_names[node.feature] if feature_names else f"X[{node.feature}]"
                    
                    # Ramo esquerdo (condição verdadeira)
                    left_condition = f"{feature_name} < {node.threshold:.4f}"
                    _extract_rules(node.left, conditions + [left_condition])
                    
                    # Ramo direito (condição falsa)
                    right_condition = f"{feature_name} >= {node.threshold:.4f}"
                    _extract_rules(node.right, conditions + [right_condition])
            
            _extract_rules(self.root)
            return rules


    def print_rules(self, feature_names=None, class_names=None):
            """Imprime as regras de decisão de forma formatada.
            
            Args:
                feature_names (list, optional): Nomes das características.
                class_names (list, optional): Nomes das classes.
            """
            rules = self.get_rules(feature_names, class_names)
            
            print("\n" + "="*60)
            print("REGRAS DE DECISÃO DA ÁRVORE")
            print("="*60)
            
            for i, rule in enumerate(rules, 1):
                print(f"\nRegra {i}:")
                print(f"  {rule}")
            
            print("\n" + "="*60 + "\n")


    def get_feature_importance(self, n_features):
            """Calcula a importância de cada característica.
            
            Args:
                n_features (int): Número total de características.
            
            Returns:
                dict: Dicionário com índice da feature e sua importância.
            """
            importance = {i: 0 for i in range(n_features)}
            
            def _calculate_importance(node):
                if node is None or node.value is not None:
                    return
                
                importance[node.feature] += 1
                _calculate_importance(node.left)
                _calculate_importance(node.right)
            
            _calculate_importance(self.root)
            
            # Normalizar
            total = sum(importance.values())
            if total > 0:
                importance = {k: v/total for k, v in importance.items()}
            
            return importance


    def print_feature_importance(self, feature_names=None, n_features=None):
            """Imprime a importância das características ordenadas.
            
            Args:
                feature_names (list, optional): Nomes das características.
                n_features (int, optional): Número de características. 
                    Se None, tenta inferir de feature_names.
            """
            if n_features is None:
                if feature_names:
                    n_features = len(feature_names)
                else:
                    print("Erro: forneça n_features ou feature_names")
                    return
            
            importance = self.get_feature_importance(n_features)
            
            # Ordenar por importância
            sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            
            print("\n" + "="*50)
            print("IMPORTÂNCIA DAS CARACTERÍSTICAS")
            print("="*50)
            
            for feature_idx, imp in sorted_importance:
                if imp > 0:
                    feature_name = feature_names[feature_idx] if feature_names else f"Feature {feature_idx}"
                    bar = "█" * int(imp * 50)
                    print(f"{feature_name:20s} | {bar} {imp:.4f}")
            
            print("="*50 + "\n")