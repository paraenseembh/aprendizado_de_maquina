import numpy as np
from collections import Counter
from typing import Optional, Union, List, Dict
import warnings
warnings.filterwarnings('ignore')


class NodeID3:
    """
    Representa um nó na árvore de decisão ID3.
    
    Attributes:
        feature_idx: índice do atributo usado para divisão
        feature_value: valor específico do atributo (para categóricos)
        children: dicionário de filhos {valor: Node}
        value: classe predita (para nós folha)
        entropy: entropia do nó
        n_samples: número de amostras no nó
        class_distribution: distribuição de classes
        is_leaf: indica se é nó folha
    """
    def __init__(self, feature_idx=None, feature_value=None, children=None,
                 value=None, entropy=None, n_samples=None, 
                 class_distribution=None, is_leaf=False):
        self.feature_idx = feature_idx
        self.feature_value = feature_value
        self.children = children if children else {}
        self.value = value
        self.entropy = entropy
        self.n_samples = n_samples
        self.class_distribution = class_distribution
        self.is_leaf = is_leaf


class ID3Classifier:
    """
    Implementação simplificada do algoritmo ID3 (Iterative Dichotomiser 3).
    
    Características principais:
    - Usa Ganho de Informação (Information Gain) como critério
    - Trabalha APENAS com atributos categóricos
    - Cria divisões multi-way (um filho por valor do atributo)
    - Não realiza poda
    
    DIFERENÇAS EM RELAÇÃO AO CART:
    - ID3: Information Gain (baseado em entropia)
    - CART: Gini Impurity
    
    - ID3: Divisões multi-way (múltiplos filhos)
    - CART: Divisões binárias (sempre 2 filhos)
    
    - ID3: Apenas atributos categóricos
    - CART: Atributos categóricos e contínuos
    
    Parameters:
        max_depth: profundidade máxima da árvore
        min_samples_split: número mínimo de amostras para dividir
        min_samples_leaf: número mínimo de amostras em folha
        min_information_gain: ganho mínimo para realizar divisão
    """
    
    def __init__(self, max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 min_information_gain: float = 0.0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_information_gain = min_information_gain
        self.root = None
        self.n_classes_ = None
        self.n_features_ = None
        
    def _entropy(self, y: np.ndarray) -> float:
        """
        Calcula a entropia de Shannon para um conjunto de labels.
        
        Entropy = -Σ(p_i * log2(p_i)), onde p_i é a proporção da classe i
        
        Interpretação:
        - Entropia = 0: conjunto puro (uma única classe)
        - Entropia máxima: distribuição uniforme das classes
        
        Args:
            y: array de labels
            
        Returns:
            float: entropia (em bits)
        """
        if len(y) == 0:
            return 0.0
        
        # Calcula proporções de cada classe
        _, counts = np.unique(y, return_counts=True)
        proportions = counts / len(y)
        
        # Calcula entropia (evita log(0) usando proporções > 0)
        entropy = -np.sum([p * np.log2(p) for p in proportions if p > 0])
        
        return entropy
    
    def _information_gain(self, y_parent: np.ndarray, 
                         splits: Dict[any, np.ndarray]) -> float:
        """
        Calcula o Ganho de Informação de uma divisão.
        
        Information Gain = Entropy(parent) - Σ[(|child_i|/|parent|) * Entropy(child_i)]
        
        Mede quanto a divisão reduz a incerteza sobre as classes.
        
        Args:
            y_parent: labels do nó pai
            splits: dicionário {valor: labels_filho}
            
        Returns:
            float: ganho de informação
        """
        n_parent = len(y_parent)
        if n_parent == 0:
            return 0.0
        
        # Entropia do pai
        parent_entropy = self._entropy(y_parent)
        
        # Entropia ponderada dos filhos
        weighted_child_entropy = 0.0
        for value, y_child in splits.items():
            if len(y_child) > 0:
                weight = len(y_child) / n_parent
                weighted_child_entropy += weight * self._entropy(y_child)
        
        # Ganho de informação
        information_gain = parent_entropy - weighted_child_entropy
        
        return information_gain
    
    def _split_data(self, X: np.ndarray, y: np.ndarray, 
                    feature_idx: int) -> Dict:
        """
        Divide os dados baseado em um atributo categórico.
        
        Cria um subconjunto para cada valor único do atributo.
        NOTA: ID3 faz divisões multi-way, não binárias!
        
        Args:
            X: matriz de features
            y: array de labels
            feature_idx: índice do atributo
            
        Returns:
            Dict: {valor: (X_subset, y_subset)}
        """
        feature_values = X[:, feature_idx]
        unique_values = np.unique(feature_values)
        
        splits = {}
        for value in unique_values:
            mask = feature_values == value
            splits[value] = (X[mask], y[mask])
        
        return splits
    
    def _find_best_feature(self, X: np.ndarray, y: np.ndarray, 
                          available_features: List[int]) -> tuple:
        """
        Encontra o melhor atributo para dividir baseado no ganho de informação.
        
        Testa cada atributo disponível e seleciona aquele com maior ganho.
        
        Args:
            X: matriz de features
            y: array de labels
            available_features: lista de índices de features disponíveis
            
        Returns:
            tuple: (melhor_feature_idx, melhor_ganho)
        """
        if len(available_features) == 0:
            return None, 0.0
        
        best_gain = 0.0
        best_feature = None
        
        for feature_idx in available_features:
            # Divide dados
            splits = self._split_data(X, y, feature_idx)
            
            # Verifica se todas as divisões respeitam min_samples_leaf
            valid_split = True
            y_splits = {}
            for value, (X_sub, y_sub) in splits.items():
                if len(y_sub) < self.min_samples_leaf:
                    valid_split = False
                    break
                y_splits[value] = y_sub
            
            if not valid_split:
                continue
            
            # Calcula ganho de informação
            gain = self._information_gain(y, y_splits)
            
            # Atualiza melhor feature
            if gain > best_gain:
                best_gain = gain
                best_feature = feature_idx
        
        return best_feature, best_gain
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, 
                    available_features: List[int], depth: int = 0) -> NodeID3:
        """
        Constrói a árvore ID3 recursivamente.
        
        ALGORITMO ID3:
        1. Se todos exemplos são da mesma classe → retorna folha
        2. Se não há mais atributos → retorna folha com classe majoritária
        3. Senão:
           a) Seleciona melhor atributo (maior ganho de informação)
           b) Cria nó de decisão para este atributo
           c) Para cada valor do atributo, cria subárvore recursivamente
        
        Args:
            X: matriz de features
            y: array de labels
            available_features: features ainda não usadas
            depth: profundidade atual
            
        Returns:
            NodeID3: nó raiz da (sub)árvore
        """
        n_samples = len(y)
        n_classes = len(np.unique(y))
        
        # Distribuição de classes
        class_counts = Counter(y)
        predicted_class = max(class_counts, key=class_counts.get)
        entropy = self._entropy(y)
        
        # Cria nó base
        node = NodeID3(
            entropy=entropy,
            n_samples=n_samples,
            class_distribution=dict(class_counts),
            value=predicted_class
        )
        
        # CRITÉRIOS DE PARADA
        
        # 1. Profundidade máxima atingida
        if self.max_depth is not None and depth >= self.max_depth:
            node.is_leaf = True
            return node
        
        # 2. Amostras insuficientes para dividir
        if n_samples < self.min_samples_split:
            node.is_leaf = True
            return node
        
        # 3. Nó puro (todas amostras da mesma classe)
        if n_classes == 1:
            node.is_leaf = True
            return node
        
        # 4. Não há mais atributos disponíveis
        if len(available_features) == 0:
            node.is_leaf = True
            return node
        
        # ENCONTRA MELHOR ATRIBUTO
        best_feature, best_gain = self._find_best_feature(
            X, y, available_features
        )
        
        # 5. Ganho de informação insuficiente
        if best_feature is None or best_gain < self.min_information_gain:
            node.is_leaf = True
            return node
        
        # DIVIDE DADOS
        node.feature_idx = best_feature
        splits = self._split_data(X, y, best_feature)
        
        # Remove feature usada da lista de disponíveis
        # IMPORTANTE: ID3 clássico não reutiliza atributos
        remaining_features = [f for f in available_features if f != best_feature]
        
        # CRIA SUBÁRVORES PARA CADA VALOR DO ATRIBUTO
        for value, (X_subset, y_subset) in splits.items():
            if len(y_subset) > 0:
                # Recursão
                child_node = self._build_tree(
                    X_subset, y_subset, remaining_features, depth + 1
                )
                child_node.feature_value = value
                node.children[value] = child_node
            else:
                # Se não há amostras com este valor, cria folha com classe majoritária
                leaf = NodeID3(
                    value=predicted_class,
                    n_samples=0,
                    is_leaf=True,
                    feature_value=value
                )
                node.children[value] = leaf
        
        return node
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Treina o classificador ID3.
        
        NOTA: X deve conter apenas atributos categóricos!
        Para atributos contínuos, use discretização prévia.
        
        Args:
            X: matriz de features categóricas (n_samples, n_features)
            y: array de labels (n_samples,)
            
        Returns:
            self
        """
        # Converte para numpy array
        X = np.array(X)
        y = np.array(y)
        
        # Armazena metadados
        self.n_classes_ = len(np.unique(y))
        self.n_features_ = X.shape[1]
        
        # Lista de todas as features disponíveis
        available_features = list(range(self.n_features_))
        
        # Constrói árvore
        self.root = self._build_tree(X, y, available_features)
        
        return self
    
    def _predict_sample(self, x: np.ndarray, node: NodeID3) -> int:
        """
        Prediz a classe para uma única amostra.
        
        Args:
            x: vetor de features
            node: nó atual
            
        Returns:
            int: classe predita
        """
        # Se é folha, retorna valor
        if node.is_leaf or len(node.children) == 0:
            return node.value
        
        # Obtém valor do atributo de decisão
        feature_value = x[node.feature_idx]
        
        # Verifica se existe filho para este valor
        if feature_value in node.children:
            return self._predict_sample(x, node.children[feature_value])
        else:
            # Valor não visto no treino → retorna classe majoritária do nó
            return node.value
    
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
    
    def _print_tree(self, node: NodeID3, depth: int = 0, 
                    parent_value: str = "Root"):
        """
        Imprime a estrutura da árvore de forma legível.
        
        Args:
            node: nó atual
            depth: profundidade atual
            parent_value: valor que levou a este nó
        """
        if node is None:
            return
        
        indent = "  " * depth
        prefix = f"{parent_value}: " if depth > 0 else "Root: "
        
        # Informações do nó
        info = f"{prefix}Entropy={node.entropy:.3f}, samples={node.n_samples}, "
        info += f"value={node.value}, dist={node.class_distribution}"
        
        if node.is_leaf:
            info += " [LEAF]"
        
        print(indent + info)
        
        # Se não é folha, imprime filhos
        if not node.is_leaf and node.children:
            feature_info = f"Split on: Feature {node.feature_idx}"
            print(indent + f"  {feature_info}")
            
            for value, child in sorted(node.children.items()):
                self._print_tree(child, depth + 1, f"Value={value}")
    
    def print_tree(self):
        """Imprime a árvore completa."""
        print("\n=== Estrutura da Árvore ID3 ===\n")
        self._print_tree(self.root)
    
    def get_rules(self) -> List[str]:
        """
        Extrai regras de decisão da árvore em formato legível.
        
        Returns:
            List[str]: lista de regras IF-THEN
        """
        rules = []
        
        def extract_rules(node, path, feature_names=None):
            """Extrai regras recursivamente."""
            if node.is_leaf or len(node.children) == 0:
                # Formata regra
                if len(path) > 0:
                    conditions = " AND ".join(path)
                    rule = f"IF {conditions} THEN class={node.value}"
                else:
                    rule = f"IF (root) THEN class={node.value}"
                rules.append(rule)
                return
            
            # Recursão para cada filho
            feature_name = f"X{node.feature_idx}" if feature_names is None \
                          else feature_names[node.feature_idx]
            
            for value, child in node.children.items():
                condition = f"{feature_name}={value}"
                extract_rules(child, path + [condition])
        
        extract_rules(self.root, [])
        return rules


# ============================================================================
# Exemplo de uso e comparação com CART
# ============================================================================

if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.preprocessing import KBinsDiscretizer
    
    print("=" * 80)
    print("DEMONSTRAÇÃO: Algoritmo ID3 (Simplified)")
    print("=" * 80)
    
    # ========================================================================
    # Dataset: Iris (discretizado para usar ID3)
    # ========================================================================
    print("\n### DATASET: IRIS (Discretizado) ###\n")
    
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # IMPORTANTE: ID3 trabalha apenas com atributos categóricos
    # Precisamos discretizar os atributos contínuos
    print("Discretizando atributos contínuos em 3 bins...")
    discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')
    X_discrete = discretizer.fit_transform(X).astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_discrete, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Treina ID3
    print("\nTreinando ID3...")
    id3 = ID3Classifier(
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        min_information_gain=0.01
    )
    id3.fit(X_train, y_train)
    
    # Predições
    y_pred_train = id3.predict(X_train)
    y_pred_test = id3.predict(X_test)
    
    # Resultados
    print("\n--- Resultados ID3 ---")
    print(f"Acurácia Treino: {accuracy_score(y_train, y_pred_train):.4f}")
    print(f"Acurácia Teste:  {accuracy_score(y_test, y_pred_test):.4f}")
    
    print("\nClassification Report (Teste):")
    print(classification_report(y_test, y_pred_test, target_names=iris.target_names))
    
    # Estrutura da árvore
    id3.print_tree()
    
    # Regras de decisão
    print("\n--- Regras de Decisão Extraídas ---\n")
    rules = id3.get_rules()
    for i, rule in enumerate(rules, 1):
        print(f"{i}. {rule}")
    
    # ========================================================================
    # Comparação: ID3 vs CART
    # ========================================================================
    print("\n\n### COMPARAÇÃO: ID3 vs CART ###\n")
    
    from sklearn.tree import DecisionTreeClassifier
    
    # CART (sklearn)
    cart = DecisionTreeClassifier(
        criterion='gini', 
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    cart.fit(X_train, y_train)
    y_pred_cart = cart.predict(X_test)
    
    print("Resultados Comparativos:")
    print(f"ID3 - Acurácia:  {accuracy_score(y_test, y_pred_test):.4f}")
    print(f"CART - Acurácia: {accuracy_score(y_test, y_pred_cart):.4f}")
    
    print("\n" + "=" * 80)
    print("DIFERENÇAS PRINCIPAIS:")
    print("=" * 80)
    print("""
    1. CRITÉRIO DE DIVISÃO:
       - ID3:  Information Gain (baseado em entropia)
       - CART: Gini Impurity
    
    2. TIPO DE DIVISÃO:
       - ID3:  Multi-way (múltiplos filhos por nó)
       - CART: Binária (sempre 2 filhos)
    
    3. TIPOS DE ATRIBUTOS:
       - ID3:  Apenas categóricos
       - CART: Categóricos E contínuos
    
    4. REUTILIZAÇÃO DE ATRIBUTOS:
       - ID3:  Não reutiliza atributos (clássico)
       - CART: Pode reutilizar atributos
    
    5. PODA:
       - ID3:  Não implementa poda (versão original)
       - CART: Implementa poda de complexidade
    """)
    
    # ========================================================================
    # Dataset Categórico Simples (Exemplo Didático)
    # ========================================================================
    print("\n\n### EXEMPLO DIDÁTICO: DATASET CATEGÓRICO PURO ###\n")
    
    # Dataset: Condições para jogar tênis
    X_tennis = np.array([
        [0, 0, 0, 0],  # Sol, Quente, Alta, Fraco -> Não
        [0, 0, 0, 1],  # Sol, Quente, Alta, Forte -> Não
        [1, 0, 0, 0],  # Nublado, Quente, Alta, Fraco -> Sim
        [2, 1, 0, 0],  # Chuva, Ameno, Alta, Fraco -> Sim
        [2, 2, 1, 0],  # Chuva, Frio, Normal, Fraco -> Sim
        [2, 2, 1, 1],  # Chuva, Frio, Normal, Forte -> Não
        [1, 2, 1, 1],  # Nublado, Frio, Normal, Forte -> Sim
        [0, 1, 0, 0],  # Sol, Ameno, Alta, Fraco -> Não
        [0, 2, 1, 0],  # Sol, Frio, Normal, Fraco -> Sim
        [2, 1, 1, 0],  # Chuva, Ameno, Normal, Fraco -> Sim
        [0, 1, 1, 1],  # Sol, Ameno, Normal, Forte -> Sim
        [1, 1, 0, 1],  # Nublado, Ameno, Alta, Forte -> Sim
        [1, 0, 1, 0],  # Nublado, Quente, Normal, Fraco -> Sim
        [2, 1, 0, 1],  # Chuva, Ameno, Alta, Forte -> Não
    ])
    
    y_tennis = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0])
    # 0 = Não joga, 1 = Joga
    
    feature_names = ['Tempo', 'Temperatura', 'Umidade', 'Vento']
    
    print("Treinando ID3 no dataset Tennis...")
    id3_tennis = ID3Classifier(max_depth=4)
    id3_tennis.fit(X_tennis, y_tennis)
    
    print("\nÁrvore de Decisão:")
    id3_tennis.print_tree()
    
    print("\nRegras Extraídas:")
    rules_tennis = id3_tennis.get_rules()
    for i, rule in enumerate(rules_tennis, 1):
        print(f"{i}. {rule}")
    
    # Teste
    y_pred_tennis = id3_tennis.predict(X_tennis)
    print(f"\nAcurácia (treino): {accuracy_score(y_tennis, y_pred_tennis):.4f}")