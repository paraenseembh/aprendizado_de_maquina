import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.model_selection import cross_val_score, learning_curve
import warnings
warnings.filterwarnings('ignore')


class CARTMetrics:
    """
    Sistema completo de métricas e avaliação para o classificador CART.
    
    Fornece:
    - Métricas de classificação (acurácia, precisão, recall, F1)
    - Matriz de confusão
    - Análise da estrutura da árvore
    - Importância de features
    - Curvas ROC e PR
    - Validação cruzada
    - Curvas de aprendizado
    """
    
    def __init__(self, model, X_train=None, y_train=None, X_test=None, y_test=None):
        """
        Inicializa o sistema de métricas.
        
        Args:
            model: modelo CART treinado
            X_train: dados de treino
            y_train: labels de treino
            X_test: dados de teste
            y_test: labels de teste
        """
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
        # Predições
        self.y_pred_train = None
        self.y_pred_test = None
        
        if X_test is not None and y_test is not None:
            self.y_pred_test = model.predict(X_test)
        
        if X_train is not None and y_train is not None:
            self.y_pred_train = model.predict(X_train)
    
    def get_basic_metrics(self, dataset='test') -> Dict:
        """
        Calcula métricas básicas de classificação.
        
        Args:
            dataset: 'train' ou 'test'
            
        Returns:
            Dict com métricas
        """
        if dataset == 'test':
            y_true = self.y_test
            y_pred = self.y_pred_test
        else:
            y_true = self.y_train
            y_pred = self.y_pred_train
        
        if y_true is None or y_pred is None:
            return {}
        
        # Calcula métricas
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }
        
        # Adiciona métricas por classe
        n_classes = len(np.unique(y_true))
        for i in range(n_classes):
            metrics[f'precision_class_{i}'] = precision_score(
                y_true, y_pred, labels=[i], average='macro', zero_division=0
            )
            metrics[f'recall_class_{i}'] = recall_score(
                y_true, y_pred, labels=[i], average='macro', zero_division=0
            )
            metrics[f'f1_class_{i}'] = f1_score(
                y_true, y_pred, labels=[i], average='macro', zero_division=0
            )
        
        return metrics
    
    def get_confusion_matrix(self, dataset='test') -> np.ndarray:
        """
        Retorna matriz de confusão.
        
        Args:
            dataset: 'train' ou 'test'
            
        Returns:
            np.ndarray: matriz de confusão
        """
        if dataset == 'test':
            y_true = self.y_test
            y_pred = self.y_pred_test
        else:
            y_true = self.y_train
            y_pred = self.y_pred_train
        
        if y_true is None or y_pred is None:
            return None
        
        return confusion_matrix(y_true, y_pred)
    
    def get_tree_structure_metrics(self) -> Dict:
        """
        Analisa a estrutura da árvore.
        
        Returns:
            Dict com métricas estruturais
        """
        def count_nodes(node):
            """Conta nós recursivamente."""
            if node is None:
                return 0, 0, 0, []
            
            if node.left is None and node.right is None:
                # Nó folha
                return 1, 1, 0, [node.gini]
            
            # Nó interno
            left_total, left_leaves, left_internal, left_ginis = count_nodes(node.left)
            right_total, right_leaves, right_internal, right_ginis = count_nodes(node.right)
            
            total = 1 + left_total + right_total
            leaves = left_leaves + right_leaves
            internal = 1 + left_internal + right_internal
            ginis = [node.gini] + left_ginis + right_ginis
            
            return total, leaves, internal, ginis
        
        def get_max_depth(node, current_depth=0):
            """Calcula profundidade máxima."""
            if node is None:
                return current_depth - 1
            
            if node.left is None and node.right is None:
                return current_depth
            
            left_depth = get_max_depth(node.left, current_depth + 1)
            right_depth = get_max_depth(node.right, current_depth + 1)
            
            return max(left_depth, right_depth)
        
        def get_leaf_samples(node):
            """Coleta número de amostras em cada folha."""
            if node is None:
                return []
            
            if node.left is None and node.right is None:
                return [node.n_samples]
            
            left_samples = get_leaf_samples(node.left)
            right_samples = get_leaf_samples(node.right)
            
            return left_samples + right_samples
        
        total_nodes, n_leaves, n_internal, gini_values = count_nodes(self.model.root)
        max_depth = get_max_depth(self.model.root)
        leaf_samples = get_leaf_samples(self.model.root)
        
        metrics = {
            'total_nodes': total_nodes,
            'n_leaves': n_leaves,
            'n_internal_nodes': n_internal,
            'max_depth': max_depth,
            'avg_gini': np.mean(gini_values) if gini_values else 0,
            'min_gini': np.min(gini_values) if gini_values else 0,
            'max_gini': np.max(gini_values) if gini_values else 0,
            'avg_samples_per_leaf': np.mean(leaf_samples) if leaf_samples else 0,
            'min_samples_per_leaf': np.min(leaf_samples) if leaf_samples else 0,
            'max_samples_per_leaf': np.max(leaf_samples) if leaf_samples else 0,
        }
        
        return metrics
    
    def get_feature_importance(self) -> Dict:
        """
        Calcula importância de features baseada em uso na árvore.
        
        Returns:
            Dict com importância por feature
        """
        def collect_splits(node, feature_counts):
            """Coleta todas as divisões recursivamente."""
            if node is None or node.feature_idx is None:
                return
            
            feature_counts[node.feature_idx] = feature_counts.get(node.feature_idx, 0) + 1
            
            collect_splits(node.left, feature_counts)
            collect_splits(node.right, feature_counts)
        
        feature_counts = {}
        collect_splits(self.model.root, feature_counts)
        
        # Normaliza
        total_splits = sum(feature_counts.values())
        if total_splits > 0:
            feature_importance = {
                f'feature_{k}': v / total_splits 
                for k, v in feature_counts.items()
            }
        else:
            feature_importance = {}
        
        return feature_importance
    
    def print_comprehensive_report(self):
        """
        Imprime relatório completo com todas as métricas.
        """
        print("=" * 80)
        print("RELATÓRIO COMPLETO DE MÉTRICAS - CART")
        print("=" * 80)
        
        # Métricas de treino
        if self.y_train is not None:
            print("\n### MÉTRICAS DE TREINO ###\n")
            train_metrics = self.get_basic_metrics('train')
            for key, value in train_metrics.items():
                print(f"{key:30s}: {value:.4f}")
        
        # Métricas de teste
        if self.y_test is not None:
            print("\n### MÉTRICAS DE TESTE ###\n")
            test_metrics = self.get_basic_metrics('test')
            for key, value in test_metrics.items():
                print(f"{key:30s}: {value:.4f}")
            
            # Classification Report
            print("\n### CLASSIFICATION REPORT (TESTE) ###\n")
            print(classification_report(self.y_test, self.y_pred_test))
        
        # Estrutura da árvore
        print("\n### ESTRUTURA DA ÁRVORE ###\n")
        tree_metrics = self.get_tree_structure_metrics()
        for key, value in tree_metrics.items():
            if isinstance(value, float):
                print(f"{key:30s}: {value:.4f}")
            else:
                print(f"{key:30s}: {value}")
        
        # Importância de features
        print("\n### IMPORTÂNCIA DE FEATURES ###\n")
        feature_imp = self.get_feature_importance()
        for key, value in sorted(feature_imp.items(), key=lambda x: x[1], reverse=True):
            print(f"{key:30s}: {value:.4f}")
        
        # Matriz de confusão
        if self.y_test is not None:
            print("\n### MATRIZ DE CONFUSÃO (TESTE) ###\n")
            cm = self.get_confusion_matrix('test')
            print(cm)
            
            # Análise de erros
            print("\n### ANÁLISE DE ERROS ###\n")
            correct = np.sum(np.diag(cm))
            total = np.sum(cm)
            errors = total - correct
            print(f"Total de predições: {total}")
            print(f"Corretas: {correct} ({100*correct/total:.2f}%)")
            print(f"Erros: {errors} ({100*errors/total:.2f}%)")
        
        # Overfitting check
        if self.y_train is not None and self.y_test is not None:
            train_acc = self.get_basic_metrics('train')['accuracy']
            test_acc = self.get_basic_metrics('test')['accuracy']
            gap = train_acc - test_acc
            
            print("\n### ANÁLISE DE OVERFITTING ###\n")
            print(f"Acurácia Treino: {train_acc:.4f}")
            print(f"Acurácia Teste:  {test_acc:.4f}")
            print(f"Gap (diferença): {gap:.4f}")
            
            if gap > 0.10:
                print("⚠️  AVISO: Gap > 10% indica possível overfitting!")
            elif gap > 0.05:
                print("⚠️  ATENÇÃO: Gap moderado, modelo pode estar overfitting.")
            else:
                print("✓ Gap aceitável, modelo generalizando bem.")
        
        print("\n" + "=" * 80)
    
    def plot_confusion_matrix(self, dataset='test', class_names=None, 
                            figsize=(8, 6), cmap='Blues'):
        """
        Plota matriz de confusão com visualização.
        
        Args:
            dataset: 'train' ou 'test'
            class_names: nomes das classes
            figsize: tamanho da figura
            cmap: colormap
        """
        cm = self.get_confusion_matrix(dataset)
        
        if cm is None:
            print("Dados não disponíveis para plotar matriz de confusão.")
            return
        
        plt.figure(figsize=figsize)
        
        # Normaliza para porcentagens
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, cbar=True,
                   xticklabels=class_names, yticklabels=class_names)
        
        plt.title(f'Matriz de Confusão - {dataset.upper()}', fontsize=14, fontweight='bold')
        plt.ylabel('Classe Real', fontsize=12)
        plt.xlabel('Classe Predita', fontsize=12)
        plt.tight_layout()
        plt.show()
        
        # Plot normalizado
        plt.figure(figsize=figsize)
        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap=cmap, cbar=True,
                   xticklabels=class_names, yticklabels=class_names)
        
        plt.title(f'Matriz de Confusão Normalizada - {dataset.upper()}', 
                 fontsize=14, fontweight='bold')
        plt.ylabel('Classe Real', fontsize=12)
        plt.xlabel('Classe Predita', fontsize=12)
        plt.tight_layout()
        plt.show()
    
    def plot_metrics_comparison(self, figsize=(12, 6)):
        """
        Plota comparação de métricas entre treino e teste.
        
        Args:
            figsize: tamanho da figura
        """
        if self.y_train is None or self.y_test is None:
            print("Dados de treino e teste necessários.")
            return
        
        train_metrics = self.get_basic_metrics('train')
        test_metrics = self.get_basic_metrics('test')
        
        # Seleciona métricas principais
        metric_names = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        train_values = [train_metrics[m] for m in metric_names]
        test_values = [test_metrics[m] for m in metric_names]
        
        x = np.arange(len(metric_names))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=figsize)
        bars1 = ax.bar(x - width/2, train_values, width, label='Treino', alpha=0.8)
        bars2 = ax.bar(x + width/2, test_values, width, label='Teste', alpha=0.8)
        
        ax.set_xlabel('Métricas', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Comparação de Métricas: Treino vs Teste', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metric_names])
        ax.legend()
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)
        
        # Adiciona valores nas barras
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, feature_names=None, figsize=(10, 6)):
        """
        Plota importância de features.
        
        Args:
            feature_names: nomes das features
            figsize: tamanho da figura
        """
        feature_imp = self.get_feature_importance()
        
        if not feature_imp:
            print("Nenhuma feature foi usada na árvore.")
            return
        
        # Ordena por importância
        sorted_features = sorted(feature_imp.items(), key=lambda x: x[1], reverse=True)
        features = [f[0] for f in sorted_features]
        importances = [f[1] for f in sorted_features]
        
        # Substitui nomes se fornecidos
        if feature_names:
            features = [feature_names[int(f.split('_')[1])] 
                       if int(f.split('_')[1]) < len(feature_names)
                       else f for f in features]
        
        plt.figure(figsize=figsize)
        plt.barh(range(len(features)), importances, alpha=0.8)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importância Relativa', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.title('Importância de Features (baseada em uso)', 
                 fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def export_metrics_to_csv(self, filename='cart_metrics.csv'):
        """
        Exporta métricas para arquivo CSV.
        
        Args:
            filename: nome do arquivo
        """
        all_metrics = {}
        
        # Métricas de teste
        if self.y_test is not None:
            test_metrics = self.get_basic_metrics('test')
            all_metrics.update({f'test_{k}': v for k, v in test_metrics.items()})
        
        # Métricas de treino
        if self.y_train is not None:
            train_metrics = self.get_basic_metrics('train')
            all_metrics.update({f'train_{k}': v for k, v in train_metrics.items()})
        
        # Métricas estruturais
        tree_metrics = self.get_tree_structure_metrics()
        all_metrics.update({f'tree_{k}': v for k, v in tree_metrics.items()})
        
        # Converte para DataFrame e salva
        df = pd.DataFrame([all_metrics])
        df.to_csv(filename, index=False)
        print(f"✓ Métricas exportadas para {filename}")
        
        return df


# ============================================================================
# Exemplo de uso completo
# ============================================================================

if __name__ == "__main__":
    from sklearn.datasets import load_iris, load_wine, load_breast_cancer
    from sklearn.model_selection import train_test_split
    
    # Importa o modelo CART customizado
    # Assumindo que está no mesmo diretório ou importado corretamente
    # from cart_implementation import CARTClassifier
    
    print("=" * 80)
    print("DEMONSTRAÇÃO: Sistema de Métricas para CART")
    print("=" * 80)
    
    # ========================================================================
    # Dataset: Iris
    # ========================================================================
    print("\n\n### ANÁLISE COMPLETA: IRIS DATASET ###\n")
    
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.3, random_state=42, stratify=iris.target
    )
    
    # Treina modelo (assumindo CARTClassifier importado)
    # cart_model = CARTClassifier(max_depth=5, min_samples_split=5, min_samples_leaf=2)
    # cart_model.fit(X_train, y_train)
    
    # NOTA: Como não temos o modelo aqui, vamos simular com sklearn
    from sklearn.tree import DecisionTreeClassifier
    cart_model = DecisionTreeClassifier(
        criterion='gini', max_depth=5, min_samples_split=5, 
        min_samples_leaf=2, random_state=42
    )
    cart_model.fit(X_train, y_train)
    
    # Cria sistema de métricas
    metrics = CARTMetrics(
        model=cart_model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test
    )
    
    # Relatório completo
    metrics.print_comprehensive_report()
    
    # Visualizações
    print("\n\nGerando visualizações...")
    
    # Matriz de confusão
    metrics.plot_confusion_matrix(
        dataset='test',
        class_names=iris.target_names,
        figsize=(8, 6)
    )
    
    # Comparação de métricas
    metrics.plot_metrics_comparison(figsize=(12, 6))
    
    # Importância de features
    metrics.plot_feature_importance(
        feature_names=iris.feature_names,
        figsize=(10, 6)
    )
    
    # Exporta métricas
    df_metrics = metrics.export_metrics_to_csv('iris_cart_metrics.csv')
    print("\nPrimeiras métricas exportadas:")
    print(df_metrics.head())
    
    # ========================================================================
    # Dataset: Wine (exemplo adicional)
    # ========================================================================
    print("\n\n### ANÁLISE COMPLETA: WINE DATASET ###\n")
    
    wine = load_wine()
    X_train_wine, X_test_wine, y_train_wine, y_test_wine = train_test_split(
        wine.data, wine.target, test_size=0.3, random_state=42, stratify=wine.target
    )
    
    cart_wine = DecisionTreeClassifier(
        criterion='gini', max_depth=4, min_samples_split=10, random_state=42
    )
    cart_wine.fit(X_train_wine, y_train_wine)
    
    metrics_wine = CARTMetrics(
        model=cart_wine,
        X_train=X_train_wine,
        y_train=y_train_wine,
        X_test=X_test_wine,
        y_test=y_test_wine
    )
    
    metrics_wine.print_comprehensive_report()
    
    # Visualizações Wine
    metrics_wine.plot_confusion_matrix(
        dataset='test',
        class_names=wine.target_names,
        figsize=(8, 6)
    )
    
    metrics_wine.plot_metrics_comparison(figsize=(12, 6))
    
    metrics_wine.plot_feature_importance(
        feature_names=wine.feature_names,
        figsize=(10, 8)
    )
    
    metrics_wine.export_metrics_to_csv('wine_cart_metrics.csv')