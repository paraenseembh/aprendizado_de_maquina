# ============================================================================
# ARQUIVO: titanic_analysis/analyzer.py
# ============================================================================
"""
Classe principal que orquestra toda a análise
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from .data_loader import load_titanic_data, analyze_missing_data, check_redundancy
from .preprocessing import preprocess_data, impute_missing_values, apply_balancing
from .models import train_decision_tree, train_random_forest
from .evaluation import evaluate_model, compare_models
from .reporting import generate_report


class TitanicAnalyzer:
    """
    Classe principal para análise do dataset Titanic
    
    Esta classe orquestra todo o pipeline de análise, desde o carregamento
    dos dados até a geração de relatórios.
    
    Attributes:
        df (pd.DataFrame): Dataset original
        X_train (pd.DataFrame): Features de treino
        X_test (pd.DataFrame): Features de teste
        y_train (pd.Series): Target de treino
        y_test (pd.Series): Target de teste
        models (dict): Modelos treinados
        results (pd.DataFrame): Resultados comparativos
    
    Examples:
        >>> analyzer = TitanicAnalyzer()
        >>> analyzer.load_data()
        >>> analyzer.preprocess()
        >>> analyzer.train_all_models()
        >>> analyzer.generate_report()
    """
    
    def __init__(self, test_size=0.2, random_state=42):
        """
        Inicializa o analisador
        
        Args:
            test_size (float): Proporção do conjunto de teste
            random_state (int): Seed para reprodutibilidade
        """
        self.test_size = test_size
        self.random_state = random_state
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = []
        
    def load_data(self, url=None):
        """Carrega o dataset"""
        self.df = load_titanic_data(url)
        return self
    
    def explore_data(self):
        """Realiza análise exploratória"""
        analyze_missing_data(self.df)
        check_redundancy(self.df)
        return self
    
    def preprocess(self):
        """Preprocessa os dados"""
        X, y, _ = preprocess_data(self.df)
        
        # Split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        # Imputação
        self.X_train, self.X_test = impute_missing_values(
            self.X_train, self.X_test, method='knn'
        )
        
        return self
    
    def train_all_models(self, strategies=['baseline', 'smote', 'undersample']):
        """
        Treina todos os modelos com diferentes estratégias
        
        Args:
            strategies (list): Lista de estratégias de balanceamento
        """
        for strategy in strategies:
            # Preparar dados
            if strategy == 'baseline':
                X_train_bal = self.X_train
                y_train_bal = self.y_train
            elif strategy == 'smote':
                X_train_bal, y_train_bal = apply_balancing(
                    self.X_train, self.y_train, method='smote'
                )
            elif strategy == 'undersample':
                X_train_bal, y_train_bal = apply_balancing(
                    self.X_train, self.y_train, method='undersample'
                )
            
            # Decision Tree
            dt_name = f"Decision Tree ({strategy.title()})"
            dt = train_decision_tree(X_train_bal, y_train_bal)
            self.models[dt_name] = dt
            
            results_dt = evaluate_model(
                dt, X_train_bal, self.X_test, y_train_bal, self.y_test, dt_name
            )
            self.results.append(results_dt)
            
            # Random Forest
            rf_name = f"Random Forest ({strategy.title()})"
            rf = train_random_forest(X_train_bal, y_train_bal)
            self.models[rf_name] = rf
            
            results_rf = evaluate_model(
                rf, X_train_bal, self.X_test, y_train_bal, self.y_test, rf_name
            )
            self.results.append(results_rf)
        
        return self
    
    def compare_results(self):
        """Compara resultados de todos os modelos"""
        df_comparison = compare_models(self.results)
        return df_comparison
    
    def generate_report(self, output_dir='relatorio_titanic'):
        """Gera relatório completo"""
        df_comparison = self.compare_results()
        
        return generate_report(
            df_comparison=df_comparison,
            model_objects=self.models,
            X_train=self.X_train,
            X_test=self.X_test,
            y_train=self.y_train,
            y_test=self.y_test,
            output_dir=output_dir
        )
    
    def run_full_analysis(self, output_dir='relatorio_titanic'):
        """
        Executa análise completa de ponta a ponta
        
        Args:
            output_dir (str): Diretório para salvar relatórios
        
        Returns:
            pd.DataFrame: Comparação de resultados
        """
        print("Iniciando análise completa...")
        
        self.load_data()
        self.explore_data()
        self.preprocess()
        self.train_all_models()
        
        print("\nGerando relatório...")
        self.generate_report(output_dir)
        
        return self.compare_results()