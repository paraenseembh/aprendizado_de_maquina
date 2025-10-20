# ============================================================================
# ARQUIVO: titanic_analysis/__init__.py
# ============================================================================
"""
Análise de Machine Learning para o Dataset Titanic

Este pacote fornece ferramentas para análise comparativa entre
Decision Tree e Random Forest com diferentes estratégias de balanceamento.

Exemplo básico de uso:
    >>> from titanic_analysis import TitanicAnalyzer
    >>> analyzer = TitanicAnalyzer()
    >>> results = analyzer.run_full_analysis()
    >>> analyzer.generate_report(results)
"""

__version__ = '1.0.0'
__author__ = 'Lucas Rafael P. do Nascimento'
__email__ = 'slucasrafael.pessoadn@gmail.com'

# Importações principais para facilitar o acesso
from .data_loader import load_titanic_data, analyze_missing_data, check_redundancy
from .preprocessing import preprocess_data, impute_missing_values, apply_balancing
from .models import train_decision_tree, train_random_forest
from .evaluation import evaluate_model, compare_models
from .reporting import generate_report

# Classe principal que orquestra tudo
from .analyzer import TitanicAnalyzer

__all__ = [
    'TitanicAnalyzer',
    'load_titanic_data',
    'analyze_missing_data',
    'check_redundancy',
    'preprocess_data',
    'impute_missing_values',
    'apply_balancing',
    'train_decision_tree',
    'train_random_forest',
    'evaluate_model',
    'compare_models',
    'generate_report',
]
