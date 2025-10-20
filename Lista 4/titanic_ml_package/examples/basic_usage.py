# ============================================================================
# ARQUIVO: examples/basic_usage.py
# ============================================================================
"""
Exemplo básico de uso do módulo
"""

from titanic_analysis import TitanicAnalyzer

# Método 1: Análise completa automática
analyzer = TitanicAnalyzer()
results = analyzer.run_full_analysis(output_dir='meu_relatorio')

print("\nMelhores resultados:")
print(results.nlargest(3, 'test_acc'))


# Método 2: Controle passo a passo
analyzer2 = TitanicAnalyzer(test_size=0.25, random_state=123)

analyzer2.load_data()
analyzer2.explore_data()
analyzer2.preprocess()
analyzer2.train_all_models(strategies=['baseline', 'smote'])
results2 = analyzer2.compare_results()

print(results2)


# Método 3: Usando funções individuais
from titanic_analysis import (
    load_titanic_data,
    preprocess_data,
    train_random_forest,
    evaluate_model
)

df = load_titanic_data()
X, y, _ = preprocess_data(df)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Imputar dados ausentes primeiro
from titanic_analysis import impute_missing_values
X_train, X_test = impute_missing_values(X_train, X_test)

model = train_random_forest(X_train, y_train)
results = evaluate_model(model, X_train, X_test, y_train, y_test, "Meu RF")