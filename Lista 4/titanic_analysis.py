import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, roc_curve)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# FASE 1: CARREGAMENTO E EXPLORAÇÃO INICIAL
# ============================================================================

def load_titanic_data():
    """Carrega o dataset Titanic"""
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    print("Dataset carregado com sucesso!")
    print(f"Shape: {df.shape}")
    print(f"\nPrimeiras linhas:\n{df.head()}")
    print(f"\nInformações do dataset:\n{df.info()}")
    print(f"\nEstatísticas descritivas:\n{df.describe()}")
    return df

# ============================================================================
# FASE 2: PRÉ-PROCESSAMENTO
# ============================================================================

def analyze_missing_data(df):
    """Analisa dados ausentes"""
    missing = df.isnull().sum()
    missing_pct = 100 * missing / len(df)
    missing_table = pd.DataFrame({
        'Missing Values': missing,
        'Percentage': missing_pct
    })
    print("\n=== ANÁLISE DE DADOS AUSENTES ===")
    print(missing_table[missing_table['Missing Values'] > 0].sort_values('Percentage', ascending=False))
    return missing_table

def check_redundancy(df):
    """Verifica redundância através de correlações"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    print("\n=== ANÁLISE DE REDUNDÂNCIA (Correlações > 0.8) ===")
    high_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.8:
                high_corr.append((corr_matrix.columns[i], 
                                corr_matrix.columns[j], 
                                corr_matrix.iloc[i, j]))
                print(f"{corr_matrix.columns[i]} <-> {corr_matrix.columns[j]}: {corr_matrix.iloc[i, j]:.3f}")
    
    if not high_corr:
        print("Nenhuma correlação alta detectada (> 0.8)")
    
    return corr_matrix

def preprocess_data(df):
    """Pipeline completo de pré-processamento"""
    df_processed = df.copy()
    
    print("\n=== INICIANDO PRÉ-PROCESSAMENTO ===")
    
    # 1. Eliminação de features irrelevantes
    cols_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    df_processed = df_processed.drop(columns=cols_to_drop)
    print(f"✓ Colunas removidas: {cols_to_drop}")
    
    # 2. Tratamento de variáveis categóricas
    le = LabelEncoder()
    df_processed['Sex'] = le.fit_transform(df_processed['Sex'])
    df_processed['Embarked'] = df_processed['Embarked'].fillna(df_processed['Embarked'].mode()[0])
    df_processed['Embarked'] = le.fit_transform(df_processed['Embarked'])
    print("✓ Variáveis categóricas codificadas")
    
    # 3. Verificação de inconsistências
    print(f"✓ Valores negativos em Age: {(df_processed['Age'] < 0).sum()}")
    print(f"✓ Valores negativos em Fare: {(df_processed['Fare'] < 0).sum()}")
    
    # 4. Separação de features e target
    X = df_processed.drop('Survived', axis=1)
    y = df_processed['Survived']
    
    return X, y, df_processed

def impute_missing_values(X_train, X_test, method='knn'):
    """Imputa valores ausentes usando KNN"""
    print(f"\n=== IMPUTAÇÃO DE DADOS (Método: {method.upper()}) ===")
    
    if method == 'knn':
        imputer = KNNImputer(n_neighbors=5)
    else:
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        from sklearn.ensemble import RandomForestRegressor
        imputer = IterativeImputer(
            estimator=RandomForestRegressor(n_estimators=10, random_state=42),
            random_state=42
        )
    
    X_train_imputed = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    
    X_test_imputed = pd.DataFrame(
        imputer.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    print(f"✓ Valores ausentes após imputação (Train): {X_train_imputed.isnull().sum().sum()}")
    print(f"✓ Valores ausentes após imputação (Test): {X_test_imputed.isnull().sum().sum()}")
    
    return X_train_imputed, X_test_imputed

# ============================================================================
# FASE 3: ANÁLISE DE DESBALANCEAMENTO
# ============================================================================

def analyze_class_balance(y):
    """Analisa o balanceamento das classes"""
    print("\n=== ANÁLISE DE BALANCEAMENTO DE CLASSES ===")
    class_counts = y.value_counts()
    class_pct = 100 * class_counts / len(y)
    
    print(f"Classe 0 (Não Sobreviveu): {class_counts[0]} ({class_pct[0]:.1f}%)")
    print(f"Classe 1 (Sobreviveu): {class_counts[1]} ({class_pct[1]:.1f}%)")
    print(f"Razão de desbalanceamento: {class_counts[0]/class_counts[1]:.2f}:1")
    
    return class_counts

def apply_balancing(X_train, y_train, method='smote'):
    """Aplica técnicas de balanceamento"""
    print(f"\n=== APLICANDO BALANCEAMENTO (Método: {method.upper()}) ===")
    print(f"Distribuição original: {y_train.value_counts().to_dict()}")
    
    if method == 'smote':
        sampler = SMOTE(random_state=42)
    elif method == 'undersample':
        sampler = RandomUnderSampler(random_state=42)
    elif method == 'combined':
        over = SMOTE(sampling_strategy=0.8, random_state=42)
        under = RandomUnderSampler(sampling_strategy=0.9, random_state=42)
        sampler = ImbPipeline([('over', over), ('under', under)])
    else:
        return X_train, y_train
    
    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    print(f"Distribuição após balanceamento: {pd.Series(y_resampled).value_counts().to_dict()}")
    
    return X_resampled, y_resampled

# ============================================================================
# FASE 4: MODELAGEM
# ============================================================================

def train_decision_tree(X_train, y_train, params=None):
    """Treina uma Árvore de Decisão"""
    if params is None:
        params = {
            'max_depth': 5,
            'min_samples_split': 20,
            'min_samples_leaf': 10,
            'random_state': 42
        }
    
    model = DecisionTreeClassifier(**params)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train, params=None):
    """Treina um Random Forest"""
    if params is None:
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 10,
            'min_samples_leaf': 4,
            'random_state': 42,
            'n_jobs': -1
        }
    
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return model

# ============================================================================
# FASE 5: AVALIAÇÃO
# ============================================================================

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Avalia o modelo de forma abrangente"""
    print(f"\n{'='*60}")
    print(f"AVALIAÇÃO: {model_name}")
    print(f"{'='*60}")
    
    # Predições
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Métricas gerais
    print("\n--- MÉTRICAS GERAIS ---")
    print(f"Acurácia (Train): {accuracy_score(y_train, y_train_pred):.4f}")
    print(f"Acurácia (Test): {accuracy_score(y_test, y_test_pred):.4f}")
    print(f"ROC-AUC (Test): {roc_auc_score(y_test, y_test_proba):.4f}")
    
    # Métricas por classe
    print("\n--- MÉTRICAS POR CLASSE (Test Set) ---")
    print(classification_report(y_test, y_test_pred, 
                               target_names=['Não Sobreviveu', 'Sobreviveu']))
    
    # Matriz de confusão
    print("\n--- MATRIZ DE CONFUSÃO (Test Set) ---")
    cm = confusion_matrix(y_test, y_test_pred)
    print(cm)
    print(f"\nVerdadeiros Negativos: {cm[0,0]}")
    print(f"Falsos Positivos: {cm[0,1]}")
    print(f"Falsos Negativos: {cm[1,0]}")
    print(f"Verdadeiros Positivos: {cm[1,1]}")
    
    # Cross-validation
    print("\n--- CROSS-VALIDATION (5-fold) ---")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"Acurácia CV: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    results = {
        'model_name': model_name,
        'train_acc': accuracy_score(y_train, y_train_pred),
        'test_acc': accuracy_score(y_test, y_test_pred),
        'test_precision_0': precision_score(y_test, y_test_pred, pos_label=0),
        'test_recall_0': recall_score(y_test, y_test_pred, pos_label=0),
        'test_f1_0': f1_score(y_test, y_test_pred, pos_label=0),
        'test_precision_1': precision_score(y_test, y_test_pred, pos_label=1),
        'test_recall_1': recall_score(y_test, y_test_pred, pos_label=1),
        'test_f1_1': f1_score(y_test, y_test_pred, pos_label=1),
        'roc_auc': roc_auc_score(y_test, y_test_proba),
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }
    
    return results

def compare_models(results_list):
    """Compara resultados de múltiplos modelos"""
    print(f"\n{'='*60}")
    print("COMPARAÇÃO DE MODELOS")
    print(f"{'='*60}\n")
    
    df_results = pd.DataFrame(results_list)
    df_results = df_results.round(4)
    
    print(df_results.to_string(index=False))
    
    # Destaca o melhor modelo
    print("\n--- MELHORES RESULTADOS ---")
    print(f"Melhor Acurácia (Test): {df_results.loc[df_results['test_acc'].idxmax(), 'model_name']}")
    print(f"Melhor ROC-AUC: {df_results.loc[df_results['roc_auc'].idxmax(), 'model_name']}")
    print(f"Melhor F1-Score (Sobreviveu): {df_results.loc[df_results['test_f1_1'].idxmax(), 'model_name']}")
    
    return df_results

# ============================================================================
# PIPELINE PRINCIPAL
# ============================================================================

def main():
    """Pipeline principal de execução"""
    
    # 1. Carregar dados
    df = load_titanic_data()
    
    # 2. Análises exploratórias
    analyze_missing_data(df)
    check_redundancy(df)
    
    # 3. Pré-processamento
    X, y, df_processed = preprocess_data(df)
    
    # 4. Split treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 5. Imputação
    X_train_imp, X_test_imp = impute_missing_values(X_train, X_test, method='knn')
    
    # 6. Análise de balanceamento
    analyze_class_balance(y_train)
    
    # 7. Lista para armazenar resultados
    all_results = []
    
    # ==================================================================
    # EXPERIMENTO 1: Modelos sem balanceamento (baseline)
    # ==================================================================
    print("\n" + "="*60)
    print("EXPERIMENTO 1: MODELOS BASELINE (Sem Balanceamento)")
    print("="*60)
    
    # Decision Tree baseline
    dt_baseline = train_decision_tree(X_train_imp, y_train)
    results_dt_baseline = evaluate_model(
        dt_baseline, X_train_imp, X_test_imp, y_train, y_test,
        "Decision Tree (Baseline)"
    )
    all_results.append(results_dt_baseline)
    
    # Random Forest baseline
    rf_baseline = train_random_forest(X_train_imp, y_train)
    results_rf_baseline = evaluate_model(
        rf_baseline, X_train_imp, X_test_imp, y_train, y_test,
        "Random Forest (Baseline)"
    )
    all_results.append(results_rf_baseline)
    
    # ==================================================================
    # EXPERIMENTO 2: Modelos com SMOTE
    # ==================================================================
    print("\n" + "="*60)
    print("EXPERIMENTO 2: MODELOS COM SMOTE")
    print("="*60)
    
    X_train_smote, y_train_smote = apply_balancing(X_train_imp, y_train, method='smote')
    
    # Decision Tree com SMOTE
    dt_smote = train_decision_tree(X_train_smote, y_train_smote)
    results_dt_smote = evaluate_model(
        dt_smote, X_train_smote, X_test_imp, y_train_smote, y_test,
        "Decision Tree (SMOTE)"
    )
    all_results.append(results_dt_smote)
    
    # Random Forest com SMOTE
    rf_smote = train_random_forest(X_train_smote, y_train_smote)
    results_rf_smote = evaluate_model(
        rf_smote, X_train_smote, X_test_imp, y_train_smote, y_test,
        "Random Forest (SMOTE)"
    )
    all_results.append(results_rf_smote)
    
    # ==================================================================
    # EXPERIMENTO 3: Modelos com Undersampling
    # ==================================================================
    print("\n" + "="*60)
    print("EXPERIMENTO 3: MODELOS COM UNDERSAMPLING")
    print("="*60)
    
    X_train_under, y_train_under = apply_balancing(X_train_imp, y_train, method='undersample')
    
    # Decision Tree com Undersampling
    dt_under = train_decision_tree(X_train_under, y_train_under)
    results_dt_under = evaluate_model(
        dt_under, X_train_under, X_test_imp, y_train_under, y_test,
        "Decision Tree (Undersample)"
    )
    all_results.append(results_dt_under)
    
    # Random Forest com Undersampling
    rf_under = train_random_forest(X_train_under, y_train_under)
    results_rf_under = evaluate_model(
        rf_under, X_train_under, X_test_imp, y_train_under, y_test,
        "Random Forest (Undersample)"
    )
    all_results.append(results_rf_under)
    
    # ==================================================================
    # COMPARAÇÃO FINAL
    # ==================================================================
    df_comparison = compare_models(all_results)
    
    return df_comparison

# ============================================================================
# FASE 6: GERAÇÃO DE RELATÓRIO
# ============================================================================

def generate_report(df_comparison, model_objects, X_train, X_test, y_train, y_test, output_dir='relatorio_titanic'):
    """
    Gera um relatório completo com todos os resultados salvando gráficos como arquivos PNG
    
    Parameters:
    -----------
    df_comparison : DataFrame com comparação dos modelos
    model_objects : dict com os modelos treinados
    X_train, X_test, y_train, y_test : dados de treino e teste
    output_dir : diretório para salvar os arquivos
    """
    from datetime import datetime
    import os
    
    print(f"\n{'='*60}")
    print("GERANDO RELATÓRIO COMPLETO")
    print(f"{'='*60}\n")
    
    # Criar diretório se não existir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✓ Diretório criado: {output_dir}/")
    
    # Dicionário para armazenar caminhos dos arquivos
    figure_files = {}
    
    # 1. Gráfico de comparação de acurácia
    fig, ax = plt.subplots(figsize=(12, 6))
    x_pos = np.arange(len(df_comparison))
    ax.bar(x_pos, df_comparison['test_acc'], alpha=0.7, color='steelblue', label='Test Accuracy')
    ax.bar(x_pos, df_comparison['train_acc'], alpha=0.5, color='lightcoral', label='Train Accuracy')
    ax.set_xlabel('Modelo', fontsize=12)
    ax.set_ylabel('Acurácia', fontsize=12)
    ax.set_title('Comparação de Acurácia entre Modelos', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df_comparison['model_name'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    filepath = os.path.join(output_dir, '1_accuracy_comparison.png')
    plt.savefig(filepath, dpi=100, bbox_inches='tight')
    figure_files['accuracy_comparison'] = filepath
    print(f"✓ Salvo: {filepath}")
    plt.close()
    
    # 2. Gráfico de F1-Score por classe
    fig, ax = plt.subplots(figsize=(12, 6))
    x_pos = np.arange(len(df_comparison))
    width = 0.35
    ax.bar(x_pos - width/2, df_comparison['test_f1_0'], width, label='Não Sobreviveu (0)', color='salmon', alpha=0.8)
    ax.bar(x_pos + width/2, df_comparison['test_f1_1'], width, label='Sobreviveu (1)', color='lightgreen', alpha=0.8)
    ax.set_xlabel('Modelo', fontsize=12)
    ax.set_ylabel('F1-Score', fontsize=12)
    ax.set_title('F1-Score por Classe', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df_comparison['model_name'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    filepath = os.path.join(output_dir, '2_f1_comparison.png')
    plt.savefig(filepath, dpi=100, bbox_inches='tight')
    figure_files['f1_comparison'] = filepath
    print(f"✓ Salvo: {filepath}")
    plt.close()
    
    # 3. Gráfico de Precision e Recall para classe positiva
    fig, ax = plt.subplots(figsize=(12, 6))
    x_pos = np.arange(len(df_comparison))
    width = 0.35
    ax.bar(x_pos - width/2, df_comparison['test_precision_1'], width, label='Precision', color='dodgerblue', alpha=0.8)
    ax.bar(x_pos + width/2, df_comparison['test_recall_1'], width, label='Recall', color='orange', alpha=0.8)
    ax.set_xlabel('Modelo', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Precision vs Recall - Classe "Sobreviveu"', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df_comparison['model_name'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    filepath = os.path.join(output_dir, '3_precision_recall.png')
    plt.savefig(filepath, dpi=100, bbox_inches='tight')
    figure_files['precision_recall'] = filepath
    print(f"✓ Salvo: {filepath}")
    plt.close()
    
    # 4. ROC-AUC Comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(model_objects)))
    
    for idx, (model_name, model) in enumerate(model_objects.items()):
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})', 
                linewidth=2, color=colors[idx])
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
    ax.set_xlabel('Taxa de Falsos Positivos', fontsize=12)
    ax.set_ylabel('Taxa de Verdadeiros Positivos', fontsize=12)
    ax.set_title('Curvas ROC - Comparação de Modelos', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    filepath = os.path.join(output_dir, '4_roc_curves.png')
    plt.savefig(filepath, dpi=100, bbox_inches='tight')
    figure_files['roc_curves'] = filepath
    print(f"✓ Salvo: {filepath}")
    plt.close()
    
    # 5. Feature Importance (apenas para Random Forest)
    rf_models = {k: v for k, v in model_objects.items() if 'Random Forest' in k}
    if rf_models:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for idx, (model_name, model) in enumerate(rf_models.items()):
            if idx < 4:
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]
                feature_names = X_train.columns
                
                axes[idx].barh(range(len(importances)), importances[indices], color='teal', alpha=0.7)
                axes[idx].set_yticks(range(len(importances)))
                axes[idx].set_yticklabels([feature_names[i] for i in indices])
                axes[idx].set_xlabel('Importância', fontsize=10)
                axes[idx].set_title(model_name, fontsize=11, fontweight='bold')
                axes[idx].grid(axis='x', alpha=0.3)
        
        # Esconder eixos não utilizados
        for idx in range(len(rf_models), 4):
            axes[idx].axis('off')
        
        plt.tight_layout()
        filepath = os.path.join(output_dir, '5_feature_importance.png')
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        figure_files['feature_importance'] = filepath
        print(f"✓ Salvo: {filepath}")
        plt.close()
    
    # 6. Matriz de Confusão do melhor modelo
    best_model_name = df_comparison.loc[df_comparison['test_acc'].idxmax(), 'model_name']
    best_model = model_objects[best_model_name]
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, 
                xticklabels=['Não Sobreviveu', 'Sobreviveu'],
                yticklabels=['Não Sobreviveu', 'Sobreviveu'],
                ax=ax, annot_kws={'size': 14})
    ax.set_xlabel('Predito', fontsize=12)
    ax.set_ylabel('Real', fontsize=12)
    ax.set_title(f'Matriz de Confusão - {best_model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    filepath = os.path.join(output_dir, '6_confusion_matrix.png')
    plt.savefig(filepath, dpi=100, bbox_inches='tight')
    figure_files['confusion_matrix'] = filepath
    print(f"✓ Salvo: {filepath}")
    plt.close()
    
    # Gerar relatório de texto
    report_text_path = os.path.join(output_dir, 'relatorio_analise.txt')
    with open(report_text_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("RELATÓRIO DE ANÁLISE - TITANIC DATASET\n")
        f.write(f"Data de Geração: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        f.write("COMPARAÇÃO DE MODELOS\n")
        f.write("-"*80 + "\n")
        f.write(df_comparison.to_string(index=False))
        f.write("\n\n")
        
        f.write("MELHOR MODELO\n")
        f.write("-"*80 + "\n")
        f.write(f"Modelo com melhor acurácia: {best_model_name}\n")
        f.write(f"Acurácia Test: {df_comparison.loc[df_comparison['test_acc'].idxmax(), 'test_acc']:.4f}\n")
        f.write(f"ROC-AUC: {df_comparison.loc[df_comparison['test_acc'].idxmax(), 'roc_auc']:.4f}\n")
        f.write("\n")
        
        f.write("GRÁFICOS GERADOS\n")
        f.write("-"*80 + "\n")
        for name, path in figure_files.items():
            f.write(f"- {name}: {path}\n")
    
    print(f"✓ Salvo: {report_text_path}")
    print(f"\n✓ Relatório completo gerado em: {output_dir}/")
    print(f"✓ Total de {len(figure_files)} gráficos salvos")
    
    return output_dir


if __name__ == "__main__":
    main()