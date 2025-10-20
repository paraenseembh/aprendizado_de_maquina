'''# Estrutura de arquivos:
meu_projeto/
├── titanic_analysis.py  # Seu código original
└── usar_analise.py      # Seu script de uso
'''
# No usar_analise.py:
import sys
sys.path.insert(0, '.')  # Adiciona diretório atual
import titanic_analysis as ta

# Usar normalmente:
"""
Script de Teste para Análise Titanic
Executa o pipeline completo e gera relatórios
"""
import os
from datetime import datetime
import pandas as pd

# Importar o módulo principal (assumindo que está salvo como titanic_analysis.py)
# Se você salvou com outro nome, ajuste aqui
try:
    import titanic_analysis as ta
except ImportError:
    print("ERRO: Não foi possível importar o módulo titanic_analysis.py")
    print("Certifique-se de que o arquivo está no mesmo diretório ou no PYTHONPATH")
    sys.exit(1)

def print_section(title):
    """Imprime um separador de seção"""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")

def test_pipeline_complete():
    """Testa o pipeline completo"""
    
    print_section("INICIANDO TESTE DO PIPELINE COMPLETO")
    print(f"Data/Hora: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    
    # ===========================================================================
    # ETAPA 1: Carregar Dados
    # ===========================================================================
    print_section("ETAPA 1/7: Carregando Dataset Titanic")
    
    try:
        df = ta.load_titanic_data()
        print("✓ Dataset carregado com sucesso!")
    except Exception as e:
        print(f"✗ ERRO ao carregar dados: {e}")
        return False
    
    # ===========================================================================
    # ETAPA 2: Análise Exploratória
    # ===========================================================================
    print_section("ETAPA 2/7: Análise Exploratória de Dados")
    
    try:
        print("\n--- Análise de Dados Ausentes ---")
        missing_table = ta.analyze_missing_data(df)
        
        print("\n--- Análise de Redundância (Correlações) ---")
        corr_matrix = ta.check_redundancy(df)
        
        print("✓ Análise exploratória concluída!")
    except Exception as e:
        print(f"✗ ERRO na análise exploratória: {e}")
        return False
    
    # ===========================================================================
    # ETAPA 3: Pré-processamento
    # ===========================================================================
    print_section("ETAPA 3/7: Pré-processamento dos Dados")
    
    try:
        X, y, df_processed = ta.preprocess_data(df)
        print(f"✓ Features shape: {X.shape}")
        print(f"✓ Target shape: {y.shape}")
        print(f"✓ Pré-processamento concluído!")
    except Exception as e:
        print(f"✗ ERRO no pré-processamento: {e}")
        return False
    
    # ===========================================================================
    # ETAPA 4: Split e Imputação
    # ===========================================================================
    print_section("ETAPA 4/7: Split Train/Test e Imputação")
    
    try:
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"✓ Train set: {X_train.shape}")
        print(f"✓ Test set: {X_test.shape}")
        
        # Imputação
        X_train_imp, X_test_imp = ta.impute_missing_values(
            X_train, X_test, method='knn'
        )
        
        print("✓ Split e imputação concluídos!")
    except Exception as e:
        print(f"✗ ERRO no split/imputação: {e}")
        return False
    
    # ===========================================================================
    # ETAPA 5: Análise de Balanceamento e Modelagem
    # ===========================================================================
    print_section("ETAPA 5/7: Treinamento de Modelos")
    
    try:
        # Analisar balanceamento
        ta.analyze_class_balance(y_train)
        
        # Armazenar resultados e modelos
        all_results = []
        model_objects = {}
        
        # --- EXPERIMENTO 1: Baseline ---
        print("\n" + "="*70)
        print("EXPERIMENTO 1: Modelos Baseline (sem balanceamento)")
        print("="*70 + "\n")
        
        # Decision Tree baseline
        print("Treinando Decision Tree (Baseline)...")
        dt_baseline = ta.train_decision_tree(X_train_imp, y_train)
        results_dt_baseline = ta.evaluate_model(
            dt_baseline, X_train_imp, X_test_imp, y_train, y_test,
            "Decision Tree (Baseline)"
        )
        all_results.append(results_dt_baseline)
        model_objects["Decision Tree (Baseline)"] = dt_baseline
        
        # Random Forest baseline
        print("\nTreinando Random Forest (Baseline)...")
        rf_baseline = ta.train_random_forest(X_train_imp, y_train)
        results_rf_baseline = ta.evaluate_model(
            rf_baseline, X_train_imp, X_test_imp, y_train, y_test,
            "Random Forest (Baseline)"
        )
        all_results.append(results_rf_baseline)
        model_objects["Random Forest (Baseline)"] = rf_baseline
        
        # --- EXPERIMENTO 2: SMOTE ---
        print("\n" + "="*70)
        print("EXPERIMENTO 2: Modelos com SMOTE")
        print("="*70 + "\n")
        
        X_train_smote, y_train_smote = ta.apply_balancing(
            X_train_imp, y_train, method='smote'
        )
        
        # Decision Tree com SMOTE
        print("Treinando Decision Tree (SMOTE)...")
        dt_smote = ta.train_decision_tree(X_train_smote, y_train_smote)
        results_dt_smote = ta.evaluate_model(
            dt_smote, X_train_smote, X_test_imp, y_train_smote, y_test,
            "Decision Tree (SMOTE)"
        )
        all_results.append(results_dt_smote)
        model_objects["Decision Tree (SMOTE)"] = dt_smote
        
        # Random Forest com SMOTE
        print("\nTreinando Random Forest (SMOTE)...")
        rf_smote = ta.train_random_forest(X_train_smote, y_train_smote)
        results_rf_smote = ta.evaluate_model(
            rf_smote, X_train_smote, X_test_imp, y_train_smote, y_test,
            "Random Forest (SMOTE)"
        )
        all_results.append(results_rf_smote)
        model_objects["Random Forest (SMOTE)"] = rf_smote
        
        # --- EXPERIMENTO 3: Undersampling ---
        print("\n" + "="*70)
        print("EXPERIMENTO 3: Modelos com Undersampling")
        print("="*70 + "\n")
        
        X_train_under, y_train_under = ta.apply_balancing(
            X_train_imp, y_train, method='undersample'
        )
        
        # Decision Tree com Undersampling
        print("Treinando Decision Tree (Undersample)...")
        dt_under = ta.train_decision_tree(X_train_under, y_train_under)
        results_dt_under = ta.evaluate_model(
            dt_under, X_train_under, X_test_imp, y_train_under, y_test,
            "Decision Tree (Undersample)"
        )
        all_results.append(results_dt_under)
        model_objects["Decision Tree (Undersample)"] = dt_under
        
        # Random Forest com Undersampling
        print("\nTreinando Random Forest (Undersample)...")
        rf_under = ta.train_random_forest(X_train_under, y_train_under)
        results_rf_under = ta.evaluate_model(
            rf_under, X_train_under, X_test_imp, y_train_under, y_test,
            "Random Forest (Undersample)"
        )
        all_results.append(results_rf_under)
        model_objects["Random Forest (Undersample)"] = rf_under
        
        print("\n✓ Todos os modelos treinados com sucesso!")
        
    except Exception as e:
        print(f"✗ ERRO no treinamento: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ===========================================================================
    # ETAPA 6: Comparação de Resultados
    # ===========================================================================
    print_section("ETAPA 6/7: Comparação de Resultados")
    
    try:
        df_comparison = ta.compare_models(all_results)
        print("\n✓ Comparação concluída!")
    except Exception as e:
        print(f"✗ ERRO na comparação: {e}")
        return False
    
    # ===========================================================================
    # ETAPA 7: Geração de Relatório
    # ===========================================================================
    print_section("ETAPA 7/7: Gerando Relatório Completo")
    
    try:
        
        report_path = ta.generate_report(
            df_comparison=df_comparison,
            model_objects=model_objects,
            X_train=X_train_imp,
            X_test=X_test_imp,
            y_train=y_train,
            y_test=y_test,
        )
        

        output_dir = '/home/lucasr/Documents/aprendizado_de_maquina/Lista 4'
        print(f"\n✓ Relatório gerado com sucesso em: {output_dir}/")
        print(f"\nArquivos gerados:")
        if os.path.exists(output_dir):
            for file in sorted(os.listdir(output_dir)):
                print(f"  - {file}")
        
    except Exception as e:
        print(f"✗ ERRO na geração do relatório: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ===========================================================================
    # SUCESSO
    # ===========================================================================
    print_section("TESTE CONCLUÍDO COM SUCESSO! ✓")
    
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                         TESTE FINALIZADO COM SUCESSO                      ║
╚═══════════════════════════════════════════════════════════════════════════╝

Resumo da Execução:
✓ Carregamento de dados
✓ Análise exploratória
✓ Pré-processamento
✓ Imputação de valores ausentes
✓ Treinamento de 6 modelos (2 algoritmos × 3 estratégias de balanceamento)
✓ Avaliação detalhada com múltiplas métricas
✓ Geração de relatório com gráficos

Próximos passos:
1. Verifique o diretório de relatório gerado
2. Analise o arquivo relatorio_analise.txt para resultados detalhados
3. Visualize os gráficos gerados (.png)
4. Compare os modelos e selecione o melhor para seu caso de uso
    """)
    
    return True

def test_individual_functions():
    """Testa funções individuais"""
    
    print_section("TESTE DE FUNÇÕES INDIVIDUAIS")
    
    # Teste 1: Carregamento
    print("\n1. Testando carregamento de dados...")
    try:
        df = ta.load_titanic_data()
        assert df.shape[0] > 0, "Dataset vazio"
        print("   ✓ Carregamento OK")
    except Exception as e:
        print(f"   ✗ ERRO: {e}")
        return False
    
    # Teste 2: Pré-processamento
    print("\n2. Testando pré-processamento...")
    try:
        X, y, df_processed = ta.preprocess_data(df)
        assert X.shape[0] == y.shape[0], "Dimensões incompatíveis"
        assert X.isnull().sum().sum() > 0, "Deveria ter valores ausentes antes da imputação"
        print("   ✓ Pré-processamento OK")
    except Exception as e:
        print(f"   ✗ ERRO: {e}")
        return False
    
    # Teste 3: Treinamento
    print("\n3. Testando treinamento de modelos...")
    try:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        X_train_imp, X_test_imp = ta.impute_missing_values(X_train, X_test)
        
        dt = ta.train_decision_tree(X_train_imp, y_train)
        rf = ta.train_random_forest(X_train_imp, y_train)
        
        assert hasattr(dt, 'predict'), "Modelo DT não tem método predict"
        assert hasattr(rf, 'predict'), "Modelo RF não tem método predict"
        
        print("   ✓ Treinamento OK")
    except Exception as e:
        print(f"   ✗ ERRO: {e}")
        return False
    
    print("\n✓ Todos os testes individuais passaram!")
    return True

def main():
    """Função principal de teste"""
    
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                    SCRIPT DE TESTE - ANÁLISE TITANIC                      ║
║            Decision Tree vs Random Forest com Balanceamento               ║
╚═══════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Menu de opções
    print("\nEscolha o tipo de teste:")
    print("1. Teste Completo (Pipeline completo + Relatório)")
    print("2. Teste de Funções Individuais (Rápido)")
    print("3. Executar Ambos")
    
    choice = input("\nDigite sua escolha (1/2/3): ").strip()
    
    if choice == '1':
        success = test_pipeline_complete()
    elif choice == '2':
        success = test_individual_functions()
    elif choice == '3':
        success_ind = test_individual_functions()
        success_full = test_pipeline_complete()
        success = success_ind and success_full
    else:
        print("Opção inválida! Executando teste completo por padrão...")
        success = test_pipeline_complete()
    
    # Status final
    if success:
        print("\n" + "="*80)
        print("STATUS FINAL: SUCESSO ✓")
        print("="*80)
        return 0
    else:
        print("\n" + "="*80)
        print("STATUS FINAL: FALHA ✗")
        print("="*80)
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)