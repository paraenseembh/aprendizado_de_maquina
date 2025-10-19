"""
Exemplo de Uso das Métricas de Avaliação

Este script demonstra como usar todas as métricas implementadas
para avaliar uma árvore de decisão.
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from arvore_de_decisao_v1 import DecisionTree
from metricas_v1 import (precision, recall, f1_score, accuracy, specificity,
                     classification_report, print_confusion_matrix, evaluate_model)

# Importar a árvore de decisão e métricas
# from decision_tree import DecisionTree
# from metrics import (precision, recall, f1_score, accuracy, specificity,
#                      classification_report, print_confusion_matrix, evaluate_model)

# ==============================================================================
# 1. PREPARAR DADOS
# ==============================================================================

# Carregar dataset
iris = load_iris()
X, y = iris.data, iris.target

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Nomes das classes
class_names = ['Setosa', 'Versicolor', 'Virginica']
feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

# ==============================================================================
# 2. TREINAR MODELO
# ==============================================================================

tree = DecisionTree(max_depth=3)
tree.fit(X_train, y_train)

# Fazer predições
y_pred = tree.predict(X_test)

print("Predições realizadas!")
print(f"Total de amostras no teste: {len(y_test)}")

# ==============================================================================
# 3. CALCULAR MÉTRICAS INDIVIDUAIS
# ==============================================================================

print("\n" + "="*70)
print("MÉTRICAS INDIVIDUAIS")
print("="*70)

# Acurácia
acc = accuracy(y_test, y_pred)
print(f"\n📊 Acurácia: {acc:.4f} ({acc*100:.2f}%)")

# Precisão
prec_macro = precision(y_test, y_pred, average='macro')
prec_weighted = precision(y_test, y_pred, average='weighted')
print(f"\n🎯 Precisão (Macro): {prec_macro:.4f}")
print(f"🎯 Precisão (Weighted): {prec_weighted:.4f}")

# Recall
rec_macro = recall(y_test, y_pred, average='macro')
rec_weighted = recall(y_test, y_pred, average='weighted')
print(f"\n🔍 Recall (Macro): {rec_macro:.4f}")
print(f"🔍 Recall (Weighted): {rec_weighted:.4f}")

# F1-Score
f1_macro = f1_score(y_test, y_pred, average='macro')
f1_weighted = f1_score(y_test, y_pred, average='weighted')
print(f"\n⚖️  F1-Score (Macro): {f1_macro:.4f}")
print(f"⚖️  F1-Score (Weighted): {f1_weighted:.4f}")

# Especificidade
spec_macro = specificity(y_test, y_pred, average='macro')
spec_weighted = specificity(y_test, y_pred, average='weighted')
print(f"\n✅ Especificidade (Macro): {spec_macro:.4f}")
print(f"✅ Especificidade (Weighted): {spec_weighted:.4f}")

# ==============================================================================
# 4. MÉTRICAS POR CLASSE
# ==============================================================================

print("\n" + "="*70)
print("MÉTRICAS POR CLASSE")
print("="*70)

prec_per_class = precision(y_test, y_pred, average=None)
rec_per_class = recall(y_test, y_pred, average=None)
f1_per_class = f1_score(y_test, y_pred, average=None)
spec_per_class = specificity(y_test, y_pred, average=None)

for cls in prec_per_class.keys():
    cls_name = class_names[cls]
    print(f"\n📌 Classe: {cls_name}")
    print(f"   Precisão:      {prec_per_class[cls]:.4f}")
    print(f"   Recall:        {rec_per_class[cls]:.4f}")
    print(f"   F1-Score:      {f1_per_class[cls]:.4f}")
    print(f"   Especificidade: {spec_per_class[cls]:.4f}")

# ==============================================================================
# 5. MATRIZ DE CONFUSÃO
# ==============================================================================

print_confusion_matrix(y_test, y_pred, class_names=class_names)

# ==============================================================================
# 6. RELATÓRIO COMPLETO
# ==============================================================================

print(classification_report(y_test, y_pred, class_names=class_names))

# ==============================================================================
# 7. AVALIAÇÃO COMPLETA (TUDO DE UMA VEZ)
# ==============================================================================

print("\n" + "="*70)
print("AVALIAÇÃO COMPLETA DO MODELO")
print("="*70)

results = evaluate_model(tree, X_test, y_test, class_names=class_names)

# Acessar resultados específicos
print("\n📦 Dicionário de resultados disponíveis:")
for key, value in results.items():
    if not isinstance(value, dict):
        print(f"   {key}: {value:.4f}")

# ==============================================================================
# 8. COMPARAR DIFERENTES CONFIGURAÇÕES
# ==============================================================================

print("\n" + "="*70)
print("COMPARAÇÃO DE DIFERENTES PROFUNDIDADES")
print("="*70)

results_comparison = []

for depth in [2, 3, 5, 10, None]:
    tree_temp = DecisionTree(max_depth=depth)
    tree_temp.fit(X_train, y_train)
    y_pred_temp = tree_temp.predict(X_test)
    
    depth_str = str(depth) if depth else "Ilimitada"
    
    results_comparison.append({
        'depth': depth_str,
        'accuracy': accuracy(y_test, y_pred_temp),
        'precision': precision(y_test, y_pred_temp, average='macro'),
        'recall': recall(y_test, y_pred_temp, average='macro'),
        'f1': f1_score(y_test, y_pred_temp, average='macro'),
        'specificity': specificity(y_test, y_pred_temp, average='macro'),
    })

# Imprimir comparação
print(f"\n{'Profundidade':<15} {'Acurácia':<12} {'Precisão':<12} {'Recall':<12} {'F1-Score':<12} {'Especif.':<12}")
print("-" * 87)

for result in results_comparison:
    print(f"{result['depth']:<15} "
          f"{result['accuracy']:<12.4f} "
          f"{result['precision']:<12.4f} "
          f"{result['recall']:<12.4f} "
          f"{result['f1']:<12.4f} "
          f"{result['specificity']:<12.4f}")

# ==============================================================================
# 9. IDENTIFICAR MELHORES E PIORES CLASSES
# ==============================================================================

print("\n" + "="*70)
print("ANÁLISE DE DESEMPENHO POR CLASSE")
print("="*70)

f1_per_class = f1_score(y_test, y_pred, average=None)

best_class = max(f1_per_class.items(), key=lambda x: x[1])
worst_class = min(f1_per_class.items(), key=lambda x: x[1])

print(f"\n🏆 Melhor classe: {class_names[best_class[0]]} (F1 = {best_class[1]:.4f})")
print(f"⚠️  Pior classe: {class_names[worst_class[0]]} (F1 = {worst_class[1]:.4f})")

# ==============================================================================
# 10. SALVAR RESULTADOS EM ARQUIVO
# ==============================================================================

def save_results_to_file(y_test, y_pred, class_names, filename='resultados.txt'):
    """Salva todos os resultados em um arquivo de texto."""
    import sys
    from io import StringIO
    
    old_stdout = sys.stdout
    sys.stdout = buffer = StringIO()
    
    print(classification_report(y_test, y_pred, class_names=class_names))
    print_confusion_matrix(y_test, y_pred, class_names=class_names)
    
    sys.stdout = old_stdout
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(buffer.getvalue())
    
    print(f"\n💾 Resultados salvos em: {filename}")

save_results_to_file(y_test, y_pred, class_names)

# ==============================================================================
# 11. VISUALIZAR MÉTRICAS (OPCIONAL - REQUER MATPLOTLIB)
# ==============================================================================

def plot_metrics_comparison(results_comparison):
    """Plota comparação de métricas para diferentes configurações."""
    try:
        import matplotlib.pyplot as plt
        
        depths = [r['depth'] for r in results_comparison]
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'specificity']
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(depths))
        width = 0.15
        
        for i, metric in enumerate(metrics):
            values = [r[metric] for r in results_comparison]
            offset = width * (i - 2)
            ax.bar(x + offset, values, width, label=metric.capitalize())
        
        ax.set_xlabel('Profundidade da Árvore', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Comparação de Métricas por Profundidade', fontsize=14, weight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(depths)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        plt.savefig('metricas_comparacao.png', dpi=150, bbox_inches='tight')
        print("\n📊 Gráfico de comparação salvo em: metricas_comparacao.png")
        plt.show()
        
    except ImportError:
        print("\n⚠️  matplotlib não instalado. Instale com: pip install matplotlib")

# Plotar comparação (se matplotlib estiver disponível)
plot_metrics_comparison(results_comparison)

# ==============================================================================
# 12. ANÁLISE DE ERROS
# ==============================================================================

print("\n" + "="*70)
print("ANÁLISE DE ERROS")
print("="*70)

# Identificar amostras classificadas incorretamente
y_test_array = np.array(y_test)
y_pred_array = np.array(y_pred)
errors = y_test_array != y_pred_array
error_indices = np.where(errors)[0]

print(f"\n❌ Total de erros: {np.sum(errors)} de {len(y_test)} ({np.sum(errors)/len(y_test)*100:.2f}%)")

if len(error_indices) > 0:
    print("\n📋 Amostras classificadas incorretamente:")
    print(f"{'Índice':<10} {'Real':<15} {'Predito':<15} {'Features'}")
    print("-" * 70)
    
    for idx in error_indices[:10]:  # Mostrar apenas as 10 primeiras
        real_class = class_names[y_test_array[idx]]
        pred_class = class_names[y_pred_array[idx]]
        features = X_test[idx]
        print(f"{idx:<10} {real_class:<15} {pred_class:<15} {features}")
    
    if len(error_indices) > 10:
        print(f"... e mais {len(error_indices) - 10} erros")

# Matriz de erros por classe
print("\n📊 Distribuição de erros por classe:")
for cls in np.unique(y_test_array):
    cls_name = class_names[cls]
    cls_mask = y_test_array == cls
    cls_errors = np.sum(errors & cls_mask)
    cls_total = np.sum(cls_mask)
    error_rate = cls_errors / cls_total if cls_total > 0 else 0
    print(f"   {cls_name:<15}: {cls_errors}/{cls_total} erros ({error_rate*100:.2f}%)")

# ==============================================================================
# 13. ENTENDENDO AS MÉTRICAS
# ==============================================================================

print("\n" + "="*70)
print("GUIA DE INTERPRETAÇÃO DAS MÉTRICAS")
print("="*70)

print("""
🎯 PRECISÃO (Precision):
   - O que mede: De todas as predições positivas, quantas estavam corretas?
   - Quando usar: Quando falsos positivos são custosos
   - Fórmula: VP / (VP + FP)
   - Exemplo: Em diagnóstico médico, evitar dizer que alguém está doente quando não está

🔍 RECALL (Sensibilidade):
   - O que mede: De todos os casos positivos reais, quantos foram identificados?
   - Quando usar: Quando falsos negativos são custosos
   - Fórmula: VP / (VP + FN)
   - Exemplo: Em detecção de fraude, é crítico identificar todas as fraudes

⚖️  F1-SCORE:
   - O que mede: Média harmônica entre precisão e recall
   - Quando usar: Quando você quer balancear precisão e recall
   - Fórmula: 2 × (Precisão × Recall) / (Precisão + Recall)
   - Vantagem: Leva em conta tanto falsos positivos quanto falsos negativos

✅ ESPECIFICIDADE:
   - O que mede: De todos os casos negativos reais, quantos foram identificados?
   - Quando usar: Para avaliar a capacidade de identificar casos negativos
   - Fórmula: VN / (VN + FP)
   - Exemplo: Identificar corretamente pessoas saudáveis em um teste médico

📊 ACURÁCIA:
   - O que mede: Proporção total de predições corretas
   - Cuidado: Pode ser enganosa em datasets desbalanceados
   - Fórmula: (VP + VN) / Total
   - Exemplo: Com 95% de uma classe, um modelo "burro" que sempre prediz 
     essa classe teria 95% de acurácia!

🔢 TIPOS DE MÉDIA:
   - Macro: Média simples (todas as classes têm peso igual)
   - Weighted: Média ponderada (classes maiores têm mais peso)
   - Micro: Calcula globalmente (soma todos os VP, FP, FN)
""")

# ==============================================================================
# 14. EXEMPLO PRÁTICO DE INTERPRETAÇÃO
# ==============================================================================

print("\n" + "="*70)
print("EXEMPLO PRÁTICO DE INTERPRETAÇÃO")
print("="*70)

# Pegar a primeira classe como exemplo
exemplo_classe = 0
cls_name = class_names[exemplo_classe]

prec = precision(y_test, y_pred, average=None)[exemplo_classe]
rec = recall(y_test, y_pred, average=None)[exemplo_classe]
f1 = f1_score(y_test, y_pred, average=None)[exemplo_classe]

print(f"\n📌 Análise para classe '{cls_name}':\n")
print(f"   Precisão = {prec:.4f}")
print(f"   → Interpretação: De todas as vezes que o modelo disse '{cls_name}',")
print(f"     ele acertou {prec*100:.1f}% das vezes.\n")

print(f"   Recall = {rec:.4f}")
print(f"   → Interpretação: De todas as amostras que realmente eram '{cls_name}',")
print(f"     o modelo identificou {rec*100:.1f}% delas.\n")

print(f"   F1-Score = {f1:.4f}")
print(f"   → Interpretação: Este é um balanceamento entre precisão e recall.")
print(f"     Um F1 de {f1:.4f} indica {'bom' if f1 > 0.8 else 'moderado' if f1 > 0.6 else 'fraco'} desempenho geral.\n")

# ==============================================================================
# 15. DECISÕES BASEADAS EM MÉTRICAS
# ==============================================================================

print("\n" + "="*70)
print("GUIA DE DECISÃO BASEADO EM MÉTRICAS")
print("="*70)

print("""
❓ QUANDO PRIORIZAR CADA MÉTRICA:

1️⃣  Alta Precisão é crítica quando:
   - Custo de falso positivo é alto
   - Recursos são limitados para investigar alertas
   - Exemplo: Sistema de spam (não quero perder emails importantes)

2️⃣  Alto Recall é crítico quando:
   - Custo de falso negativo é alto
   - É importante não perder nenhum caso positivo
   - Exemplo: Detecção de câncer (não podemos deixar passar nenhum caso)

3️⃣  F1-Score alto é importante quando:
   - Precisão e recall são igualmente importantes
   - Classes estão desbalanceadas
   - Exemplo: Classificação de documentos

4️⃣  Alta Especificidade é crítica quando:
   - É importante identificar corretamente os negativos
   - Falsos positivos podem causar alarme desnecessário
   - Exemplo: Testes de triagem médica

5️⃣  Alta Acurácia é suficiente quando:
   - Classes estão balanceadas
   - Todos os tipos de erro têm custo similar
   - Exemplo: Classificação de dígitos escritos à mão

💡 DICA: Sempre considere o contexto do problema ao escolher a métrica!
""")

print("\n" + "="*70)
print("✅ ANÁLISE COMPLETA FINALIZADA!")
print("="*70)
print(f"""
📊 Resumo dos Resultados:
   - Acurácia Geral: {accuracy(y_test, y_pred):.4f}
   - F1-Score Médio: {f1_score(y_test, y_pred, average='macro'):.4f}
   - Melhor Classe: {class_names[best_class[0]]} (F1 = {best_class[1]:.4f})
   - Pior Classe: {class_names[worst_class[0]]} (F1 = {worst_class[1]:.4f})
   - Total de Erros: {np.sum(errors)}/{len(y_test)}

📁 Arquivos Gerados:
   - resultados.txt (relatório completo)
   - metricas_comparacao.png (se matplotlib disponível)

🎓 Use as métricas apropriadas para seu contexto específico!
""")