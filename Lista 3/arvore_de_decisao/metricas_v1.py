"""
Métricas de Avaliação para Classificação

Este módulo contém funções para calcular métricas de desempenho
de modelos de classificação.

Autor: Lucas Rafael P. do Nascimento
"""

import numpy as np
from collections import Counter


def confusion_matrix(y_true, y_pred):
    """Calcula a matriz de confusão.
    
    Args:
        y_true (array-like): Rótulos verdadeiros.
        y_pred (array-like): Rótulos preditos.
    
    Returns:
        dict: Matriz de confusão com VP, VN, FP, FN para cada classe.
    
    Examples:
        >>> y_true = [0, 1, 1, 0, 1]
        >>> y_pred = [0, 1, 0, 0, 1]
        >>> cm = confusion_matrix(y_true, y_pred)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    classes = np.unique(np.concatenate([y_true, y_pred]))
    
    # Dicionário para armazenar métricas de cada classe
    cm = {}
    
    for cls in classes:
        vp = np.sum((y_true == cls) & (y_pred == cls))  # Verdadeiro Positivo
        vn = np.sum((y_true != cls) & (y_pred != cls))  # Verdadeiro Negativo
        fp = np.sum((y_true != cls) & (y_pred == cls))  # Falso Positivo
        fn = np.sum((y_true == cls) & (y_pred != cls))  # Falso Negativo
        
        cm[cls] = {
            'VP': vp,
            'VN': vn,
            'FP': fp,
            'FN': fn
        }
    
    return cm


def precision(y_true, y_pred, average='macro', zero_division=0):
    """Calcula a precisão.
    
    Precisão = VP / (VP + FP)
    Mede a proporção de predições positivas que estão corretas.
    
    Args:
        y_true (array-like): Rótulos verdadeiros.
        y_pred (array-like): Rótulos preditos.
        average (str): Tipo de média ('macro', 'micro', 'weighted', None).
        zero_division (float): Valor retornado quando divisão por zero (padrão 0).
    
    Returns:
        float ou dict: Precisão (ou dicionário com precisão por classe se average=None).
    
    Examples:
        >>> y_true = [0, 1, 1, 0, 1]
        >>> y_pred = [0, 1, 0, 0, 1]
        >>> precision(y_true, y_pred)
        0.75
    """
    cm = confusion_matrix(y_true, y_pred)
    precisions = {}
    
    for cls, metrics in cm.items():
        vp = metrics['VP']
        fp = metrics['FP']
        
        if vp + fp == 0:
            precisions[cls] = zero_division
        else:
            precisions[cls] = vp / (vp + fp)
    
    if average is None:
        return precisions
    elif average == 'macro':
        return np.mean(list(precisions.values()))
    elif average == 'micro':
        total_vp = sum(m['VP'] for m in cm.values())
        total_fp = sum(m['FP'] for m in cm.values())
        return total_vp / (total_vp + total_fp) if (total_vp + total_fp) > 0 else zero_division
    elif average == 'weighted':
        y_true = np.array(y_true)
        weights = {cls: np.sum(y_true == cls) for cls in cm.keys()}
        total = len(y_true)
        return sum(precisions[cls] * weights[cls] / total for cls in cm.keys())
    else:
        raise ValueError(f"average deve ser 'macro', 'micro', 'weighted' ou None, não '{average}'")


def recall(y_true, y_pred, average='macro', zero_division=0):
    """Calcula o recall (revocação/sensibilidade).
    
    Recall = VP / (VP + FN)
    Mede a proporção de casos positivos reais que foram identificados.
    
    Args:
        y_true (array-like): Rótulos verdadeiros.
        y_pred (array-like): Rótulos preditos.
        average (str): Tipo de média ('macro', 'micro', 'weighted', None).
        zero_division (float): Valor retornado quando divisão por zero (padrão 0).
    
    Returns:
        float ou dict: Recall (ou dicionário com recall por classe se average=None).
    
    Examples:
        >>> y_true = [0, 1, 1, 0, 1]
        >>> y_pred = [0, 1, 0, 0, 1]
        >>> recall(y_true, y_pred)
        0.75
    """
    cm = confusion_matrix(y_true, y_pred)
    recalls = {}
    
    for cls, metrics in cm.items():
        vp = metrics['VP']
        fn = metrics['FN']
        
        if vp + fn == 0:
            recalls[cls] = zero_division
        else:
            recalls[cls] = vp / (vp + fn)
    
    if average is None:
        return recalls
    elif average == 'macro':
        return np.mean(list(recalls.values()))
    elif average == 'micro':
        total_vp = sum(m['VP'] for m in cm.values())
        total_fn = sum(m['FN'] for m in cm.values())
        return total_vp / (total_vp + total_fn) if (total_vp + total_fn) > 0 else zero_division
    elif average == 'weighted':
        y_true = np.array(y_true)
        weights = {cls: np.sum(y_true == cls) for cls in cm.keys()}
        total = len(y_true)
        return sum(recalls[cls] * weights[cls] / total for cls in cm.keys())
    else:
        raise ValueError(f"average deve ser 'macro', 'micro', 'weighted' ou None, não '{average}'")


def f1_score(y_true, y_pred, average='macro', zero_division=0):
    """Calcula o F1-Score.
    
    F1 = 2 × (Precisão × Recall) / (Precisão + Recall)
    Média harmônica entre precisão e recall.
    
    Args:
        y_true (array-like): Rótulos verdadeiros.
        y_pred (array-like): Rótulos preditos.
        average (str): Tipo de média ('macro', 'micro', 'weighted', None).
        zero_division (float): Valor retornado quando divisão por zero (padrão 0).
    
    Returns:
        float ou dict: F1-Score (ou dicionário com F1 por classe se average=None).
    
    Examples:
        >>> y_true = [0, 1, 1, 0, 1]
        >>> y_pred = [0, 1, 0, 0, 1]
        >>> f1_score(y_true, y_pred)
        0.75
    """
    prec = precision(y_true, y_pred, average=None, zero_division=zero_division)
    rec = recall(y_true, y_pred, average=None, zero_division=zero_division)
    
    f1_scores = {}
    for cls in prec.keys():
        p = prec[cls]
        r = rec[cls]
        
        if p + r == 0:
            f1_scores[cls] = zero_division
        else:
            f1_scores[cls] = 2 * (p * r) / (p + r)
    
    if average is None:
        return f1_scores
    elif average == 'macro':
        return np.mean(list(f1_scores.values()))
    elif average == 'micro':
        # Para micro, F1 = Precisão = Recall
        return precision(y_true, y_pred, average='micro', zero_division=zero_division)
    elif average == 'weighted':
        y_true = np.array(y_true)
        weights = {cls: np.sum(y_true == cls) for cls in f1_scores.keys()}
        total = len(y_true)
        return sum(f1_scores[cls] * weights[cls] / total for cls in f1_scores.keys())
    else:
        raise ValueError(f"average deve ser 'macro', 'micro', 'weighted' ou None, não '{average}'")


def accuracy(y_true, y_pred):
    """Calcula a acurácia.
    
    Acurácia = (VP + VN) / Total
    Proporção de predições corretas.
    
    Args:
        y_true (array-like): Rótulos verdadeiros.
        y_pred (array-like): Rótulos preditos.
    
    Returns:
        float: Acurácia entre 0 e 1.
    
    Examples:
        >>> y_true = [0, 1, 1, 0, 1]
        >>> y_pred = [0, 1, 0, 0, 1]
        >>> accuracy(y_true, y_pred)
        0.6
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(y_true == y_pred)


def specificity(y_true, y_pred, average='macro', zero_division=0):
    """Calcula a especificidade.
    
    Especificidade = VN / (VN + FP)
    Mede a proporção de casos negativos reais que foram identificados.
    
    Args:
        y_true (array-like): Rótulos verdadeiros.
        y_pred (array-like): Rótulos preditos.
        average (str): Tipo de média ('macro', 'micro', 'weighted', None).
        zero_division (float): Valor retornado quando divisão por zero (padrão 0).
    
    Returns:
        float ou dict: Especificidade (ou dicionário por classe se average=None).
    
    Examples:
        >>> y_true = [0, 1, 1, 0, 1]
        >>> y_pred = [0, 1, 0, 0, 1]
        >>> specificity(y_true, y_pred)
        0.833
    """
    cm = confusion_matrix(y_true, y_pred)
    specificities = {}
    
    for cls, metrics in cm.items():
        vn = metrics['VN']
        fp = metrics['FP']
        
        if vn + fp == 0:
            specificities[cls] = zero_division
        else:
            specificities[cls] = vn / (vn + fp)
    
    if average is None:
        return specificities
    elif average == 'macro':
        return np.mean(list(specificities.values()))
    elif average == 'micro':
        total_vn = sum(m['VN'] for m in cm.values())
        total_fp = sum(m['FP'] for m in cm.values())
        return total_vn / (total_vn + total_fp) if (total_vn + total_fp) > 0 else zero_division
    elif average == 'weighted':
        y_true = np.array(y_true)
        # Para especificidade, peso é baseado em negativos
        weights = {cls: np.sum(y_true != cls) for cls in cm.keys()}
        total = sum(weights.values())
        return sum(specificities[cls] * weights[cls] / total for cls in cm.keys())
    else:
        raise ValueError(f"average deve ser 'macro', 'micro', 'weighted' ou None, não '{average}'")


def classification_report(y_true, y_pred, class_names=None, digits=4):
    """Gera um relatório completo de classificação.
    
    Args:
        y_true (array-like): Rótulos verdadeiros.
        y_pred (array-like): Rótulos preditos.
        class_names (list, optional): Nomes das classes.
        digits (int): Número de casas decimais (padrão 4).
    
    Returns:
        str: Relatório formatado com todas as métricas.
    
    Examples:
        >>> y_true = [0, 1, 1, 0, 1, 2, 2, 0]
        >>> y_pred = [0, 1, 0, 0, 1, 2, 1, 0]
        >>> print(classification_report(y_true, y_pred, class_names=['A', 'B', 'C']))
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    classes = np.unique(np.concatenate([y_true, y_pred]))
    
    # Calcular métricas por classe
    prec = precision(y_true, y_pred, average=None)
    rec = recall(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)
    spec = specificity(y_true, y_pred, average=None)
    
    # Suporte (número de ocorrências de cada classe)
    support = {cls: np.sum(y_true == cls) for cls in classes}
    
    # Construir relatório
    header = f"{'Classe':<15} {'Precisão':<12} {'Recall':<12} {'F1-Score':<12} {'Especif.':<12} {'Suporte':<10}"
    separator = "=" * len(header)
    
    lines = ["\n" + separator, "RELATÓRIO DE CLASSIFICAÇÃO", separator, header, "-" * len(header)]
    
    for cls in classes:
        cls_name = class_names[cls] if class_names else f"Classe {cls}"
        line = (f"{cls_name:<15} "
                f"{prec[cls]:<12.{digits}f} "
                f"{rec[cls]:<12.{digits}f} "
                f"{f1[cls]:<12.{digits}f} "
                f"{spec[cls]:<12.{digits}f} "
                f"{support[cls]:<10}")
        lines.append(line)
    
    lines.append("-" * len(header))
    
    # Médias
    lines.append(f"{'Acurácia':<15} {'':<12} {'':<12} {accuracy(y_true, y_pred):<12.{digits}f} {'':<12} {len(y_true):<10}")
    lines.append(f"{'Macro avg':<15} "
                 f"{precision(y_true, y_pred, average='macro'):<12.{digits}f} "
                 f"{recall(y_true, y_pred, average='macro'):<12.{digits}f} "
                 f"{f1_score(y_true, y_pred, average='macro'):<12.{digits}f} "
                 f"{specificity(y_true, y_pred, average='macro'):<12.{digits}f} "
                 f"{len(y_true):<10}")
    lines.append(f"{'Weighted avg':<15} "
                 f"{precision(y_true, y_pred, average='weighted'):<12.{digits}f} "
                 f"{recall(y_true, y_pred, average='weighted'):<12.{digits}f} "
                 f"{f1_score(y_true, y_pred, average='weighted'):<12.{digits}f} "
                 f"{specificity(y_true, y_pred, average='weighted'):<12.{digits}f} "
                 f"{len(y_true):<10}")
    
    lines.append(separator + "\n")
    
    return "\n".join(lines)


def print_confusion_matrix(y_true, y_pred, class_names=None):
    """Imprime a matriz de confusão em formato tabular.
    
    Args:
        y_true (array-like): Rótulos verdadeiros.
        y_pred (array-like): Rótulos preditos.
        class_names (list, optional): Nomes das classes.
    
    Examples:
        >>> y_true = [0, 1, 1, 0, 1]
        >>> y_pred = [0, 1, 0, 0, 1]
        >>> print_confusion_matrix(y_true, y_pred, class_names=['Não', 'Sim'])
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)
    
    # Criar matriz
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for i, true_cls in enumerate(classes):
        for j, pred_cls in enumerate(classes):
            matrix[i, j] = np.sum((y_true == true_cls) & (y_pred == pred_cls))
    
    # Imprimir
    print("\n" + "="*60)
    print("MATRIZ DE CONFUSÃO")
    print("="*60)
    
    # Cabeçalho
    header = "Real \\ Predito |"
    for cls in classes:
        cls_name = class_names[cls] if class_names else f"C{cls}"
        header += f" {cls_name:>6} |"
    print(header)
    print("-" * len(header))
    
    # Linhas
    for i, true_cls in enumerate(classes):
        cls_name = class_names[true_cls] if class_names else f"C{true_cls}"
        line = f"{cls_name:>14} |"
        for j in range(n_classes):
            line += f" {matrix[i, j]:>6} |"
        print(line)
    
    print("="*60 + "\n")


def evaluate_model(model, X_test, y_test, class_names=None):
    """Avalia um modelo de forma completa.
    
    Args:
        model: Modelo com método predict().
        X_test (array-like): Dados de teste.
        y_test (array-like): Rótulos verdadeiros de teste.
        class_names (list, optional): Nomes das classes.
    
    Returns:
        dict: Dicionário com todas as métricas.
    
    Examples:
        >>> from decision_tree import DecisionTree
        >>> tree = DecisionTree(max_depth=3)
        >>> tree.fit(X_train, y_train)
        >>> results = evaluate_model(tree, X_test, y_test, class_names=['A', 'B', 'C'])
    """
    # Fazer predições
    y_pred = model.predict(X_test)
    
    # Calcular métricas
    results = {
        'accuracy': accuracy(y_test, y_pred),
        'precision_macro': precision(y_test, y_pred, average='macro'),
        'recall_macro': recall(y_test, y_pred, average='macro'),
        'f1_macro': f1_score(y_test, y_pred, average='macro'),
        'specificity_macro': specificity(y_test, y_pred, average='macro'),
        'precision_weighted': precision(y_test, y_pred, average='weighted'),
        'recall_weighted': recall(y_test, y_pred, average='weighted'),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
        'specificity_weighted': specificity(y_test, y_pred, average='weighted'),
        'precision_per_class': precision(y_test, y_pred, average=None),
        'recall_per_class': recall(y_test, y_pred, average=None),
        'f1_per_class': f1_score(y_test, y_pred, average=None),
        'specificity_per_class': specificity(y_test, y_pred, average=None),
    }
    
    # Imprimir relatórios
    print(classification_report(y_test, y_pred, class_names=class_names))
    print_confusion_matrix(y_test, y_pred, class_names=class_names)
    
    return results