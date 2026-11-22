"""Carregamento e codificação da base PNS usada no Projeto 2 (KNIME) e no C4.5."""

from pathlib import Path

import pandas as pd

RAIZ_REPO = Path(__file__).resolve().parents[2]
CAMINHO_PNS = RAIZ_REPO / "c45" / "dados_c45_sembalanceamento_originais.csv"

ALVO = "Q00201"

# Atributos com ordem natural: a lista define a ordem do menor para o maior.
ORDINAIS = {
    "C008": ["39-49 anos", "50-59 anos", "60-69 anos", "70-79 anos", "80-89 anos", "90+ anos"],
    "M01601": ["Nenhuma vez", "Uma vez no ano", "Algumas vezes no ano", "Uma vez por mes",
               "De 2 a 3 vezes por mes", "Mais de uma vez"],
    "N010": ["Nenhum dia", "Menos da metade dos dias", "Mais da metade dos dias", "Quase todos os dias"],
    "N011": ["Nenhum dia", "Menos da metade dos dias", "Mais da metade dos dias", "Quase todos os dias"],
    "N016": ["Nenhum dia", "Menos da metade dos dias", "Mais da metade dos dias", "Quase todos os dias"],
    "P02601": ["Muito baixo", "Baixo", "Adequado", "Alto", "Muito alto"],
    "VDD004A": ["Sem instrucao", "Fundamental incompleto", "Fundamental completo", "Medio incompleto",
                "Medio completo", "Superior incompleto", "Superior completo"],
    "VDF004": ["Ate 1/4 salario minimo", "Entre 1/4 e 1/2 salario minimo", "Entre 1/2 e 1 salario minimo",
               "Entre 1 e 2 salarios minimos", "Entre 2 e 3 salarios minimos",
               "Entre 3 e 5 salarios minimos", "Mais de 5 salarios minimos"],
}

# Atributo numérico (nº de moradores do domicílio).
NUMERICOS = ["C001"]


def carregar_pns(caminho=CAMINHO_PNS, remover_constantes=True, rotulo_faltante="Sem resposta"):
    """Lê o CSV da PNS (separador ';').

    Remove colunas constantes e, nas categóricas, troca valores ausentes por uma
    categoria explícita (M005010 tem ~47% de ausentes, ligados ao fluxo do questionário).
    """
    df = pd.read_csv(caminho, sep=";", encoding="utf-8-sig")
    categoricas = df.columns.difference(NUMERICOS)
    df[categoricas] = df[categoricas].fillna(rotulo_faltante)
    if remover_constantes:
        constantes = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
        df = df.drop(columns=constantes)
    return df


def ordinais_presentes(df):
    """Devolve apenas as colunas ordinais conhecidas cujas categorias batem com a ordem definida."""
    validas = {}
    for col, ordem in ORDINAIS.items():
        if col in df.columns and set(df[col].dropna().unique()) <= set(ordem):
            validas[col] = ordem
    return validas
