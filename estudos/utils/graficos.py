"""Gráficos reutilizáveis para os notebooks de clusterização."""

import matplotlib.pyplot as plt
import numpy as np

PALETA = np.array(["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
                   "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"])


def cores(rotulos):
    """Converte rótulos de cluster em cores; ruído (-1) fica preto."""
    rotulos = np.asarray(rotulos)
    c = PALETA[np.mod(rotulos, len(PALETA))]
    c[rotulos == -1] = "#000000"
    return c


def dispersao_clusters(ax, X, rotulos, titulo="", s=8):
    """Plota as duas primeiras colunas de X coloridas por cluster."""
    ax.scatter(X[:, 0], X[:, 1], c=cores(rotulos), s=s, linewidths=0)
    ax.set_title(titulo, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def curva_escolha_k(ks, inercias, silhuetas, titulo="Escolha de k"):
    """Cotovelo (inércia) e silhueta lado a lado."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.5))
    a1.plot(ks, inercias, "o-")
    a1.set(xlabel="k", ylabel="Inércia / custo", title="Cotovelo")
    a2.plot(ks, silhuetas, "o-", color="#DD8452")
    a2.set(xlabel="k", ylabel="Silhueta média", title="Silhueta")
    fig.suptitle(titulo)
    fig.tight_layout()
    return fig
