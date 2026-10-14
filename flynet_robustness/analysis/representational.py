"""
@author: alexgarciaduran
"""

import numpy as np
from sklearn.decomposition import PCA


def cosine_similarity_matrix(X):
    """Row-wise cosine similarity of X (M, D) -> (M, M)."""
    X = np.asarray(X)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return Xn @ Xn.T


def trial_similarity(activity):
    """Population and per-neuron cosine similarity across trials.

    activity: list of M arrays (T_k, N). Returns (population (M,M),
    neuron (N, M, M)).
    """
    N = activity[0].shape[1]
    X = np.stack([np.asarray(a).mean(axis=0) for a in activity])  # (M, N)
    population = cosine_similarity_matrix(X)
    neuron = np.stack([cosine_similarity_matrix(X[:, n:n + 1]) for n in range(N)])
    return population, neuron


def fit_global_pca(activity_all, n_components=3, indices=None):
    """Fit PCA across all timepoints of the selected trials."""
    if indices is None:
        indices = np.arange(len(activity_all))
    data = np.concatenate([np.asarray(activity_all[k][0]) for k in indices], axis=0)
    return PCA(n_components=n_components).fit(data)
