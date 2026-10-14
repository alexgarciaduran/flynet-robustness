"""
@author: alexgarciaduran
"""

import warnings

import numpy as np

from .config import flops as _flops

_trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy < 2.0


def robustness(eps, accuracy):
    """Robustness R: mean accuracy over the swept perturbation range (AUC)."""
    eps = np.asarray(eps, dtype=float)
    order = np.argsort(eps)
    acc = np.asarray(accuracy, dtype=float)[order]
    span = eps.max() - eps.min()
    if span == 0:
        return float(acc.mean())
    return _trapezoid(acc, eps[order]) / span


def bootstrap_robustness(eps, accuracy_matrix, n_boot=1000, seed=None):
    """Bootstrap mean/std of R over trajectories.

    accuracy_matrix: (n_eps, n_trajectories) accuracy values.
    """
    rng = np.random.default_rng(seed)
    n = accuracy_matrix.shape[1]
    per_traj = np.array([robustness(eps, accuracy_matrix[:, i]) for i in range(n)])
    means = [per_traj[rng.integers(0, n, n)].mean() for _ in range(n_boot)]
    return float(np.mean(means)), float(np.std(means))


def lers(R, network_id=None, flops=None):
    """Log-efficiency robustness score: R / log10(FLOPs)."""
    if flops is None:
        flops = _flops(network_id)
    return R / np.log10(flops)


def _nanmean(a, axis):
    """nanmean that stays quiet about the all-NaN diagonal of a transfer array."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=axis)


def transfer_target_accuracy(acc):
    """Mean accuracy per target and epsilon, over sources and attacks."""
    return _nanmean(acc, axis=(0, 3))


def transfer_auc_matrix(acc, eps, eps_index=None):
    """Source x target transfer robustness, NaN on the diagonal.

    With `eps_index`, the mean accuracy there; otherwise the area under the
    accuracy-vs-epsilon curve.
    """
    mean = _nanmean(acc, axis=3)                      # (source, target, eps)
    if eps_index is not None:
        return mean[:, :, eps_index]
    n = mean.shape[0]
    out = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if i != j:
                out[i, j] = robustness(eps, mean[i, j])
    return out
