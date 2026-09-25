"""Robustness (area under the accuracy curve) and efficiency metrics."""

import numpy as np

from .config import flops as _flops


def robustness(eps, accuracy):
    """Robustness R: mean accuracy over the swept perturbation range (AUC)."""
    eps = np.asarray(eps, dtype=float)
    order = np.argsort(eps)
    acc = np.asarray(accuracy, dtype=float)[order]
    return np.trapezoid(acc, eps[order]) / (eps.max() - eps.min())


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
