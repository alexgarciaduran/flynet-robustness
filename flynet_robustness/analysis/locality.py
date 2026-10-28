"""
@author: alexgarciaduran and alejandrorg00
"""

import numpy as np


def connection_distances_from_edges(df):
    """Per-edge Euclidean distance on the hex lattice (axial->cartesian)."""
    du = df["target_u"].values - df["source_u"].values
    dv = df["target_v"].values - df["source_v"].values
    x = du + 0.5 * dv
    y = dv * np.sqrt(3) / 2
    return np.sqrt(x ** 2 + y ** 2)


def locality_from_edges(df, weight_col="n_syn"):
    """Distance statistics and fractions of local weight from an edge dataframe."""
    d = connection_distances_from_edges(df)
    w = np.abs(df[weight_col].values)
    return {
        "mean_distance": float(d.mean()),
        "median_distance": float(np.median(d)),
        "weighted_mean_distance": float(np.average(d, weights=w)),
        "frac_within_1col": float((d <= 1).mean()),
        "frac_within_3col": float((d <= 3).mean()),
        "weight_frac_within_1col": float(w[d <= 1].sum() / w.sum()),
        "weight_frac_within_3col": float(w[d <= 3].sum() / w.sum()),
    }


def normalized_locality(distances, weights):
    """1 - weighted_mean_distance / max_distance (1=local, 0=maximally global)."""
    d_max = distances.max()
    if d_max == 0:
        return 1.0
    return 1.0 - np.average(distances, weights=weights) / d_max


def compute_functional_locality(get_activity_fn, input_positions, output_positions,
                                n_probe_neurons=100, perturbation_std=0.25, seed=42):
    """Influence-weighted mean input->response distance via input perturbations.

    get_activity_fn(x) maps an input vector to an output activity vector.
    Returns locality (1 - weighted_mean_d/d_max), weighted_mean_d and d_max.
    """
    rng = np.random.default_rng(seed)
    N_in = input_positions.shape[0]
    x_base = np.zeros(N_in) + 0.5
    act_base = get_activity_fn(x_base)
    probe_idx = rng.choice(N_in, size=min(n_probe_neurons, N_in), replace=False)

    distances, influences = [], []
    for j in probe_idx:
        x_pert = x_base.copy()
        x_pert[j] += perturbation_std
        delta = np.abs(get_activity_fn(x_pert) - act_base)
        dists = np.linalg.norm(output_positions - input_positions[j], axis=1)
        distances.append(dists)
        influences.append(delta)

    distances = np.concatenate(distances)
    influences = np.concatenate(influences)
    weighted_mean_d = np.average(distances, weights=influences + 1e-10)
    d_max = distances.max()
    return {"locality": 1.0 - weighted_mean_d / d_max,
            "weighted_mean_d": weighted_mean_d, "d_max": d_max}
