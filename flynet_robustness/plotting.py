"""
@author: alexgarciaduran
"""

import numpy as np
import torch

from . import config


def _ax(ax, figsize=(3, 2.2)):
    import matplotlib.pyplot as plt
    return plt.subplots(figsize=figsize) if ax is None else (ax.figure, ax)


def plot_p_hit_trace(p_hit, dt=0.01, ax=None, label=None, color="crimson"):
    """p(hit) over time for a single trajectory."""
    fig, ax = _ax(ax)
    t = np.arange(len(p_hit)) * dt
    ax.plot(t, np.asarray(p_hit).ravel(), color=color, lw=2, label=label)
    ax.axhline(0.5, ls="--", c="gray", alpha=0.6)
    ax.set(xlabel="Time (s)", ylabel="p(hit)", ylim=(-0.05, 1.05))
    return fig, ax


def plot_accuracy_curve(magnitudes, accuracy, network_ids=None, xlabel="Perturbation",
                        ax=None, errors=None):
    """Accuracy vs perturbation magnitude for one or more networks."""
    fig, ax = _ax(ax)
    if isinstance(accuracy, dict):
        for nid, acc in accuracy.items():
            err = None if errors is None else errors.get(nid)
            ax.errorbar(magnitudes, acc, yerr=err, color=config.color(nid),
                        label=config.label(nid), lw=2, marker="o", ms=3, capsize=2)
        ax.legend(frameon=False, fontsize=7)
    else:
        ax.errorbar(magnitudes, accuracy, yerr=errors, color="forestgreen", lw=2, marker="o", ms=3)
    ax.set(xlabel=xlabel, ylabel="Accuracy", ylim=(-0.05, 1.05))
    return fig, ax


def plot_robustness_efficiency(results, ax=None, annotate=True):
    """Robustness R vs FLOPs (log-x) scatter. results: {network_id: R}."""
    fig, ax = _ax(ax, figsize=(3.5, 2.5))
    for nid, R in results.items():
        ax.scatter(config.flops(nid), R, color=config.color(nid), s=40, zorder=3)
        if annotate:
            ax.annotate(config.label(nid), (config.flops(nid), R), fontsize=6,
                        xytext=(3, 3), textcoords="offset points")
    ax.set(xscale="log", xlabel="FLOPs", ylabel="Robustness R", ylim=(0, 1.05))
    return fig, ax


def plot_lers_bars(lers_by_network, ax=None, sort=True):
    """Bar chart of LERS per network. lers_by_network: {network_id: LERS}."""
    fig, ax = _ax(ax, figsize=(4, 2.2))
    items = list(lers_by_network.items())
    if sort:
        items.sort(key=lambda kv: kv[1], reverse=True)
    ids, vals = zip(*items)
    ax.bar([config.label(i) for i in ids], vals, color=[config.color(i) for i in ids])
    ax.set_ylabel("LERS")
    ax.tick_params(axis="x", rotation=45)
    return fig, ax


def plot_transfer_matrix(matrix, network_ids, ax=None, cmap="Greens", vmin=None, vmax=None):
    """Transfer matrix (rows: source, cols: target). Scale spans the data."""
    import matplotlib.pyplot as plt
    fig, ax = _ax(ax, figsize=(4, 3.5))
    matrix = np.asarray(matrix, dtype=float)
    vmin = np.nanmin(matrix) if vmin is None else vmin
    vmax = np.nanmax(matrix) if vmax is None else vmax
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    labels = [config.label(i) for i in network_ids]
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.set(xlabel="Target", ylabel="Source")
    plt.colorbar(im, ax=ax, label="AUC", fraction=0.046)
    return fig, ax


def plot_texture(tex_logits, ax=None, from_logits=True):
    """Show a grayscale texture (sigmoid of logits by default)."""
    fig, ax = _ax(ax, figsize=(2, 2))
    tex = torch.sigmoid(tex_logits) if from_logits else tex_logits
    gray = tex.detach()[0].mean(dim=-1).clamp(0, 1).cpu().numpy()
    ax.imshow(gray, cmap="gray", vmin=0, vmax=1)
    ax.axis("off")
    return fig, ax


def plot_decoding_scatter(y_true, y_pred, ax=None, labels=("x", "y", "z")):
    """Predicted vs true scatter per component, with overall R^2."""
    fig, ax = _ax(ax, figsize=(3, 3))
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    for c in range(y_true.shape[1]):
        ax.scatter(y_true[:, c], y_pred[:, c], s=2, alpha=0.3, label=labels[c])
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "k--", lw=1)
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean(0)) ** 2).sum()
    ax.set(xlabel="True", ylabel="Predicted", title=f"$R^2$ = {1 - ss_res / ss_tot:.2f}")
    ax.legend(frameon=False, fontsize=7)
    return fig, ax


def plot_decoding_error_over_time(error, dt=0.01, ax=None, label=None, color="crimson",
                                  log=True):
    """Decoding error over time (e.g. under reverse-phi)."""
    fig, ax = _ax(ax)
    t = np.arange(len(error)) * dt
    ax.plot(t, np.asarray(error).ravel(), color=color, lw=2, label=label)
    if log:
        ax.set_yscale("log")
    ax.set(xlabel="Time (s)", ylabel="Decoding error")
    if label:
        ax.legend(frameon=False, fontsize=7)
    return fig, ax
