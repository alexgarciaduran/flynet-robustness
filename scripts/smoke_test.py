"""Smoke test for dependency-light parts of the package.

Runs without flyvis / PyTorch3D / RAFT. Exercises the registry, textures,
metrics, perturbation ops, plotting, training and neural analyses.
"""

import warnings

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

import flynet_robustness as fr
from flynet_robustness import config as C
from flynet_robustness import plotting as P
from flynet_robustness.analysis import decoding, locality, recurrence, representational
from flynet_robustness.perturbations import (
    additive_white_noise, apply_fog, flicker, motion_jitter, shot_noise)
from flynet_robustness.training import train_hit_decoder, train_position_decoder


def main():
    # registry
    assert C.n_frames("vCNN_5_frames") == 5 and C.n_frames("vCNN_10_frames") == 10
    assert C.raft_iters("RAFT_6") == 6 and not C.is_cnn("FlyNet")
    assert C.n_frames("0") == 2 and C.label("raft_large") == "RAFT-12"  # legacy ids

    for t in fr.TEXTURE_TYPES:
        assert fr.get_texture(t, 32).shape == (1, 32, 32, 1)

    R = fr.robustness([1, 2, 3], [1.0, 0.5, 0.2])
    assert fr.lers(R, "FlyNet") > 0 and fr.lers(R, "flynet_synthesized") > 0

    x = torch.rand(4, 5, 1, 721)
    for op in (lambda a: additive_white_noise(a, 0.1), lambda a: shot_noise(a, 30),
               lambda a: apply_fog(a, contrast_scale=0.5), lambda a: flicker(a, 0.05)):
        assert op(x).shape == x.shape
    assert motion_jitter(torch.rand(8, 16, 16, 3), 3, seed=1)[0].shape == (8, 16, 16, 3)

    # plotting
    P.plot_p_hit_trace(np.linspace(1, 0, 100))
    P.plot_accuracy_curve([0, 1, 2], {"FlyNet": [1, .8, .5], "RAFT_12": [1, .9, .7]})
    P.plot_robustness_efficiency({"FlyNet": 0.9, "CTRNN": 0.95})
    P.plot_lers_bars({"FlyNet": 0.12, "vCNN_2_frames": 0.05})
    P.plot_transfer_matrix(np.clip(np.random.rand(3, 3) * .5 + .5, .5, 1),
                           ["FlyNet", "CTRNN", "RAFT_12"])
    P.plot_texture(torch.randn(1, 32, 32, 3))
    yt = np.random.rand(80, 3)
    P.plot_decoding_scatter(yt, yt + 0.1 * np.random.randn(80, 3))

    # analyses
    W = np.zeros((4, 4)); W[0, 1] = W[1, 2] = W[2, 0] = 0.5
    assert recurrence.recurrence(W, n_samples=4)[1]["n_cycles"] >= 1
    assert recurrence.recurrence_weighted(W, n_samples=4)["frac_on_cycle"] > 0
    pos = np.random.rand(20, 2)
    assert 0 <= locality.compute_functional_locality(lambda a: a * 2, pos, pos,
                                                     n_probe_neurons=10)["locality"] <= 1
    pop, neu = representational.trial_similarity([np.random.rand(8, 6) for _ in range(4)])
    assert pop.shape == (4, 4)
    assert decoding.r2_score(yt, yt + 0.01) > 0.9

    # training (synthetic, no save)
    acts = [torch.tensor(np.random.randn(1, 40, 20) + (2 if k % 2 else -2), dtype=torch.float32)
            for k in range(16)]
    labels = [k % 2 for k in range(16)]
    _, test_acc = train_hit_decoder(acts, labels, [0.4] * 16, decoder="mlp",
                                    scale=True, max_iter=30, save=False)
    assert test_acc > 0.5
    Xr, Yr = torch.randn(200, 20), torch.randn(200, 3)
    train_position_decoder(Xr, Yr, Xr[:20], Yr[:20], n_epochs=3, device="cpu", save=False)

    print("Smoke test passed.")


if __name__ == "__main__":
    main()
