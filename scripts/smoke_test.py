"""Smoke test for dependency-light parts of the package.

Runs without flyvis / PyTorch3D / RAFT. Exercises textures, metrics,
perturbation ops and the neural-analysis functions.
"""

import numpy as np
import torch

import flynet_robustness as fr
from flynet_robustness.analysis import locality, recurrence, representational
from flynet_robustness.perturbations import (
    additive_white_noise,
    apply_fog,
    flicker,
    motion_jitter,
    shot_noise,
)


def main():
    for t in fr.TEXTURE_TYPES:
        x = fr.get_texture(t, size=32)
        assert x.shape == (1, 32, 32, 1) and 0 <= x.min() and x.max() <= 1

    eps = [1, 2, 3, 4, 5]
    acc = [1.0, 0.9, 0.5, 0.3, 0.1]
    R = fr.robustness(eps, acc)
    assert 0 <= R <= 1
    assert fr.lers(R, "flynet_synthesized") > 0

    x = torch.rand(4, 5, 1, 721)
    for op in (lambda a: additive_white_noise(a, 0.1),
               lambda a: shot_noise(a, 30),
               lambda a: apply_fog(a, contrast_scale=0.5),
               lambda a: flicker(a, 0.05)):
        assert op(x).shape == x.shape
    rgb = torch.rand(8, 16, 16, 3)
    assert motion_jitter(rgb, 3, seed=1)[0].shape == rgb.shape

    W = np.zeros((4, 4))
    W[0, 1] = W[1, 2] = W[2, 0] = 0.5
    assert recurrence.recurrence(W, n_samples=4)[1]["n_cycles"] >= 1
    assert recurrence.recurrence_weighted(W, n_samples=4)["frac_on_cycle"] > 0
    assert set(recurrence.recurrence_metrics(W)) >= {"self_loop_rate", "largest_scc"}

    pos = np.random.rand(20, 2)
    assert 0 <= locality.compute_functional_locality(lambda a: a * 2, pos, pos,
                                                     n_probe_neurons=10)["locality"] <= 1

    acts = [np.random.rand(8, 6) for _ in range(4)]
    pop, neu = representational.trial_similarity(acts)
    assert pop.shape == (4, 4) and neu.shape == (6, 4, 4)

    print("Smoke test passed.")


if __name__ == "__main__":
    main()
