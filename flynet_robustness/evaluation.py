"""
@author: alexgarciaduran
"""

import numpy as np
import torch

from . import config
from .config import DEFAULT_STIMULUS
from .decoders import to_logits
from .feature_extraction import extract_features
from .hex_eye import BoxEye
from .metrics import bootstrap_robustness, lers, robustness
from .models import forward_flynet
from .perturbations import texture_to_input_perturbed
from .textures import get_texture

# Trajectories averaged over; fog is deterministic, so it gets more.
N_TRAJECTORIES = {"Fog": 20}
DEFAULT_N_TRAJECTORIES = 10

# name -> (mode, magnitude kwarg, identity magnitude, magnitudes)
PERTURBATIONS = {
    "AWN_texture": ("awn_texture", "sigma", 0, (0, 0.1, 0.2, 0.4, 0.6, 0.8)),
    "AWN_frames": ("awn", "sigma", 0, (0, 5e-3, 1e-2, 2e-2, 3e-2, 4e-2, 5e-2)),
    "Shot_noise": ("shot", "photon_count", None, (1e4, 5e3, 1e3, 5e2, 1e2, 5e1, 2e1, 1e1)),
    "Flicker": ("flicker", "strength", 0, (0, 0.01, 0.02, 0.04, 0.05)),
    "Motion_jitter": ("jitter", "jitter", 0, (0, 1, 2, 3, 4, 5)),
    "Fog": ("fog", "contrast_scale", 1, (1, 0.5, 0.1, 0.05, 0.02, 0.01)),
}


def _logits(inp, net, decoder, network_id):
    if not config.is_cnn(network_id):
        model1, idxs1, model2, idxs2 = net
        activity = forward_flynet(inp, model1, idxs1, model2, idxs2).flatten(end_dim=1)
    else:
        _, activity, _ = extract_features(net, inp, network_id=network_id)
        activity = activity.view(activity.shape[0], -1)
    return decoder(activity)


def evaluate_perturbation(net, decoder, network_id, perturbation, magnitudes=None,
                          texttype="Black", base_kwargs=None, device="cuda", extent=15,
                          n_trials=10, seed=0):
    """Accuracy vs magnitude and R, for one perturbation on one trajectory.

    Noise draw t uses torch seed `seed + t`, so all networks see the same
    noise; fog is deterministic and gets one draw.
    """
    mode, param, identity, default = PERTURBATIONS[perturbation]
    magnitudes = np.asarray(default if magnitudes is None else magnitudes, dtype=float)
    base_kwargs = base_kwargs or dict(DEFAULT_STIMULUS, device=device)
    renderer_hex = BoxEye(extent, kernel_size=13)
    x_logits = to_logits(get_texture(texttype).to(device))
    n_ratio = config.input_dim(network_id) / config.input_dim("FlyNet")
    unit = {"awn": n_ratio ** -0.5, "shot": n_ratio}.get(mode, 1.0)

    def logits(mag=None, t=0):
        torch.manual_seed(seed + t)
        pert = None if mag is None or mag == identity else mode
        inp, _ = texture_to_input_perturbed(x_logits, base_kwargs, renderer_hex, network_id=network_id,
                                            perturbation=pert, seed=seed + t,
                                            **({} if mag is None else {param: mag * unit}))
        return _logits(inp, net, decoder, network_id)

    with torch.no_grad():
        correct = logits() > 0
        acc = np.array([np.mean([1 - ((logits(m, t) <= 0) & correct).sum().item() / max(correct.sum().item(), 1)
                                 for t in range(1 if m == identity or mode == "fog" else n_trials)])
                        for m in magnitudes])
    R = robustness(magnitudes, acc)
    return {"magnitudes": magnitudes, "accuracy": acc, "robustness": R, "lers": lers(R, network_id)}


def sweep_perturbation(net, decoder, network_id, perturbation, trajectories,
                       magnitudes=None, n_trajectories=None, device="cuda", seed=0, **kwargs):
    """Run one perturbation over several trajectories and bootstrap R.

    `trajectories`: dicts of radius/theta/z0. Trajectory i seeds its noise with
    `seed + 1000 * i`: different per trajectory, identical across networks.
    """
    n = n_trajectories or N_TRAJECTORIES.get(perturbation, DEFAULT_N_TRAJECTORIES)
    magnitudes = np.asarray(PERTURBATIONS[perturbation][3] if magnitudes is None else magnitudes,
                            dtype=float)
    acc = np.stack([
        evaluate_perturbation(net, decoder, network_id, perturbation, magnitudes=magnitudes,
                              base_kwargs=dict(DEFAULT_STIMULUS, **t, device=device),
                              device=device, seed=seed + 1000 * i, **kwargs)["accuracy"]
        for i, t in enumerate(trajectories[:n])], axis=1)
    R, R_std = bootstrap_robustness(magnitudes, acc)
    return {"magnitudes": magnitudes, "accuracy": acc, "robustness": R, "robustness_std": R_std,
            "lers": lers(R, network_id), "lers_std": lers(R_std, network_id),
            "n_trajectories": acc.shape[1]}
