"""Evaluate networks under video perturbations as an attack.

Sweeps a perturbation magnitude, decodes p(hit) over the sweep, and returns
per-magnitude accuracy plus the robustness R (area under the accuracy curve).
"""

import numpy as np
import torch

from . import config
from .config import DEFAULT_STIMULUS
from .decoders import to_logits
from .feature_extraction import extract_features
from .hex_eye import BoxEye
from .metrics import lers, robustness
from .models import forward_flynet
from .perturbations import texture_to_input_perturbed
from .textures import get_texture

# Perturbation name -> (kwarg name expected by texture_to_input_perturbed, mode).
PERTURBATIONS = {
    "AWN": ("awn", "sigma"), "Shot_noise": ("shot", "photon_count"),
    "Flicker": ("flicker", "strength"), "Motion_jitter": ("jitter", "jitter"),
    "Fog": ("fog", "contrast_scale"),
}


def _decode(inp, net, decoder, network_id):
    if not config.is_cnn(network_id):
        model1, idxs1, model2, idxs2 = net
        activity = forward_flynet(inp, model1, idxs1, model2, idxs2).flatten(end_dim=1)
    else:
        _, activity, _ = extract_features(net, inp, network_id=network_id)
        activity = activity.view(activity.shape[0], -1)
    return torch.sigmoid(decoder(activity))


def evaluate_perturbation(net, decoder, network_id, perturbation, magnitudes,
                          texttype="Black", base_kwargs=None, device="cuda", extent=15,
                          n_trials=1, seed=0):
    """Accuracy vs magnitude and robustness R for one perturbation.

    Returns dict with magnitudes, accuracy, robustness, lers.
    """
    assert perturbation in PERTURBATIONS, f"unknown perturbation {perturbation}"
    kw_name, _ = PERTURBATIONS[perturbation]
    base_kwargs = base_kwargs or dict(DEFAULT_STIMULUS, device=device)
    renderer_hex = BoxEye(extent, kernel_size=13)
    x_logits = to_logits(get_texture(texttype).to(device))

    acc = np.zeros(len(magnitudes))
    for j, mag in enumerate(magnitudes):
        trials = []
        for t in range(n_trials):
            inp, _ = texture_to_input_perturbed(
                x_logits, base_kwargs, renderer_hex, network_id=network_id,
                perturbation=(None if mag == 0 else kw_name), seed=seed + t, **{kw_name: mag})
            p_hit = _decode(inp, net, decoder, network_id)
            trials.append((p_hit > 0.5).float().mean().item())  # hit trials -> correct = hit
        acc[j] = float(np.mean(trials))
    R = robustness(magnitudes, acc)
    return {"magnitudes": np.asarray(magnitudes), "accuracy": acc,
            "robustness": R, "lers": lers(R, network_id)}
