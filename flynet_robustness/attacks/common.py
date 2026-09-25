"""Shared utilities for adversarial attacks (forward, IO)."""

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .. import config
from ..feature_extraction import extract_features
from ..models import forward_flynet
from ..perturbations import texture_to_hex_batch


def _eps_dir(eps):
    return f"eps_{eps:.0e}" if eps < 1e-2 else f"eps_{eps:g}"


def build_out_dir(root, texttype, network_id, norm, eps, run_id=None):
    """Structured output directory: root/texttype/network/norm/eps/run_id."""
    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(root) / texttype / network_id / norm / _eps_dir(eps) / f"run_{run_id}"
    (out_dir / "snapshots").mkdir(parents=True, exist_ok=True)
    return out_dir


def final_values(root, texttype, network_id, norm, values, runs, key="avg_flip_correct_initially"):
    """(n_values, runs) final `key` of saved runs (best restart = highest final flip); NaN if missing."""
    out = np.full((len(values), runs), np.nan)
    for i, v in enumerate(values):
        for r in range(runs):
            f = Path(root) / texttype / network_id / norm / _eps_dir(v) / f"run_{r + 1}" / "loss_history.pt"
            if f.exists():
                last = {e.get("restart", 0): e for e in torch.load(f, map_location="cpu", weights_only=False)}
                out[i, r] = max(last.values(), key=lambda e: e["avg_flip_correct_initially"])[key]
    return out


def decode_logits(tex_logits, net, decoder, network_id, base_kwargs, renderer_hex,
                  amp_dtype=torch.float16, return_activity=False):
    """Texture logits -> decoder logits, through the appropriate network.

    net is (model1, idxs1, model2, idxs2) for FlyNet/CTRNN, else a CNN/RAFT
    feature-extractor module.
    """
    hex_batch = texture_to_hex_batch(tex_logits, base_kwargs, renderer_hex, network_id=network_id)
    if not config.is_cnn(network_id):
        model1, idxs1, model2, idxs2 = net
        activity = forward_flynet(hex_batch, model1, idxs1, model2, idxs2).flatten(end_dim=1)
    elif config.kind(network_id) in ("raft", "raft_hex"):
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            _, activity, _ = extract_features(net, hex_batch, network_id=network_id)
        activity = activity.view(activity.shape[0], -1)
    else:
        _, activity, _ = extract_features(net, hex_batch, network_id=network_id)
        activity = activity.view(activity.shape[0], -1)
    logits = decoder(activity)
    return (logits, activity) if return_activity else logits


def flip_rates(logits, correct):
    """Frames flipped to miss: overall, and among the frames initially classified as hit."""
    flipped = logits <= 0
    return {"label_flip": flipped.float().mean().item(),
            "avg_flip_correct_initially": (flipped & correct).sum().item() / max(correct.sum().item(), 1)}


def save_texture_gray(tex_logits, png_path):
    """Save the grayscale texture (sigmoid of logits) as a PNG."""
    with torch.no_grad():
        gray = torch.sigmoid(tex_logits)[0].mean(dim=-1).clamp(0, 1).cpu().numpy()
    plt.imsave(png_path, gray, cmap="gray", vmin=0, vmax=1)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def amp_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16
