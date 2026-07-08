"""Shared utilities for adversarial attacks (forward, IO)."""

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from ..feature_extraction import extract_features
from ..models import forward_flynet
from ..perturbations import texture_to_hex_batch


def build_out_dir(root, texttype, cnn, norm, eps, cnn_id="", run_id=None):
    """Structured output directory: root/texttype/model/norm/eps/run_id."""
    model_name = "cnn_" + cnn_id if cnn else cnn_id
    eps_str = f"eps_{eps:.0e}" if eps < 1e-2 else f"eps_{eps:g}"
    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(root) / texttype / model_name / norm / eps_str / f"run_{run_id}"
    (out_dir / "snapshots").mkdir(parents=True, exist_ok=True)
    return out_dir


def decode_logits(tex_logits, models, cnn, cnn_id, base_kwargs, renderer_hex, decoder,
                  n_frames=2, amp_dtype=torch.float16):
    """Texture logits -> decoder logits, through the appropriate network."""
    hex_batch = texture_to_hex_batch(tex_logits, base_kwargs, renderer_hex, cnn_id=cnn_id)
    if not cnn:
        model1, model2, idxs1, idxs2, _ = models
        two_model = not ("no_connectome" in cnn_id or "synthesized" in cnn_id or model2 is None)
        activity = forward_flynet(hex_batch, model1, idxs1,
                                  model2 if two_model else None,
                                  idxs2 if two_model else None).flatten(end_dim=1)
    else:
        model, _, _ = models
        if "raft" in cnn_id:
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                _, activity, _ = extract_features(model, hex_batch, n_frames=2, cnn_id=cnn_id)
        else:
            _, activity, _ = extract_features(model, hex_batch, n_frames=n_frames, cnn_id=cnn_id)
        activity = activity.view(activity.shape[0], -1)
    return decoder(activity)


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
