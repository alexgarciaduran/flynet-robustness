"""
@author: alexgarciaduran
"""

import json
import pickle
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .. import config
from ..feature_extraction import extract_features
from ..hex_eye import BoxEye
from ..metrics import transfer_auc_matrix, transfer_target_accuracy  # noqa: F401  (re-exported)
from ..models import forward_flynet
from ..perturbations import texture_to_hex_batch
from .common import run_dir


def load_adversarial_texture(source_id, norm, eps, run_id, texttype="Black", root="PGD",
                             device="cuda"):
    """(texture logits, stimulus kwargs) of a PGD run's best restart."""
    d = run_dir(root, texttype, source_id, norm, eps, run_id, existing=True)
    best = d / "best_texture.pt"
    if not best.exists():
        best = _best_restart_snapshot(d)
    x = torch.load(best, map_location=device, weights_only=False)
    with open(d / "config.json", encoding="utf-8") as f:
        base_kwargs = json.load(f)
    return x, base_kwargs


def _best_restart_snapshot(d):
    """Last snapshot of the restart with the highest final label-flip rate."""
    history = torch.load(d / "loss_history.pt", map_location="cpu", weights_only=False)
    last = {e.get("restart", 0): e for e in history}
    restart = max(last, key=lambda r: last[r]["avg_flip_correct_initially"])
    snaps = sorted((d / "snapshots" / f"restart_{restart:02d}").glob("step_*.pt"))
    if not snaps:                                   # layout without restart folders
        snaps = sorted((d / "snapshots").glob("step_*.pt"))
    if not snaps:
        raise FileNotFoundError(f"No saved texture under {d}")
    return snaps[-1]


def evaluate_texture(tex_logits, base_kwargs, net, decoder, network_id, extent=15):
    """Fraction of frames the target network still classifies as hit."""
    renderer_hex = BoxEye(extent, kernel_size=13)
    with torch.inference_mode():
        inp = texture_to_hex_batch(tex_logits, base_kwargs, renderer_hex, network_id=network_id)
        if config.is_cnn(network_id):
            _, activity, _ = extract_features(net, inp, network_id=network_id)
            activity = activity.reshape(activity.shape[0], -1)
        else:
            model1, idxs1, model2, idxs2 = net
            activity = forward_flynet(inp, model1, idxs1, model2, idxs2).flatten(end_dim=1)
        return float((decoder(activity) > 0).float().mean())


def transfer_matrix(nets, eps_list, norm="l2", attacks=15, texttype="Black", root="PGD",
                    device="cuda", network_ids=None):
    """Accuracy of every target under every source's PGD textures.

    `nets`: {id: (net, decoder)}. Returns (ids, accuracy), accuracy shaped
    (sources, targets, eps, attacks) with NaN on the diagonal.
    """
    ids = list(network_ids or nets)
    acc = np.full((len(ids), len(ids), len(eps_list), attacks), np.nan)
    for i, source in enumerate(ids):
        for e, eps in enumerate(eps_list):
            for a in tqdm(range(attacks), desc=f"{source} {norm} eps={eps:g}", leave=False):
                try:
                    x, base_kwargs = load_adversarial_texture(
                        source, norm, eps, a + 1, texttype, root, device)
                except FileNotFoundError:
                    continue
                base_kwargs = dict(base_kwargs, device=device)
                for j, target in enumerate(ids):
                    if target == source:
                        continue
                    net, decoder = nets[target]
                    acc[i, j, e, a] = evaluate_texture(x, base_kwargs, net, decoder, target)
    return ids, acc


def save_transfer(path, ids, acc, eps_list, norm):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"network_ids": ids, "accuracy": acc,
                     "eps_list": list(eps_list), "norm": norm}, f)
    return path


def load_transfer(path):
    with open(path, "rb") as f:
        return pickle.load(f)
