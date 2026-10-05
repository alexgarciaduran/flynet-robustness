"""Generate the looming dataset and the illusion-analysis simulations."""

import glob
import json
import os

import numpy as np
import torch
from tqdm import tqdm

from .hex_eye import BoxEye
from .models import forward_flynet, get_stimulus_and_return_activity
from .rendering import generate_looming_stimulus
from .reverse_phi import reverse_phi
from .textures import get_texture

_STOP_MARGIN = {"Sphere": 0.6, "Cube": 1.0, "Tetrahedron": 0.0}


def seed_everything(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_simulations(model1, idxs1, model2=None, idxs2=None, config_path=None,
                     out_dir="simulations", device="cpu", out_size=416, test=False,
                     save_every=500):
    """Render the fixed trajectory set, run the network and save activity + positions.

    Writes partial results every `save_every` trials so a killed job is resumable.
    """
    np.random.seed(1 if test else 0)
    data = torch.load(config_path, weights_only=False)
    os.makedirs(out_dir, exist_ok=True)
    suffix = "_test" if test else ""

    def dump():
        torch.save({"network_activity_all": activity_all}, f"{out_dir}/activity{suffix}.pt")
        torch.save({"positions": positions}, f"{out_dir}/positions{suffix}.pt")

    activity_all, positions = [], []
    for sim in tqdm(range(len(data["shape_all"]))):
        seed_everything(seed=sim)
        shape = data["shape_all"][sim]
        with torch.no_grad():
            activity, _, _, position = get_stimulus_and_return_activity(
                model1, idxs1, model2, idxs2, hit=data["true_labels"][sim],
                radius=data["radius_conditions"][sim], theta=data["theta_conditions"][sim],
                shape=shape, texture=get_texture(data["textures_all"][sim]),
                time=data["time_duration_conditions"][sim], device=device,
                z0=data["z0_conditions"][sim], rotation=data["rotation_conditions"][sim],
                stop_margin_hit=_STOP_MARGIN[shape], image_size=out_size)
        activity_all.append(activity.cpu())
        positions.append(position)
        if save_every and sim and sim % save_every == 0:
            dump()

    dump()
    return activity_all, positions


def prepare_dataset(activity_all, position_all, t_start=10):
    """Concatenate trials into (samples, neurons) and (samples, 3) tensors."""
    X = torch.cat([a[0][t_start:] for a in activity_all], dim=0)
    Y = torch.cat([p.T[t_start:] for p in position_all], dim=0)
    return X, Y


# --- Illusion analyses: counterfactual trajectories -> condition simulations ------

CONDITION_STEPS = {"original": 0, "illusion": 250, "recovered": 500, "reverse_phi": 0}
_NOT_STIMULUS = ("text_type", "lambda_tex_base", "lambda_nc", "return_rgb", "mode")


def simulate_texture(model1, idxs1, tex_logits, base_kwargs, T_flip=None, mode="avg_pool_adaptive", dt=0.01):
    """Render texture logits on a trajectory (reverse-phi every T_flip frames if set), run the net.

    Returns numpy (activity (T, len(idxs1)), hex input (T, 721)).
    """
    kw = {k: v for k, v in base_kwargs.items() if k not in _NOT_STIMULUS}
    with torch.no_grad():
        frames = generate_looming_stimulus(texture=torch.sigmoid(tex_logits), mode=mode, **kw)[0]
        if T_flip:
            frames = reverse_phi(frames, T_flip)
        hex_frames = BoxEye(model1.connectome.config.extent, kernel_size=13)(frames[None])
        act = forward_flynet(hex_frames, model1, idxs1, dt=dt)[0]
    return act.cpu().numpy(), hex_frames[0, :, 0].cpu().numpy()


def save_condition_simulations(model1, idxs1, traj_root, out_root, T_flip=33, steps=CONDITION_STEPS):
    """Original / illusion / recovered / reverse-phi simulations of every counterfactual trajectory."""
    for folder in tqdm(sorted(glob.glob(f"{traj_root}/optimal_trajectories_fooling_*"))):
        n = int(folder.rsplit("_", 1)[1])
        cfg = json.load(open(f"{folder}/config.json"))
        tex, shape = cfg["text_type"], cfg["shape"]
        for cond, step in steps.items():
            x = torch.load(f"{folder}/{tex}_illusion_{step:04d}.pt", weights_only=False)
            act, hexf = simulate_texture(model1, idxs1, x, cfg, T_flip if cond == "reverse_phi" else None)
            d = f"{out_root}/{cond}/{shape}"
            os.makedirs(f"{d}/stimulus", exist_ok=True)
            np.save(f"{d}/simulation_{n}_{tex}_responses_t4t5_{shape}_{cond}.npy", act)
            np.save(f"{d}/stimulus/input_stimulus_{n}_{tex}__{shape}_{cond}.npy", hexf)
