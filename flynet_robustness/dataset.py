"""Generate the looming dataset (network activity + positions) from configs."""

import os

import numpy as np
import torch
from tqdm import tqdm

from .models import get_stimulus_and_return_activity
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
                     out_dir="simulations", device="cpu", out_size=416, test=False):
    """Render the fixed trajectory set, run the network and save activity + positions."""
    np.random.seed(1 if test else 0)
    data = torch.load(config_path, weights_only=False)
    os.makedirs(out_dir, exist_ok=True)

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

    suffix = "_test" if test else ""
    torch.save({"network_activity_all": activity_all}, f"{out_dir}/activity{suffix}.pt")
    torch.save({"positions": positions}, f"{out_dir}/positions{suffix}.pt")
    return activity_all, positions


def prepare_dataset(activity_all, position_all, t_start=10):
    """Concatenate trials into (samples, neurons) and (samples, 3) tensors."""
    X = torch.cat([a[0][t_start:] for a in activity_all], dim=0)
    Y = torch.cat([p.T[t_start:] for p in position_all], dim=0)
    return X, Y
