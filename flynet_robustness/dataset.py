"""
@author: alexgarciaduran
"""

import glob
import json
import os

import numpy as np
import torch
from tqdm import tqdm

from . import config
from .hex_eye import BoxEye
from .models import forward_flynet
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


def network_activity(net, network_id="FlyNet", *, hit=False, radius=5.0, theta=0.0,
                     shape="Sphere", texture=None, time=2.0, fps=100,
                     stop_margin_hit=0.5, device="cuda", z0=15.0, rotation=0.0,
                     image_size=416, dt=0.01, renderer_hex=None):
    """Decoder-input activity for one stimulus, for any registry id.

    Returns (activity, position); activity is (samples, features) on the CPU.
    CNN/RAFT read windows of n_frames, so they give T - n_frames + 1 samples.
    """
    from .feature_extraction import extract_features

    # Cartesian RAFT takes RGB frames, everything else the hex input.
    rgb = config.kind(network_id) == "raft"
    out = generate_looming_stimulus(
        hit=hit, radius=radius, theta=theta, shape=shape, texture=texture, time=time,
        fps=fps, stop_margin_hit=stop_margin_hit, device=device, z0=z0,
        out_size=image_size, rotation=rotation, return_rgb=rgb)
    if rgb:
        vis, frames_rgb, _, _, position = out
    else:
        vis, _, _, position = out

    with torch.no_grad():
        if rgb:
            model_input = frames_rgb
        else:
            renderer_hex = renderer_hex or BoxEye(15, kernel_size=13)
            model_input = renderer_hex(vis[None].detach()).to(device)
        if config.is_cnn(network_id):
            _, activity, _ = extract_features(net, model_input, network_id=network_id)
            activity = activity.reshape(activity.shape[0], -1)
        else:
            model1, idxs1, model2, idxs2 = net
            activity = forward_flynet(model_input, model1, idxs1, model2, idxs2,
                                      fps=fps, dt=dt)[0]
    return activity.cpu(), position


def save_simulations(net, network_id="FlyNet", config_path=None, out_dir="simulations",
                     device="cuda", out_size=416, test=False, save_every=500):
    """Render the trajectory set, save each trial's activity + position.

    Dumps every `save_every` trials, so a killed job can be restarted.
    """
    np.random.seed(1 if test else 0)
    data = torch.load(config_path, weights_only=False)
    os.makedirs(out_dir, exist_ok=True)
    suffix = "_test" if test else ""
    renderer_hex = BoxEye(15, kernel_size=13)

    def dump():
        torch.save({"network_activity_all": activity_all}, f"{out_dir}/activity{suffix}.pt")
        torch.save({"positions": positions}, f"{out_dir}/positions{suffix}.pt")

    activity_all, positions = [], []
    for sim in tqdm(range(len(data["shape_all"]))):
        seed_everything(seed=sim)
        shape = data["shape_all"][sim]
        activity, position = network_activity(
            net, network_id, hit=data["true_labels"][sim],
            radius=data["radius_conditions"][sim], theta=data["theta_conditions"][sim],
            shape=shape, texture=get_texture(data["textures_all"][sim]),
            time=data["time_duration_conditions"][sim], device=device,
            z0=data["z0_conditions"][sim], rotation=data["rotation_conditions"][sim],
            stop_margin_hit=_STOP_MARGIN[shape], image_size=out_size,
            renderer_hex=renderer_hex)
        activity_all.append(activity)
        positions.append(position)
        if save_every and sim and sim % save_every == 0:
            dump()

    dump()
    return activity_all, positions


def prepare_dataset(activity_all, position_all, t_start=10, network_id="FlyNet"):
    """Concatenate trials into (samples, features) and (samples, 3).

    Positions are truncated to the number of samples the network emitted.
    """
    n_frames = config.n_frames(network_id)
    X_list, Y_list = [], []
    for activity, pos in zip(activity_all, position_all):
        a = activity[0] if activity.ndim == 3 else activity
        Y = pos.T
        if n_frames:
            Y = Y[:Y.shape[0] - n_frames + 1]
        X_list.append(a[t_start:])
        Y_list.append(Y[t_start:])
    return torch.cat(X_list, dim=0), torch.cat(Y_list, dim=0)


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
