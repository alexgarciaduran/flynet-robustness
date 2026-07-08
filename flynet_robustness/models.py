"""FlyNet / CTRNN loading and forward passes (flyvis)."""

import numpy as np
import torch
from flyvis import EnsembleView, NetworkView
from flyvis.network.stimulus import Stimulus

from .config import MOTION_CELL_TYPES
from .hex_eye import BoxEye
from .rendering import generate_looming_stimulus


def _motion_indices(connectome, cell_types=MOTION_CELL_TYPES):
    return np.concatenate([connectome.nodes.layer_index[ct.decode()]
                           for ct in connectome.unique_cell_types
                           if ct.decode() in cell_types])


def get_flynet(network_id="flynet_synthesized", device="cuda"):
    """Load a FlyNet/CTRNN network and its decoded T4/T5 indices.

    Returns (model1, model2, idxs1, idxs2). model2/idxs2 are None except for
    the two-model ensemble variant ('flynet').
    """
    ckpt = {"validation_subdir": "validation", "loss_file_name": "loss"}
    if network_id == "flynet_no_connectome":
        model1 = NetworkView("flow/0003/000", best_checkpoint_fn_kwargs=ckpt).init_network().to(device)
        return model1, None, _motion_indices(model1.connectome), None
    if network_id == "flynet_synthesized":
        model1 = NetworkView("flow/0000/synthesized_t4d",
                             best_checkpoint_fn_kwargs=ckpt).init_network().to(device)
        return model1, None, _motion_indices(model1.connectome), None

    # Two-model ensemble: best overall + best-T4d cluster model.
    model1 = NetworkView("flow/0000/000", best_checkpoint_fn_kwargs=ckpt).init_network().to(device)
    clustering = EnsembleView("flow/0000").clustering("T4d")
    label = str(np.arange(50)[clustering.labels == 0][0])
    model2 = NetworkView(f"flow/0000/0{label}",
                         best_checkpoint_fn_kwargs=ckpt).init_network().to(device)
    base = ["T4a", "T4b", "T4c", "T5a", "T5b", "T5c", "T5d"]
    return model1, model2, _motion_indices(model1.connectome, base), \
        _motion_indices(model2.connectome, ["T4d"])


def forward_flynet(stimulus, model1, idxs1, model2=None, idxs2=None, fps=100, dt=0.01):
    """Run one or two networks on a hex stimulus; concatenate decoded activity."""
    B = stimulus.shape[0]
    state1 = model1.steady_state(t_pre=1.0, dt=dt, batch_size=B)
    model1.stimulus.zero()
    stim1 = Stimulus(model1.connectome, B, fps)
    stim1.add_input(stimulus)
    act1 = model1(stim1(), dt=dt, state=state1)[:, :, idxs1]
    if model2 is None:
        return act1
    state2 = model2.steady_state(t_pre=1.0, dt=dt, batch_size=B)
    model2.stimulus.zero()
    stim2 = Stimulus(model2.connectome, B, fps)
    stim2.add_input(stimulus)
    act2 = model2(stim2(), dt=dt, state=state2)[:, :, idxs2]
    return torch.cat((act1, act2), dim=-1)


def get_stimulus_and_return_activity(model1, idxs1, model2=None, idxs2=None, *,
                                     hit=False, radius=5.0, theta=0.0, shape="Sphere",
                                     texture=None, time=2.0, fps=100, stop_margin_hit=0.5,
                                     device="cuda", z0=15.0, image_size=416, dt=0.01,
                                     rotation=0.0):
    """Generate a looming stimulus, hex-render it and return network activity."""
    vis, speed, angle, position = generate_looming_stimulus(
        hit=hit, radius=radius, theta=theta, shape=shape, texture=texture, time=time,
        fps=fps, stop_margin_hit=stop_margin_hit, device=device, z0=z0,
        out_size=image_size, rotation=rotation)
    renderer = BoxEye(model1.connectome.config.extent, kernel_size=13)
    hex_frames = renderer(vis[None].detach().cpu().numpy()).to(device)
    activity = forward_flynet(hex_frames, model1, idxs1, model2, idxs2, fps=fps, dt=dt)
    return activity.cpu(), speed, angle, position
