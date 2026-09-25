"""Texture optimization of the input's radial 2-point (Hassenstein-Reichardt) correlation."""

import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

from ..analysis.illusion import two_point_torch
from ..config import DEFAULT_STIMULUS
from ..decoders import to_logits
from ..hex_eye import BoxEye
from ..models import forward_flynet
from ..rendering import generate_looming_stimulus
from ..textures import get_texture
from .common import save_json, save_texture_gray


def run_correlation_optimization(net, decoder, maximize=False, delays=(8,), steps=300, lr=1e-1,
                                 tex_res=128, texttype="Black", base_kwargs=None,
                                 out_dir="optimized_textures_correlations", save_every=10,
                                 pos_vel=None, device="cuda", extent=15):
    """Adam on the texture logits to min/maximise the input 2-point averaged over `delays`.

    The network is not in the loss: p(hit) (and position/velocity MSE with
    pos_vel=(pos_decoder, vel_decoder)) is only read out. net = (model1, idxs1,
    model2, idxs2) of a flyvis network. Saves `{texttype}_illusion_{step:04d}.pt`
    and `{texttype}_loss_history.json` in out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)
    kw = dict(base_kwargs or DEFAULT_STIMULUS, device=device, return_rgb=False)
    renderer = BoxEye(extent, kernel_size=13)
    x = to_logits(get_texture(texttype, tex_res).to(device).clamp(0, 1)).detach().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=lr, eps=1e-16)
    sign = 1.0 if maximize else -1.0
    history = []

    def snapshot(step):
        torch.save(x.detach().cpu(), f"{out_dir}/{texttype}_illusion_{step:04d}.pt")
        save_texture_gray(x, f"{out_dir}/{texttype}_illusion_{step:04d}_gray.png")

    snapshot(0)
    for step in tqdm(range(1, steps + 1)):
        frames, _, _, position = generate_looming_stimulus(texture=torch.sigmoid(x), **kw)
        hex_frames = renderer(frames[None]).to(device)
        corr = torch.stack([two_point_torch(hex_frames[0, :, 0], D) for D in delays]).mean()
        loss = -sign * corr * 1e5
        opt.zero_grad()
        loss.backward()
        opt.step()

        with torch.no_grad():
            act = forward_flynet(hex_frames.detach(), *net).flatten(end_dim=1)
            logits = decoder(act)
            row = {"step": step, "loss": float(loss), "correlation": float(corr),
                   "p_hit": torch.sigmoid(logits).mean().item(),
                   "accuracy_hit": (logits > 0).float().mean().item()}
            if pos_vel is not None:
                pos = position.to(act)
                vel = torch.gradient(pos, dim=1)[0]
                row["mse_pos"] = F.mse_loss(pos_vel[0](act)[10:], pos.T[10:]).item()
                row["mse_vel"] = F.mse_loss(pos_vel[1](act)[10:], vel.T[10:]).item()
        history.append(row)
        if step % save_every == 0 or step == steps:
            snapshot(step)
            save_json(history, f"{out_dir}/{texttype}_loss_history.json")
    return x.detach().cpu(), history
