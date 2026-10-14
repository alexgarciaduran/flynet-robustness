"""
@author: alexgarciaduran
"""

import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

from ..config import DEFAULT_STIMULUS
from ..decoders import to_logits
from ..hex_eye import BoxEye
from ..rendering import generate_looming_stimulus
from ..textures import get_texture
from .common import amp_dtype, decode_logits, save_json, save_texture_gray


def _optimise(w_init, anchor, target, net, decoder, network_id, base_kwargs,
              renderer_hex, steps, lr, lambda_anchor, log=None):
    """Optimise cross-entropy to `target` with an L2 anchor to `anchor` (both texture logits)."""
    w = w_init.clone().requires_grad_(True)
    opt = torch.optim.Adam([w], lr=lr)
    ce = torch.nn.BCEWithLogitsLoss()
    adtype = amp_dtype()
    history = []
    for _ in tqdm(range(steps)):
        logits, activity = decode_logits(w, net, decoder, network_id, base_kwargs, renderer_hex,
                                         adtype, return_activity=True)
        tgt = torch.full_like(logits, float(target))
        loss = ce(logits, tgt) + lambda_anchor * (w - anchor).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        history.append(float(loss.detach()))
        if log is not None:
            with torch.no_grad():
                log(w, logits.detach(), activity.detach(), history[-1])
    return w.detach(), history


def counterfactual_restoration(net, decoder, network_id="FlyNet", steps=250, lr=1e-1,
                               tex_res=128, lambda_base=0.02, texttype="Black",
                               base_kwargs=None, device="cuda", extent=15,
                               out_dir=None, save_every=10, pos_vel=None):
    """Return (illusion_texture, restored_texture) and their loss histories.

    With out_dir, saves the trajectory read by the illusion analyses: config.json,
    `{texttype}_illusion_{step:04d}.pt` every save_every steps (0 = base, steps =
    illusion, 2*steps = restored) and `{texttype}_loss_history.json` with p(hit),
    plus position/velocity MSE if pos_vel=(pos_decoder, vel_decoder) (flyvis nets).
    """
    base_kwargs = base_kwargs or dict(DEFAULT_STIMULUS, device=device)
    renderer_hex = BoxEye(extent, kernel_size=13)
    base = get_texture(texttype, tex_res).to(device).clamp(0, 1)
    w_base = to_logits(base).detach()
    tex_base = torch.sigmoid(w_base)

    log, rows = None, []
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        save_json(dict(base_kwargs, text_type=texttype, lambda_tex_base=lambda_base),
                  f"{out_dir}/config.json")
        position = generate_looming_stimulus(texture=tex_base, **base_kwargs)[-1] if pos_vel else None

        def snapshot(w, step):
            torch.save(w.detach().cpu(), f"{out_dir}/{texttype}_illusion_{step:04d}.pt")
            save_texture_gray(w, f"{out_dir}/{texttype}_illusion_{step:04d}_gray.png")

        def _log(w, logits, activity, loss):
            row = {"step": len(rows) + 1, "loss": loss, "p_hit": torch.sigmoid(logits).mean().item()}
            if pos_vel is not None:
                pos = position.to(activity)
                vel = torch.gradient(pos, dim=1)[0]
                row["mse_pos"] = F.mse_loss(pos_vel[0](activity)[10:], pos.T[10:]).item()
                row["mse_vel"] = F.mse_loss(pos_vel[1](activity)[10:], vel.T[10:]).item()
            rows.append(row)
            if row["step"] % save_every == 0:
                snapshot(w, row["step"])
                save_json(rows, f"{out_dir}/{texttype}_loss_history.json")

        log = _log
        snapshot(w_base, 0)

    illusion, hist1 = _optimise(w_base, w_base, 0, net, decoder, network_id, base_kwargs,
                                renderer_hex, steps, lr, lambda_base, log)
    restored, hist2 = _optimise(illusion, illusion, 1, net, decoder, network_id, base_kwargs,
                                renderer_hex, steps, lr, lambda_base, log)
    if out_dir is not None:
        save_json(rows, f"{out_dir}/{texttype}_loss_history.json")
    return illusion, restored, hist1, hist2
