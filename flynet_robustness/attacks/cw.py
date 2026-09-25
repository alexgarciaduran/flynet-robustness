"""Carlini-Wagner attack on the object texture."""

import torch
from tqdm import tqdm

from ..config import DEFAULT_STIMULUS
from ..decoders import to_logits
from ..hex_eye import BoxEye
from ..textures import get_texture
from .common import amp_dtype, build_out_dir, decode_logits, flip_rates, save_json, save_texture_gray


def _dist(tex_adv, tex_clean, norm):
    d = (tex_adv - tex_clean).view(tex_adv.shape[0], -1)
    if norm == "l2":
        return d.pow(2).sum(dim=1).mean()
    if norm == "linf":
        return d.abs().max(dim=1)[0].mean()
    raise ValueError("CW norm must be 'l2' or 'linf'")


def run_cw(net, decoder, network_id="FlyNet", steps=150, lr=1e-1, tex_res=128, c=1e-2,
           kappa=0.0, base_kwargs=None, save_every=10, texttype="Black", run_id=1,
           norm="l2", device="cuda", extent=15):
    """Minimise dist(tex, tex_clean) + c*max(logit+kappa, 0) (untargeted, clean=hit)."""
    out_dir = build_out_dir("CW", texttype, network_id, norm, c, run_id)
    base_kwargs = base_kwargs or dict(DEFAULT_STIMULUS, device=device)
    save_json(base_kwargs, out_dir / "config.json")

    w0 = to_logits(get_texture(texttype, tex_res).to(device).clamp(0, 1), eps=1e-2).detach()
    w = w0.clone().requires_grad_(True)
    tex_clean = torch.sigmoid(w0)
    renderer_hex = BoxEye(extent, kernel_size=13)
    opt = torch.optim.Adam([w], lr=lr)
    adtype = amp_dtype()

    def logits_of(x):
        return decode_logits(x, net, decoder, network_id, base_kwargs, renderer_hex, adtype)

    with torch.no_grad():
        correct = logits_of(w0) > 0

    loss_history = []
    for step in tqdm(range(1, steps + 1)):
        logits = logits_of(w)
        dist = _dist(torch.sigmoid(w), tex_clean, norm)
        loss = dist + c * torch.clamp(logits + kappa, min=0.0).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        with torch.no_grad():
            d = (torch.sigmoid(w) - tex_clean).flatten(1)
            size = (d.norm(dim=1) if norm == "l2" else d.abs().amax(1)).mean().item()
            loss_history.append({"step": step, "loss": float(loss.detach()), "dist": float(dist.detach()),
                                 "max_delta": size, **flip_rates(logits_of(w), correct)})
        if step % save_every == 0 or step == 1:
            save_texture_gray(w, out_dir / "snapshots" / f"step_{step:04d}_gray.png")
            torch.save(w.detach().cpu(), out_dir / "snapshots" / f"step_{step:04d}.pt")

    torch.save(loss_history, out_dir / "loss_history.pt")
    return out_dir
