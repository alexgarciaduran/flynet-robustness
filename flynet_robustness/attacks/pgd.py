"""
@author: alexgarciaduran
"""

import torch
from tqdm import tqdm

from ..config import DEFAULT_STIMULUS
from ..decoders import to_logits
from ..hex_eye import BoxEye
from ..textures import get_texture
from .common import amp_dtype, build_out_dir, decode_logits, flip_rates, save_json, save_texture_gray


def _random_start(x0, eps, norm):
    if eps <= 0:
        return x0.clone()
    if norm == "linf":
        return x0 + torch.empty_like(x0).uniform_(-eps, eps)
    eta = torch.randn_like(x0)
    eta = eta / eta.view(eta.size(0), -1).norm(dim=1).view(-1, 1, 1, 1).clamp(min=1e-12)
    return x0 + eps * eta


def _project(x, x0, eps, norm):
    if norm == "linf":
        return torch.max(torch.min(x, x0 + eps), x0 - eps)
    delta = x - x0
    dn = delta.view(delta.shape[0], -1).norm(dim=1).clamp(min=1e-12)
    factor = torch.min(torch.ones_like(dn), eps / dn).view(-1, *([1] * (x.dim() - 1)))
    return x0 + delta * factor


def run_pgd(net, decoder, network_id="FlyNet", steps=100, lr=1e-2, tex_res=128, eps=1e-2,
            base_kwargs=None, save_every=10, texttype="Black", run_id=1, norm="linf",
            device="cuda", extent=15, n_restarts=2, checkpoint_frac=0.2,
            alpha_min_factor=0.1, patience=50, rel_thresh=1e-3):
    """Optimise the texture to flip hit->miss, from `n_restarts` random starts.

    The step halves and rewinds to the best iterate every
    `checkpoint_frac * steps` steps, and the restart stops once the loss has
    moved by less than `rel_thresh` over `patience` entries (patience=None
    disables it). Writes per-restart snapshots, plus best_texture.pt
    (highest-flip restart), loss_history.pt and best.json.
    """
    out_dir = build_out_dir("PGD", texttype, network_id, norm, eps, run_id)
    base_kwargs = base_kwargs or dict(DEFAULT_STIMULUS, device=device)
    save_json(base_kwargs, out_dir / "config.json")

    x0 = to_logits(get_texture(texttype, tex_res).to(device).clamp(0, 1)).detach()
    renderer_hex = BoxEye(extent, kernel_size=13)
    criterion = torch.nn.BCEWithLogitsLoss()
    adtype = amp_dtype()

    def loss_of(x):
        logits = decode_logits(x, net, decoder, network_id, base_kwargs, renderer_hex, adtype)
        target = torch.ones_like(logits)
        return logits, criterion(logits, target), target

    save_texture_gray(x0, out_dir / "snapshots" / "step_0000_gray.png")
    with torch.no_grad():
        correct = loss_of(x0)[0] > 0
    loss_history, best_x, best_flip, best_restart = [], None, -float("inf"), None

    for restart in range(n_restarts):
        snap_dir = out_dir / "snapshots" / f"restart_{restart:02d}"
        snap_dir.mkdir(parents=True, exist_ok=True)
        x = _random_start(x0, eps, norm).detach().requires_grad_(True)
        alpha, alpha_min = lr, lr * alpha_min_factor
        ckpt_int = max(1, int(steps * checkpoint_frac))
        best_ckpt_loss, best_ckpt_x = -float("inf"), x.detach().clone()

        for step in tqdm(range(1, steps + 1), desc=f"restart {restart + 1}/{n_restarts}"):
            x.requires_grad_(True)
            logits, loss, _ = loss_of(x)
            if x.grad is not None:
                x.grad.zero_()
            loss.backward()

            with torch.no_grad():
                grad = x.grad
                if norm == "linf":
                    x = x + alpha * grad.sign()
                else:
                    g = grad.view(grad.shape[0], -1)
                    g = (g / g.norm(dim=1, keepdim=True).clamp(min=1e-12)).view_as(x)
                    x = x + alpha * g
                x = _project(x, x0, eps, norm)

                if loss.item() > best_ckpt_loss:
                    best_ckpt_loss, best_ckpt_x = loss.item(), x.clone()
                if step % ckpt_int == 0:
                    # re-read at the updated x, so the logged flip rate is the
                    # one of the iterate this step actually reached
                    logits, loss_true, _ = loss_of(x)
                    if loss_true.item() <= best_ckpt_loss:
                        alpha = max(alpha * 0.5, alpha_min)
                        x = best_ckpt_x.clone()
                    else:
                        best_ckpt_loss, best_ckpt_x = loss_true.item(), x.clone()

            loss_history.append({"restart": restart, "step": step, "loss": float(loss.detach()),
                                 **flip_rates(logits.detach(), correct)})
            x = x.detach()

            if step % save_every == 0 or step == 1:
                save_texture_gray(x, snap_dir / f"step_{step:04d}_gray.png")
                torch.save(x.cpu(), snap_dir / f"step_{step:04d}.pt")

            if patience and step > 50 and len(loss_history) > patience:
                prev = loss_history[-patience - 1]["loss"]
                if abs(loss.item() - prev) / max(prev, 1e-6) < rel_thresh:
                    break

        torch.save(x.cpu(), snap_dir / "final.pt")
        flip = loss_history[-1]["avg_flip_correct_initially"]
        if flip > best_flip:
            best_flip, best_restart, best_x = flip, restart, x.clone()

    torch.save(best_x.cpu(), out_dir / "best_texture.pt")
    save_texture_gray(best_x, out_dir / "best_texture_gray.png")
    torch.save(loss_history, out_dir / "loss_history.pt")
    save_json({"restart": best_restart, "label_flip": best_flip,
               "network_id": network_id, "norm": norm, "eps": eps},
              out_dir / "best.json")
    return out_dir
