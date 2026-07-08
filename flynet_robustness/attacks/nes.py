"""Natural Evolution Strategies (gradient-free) attack on the object texture."""

import torch
from tqdm import tqdm

from ..config import DEFAULT_STIMULUS
from ..decoders import to_logits
from ..hex_eye import BoxEye
from ..textures import get_texture
from .common import amp_dtype, build_out_dir, decode_logits, save_json, save_texture_gray


def run_nes(models, cnn=False, steps=100, lr=1e-1, tex_res=128, sigma=1e-2,
            n_samples=15, base_kwargs=None, save_every=10, texttype="Black",
            run_id=1, device="cuda", cnn_id="flynet_synthesized"):
    """Estimate the gradient with symmetric Gaussian perturbations and ascend the loss."""
    out_dir = build_out_dir("NES", texttype, cnn, "gaussian", sigma, run_id, cnn_id)
    base_kwargs = base_kwargs or dict(DEFAULT_STIMULUS, device=device)
    save_json(base_kwargs, out_dir / "config.json")

    x = to_logits(get_texture(texttype, tex_res).to(device).clamp(0, 1)).detach()
    x0 = x.clone()
    renderer_hex = BoxEye(models[0].connectome.config.extent, kernel_size=13)
    criterion = torch.nn.BCEWithLogitsLoss()
    adtype = amp_dtype()

    def loss_logits(w):
        logits = decode_logits(w, models, cnn, cnn_id, base_kwargs, renderer_hex,
                               models[-1], amp_dtype=adtype)
        return criterion(logits, torch.ones_like(logits)), logits

    with torch.inference_mode():
        _, logits0 = loss_logits(x)
        correct = (logits0 > 0).float()

    loss_history = []
    for step in tqdm(range(1, steps + 1)):
        grad = torch.zeros_like(x)
        for _ in range(n_samples):
            mu = torch.randn_like(x)
            with torch.inference_mode():
                loss_pos, _ = loss_logits(x + sigma * mu)
                loss_neg, _ = loss_logits(x - sigma * mu)
                grad += ((loss_pos - loss_neg) / (2 * sigma)) * mu
        grad /= n_samples
        grad /= grad.abs().mean().clamp_min(1e-12)

        with torch.inference_mode():
            x.add_(lr * grad)
            loss, logits = loss_logits(x)
            pred = (logits > 0).float()
            flip = ((pred != 1) & (correct == 1)).float().mean().item()
        loss_history.append({"step": step, "loss": float(loss), "label_flip": flip})
        if step % save_every == 0 or step == 1:
            save_texture_gray(x, out_dir / "snapshots" / f"step_{step:04d}_gray.png")
            torch.save(x.detach().cpu(), out_dir / "snapshots" / f"step_{step:04d}.pt")

    torch.save(loss_history, out_dir / "loss_history.pt")
    return out_dir
