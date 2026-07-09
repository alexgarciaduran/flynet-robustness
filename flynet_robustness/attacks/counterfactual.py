"""Counterfactual restoration (paper Eqs. 2-3).

Stage 1: optimise a texture to suppress p(hit), anchored to the base texture.
Stage 2: find the nearest texture that recovers correct classification,
anchored to the illusion texture.
"""

import torch
from tqdm import tqdm

from ..config import DEFAULT_STIMULUS
from ..decoders import to_logits
from ..hex_eye import BoxEye
from ..textures import get_texture
from .common import amp_dtype, decode_logits


def _optimise(w_init, anchor, target, net, decoder, network_id, base_kwargs,
              renderer_hex, steps, lr, lambda_anchor):
    """Optimise cross-entropy to `target` with an L2 anchor to `anchor`."""
    w = w_init.clone().requires_grad_(True)
    opt = torch.optim.Adam([w], lr=lr)
    ce = torch.nn.BCEWithLogitsLoss()
    adtype = amp_dtype()
    history = []
    for _ in tqdm(range(steps)):
        logits = decode_logits(w, net, decoder, network_id, base_kwargs, renderer_hex, adtype)
        tgt = torch.full_like(logits, float(target))
        loss = ce(logits, tgt) + lambda_anchor * (torch.sigmoid(w) - anchor).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        history.append(float(loss.detach()))
    return w.detach(), history


def counterfactual_restoration(net, decoder, network_id="FlyNet", steps=250, lr=1e-2,
                               tex_res=128, lambda_base=0.02, texttype="Black",
                               base_kwargs=None, device="cuda", extent=15):
    """Return (illusion_texture, restored_texture) and their loss histories."""
    base_kwargs = base_kwargs or dict(DEFAULT_STIMULUS, device=device)
    renderer_hex = BoxEye(extent, kernel_size=13)
    base = get_texture(texttype, tex_res).to(device).clamp(0, 1)
    w_base = to_logits(base).detach()
    tex_base = torch.sigmoid(w_base)

    illusion, hist1 = _optimise(w_base, tex_base, 0, net, decoder, network_id, base_kwargs,
                                renderer_hex, steps, lr, lambda_base)
    tex_illusion = torch.sigmoid(illusion)
    restored, hist2 = _optimise(illusion, tex_illusion, 1, net, decoder, network_id, base_kwargs,
                                renderer_hex, steps, lr, lambda_base)
    return illusion, restored, hist1, hist2
