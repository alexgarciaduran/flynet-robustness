"""
@author: alexgarciaduran
"""

import torch


# --- Raw perturbation ops ---------------------------------------------------


def additive_white_noise(x, sigma):
    """Zero-mean Gaussian noise of std sigma added to the input (shared across RGB channels)."""
    shape = x.shape[:-1] + (1,) if x.shape[-1] == 3 else x.shape
    return x + sigma * torch.randn(shape, device=x.device, dtype=x.dtype)


def flicker(frames, strength, generator=None):
    """Per-frame multiplicative gain in [1-strength, 1+strength].

    Frame axis is 1 for hex input (B,T,1,P), else axis 0 (raw frames T,...).
    """
    if frames.ndim == 4 and frames.shape[2] == 1:  # hex (B,T,1,P)
        shape = (frames.shape[0], frames.shape[1]) + (1,) * (frames.ndim - 2)
    else:  # raw frames, frame axis 0
        shape = (frames.shape[0],) + (1,) * (frames.ndim - 1)
    gain = (2 * strength) * torch.rand(shape, device=frames.device, generator=generator) + (1 - strength)
    return (frames * gain).clamp(0, 1)


def shot_noise(x, c=60):
    """Normalized Poisson (photon) noise modulated by photon count c."""
    if x.ndim == 4 and x.shape[-1] == 3:  # RGB frames (T,H,W,3)
        intensity = x.mean(dim=-1, keepdim=True)
        noisy = torch.poisson(intensity * c) / c
        return (x * (noisy / (intensity + 1e-8))).clamp(0, 1)
    if x.ndim == 4 and x.shape[2] == 1:  # luminance (B,T,1,P)
        return (torch.poisson(x * c) / c).clamp(0, 1)
    raise ValueError(f"Unsupported shape: {x.shape}")


def apply_fog(x, background_val=0.5, contrast_scale=1.0):
    """Reduce contrast against the background (contrast_scale in [0,1])."""
    return (x - background_val) * contrast_scale + background_val


def motion_jitter(frames, max_shift=5, seed=None, device="cpu"):
    """Random per-frame integer shift on raw RGB frames (T,H,W,C), edge-padded.

    The shifts are drawn on `device` so the generator and the draw always match.
    """
    if not (frames.ndim == 4 and frames.shape[-1] in (1, 3)):
        raise ValueError("motion_jitter expects raw frames (T,H,W,C)")
    T, H, W, _ = frames.shape
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)
    out = torch.ones_like(frames) * 0.5
    sh = torch.randint(-max_shift, max_shift + 1, (T,), generator=gen, device=device)
    sw = torch.randint(-max_shift, max_shift + 1, (T,), generator=gen, device=device)
    for t in range(T):
        h, w = sh[t].item(), sw[t].item()
        out[t, max(0, h):min(H, H + h), max(0, w):min(W, W + w)] = \
            frames[t, max(0, -h):min(H, H - h), max(0, -w):min(W, W - w)]
    return out.clamp(0, 1), sh, sw


# --- Rendering wrappers -----------------------------------------------------


def _is_raft(network_id):
    from . import config
    return config.kind(network_id) == "raft"


def _render(tex_logits, base_kwargs):
    from .rendering import generate_looming_stimulus
    tex = torch.sigmoid(tex_logits)
    kwargs = dict(base_kwargs, return_rgb=True)  # RGB frames are needed for RAFT and jitter
    frames_gray, frames_rgb, _, _, position = generate_looming_stimulus(texture=tex, **kwargs)
    return frames_gray, frames_rgb, position


def texture_to_hex_batch(tex_logits, base_kwargs, renderer_hex, network_id="RAFT_12",
                         hex_rendering=True):
    """Texture logits -> model input (hex batch for FlyNet/CNN, RGB frames for RAFT)."""
    frames_gray, frames_rgb, _ = _render(tex_logits, base_kwargs)
    if _is_raft(network_id):
        return frames_rgb
    return renderer_hex(frames_gray[None]) if hex_rendering else frames_gray[None]


def texture_to_input_perturbed(tex_logits, base_kwargs, renderer_hex, network_id="RAFT_12",
                               perturbation=None, **kwargs):
    """Render a texture and apply a named video perturbation before decoding.

    perturbation in {None, 'awn_texture', 'awn', 'fog', 'shot', 'flicker', 'jitter'}.
    Returns (model_input, position).
    """
    if perturbation == "awn_texture":  # noise on the texture pixels
        p = (torch.sigmoid(tex_logits) + kwargs.get("sigma", 0.0) * torch.randn_like(tex_logits))
        p = p.clamp(1e-4, 1 - 1e-4)
        tex_logits = torch.log(p / (1 - p))
    frames_gray, frames_rgb, position = _render(tex_logits, base_kwargs)
    is_raft = _is_raft(network_id)

    if perturbation == "fog":
        frames_gray = apply_fog(frames_gray, contrast_scale=kwargs.get("contrast_scale", 1.0))
        frames_rgb = apply_fog(frames_rgb, contrast_scale=kwargs.get("contrast_scale", 1.0))
    if perturbation == "jitter":
        frames_rgb, _, _ = motion_jitter(frames_rgb, max_shift=int(kwargs.get("jitter", 0)),
                                         seed=kwargs.get("seed"), device=frames_rgb.device.type)
        frames_gray = frames_rgb[..., 0]

    inp = frames_rgb if is_raft else (
        renderer_hex(frames_gray[None]) if kwargs.get("hex_rendering", True) else frames_gray[None])

    if perturbation == "awn":
        inp = additive_white_noise(inp, kwargs.get("sigma", 0.0))
    elif perturbation == "shot":
        inp = shot_noise(inp, c=kwargs.get("photon_count", 60))
    elif perturbation == "flicker":
        inp = flicker(inp, kwargs.get("strength", 0.0))
    return inp, position
