"""
@author: alexgarciaduran and alejandrorg00
"""

import numpy as np
import torch


def reverse_phi(frames, T_flip=5):
    """Invert contrast (x -> 1-x) on alternating blocks of T_flip frames.

    Accepts a numpy array or torch tensor of shape (T, ...).
    """
    is_torch = isinstance(frames, torch.Tensor)
    out = (frames.clone() if is_torch else frames.astype(float).copy())
    for t in range(len(out)):
        if (t // T_flip) % 2 == 1:
            out[t] = 1 - out[t]
    return out


def reverse_phi_p_hit(frames_gray, renderer_hex, decoder, forward_fn, T_flip=1):
    """Run reverse-phi frames through hex-render, network and decoder.

    forward_fn(hex_input) -> flattened activity; returns p(hit) over time.
    """
    frames = reverse_phi(frames_gray.detach().cpu().numpy(), T_flip=T_flip)
    hex_frames = renderer_hex(frames[None])
    activity = forward_fn(hex_frames).flatten(end_dim=1)
    with torch.no_grad():
        return torch.sigmoid(decoder(activity)).cpu().numpy()


def sign_agreement(true_vel, pred_vel):
    """Mean product of the signs of true and predicted velocity (per component)."""
    return np.mean(np.sign(true_vel) * np.sign(pred_vel), axis=0)
