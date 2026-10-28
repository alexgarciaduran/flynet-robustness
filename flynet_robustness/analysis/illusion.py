"""
@author: alexgarciaduran and alejandrorg00
"""

import glob
import os
import re
from functools import lru_cache

import numpy as np

CONDITIONS = ("original", "illusion", "recovered", "reverse_phi")


@lru_cache(maxsize=None)
def radial_pairs(extent=15):
    """(inner, outer) hex indices: every non-central column and its most outward neighbour."""
    uv = [(u, v) for u in range(-extent, extent + 1)
          for v in range(max(-extent, -extent - u), min(extent, extent - u) + 1)]
    index = {p: i for i, p in enumerate(uv)}
    xy = np.array([(1.5 * v, -np.sqrt(3) * (u + v / 2)) for u, v in uv])
    inner, outer = [], []
    for i, (u, v) in enumerate(uv):
        r = np.linalg.norm(xy[i])
        if r < 1e-6:
            continue
        nb = [index[(u + du, v + dv)] for du, dv in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]
              if (u + du, v + dv) in index]
        inner.append(i)
        outer.append(nb[int(np.argmax([(xy[j] - xy[i]) @ xy[i] / r for j in nb]))])
    return np.array(inner), np.array(outer)


def two_point(c, D=8, frames=None):
    """Mean 2-point of a (T, 721) input over hexes and frame-pairs (optionally masked (T-D,))."""
    C = c - c.mean(1, keepdims=True)
    i, o = radial_pairs()
    g = C[:-D, i] * C[D:, o] - C[:-D, o] * C[D:, i]
    return g[frames].mean() if frames is not None else g.mean()


def flip_mask(T, D=8, T_flip=33, W=1):
    """Frame-pairs (t, t+D) spanning a reverse-phi contrast flip (+-W frames)."""
    t, out = np.arange(T - D), np.zeros(T - D, bool)
    for b in range(T_flip, T, T_flip):
        out |= (b >= t - W) & (b <= t + D + W)
    return out


def two_point_torch(c, D=8):
    """Differentiable mean 2-point of a (T, 721) torch input."""
    import torch
    i, o = (torch.as_tensor(a, device=c.device) for a in radial_pairs())
    C = c - c.mean(1, keepdim=True)
    return (C[:-D, i] * C[D:, o] - C[:-D, o] * C[D:, i]).mean()


def _file(root, n, cond, response):
    pat = (f"{root}/{cond}/Sphere/simulation_{n}_*_Sphere_{cond}.npy" if response else
           f"{root}/{cond}/Sphere/stimulus/input_stimulus_{n}_*__Sphere_{cond}.npy")
    files = sorted(f for f in glob.glob(pat) if "t4t5miss" not in os.path.basename(f))
    if not files:
        raise FileNotFoundError(pat)
    return files[0]


def load_stimulus(root, n, cond):
    """(T, 721) hex input of trajectory n."""
    return np.squeeze(np.load(_file(root, n, cond, False))).astype(np.float32)


def load_response(root, n, cond):
    """(T, 5768) T4/T5 activity of trajectory n."""
    return np.squeeze(np.load(_file(root, n, cond, True))).astype(np.float32)


def trials(root, conds=CONDITIONS):
    """Trajectories simulated for every condition."""
    ids = [{int(re.search(r"input_stimulus_(\d+)_", f).group(1)) for f in
            glob.glob(f"{root}/{c}/Sphere/stimulus/input_stimulus_*__Sphere_{c}.npy")} for c in conds]
    return sorted(set.intersection(*ids))


def switch_locked(resp, T_flip=33, win=20, first_only=True):
    """Mean activity in [s, s+win) after the reverse-phi switches s (first switch only by default)."""
    starts = [s for s in range(T_flip, len(resp), T_flip) if s + win <= len(resp)]
    starts = starts[:1] if first_only else starts
    return np.concatenate([resp[s:s + win] for s in starts]).mean(0)


def closeness(patterns, ref="reverse_phi", base="original"):
    """Per-trial closeness of each condition's pattern to `ref`: correlation, RMS distance, projection.

    patterns: {cond: (n_trials, n_units)}. Projection onto base->ref (0 = base, 1 = ref).
    """
    R, B = patterns[ref], patterns[base]
    u = R - B
    Rc = R - R.mean(1, keepdims=True)
    out = {"rsa": {}, "euclid": {}, "projection": {}}
    for c, P in patterns.items():
        Pc = P - P.mean(1, keepdims=True)
        out["rsa"][c] = (Pc * Rc).sum(1) / (np.linalg.norm(Pc, axis=1) * np.linalg.norm(Rc, axis=1))
        out["euclid"][c] = np.linalg.norm(P - R, axis=1) / np.sqrt(P.shape[1])
        out["projection"][c] = ((P - B) * u).sum(1) / (u * u).sum(1)
    return out
