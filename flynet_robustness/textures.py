"""
@author: alexgarciaduran
"""

import numpy as np
import torch

TEXTURE_TYPES = ["Black", "White", "Checkerboard", "Wavy", "Random"]


def make_checkerboard(size: int = 128, num_checks: int = 6) -> torch.Tensor:
    """Square checkerboard in [0,1], shape [1,H,W,1]."""
    xx, yy = np.meshgrid(np.arange(size), np.arange(size))
    checks = ((xx // (size // num_checks)) + (yy // (size // num_checks))) % 2
    img = checks.astype(np.float32)[..., None]
    return torch.from_numpy(img)[None]


def make_tri_checkerboard(size: int = 128, num_tri: int = 4) -> torch.Tensor:
    """Triangular ('wavy') checkerboard in [0,1], shape [1,H,W,1]."""
    xx, yy = np.meshgrid(np.arange(size), np.arange(size))
    u, v = xx / size * num_tri, yy / size * num_tri
    ui, vi = np.floor(u).astype(int), np.floor(v).astype(int)
    tri = ((u - ui) + (v - vi)) < 1.0
    checks = (ui + vi + tri.astype(int)) % 2
    img = checks.astype(np.float32)[..., None]
    return torch.from_numpy(img)[None]


def get_texture(texture_type: str, size: int = 128) -> torch.Tensor:
    """Return a texture of the given type as a [1,H,W,1] float tensor in [0,1]."""
    if texture_type == "Black":
        return torch.zeros(1, size, size, 1)
    if texture_type == "White":
        return torch.ones(1, size, size, 1)
    if texture_type == "Checkerboard":
        return make_checkerboard(size, num_checks=6).clamp(0, 1)
    if texture_type == "Wavy":
        return make_tri_checkerboard(size, num_tri=4).clamp(0, 1)
    if texture_type == "Random":
        base = torch.rand(1, size, size, 1)          # drawn before the offset
        return torch.clamp(base + (0.2 + 0.6 * torch.rand(1) - 0.5), 0.0, 1.0)
    raise ValueError(f"Unknown texture_type: {texture_type}")
