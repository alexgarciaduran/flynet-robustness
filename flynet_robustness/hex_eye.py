"""Cartesian-to-hexagonal transduction (BoxEye photoreceptor array).

Adapted from flyvis to keep the pipeline self-contained.
"""

from typing import Iterator, Literal, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as ttf
from torch import nn


class BoxEye:
    """Box filter producing hexals matching the photoreceptor array.

    Args:
        extent: radius, in receptors, of the hexagonal array.
        kernel_size: photon collection radius, in pixels.
    """

    def __init__(self, extent: int = 15, kernel_size: int = 13):
        self.extent = extent
        self.kernel_size = kernel_size
        self.receptor_centers = torch.tensor([*self._receptor_centers()], dtype=torch.long)
        self.hexals = len(self.receptor_centers)
        self.min_frame_size = (self.receptor_centers.max(0).values
                               - self.receptor_centers.min(0).values + 1)
        self._set_filter()
        pad = (self.kernel_size - 1) / 2
        self.pad = (int(np.ceil(pad)), int(np.floor(pad)),
                    int(np.ceil(pad)), int(np.floor(pad)))

    def _receptor_centers(self) -> Iterator[Tuple[float, float]]:
        n, d = self.extent, self.kernel_size
        for u in range(-n, n + 1):
            for v in range(max(-n, -n - u), min(n, n - u) + 1):
                yield d * (u + v / 2), d * v

    def _set_filter(self) -> None:
        self.conv = nn.Conv2d(1, 1, kernel_size=self.kernel_size, stride=1, padding=0)
        self.conv.weight.data /= self.conv.weight.data
        self.conv.bias.data.fill_(0)
        self.conv.weight.requires_grad = False
        self.conv.bias.requires_grad = False

    def _align(self, sequence: torch.Tensor) -> None:
        """Move the box filter to the sequence's device/dtype (kept across calls)."""
        w = self.conv.weight
        if w.device != sequence.device or w.dtype != sequence.dtype:
            self.conv = self.conv.to(device=sequence.device, dtype=sequence.dtype)
            self.receptor_centers = self.receptor_centers.to(sequence.device)

    def __call__(self, sequence: torch.Tensor,
                 ftype: Literal["mean", "sum"] = "mean",
                 hex_sample: bool = True) -> torch.Tensor:
        """Apply the box kernel to (samples, frames, H, W) sequences."""
        if not isinstance(sequence, torch.Tensor):
            sequence = torch.as_tensor(sequence, dtype=torch.float32)
        self._align(sequence)
        samples, frames, h, w = sequence.shape
        if (self.min_frame_size > torch.tensor([h, w])).any():
            sequence = ttf.resize(sequence, self.min_frame_size.tolist())
            h, w = sequence.shape[2:]

        def _convolve():
            conv = lambda x: self.conv(x.unsqueeze(1))
            return torch.cat(tuple(map(conv, torch.unbind(F.pad(sequence, self.pad), 0))), 0)

        if ftype == "mean":
            out = _convolve() / self.kernel_size ** 2
        elif ftype == "sum":
            out = _convolve()
        else:
            raise ValueError("ftype must be 'sum' or 'mean'")
        if hex_sample:
            return self.hex_render(out).reshape(samples, frames, 1, -1)
        return out.reshape(samples, frames, h, w)

    def hex_render(self, sequence: torch.Tensor) -> torch.Tensor:
        """Sample receptor locations from cartesian frames -> (S, F, 1, hexals)."""
        h, w = sequence.shape[2:]
        if (self.min_frame_size > torch.tensor([h, w])).any():
            sequence = ttf.resize(sequence, self.min_frame_size.tolist())
            h, w = sequence.shape[2:]
        c = self.receptor_centers.to(sequence.device) + torch.tensor(
            [h // 2, w // 2], device=sequence.device)
        out = sequence[:, :, c[:, 0], c[:, 1]].clone()
        return out.view(*sequence.shape[:2], 1, -1).contiguous()
