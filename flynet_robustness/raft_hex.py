"""
@author: alexgarciaduran
"""

import torch
import torch.nn.functional as F

from .cnn import RegularHexToCartesianMap
from .hex_eye import BoxEye


class RAFTFlyVisHexInput:
    """FlyVis hex sampling for RAFT, as RAFT-HexSintel was trained with it.

    BoxEye -> 721 hexals -> cartesian 31x31 -> pad 32 -> upsample to
    `output_size`. Scale-agnostic: pass images already in RAFT's [0, 255].
    """

    def __init__(self, extent=15, kernel_size=13, output_size=256, device="cpu"):
        self.extent, self.kernel_size, self.output_size = extent, kernel_size, output_size
        self.device = torch.device(device)
        self.eye = BoxEye(extent=extent, kernel_size=kernel_size)
        self.to_cartesian = RegularHexToCartesianMap(extent=extent)

    # --- building blocks ----------------------------------------------------

    def _sample(self, x):
        """[B,C,H,W] -> [B,1,C,721], each channel sampled on its own."""
        return torch.cat([self.eye(x[:, c][:, None], ftype="mean", hex_sample=True)
                          for c in range(x.shape[1])], dim=2)

    def _to_cart32(self, hexals):
        """[B,1,C,721] -> [B,C,32,32]."""
        cart = self.to_cartesian(hexals)
        cart = cart[:, 0][:, None] if cart.ndim == 4 else cart[:, 0]
        return F.pad(cart, (0, 1, 0, 1))

    def _resize(self, x, mode="bilinear"):
        if self.output_size == 32:
            return x
        kw = {"align_corners": False} if mode == "bilinear" else {}
        return F.interpolate(x, size=(self.output_size,) * 2, mode=mode, **kw)

    # --- batch API ----------------------------------------------------------

    def batch_image_to_raft_input(self, images, input_mode="rgb"):
        """[B,3,H,W] (or [B,1,1,721] hexals) -> [B,3,output_size,output_size].

        `input_mode`: "rgb" keeps the three channels, "lum" samples one channel
        and repeats it, "hex" skips BoxEye and takes hexals directly.
        """
        images = images.to(self.device, non_blocking=True)
        if input_mode == "hex":
            image = self._resize(self._to_cart32(images))
        elif input_mode == "lum":
            image = self._resize(self._to_cart32(self._sample(images[:, :1])))
        elif input_mode == "rgb":
            return self._resize(self._to_cart32(self._sample(images)))
        else:
            raise ValueError(f"Unknown input_mode: {input_mode}")
        return image.repeat(1, 3, 1, 1)

    def batch_flow_to_raft_target(self, flows):
        """[B,2,H,W] -> [B,2,output_size,output_size].

        Densifies the grid only; flow magnitudes keep the sampled-Sintel
        convention, so do not rescale them by output_size / 32.
        """
        flows = flows.to(self.device, non_blocking=True)
        return self._resize(self._to_cart32(self._sample(flows)))

    def batch_valid_to_raft_mask(self, valid):
        """[B,H,W] -> [B,output_size,output_size]."""
        valid = valid.to(self.device, non_blocking=True).float()
        hexals = self.eye(valid[:, None], ftype="mean", hex_sample=True)
        mask = (self._to_cart32(hexals)[:, 0] > 0).float()
        return self._resize(mask[:, None], mode="nearest")[:, 0]

    # --- single-sample API (Sintel loaders) ---------------------------------

    def image_to_raft_input(self, img, input_mode="rgb"):
        """[3,H,W] -> [3,output_size,output_size]."""
        return self.batch_image_to_raft_input(img[None], input_mode)[0]

    def flow_to_raft_target(self, flow):
        """[2,H,W] -> [2,output_size,output_size]."""
        return self.batch_flow_to_raft_target(flow[None])[0]

    def valid_to_raft_mask(self, valid):
        """[H,W] -> [output_size,output_size]."""
        return self.batch_valid_to_raft_mask(valid[None])[0]


def hex_to_raft_input(hex_frames, output_size=256, extent=15, scale=255.0):
    """(1,T,1,721) hex input in [0,1] -> (T,3,size,size) in RAFT's [0,255]."""
    pre = RAFTFlyVisHexInput(extent=extent, output_size=output_size,
                             device=hex_frames.device)
    return scale * pre.batch_image_to_raft_input(
        hex_frames.permute(1, 0, 2, 3), input_mode="hex")
