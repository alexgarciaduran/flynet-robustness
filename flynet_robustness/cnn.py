"""
@author: alexgarciaduran and alejandrorg00
"""

import flyvis
import torch
from torch import nn
from torch.nn.init import constant_, kaiming_normal_


def conv_block(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, nonlinearity="ELU"):
    """Conv (+BatchNorm) (+nonlinearity). nonlinearity=None -> bare conv."""
    pad = (kernel_size - 1) // 2
    if nonlinearity is None or nonlinearity is False:
        return nn.Conv2d(in_planes, out_planes, kernel_size, stride, pad, bias=True)
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, pad, bias=False),
            nn.BatchNorm2d(out_planes), getattr(nn, nonlinearity)())
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size, stride, pad, bias=True),
        getattr(nn, nonlinearity)())


def _init_weights(module):
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            kaiming_normal_(m.weight, 0.1)
            if m.bias is not None:
                constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            constant_(m.weight, 1)
            constant_(m.bias, 0)


class RegularHexToCartesianMap(nn.Module):
    """Scatter hexal activity onto a regular cartesian map."""

    def __init__(self, extent=15):
        super().__init__()
        self.u, self.v = flyvis.utils.hex_utils.get_hex_coords(extent)
        self.u -= self.u.min()
        self.v -= self.v.min()
        self.H, self.W = self.u.max() + 1, self.v.max() + 1

    def forward(self, x):
        n, f, c, _ = x.shape
        out = torch.zeros([n, f, c, self.H, self.W], device=x.device, dtype=x.dtype)
        out[:, :, :, self.u, self.v] = x
        return out.squeeze(dim=2)


class CartesianMapToRegularHex(nn.Module):
    """Gather a cartesian map back to hexal activity."""

    def __init__(self, extent=15):
        super().__init__()
        self.u, self.v = flyvis.utils.hex_utils.get_hex_coords(extent)
        self.u -= self.u.min()
        self.v -= self.v.min()
        self.H, self.W = self.u.max() + 1, self.v.max() + 1

    def forward(self, x):
        n, c = x.shape[:2]
        return x.view(n, 1, c, self.H, self.W)[:, :, :, self.u, self.v]


class VanillaHexCNNBaseline(nn.Module):
    """Feedforward hex CNN with local receptive fields (~400k params)."""

    def __init__(self, n_frames=4, conv_kernel_sizes=None, conv_strides=None,
                 batchNorm=True, shape=None, nonlinearity="ELU"):
        super().__init__()
        shape = shape or [64, 32, 16, 8, 2]
        conv_kernel_sizes = conv_kernel_sizes or [1, 3, 3, 3, 5]
        conv_strides = conv_strides or [1, 1, 1, 1, 1]

        self.to_cartesian = RegularHexToCartesianMap(extent=15)
        blocks = [conv_block(batchNorm, n_frames, shape[0],
                             conv_kernel_sizes[0], conv_strides[0], nonlinearity)]
        in_planes = shape[0]
        for i, out_planes in enumerate(shape[1:-1]):
            blocks.append(conv_block(batchNorm, in_planes, out_planes,
                                     conv_kernel_sizes[i + 1], conv_strides[i + 1], nonlinearity))
            in_planes = out_planes
        blocks.append(conv_block(batchNorm, in_planes, shape[-1],
                                 conv_kernel_sizes[-1], conv_strides[-1], nonlinearity=None))
        self.conv = nn.Sequential(*blocks)
        self.to_hex = CartesianMapToRegularHex(extent=15)
        _init_weights(self)

    def forward(self, x):
        return self.to_hex(self.conv(self.to_cartesian(x)))


class VanillaHexCNNBaselineGlobalInformation(nn.Module):
    """Hex CNN with a global-average-pooling context injected mid-network."""

    def __init__(self, n_frames=4, conv_kernel_sizes=None, conv_strides=None,
                 batchNorm=True, shape=None, nonlinearity="ELU", global_inject_after=2):
        super().__init__()
        shape = shape or [64, 32, 16, 8, 2]
        conv_kernel_sizes = conv_kernel_sizes or [1, 3, 3, 3, 5]
        conv_strides = conv_strides or [1, 1, 1, 1, 1]
        assert 0 < global_inject_after < len(shape)

        self.global_inject_after = global_inject_after
        self.to_cartesian = RegularHexToCartesianMap(extent=15)
        blocks = [conv_block(batchNorm, n_frames, shape[0],
                             conv_kernel_sizes[0], conv_strides[0], nonlinearity)]
        in_planes = shape[0]
        for i in range(1, len(shape) - 1):
            blocks.append(conv_block(batchNorm, in_planes, shape[i],
                                     conv_kernel_sizes[i], conv_strides[i], nonlinearity))
            in_planes = shape[i]
        blocks.append(conv_block(batchNorm, in_planes, shape[-1],
                                 conv_kernel_sizes[-1], conv_strides[-1], nonlinearity=None))
        self.conv = nn.Sequential(*blocks)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.to_hex = CartesianMapToRegularHex(extent=15)
        _init_weights(self)

    def forward(self, x):
        x = self.to_cartesian(x)
        feat = self.conv[:self.global_inject_after](x)
        feat = feat + self.global_pool(feat).expand_as(feat)
        feat = self.conv[self.global_inject_after:](feat)
        return self.to_hex(feat)


class FeatureExtractor(nn.Module):
    """Expose the final hidden representation of a vanilla hex CNN."""

    def __init__(self, base_model):
        super().__init__()
        self.base = base_model

    def forward(self, x):
        x = self.base.to_cartesian(x)
        feats = []
        for block in self.base.conv:
            x = block(x)
            feats.append(x)
        return self.base.to_hex(feats[-1]), feats[-2], feats[-3]


class FeatureExtractorGlobalInformation(nn.Module):
    """Feature extractor variant for the global-pooling CNN."""

    def __init__(self, base_model):
        super().__init__()
        self.base = base_model

    def forward(self, x):
        x = self.base.to_cartesian(x)
        feats, cur = [], x
        for i, block in enumerate(self.base.conv):
            cur = block(cur)
            if i == self.base.global_inject_after - 1:
                cur = cur + self.base.global_pool(cur).expand_as(cur)
            feats.append(cur)
        return self.base.to_hex(feats[-1]), feats[-2], feats[-3]
