"""
@author: alexgarciaduran
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from . import config
from .cnn import (
    FeatureExtractor,
    FeatureExtractorGlobalInformation,
    RegularHexToCartesianMap,
    VanillaHexCNNBaseline,
    VanillaHexCNNBaselineGlobalInformation,
)

_CNN_SHAPE = [32, 92, 136, 8, 2]
_CNN_KERNELS = [1, 5, 5, 5, 5]


def load_cnn_network(network_id="RAFT_12", device="cuda"):
    """Load a CNN/gCNN/RAFT baseline as a feature-extractor module."""
    s = config.spec(network_id)
    kind = s["kind"]

    if kind in ("raft", "raft_hex"):
        from core.raft import RAFT  # external RAFT checkout on sys.path

        class Args:
            def __init__(self, small):
                self.small, self.mixed_precision, self.alternate_corr = small, False, False
                self.model = config.checkpoint_path(network_id)

            def __contains__(self, key):
                return hasattr(self, key)

        args = Args(s["small"])
        model = RAFT(args)
        state = torch.load(args.model, map_location=device, weights_only=False)
        state = state.get("state_dict", state)
        model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()})
        model = model.eval().to(device)
        for p in model.parameters():  # attacks only need gradients w.r.t. the input
            p.requires_grad_(False)
        return model

    n_frames = s["n_frames"]
    if kind == "gcnn":
        model = VanillaHexCNNBaselineGlobalInformation(
            n_frames=n_frames, shape=_CNN_SHAPE, conv_kernel_sizes=_CNN_KERNELS, batchNorm=True)
        extractor = FeatureExtractorGlobalInformation
    else:
        model = VanillaHexCNNBaseline(
            n_frames=n_frames, shape=_CNN_SHAPE, conv_kernel_sizes=_CNN_KERNELS, batchNorm=True)
        extractor = FeatureExtractor

    state = torch.load(config.checkpoint_path(network_id), map_location="cpu",
                       weights_only=False)[s["state_key"]]
    # The gCNN checkpoint also carries early_conv/late_conv aliases of `conv`.
    expected = model.state_dict()
    missing = [k for k in expected if k not in state]
    if missing:
        raise KeyError(f"{network_id} checkpoint is missing {len(missing)} keys, e.g. {missing[:3]}")
    model.load_state_dict({k: state[k] for k in expected})
    return extractor(model.eval()).eval().to(device)


def _raft_forward(model, a1, a2, n_iters):
    """Final recurrent hidden state of RAFT (read from the update block)."""
    hidden = []
    handle = model.update_block.register_forward_hook(lambda m, i, o: hidden.append(o[0]))
    try:
        model(a1, a2, iters=n_iters)
    finally:
        handle.remove()
    return hidden[-1]


def raft_hex_frames(hex_frames, size=256, extent=15):
    """RAFT-S input: hex frames (1, T, 1, 721) in [0, 1] -> (T, 3, size, size) in [0, 255]."""
    cart = F.pad(RegularHexToCartesianMap(extent)(hex_frames.permute(1, 0, 2, 3)), (0, 1, 0, 1))
    return 255 * F.interpolate(cart, size=(size, size), mode="bilinear", align_corners=False).repeat(1, 3, 1, 1)


def extract_features(model, hex_frames, network_id="RAFTsmall"):
    """Return the final hidden representation used as decoder input.

    CNN: (final_flow, conv3_2, conv2_2). RAFT: (None, pooled_features, None).
    The CNN branch stacks ``config.n_frames(network_id)`` consecutive frames.
    """
    kind = config.kind(network_id)
    if kind not in ("raft", "raft_hex"):
        n_frames = config.n_frames(network_id)
        seqs = [hex_frames[:, t:t + n_frames]
                for t in range(hex_frames.shape[1] - n_frames + 1)]
        return model(torch.cat(seqs, dim=0))

    n_iters = config.raft_iters(network_id)
    if kind == "raft_hex":
        frames, avg_pool, chunk = raft_hex_frames(hex_frames), nn.AdaptiveAvgPool2d((7, 7)), 4
    else:
        frames, avg_pool, chunk = hex_frames.permute(0, 3, 1, 2), nn.AvgPool2d(kernel_size=7, stride=7), 8
    img1, img2 = frames[:-1], frames[1:]
    outputs = []
    for i in range(0, img1.shape[0], chunk):
        net = checkpoint(_raft_forward, model, img1[i:i + chunk], img2[i:i + chunk], n_iters,
                         use_reentrant=False)
        outputs.append(avg_pool(net))
    return None, torch.cat(outputs, dim=0), None
