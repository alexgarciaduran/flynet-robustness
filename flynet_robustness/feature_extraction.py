"""Load CNN/RAFT baselines and extract decoder-input features.

The vanilla CNN stacks ``n_frames`` consecutive frames as input channels;
``n_frames`` (2/5/10), the checkpoint path and the state-dict key all come
from the network registry, and the same ``n_frames`` is used in the forward.
RAFT is imported lazily from an external checkout (see README).
"""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from . import config
from .cnn import (
    FeatureExtractor,
    FeatureExtractorGlobalInformation,
    VanillaHexCNNBaseline,
    VanillaHexCNNBaselineGlobalInformation,
)

_CNN_SHAPE = [32, 92, 136, 8, 2]
_CNN_KERNELS = [1, 5, 5, 5, 5]


def load_cnn_network(network_id="RAFT_12", device="cuda"):
    """Load a CNN/gCNN/RAFT baseline as a feature-extractor module."""
    s = config.spec(network_id)
    kind = s["kind"]

    if kind == "raft":
        from core.raft import RAFT  # external RAFT checkout on sys.path

        class Args:
            def __init__(self, small):
                self.small, self.mixed_precision, self.alternate_corr = small, False, False
                self.model = config.checkpoint_path(network_id)

            def __contains__(self, key):
                return hasattr(self, key)

        model = RAFT(Args(s["small"]))
        state = torch.load(Args(s["small"]).model, map_location=device, weights_only=False)
        model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()})
        return model.eval().to(device)

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
    model.load_state_dict(state)
    return extractor(model.eval()).eval().to(device)


def _raft_forward(model, a1, a2, n_iters):
    _, net = model(a1, a2, iters=n_iters)
    return net


def extract_features(model, hex_frames, network_id="RAFTsmall"):
    """Return the final hidden representation used as decoder input.

    CNN: (final_flow, conv3_2, conv2_2). RAFT: (None, pooled_features, None).
    The CNN branch stacks ``config.n_frames(network_id)`` consecutive frames.
    """
    if config.kind(network_id) != "raft":
        n_frames = config.n_frames(network_id)
        seqs = [hex_frames[:, t:t + n_frames]
                for t in range(hex_frames.shape[1] - n_frames + 1)]
        return model(torch.cat(seqs, dim=0))

    n_iters = config.raft_iters(network_id)
    avg_pool = nn.AvgPool2d(kernel_size=7, stride=7)
    frames = hex_frames.permute(0, 3, 1, 2)
    img1, img2 = frames[:-1], frames[1:]
    outputs = []
    for i in range(0, img1.shape[0], 8):
        net = checkpoint(_raft_forward, model, img1[i:i + 8], img2[i:i + 8], n_iters)
        outputs.append(avg_pool(net))
    return None, torch.cat(outputs, dim=0), None
