"""Load CNN/RAFT baselines and extract decoder input features.

RAFT is imported lazily from an external checkout (see README); only needed
for the RAFT baselines.
"""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .cnn import (
    FeatureExtractor,
    FeatureExtractorGlobalInformation,
    VanillaHexCNNBaseline,
    VanillaHexCNNBaselineGlobalInformation,
)

_CNN_SHAPE = [32, 92, 136, 8, 2]
_CNN_KERNELS = [1, 5, 5, 5, 5]
# Recurrent refinement iterations per RAFT variant.
_RAFT_ITERS = {"no_recurrence": 1, "half_recurrence": 6}


def _raft_iters(cnn_id):
    for key, n in _RAFT_ITERS.items():
        if key in cnn_id:
            return n
    return 12


def load_cnn_network(cnn_id="raft_large", device="cuda"):
    """Load a CNN/RAFT baseline as a feature-extractor module."""
    if "raft" in cnn_id:
        from core.raft import RAFT  # external RAFT checkout on sys.path

        class Args:
            def __init__(self, small):
                self.small = small
                self.model = "models/raft-small.pth" if small else "models/raft-sintel.pth"
                self.mixed_precision = False
                self.alternate_corr = False

            def __contains__(self, key):
                return hasattr(self, key)

        args = Args(small="small" in cnn_id)
        model = RAFT(args)
        chkpt = {k.replace("module.", ""): v
                 for k, v in torch.load(args.model, map_location=device, weights_only=False).items()}
        model.load_state_dict(chkpt)
        return model.eval().to(device)

    if "global" in cnn_id:
        path = "cnn_networks/global_pooling/global_pooling_network_000.chkpt"
        model = VanillaHexCNNBaselineGlobalInformation(
            n_frames=2, shape=_CNN_SHAPE, conv_kernel_sizes=_CNN_KERNELS, batchNorm=True)
        extractor = FeatureExtractorGlobalInformation
    else:
        path = f"cnn_networks/2_frame/network_00{cnn_id}.chkpt"
        model = VanillaHexCNNBaseline(
            n_frames=2, shape=_CNN_SHAPE, conv_kernel_sizes=_CNN_KERNELS, batchNorm=True)
        extractor = FeatureExtractor

    state = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict(state)
    return extractor(model.eval()).eval().to(device)


def _raft_forward(model, a1, a2, n_iters):
    _, net = model(a1, a2, iters=n_iters)
    return net


def extract_features(model, hex_frames, n_frames=2, cnn_id="raft_small"):
    """Return the final hidden representation used as decoder input.

    CNN: (final_flow, conv3_2, conv2_2). RAFT: (None, pooled_features, None).
    """
    if "raft" not in cnn_id:
        seqs = [hex_frames[:, t:t + n_frames]
                for t in range(hex_frames.shape[1] - n_frames + 1)]
        return model(torch.cat(seqs, dim=0))

    n_iters = _raft_iters(cnn_id)
    avg_pool = nn.AvgPool2d(kernel_size=7, stride=7)
    frames = hex_frames.permute(0, 3, 1, 2)
    if "flyres" in cnn_id:
        frames = nn.AvgPool2d(13, stride=1, padding=6)(frames)
    img1, img2 = frames[:-1], frames[1:]

    outputs = []
    for i in range(0, img1.shape[0], 8):
        net = checkpoint(_raft_forward, model, img1[i:i + 8], img2[i:i + 8], n_iters)
        outputs.append(avg_pool(net))
    return None, torch.cat(outputs, dim=0), None
