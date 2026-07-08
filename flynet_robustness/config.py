"""Network identifiers, cost tables and plotting metadata.

Network ids used throughout the code:
    flynet_synthesized  -> FlyNet (connectome-constrained, hybrid T4d)
    flynet_no_connectome-> CTRNN (unconstrained, same dynamics)
    0                   -> vCNN
    global_pooling      -> gCNN
    raft_small          -> RAFTsmall
    raft_large_no_recurrence   -> RAFT-1
    raft_large_half_recurrence -> RAFT-6
    raft_large                 -> RAFT-12
"""

# Motion cell types decoded from FlyNet/CTRNN (T4/T5, 5768 units).
MOTION_CELL_TYPES = ["T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"]

# Input dimensionality (columns / pixels) per network, for noise normalization.
INPUT_DIM = {
    "flynet_synthesized": 721, "flynet_no_connectome": 721, "0": 721,
    "global_pooling": 721, "raft_small": 416 ** 2, "raft_large": 416 ** 2,
    "raft_large_half_recurrence": 416 ** 2, "raft_large_no_recurrence": 416 ** 2,
}

# Inference cost in FLOPs (100-frame video, torch.profiler).
FLOPS = {
    "flynet_synthesized": 186_317_862, "flynet_no_connectome": 783_232_600,
    "0": 78_787_269_792, "global_pooling": 78_796_022_580,
    "raft_small": 6_138_082_963_584, "raft_large": 27_401_820_855_552,
    "raft_large_half_recurrence": 17_380_072_854_144,
    "raft_large_no_recurrence": 9_028_616_186_304,
}

FREE_PARAMETERS = {
    "flynet_synthesized": 734, "flynet_no_connectome": 11_593, "0": 414_064,
    "global_pooling": 414_064, "raft_small": 1_000_000, "raft_large": 5_257_512,
    "raft_large_half_recurrence": 5_257_512, "raft_large_no_recurrence": 5_257_512,
}

SYNAPSES = {
    "flynet_synthesized": 1_513_231, "flynet_no_connectome": 7_512_643,
    "0": 397_903_504, "global_pooling": 397_903_504,
}

LABELS = {
    "flynet_synthesized": "FlyNet", "flynet_no_connectome": "CTRNN",
    "0": "vCNN", "global_pooling": "gCNN", "raft_small": "RAFTsmall",
    "raft_large": "RAFT-12", "raft_large_half_recurrence": "RAFT-6",
    "raft_large_no_recurrence": "RAFT-1",
}

COLORS = {
    "flynet_synthesized": "forestgreen", "flynet_no_connectome": "yellowgreen",
    "0": "royalblue", "global_pooling": "lightblue", "raft_small": "darkorange",
    "raft_large": "crimson", "raft_large_half_recurrence": "chocolate",
    "raft_large_no_recurrence": "lightcoral",
}

# Perturbation name -> (axis label, key stored in result dicts).
PERTURBATION_AXIS = {
    "AWN": (r"Noise strength, $\sigma$", "sigma"),
    "AWN_frames": (r"Noise strength, $\sigma$", "sigma"),
    "Flicker": (r"Flickering strength, $\epsilon$", "flicker_strength"),
    "Contrast_fog": (r"Contrast scale, $\alpha$", "contrast_scale"),
    "Shot_noise": (r"Photon count, $c$", "poisson_scale"),
    "Motion_jitter": (r"Jitter, $j$", "motion_shift"),
}

# Default looming stimulus for attacks / perturbations.
DEFAULT_STIMULUS = dict(
    hit=True, radius=4.0, shape="Sphere", theta=0, time=1, fps=100,
    stop_margin_hit=0.6, z0=15, out_size=416, return_rgb=True,
)


def is_cnn(network_id: str) -> bool:
    """CNN/RAFT networks are everything that is not a FlyNet/CTRNN variant."""
    return "flynet" not in network_id
