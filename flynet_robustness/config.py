"""
@author: alexgarciaduran
"""

import os

# T4/T5, the 5768 units the decoders read.
MOTION_CELL_TYPES = ["T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"]

# Overridable with the matching environment variables.
CHECKPOINT_DIR = os.environ.get("FLYNET_CHECKPOINTS", "checkpoints")
DECODER_DIR = os.environ.get("FLYNET_DECODERS", "decoders")
RAFT_MODEL_DIR = os.environ.get("FLYNET_RAFT_MODELS", "models")
# Root for the PGD/ CW/ NES/ Transfer/ output trees.
RESULTS_DIR = os.environ.get("FLYNET_RESULTS", ".")


def _cnn(n_frames, state_key, flops, params, label, color):
    return dict(kind="cnn", n_frames=n_frames, state_key=state_key,
                flops=flops, params=params, label=label, color=color)


def _raft(iters, small, flops, params, label, color, kind="raft", ckpt=None):
    return dict(kind=kind, iters=iters, small=small, n_frames=2, ckpt=ckpt,
                flops=flops, params=params, label=label, color=color)


# RAFT trained on hex-sampled Sintel, i.e. FlyNet's 721-column input.
RAFT_HEX_CKPT = "raft-sintel-flyvis-split-hex-lum-gridflow-256-bs8-50k_20260615-124452.pth"


# --- Network registry -------------------------------------------------------
# `state_key`: key in the CNN checkpoint. `legacy`: old id, for old paths.
# `flops`: torch.profiler count over a 100-frame video, as the exact integer
# the reported LERS was computed from.
NETWORKS = {
    "FlyNet": dict(kind="flynet", flyvis="flow/0000/synthesized_t4d", n_frames=None,
                   flops=186_317_862, params=734, label="FlyNet", color="forestgreen",
                   legacy="_synthesized_model"),
    "CTRNN": dict(kind="flynet", flyvis="flow/0003/000", n_frames=None,
                  flops=783_232_600, params=11593, label="CTRNN", color="yellowgreen",
                  legacy="no_connectome"),
    # CTRNN pruned to FlyNet's edge count.
    "sCTRNN": dict(kind="flynet", flyvis="flow/0215_pruned_current_ft_compact/000", n_frames=None,
                   flops=183_291_400, params=None, label="sCTRNN", color="olivedrab",
                   legacy="_pruned_ctrnn"),
    # Rewired FlyNet controls: FlyNet's nodes and edge count, 0 = best task error.
    **{f"flynet_degree_matched_{i}": dict(kind="flynet", flyvis=f"flow/0301_best10/{i:03d}",
                                          n_frames=None, flops=186_317_862, params=734,
                                          label=f"FlyNet-DM{i}", color="mediumseagreen")
       for i in range(10)},
    "vCNN_2_frames": _cnn(2, "state_dict", 78_787_269_792, 414064, "vCNN", "royalblue"),
    "vCNN_5_frames": _cnn(5, "network", 75_621_474_400, 414224, "vCNN (5f)", "cornflowerblue"),
    "vCNN_10_frames": _cnn(10, "network", 71_669_073_600, 414624, "vCNN (10f)", "mediumblue"),
    "gCNN_2_frames": dict(kind="gcnn", n_frames=2, state_key="state_dict",
                          flops=78_796_022_580, params=414064, label="gCNN", color="lightblue"),
    "RAFTsmall": _raft(12, True, 6_138_082_963_584, 1_000_000, "RAFTsmall", "darkorange"),
    "RAFT_1": _raft(1, False, 9_028_616_186_304, 5_257_512, "RAFT-1", "lightcoral"),
    "RAFT_6": _raft(6, False, 17_380_072_854_144, 5_257_512, "RAFT-6", "chocolate"),
    "RAFT_12": _raft(12, False, 27_401_820_855_552, 5_257_512, "RAFT-12", "crimson"),
    **{f"RAFT_{i}_S": _raft(i, False, f, 5_257_512, f"RAFT-{i}-S", c, kind="raft_hex", ckpt=RAFT_HEX_CKPT)
       for i, f, c in [(1, 3_331_921_342_464, "plum"), (6, 6_494_603_157_504, "orchid"),
                       (12, 10_289_821_335_552, "darkviolet")]},
}

# Benchmarked networks, in reporting order.
PAPER_NETWORKS = [
    "FlyNet", "sCTRNN", "CTRNN",
    "vCNN_2_frames", "vCNN_5_frames", "vCNN_10_frames", "gCNN_2_frames",
    "RAFTsmall", "RAFT_1", "RAFT_6", "RAFT_12",
    "RAFT_1_S", "RAFT_6_S", "RAFT_12_S",
]

# Old id -> readable id.
LEGACY_IDS = {
    "flynet_synthesized": "FlyNet", "flynet_no_connectome": "CTRNN",
    "flynet_pruned_ctrnn": "sCTRNN",
    "0": "vCNN_2_frames", "5_frames": "vCNN_5_frames", "10_frames": "vCNN_10_frames",
    "global_pooling": "gCNN_2_frames",
    "raft_small": "RAFTsmall", "raft_large_no_recurrence": "RAFT_1",
    "raft_large_half_recurrence": "RAFT_6", "raft_large": "RAFT_12",
    "raft_hexsintel_large_no_recurrence": "RAFT_1_S", "raft_hexsintel_large_half_recurrence": "RAFT_6_S",
    "raft_hexsintel_large": "RAFT_12_S",
}


def spec(network_id):
    """Return the registry spec, accepting readable or legacy ids."""
    if network_id in NETWORKS:
        return NETWORKS[network_id]
    if network_id in LEGACY_IDS:
        return NETWORKS[LEGACY_IDS[network_id]]
    raise KeyError(f"Unknown network id: {network_id}. Known: {list(NETWORKS)}")


def canonical(network_id):
    """Readable id for a readable or legacy id (the one used in paths/plots)."""
    if network_id in NETWORKS:
        return network_id
    s = spec(network_id)
    return next(k for k, v in NETWORKS.items() if v is s)


def legacy_dir_name(network_id):
    """Result-tree folder name used by the original scripts: the legacy id,
    prefixed by ``cnn_`` for everything but the flyvis networks."""
    cid = canonical(network_id)
    old = next((k for k, v in LEGACY_IDS.items() if v == cid), None)
    if old is None:
        return cid
    return old if NETWORKS[cid]["kind"] == "flynet" else f"cnn_{old}"


def resolve_networks(names):
    """Expand a list of ids; 'all' means PAPER_NETWORKS."""
    out = []
    for name in names:
        out += PAPER_NETWORKS if name == "all" else [canonical(name)]
    return list(dict.fromkeys(out))


def kind(network_id):
    return spec(network_id)["kind"]


def is_cnn(network_id):
    """True for CNN/RAFT networks (anything that is not FlyNet/CTRNN)."""
    return spec(network_id)["kind"] != "flynet"


def n_frames(network_id):
    """Number of input frames the network stacks (None for FlyNet/CTRNN)."""
    return spec(network_id)["n_frames"]


def flops(network_id):
    return spec(network_id)["flops"]


def free_parameters(network_id):
    return spec(network_id)["params"]


def label(network_id):
    return spec(network_id)["label"]


def color(network_id):
    return spec(network_id)["color"]


def flyvis_id(network_id):
    return spec(network_id)["flyvis"]


def raft_iters(network_id):
    return spec(network_id)["iters"]


def checkpoint_path(network_id):
    """Path to a CNN/gCNN/RAFT checkpoint under CHECKPOINT_DIR / RAFT_MODEL_DIR."""
    s = spec(network_id)
    if s["kind"] in ("raft", "raft_hex"):
        fname = s["ckpt"] or ("raft-small.pth" if s["small"] else "raft-sintel.pth")
        return os.path.join(RAFT_MODEL_DIR, fname)
    return os.path.join(CHECKPOINT_DIR, canonical(network_id), "network.chkpt")


def decoder_path(network_id, kind_="hit", base=None):
    """Readable decoder path; falls back to the legacy layout if present."""
    base = base or DECODER_DIR
    ext = "pkl" if kind_ == "hit" else "pth"
    path = os.path.join(base, canonical(network_id), f"{kind_}.{ext}")
    if os.path.exists(path):
        return path
    legacy = _legacy_decoder_path(network_id, kind_)
    return legacy if legacy and os.path.exists(legacy) else path


def _legacy_decoder_path(network_id, kind_="hit"):
    """Reconstruct the original flat decoder path (pre-reorg)."""
    s = spec(network_id)
    if s["kind"] == "flynet":
        tag = s.get("legacy")
        if tag is None:
            return None
        if kind_ == "hit":
            return f"mlp_models{tag}/model_hit{tag}.pkl"
        return f"{'velocity' if kind_ == 'velocity' else 'position'}_decoders/" \
               f"{'velocity' if kind_ == 'velocity' else 'position'}_decoder{tag}.pth"
    for old, new in LEGACY_IDS.items():
        if NETWORKS.get(new) is s and kind_ == "hit":
            return f"mlp_models_cnn_{old}/model_hit_cnn_{old}.pkl"
    return None


# Input size per network, used to scale the noise.
INPUT_DIM = {"flynet": 721, "cnn": 721, "gcnn": 721, "raft": 416 ** 2, "raft_hex": 721}


def input_dim(network_id):
    return INPUT_DIM[kind(network_id)]


# Perturbation -> (axis label, magnitude key).
PERTURBATION_AXIS = {
    "AWN_texture": (r"Noise strength, $\sigma$", "sigma"),
    "AWN_frames": (r"Noise strength, $\sigma$", "sigma"),
    "Flicker": (r"Flickering strength, $\epsilon$", "strength"),
    "Fog": (r"Contrast scale, $\alpha$", "contrast_scale"),
    "Shot_noise": (r"Photon count, $c$", "photon_count"),
    "Motion_jitter": (r"Jitter, $j$", "jitter"),
}

# Default looming stimulus.
DEFAULT_STIMULUS = dict(
    hit=True, radius=4.0, shape="Sphere", theta=0, time=1, fps=100,
    stop_margin_hit=0.6, z0=15, out_size=416, return_rgb=True,
)
