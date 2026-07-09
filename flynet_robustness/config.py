"""Central network registry, cost tables, paths and plotting metadata.

Readable network ids are the public API everywhere (models, decoders,
attacks, plotting). Each id maps to a spec holding the internal loading
details (flyvis id or checkpoint path, input frame count, state-dict key),
its cost, and its display label/color.
"""

import os

# Motion cell types decoded from FlyNet/CTRNN (T4/T5, 5768 units).
MOTION_CELL_TYPES = ["T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"]

# Base directories (override via environment variables).
CHECKPOINT_DIR = os.environ.get("FLYNET_CHECKPOINTS", "checkpoints")
DECODER_DIR = os.environ.get("FLYNET_DECODERS", "decoders")
RAFT_MODEL_DIR = os.environ.get("FLYNET_RAFT_MODELS", "models")


def _cnn(n_frames, state_key, flops, params, label, color):
    return dict(kind="cnn", n_frames=n_frames, state_key=state_key,
                flops=flops, params=params, label=label, color=color)


def _raft(iters, small, flops, params, label, color):
    return dict(kind="raft", iters=iters, small=small, n_frames=2,
                flops=flops, params=params, label=label, color=color)


# --- Network registry -------------------------------------------------------
# id -> spec. `flyvis` = flyvis network id; `state_key` = key in the CNN
# checkpoint dict; `legacy` = old identifier for backward-compatible paths.
NETWORKS = {
    "FlyNet": dict(kind="flynet", flyvis="flow/0000/synthesized_t4d", n_frames=None,
                   flops=1.863e8, params=734, label="FlyNet", color="forestgreen",
                   legacy="_synthesized_model"),
    "CTRNN": dict(kind="flynet", flyvis="flow/0003/000", n_frames=None,
                  flops=7.832e8, params=11593, label="CTRNN", color="yellowgreen",
                  legacy="no_connectome"),
    "vCNN_2_frames": _cnn(2, "state_dict", 7.879e10, 414064, "vCNN", "royalblue"),
    "vCNN_5_frames": _cnn(5, "network", 7.879e10, 414224, "vCNN (5f)", "cornflowerblue"),
    "vCNN_10_frames": _cnn(10, "network", 7.879e10, 414624, "vCNN (10f)", "mediumblue"),
    "gCNN_2_frames": dict(kind="gcnn", n_frames=2, state_key="state_dict",
                          flops=7.880e10, params=414064, label="gCNN", color="lightblue"),
    "RAFTsmall": _raft(12, True, 6.138e12, 1_000_000, "RAFTsmall", "darkorange"),
    "RAFT_1": _raft(1, False, 9.029e12, 5_257_512, "RAFT-1", "lightcoral"),
    "RAFT_6": _raft(6, False, 1.738e13, 5_257_512, "RAFT-6", "chocolate"),
    "RAFT_12": _raft(12, False, 2.740e13, 5_257_512, "RAFT-12", "crimson"),
}

# Old id -> readable id, for migrating existing paths/results.
LEGACY_IDS = {
    "flynet_synthesized": "FlyNet", "flynet_no_connectome": "CTRNN",
    "0": "vCNN_2_frames", "global_pooling": "gCNN_2_frames",
    "raft_small": "RAFTsmall", "raft_large_no_recurrence": "RAFT_1",
    "raft_large_half_recurrence": "RAFT_6", "raft_large": "RAFT_12",
}


def spec(network_id):
    """Return the registry spec, accepting readable or legacy ids."""
    if network_id in NETWORKS:
        return NETWORKS[network_id]
    if network_id in LEGACY_IDS:
        return NETWORKS[LEGACY_IDS[network_id]]
    raise KeyError(f"Unknown network id: {network_id}. Known: {list(NETWORKS)}")


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
    if s["kind"] == "raft":
        fname = "raft-small.pth" if s["small"] else "raft-sintel.pth"
        return os.path.join(RAFT_MODEL_DIR, fname)
    canonical = next(k for k, v in NETWORKS.items() if v is s)
    return os.path.join(CHECKPOINT_DIR, canonical, "network.chkpt")


def decoder_path(network_id, kind_="hit", base=None):
    """Readable decoder path; falls back to the legacy layout if present."""
    base = base or DECODER_DIR
    canonical = next(k for k, v in NETWORKS.items() if v is spec(network_id))
    ext = "pkl" if kind_ == "hit" else "pth"
    path = os.path.join(base, canonical, f"{kind_}.{ext}")
    if os.path.exists(path):
        return path
    legacy = _legacy_decoder_path(network_id, kind_)
    return legacy if legacy and os.path.exists(legacy) else path


def _legacy_decoder_path(network_id, kind_="hit"):
    """Reconstruct the original flat decoder path (pre-reorg)."""
    s = spec(network_id)
    if s["kind"] == "flynet":
        tag = s.get("legacy", "")
        if kind_ == "hit":
            return f"mlp_models{tag}/model_hit{tag}.pkl"
        return f"{'velocity' if kind_ == 'velocity' else 'position'}_decoders/" \
               f"{'velocity' if kind_ == 'velocity' else 'position'}_decoder{tag}.pth"
    for old, new in LEGACY_IDS.items():
        if NETWORKS.get(new) is s and kind_ == "hit":
            return f"mlp_models_cnn_{old}/model_hit_cnn_{old}.pkl"
    return None


# Input dimensionality (columns / pixels) per network, for noise normalization.
INPUT_DIM = {"flynet": 721, "cnn": 721, "gcnn": 721, "raft": 416 ** 2}


def input_dim(network_id):
    return INPUT_DIM[kind(network_id)]


# Perturbation name -> (axis label, magnitude key in result dicts).
PERTURBATION_AXIS = {
    "AWN_texture": (r"Noise strength, $\sigma$", "sigma"),
    "AWN_frames": (r"Noise strength, $\sigma$", "sigma"),
    "Flicker": (r"Flickering strength, $\epsilon$", "strength"),
    "Fog": (r"Contrast scale, $\alpha$", "contrast_scale"),
    "Shot_noise": (r"Photon count, $c$", "photon_count"),
    "Motion_jitter": (r"Jitter, $j$", "jitter"),
}

# Default looming stimulus for attacks / perturbations.
DEFAULT_STIMULUS = dict(
    hit=True, radius=4.0, shape="Sphere", theta=0, time=1, fps=100,
    stop_margin_hit=0.6, z0=15, out_size=416, return_rgb=True,
)
