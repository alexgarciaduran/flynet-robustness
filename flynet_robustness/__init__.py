"""FlyNet robustness & efficiency benchmark.

Measuring robustness and efficiency in a connectome-constrained fly visual
system model on a collision-detection task.

Lightweight utilities (textures, metrics, config, perturbation ops) are
importable directly. Modules that need flyvis / PyTorch3D / RAFT
(`rendering`, `models`, `cnn`, `feature_extraction`, `attacks`, `dataset`)
are imported explicitly, e.g. ``from flynet_robustness.models import get_flynet``.
"""

__version__ = "0.1.0"

from . import config, metrics, perturbations, textures
from .metrics import lers, robustness
from .textures import TEXTURE_TYPES, get_texture

__all__ = [
    "config", "metrics", "perturbations", "textures",
    "get_texture", "TEXTURE_TYPES", "robustness", "lers", "__version__",
]
