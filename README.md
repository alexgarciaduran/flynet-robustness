# Measuring Robustness and Efficiency in a Connectome-Constrained Fly Visual System Model

Code for the collision-detection robustness/efficiency benchmark built around
**FlyNet** (a connectome-constrained, task-optimized model of the *Drosophila*
optic lobe). The pipeline renders looming stimuli in a differentiable 3D scene,
runs them through FlyNet and a set of baselines (CTRNN, vCNN, gCNN, RAFT),
decodes hit/miss and object position/velocity, and evaluates robustness under
adversarial attacks and common video corruptions relative to computational cost.

## Overview

- **Differentiable 3D pipeline** — textured objects on parametric looming
  trajectories, rendered with PyTorch3D and sampled onto a hexagonal lattice.
- **Networks** — FlyNet and CTRNN (`flyvis`), vanilla/global-pooling hex CNNs,
  and RAFT optical-flow baselines.
- **Attacks** — PGD, Carlini-Wagner, NES, and counterfactual restoration on the
  object texture.
- **Perturbations** — additive white noise, shot noise, flicker, motion jitter,
  fog/contrast.
- **Illusion** — reverse-phi contrast inversion.
- **Metrics** — robustness `R` (area under the accuracy curve) and the
  log-efficiency robustness score `LERS = R / log10(FLOPs)`.
- **Neural analyses** — structural/weighted recurrence, structural/functional
  locality, and representational similarity.

## Installation

```bash
git clone <repo-url>
cd flynet-robustness
pip install -e .
```

Three dependencies are environment-specific and installed separately:

| Dependency | Purpose | Link |
|---|---|---|
| `pytorch3d` | differentiable 3D rendering | https://github.com/facebookresearch/pytorch3d |
| `flyvis` | FlyNet / CTRNN | https://github.com/TuragaLab/flyvis |
| RAFT | optical-flow baseline | https://github.com/princeton-vl/RAFT |

Install `torch`/`torchvision` for your CUDA/CPU setup first, then PyTorch3D and
flyvis per their instructions. For the RAFT baselines, clone the RAFT repo and
add it to `PYTHONPATH`, with `raft-small.pth` / `raft-sintel.pth` under
`models/`.

The lightweight parts (textures, metrics, perturbation ops, neural analyses)
run with only the core scientific stack — verify with:

```bash
python scripts/smoke_test.py
```

## Repository layout

```
flynet_robustness/
  rendering.py           differentiable looming-stimulus pipeline (PyTorch3D)
  textures.py            object textures (Black/White/Checkerboard/Wavy/Random)
  hex_eye.py             cartesian -> hexagonal transduction (BoxEye)
  models.py              FlyNet / CTRNN loading and forward passes
  cnn.py                 vanilla (vCNN) and global-pooling (gCNN) hex CNNs
  feature_extraction.py  CNN/RAFT loading and decoder-feature extraction
  decoders.py            hit/miss classifier + position/velocity regressors
  perturbations.py       common video perturbations and stimulus->input helpers
  reverse_phi.py         reverse-phi illusion
  metrics.py             robustness R, bootstrap, LERS
  config.py              network ids, FLOPs/params, colors, default stimulus
  dataset.py             simulation generation and decoder training
  attacks/
    pgd.py  cw.py  nes.py  counterfactual.py  common.py
  analysis/
    recurrence.py        cycle length, weighted recurrence, SCC metrics
    locality.py          edge-distance and functional-perturbation locality
    representational.py   cosine similarity and PCA
notebooks/               worked examples (see below)
scripts/smoke_test.py    dependency-light self-test
data/configs/            fixed train/test trajectory configurations
```

## Quickstart

```python
import torch
from flynet_robustness.models import get_flynet, get_stimulus_and_return_activity
from flynet_robustness.decoders import load_decoder
from flynet_robustness.textures import get_texture

device = "cuda" if torch.cuda.is_available() else "cpu"
model1, model2, idxs1, idxs2 = get_flynet("flynet_synthesized", device=device)
decoder = load_decoder(cnn=False, cnn_id="flynet_synthesized")

activity, speed, angle, position = get_stimulus_and_return_activity(
    model1, idxs1, model2, idxs2, hit=True, shape="Sphere",
    texture=get_texture("Black"), time=1.0, device=device)
p_hit = torch.sigmoid(decoder(activity.flatten(end_dim=1)))
```

## Notebooks

| Notebook | Contents |
|---|---|
| `01_generate_stimuli` | render looming trajectories, angular size, GIFs |
| `02_forward_and_decode` | load FlyNet, run forward, decode p(hit) and position/velocity |
| `03_video_perturbations` | AWN, shot, flicker, jitter, fog; robustness `R` and `LERS` |
| `04_adversarial_attacks` | PGD, CW, NES, counterfactual restoration |
| `05_reverse_phi_and_neural` | reverse-phi + recurrence, locality, representational similarity |

## Network identifiers

| id | model | FLOPs | free params |
|---|---|---|---|
| `flynet_synthesized` | FlyNet | 1.86e8 | 734 |
| `flynet_no_connectome` | CTRNN | 7.83e8 | 11,593 |
| `0` | vCNN | 7.88e10 | 414,064 |
| `global_pooling` | gCNN | 7.88e10 | 414,064 |
| `raft_small` | RAFTsmall | 6.14e12 | 1.0M |
| `raft_large_no_recurrence` | RAFT-1 | 9.03e12 | 5.3M |
| `raft_large_half_recurrence` | RAFT-6 | 1.74e13 | 5.3M |
| `raft_large` | RAFT-12 | 2.74e13 | 5.3M |

## Data and checkpoints

Trained decoders, network checkpoints and rendered simulations are not tracked
here. `data/configs/{train,test}_configuration.pt` hold the fixed trajectory
parameters (radius, theta, z0, duration, shape, texture, label); regenerate
activity and train decoders with `flynet_robustness.dataset`.

## Citation

Anonymous. *Measuring Robustness and Efficiency in a Connectome-Constrained Fly
Visual System Model on a Collision-Detection Task.* NeurIPS 2026 (under review).

## License

MIT — see [LICENSE](LICENSE).
