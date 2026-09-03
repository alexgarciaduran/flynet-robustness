# Measuring Robustness and Efficiency in a Connectome-Constrained Fly Visual System Model

Code for the collision-detection robustness/efficiency benchmark built around
**FlyNet** (a connectome-constrained, task-optimized model of the *Drosophila*
optic lobe). The pipeline renders looming stimuli in a differentiable 3D scene,
runs them through FlyNet and a set of baselines (CTRNN, vCNN, gCNN, RAFT),
decodes hit/miss and object position/velocity, and evaluates robustness under
adversarial attacks and common video corruptions relative to computational cost.

## Overview

- **Differentiable 3D pipeline** — textured objects on parametric looming
  trajectories (PyTorch3D), sampled onto a hexagonal lattice.
- **Networks** — FlyNet and CTRNN (`flyvis`), vanilla/global-pooling hex CNNs
  (2/5/10 input frames), and RAFT optical-flow baselines.
- **Attacks** — PGD, Carlini-Wagner, NES, counterfactual restoration.
- **Perturbations** — additive white noise, shot noise, flicker, motion jitter,
  fog; runnable as an attack (sweep → accuracy → robustness `R`).
- **Illusion** — reverse-phi contrast inversion.
- **Metrics** — robustness `R` (area under the accuracy curve) and
  `LERS = R / log10(FLOPs)`.
- **Neural analyses** — recurrence, locality, representational similarity,
  position/velocity decoding.
- **Training & plotting** — decoder training and the main paper figures.

## Installation

```bash
git clone https://github.com/alexgarciaduran/flynet-robustness.git
cd flynet-robustness
pip install -e .
```

Environment-specific dependencies (install separately):

| Dependency | Purpose | Link |
|---|---|---|
| `pytorch3d` | differentiable 3D rendering | https://github.com/facebookresearch/pytorch3d |
| `flyvis` | FlyNet / CTRNN | https://github.com/TuragaLab/flyvis |
| RAFT | optical-flow baseline (on `PYTHONPATH`) | https://github.com/princeton-vl/RAFT |

The lightweight parts (registry, textures, metrics, perturbation ops,
plotting, training, neural analyses) run with only the core stack — verify:

```bash
python scripts/smoke_test.py
```

## Networks

Readable ids are the public API everywhere (loaders, decoders, attacks,
plotting). The registry (`flynet_robustness.config.NETWORKS`) maps each id to
its loading details, cost and display metadata; legacy ids still resolve.

| id | model | frames | FLOPs | params |
|---|---|---|---|---|
| `FlyNet` | connectome-constrained | – | 1.86e8 | 734 |
| `CTRNN` | unconstrained, same dynamics | – | 7.83e8 | 11,593 |
| `vCNN_2_frames` | vanilla hex CNN | 2 | 7.88e10 | 414,064 |
| `vCNN_5_frames` | vanilla hex CNN | 5 | 7.88e10 | 414,224 |
| `vCNN_10_frames` | vanilla hex CNN | 10 | 7.88e10 | 414,624 |
| `gCNN_2_frames` | global-pooling hex CNN | 2 | 7.88e10 | 414,064 |
| `RAFTsmall` | RAFT (small) | 2 | 6.14e12 | 1.0M |
| `RAFT_1` / `RAFT_6` / `RAFT_12` | RAFT (1/6/12 iters) | 2 | 9.03e12 – 2.74e13 | 5.3M |

The CNN stacks `frames` consecutive frames as input channels; `config.n_frames(id)`
is used consistently at loading (model input channels, checkpoint path,
state-dict key) and in the forward (`extract_features`). Legacy → readable id
map: `config.LEGACY_IDS`.

## Repository layout

```
flynet_robustness/
  rendering.py           differentiable looming-stimulus pipeline (PyTorch3D)
  textures.py            object textures
  hex_eye.py             cartesian -> hexagonal transduction (BoxEye)
  config.py              network registry, costs, paths, plotting metadata
  models.py              FlyNet / CTRNN loading and forward passes
  cnn.py                 vanilla (vCNN) and global-pooling (gCNN) hex CNNs
  feature_extraction.py  CNN/RAFT loading + decoder-feature extraction (n_frames)
  decoders.py            hit/miss classifier + position/velocity regressors
  training.py            train hit/miss and position/velocity decoders
  perturbations.py       common video perturbations and stimulus->input helpers
  evaluation.py          run a perturbation sweep as an attack (accuracy, R, LERS)
  reverse_phi.py         reverse-phi illusion
  metrics.py             robustness R, bootstrap, LERS
  plotting.py            accuracy curves, R-vs-FLOPs, LERS bars, transfer, decoding
  dataset.py             simulation generation
  attacks/               pgd, cw, nes, counterfactual (+ common)
  analysis/              recurrence, locality, representational, decoding
notebooks/               worked examples (01-06)
scripts/                 smoke_test.py, download_assets.py
checkpoints/, decoders/  asset folders (weights not tracked; see their READMEs)
data/configs/            fixed train/test trajectory configurations
```

## Quickstart

```python
import torch
from flynet_robustness.models import get_flynet, get_stimulus_and_return_activity
from flynet_robustness.decoders import load_decoder
from flynet_robustness.textures import get_texture

device = "cuda" if torch.cuda.is_available() else "cpu"
model1, model2, idxs1, idxs2 = get_flynet("FlyNet", device=device)
decoder = load_decoder("FlyNet")

activity, *_ = get_stimulus_and_return_activity(
    model1, idxs1, model2, idxs2, hit=True, shape="Sphere",
    texture=get_texture("Black"), time=1.0, device=device)
p_hit = torch.sigmoid(decoder(activity.flatten(end_dim=1)))
```

Attacks and perturbation evaluation take `(net, decoder, network_id)`, where
`net` is `(model1, idxs1, model2, idxs2)` for FlyNet/CTRNN or a CNN/RAFT
feature-extractor module:

```python
from flynet_robustness.attacks import run_pgd
from flynet_robustness.evaluation import evaluate_perturbation

net = (model1, idxs1, model2, idxs2)
run_pgd(net, decoder, "FlyNet", norm="l2", eps=1000, steps=100, device=device)
res = evaluate_perturbation(net, decoder, "FlyNet", "AWN",
                            magnitudes=[0, 0.02, 0.04, 0.06, 0.08], device=device)
print(res["robustness"], res["lers"])
```

## Notebooks

| Notebook | Contents |
|---|---|
| `01_generate_stimuli` | render trajectories, angular size, GIFs |
| `02_forward_and_decode` | FlyNet forward, p(hit), position/velocity |
| `03_video_perturbations` | perturbations as attack; `R`, `LERS`, plots |
| `04_adversarial_attacks` | PGD, CW, NES, counterfactual restoration |
| `05_reverse_phi_and_neural` | reverse-phi + recurrence/locality/RSA |
| `06_train_decoders` | train hit/miss and position/velocity decoders |

## Assets

Weights are not tracked (they are large — added at final push). Place them per
`checkpoints/README.md` and `decoders/README.md`; FlyNet/CTRNN weights come
from flyvis (`python scripts/download_assets.py`). Decoder paths are resolved
by `config.decoder_path(id, kind)`, which falls back to the legacy flat layout
(`mlp_models_synthesized_model/…`) if present.

## Citation

Anonymous. *Measuring Robustness and Efficiency in a Connectome-Constrained Fly
Visual System Model on a Collision-Detection Task.* NeurIPS 2026 (under review).

## License

MIT — see [LICENSE](LICENSE).
