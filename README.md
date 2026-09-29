# Measuring Robustness and Efficiency in a Connectome-Constrained Fly Visual System Model

Code for the collision-detection robustness/efficiency benchmark built around
**FlyNet** (a connectome-constrained, task-optimized model of the *Drosophila*
optic lobe). The pipeline renders looming stimuli in a differentiable 3D scene,
runs them through FlyNet and a set of baselines (CTRNN, sparse CTRNN, degree-matched
FlyNet controls, vCNN, gCNN, RAFT, RAFT on hex-sampled input),
decodes hit/miss and object position/velocity, and evaluates robustness under
adversarial attacks and common video corruptions relative to computational cost.

## Overview

- **Differentiable 3D pipeline** — textured objects on parametric looming
  trajectories (PyTorch3D), sampled onto a hexagonal lattice.
- **Networks** — FlyNet, CTRNN, sparse CTRNN and 10 degree-matched FlyNet
  controls (`flyvis`), vanilla/global-pooling hex CNNs (2/5/10 input frames),
  and RAFT optical-flow baselines (Sintel RGB, and hex-sampled Sintel: RAFT-S).
- **Attacks** — PGD, Carlini-Wagner, NES, counterfactual restoration, and
  texture optimization of the input's 2-point (Hassenstein–Reichardt) correlation.
- **Perturbations** — additive white noise (texture / frames), shot noise,
  flicker, motion jitter, fog; runnable as an attack (sweep → accuracy → robustness `R`).
- **Illusion** — reverse-phi contrast inversion.
- **Metrics** — robustness `R` (area under the accuracy curve) and
  `LERS = R / log10(FLOPs)`.
- **Neural analyses** — recurrence, locality, representational similarity,
  position/velocity decoding, illusion mechanism (2-point correlation,
  closeness of the T4/T5 code to reverse-phi).
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
| `pytorch3d` | differentiable 3D rendering (Linux only) | https://github.com/facebookresearch/pytorch3d |
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
| `sCTRNN` | CTRNN pruned to FlyNet's 1,513,231 edges | – | 1.86e8* | – |
| `flynet_degree_matched_0`…`_9` | FlyNet on locality-preserving rewired connectomes | – | 1.86e8* | 734 |
| `vCNN_2_frames` | vanilla hex CNN | 2 | 7.88e10 | 414,064 |
| `vCNN_5_frames` | vanilla hex CNN | 5 | 7.56e10 | 414,224 |
| `vCNN_10_frames` | vanilla hex CNN | 10 | 7.17e10 | 414,624 |
| `gCNN_2_frames` | global-pooling hex CNN | 2 | 7.88e10 | 414,064 |
| `RAFTsmall` | RAFT (small) | 2 | 6.14e12 | 1.0M |
| `RAFT_1` / `RAFT_6` / `RAFT_12` | RAFT (1/6/12 iters) | 2 | 9.03e12 – 2.74e13 | 5.3M |
| `RAFT_1_S` / `RAFT_6_S` / `RAFT_12_S` | RAFT trained on hex-sampled Sintel (FlyNet's input) | 2 | 3.33e12 – 1.03e13 | 5.3M |

\* same nodes and edge count as FlyNet (FLOPs scale with edges).

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
  models.py              flyvis-network loading and forward passes
  cnn.py                 vanilla (vCNN) and global-pooling (gCNN) hex CNNs
  feature_extraction.py  CNN/RAFT loading + decoder-feature extraction (n_frames)
  decoders.py            hit/miss classifier + position/velocity regressors
  training.py            train hit/miss and position/velocity decoders
  perturbations.py       common video perturbations and stimulus->input helpers
  evaluation.py          run a perturbation sweep as an attack (accuracy, R, LERS)
  reverse_phi.py         reverse-phi illusion
  metrics.py             robustness R, bootstrap, LERS
  plotting.py            accuracy curves, R-vs-FLOPs, LERS bars, transfer, decoding
  dataset.py             simulation generation (dataset, illusion conditions)
  attacks/               pgd, cw, nes, counterfactual, correlation (+ common)
  analysis/              recurrence, locality, representational, decoding, illusion
notebooks/               worked examples (01-07)
scripts/                 smoke_test.py, download_assets.py
checkpoints/, decoders/  asset folders (weights not tracked; see their READMEs)
data/configs/            train/test configurations, attack and illusion trajectories
```

## Quickstart

```python
import torch
from flynet_robustness.models import get_flynet, get_stimulus_and_return_activity
from flynet_robustness.decoders import load_decoder
from flynet_robustness.textures import get_texture

device = "cuda" if torch.cuda.is_available() else "cpu"
model1, model2, idxs1, idxs2 = get_flynet("FlyNet", device=device)
decoder = load_decoder("FlyNet", device=device)

activity, *_ = get_stimulus_and_return_activity(
    model1, idxs1, model2, idxs2, hit=True, shape="Sphere",
    texture=get_texture("Black"), time=1.0, device=device)
p_hit = torch.sigmoid(decoder(activity.flatten(end_dim=1).to(device)))
```

Attacks and perturbation evaluation take `(net, decoder, network_id)`, where
`net` is `(model1, idxs1, model2, idxs2)` for flyvis networks or a CNN/RAFT
feature-extractor module:

```python
from flynet_robustness.attacks import run_pgd
from flynet_robustness.evaluation import evaluate_perturbation

net = (model1, idxs1, model2, idxs2)
run_pgd(net, decoder, "FlyNet", norm="l2", eps=1000, steps=100, device=device)
res = evaluate_perturbation(net, decoder, "FlyNet", "AWN_frames", device=device)  # paper sweep
print(res["robustness"], res["lers"])
```

## Notebooks

| Notebook | Contents |
|---|---|
| `01_generate_stimuli` | render trajectories, angular size, GIFs |
| `02_forward_and_decode` | FlyNet forward, p(hit), position/velocity |
| `03_video_perturbations` | perturbations as attack (paper protocol); `R`, `LERS` |
| `04_adversarial_attacks` | PGD, CW, NES sweeps; `R`, `LERS`, CW distances |
| `05_reverse_phi_and_neural` | reverse-phi + recurrence/locality/RSA |
| `06_train_decoders` | train hit/miss and position/velocity decoders |
| `07_illusion_mechanism` | counterfactual restoration, 2-point correlation, closeness to reverse-phi, 2-point texture optimization (rebuttal analyses) |

## Assets

Weights are not tracked (they are large — added at final push). Place them per
`checkpoints/README.md` and `decoders/README.md`. The flyvis networks are set up with

```bash
python scripts/download_assets.py 0003.tar.gz ctrnn_edge_matched.zip flow_0301_best10.zip
```

which downloads the flyvis ensemble, unpacks the CTRNN, sCTRNN and degree-matched
archives into the flyvis results folder, and synthesizes FlyNet
(`flow/0000/synthesized_t4d`: best ensemble model with the T4d parameters of the
2nd best). sCTRNN and the degree-matched controls need a flyvis version with
`ConnectomeWithPrunedEdges` / `ConnectomeWithRewiredEdges`. The illusion analyses
(notebook 07) read their data from `FLYNET_ILLUSION_DATA` (default
`data/illusion`). Decoder paths are resolved
by `config.decoder_path(id, kind)`, which falls back to the legacy flat layout
(`mlp_models_synthesized_model/…`) if present.

## Citation

Anonymous. *Measuring Robustness and Efficiency in a Connectome-Constrained Fly
Visual System Model on a Collision-Detection Task.* NeurIPS 2026 (under review).

## License

MIT — see [LICENSE](LICENSE).
