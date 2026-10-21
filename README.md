# flynet-robustness

Robustness and efficiency of FlyNet and a set of optical-flow baselines on a
looming collision-detection task.

## Install

```bash
pip install -e .
```

Plus, separately: [`pytorch3d`](https://github.com/facebookresearch/pytorch3d)
(Linux only), [`flyvis`](https://github.com/TuragaLab/flyvis), and a
[RAFT](https://github.com/princeton-vl/RAFT) checkout on `PYTHONPATH` (only
`core.raft` is needed; the hex-Sintel preprocessing is vendored in
`flynet_robustness/raft_hex.py`).

Check the parts that need none of those:

```bash
python scripts/smoke_test.py
```

## Assets

Decoders and the CNN / RAFT-HexSintel checkpoints:

```bash
python scripts/download_assets.py --weights          # ~560 MB, skips what you have
```

from [huggingface.co/alejandrorg00/flynet-robustness](https://huggingface.co/alejandrorg00/flynet-robustness).
Two sets are not there:

- **vanilla RAFT** (`RAFTsmall`, `RAFT_1/6/12`) — `models.zip` from the RAFT
  authors; put `raft-small.pth` and `raft-sintel.pth` in `models/`
- **CTRNN and sCTRNN** — flyvis archives, supplied separately

Then the flyvis networks, which also synthesises FlyNet:

```bash
python scripts/download_assets.py 0003.tar.gz ctrnn_edge_matched.zip
```

sCTRNN needs a flyvis build with `ConnectomeWithPrunedEdges`.
`$FLYNET_CHECKPOINTS`, `$FLYNET_RAFT_MODELS` and `$FLYNET_DECODERS` override
where these land; set them before importing `flynet_robustness`.

## Networks

`config.PAPER_NETWORKS`, in reporting order:

```
FlyNet  sCTRNN  CTRNN
vCNN_2_frames  vCNN_5_frames  vCNN_10_frames  gCNN_2_frames
RAFTsmall  RAFT_1  RAFT_6  RAFT_12
RAFT_1_S  RAFT_6_S  RAFT_12_S
```

Old ids (`raft_large`, `0`, `flynet_synthesized`, ...) still resolve.

## Run

```bash
# adversarial attacks -> PGD/ CW/ NES/ under --results
python scripts/run_attacks.py --networks all --attack pgd --results results
python scripts/run_attacks.py --networks all --attack cw  --results results
python scripts/run_attacks.py --networks all --attack nes --results results

# transfer: each network's PGD textures on the others (needs the PGD runs above)
python scripts/run_transfer.py --norm both --pgd-root results --results results

# video perturbations -> results/Environmental/*.csv
python scripts/run_perturbations.py --networks all --results results
```

`--networks` also takes any subset of ids. Split a long job into blocks with
`--start-trajectory`. `-h` lists the rest.

The CSVs are `network,aucs,aucs_std,lers,flops`. `data/reference_results/` holds
the same table from the published run, to check a rerun against.

## Notebooks

| | |
|---|---|
| `01_generate_stimuli` | render trajectories, angular size, GIFs |
| `02_forward_and_decode` | forward pass, p(hit), position/velocity |
| `03_video_perturbations` | noise, flicker, jitter, fog sweeps |
| `04_adversarial_attacks` | PGD, CW, NES |
| `05_reverse_phi_and_neural` | reverse-phi, recurrence, locality, RSA |
| `06_train_decoders` | generate activity, train the decoders |
| `07_illusion_mechanism` | what the adversarial texture does to the motion code |
| `08_transfer_attacks` | transfer matrix across networks |

Run them from the repository root.

## License

MIT — see [LICENSE](LICENSE).
