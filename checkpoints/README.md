# Checkpoints

Not tracked. FlyNet, CTRNN and sCTRNN come from flyvis
(`scripts/download_assets.py`); the rest go here:

```
checkpoints/
  vCNN_2_frames/network.chkpt      # state_dict key: "state_dict"
  vCNN_5_frames/network.chkpt      # state_dict key: "network"
  vCNN_10_frames/network.chkpt     # state_dict key: "network"
  gCNN_2_frames/network.chkpt      # state_dict key: "state_dict"

models/
  raft-small.pth                   # RAFTsmall
  raft-sintel.pth                  # RAFT_1 / RAFT_6 / RAFT_12
  raft-sintel-flyvis-split-hex-lum-gridflow-256-bs8-50k_20260615-124452.pth
                                   # RAFT_1_S / RAFT_6_S / RAFT_12_S
```

Resolved by `config.checkpoint_path(id)`; `$FLYNET_CHECKPOINTS` and
`$FLYNET_RAFT_MODELS` override the two roots.
