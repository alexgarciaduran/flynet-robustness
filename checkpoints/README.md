# Checkpoints

Place CNN / gCNN / RAFT weights here (not tracked; see `.gitignore`).

```
checkpoints/
  vCNN_2_frames/network.chkpt      # state_dict key: "state_dict"
  vCNN_5_frames/network.chkpt      # state_dict key: "network"
  vCNN_10_frames/network.chkpt     # state_dict key: "network"
  gCNN_2_frames/network.chkpt      # state_dict key: "state_dict"
```

RAFT weights go in `../models/` as `raft-small.pth` / `raft-sintel.pth`.
FlyNet / CTRNN weights are managed by flyvis (see `scripts/download_assets.py`).
Paths are resolved by `flynet_robustness.config.checkpoint_path(network_id)`.
