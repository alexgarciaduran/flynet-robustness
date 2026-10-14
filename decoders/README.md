# Decoders

Not tracked. One folder per network id; train them with notebook 06.

```
decoders/
  FlyNet/   hit.pkl  position.pth  velocity.pth
  CTRNN/    hit.pkl  position.pth  velocity.pth
  <every other id>/   hit.pkl
```

`hit.pkl` is a scikit-learn Pipeline (StandardScaler + MLPClassifier), the
`.pth` files are `PositionDecoder` state dicts. Each network needs its own,
since the input width differs: 5,768 T4/T5 units for the flyvis networks, 7,688
for the CNNs, 6,272 for RAFT.

Resolved by `config.decoder_path(id, kind)`, which falls back to the old flat
layout (`mlp_models_cnn_raft_large/...`) if present. `$FLYNET_DECODERS`
overrides the root.
