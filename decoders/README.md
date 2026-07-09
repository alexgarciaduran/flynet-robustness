# Decoders

Trained task decoders, one folder per network id (not tracked; see `.gitignore`).

```
decoders/
  FlyNet/  hit.pkl  position.pth  velocity.pth
  CTRNN/   hit.pkl  ...
  vCNN_2_frames/ hit.pkl
  ...
```

`hit.pkl` is a scikit-learn Pipeline (StandardScaler + MLPClassifier); the
position/velocity `.pth` files are `PositionDecoder` state dicts. Train with
`flynet_robustness.training`. Paths are resolved by
`flynet_robustness.config.decoder_path(network_id, kind)`, which also falls
back to the legacy flat layout (`mlp_models_synthesized_model/…`) if present.
