"""Task decoders: hit/miss classifier (MLP) and position/velocity regressors."""

import pickle

import torch
from torch import nn

from . import config


class TorchMLP(nn.Module):
    """Frozen torch port of a scikit-learn StandardScaler + MLPClassifier pipeline."""

    def __init__(self, sklearn_pipeline):
        super().__init__()
        scaler = sklearn_pipeline.named_steps["scaler"]
        mlp = sklearn_pipeline.named_steps["clf"]
        self.register_buffer("mean", torch.tensor(scaler.mean_, dtype=torch.float32))
        self.register_buffer("scale", torch.tensor(scaler.scale_, dtype=torch.float32))
        (W1, W2), (b1, b2) = mlp.coefs_, mlp.intercepts_
        self.fc1 = nn.Linear(W1.shape[0], W1.shape[1])
        self.fc2 = nn.Linear(W2.shape[0], W2.shape[1])
        self.fc1.weight.data = torch.tensor(W1.T, dtype=torch.float32)
        self.fc1.bias.data = torch.tensor(b1, dtype=torch.float32)
        self.fc2.weight.data = torch.tensor(W2.T, dtype=torch.float32)
        self.fc2.bias.data = torch.tensor(b2, dtype=torch.float32)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = (x - self.mean) / self.scale
        return self.fc2(torch.tanh(self.fc1(x)))


class PositionDecoder(nn.Module):
    """MLP regressor for 3D object position or velocity."""

    def __init__(self, input_dim, hidden_dims=(128, 16), output_dim=3):
        super().__init__()
        layers, last = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers.append(nn.Linear(last, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def load_decoder(network_id="FlyNet", path=None, device=None):
    """Load the hit/miss classifier for a network as a frozen TorchMLP.

    `device` must match the device the activity lives on; None keeps torch's
    default (importing flyvis sets it to cuda).
    """
    path = path or config.decoder_path(network_id, "hit")
    with open(path, "rb") as f:
        pipeline = pickle.load(f)
    model = TorchMLP(pipeline).eval()
    return model if device is None else model.to(device)


def load_decoder_pos_vel(network_id="FlyNet", load_velocity=True, input_dim=5768,
                         device="cuda", path=None):
    """Load a position or velocity regressor for a network."""
    kind = "velocity" if load_velocity else "position"
    path = path or config.decoder_path(network_id, kind)
    model = PositionDecoder(input_dim=input_dim, hidden_dims=(128, 16), output_dim=3)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()


def to_logits(p, eps=1e-4):
    """Inverse sigmoid of a probability/texture in (0,1)."""
    p = p.clamp(eps, 1.0 - eps)
    return torch.log(p / (1.0 - p))
