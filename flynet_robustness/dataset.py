"""Generate the looming dataset and train task decoders."""

import os
import pickle

import numpy as np
import torch
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from .decoders import PositionDecoder
from .models import get_stimulus_and_return_activity
from .textures import get_texture

_STOP_MARGIN = {"Sphere": 0.6, "Cube": 1.0, "Tetrahedron": 0.0}


def seed_everything(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_simulations(model1, idxs1, model2=None, idxs2=None, config_path=None,
                     out_dir="simulations", device="cpu", out_size=416, test=False):
    """Render the fixed trajectory set, run the network and save activity + positions."""
    seed = 1 if test else 0
    np.random.seed(seed)
    data = torch.load(config_path, weights_only=False)
    os.makedirs(out_dir, exist_ok=True)

    activity_all, positions = [], []
    n = len(data["shape_all"])
    for sim in tqdm(range(n)):
        seed_everything(seed=sim)
        shape = data["shape_all"][sim]
        with torch.no_grad():
            activity, _, _, position = get_stimulus_and_return_activity(
                model1, idxs1, model2, idxs2, hit=data["true_labels"][sim],
                radius=data["radius_conditions"][sim], theta=data["theta_conditions"][sim],
                shape=shape, texture=get_texture(data["textures_all"][sim]),
                time=data["time_duration_conditions"][sim], device=device,
                z0=data["z0_conditions"][sim], rotation=data["rotation_conditions"][sim],
                stop_margin_hit=_STOP_MARGIN[shape], image_size=out_size)
        activity_all.append(activity.cpu())
        positions.append(position)

    suffix = "_test" if test else ""
    torch.save({"network_activity_all": activity_all}, f"{out_dir}/activity{suffix}.pt")
    torch.save({"positions": positions}, f"{out_dir}/positions{suffix}.pt")
    return activity_all, positions


def prepare_dataset(activity_all, position_all, t_start=10):
    """Concatenate trials into (samples, neurons) and (samples, 3) tensors."""
    X = torch.cat([a[0][t_start:] for a in activity_all], dim=0)
    Y = torch.cat([p.T[t_start:] for p in position_all], dim=0)
    return X, Y


def train_hit_decoder(X_train, y_train, out_path="mlp_models/model_hit.pkl"):
    """Train the hit/miss MLP classifier (StandardScaler + MLPClassifier)."""
    from sklearn.neural_network import MLPClassifier
    mlp = MLPClassifier(hidden_layer_sizes=(512,), activation="tanh", solver="adam",
                        alpha=1e-1, batch_size=1024, learning_rate_init=1e-3,
                        max_iter=50, early_stopping=True, n_iter_no_change=10, random_state=0)
    clf = Pipeline([("scaler", StandardScaler()), ("clf", mlp)]).fit(X_train, y_train)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(clf, f)
    return clf


def train_position_decoder(X, Y, X_test=None, Y_test=None, hidden_dims=(128, 16),
                           n_epochs=200, batch_size=1024, lr=1e-3, patience=50,
                           min_delta=1e-5, device="cuda"):
    """Train a position/velocity MLP regressor with early stopping."""
    model = PositionDecoder(X.shape[1], hidden_dims=hidden_dims, output_dim=Y.shape[1]).to(device)
    criterion = torch.nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, Y),
                                         batch_size=batch_size, shuffle=True)
    best, no_improve, best_state = float("inf"), 0, None
    for _ in range(n_epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = criterion(model(xb.to(device)), yb.to(device))
            loss.backward()
            opt.step()
        if X_test is not None:
            model.eval()
            with torch.no_grad():
                val = criterion(model(X_test.to(device)), Y_test.to(device)).item()
            if val < best - min_delta:
                best, no_improve, best_state = val, 0, model.state_dict()
            else:
                no_improve += 1
                if no_improve >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model
