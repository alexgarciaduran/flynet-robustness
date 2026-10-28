"""
@author: alexgarciaduran and alejandrorg00
"""

import os
import pickle

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .decoders import PositionDecoder


def _split_by_trial(activity_all, labels, test_size=0.1, seed=1):
    """Split trials into train/test, expanding each trial over its timepoints."""
    rng = np.random.default_rng(seed)
    n_trials = len(activity_all)
    n_test = min(max(1, int(round(test_size * n_trials))), n_trials - 1)
    test_trials = set(rng.choice(n_trials, size=n_test, replace=False).tolist())
    Xtr, ytr, Xte, yte, idx_te = [], [], [], [], []
    trial = 0
    for k, (act, lab) in enumerate(zip(activity_all, labels)):
        a = act[0] if act.ndim == 3 else act
        n = a.shape[0]
        y = np.full(n, lab)
        if k in test_trials:
            Xte.append(np.asarray(a)); yte.append(y); idx_te.append(np.arange(trial, trial + n))
        else:
            Xtr.append(np.asarray(a)); ytr.append(y)
        trial += n
    Xtr, ytr = np.concatenate(Xtr), np.concatenate(ytr)
    Xte, yte = np.concatenate(Xte), np.concatenate(yte)
    idx_te = np.concatenate(idx_te)
    perm = rng.permutation(len(Xtr))
    return Xtr[perm], ytr[perm], Xte, yte, idx_te


def _build_classifier(decoder, scale, C, max_iter, penalty):
    if decoder == "mlp":
        clf = MLPClassifier(hidden_layer_sizes=(512,), activation="tanh", solver="adam",
                            alpha=1e-1, batch_size=1024, learning_rate_init=1e-3,
                            max_iter=max_iter, early_stopping=True, n_iter_no_change=10,
                            random_state=0)
    elif decoder == "logistic_regression":
        solver = "saga" if penalty == "l1" else "lbfgs"
        clf = LogisticRegression(penalty=penalty, solver=solver, C=C, max_iter=max_iter,
                                 class_weight="balanced")
    else:
        raise ValueError("decoder must be 'mlp' or 'logistic_regression'")
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)]) if scale else clf


def train_hit_decoder(activity_all, hit_labels, times_duration, network_id="FlyNet",
                      decoder="mlp", scale=True, test_size=0.1, C=0.1, max_iter=50,
                      penalty="l2", save=True, n_bins=20, dt=0.01):
    """Train and save the hit/miss decoder; return time-binned test accuracy."""
    Xtr, ytr, Xte, yte, idx_te = _split_by_trial(activity_all, hit_labels, test_size)
    clf = _build_classifier(decoder, scale, C, max_iter, penalty).fit(Xtr, ytr)

    if save:
        path = config.decoder_path(network_id, "hit")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(clf, f)

    # Windowed nets give T - n_frames + 1 samples per trial, so idx_te must be
    # split by the real lengths; sample j then ends on frame j + T - S.
    lengths = np.array([(a[0] if a.ndim == 3 else a).shape[0] for a in activity_all])
    n_tp = np.round(np.asarray(times_duration) / dt).astype(int)
    starts = np.concatenate(([0], np.cumsum(lengths)[:-1]))
    sim_idx = np.searchsorted(np.cumsum(lengths), idx_te, side="right")
    rel_time = (idx_te - starts[sim_idx] + n_tp[sim_idx] - lengths[sim_idx]) / n_tp[sim_idx]
    bins = np.linspace(0, 1, n_bins + 1)
    acc = np.full(n_bins, np.nan)
    for i in range(n_bins):
        mask = (rel_time >= bins[i]) & (rel_time < bins[i + 1])
        if mask.any():
            acc[i] = np.mean(clf.predict(Xte[mask]) == yte[mask])
    return acc, float(np.mean(clf.predict(Xte) == yte))


def train_position_decoder(X, Y, X_test=None, Y_test=None, network_id="FlyNet",
                           velocity=False, hidden_dims=(128, 16), n_epochs=200,
                           batch_size=1024, lr=1e-3, patience=50, min_delta=1e-5,
                           device="cuda", save=True):
    """Train a position/velocity MLP regressor with early stopping; optionally save."""
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
            criterion(model(xb.to(device)), yb.to(device)).backward()
            opt.step()
        if X_test is not None:
            model.eval()
            with torch.no_grad():
                val = criterion(model(X_test.to(device)), Y_test.to(device)).item()
            if val < best - min_delta:
                best, no_improve = val, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                no_improve += 1
                if no_improve >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    if save:
        path = config.decoder_path(network_id, "velocity" if velocity else "position")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(model.state_dict(), path)
    return model
