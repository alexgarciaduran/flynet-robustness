"""
@author: alexgarciaduran
"""

import numpy as np


def r2_score(y_true, y_pred):
    """Overall coefficient of determination."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean(0)) ** 2).sum()
    return 1.0 - ss_res / ss_tot


def decoding_error(y_true, y_pred, per_timepoint=True):
    """Euclidean decoding error (per timepoint or mean)."""
    err = np.linalg.norm(np.asarray(y_true) - np.asarray(y_pred), axis=-1)
    return err if per_timepoint else float(err.mean())


def error_over_time(decoder_fn, activity, positions):
    """Mean decoding error at each timepoint across trials.

    decoder_fn(activity_matrix) -> predictions; activity/positions are lists
    of aligned (T, .) arrays.
    """
    errs = []
    for act, pos in zip(activity, positions):
        pred = decoder_fn(np.asarray(act))
        errs.append(decoding_error(np.asarray(pos), pred))
    T = min(len(e) for e in errs)
    return np.stack([e[:T] for e in errs]).mean(axis=0)
