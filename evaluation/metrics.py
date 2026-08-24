"""Evaluation metrics from the original DeepDTA `emetrics.py` plus vectorized CI."""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np

ArrayLike = Union[Sequence[float], np.ndarray]


def _ravel(arr: ArrayLike) -> np.ndarray:
    if hasattr(arr, "A"):
        arr = arr.A
    return np.asarray(arr, dtype=np.float64).ravel()


def mse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean squared error; also the training loss and early-stopping criterion."""
    yt = _ravel(y_true)
    yp = _ravel(y_pred)
    return float(np.mean((yt - yp) ** 2))


def get_cindex(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Pairwise concordance index (ranking): P(pred order matches true order). Slow O(n^2) reference."""
    y = _ravel(y_true)
    p = _ravel(y_pred)
    summ = 0.0
    pair = 0
    for i in range(1, len(y)):
        for j in range(i):
            if y[i] > y[j]:
                pair += 1
                summ += 1.0 * (p[i] > p[j]) + 0.5 * (p[i] == p[j])
    if pair == 0:
        return 0.0
    return summ / pair


def concordance_index(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Same CI as ``get_cindex``, faster numpy inner loop. Used during training logs."""
    y = _ravel(y_true)
    p = _ravel(y_pred)
    n = len(y)
    if n < 2:
        return 0.0

    order = np.argsort(y, kind="mergesort")
    y = y[order]
    p = p[order]

    n_pairs = 0.0
    n_correct = 0.0
    for i in range(1, n):
        smaller = y[:i] < y[i]
        count = int(smaller.sum())
        if count == 0:
            continue
        n_pairs += count
        preds_j = p[:i][smaller]
        n_correct += float((p[i] > preds_j).sum()) + 0.5 * float((p[i] == preds_j).sum())

    if n_pairs == 0:
        return 0.0
    return n_correct / n_pairs


def r_squared_error(y_obs: ArrayLike, y_pred: ArrayLike) -> float:
    y_obs = _ravel(y_obs)
    y_pred = _ravel(y_pred)
    y_obs_mean = np.mean(y_obs)
    y_pred_mean = np.mean(y_pred)
    mult = np.sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult
    y_obs_sq = np.sum((y_obs - y_obs_mean) ** 2)
    y_pred_sq = np.sum((y_pred - y_pred_mean) ** 2)
    denom = float(y_obs_sq * y_pred_sq)
    if denom == 0:
        return 0.0
    return float(mult / denom)


def get_k(y_obs: ArrayLike, y_pred: ArrayLike) -> float:
    y_obs = _ravel(y_obs)
    y_pred = _ravel(y_pred)
    denom = float(np.sum(y_pred * y_pred))
    if denom == 0:
        return 0.0
    return float(np.sum(y_obs * y_pred) / denom)


def squared_error_zero(y_obs: ArrayLike, y_pred: ArrayLike) -> float:
    k = get_k(y_obs, y_pred)
    y_obs = _ravel(y_obs)
    y_pred = _ravel(y_pred)
    y_obs_mean = np.mean(y_obs)
    upp = np.sum((y_obs - (k * y_pred)) ** 2)
    down = np.sum((y_obs - y_obs_mean) ** 2)
    if down == 0:
        return 0.0
    return float(1.0 - (upp / down))


def get_rm2(ys_orig: ArrayLike, ys_line: ArrayLike) -> float:
    """Roy's rm^2: R² penalized when the origin-forced fit (r0²) disagrees."""
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)
    return float(r2 * (1.0 - np.sqrt(np.abs((r2 * r2) - (r02 * r02)))))


def _average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Binary average precision without scikit-learn (sklearn-compatible step integral)."""
    order = np.argsort(-y_score, kind="mergesort")
    y_true = y_true[order]
    tp = np.cumsum(y_true)
    fp = np.cumsum(1.0 - y_true)
    n_pos = tp[-1]
    if n_pos <= 0:
        return 0.0
    recall = tp / n_pos
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = np.concatenate(([0.0], recall))
    precision = np.concatenate(([1.0], precision))
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def get_aupr(y_true: ArrayLike, y_pred: ArrayLike, threshold: float) -> float:
    """AUPR after binarizing affinities at ``threshold`` (paper: Davis 7, KIBA 12.1)."""
    y = _ravel(y_true)
    p = _ravel(y_pred)
    labels = (y >= threshold).astype(np.float64)
    if labels.min() == labels.max():
        return 0.0
    return _average_precision(labels, p)


def evaluate_all(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    aupr_threshold: float,
) -> dict[str, float]:
    """Paper metrics on a split: MSE, CI, rm2, and AUPR after binarizing at ``aupr_threshold``."""
    y_true = _ravel(y_true)
    y_pred = _ravel(y_pred)
    return {
        "mse": mse(y_true, y_pred),
        "ci": concordance_index(y_true, y_pred),
        "rm2": get_rm2(y_true, y_pred),
        "aupr": get_aupr(y_true, y_pred, aupr_threshold),
    }
