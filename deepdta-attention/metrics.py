"""Paper metrics: MSE, concordance index, rm^2, AUPR."""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np

ArrayLike = Union[Sequence[float], np.ndarray]


def _ravel(arr: ArrayLike) -> np.ndarray:
    return np.asarray(arr, dtype=np.float64).ravel()


def mse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    yt, yp = _ravel(y_true), _ravel(y_pred)
    return float(np.mean((yt - yp) ** 2))


def concordance_index(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    y, p = _ravel(y_true), _ravel(y_pred)
    n = len(y)
    if n < 2:
        return 0.0
    order = np.argsort(y, kind="mergesort")
    y, p = y[order], p[order]
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
    return 0.0 if n_pairs == 0 else n_correct / n_pairs


def _r2(y_obs: np.ndarray, y_pred: np.ndarray) -> float:
    y_obs_mean = np.mean(y_obs)
    y_pred_mean = np.mean(y_pred)
    num = np.sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean)) ** 2
    den = float(np.sum((y_obs - y_obs_mean) ** 2) * np.sum((y_pred - y_pred_mean) ** 2))
    return 0.0 if den == 0 else float(num / den)


def rm2(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    y_obs, y_hat = _ravel(y_true), _ravel(y_pred)
    r2 = _r2(y_obs, y_hat)
    denom = float(np.sum(y_hat * y_hat))
    k = 0.0 if denom == 0 else float(np.sum(y_obs * y_hat) / denom)
    down = float(np.sum((y_obs - np.mean(y_obs)) ** 2))
    r02 = 0.0 if down == 0 else float(1.0 - np.sum((y_obs - k * y_hat) ** 2) / down)
    return float(r2 * (1.0 - np.sqrt(np.abs(r2 * r2 - r02 * r02))))


def aupr(y_true: ArrayLike, y_pred: ArrayLike, threshold: float) -> float:
    y, p = _ravel(y_true), _ravel(y_pred)
    labels = (y >= threshold).astype(np.float64)
    if labels.min() == labels.max():
        return 0.0
    order = np.argsort(-p, kind="mergesort")
    labels = labels[order]
    tp = np.cumsum(labels)
    fp = np.cumsum(1.0 - labels)
    if tp[-1] <= 0:
        return 0.0
    recall = np.concatenate(([0.0], tp / tp[-1]))
    precision = np.concatenate(([1.0], tp / np.maximum(tp + fp, 1e-12)))
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def evaluate_all(y_true: ArrayLike, y_pred: ArrayLike, aupr_threshold: float) -> dict[str, float]:
    return {
        "mse": mse(y_true, y_pred),
        "ci": concordance_index(y_true, y_pred),
        "rm2": rm2(y_true, y_pred),
        "aupr": aupr(y_true, y_pred, aupr_threshold),
    }
