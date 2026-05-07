from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_squared_error


def rmse(y_true, y_pred) -> float:
    """RMSE

    Args:
        y_true: Истинные значения.
        y_pred: Предсказанные значения.

    Returns:
        Значение RMSE.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(mean_squared_error(yt, yp)))


def wape(y_true, y_pred) -> float:
    """WAPE в процентах

    Args:
        y_true: Истинные значения.
        y_pred: Предсказанные значения.

    Returns:
        Значение WAPE, %.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    denom = np.sum(np.abs(yt))
    if denom == 0:
        return float("nan")
    return float(np.sum(np.abs(yt - yp)) / denom * 100)


def mae(y_true, y_pred) -> float:
    """MAE

    Args:
        y_true: Истинные значения.
        y_pred: Предсказанные значения.

    Returns:
        Значение MAE.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(yt - yp)))


def round_pred_counts(y_pred) -> np.ndarray:
    """Округление до целых счётчиков без отрицательных значений.

    Args:
        y_pred: Предсказания (вещественные).

    Returns:
        Целочисленные предсказания (numpy array).
    """
    return np.maximum(0, np.round(np.asarray(y_pred, dtype=float))).astype(np.int64)


def accuracy_zero_nonzero(y_true, y_pred) -> float:
    """Accuracy для задачи «0 / не 0»

    Args:
        y_true: Истинные значения.
        y_pred: Предсказанные значения.

    Returns:
        Значение accuracy.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = round_pred_counts(y_pred)
    true_nz = yt > 0
    pred_nz = yp > 0
    return float(np.mean(true_nz == pred_nz))


def precision_zero_nonzero(y_true, y_pred) -> float:
    """Precision положительного класса

    Args:
        y_true: Истинные значения.
        y_pred: Предсказанные значения.

    Returns:
        Значение precision.
    """
    yt = np.asarray(y_true, dtype=float) > 0
    yp = round_pred_counts(y_pred) > 0
    tp = np.sum(yp & yt)
    fp = np.sum(yp & (~yt))
    denom = tp + fp
    return float(tp / denom) if denom > 0 else 0.0
