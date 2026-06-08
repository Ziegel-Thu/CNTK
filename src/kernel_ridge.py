"""Closed-form kernel/feature ridge classifiers and margin diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RidgeBinaryResult:
    train_accuracy: float
    test_accuracy: float
    train_margin_mean: float
    train_margin_median: float
    test_margin_mean: float
    test_margin_median: float
    rkhs_norm: float
    source_norm: float


@dataclass(frozen=True)
class RidgeMulticlassResult:
    train_accuracy: float
    test_accuracy: float
    train_margin_mean: float
    train_margin_median: float
    test_margin_mean: float
    test_margin_median: float
    source_norm: float


def _regularized_solve(k_train: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    k_train = np.asarray(k_train, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = k_train.shape[0]
    return np.linalg.solve(k_train + ridge * n * np.eye(n), y)


def binary_margins(scores: np.ndarray, y_sign: np.ndarray) -> np.ndarray:
    return np.asarray(scores, dtype=np.float64).reshape(-1) * np.asarray(y_sign, dtype=np.float64).reshape(-1)


def multiclass_margins(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    true_scores = scores[np.arange(len(labels)), labels]
    masked = scores.copy()
    masked[np.arange(len(labels)), labels] = -np.inf
    other_scores = np.max(masked, axis=1)
    return true_scores - other_scores


def source_norm_binary(k_train: np.ndarray, y_sign: np.ndarray, ridge: float = 1e-6) -> float:
    """Regularized y^T (K + ridge*n I)^-1 y source/RKHS norm proxy."""
    alpha = _regularized_solve(k_train, np.asarray(y_sign, dtype=np.float64).reshape(-1), ridge)
    return float(np.sqrt(max(0.0, float(np.dot(y_sign, alpha)))))


def source_norm_multiclass(k_train: np.ndarray, labels: np.ndarray, ridge: float = 1e-6) -> float:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    classes = np.unique(labels)
    y = np.zeros((len(labels), len(classes)), dtype=np.float64)
    for col, cls in enumerate(classes):
        y[:, col] = labels == cls
    y = y - y.mean(axis=0, keepdims=True)
    alpha = _regularized_solve(k_train, y, ridge)
    value = float(np.sum(y * alpha))
    return float(np.sqrt(max(0.0, value)))


def fit_binary_kernel_ridge(
    k_train: np.ndarray,
    y_train: np.ndarray,
    k_test_train: np.ndarray,
    y_test: np.ndarray,
    ridge: float = 1e-3,
    norm_ridge: float = 1e-6,
) -> RidgeBinaryResult:
    """Fit binary kernel ridge classifier from train and test-train Gram blocks."""
    y_train = np.asarray(y_train, dtype=np.float64).reshape(-1)
    y_test = np.asarray(y_test, dtype=np.float64).reshape(-1)
    alpha = _regularized_solve(k_train, y_train, ridge)
    train_scores = np.asarray(k_train, dtype=np.float64) @ alpha
    test_scores = np.asarray(k_test_train, dtype=np.float64) @ alpha
    train_margins = binary_margins(train_scores, y_train)
    test_margins = binary_margins(test_scores, y_test)
    rkhs_norm = float(np.sqrt(max(0.0, float(alpha @ np.asarray(k_train, dtype=np.float64) @ alpha))))
    return RidgeBinaryResult(
        train_accuracy=float(np.mean(train_margins > 0.0)),
        test_accuracy=float(np.mean(test_margins > 0.0)),
        train_margin_mean=float(np.mean(train_margins)),
        train_margin_median=float(np.median(train_margins)),
        test_margin_mean=float(np.mean(test_margins)),
        test_margin_median=float(np.median(test_margins)),
        rkhs_norm=rkhs_norm,
        source_norm=source_norm_binary(k_train, y_train, ridge=norm_ridge),
    )


def fit_multiclass_kernel_ridge(
    k_train: np.ndarray,
    y_train: np.ndarray,
    k_test_train: np.ndarray,
    y_test: np.ndarray,
    ridge: float = 1e-3,
    norm_ridge: float = 1e-6,
) -> RidgeMulticlassResult:
    """Fit one-hot multiclass kernel ridge classifier."""
    y_train = np.asarray(y_train, dtype=np.int64).reshape(-1)
    y_test = np.asarray(y_test, dtype=np.int64).reshape(-1)
    classes = np.unique(y_train)
    class_to_col = {int(cls): col for col, cls in enumerate(classes)}
    y_train_col = np.asarray([class_to_col[int(label)] for label in y_train], dtype=np.int64)
    y_test_col = np.asarray([class_to_col[int(label)] for label in y_test], dtype=np.int64)
    y_mat = np.zeros((len(y_train), len(classes)), dtype=np.float64)
    for col, cls in enumerate(classes):
        y_mat[:, col] = y_train == cls
    y_mat = y_mat - y_mat.mean(axis=0, keepdims=True)
    alpha = _regularized_solve(k_train, y_mat, ridge)
    train_scores = np.asarray(k_train, dtype=np.float64) @ alpha
    test_scores = np.asarray(k_test_train, dtype=np.float64) @ alpha
    train_pred = classes[np.argmax(train_scores, axis=1)]
    test_pred = classes[np.argmax(test_scores, axis=1)]
    train_margins = multiclass_margins(train_scores, y_train_col)
    test_margins = multiclass_margins(test_scores, y_test_col)
    return RidgeMulticlassResult(
        train_accuracy=float(np.mean(train_pred == y_train)),
        test_accuracy=float(np.mean(test_pred == y_test)),
        train_margin_mean=float(np.mean(train_margins)),
        train_margin_median=float(np.median(train_margins)),
        test_margin_mean=float(np.mean(test_margins)),
        test_margin_median=float(np.median(test_margins)),
        source_norm=source_norm_multiclass(k_train, y_train, ridge=norm_ridge),
    )


def to_jsonable(result: RidgeBinaryResult | RidgeMulticlassResult) -> dict[str, float]:
    return {key: float(value) for key, value in result.__dict__.items()}
