"""Kernel and Gram-matrix utilities."""

from __future__ import annotations

import numpy as np


def pairwise_sq_dists(x: np.ndarray, z: np.ndarray | None = None) -> np.ndarray:
    """Return pairwise squared Euclidean distances."""
    x = np.asarray(x, dtype=np.float64)
    z = x if z is None else np.asarray(z, dtype=np.float64)
    x_norm = np.sum(x * x, axis=1, keepdims=True)
    z_norm = np.sum(z * z, axis=1, keepdims=True).T
    d2 = x_norm + z_norm - 2.0 * x @ z.T
    return np.maximum(d2, 0.0)


def median_bandwidth(x: np.ndarray, eps: float = 1e-12) -> float:
    """Median pairwise distance bandwidth, excluding diagonal zeros."""
    d2 = pairwise_sq_dists(x)
    tri = d2[np.triu_indices_from(d2, k=1)]
    tri = tri[tri > eps]
    if tri.size == 0:
        return 1.0
    return float(np.sqrt(np.median(tri)))


def rbf_kernel(x: np.ndarray, bandwidth: float | None = None) -> np.ndarray:
    """Gaussian RBF kernel."""
    sigma = median_bandwidth(x) if bandwidth is None else float(bandwidth)
    d2 = pairwise_sq_dists(x)
    return np.exp(-d2 / (2.0 * sigma * sigma + 1e-12))


def laplace_kernel(x: np.ndarray, bandwidth: float | None = None) -> np.ndarray:
    """Laplace kernel based on Euclidean distance."""
    sigma = median_bandwidth(x) if bandwidth is None else float(bandwidth)
    d = np.sqrt(pairwise_sq_dists(x))
    return np.exp(-d / (sigma + 1e-12))


def linear_kernel(x: np.ndarray) -> np.ndarray:
    """Linear Gram matrix with dimension normalization."""
    x = np.asarray(x, dtype=np.float64)
    return x @ x.T / max(1, x.shape[1])


def random_fourier_features(
    x: np.ndarray,
    n_features: int = 512,
    bandwidth: float | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Random Fourier features approximating an RBF kernel."""
    x = np.asarray(x, dtype=np.float64)
    sigma = median_bandwidth(x) if bandwidth is None else float(bandwidth)
    rng = np.random.default_rng(seed)
    omega = rng.normal(scale=1.0 / (sigma + 1e-12), size=(x.shape[1], n_features))
    phase = rng.uniform(0.0, 2.0 * np.pi, size=n_features)
    return np.sqrt(2.0 / n_features) * np.cos(x @ omega + phase)


def feature_gram(features: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Feature Gram matrix."""
    features = np.asarray(features, dtype=np.float64)
    k = features @ features.T / max(1, features.shape[1])
    if normalize:
        k = diagonal_normalize(k)
    return k


def center_gram(k: np.ndarray) -> np.ndarray:
    """Double-center a Gram matrix."""
    k = np.asarray(k, dtype=np.float64)
    row_mean = k.mean(axis=1, keepdims=True)
    col_mean = k.mean(axis=0, keepdims=True)
    return k - row_mean - col_mean + k.mean()


def diagonal_normalize(k: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize a PSD Gram matrix to unit diagonal."""
    k = np.asarray(k, dtype=np.float64)
    diag = np.sqrt(np.maximum(np.diag(k), 0.0))
    denom = np.outer(diag, diag)
    return k / (denom + eps)


def kernel_metric_squared(k: np.ndarray) -> np.ndarray:
    """Squared RKHS distance matrix induced by a Gram matrix."""
    k = np.asarray(k, dtype=np.float64)
    diag = np.diag(k)
    d2 = diag[:, None] + diag[None, :] - 2.0 * k
    d2 = np.maximum(d2, 0.0)
    np.fill_diagonal(d2, 0.0)
    return d2
