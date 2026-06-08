"""Exact static-kernel gradient-flow diagnostics."""

from __future__ import annotations

import numpy as np


def residual_norm_curve(
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    y: np.ndarray,
    times: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """Compute ||exp(-tK/n)y||^2 for each time."""
    eigvals = np.asarray(eigvals, dtype=np.float64).reshape(-1)
    eigvecs = np.asarray(eigvecs, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    coeff2 = (eigvecs.T @ y) ** 2
    n = len(y)
    decay = np.exp(-2.0 * np.outer(times, eigvals) / max(1, n))
    norms = decay @ coeff2
    if normalize:
        norms = norms / max(float(np.dot(y, y)), 1e-12)
    return norms


def time_to_fraction(times: np.ndarray, curve: np.ndarray, fraction: float = 0.1) -> float | None:
    """First time where a normalized curve drops below `fraction`."""
    times = np.asarray(times, dtype=np.float64)
    curve = np.asarray(curve, dtype=np.float64)
    hit = np.flatnonzero(curve <= fraction)
    if hit.size == 0:
        return None
    return float(times[int(hit[0])])
