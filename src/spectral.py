"""Spectral diagnostics for labels under a fixed kernel."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .kernels import center_gram


@dataclass(frozen=True)
class SpectralSummary:
    eigvals: np.ndarray
    energy: np.ndarray
    tail: np.ndarray
    tail_auc: float
    tail_at_10pct: float
    alignment: float


@dataclass(frozen=True)
class MulticlassSpectralSummary:
    eigvals: np.ndarray
    energy: np.ndarray
    tail: np.ndarray
    tail_auc: float
    tail_at_10pct: float
    alignment: float
    n_classes: int


def eigendecompose(k: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric eigendecomposition sorted by descending eigenvalue."""
    k = np.asarray(k, dtype=np.float64)
    k = 0.5 * (k + k.T)
    eigvals, eigvecs = np.linalg.eigh(k)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 0.0)
    eigvecs = eigvecs[:, order]
    eigvals[eigvals < eps] = 0.0
    return eigvals, eigvecs


def label_energy_curve(y: np.ndarray, eigvecs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return cumulative spectral energy and tail for labels."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    coeff2 = (eigvecs.T @ y) ** 2
    total = float(np.dot(y, y))
    if total <= 0:
        raise ValueError("label vector has zero norm")
    energy = np.cumsum(coeff2) / total
    energy = np.clip(energy, 0.0, 1.0)
    tail = 1.0 - energy
    return energy, tail


def one_hot_labels(y: np.ndarray, classes: np.ndarray | None = None, center: bool = True) -> np.ndarray:
    """One-hot label matrix, optionally column-centered."""
    y = np.asarray(y).reshape(-1)
    if classes is None:
        classes = np.unique(y)
    classes = np.asarray(classes)
    out = np.zeros((len(y), len(classes)), dtype=np.float64)
    for col, cls in enumerate(classes):
        out[:, col] = y == cls
    if center:
        out = out - out.mean(axis=0, keepdims=True)
    return out


def label_subspace_energy_curve(labels: np.ndarray, eigvecs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative spectral energy/tail for a multiclass one-hot label subspace."""
    y_mat = one_hot_labels(labels, center=True)
    coeff2 = np.sum((eigvecs.T @ y_mat) ** 2, axis=1)
    total = float(np.sum(y_mat * y_mat))
    if total <= 0:
        raise ValueError("label matrix has zero norm")
    energy = np.cumsum(coeff2) / total
    energy = np.clip(energy, 0.0, 1.0)
    tail = 1.0 - energy
    return energy, tail


def kernel_target_alignment(k: np.ndarray, y: np.ndarray, center: bool = True) -> float:
    """Frobenius alignment between kernel and yy^T."""
    k = np.asarray(k, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    target = y @ y.T
    if center:
        k = center_gram(k)
        target = center_gram(target)
    denom = np.linalg.norm(k, "fro") * np.linalg.norm(target, "fro")
    if denom <= 0:
        return 0.0
    return float(np.sum(k * target) / denom)


def label_subspace_alignment(k: np.ndarray, labels: np.ndarray, center: bool = True) -> float:
    """Frobenius alignment between a kernel and the multiclass same-label target."""
    k = np.asarray(k, dtype=np.float64)
    y_mat = one_hot_labels(labels, center=True)
    target = y_mat @ y_mat.T
    if center:
        k = center_gram(k)
        target = center_gram(target)
    denom = np.linalg.norm(k, "fro") * np.linalg.norm(target, "fro")
    if denom <= 0:
        return 0.0
    return float(np.sum(k * target) / denom)


def tail_at_fraction(tail: np.ndarray, fraction: float) -> float:
    """Tail value after the top `fraction` eigendirections."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    idx = max(0, min(len(tail) - 1, int(np.ceil(fraction * len(tail))) - 1))
    return float(tail[idx])


def summarize(k: np.ndarray, y: np.ndarray) -> SpectralSummary:
    """Compute the standard spectral summary used by experiments."""
    eigvals, eigvecs = eigendecompose(k)
    energy, tail = label_energy_curve(y, eigvecs)
    return SpectralSummary(
        eigvals=eigvals,
        energy=energy,
        tail=tail,
        tail_auc=float(np.mean(tail)),
        tail_at_10pct=tail_at_fraction(tail, 0.10),
        alignment=kernel_target_alignment(k, y),
    )


def summarize_multiclass(k: np.ndarray, labels: np.ndarray) -> MulticlassSpectralSummary:
    """Compute spectral summary for a multiclass centered one-hot label subspace."""
    eigvals, eigvecs = eigendecompose(k)
    energy, tail = label_subspace_energy_curve(labels, eigvecs)
    return MulticlassSpectralSummary(
        eigvals=eigvals,
        energy=energy,
        tail=tail,
        tail_auc=float(np.mean(tail)),
        tail_at_10pct=tail_at_fraction(tail, 0.10),
        alignment=label_subspace_alignment(k, labels),
        n_classes=int(len(np.unique(labels))),
    )


def to_jsonable(summary: SpectralSummary) -> dict[str, float | list[float]]:
    """Compact JSON representation."""
    return {
        "eigvals": summary.eigvals.tolist(),
        "energy": summary.energy.tolist(),
        "tail": summary.tail.tolist(),
        "tail_auc": summary.tail_auc,
        "tail_at_10pct": summary.tail_at_10pct,
        "alignment": summary.alignment,
    }


def multiclass_to_jsonable(summary: MulticlassSpectralSummary) -> dict[str, float | int | list[float]]:
    """Compact JSON representation for multiclass spectral summaries."""
    return {
        "eigvals": summary.eigvals.tolist(),
        "energy": summary.energy.tolist(),
        "tail": summary.tail.tolist(),
        "tail_auc": summary.tail_auc,
        "tail_at_10pct": summary.tail_at_10pct,
        "alignment": summary.alignment,
        "n_classes": summary.n_classes,
    }
