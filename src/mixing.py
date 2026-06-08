"""Local label-mixing diagnostics under a metric."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MixingSummary:
    opposite_nn_mean: float
    opposite_nn_median: float
    same_nn_mean: float
    knn_opposite_ratio: float
    local_label_entropy: float
    collision_count: int
    collision_rate: float
    rho: float
    graph_dirichlet: float
    graph_disagreement: float


@dataclass(frozen=True)
class MulticlassMixingSummary:
    nearest_other_mean: float
    nearest_other_median: float
    nearest_same_mean: float
    knn_disagreement_ratio: float
    knn_same_label_ratio: float
    local_label_entropy: float
    local_label_entropy_normalized: float
    collision_count: int
    collision_rate: float
    rho: float
    graph_dirichlet: float
    graph_disagreement: float


def _offdiag_inf(d2: np.ndarray) -> np.ndarray:
    out = np.asarray(d2, dtype=np.float64).copy()
    np.fill_diagonal(out, np.inf)
    return out


def nearest_distances_by_label(d2: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nearest same-label and opposite-label distances for every sample."""
    d = np.sqrt(np.maximum(np.asarray(d2, dtype=np.float64), 0.0))
    y = np.asarray(y).reshape(-1)
    n = len(y)
    same = np.full(n, np.inf)
    opp = np.full(n, np.inf)
    for i in range(n):
        same_mask = y == y[i]
        opp_mask = y != y[i]
        same_mask[i] = False
        if np.any(same_mask):
            same[i] = np.min(d[i, same_mask])
        if np.any(opp_mask):
            opp[i] = np.min(d[i, opp_mask])
    return same, opp


def knn_opposite_ratio(d2: np.ndarray, y: np.ndarray, k: int = 10) -> float:
    """Average fraction of k nearest neighbors with the opposite label."""
    dist = _offdiag_inf(d2)
    y = np.asarray(y).reshape(-1)
    k = max(1, min(k, len(y) - 1))
    idx = np.argpartition(dist, kth=k - 1, axis=1)[:, :k]
    return float(np.mean(y[idx] != y[:, None]))


def knn_adjacency(d2: np.ndarray, k: int = 10, symmetrize: bool = True, weighted: bool = True) -> np.ndarray:
    """kNN graph adjacency from squared distances."""
    d2 = np.asarray(d2, dtype=np.float64)
    n = d2.shape[0]
    if d2.shape != (n, n):
        raise ValueError("d2 must be square")
    k = max(1, min(k, n - 1))
    dist = _offdiag_inf(d2)
    idx = np.argpartition(dist, kth=k - 1, axis=1)[:, :k]
    adj = np.zeros((n, n), dtype=np.float64)
    if weighted:
        finite = dist[np.isfinite(dist)]
        sigma2 = float(np.median(finite[finite > 1e-12])) if np.any(finite > 1e-12) else 1.0
        weights = np.exp(-d2 / (sigma2 + 1e-12))
    else:
        weights = np.ones_like(d2, dtype=np.float64)
    rows = np.arange(n)[:, None]
    adj[rows, idx] = weights[rows, idx]
    if symmetrize:
        adj = np.maximum(adj, adj.T)
    np.fill_diagonal(adj, 0.0)
    return adj


def graph_disagreement_ratio(adj: np.ndarray, y: np.ndarray) -> float:
    """Fraction of graph edge weight crossing different labels."""
    adj = np.asarray(adj, dtype=np.float64)
    y = np.asarray(y).reshape(-1)
    denom = float(np.sum(adj))
    if denom <= 0:
        return 0.0
    cross = adj * (y[:, None] != y[None, :])
    return float(np.sum(cross) / denom)


def graph_dirichlet_energy(adj: np.ndarray, labels: np.ndarray) -> float:
    """Normalized graph Dirichlet energy for binary or multiclass labels.

    Binary labels use the provided scalar labels. Multiclass labels use centered
    one-hot vectors, so the energy measures label-subspace roughness on the
    local metric graph.
    """
    adj = np.asarray(adj, dtype=np.float64)
    labels = np.asarray(labels).reshape(-1)
    classes = np.unique(labels)
    if len(classes) <= 2 and np.all(np.isin(labels, [-1, 1, -1.0, 1.0])):
        values = labels.astype(np.float64)[:, None]
    else:
        values = np.zeros((len(labels), len(classes)), dtype=np.float64)
        for col, cls in enumerate(classes):
            values[:, col] = labels == cls
        values = values - values.mean(axis=0, keepdims=True)

    denom = float(np.sum(adj))
    if denom <= 0:
        return 0.0
    diff = values[:, None, :] - values[None, :, :]
    sq = np.sum(diff * diff, axis=2)
    raw = 0.5 * float(np.sum(adj * sq))
    label_scale = float(np.mean(np.sum(values * values, axis=1)))
    return raw / (denom * max(label_scale, 1e-12))


def local_label_entropy(d2: np.ndarray, y: np.ndarray, k: int = 10) -> float:
    """Average binary entropy of labels in each k-neighborhood."""
    dist = _offdiag_inf(d2)
    y = np.asarray(y).reshape(-1)
    k = max(1, min(k, len(y) - 1))
    idx = np.argpartition(dist, kth=k - 1, axis=1)[:, :k]
    p_opp = np.mean(y[idx] != y[:, None], axis=1)
    p = np.clip(p_opp, 1e-12, 1.0 - 1e-12)
    entropy = -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))
    return float(np.mean(entropy))


def multiclass_local_label_entropy(d2: np.ndarray, y: np.ndarray, k: int = 10, normalize: bool = False) -> float:
    """Average entropy of the empirical class distribution in each k-neighborhood."""
    dist = _offdiag_inf(d2)
    y = np.asarray(y).reshape(-1)
    classes = np.unique(y)
    k = max(1, min(k, len(y) - 1))
    idx = np.argpartition(dist, kth=k - 1, axis=1)[:, :k]
    entropies = []
    for neigh in idx:
        _, counts = np.unique(y[neigh], return_counts=True)
        p = counts.astype(np.float64) / float(k)
        p = np.clip(p, 1e-12, 1.0)
        entropies.append(float(-np.sum(p * np.log2(p))))
    entropy = float(np.mean(entropies))
    if normalize and len(classes) > 1:
        entropy = entropy / float(np.log2(len(classes)))
    return entropy


def greedy_disjoint_opposite_pairs(
    d2: np.ndarray,
    y: np.ndarray,
    rho: float,
) -> list[tuple[int, int, float]]:
    """Greedily select disjoint opposite-label pairs with distance <= rho."""
    d = np.sqrt(np.maximum(np.asarray(d2, dtype=np.float64), 0.0))
    y = np.asarray(y).reshape(-1)
    pairs: list[tuple[int, int, float]] = []
    n = len(y)
    for i in range(n):
        for j in range(i + 1, n):
            if y[i] != y[j] and d[i, j] <= rho:
                pairs.append((i, j, float(d[i, j])))
    pairs.sort(key=lambda item: item[2])

    used: set[int] = set()
    selected: list[tuple[int, int, float]] = []
    for i, j, dist in pairs:
        if i in used or j in used:
            continue
        selected.append((i, j, dist))
        used.add(i)
        used.add(j)
    return selected


def summarize(d2: np.ndarray, y: np.ndarray, k: int = 10, rho_quantile: float = 0.10) -> MixingSummary:
    """Compute standard local-mixing diagnostics."""
    same, opp = nearest_distances_by_label(d2, y)
    finite_opp = opp[np.isfinite(opp)]
    if finite_opp.size == 0:
        rho = 0.0
    else:
        rho = float(np.quantile(finite_opp, rho_quantile))
    pairs = greedy_disjoint_opposite_pairs(d2, y, rho=rho)
    n = len(np.asarray(y).reshape(-1))
    adj = knn_adjacency(d2, k=k)
    return MixingSummary(
        opposite_nn_mean=float(np.mean(finite_opp)) if finite_opp.size else float("inf"),
        opposite_nn_median=float(np.median(finite_opp)) if finite_opp.size else float("inf"),
        same_nn_mean=float(np.mean(same[np.isfinite(same)])),
        knn_opposite_ratio=knn_opposite_ratio(d2, y, k=k),
        local_label_entropy=local_label_entropy(d2, y, k=k),
        collision_count=len(pairs),
        collision_rate=float(len(pairs) / max(1, n)),
        rho=rho,
        graph_dirichlet=graph_dirichlet_energy(adj, y),
        graph_disagreement=graph_disagreement_ratio(adj, y),
    )


def summarize_multiclass(d2: np.ndarray, y: np.ndarray, k: int = 10, rho_quantile: float = 0.10) -> MulticlassMixingSummary:
    """Compute local-mixing diagnostics for multiclass labels."""
    same, other = nearest_distances_by_label(d2, y)
    finite_other = other[np.isfinite(other)]
    if finite_other.size == 0:
        rho = 0.0
    else:
        rho = float(np.quantile(finite_other, rho_quantile))
    pairs = greedy_disjoint_opposite_pairs(d2, y, rho=rho)
    n = len(np.asarray(y).reshape(-1))
    disagreement = knn_opposite_ratio(d2, y, k=k)
    adj = knn_adjacency(d2, k=k)
    return MulticlassMixingSummary(
        nearest_other_mean=float(np.mean(finite_other)) if finite_other.size else float("inf"),
        nearest_other_median=float(np.median(finite_other)) if finite_other.size else float("inf"),
        nearest_same_mean=float(np.mean(same[np.isfinite(same)])),
        knn_disagreement_ratio=disagreement,
        knn_same_label_ratio=float(1.0 - disagreement),
        local_label_entropy=multiclass_local_label_entropy(d2, y, k=k, normalize=False),
        local_label_entropy_normalized=multiclass_local_label_entropy(d2, y, k=k, normalize=True),
        collision_count=len(pairs),
        collision_rate=float(len(pairs) / max(1, n)),
        rho=rho,
        graph_dirichlet=graph_dirichlet_energy(adj, y),
        graph_disagreement=graph_disagreement_ratio(adj, y),
    )


def disjoint_collision_tail_bound(
    n: int,
    q_rho: int,
    rho: float,
    lambda_m: float,
) -> float:
    """Normalized corollary-style lower-bound proxy for tail energy.

    This uses the simple disjoint-pairs expression from the draft:

    eT_m(y) >= q_rho / 2 * (2 - rho * sqrt(n / lambda_m))_+^2

    and divides by ||y||^2 = n.
    """
    if n <= 0 or q_rho <= 0 or lambda_m <= 0:
        return 0.0
    slack = max(0.0, 2.0 - float(rho) * np.sqrt(float(n) / float(lambda_m)))
    return float((q_rho / 2.0) * slack * slack / n)


def to_jsonable(summary: MixingSummary) -> dict[str, float | int]:
    return {
        "opposite_nn_mean": summary.opposite_nn_mean,
        "opposite_nn_median": summary.opposite_nn_median,
        "same_nn_mean": summary.same_nn_mean,
        "knn_opposite_ratio": summary.knn_opposite_ratio,
        "local_label_entropy": summary.local_label_entropy,
        "collision_count": summary.collision_count,
        "collision_rate": summary.collision_rate,
        "rho": summary.rho,
        "graph_dirichlet": summary.graph_dirichlet,
        "graph_disagreement": summary.graph_disagreement,
    }


def multiclass_to_jsonable(summary: MulticlassMixingSummary) -> dict[str, float | int]:
    return {
        "nearest_other_mean": summary.nearest_other_mean,
        "nearest_other_median": summary.nearest_other_median,
        "nearest_same_mean": summary.nearest_same_mean,
        "knn_disagreement_ratio": summary.knn_disagreement_ratio,
        "knn_same_label_ratio": summary.knn_same_label_ratio,
        "local_label_entropy": summary.local_label_entropy,
        "local_label_entropy_normalized": summary.local_label_entropy_normalized,
        "collision_count": summary.collision_count,
        "collision_rate": summary.collision_rate,
        "rho": summary.rho,
        "graph_dirichlet": summary.graph_dirichlet,
        "graph_disagreement": summary.graph_disagreement,
    }
