"""Small dataset builders used by the first diagnostic experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Dataset:
    name: str
    x: np.ndarray
    y: np.ndarray


def standardize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Column-standardize a feature matrix."""
    x = np.asarray(x, dtype=np.float64)
    return (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + eps)


def make_two_moons(n: int = 400, noise: float = 0.08, seed: int = 0) -> Dataset:
    """Generate a two-moons binary dataset without relying on sklearn."""
    rng = np.random.default_rng(seed)
    n0 = n // 2
    n1 = n - n0

    theta0 = rng.uniform(0.0, np.pi, size=n0)
    theta1 = rng.uniform(0.0, np.pi, size=n1)

    moon0 = np.stack([np.cos(theta0), np.sin(theta0)], axis=1)
    moon1 = np.stack([1.0 - np.cos(theta1), 0.5 - np.sin(theta1)], axis=1)
    x = np.concatenate([moon0, moon1], axis=0)
    y = np.concatenate([np.ones(n0), -np.ones(n1)], axis=0)

    x = x + noise * rng.normal(size=x.shape)
    perm = rng.permutation(n)
    return Dataset(name=f"two_moons_noise{noise:g}", x=standardize(x[perm]), y=y[perm])


def make_xor(n: int = 400, noise: float = 0.2, seed: int = 0) -> Dataset:
    """Generate a noisy XOR dataset."""
    rng = np.random.default_rng(seed)
    centers = np.array(
        [
            [-1.0, -1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )
    labels = np.array([1.0, -1.0, -1.0, 1.0])
    idx = rng.integers(0, 4, size=n)
    x = centers[idx] + noise * rng.normal(size=(n, 2))
    y = labels[idx]
    return Dataset(name=f"xor_noise{noise:g}", x=standardize(x), y=y)


def make_collision_pairs(
    n_pairs: int = 200,
    separation: float = 0.05,
    seed: int = 0,
) -> Dataset:
    """Generate disjoint opposite-label near-collisions in 2D.

    Each latent anchor creates one positive and one negative sample separated by
    roughly `separation`. This is the cleanest finite-sample stress test for the
    local label mixing obstruction.
    """
    rng = np.random.default_rng(seed)
    anchors = rng.normal(size=(n_pairs, 2))
    directions = rng.normal(size=(n_pairs, 2))
    directions = directions / (np.linalg.norm(directions, axis=1, keepdims=True) + 1e-12)
    offset = 0.5 * separation * directions

    x_pos = anchors + offset
    x_neg = anchors - offset
    x = np.concatenate([x_pos, x_neg], axis=0)
    y = np.concatenate([np.ones(n_pairs), -np.ones(n_pairs)], axis=0)

    perm = rng.permutation(2 * n_pairs)
    return Dataset(name=f"collision_pairs_sep{separation:g}", x=standardize(x[perm]), y=y[perm])


def toy_suite(n: int = 400, seed: int = 0) -> list[Dataset]:
    """Default toy suite for experiment 001 quick runs."""
    return [
        make_two_moons(n=n, noise=0.05, seed=seed),
        make_two_moons(n=n, noise=0.2, seed=seed + 1),
        make_xor(n=n, noise=0.25, seed=seed + 2),
        make_collision_pairs(n_pairs=n // 2, separation=0.03, seed=seed + 3),
        make_collision_pairs(n_pairs=n // 2, separation=0.25, seed=seed + 4),
    ]
