"""Small dataset builders used by the first diagnostic experiments."""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import pickle
import tarfile
import urllib.request
from pathlib import Path

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


MNIST_URLS = {
    "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_LABELS = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(path)


def _read_mnist_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        if magic != 2051:
            raise ValueError(f"bad MNIST image magic in {path}: {magic}")
        n = int.from_bytes(f.read(4), "big")
        rows = int.from_bytes(f.read(4), "big")
        cols = int.from_bytes(f.read(4), "big")
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows * cols)
    return data.astype(np.float64) / 255.0


def _read_mnist_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        if magic != 2049:
            raise ValueError(f"bad MNIST label magic in {path}: {magic}")
        n = int.from_bytes(f.read(4), "big")
        data = np.frombuffer(f.read(), dtype=np.uint8, count=n)
    return data.astype(np.int64)


def load_mnist(root: str | Path, split: str = "train") -> tuple[np.ndarray, np.ndarray]:
    """Load MNIST, downloading the official IDX gzip files if needed."""
    root = Path(root)
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    image_key = f"{split}_images"
    label_key = f"{split}_labels"
    image_path = root / "mnist" / Path(MNIST_URLS[image_key]).name
    label_path = root / "mnist" / Path(MNIST_URLS[label_key]).name
    _download(MNIST_URLS[image_key], image_path)
    _download(MNIST_URLS[label_key], label_path)
    return _read_mnist_images(image_path), _read_mnist_labels(label_path)


def make_mnist_binary(
    root: str | Path,
    classes: tuple[int, int] = (3, 8),
    n_per_class: int = 150,
    split: str = "train",
    seed: int = 0,
) -> Dataset:
    """MNIST binary subset with labels +1 for classes[0] and -1 for classes[1]."""
    x, labels = load_mnist(root, split=split)
    rng = np.random.default_rng(seed)
    selected_x = []
    selected_y = []
    for sign, cls in [(1.0, classes[0]), (-1.0, classes[1])]:
        idx = np.flatnonzero(labels == cls)
        if len(idx) < n_per_class:
            raise ValueError(f"not enough samples for MNIST class {cls}: {len(idx)}")
        idx = rng.choice(idx, size=n_per_class, replace=False)
        selected_x.append(x[idx])
        selected_y.append(np.full(n_per_class, sign))
    out_x = np.concatenate(selected_x, axis=0)
    out_y = np.concatenate(selected_y, axis=0)
    perm = rng.permutation(len(out_y))
    name = f"mnist_{classes[0]}vs{classes[1]}_{split}_n{n_per_class}"
    return Dataset(name=name, x=standardize(out_x[perm]), y=out_y[perm])


def _ensure_cifar10(root: Path) -> Path:
    root = Path(root)
    archive = root / "cifar-10-python.tar.gz"
    extracted = root / "cifar-10-batches-py"
    if extracted.exists():
        return extracted
    _download(CIFAR10_URL, archive)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(root)
    return extracted


def _load_cifar_batch(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as f:
        batch = pickle.load(f, encoding="latin1")
    x = batch["data"].astype(np.float64) / 255.0
    y = np.asarray(batch["labels"], dtype=np.int64)
    return x, y


def load_cifar10(root: str | Path, split: str = "train") -> tuple[np.ndarray, np.ndarray]:
    """Load CIFAR-10 python batches, downloading if needed."""
    base = _ensure_cifar10(Path(root) / "cifar10")
    if split == "train":
        xs, ys = [], []
        for i in range(1, 6):
            x, y = _load_cifar_batch(base / f"data_batch_{i}")
            xs.append(x)
            ys.append(y)
        return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)
    if split == "test":
        return _load_cifar_batch(base / "test_batch")
    raise ValueError("split must be 'train' or 'test'")


def _cifar_class_id(cls: str | int) -> int:
    if isinstance(cls, int):
        return cls
    return CIFAR10_LABELS.index(cls)


def make_cifar10_binary(
    root: str | Path,
    classes: tuple[str | int, str | int] = ("cat", "dog"),
    n_per_class: int = 150,
    split: str = "train",
    seed: int = 0,
) -> Dataset:
    """CIFAR-10 binary subset with labels +1 for classes[0] and -1 for classes[1]."""
    x, labels = load_cifar10(root, split=split)
    class_ids = (_cifar_class_id(classes[0]), _cifar_class_id(classes[1]))
    rng = np.random.default_rng(seed)
    selected_x = []
    selected_y = []
    for sign, cls in [(1.0, class_ids[0]), (-1.0, class_ids[1])]:
        idx = np.flatnonzero(labels == cls)
        if len(idx) < n_per_class:
            raise ValueError(f"not enough samples for CIFAR-10 class {cls}: {len(idx)}")
        idx = rng.choice(idx, size=n_per_class, replace=False)
        selected_x.append(x[idx])
        selected_y.append(np.full(n_per_class, sign))
    out_x = np.concatenate(selected_x, axis=0)
    out_y = np.concatenate(selected_y, axis=0)
    perm = rng.permutation(len(out_y))
    left = classes[0] if isinstance(classes[0], str) else CIFAR10_LABELS[class_ids[0]]
    right = classes[1] if isinstance(classes[1], str) else CIFAR10_LABELS[class_ids[1]]
    name = f"cifar10_{left}vs{right}_{split}_n{n_per_class}"
    return Dataset(name=name, x=standardize(out_x[perm]), y=out_y[perm])


def _binary_subset_from_arrays(
    x: np.ndarray,
    labels: np.ndarray,
    class_ids: tuple[int, int],
    class_names: tuple[str, str],
    n_per_class: int,
    seed: int,
    name: str,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
    image_shape: tuple[int, ...] | None = None,
) -> tuple[Dataset, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    xs = []
    ys = []
    for sign, cls in [(1.0, class_ids[0]), (-1.0, class_ids[1])]:
        idx = np.flatnonzero(labels == cls)
        if len(idx) < n_per_class:
            raise ValueError(f"not enough samples for class {cls}: {len(idx)}")
        idx = rng.choice(idx, size=n_per_class, replace=False)
        xs.append(x[idx])
        ys.append(np.full(n_per_class, sign))
    out_x = np.concatenate(xs, axis=0)
    out_y = np.concatenate(ys, axis=0)
    perm = rng.permutation(len(out_y))
    out_x = out_x[perm]
    out_y = out_y[perm]

    if mean is None:
        mean = out_x.mean(axis=0, keepdims=True)
    if std is None:
        std = out_x.std(axis=0, keepdims=True) + 1e-12
    out_x = (out_x - mean) / std
    if image_shape is not None:
        out_x = out_x.reshape((out_x.shape[0],) + image_shape)
    return Dataset(name=name, x=out_x, y=out_y), mean, std


def make_mnist_binary_train_test(
    root: str | Path,
    classes: tuple[int, int] = (3, 8),
    n_per_class: int = 150,
    seed: int = 0,
) -> tuple[Dataset, Dataset]:
    """MNIST train/test binary subsets using train-subset normalization."""
    x_train, y_train = load_mnist(root, split="train")
    x_test, y_test = load_mnist(root, split="test")
    label = f"{classes[0]}vs{classes[1]}_n{n_per_class}"
    train, mean, std = _binary_subset_from_arrays(
        x_train,
        y_train,
        class_ids=classes,
        class_names=(str(classes[0]), str(classes[1])),
        n_per_class=n_per_class,
        seed=seed,
        name=f"mnist_{label}_train",
    )
    test, _, _ = _binary_subset_from_arrays(
        x_test,
        y_test,
        class_ids=classes,
        class_names=(str(classes[0]), str(classes[1])),
        n_per_class=n_per_class,
        seed=seed + 100,
        name=f"mnist_{label}_test",
        mean=mean,
        std=std,
    )
    return train, test


def make_cifar10_binary_train_test(
    root: str | Path,
    classes: tuple[str | int, str | int] = ("cat", "dog"),
    n_per_class: int = 150,
    seed: int = 0,
    as_images: bool = False,
) -> tuple[Dataset, Dataset]:
    """CIFAR-10 train/test binary subsets using train-subset normalization."""
    x_train, y_train = load_cifar10(root, split="train")
    x_test, y_test = load_cifar10(root, split="test")
    class_ids = (_cifar_class_id(classes[0]), _cifar_class_id(classes[1]))
    class_names = (
        classes[0] if isinstance(classes[0], str) else CIFAR10_LABELS[class_ids[0]],
        classes[1] if isinstance(classes[1], str) else CIFAR10_LABELS[class_ids[1]],
    )
    image_shape = (3, 32, 32) if as_images else None
    label = f"{class_names[0]}vs{class_names[1]}_n{n_per_class}"
    train, mean, std = _binary_subset_from_arrays(
        x_train,
        y_train,
        class_ids=class_ids,
        class_names=class_names,
        n_per_class=n_per_class,
        seed=seed,
        name=f"cifar10_{label}_train",
        image_shape=image_shape,
    )
    test, _, _ = _binary_subset_from_arrays(
        x_test,
        y_test,
        class_ids=class_ids,
        class_names=class_names,
        n_per_class=n_per_class,
        seed=seed + 100,
        name=f"cifar10_{label}_test",
        mean=mean,
        std=std,
        image_shape=image_shape,
    )
    return train, test
