#!/usr/bin/env python3
"""Fixed-representation sweep for experiment 003."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from src import datasets, kernels, mixing, models, plotting, spectral


EXP_DIR = ROOT / "experiments" / "003-fixed-representation-sweep"


def flatten(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], -1)


def standardize_features(train_z: np.ndarray, test_z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_z.mean(axis=0, keepdims=True)
    std = train_z.std(axis=0, keepdims=True) + 1e-12
    return (train_z - mean) / std, (test_z - mean) / std


def rff_pair(train_x: np.ndarray, test_x: np.ndarray, n_features: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    sigma = kernels.median_bandwidth(train_x)
    rng = np.random.default_rng(seed)
    omega = rng.normal(scale=1.0 / (sigma + 1e-12), size=(train_x.shape[1], n_features))
    phase = rng.uniform(0.0, 2.0 * np.pi, size=n_features)

    def transform(x: np.ndarray) -> np.ndarray:
        return np.sqrt(2.0 / n_features) * np.cos(x @ omega + phase)

    return transform(train_x), transform(test_x)


def train_mlp_features(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    epochs: int,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    models.set_seed(seed)
    model = models.MLP(input_dim=train_x.shape[1], width=256, depth=2).to(device)
    x_train = torch.tensor(train_x, dtype=torch.float32, device=device)
    y_train = torch.tensor((train_y > 0).astype(np.float32), device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x_train), y_train)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        train_z = model.features(x_train).detach().cpu().numpy()
        test_z = model.features(torch.tensor(test_x, dtype=torch.float32, device=device)).detach().cpu().numpy()
    return train_z, test_z


def random_mlp_features(
    train_x: np.ndarray,
    test_x: np.ndarray,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    models.set_seed(seed)
    model = models.MLP(input_dim=train_x.shape[1], width=256, depth=2).to(device)
    model.eval()
    with torch.no_grad():
        train_z = model.features(torch.tensor(train_x, dtype=torch.float32, device=device)).detach().cpu().numpy()
        test_z = model.features(torch.tensor(test_x, dtype=torch.float32, device=device)).detach().cpu().numpy()
    return train_z, test_z


def train_cnn_features(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    epochs: int,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    models.set_seed(seed)
    model = models.SmallCNN(width=32, feature_dim=128).to(device)
    x_train = torch.tensor(train_x, dtype=torch.float32, device=device)
    y_train = torch.tensor((train_y > 0).astype(np.float32), device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x_train), y_train)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        train_z = model.features(x_train).detach().cpu().numpy()
        test_z = model.features(torch.tensor(test_x, dtype=torch.float32, device=device)).detach().cpu().numpy()
    return train_z, test_z


def random_cnn_features(
    train_x: np.ndarray,
    test_x: np.ndarray,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    models.set_seed(seed)
    model = models.SmallCNN(width=32, feature_dim=128).to(device)
    model.eval()
    with torch.no_grad():
        train_z = model.features(torch.tensor(train_x, dtype=torch.float32, device=device)).detach().cpu().numpy()
        test_z = model.features(torch.tensor(test_x, dtype=torch.float32, device=device)).detach().cpu().numpy()
    return train_z, test_z


def linear_probe_accuracy(
    train_z: np.ndarray,
    train_y: np.ndarray,
    test_z: np.ndarray,
    test_y: np.ndarray,
    seed: int,
    device: torch.device,
    epochs: int = 250,
) -> tuple[float, float]:
    train_z, test_z = standardize_features(train_z, test_z)
    models.set_seed(seed)
    x_train = torch.tensor(train_z, dtype=torch.float32, device=device)
    y_train = torch.tensor((train_y > 0).astype(np.float32), device=device)
    x_test = torch.tensor(test_z, dtype=torch.float32, device=device)
    y_test = torch.tensor((test_y > 0).astype(np.float32), device=device)
    head = nn.Linear(train_z.shape[1], 1).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-2)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(head(x_train).squeeze(-1), y_train)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        train_acc = models.binary_accuracy(head(x_train).squeeze(-1), y_train)
        test_acc = models.binary_accuracy(head(x_test).squeeze(-1), y_test)
    return train_acc, test_acc


def diagnostics(z: np.ndarray, y: np.ndarray) -> dict:
    gram = kernels.feature_gram(z, normalize=True)
    d2 = kernels.kernel_metric_squared(gram)
    spec = spectral.summarize(gram, y)
    mix = mixing.summarize(d2, y, k=10)
    return {
        "tail_at_10pct": spec.tail_at_10pct,
        "tail_auc": spec.tail_auc,
        "alignment": spec.alignment,
        "knn_opposite_ratio": mix.knn_opposite_ratio,
        "local_label_entropy": mix.local_label_entropy,
    }


def summarize_rep(
    dataset_name: str,
    rep_name: str,
    train_z: np.ndarray,
    test_z: np.ndarray,
    train_y: np.ndarray,
    test_y: np.ndarray,
    seed: int,
    device: torch.device,
) -> dict:
    train_acc, test_acc = linear_probe_accuracy(train_z, train_y, test_z, test_y, seed=seed, device=device)
    train_diag = diagnostics(train_z, train_y)
    test_diag = diagnostics(test_z, test_y)
    return {
        "dataset": dataset_name,
        "representation": rep_name,
        "feature_dim": int(train_z.shape[1]),
        "linear_probe_train_acc": train_acc,
        "linear_probe_test_acc": test_acc,
        "train": train_diag,
        "test": test_diag,
    }


def build_tasks(root: Path, n_per_class: int, seed: int) -> list[dict]:
    tasks = []
    for idx, classes in enumerate([(3, 8), (4, 9)]):
        train, test = datasets.make_mnist_binary_train_test(root, classes=classes, n_per_class=n_per_class, seed=seed + idx)
        tasks.append({"name": train.name, "train_flat": train, "test_flat": test, "train_img": None, "test_img": None})
    for idx, classes in enumerate([("cat", "dog"), ("automobile", "truck")]):
        train_flat, test_flat = datasets.make_cifar10_binary_train_test(root, classes=classes, n_per_class=n_per_class, seed=seed + 10 + idx)
        train_img, test_img = datasets.make_cifar10_binary_train_test(root, classes=classes, n_per_class=n_per_class, seed=seed + 10 + idx, as_images=True)
        tasks.append({"name": train_flat.name, "train_flat": train_flat, "test_flat": test_flat, "train_img": train_img, "test_img": test_img})
    return tasks


def run_task(task: dict, args: argparse.Namespace, task_idx: int, device: torch.device) -> list[dict]:
    train_flat = task["train_flat"]
    test_flat = task["test_flat"]
    x_train = flatten(train_flat.x)
    x_test = flatten(test_flat.x)
    y_train = train_flat.y
    y_test = test_flat.y
    results = []

    reps: list[tuple[str, np.ndarray, np.ndarray]] = []
    reps.append(("raw_pixels", x_train, x_test))
    reps.append(("rff_512", *rff_pair(x_train, x_test, n_features=512, seed=args.seed + task_idx)))
    reps.append(("random_mlp", *random_mlp_features(x_train, x_test, seed=args.seed + 100 + task_idx, device=device)))
    reps.append(("trained_mlp", *train_mlp_features(x_train, y_train, x_test, epochs=args.mlp_epochs, seed=args.seed + 200 + task_idx, device=device)))

    if task["train_img"] is not None:
        train_img = task["train_img"]
        test_img = task["test_img"]
        reps.append(("random_cnn", *random_cnn_features(train_img.x, test_img.x, seed=args.seed + 300 + task_idx, device=device)))
        reps.append(("trained_cnn", *train_cnn_features(train_img.x, train_img.y, test_img.x, epochs=args.cnn_epochs, seed=args.seed + 400 + task_idx, device=device)))

    for rep_idx, (rep_name, train_z, test_z) in enumerate(reps):
        results.append(
            summarize_rep(
                dataset_name=task["name"],
                rep_name=rep_name,
                train_z=train_z,
                test_z=test_z,
                train_y=y_train,
                test_y=y_test,
                seed=args.seed + 1000 * task_idx + rep_idx,
                device=device,
            )
        )
    return results


def plot_results(results: list[dict]) -> None:
    plotting.setup()
    plt.figure(figsize=(6.6, 4.6))
    for rep_name in sorted({r["representation"] for r in results}):
        rs = [r for r in results if r["representation"] == rep_name]
        plt.scatter(
            [r["test"]["tail_at_10pct"] for r in rs],
            [r["linear_probe_test_acc"] for r in rs],
            label=rep_name,
            s=34,
            alpha=0.85,
        )
    plt.xlabel("test tail@10%")
    plt.ylabel("linear probe test accuracy")
    plt.title("Representation sweep: tail vs accuracy")
    plt.legend(fontsize=7)
    plotting.savefig(EXP_DIR / "figures" / "tail_vs_accuracy.png")

    plotting.setup()
    plt.figure(figsize=(6.6, 4.6))
    for rep_name in sorted({r["representation"] for r in results}):
        rs = [r for r in results if r["representation"] == rep_name]
        plt.scatter(
            [r["test"]["knn_opposite_ratio"] for r in rs],
            [r["test"]["tail_at_10pct"] for r in rs],
            label=rep_name,
            s=34,
            alpha=0.85,
        )
    plt.xlabel("test opposite-kNN ratio")
    plt.ylabel("test tail@10%")
    plt.title("Representation sweep: local mixing vs tail")
    plt.legend(fontsize=7)
    plotting.savefig(EXP_DIR / "figures" / "mixing_vs_tail.png")


def write_result_md(results: list[dict], command: str, out: Path) -> None:
    x = np.asarray([r["test"]["knn_opposite_ratio"] for r in results])
    tail = np.asarray([r["test"]["tail_at_10pct"] for r in results])
    acc = np.asarray([r["linear_probe_test_acc"] for r in results])
    corr_mix_tail = float(np.corrcoef(x, tail)[0, 1])
    corr_tail_acc = float(np.corrcoef(tail, acc)[0, 1])
    lines = [
        "# Result",
        "",
        "## Run",
        "",
        f"Command: `{command}`",
        "",
        "## Summary",
        "",
        f"- corr(test opposite-kNN ratio, test tail@10%) = `{corr_mix_tail:.3f}`",
        f"- corr(test tail@10%, linear probe test acc) = `{corr_tail_acc:.3f}`",
        "",
        "Interpretation checklist:",
        "",
        "- A useful fixed representation should have low test local mixing, low test",
        "  spectral tail, and high linear-probe accuracy.",
        "- If a trained representation only improves train metrics, it should be",
        "  treated as memorization rather than transferable metric adaptation.",
        "- CIFAR `cat vs dog` is expected to remain hard for small local runs.",
        "",
        "| dataset | representation | dim | test tail@10% | test mix | alignment | probe train acc | probe test acc |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in sorted(results, key=lambda item: (item["dataset"], item["representation"])):
        lines.append(
            "| {dataset} | {rep} | {dim} | {tail:.3f} | {mix:.3f} | {align:.3f} | {tr:.3f} | {te:.3f} |".format(
                dataset=r["dataset"],
                rep=r["representation"],
                dim=r["feature_dim"],
                tail=r["test"]["tail_at_10pct"],
                mix=r["test"]["knn_opposite_ratio"],
                align=r["test"]["alignment"],
                tr=r["linear_probe_train_acc"],
                te=r["linear_probe_test_acc"],
            )
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `metrics.json`",
            "- `figures/tail_vs_accuracy.png`",
            "- `figures/mixing_vs_tail.png`",
        ]
    )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--n-per-class", type=int, default=100)
    parser.add_argument("--mlp-epochs", type=int, default=80)
    parser.add_argument("--cnn-epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    tasks = build_tasks(args.data_root, args.n_per_class, args.seed)
    results = []
    for task_idx, task in enumerate(tasks):
        results.extend(run_task(task, args, task_idx, device=device))

    metrics_path = EXP_DIR / "metrics.json"
    metrics_path.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
    plot_results(results)
    command = "python experiments/003-fixed-representation-sweep/scripts/run_sweep.py"
    command += f" --n-per-class {args.n_per_class} --mlp-epochs {args.mlp_epochs} --cnn-epochs {args.cnn_epochs} --seed {args.seed} --device {args.device}"
    write_result_md(results, command, EXP_DIR / "result.md")
    print(f"Wrote {metrics_path}")
    print(f"Wrote {EXP_DIR / 'result.md'}")


if __name__ == "__main__":
    main()
