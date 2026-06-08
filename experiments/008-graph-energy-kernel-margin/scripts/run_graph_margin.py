#!/usr/bin/env python3
"""Graph-energy and kernel-margin diagnostics for experiment 008."""

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

from src import datasets, kernel_ridge, kernels, mixing, models, plotting, spectral


EXP_DIR = ROOT / "experiments" / "008-graph-energy-kernel-margin"


def flatten(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], -1)


def standardize_features(train_z: np.ndarray, test_z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_z.mean(axis=0, keepdims=True)
    std = train_z.std(axis=0, keepdims=True) + 1e-12
    return (train_z - mean) / std, (test_z - mean) / std


def row_normalize(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    return z / (norms + 1e-12)


def linear_gram_blocks(train_z: np.ndarray, test_z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_z, test_z = standardize_features(train_z, test_z)
    train_z = row_normalize(train_z)
    test_z = row_normalize(test_z)
    k_train = train_z @ train_z.T
    k_test = test_z @ test_z.T
    k_test_train = test_z @ train_z.T
    return k_train, k_test, k_test_train


def rbf_blocks(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sigma = kernels.median_bandwidth(train_x)
    train_d2 = kernels.pairwise_sq_dists(train_x)
    test_d2 = kernels.pairwise_sq_dists(test_x)
    cross_d2 = kernels.pairwise_sq_dists(test_x, train_x)
    denom = 2.0 * sigma * sigma + 1e-12
    return np.exp(-train_d2 / denom), np.exp(-test_d2 / denom), np.exp(-cross_d2 / denom)


def laplace_blocks(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sigma = kernels.median_bandwidth(train_x)
    train_d = np.sqrt(kernels.pairwise_sq_dists(train_x))
    test_d = np.sqrt(kernels.pairwise_sq_dists(test_x))
    cross_d = np.sqrt(kernels.pairwise_sq_dists(test_x, train_x))
    denom = sigma + 1e-12
    return np.exp(-train_d / denom), np.exp(-test_d / denom), np.exp(-cross_d / denom)


def rff_pair(train_x: np.ndarray, test_x: np.ndarray, n_features: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    sigma = kernels.median_bandwidth(train_x)
    rng = np.random.default_rng(seed)
    omega = rng.normal(scale=1.0 / (sigma + 1e-12), size=(train_x.shape[1], n_features))
    phase = rng.uniform(0.0, 2.0 * np.pi, size=n_features)

    def transform(x: np.ndarray) -> np.ndarray:
        return np.sqrt(2.0 / n_features) * np.cos(x @ omega + phase)

    return transform(train_x), transform(test_x)


def random_mlp_features(train_x: np.ndarray, test_x: np.ndarray, seed: int, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    models.set_seed(seed)
    model = models.MLP(input_dim=train_x.shape[1], width=256, depth=2).to(device)
    model.eval()
    with torch.no_grad():
        train_z = model.features(torch.tensor(train_x, dtype=torch.float32, device=device)).cpu().numpy()
        test_z = model.features(torch.tensor(test_x, dtype=torch.float32, device=device)).cpu().numpy()
    return train_z, test_z


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
        train_z = model.features(x_train).cpu().numpy()
        test_z = model.features(torch.tensor(test_x, dtype=torch.float32, device=device)).cpu().numpy()
    return train_z, test_z


def random_cnn_features(train_x: np.ndarray, test_x: np.ndarray, seed: int, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    models.set_seed(seed)
    model = models.SmallCNN(width=32, feature_dim=128).to(device)
    model.eval()
    with torch.no_grad():
        train_z = model.features(torch.tensor(train_x, dtype=torch.float32, device=device)).cpu().numpy()
        test_z = model.features(torch.tensor(test_x, dtype=torch.float32, device=device)).cpu().numpy()
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
        train_z = model.features(x_train).cpu().numpy()
        test_z = model.features(torch.tensor(test_x, dtype=torch.float32, device=device)).cpu().numpy()
    return train_z, test_z


def diagnostics_from_blocks(
    dataset: str,
    representation: str,
    k_train: np.ndarray,
    k_test: np.ndarray,
    k_test_train: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    ridge: float,
) -> dict:
    test_d2 = kernels.kernel_metric_squared(k_test)
    spec = spectral.summarize(k_test, y_test)
    mix = mixing.summarize(test_d2, y_test, k=10)
    clf = kernel_ridge.fit_binary_kernel_ridge(k_train, y_train, k_test_train, y_test, ridge=ridge)
    return {
        "dataset": dataset,
        "representation": representation,
        "test_tail_at_10pct": spec.tail_at_10pct,
        "test_tail_auc": spec.tail_auc,
        "test_alignment": spec.alignment,
        "test_knn_opposite_ratio": mix.knn_opposite_ratio,
        "test_local_label_entropy": mix.local_label_entropy,
        "test_graph_disagreement": mix.graph_disagreement,
        "test_graph_dirichlet": mix.graph_dirichlet,
        "test_collision_rate": mix.collision_rate,
        "kernel_ridge": kernel_ridge.to_jsonable(clf),
    }


def build_tasks(root: Path, n_per_class: int, seed: int) -> list[dict]:
    tasks = []
    for idx, classes in enumerate([(3, 8), (4, 9)]):
        train, test = datasets.make_mnist_binary_train_test(root, classes=classes, n_per_class=n_per_class, seed=seed + idx)
        tasks.append({"name": train.name, "train_flat": train, "test_flat": test, "train_img": None, "test_img": None})
    for idx, classes in enumerate([("cat", "dog"), ("automobile", "truck")]):
        train_flat, test_flat = datasets.make_cifar10_binary_train_test(root, classes=classes, n_per_class=n_per_class, seed=seed + 10 + idx)
        train_img, test_img = datasets.make_cifar10_binary_train_test(
            root,
            classes=classes,
            n_per_class=n_per_class,
            seed=seed + 10 + idx,
            as_images=True,
        )
        tasks.append({"name": train_flat.name, "train_flat": train_flat, "test_flat": test_flat, "train_img": train_img, "test_img": test_img})
    return tasks


def run_task(task: dict, task_idx: int, args: argparse.Namespace, device: torch.device) -> list[dict]:
    train = task["train_flat"]
    test = task["test_flat"]
    x_train = flatten(train.x)
    x_test = flatten(test.x)
    y_train = train.y
    y_test = test.y
    rows = []

    kernel_blocks = {
        "raw_linear": linear_gram_blocks(x_train, x_test),
        "rbf": rbf_blocks(x_train, x_test),
        "laplace": laplace_blocks(x_train, x_test),
    }
    for name, (k_train, k_test, k_test_train) in kernel_blocks.items():
        rows.append(diagnostics_from_blocks(task["name"], name, k_train, k_test, k_test_train, y_train, y_test, ridge=args.ridge))

    feature_reps: list[tuple[str, np.ndarray, np.ndarray]] = [
        ("rff_512", *rff_pair(x_train, x_test, n_features=512, seed=args.seed + task_idx)),
        ("random_mlp", *random_mlp_features(x_train, x_test, seed=args.seed + 100 + task_idx, device=device)),
        (
            "trained_mlp",
            *train_mlp_features(
                x_train,
                y_train,
                x_test,
                epochs=args.mlp_epochs,
                seed=args.seed + 200 + task_idx,
                device=device,
            ),
        ),
    ]
    if task["train_img"] is not None:
        train_img = task["train_img"]
        test_img = task["test_img"]
        feature_reps.extend(
            [
                ("random_cnn", *random_cnn_features(train_img.x, test_img.x, seed=args.seed + 300 + task_idx, device=device)),
                (
                    "trained_cnn",
                    *train_cnn_features(
                        train_img.x,
                        train_img.y,
                        test_img.x,
                        epochs=args.cnn_epochs,
                        seed=args.seed + 400 + task_idx,
                        device=device,
                    ),
                ),
            ]
        )

    for rep_name, train_z, test_z in feature_reps:
        k_train, k_test, k_test_train = linear_gram_blocks(train_z, test_z)
        rows.append(diagnostics_from_blocks(task["name"], rep_name, k_train, k_test, k_test_train, y_train, y_test, ridge=args.ridge))
    return rows


def safe_corr(xs: list[float], ys: list[float]) -> float:
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def plot_results(rows: list[dict]) -> None:
    plotting.setup()
    plt.figure(figsize=(6.8, 4.8))
    for rep in sorted({r["representation"] for r in rows}):
        rs = [r for r in rows if r["representation"] == rep]
        plt.scatter([r["test_graph_dirichlet"] for r in rs], [r["test_tail_at_10pct"] for r in rs], s=36, alpha=0.82, label=rep)
    plt.xlabel("test graph Dirichlet energy")
    plt.ylabel("test tail@10%")
    plt.legend(fontsize=7)
    plotting.savefig(EXP_DIR / "figures" / "graph_energy_vs_tail.png")

    plotting.setup()
    plt.figure(figsize=(6.8, 4.8))
    for rep in sorted({r["representation"] for r in rows}):
        rs = [r for r in rows if r["representation"] == rep]
        plt.scatter([r["test_tail_at_10pct"] for r in rs], [r["kernel_ridge"]["test_margin_median"] for r in rs], s=36, alpha=0.82, label=rep)
    plt.xlabel("test tail@10%")
    plt.ylabel("kernel ridge test margin median")
    plt.legend(fontsize=7)
    plotting.savefig(EXP_DIR / "figures" / "tail_vs_margin.png")

    plotting.setup()
    plt.figure(figsize=(6.8, 4.8))
    for rep in sorted({r["representation"] for r in rows}):
        rs = [r for r in rows if r["representation"] == rep]
        plt.scatter([r["kernel_ridge"]["source_norm"] for r in rs], [r["kernel_ridge"]["test_margin_median"] for r in rs], s=36, alpha=0.82, label=rep)
    plt.xlabel("source norm proxy")
    plt.ylabel("kernel ridge test margin median")
    plt.legend(fontsize=7)
    plotting.savefig(EXP_DIR / "figures" / "source_norm_vs_margin.png")


def write_result_md(rows: list[dict], command: str) -> None:
    tail = [r["test_tail_at_10pct"] for r in rows]
    mix = [r["test_knn_opposite_ratio"] for r in rows]
    graph_dis = [r["test_graph_disagreement"] for r in rows]
    graph_dir = [r["test_graph_dirichlet"] for r in rows]
    margin = [r["kernel_ridge"]["test_margin_median"] for r in rows]
    acc = [r["kernel_ridge"]["test_accuracy"] for r in rows]
    source_norm = [r["kernel_ridge"]["source_norm"] for r in rows]
    lines = [
        "# Result",
        "",
        "## Run",
        "",
        f"Command: `{command}`",
        "",
        "## Summary",
        "",
        f"- corr(test kNN opposite ratio, test tail@10%) = `{safe_corr(mix, tail):.3f}`",
        f"- corr(test graph disagreement, test tail@10%) = `{safe_corr(graph_dis, tail):.3f}`",
        f"- corr(test graph Dirichlet, test tail@10%) = `{safe_corr(graph_dir, tail):.3f}`",
        f"- corr(test tail@10%, kernel ridge test margin median) = `{safe_corr(tail, margin):.3f}`",
        f"- corr(kernel ridge test margin median, test accuracy) = `{safe_corr(margin, acc):.3f}`",
        f"- corr(source norm proxy, test margin median) = `{safe_corr(source_norm, margin):.3f}`",
        "",
        "Interpretation:",
        "",
        "- Graph disagreement/Dirichlet energy give a graph-level version of local",
        "  label mixing and should move with spectral tail.",
        "- Kernel ridge margin/source norm connect the geometric obstruction to a",
        "  classifier consequence rather than only a spectral statistic.",
        "",
        "| dataset | representation | tail@10% | kNN mix | graph dis | graph dir | align | ridge acc | ridge margin | source norm |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in sorted(rows, key=lambda item: (item["dataset"], item["representation"])):
        kr = r["kernel_ridge"]
        lines.append(
            "| {dataset} | {rep} | {tail:.3f} | {mix:.3f} | {gdis:.3f} | {gdir:.3f} | {align:.3f} | {acc:.3f} | {margin:.3f} | {norm:.2f} |".format(
                dataset=r["dataset"],
                rep=r["representation"],
                tail=r["test_tail_at_10pct"],
                mix=r["test_knn_opposite_ratio"],
                gdis=r["test_graph_disagreement"],
                gdir=r["test_graph_dirichlet"],
                align=r["test_alignment"],
                acc=kr["test_accuracy"],
                margin=kr["test_margin_median"],
                norm=kr["source_norm"],
            )
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `metrics.json`",
            "- `figures/graph_energy_vs_tail.png`",
            "- `figures/tail_vs_margin.png`",
            "- `figures/source_norm_vs_margin.png`",
        ]
    )
    (EXP_DIR / "result.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--n-per-class", type=int, default=80)
    parser.add_argument("--mlp-epochs", type=int, default=60)
    parser.add_argument("--cnn-epochs", type=int, default=50)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    tasks = build_tasks(args.data_root, n_per_class=args.n_per_class, seed=args.seed)
    rows = []
    for task_idx, task in enumerate(tasks):
        print(f"Running {task['name']}", flush=True)
        rows.extend(run_task(task, task_idx=task_idx, args=args, device=device))

    payload = {
        "config": {
            "n_per_class": args.n_per_class,
            "mlp_epochs": args.mlp_epochs,
            "cnn_epochs": args.cnn_epochs,
            "ridge": args.ridge,
            "seed": args.seed,
            "device": args.device,
        },
        "results": rows,
    }
    (EXP_DIR / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot_results(rows)
    command = "python3 experiments/008-graph-energy-kernel-margin/scripts/run_graph_margin.py"
    command += f" --n-per-class {args.n_per_class} --mlp-epochs {args.mlp_epochs}"
    command += f" --cnn-epochs {args.cnn_epochs} --ridge {args.ridge}"
    command += f" --seed {args.seed} --device {args.device}"
    write_result_md(rows, command)
    print(f"Wrote {EXP_DIR / 'metrics.json'}")
    print(f"Wrote {EXP_DIR / 'result.md'}")


if __name__ == "__main__":
    main()
