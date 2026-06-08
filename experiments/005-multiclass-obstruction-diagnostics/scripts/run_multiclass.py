#!/usr/bin/env python3
"""Multiclass fixed-metric obstruction diagnostics for experiment 005."""

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


EXP_DIR = ROOT / "experiments" / "005-multiclass-obstruction-diagnostics"


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


def random_mlp_features(
    train_x: np.ndarray,
    test_x: np.ndarray,
    num_classes: int,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    models.set_seed(seed)
    model = models.ClassifierMLP(input_dim=train_x.shape[1], num_classes=num_classes, width=256, depth=2).to(device)
    model.eval()
    with torch.no_grad():
        train_z = model.features(torch.tensor(train_x, dtype=torch.float32, device=device)).detach().cpu().numpy()
        test_z = model.features(torch.tensor(test_x, dtype=torch.float32, device=device)).detach().cpu().numpy()
    return train_z, test_z


def train_mlp_features(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    num_classes: int,
    epochs: int,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    models.set_seed(seed)
    model = models.ClassifierMLP(input_dim=train_x.shape[1], num_classes=num_classes, width=256, depth=2).to(device)
    x_train = torch.tensor(train_x, dtype=torch.float32, device=device)
    y_train = torch.tensor(train_y, dtype=torch.long, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
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
    num_classes: int,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    models.set_seed(seed)
    model = models.ClassifierSmallCNN(
        num_classes=num_classes,
        width=32,
        feature_dim=128,
        in_channels=train_x.shape[1],
    ).to(device)
    model.eval()
    with torch.no_grad():
        train_z = model.features(torch.tensor(train_x, dtype=torch.float32, device=device)).detach().cpu().numpy()
        test_z = model.features(torch.tensor(test_x, dtype=torch.float32, device=device)).detach().cpu().numpy()
    return train_z, test_z


def train_cnn_features(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    num_classes: int,
    epochs: int,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    models.set_seed(seed)
    model = models.ClassifierSmallCNN(
        num_classes=num_classes,
        width=32,
        feature_dim=128,
        in_channels=train_x.shape[1],
    ).to(device)
    x_train = torch.tensor(train_x, dtype=torch.float32, device=device)
    y_train = torch.tensor(train_y, dtype=torch.long, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x_train), y_train)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        train_z = model.features(x_train).detach().cpu().numpy()
        test_z = model.features(torch.tensor(test_x, dtype=torch.float32, device=device)).detach().cpu().numpy()
    return train_z, test_z


def linear_probe_accuracy(
    train_z: np.ndarray,
    train_y: np.ndarray,
    test_z: np.ndarray,
    test_y: np.ndarray,
    num_classes: int,
    seed: int,
    device: torch.device,
    epochs: int = 250,
) -> tuple[float, float]:
    train_z, test_z = standardize_features(train_z, test_z)
    models.set_seed(seed)
    x_train = torch.tensor(train_z, dtype=torch.float32, device=device)
    y_train = torch.tensor(train_y, dtype=torch.long, device=device)
    x_test = torch.tensor(test_z, dtype=torch.float32, device=device)
    y_test = torch.tensor(test_y, dtype=torch.long, device=device)
    head = nn.Linear(train_z.shape[1], num_classes).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-2)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(head(x_train), y_train)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        train_acc = models.multiclass_accuracy(head(x_train), y_train)
        test_acc = models.multiclass_accuracy(head(x_test), y_test)
    return train_acc, test_acc


def diagnostics(z: np.ndarray, y: np.ndarray, k_neighbors: int) -> dict:
    gram = kernels.feature_gram(z, normalize=True)
    d2 = kernels.kernel_metric_squared(gram)
    spec = spectral.summarize_multiclass(gram, y)
    mix = mixing.summarize_multiclass(d2, y, k=k_neighbors)
    return {
        "tail_at_10pct": spec.tail_at_10pct,
        "tail_auc": spec.tail_auc,
        "alignment": spec.alignment,
        "knn_disagreement_ratio": mix.knn_disagreement_ratio,
        "knn_same_label_ratio": mix.knn_same_label_ratio,
        "local_label_entropy": mix.local_label_entropy,
        "local_label_entropy_normalized": mix.local_label_entropy_normalized,
        "nearest_other_mean": mix.nearest_other_mean,
        "collision_rate": mix.collision_rate,
    }


def summarize_rep(
    task: dict,
    rep_name: str,
    train_z: np.ndarray,
    test_z: np.ndarray,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    train_y = task["train_flat"].y
    test_y = task["test_flat"].y
    num_classes = len(task["class_names"])
    train_acc, test_acc = linear_probe_accuracy(
        train_z,
        train_y,
        test_z,
        test_y,
        num_classes=num_classes,
        seed=seed,
        device=device,
        epochs=args.probe_epochs,
    )
    return {
        "dataset": task["name"],
        "class_names": list(task["class_names"]),
        "representation": rep_name,
        "feature_dim": int(train_z.shape[1]),
        "linear_probe_train_acc": train_acc,
        "linear_probe_test_acc": test_acc,
        "train": diagnostics(train_z, train_y, k_neighbors=args.k_neighbors),
        "test": diagnostics(test_z, test_y, k_neighbors=args.k_neighbors),
    }


def build_tasks(root: Path, n_per_class: int, seed: int) -> list[dict]:
    tasks = []
    mnist_specs = [
        ("mnist_all10", tuple(range(10))),
        ("mnist_hard5", (3, 4, 5, 8, 9)),
    ]
    for offset, (name, classes) in enumerate(mnist_specs):
        train, test, class_names = datasets.make_mnist_multiclass_train_test(
            root,
            classes=classes,
            n_per_class=n_per_class,
            seed=seed + offset,
        )
        tasks.append(
            {
                "name": name,
                "train_flat": train,
                "test_flat": test,
                "train_img": None,
                "test_img": None,
                "class_names": class_names,
            }
        )

    cifar_specs: list[tuple[str, tuple[str, ...]]] = [
        ("cifar10_all10", tuple(datasets.CIFAR10_LABELS)),
        ("cifar10_animals6", ("bird", "cat", "deer", "dog", "frog", "horse")),
        ("cifar10_vehicles4", ("airplane", "automobile", "ship", "truck")),
    ]
    for offset, (name, classes) in enumerate(cifar_specs):
        train_flat, test_flat, class_names = datasets.make_cifar10_multiclass_train_test(
            root,
            classes=classes,
            n_per_class=n_per_class,
            seed=seed + 10 + offset,
        )
        train_img, test_img, _ = datasets.make_cifar10_multiclass_train_test(
            root,
            classes=classes,
            n_per_class=n_per_class,
            seed=seed + 10 + offset,
            as_images=True,
        )
        tasks.append(
            {
                "name": name,
                "train_flat": train_flat,
                "test_flat": test_flat,
                "train_img": train_img,
                "test_img": test_img,
                "class_names": class_names,
            }
        )
    return tasks


def run_task(task: dict, args: argparse.Namespace, task_idx: int, device: torch.device) -> list[dict]:
    train_flat = task["train_flat"]
    test_flat = task["test_flat"]
    x_train = flatten(train_flat.x)
    x_test = flatten(test_flat.x)
    y_train = train_flat.y
    num_classes = len(task["class_names"])

    reps: list[tuple[str, np.ndarray, np.ndarray]] = [
        ("raw_pixels", x_train, x_test),
        ("rff_512", *rff_pair(x_train, x_test, n_features=512, seed=args.seed + task_idx)),
        (
            "random_mlp",
            *random_mlp_features(
                x_train,
                x_test,
                num_classes=num_classes,
                seed=args.seed + 100 + task_idx,
                device=device,
            ),
        ),
        (
            "trained_mlp",
            *train_mlp_features(
                x_train,
                y_train,
                x_test,
                num_classes=num_classes,
                epochs=args.mlp_epochs,
                seed=args.seed + 200 + task_idx,
                device=device,
            ),
        ),
    ]

    if task["train_img"] is not None:
        train_img = task["train_img"]
        test_img = task["test_img"]
        reps.extend(
            [
                (
                    "random_cnn",
                    *random_cnn_features(
                        train_img.x,
                        test_img.x,
                        num_classes=num_classes,
                        seed=args.seed + 300 + task_idx,
                        device=device,
                    ),
                ),
                (
                    "trained_cnn",
                    *train_cnn_features(
                        train_img.x,
                        train_img.y,
                        test_img.x,
                        num_classes=num_classes,
                        epochs=args.cnn_epochs,
                        seed=args.seed + 400 + task_idx,
                        device=device,
                    ),
                ),
            ]
        )

    results = []
    for rep_idx, (rep_name, train_z, test_z) in enumerate(reps):
        results.append(
            summarize_rep(
                task=task,
                rep_name=rep_name,
                train_z=train_z,
                test_z=test_z,
                seed=args.seed + 1000 * task_idx + rep_idx,
                args=args,
                device=device,
            )
        )
    return results


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def plot_results(results: list[dict]) -> None:
    plotting.setup()
    plt.figure(figsize=(6.8, 4.8))
    for rep_name in sorted({r["representation"] for r in results}):
        rs = [r for r in results if r["representation"] == rep_name]
        plt.scatter(
            [r["test"]["knn_disagreement_ratio"] for r in rs],
            [r["test"]["tail_at_10pct"] for r in rs],
            label=rep_name,
            s=38,
            alpha=0.85,
        )
    plt.xlabel("test kNN disagreement ratio")
    plt.ylabel("test multiclass tail@10%")
    plt.title("Multiclass local mixing vs spectral tail")
    plt.legend(fontsize=7)
    plotting.savefig(EXP_DIR / "figures" / "mixing_vs_tail.png")

    plotting.setup()
    plt.figure(figsize=(6.8, 4.8))
    for rep_name in sorted({r["representation"] for r in results}):
        rs = [r for r in results if r["representation"] == rep_name]
        plt.scatter(
            [r["test"]["tail_at_10pct"] for r in rs],
            [r["linear_probe_test_acc"] for r in rs],
            label=rep_name,
            s=38,
            alpha=0.85,
        )
    plt.xlabel("test multiclass tail@10%")
    plt.ylabel("linear probe test accuracy")
    plt.title("Multiclass spectral tail vs accuracy")
    plt.legend(fontsize=7)
    plotting.savefig(EXP_DIR / "figures" / "tail_vs_accuracy.png")

    plotting.setup()
    plt.figure(figsize=(6.8, 4.8))
    for rep_name in sorted({r["representation"] for r in results}):
        rs = [r for r in results if r["representation"] == rep_name]
        plt.scatter(
            [r["test"]["local_label_entropy_normalized"] for r in rs],
            [r["test"]["tail_at_10pct"] for r in rs],
            label=rep_name,
            s=38,
            alpha=0.85,
        )
    plt.xlabel("test normalized local label entropy")
    plt.ylabel("test multiclass tail@10%")
    plt.title("Multiclass local entropy vs spectral tail")
    plt.legend(fontsize=7)
    plotting.savefig(EXP_DIR / "figures" / "entropy_vs_tail.png")


def write_result_md(results: list[dict], command: str, out: Path) -> None:
    mix = np.asarray([r["test"]["knn_disagreement_ratio"] for r in results])
    entropy = np.asarray([r["test"]["local_label_entropy_normalized"] for r in results])
    tail = np.asarray([r["test"]["tail_at_10pct"] for r in results])
    acc = np.asarray([r["linear_probe_test_acc"] for r in results])
    corr_mix_tail = safe_corr(mix, tail)
    corr_entropy_tail = safe_corr(entropy, tail)
    corr_tail_acc = safe_corr(tail, acc)

    lines = [
        "# Result",
        "",
        "## Run",
        "",
        f"Command: `{command}`",
        "",
        "## Summary",
        "",
        f"- corr(test kNN disagreement, test multiclass tail@10%) = `{corr_mix_tail:.3f}`",
        f"- corr(test normalized local entropy, test multiclass tail@10%) = `{corr_entropy_tail:.3f}`",
        f"- corr(test multiclass tail@10%, linear probe test acc) = `{corr_tail_acc:.3f}`",
        "",
        "Interpretation:",
        "",
        "- The binary obstruction signal extends if disagreement/entropy remain",
        "  positively correlated with multiclass label-subspace tail.",
        "- A representation should be treated as transferable only when the test",
        "  tail/entropy improve together with probe accuracy.",
        "- A train-only improvement is a memorization signal, not evidence that the",
        "  fixed test metric improved.",
        "",
        "| dataset | representation | classes | dim | test tail@10% | test disagreement | test entropy | alignment | probe train acc | probe test acc |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in sorted(results, key=lambda item: (item["dataset"], item["representation"])):
        lines.append(
            "| {dataset} | {rep} | {classes} | {dim} | {tail:.3f} | {mix:.3f} | {ent:.3f} | {align:.3f} | {tr:.3f} | {te:.3f} |".format(
                dataset=r["dataset"],
                rep=r["representation"],
                classes=len(r["class_names"]),
                dim=r["feature_dim"],
                tail=r["test"]["tail_at_10pct"],
                mix=r["test"]["knn_disagreement_ratio"],
                ent=r["test"]["local_label_entropy_normalized"],
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
            "- `figures/mixing_vs_tail.png`",
            "- `figures/tail_vs_accuracy.png`",
            "- `figures/entropy_vs_tail.png`",
        ]
    )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--n-per-class", type=int, default=40)
    parser.add_argument("--mlp-epochs", type=int, default=80)
    parser.add_argument("--cnn-epochs", type=int, default=60)
    parser.add_argument("--probe-epochs", type=int, default=250)
    parser.add_argument("--k-neighbors", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    tasks = build_tasks(args.data_root, n_per_class=args.n_per_class, seed=args.seed)
    results = []
    for task_idx, task in enumerate(tasks):
        print(f"Running {task['name']} ({len(task['class_names'])} classes)")
        results.extend(run_task(task, args=args, task_idx=task_idx, device=device))

    payload = {
        "config": {
            "n_per_class": args.n_per_class,
            "mlp_epochs": args.mlp_epochs,
            "cnn_epochs": args.cnn_epochs,
            "probe_epochs": args.probe_epochs,
            "k_neighbors": args.k_neighbors,
            "seed": args.seed,
            "device": args.device,
        },
        "results": results,
    }
    metrics_path = EXP_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot_results(results)
    command = "python3 experiments/005-multiclass-obstruction-diagnostics/scripts/run_multiclass.py"
    command += f" --n-per-class {args.n_per_class} --mlp-epochs {args.mlp_epochs}"
    command += f" --cnn-epochs {args.cnn_epochs} --probe-epochs {args.probe_epochs}"
    command += f" --k-neighbors {args.k_neighbors} --seed {args.seed} --device {args.device}"
    write_result_md(results, command, EXP_DIR / "result.md")
    print(f"Wrote {metrics_path}")
    print(f"Wrote {EXP_DIR / 'result.md'}")


if __name__ == "__main__":
    main()
