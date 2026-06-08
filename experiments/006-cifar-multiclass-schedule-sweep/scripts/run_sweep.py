#!/usr/bin/env python3
"""CIFAR multiclass schedule sweep for experiment 006."""

from __future__ import annotations

import argparse
from collections import defaultdict
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
import torch.nn.functional as F
from torch import nn

from src import datasets, kernels, mixing, models, plotting, spectral


EXP_DIR = ROOT / "experiments" / "006-cifar-multiclass-schedule-sweep"


def group_norm(channels: int) -> nn.GroupNorm:
    groups = 8 if channels % 8 == 0 else 1
    return nn.GroupNorm(groups, channels)


class CifarMetricCNN(nn.Module):
    """Small CIFAR CNN used to isolate schedule effects."""

    def __init__(self, num_classes: int, width: int = 48, feature_dim: int = 192) -> None:
        super().__init__()
        self.feature_net = nn.Sequential(
            nn.Conv2d(3, width, kernel_size=3, padding=1, bias=False),
            group_norm(width),
            nn.ReLU(),
            nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False),
            group_norm(width),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(width, 2 * width, kernel_size=3, padding=1, bias=False),
            group_norm(2 * width),
            nn.ReLU(),
            nn.Conv2d(2 * width, 2 * width, kernel_size=3, padding=1, bias=False),
            group_norm(2 * width),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(2 * width, feature_dim, kernel_size=3, padding=1, bias=False),
            group_norm(feature_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.head = nn.Linear(feature_dim, num_classes)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_net(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def margin_stats(logits: torch.Tensor, targets: torch.Tensor) -> tuple[float, float]:
    true_logits = logits.gather(1, targets[:, None]).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, targets[:, None], -torch.inf)
    other_logits = masked.max(dim=1).values
    margins = true_logits - other_logits
    return float(margins.mean().item()), float(margins.median().item())


def extract_features_logits(
    model: nn.Module,
    x_np: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    feats = []
    logits = []
    with torch.no_grad():
        for start in range(0, len(x_np), batch_size):
            x = torch.tensor(x_np[start : start + batch_size], dtype=torch.float32, device=device)
            feats.append(model.features(x).detach().cpu().numpy())
            logits.append(model(x).detach().cpu().numpy())
    return np.concatenate(feats, axis=0), np.concatenate(logits, axis=0)


def representation_diagnostics(z: np.ndarray, y: np.ndarray, k_neighbors: int) -> dict:
    gram = kernels.feature_gram(z, normalize=True)
    d2 = kernels.kernel_metric_squared(gram)
    spec = spectral.summarize_multiclass(gram, y)
    mix = mixing.summarize_multiclass(d2, y, k=k_neighbors)
    return {
        "tail_at_10pct": spec.tail_at_10pct,
        "tail_auc": spec.tail_auc,
        "alignment": spec.alignment,
        "knn_disagreement_ratio": mix.knn_disagreement_ratio,
        "local_label_entropy_normalized": mix.local_label_entropy_normalized,
        "collision_rate": mix.collision_rate,
    }


def standardize_features(train_z: np.ndarray, test_z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_z.mean(axis=0, keepdims=True)
    std = train_z.std(axis=0, keepdims=True) + 1e-12
    return (train_z - mean) / std, (test_z - mean) / std


def linear_probe_metrics(
    train_z: np.ndarray,
    train_y: np.ndarray,
    test_z: np.ndarray,
    test_y: np.ndarray,
    num_classes: int,
    seed: int,
    device: torch.device,
    epochs: int,
) -> dict:
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
        train_logits = head(x_train)
        test_logits = head(x_test)
        train_margin_mean, train_margin_median = margin_stats(train_logits, y_train)
        test_margin_mean, test_margin_median = margin_stats(test_logits, y_test)
        return {
            "probe_train_acc": models.multiclass_accuracy(train_logits, y_train),
            "probe_test_acc": models.multiclass_accuracy(test_logits, y_test),
            "probe_train_margin_mean": train_margin_mean,
            "probe_train_margin_median": train_margin_median,
            "probe_test_margin_mean": test_margin_mean,
            "probe_test_margin_median": test_margin_median,
        }


def random_crop_flip(x: torch.Tensor, padding: int = 4) -> torch.Tensor:
    if padding <= 0:
        out = x.clone()
    else:
        padded = F.pad(x, (padding, padding, padding, padding), mode="reflect")
        out = torch.empty_like(x)
        max_offset = 2 * padding
        ys = torch.randint(0, max_offset + 1, (x.shape[0],), device=x.device)
        xs = torch.randint(0, max_offset + 1, (x.shape[0],), device=x.device)
        for i in range(x.shape[0]):
            out[i] = padded[i, :, ys[i] : ys[i] + x.shape[2], xs[i] : xs[i] + x.shape[3]]
    flips = torch.rand(x.shape[0], device=x.device) < 0.5
    out[flips] = torch.flip(out[flips], dims=[3])
    return out


def evaluate_model(
    model: nn.Module,
    train: datasets.Dataset,
    test: datasets.Dataset,
    k0_train: np.ndarray | None,
    k0_test: np.ndarray | None,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    train_z, train_logits_np = extract_features_logits(model, train.x, device=device, batch_size=args.eval_batch_size)
    test_z, test_logits_np = extract_features_logits(model, test.x, device=device, batch_size=args.eval_batch_size)
    train_diag = representation_diagnostics(train_z, train.y, k_neighbors=args.k_neighbors)
    test_diag = representation_diagnostics(test_z, test.y, k_neighbors=args.k_neighbors)
    train_gram = kernels.feature_gram(train_z, normalize=True)
    test_gram = kernels.feature_gram(test_z, normalize=True)
    movement_train = None
    movement_test = None
    if k0_train is not None and k0_test is not None:
        movement_train = float(np.linalg.norm(train_gram - k0_train, "fro") / (np.linalg.norm(k0_train, "fro") + 1e-12))
        movement_test = float(np.linalg.norm(test_gram - k0_test, "fro") / (np.linalg.norm(k0_test, "fro") + 1e-12))

    y_train_t = torch.tensor(train.y, dtype=torch.long)
    y_test_t = torch.tensor(test.y, dtype=torch.long)
    train_logits = torch.tensor(train_logits_np, dtype=torch.float32)
    test_logits = torch.tensor(test_logits_np, dtype=torch.float32)
    head_train_margin_mean, head_train_margin_median = margin_stats(train_logits, y_train_t)
    head_test_margin_mean, head_test_margin_median = margin_stats(test_logits, y_test_t)
    probe = linear_probe_metrics(
        train_z,
        train.y,
        test_z,
        test.y,
        num_classes=int(len(np.unique(train.y))),
        seed=seed,
        device=device,
        epochs=args.probe_epochs,
    )
    return {
        "feature_dim": int(train_z.shape[1]),
        "train": train_diag,
        "test": test_diag,
        "kernel_movement_train": movement_train,
        "kernel_movement_test": movement_test,
        "head_train_acc": models.multiclass_accuracy(train_logits, y_train_t),
        "head_test_acc": models.multiclass_accuracy(test_logits, y_test_t),
        "head_train_margin_mean": head_train_margin_mean,
        "head_train_margin_median": head_train_margin_median,
        "head_test_margin_mean": head_test_margin_mean,
        "head_test_margin_median": head_test_margin_median,
        **probe,
    }


def initial_grams(
    model: nn.Module,
    train: datasets.Dataset,
    test: datasets.Dataset,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    train_z, _ = extract_features_logits(model, train.x, device=device, batch_size=args.eval_batch_size)
    test_z, _ = extract_features_logits(model, test.x, device=device, batch_size=args.eval_batch_size)
    return kernels.feature_gram(train_z, normalize=True), kernels.feature_gram(test_z, normalize=True)


def train_fullbatch(
    train: datasets.Dataset,
    test: datasets.Dataset,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    models.set_seed(seed)
    num_classes = int(len(np.unique(train.y)))
    model = CifarMetricCNN(num_classes=num_classes, width=args.width, feature_dim=args.feature_dim).to(device)
    k0_train, k0_test = initial_grams(model, train, test, args=args, device=device)
    x_train = torch.tensor(train.x, dtype=torch.float32, device=device)
    y_train = torch.tensor(train.y, dtype=torch.long, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.short_lr)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(args.short_epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x_train), y_train)
        loss.backward()
        optimizer.step()
    out = evaluate_model(model, train, test, k0_train, k0_test, seed=seed + 10_000, args=args, device=device)
    out.update({"regime": "short_fullbatch", "seed": seed, "epochs": args.short_epochs})
    return out


def train_strong_minibatch(
    train: datasets.Dataset,
    test: datasets.Dataset,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    models.set_seed(seed)
    num_classes = int(len(np.unique(train.y)))
    model = CifarMetricCNN(num_classes=num_classes, width=args.width, feature_dim=args.feature_dim).to(device)
    k0_train, k0_test = initial_grams(model, train, test, args=args, device=device)
    x_train = torch.tensor(train.x, dtype=torch.float32, device=device)
    y_train = torch.tensor(train.y, dtype=torch.long, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    n = len(train.y)
    for _ in range(args.epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for start in range(0, n, args.batch_size):
            idx = perm[start : start + args.batch_size]
            xb = random_crop_flip(x_train[idx], padding=args.crop_padding)
            yb = y_train[idx]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()
    out = evaluate_model(model, train, test, k0_train, k0_test, seed=seed + 20_000, args=args, device=device)
    out.update({"regime": "strong_minibatch", "seed": seed, "epochs": args.epochs})
    return out


def run_random_cnn(
    train: datasets.Dataset,
    test: datasets.Dataset,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    models.set_seed(seed)
    num_classes = int(len(np.unique(train.y)))
    model = CifarMetricCNN(num_classes=num_classes, width=args.width, feature_dim=args.feature_dim).to(device)
    out = evaluate_model(model, train, test, None, None, seed=seed + 30_000, args=args, device=device)
    out.update({"regime": "random_cnn", "seed": seed, "epochs": 0})
    return out


def build_tasks(root: Path, n_per_class: int, seed: int) -> list[dict]:
    specs: list[tuple[str, tuple[str, ...]]] = [
        ("cifar10_all10", tuple(datasets.CIFAR10_LABELS)),
        ("cifar10_animals6", ("bird", "cat", "deer", "dog", "frog", "horse")),
        ("cifar10_vehicles4", ("airplane", "automobile", "ship", "truck")),
    ]
    tasks = []
    for offset, (name, classes) in enumerate(specs):
        train, test, class_names = datasets.make_cifar10_multiclass_train_test(
            root,
            classes=classes,
            n_per_class=n_per_class,
            seed=seed + offset,
            as_images=True,
        )
        tasks.append({"name": name, "train": train, "test": test, "class_names": class_names})
    return tasks


def run_task(task: dict, args: argparse.Namespace, device: torch.device) -> list[dict]:
    rows = []
    for seed in args.seeds:
        print(f"Running {task['name']} seed={seed} random_cnn", flush=True)
        random_row = run_random_cnn(task["train"], task["test"], seed=seed, args=args, device=device)
        rows.append({"dataset": task["name"], "class_names": list(task["class_names"]), **random_row})

        print(f"Running {task['name']} seed={seed} short_fullbatch", flush=True)
        short_row = train_fullbatch(task["train"], task["test"], seed=seed, args=args, device=device)
        rows.append({"dataset": task["name"], "class_names": list(task["class_names"]), **short_row})

        print(f"Running {task['name']} seed={seed} strong_minibatch", flush=True)
        strong_row = train_strong_minibatch(task["train"], task["test"], seed=seed, args=args, device=device)
        rows.append({"dataset": task["name"], "class_names": list(task["class_names"]), **strong_row})
    return rows


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def aggregate_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["regime"])].append(row)
    fields = [
        ("test_tail_at_10pct", lambda r: r["test"]["tail_at_10pct"]),
        ("test_disagreement", lambda r: r["test"]["knn_disagreement_ratio"]),
        ("test_entropy", lambda r: r["test"]["local_label_entropy_normalized"]),
        ("probe_test_acc", lambda r: r["probe_test_acc"]),
        ("probe_test_margin_median", lambda r: r["probe_test_margin_median"]),
        ("kernel_movement_test", lambda r: r["kernel_movement_test"] if r["kernel_movement_test"] is not None else 0.0),
    ]
    out = []
    for (dataset, regime), rs in sorted(grouped.items()):
        item = {"dataset": dataset, "regime": regime, "n": len(rs)}
        for name, getter in fields:
            values = np.asarray([getter(r) for r in rs], dtype=np.float64)
            item[f"{name}_mean"] = float(np.mean(values))
            item[f"{name}_std"] = float(np.std(values))
        out.append(item)
    return out


def plot_by_regime(rows: list[dict], metric_getter, ylabel: str, out: Path) -> None:
    plotting.setup()
    regimes = ["random_cnn", "short_fullbatch", "strong_minibatch"]
    x_pos = {regime: i for i, regime in enumerate(regimes)}
    offsets = {"cifar10_all10": -0.18, "cifar10_animals6": 0.0, "cifar10_vehicles4": 0.18}
    plt.figure(figsize=(7.0, 4.7))
    for row in rows:
        x = x_pos[row["regime"]] + offsets.get(row["dataset"], 0.0)
        plt.scatter(x, metric_getter(row), s=34, alpha=0.75, label=row["dataset"])
    handles, labels = plt.gca().get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    plt.xticks(range(len(regimes)), regimes, rotation=15)
    plt.ylabel(ylabel)
    plt.legend(uniq.values(), uniq.keys(), fontsize=7)
    plotting.savefig(out)


def plot_scatter(rows: list[dict], x_getter, y_getter, xlabel: str, ylabel: str, out: Path) -> None:
    plotting.setup()
    plt.figure(figsize=(6.7, 4.7))
    for regime in sorted({row["regime"] for row in rows}):
        rs = [row for row in rows if row["regime"] == regime]
        plt.scatter([x_getter(r) for r in rs], [y_getter(r) for r in rs], s=42, alpha=0.82, label=regime)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(fontsize=7)
    plotting.savefig(out)


def plot_results(rows: list[dict]) -> None:
    plot_by_regime(
        rows,
        metric_getter=lambda r: r["test"]["tail_at_10pct"],
        ylabel="test multiclass tail@10%",
        out=EXP_DIR / "figures" / "tail_by_regime.png",
    )
    plot_by_regime(
        rows,
        metric_getter=lambda r: r["probe_test_acc"],
        ylabel="linear probe test accuracy",
        out=EXP_DIR / "figures" / "accuracy_by_regime.png",
    )
    plot_scatter(
        rows,
        x_getter=lambda r: r["test"]["tail_at_10pct"],
        y_getter=lambda r: r["probe_test_acc"],
        xlabel="test multiclass tail@10%",
        ylabel="linear probe test accuracy",
        out=EXP_DIR / "figures" / "tail_vs_accuracy.png",
    )
    plot_scatter(
        rows,
        x_getter=lambda r: r["probe_test_margin_median"],
        y_getter=lambda r: r["probe_test_acc"],
        xlabel="probe test margin median",
        ylabel="linear probe test accuracy",
        out=EXP_DIR / "figures" / "margin_vs_accuracy.png",
    )


def write_result_md(rows: list[dict], aggregates: list[dict], command: str, out: Path) -> None:
    tail = np.asarray([row["test"]["tail_at_10pct"] for row in rows])
    disagreement = np.asarray([row["test"]["knn_disagreement_ratio"] for row in rows])
    entropy = np.asarray([row["test"]["local_label_entropy_normalized"] for row in rows])
    acc = np.asarray([row["probe_test_acc"] for row in rows])
    margin = np.asarray([row["probe_test_margin_median"] for row in rows])
    corr_dis_tail = safe_corr(disagreement, tail)
    corr_ent_tail = safe_corr(entropy, tail)
    corr_tail_acc = safe_corr(tail, acc)
    corr_margin_acc = safe_corr(margin, acc)

    lines = [
        "# Result",
        "",
        "## Run",
        "",
        f"Command: `{command}`",
        "",
        "## Summary",
        "",
        f"- corr(test kNN disagreement, test multiclass tail@10%) = `{corr_dis_tail:.3f}`",
        f"- corr(test normalized local entropy, test multiclass tail@10%) = `{corr_ent_tail:.3f}`",
        f"- corr(test multiclass tail@10%, probe test acc) = `{corr_tail_acc:.3f}`",
        f"- corr(probe test margin median, probe test acc) = `{corr_margin_acc:.3f}`",
        "",
        "Aggregate rows report mean/std over seeds.",
        "",
        "| dataset | regime | seeds | test tail mean | test tail std | disagreement mean | entropy mean | movement mean | probe acc mean | probe acc std | margin median mean |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregates:
        lines.append(
            "| {dataset} | {regime} | {n} | {tail:.3f} | {tail_std:.3f} | {dis:.3f} | {ent:.3f} | {move:.3f} | {acc:.3f} | {acc_std:.3f} | {margin:.3f} |".format(
                dataset=row["dataset"],
                regime=row["regime"],
                n=row["n"],
                tail=row["test_tail_at_10pct_mean"],
                tail_std=row["test_tail_at_10pct_std"],
                dis=row["test_disagreement_mean"],
                ent=row["test_entropy_mean"],
                move=row["kernel_movement_test_mean"],
                acc=row["probe_test_acc_mean"],
                acc_std=row["probe_test_acc_std"],
                margin=row["probe_test_margin_median_mean"],
            )
        )

    lines.extend(
        [
            "",
            "## Single-Seed Rows",
            "",
            "| dataset | regime | seed | test tail | test disagreement | test entropy | movement | head test acc | probe test acc | probe margin median |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(rows, key=lambda item: (item["dataset"], item["regime"], item["seed"])):
        movement = row["kernel_movement_test"]
        lines.append(
            "| {dataset} | {regime} | {seed} | {tail:.3f} | {dis:.3f} | {ent:.3f} | {move:.3f} | {head:.3f} | {probe:.3f} | {margin:.3f} |".format(
                dataset=row["dataset"],
                regime=row["regime"],
                seed=row["seed"],
                tail=row["test"]["tail_at_10pct"],
                dis=row["test"]["knn_disagreement_ratio"],
                ent=row["test"]["local_label_entropy_normalized"],
                move=movement if movement is not None else 0.0,
                head=row["head_test_acc"],
                probe=row["probe_test_acc"],
                margin=row["probe_test_margin_median"],
            )
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `metrics.json`",
            "- `figures/tail_by_regime.png`",
            "- `figures/accuracy_by_regime.png`",
            "- `figures/tail_vs_accuracy.png`",
            "- `figures/margin_vs_accuracy.png`",
        ]
    )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--n-per-class", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--short-epochs", type=int, default=60)
    parser.add_argument("--probe-epochs", type=int, default=250)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--feature-dim", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--short-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--crop-padding", type=int, default=4)
    parser.add_argument("--k-neighbors", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    tasks = build_tasks(args.data_root, n_per_class=args.n_per_class, seed=args.seed)
    rows = []
    for task in tasks:
        rows.extend(run_task(task, args=args, device=device))

    aggregates = aggregate_rows(rows)
    payload = {
        "config": {
            "n_per_class": args.n_per_class,
            "epochs": args.epochs,
            "short_epochs": args.short_epochs,
            "probe_epochs": args.probe_epochs,
            "seeds": args.seeds,
            "width": args.width,
            "feature_dim": args.feature_dim,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "short_lr": args.short_lr,
            "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing,
            "crop_padding": args.crop_padding,
            "k_neighbors": args.k_neighbors,
            "seed": args.seed,
            "device": args.device,
        },
        "aggregates": aggregates,
        "results": rows,
    }
    metrics_path = EXP_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot_results(rows)
    command = "python3 experiments/006-cifar-multiclass-schedule-sweep/scripts/run_sweep.py"
    command += f" --n-per-class {args.n_per_class} --epochs {args.epochs} --short-epochs {args.short_epochs}"
    command += f" --probe-epochs {args.probe_epochs} --seeds {' '.join(str(s) for s in args.seeds)}"
    command += f" --width {args.width} --feature-dim {args.feature_dim} --batch-size {args.batch_size}"
    command += f" --lr {args.lr} --short-lr {args.short_lr} --weight-decay {args.weight_decay}"
    command += f" --label-smoothing {args.label_smoothing} --crop-padding {args.crop_padding}"
    command += f" --k-neighbors {args.k_neighbors} --seed {args.seed} --device {args.device}"
    write_result_md(rows, aggregates, command, EXP_DIR / "result.md")
    print(f"Wrote {metrics_path}")
    print(f"Wrote {EXP_DIR / 'result.md'}")


if __name__ == "__main__":
    main()
