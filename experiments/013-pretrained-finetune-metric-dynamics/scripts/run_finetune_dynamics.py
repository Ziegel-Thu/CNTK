#!/usr/bin/env python3
"""Pretrained ResNet18 fine-tuning metric dynamics for experiment 013."""

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
import torch.nn.functional as F
from torch import nn
from torchvision import models as tv_models

from src import datasets, kernel_ridge, kernels, mixing, models, plotting, spectral


EXP_DIR = ROOT / "experiments" / "013-pretrained-finetune-metric-dynamics"
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def class_id(cls: str | int) -> int:
    if isinstance(cls, int):
        return cls
    return datasets.CIFAR10_LABELS.index(cls)


def select_cifar_subset(
    root: Path,
    classes: tuple[str | int, ...],
    n_per_class: int,
    split: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    x, labels = datasets.load_cifar10(root, split=split)
    ids = tuple(class_id(cls) for cls in classes)
    names = tuple(datasets.CIFAR10_LABELS[idx] for idx in ids)
    rng = np.random.default_rng(seed)
    xs = []
    ys = []
    for out_label, cls_id in enumerate(ids):
        idx = np.flatnonzero(labels == cls_id)
        if len(idx) < n_per_class:
            raise ValueError(f"not enough CIFAR samples for {names[out_label]}: {len(idx)}")
        idx = rng.choice(idx, size=n_per_class, replace=False)
        xs.append(x[idx])
        ys.append(np.full(n_per_class, out_label, dtype=np.int64))
    out_x = np.concatenate(xs, axis=0).reshape(-1, 3, 32, 32)
    out_y = np.concatenate(ys, axis=0)
    perm = rng.permutation(len(out_y))
    return out_x[perm], out_y[perm], names


def build_tasks(root: Path, binary_n_per_class: int, multi_n_per_class: int, seed: int) -> list[dict]:
    specs: list[tuple[str, tuple[str | int, ...], int]] = [
        ("cifar10_catvsdog", ("cat", "dog"), binary_n_per_class),
        ("cifar10_automobilevstruck", ("automobile", "truck"), binary_n_per_class),
        ("cifar10_vehicles4", ("airplane", "automobile", "ship", "truck"), multi_n_per_class),
    ]
    tasks = []
    for offset, (name, classes, n_per_class) in enumerate(specs):
        train_x, train_y, names = select_cifar_subset(root, classes, n_per_class, "train", seed + offset)
        test_x, test_y, _ = select_cifar_subset(root, classes, n_per_class, "test", seed + 100 + offset)
        tasks.append(
            {
                "name": name,
                "classes": names,
                "binary": len(names) == 2,
                "train_x": train_x,
                "train_y": train_y,
                "test_x": test_x,
                "test_y": test_y,
            }
        )
    return tasks


class ResNet18MetricModel(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        weights = tv_models.ResNet18_Weights.IMAGENET1K_V1
        self.backbone = tv_models.resnet18(weights=weights)
        feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.head = nn.Linear(feature_dim, num_classes)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def set_trainable(model: ResNet18MetricModel, regime: str) -> None:
    for param in model.parameters():
        param.requires_grad = False
    for param in model.head.parameters():
        param.requires_grad = True
    if regime == "frozen_head":
        return
    if regime == "finetune_layer4":
        for param in model.backbone.layer4.parameters():
            param.requires_grad = True
        return
    if regime == "finetune_all":
        for param in model.backbone.parameters():
            param.requires_grad = True
        return
    raise ValueError(f"unknown regime: {regime}")


def make_optimizer(model: ResNet18MetricModel, regime: str, args: argparse.Namespace) -> torch.optim.Optimizer:
    if regime == "frozen_head":
        return torch.optim.AdamW(model.head.parameters(), lr=args.head_lr, weight_decay=args.weight_decay)
    if regime == "finetune_layer4":
        return torch.optim.AdamW(
            [
                {"params": model.backbone.layer4.parameters(), "lr": args.backbone_lr},
                {"params": model.head.parameters(), "lr": args.head_lr},
            ],
            weight_decay=args.weight_decay,
        )
    if regime == "finetune_all":
        return torch.optim.AdamW(
            [
                {"params": model.backbone.parameters(), "lr": args.full_backbone_lr},
                {"params": model.head.parameters(), "lr": args.head_lr},
            ],
            weight_decay=args.weight_decay,
        )
    raise ValueError(f"unknown regime: {regime}")


def set_epoch_train_mode(model: ResNet18MetricModel, regime: str) -> None:
    """Set train/eval modes without letting frozen BatchNorm stats drift."""
    model.train()
    if regime == "frozen_head":
        model.backbone.eval()
        model.head.train()
        return
    if regime == "finetune_layer4":
        model.backbone.eval()
        model.backbone.layer4.train()
        model.head.train()
        return
    if regime == "finetune_all":
        model.train()
        return
    raise ValueError(f"unknown regime: {regime}")


def preprocess_batch(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    x = x.to(device=device, dtype=torch.float32)
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)
    return (x - mean) / std


def random_crop_flip(x: torch.Tensor, padding: int = 4) -> torch.Tensor:
    padded = F.pad(x, (padding, padding, padding, padding), mode="reflect")
    out = torch.empty_like(x)
    max_offset = 2 * padding
    ys = torch.randint(0, max_offset + 1, (x.shape[0],))
    xs = torch.randint(0, max_offset + 1, (x.shape[0],))
    for i in range(x.shape[0]):
        out[i] = padded[i, :, ys[i] : ys[i] + x.shape[2], xs[i] : xs[i] + x.shape[3]]
    flips = torch.rand(x.shape[0]) < 0.5
    out[flips] = torch.flip(out[flips], dims=[3])
    return out


def label_for_spectral(y: np.ndarray, binary: bool) -> np.ndarray:
    y = np.asarray(y).reshape(-1)
    if binary:
        return np.where(y == 0, 1.0, -1.0)
    return y.astype(np.int64)


def standardize_features(train_z: np.ndarray, test_z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_z.mean(axis=0, keepdims=True)
    std = train_z.std(axis=0, keepdims=True) + 1e-12
    return (train_z - mean) / std, (test_z - mean) / std


def row_normalize(z: np.ndarray) -> np.ndarray:
    return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)


def linear_gram_blocks(train_z: np.ndarray, test_z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_z, test_z = standardize_features(train_z, test_z)
    train_z = row_normalize(train_z)
    test_z = row_normalize(test_z)
    return train_z @ train_z.T, test_z @ test_z.T, test_z @ train_z.T


def margin_stats_np(logits: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    pred = np.argmax(logits, axis=1)
    true_scores = logits[np.arange(len(labels)), labels]
    masked = logits.copy()
    masked[np.arange(len(labels)), labels] = -np.inf
    margins = true_scores - np.max(masked, axis=1)
    return float(np.mean(pred == labels)), float(np.mean(margins)), float(np.median(margins))


def split_diagnostics(k: np.ndarray, labels: np.ndarray, binary: bool, k_neighbors: int) -> dict:
    d2 = kernels.kernel_metric_squared(k)
    spec_labels = label_for_spectral(labels, binary)
    if binary:
        spec = spectral.summarize(k, spec_labels)
        mix = mixing.summarize(d2, spec_labels, k=k_neighbors)
        return {
            "tail_at_10pct": spec.tail_at_10pct,
            "tail_auc": spec.tail_auc,
            "alignment": spec.alignment,
            "mixing": mix.knn_opposite_ratio,
            "local_entropy": mix.local_label_entropy,
            "graph_dirichlet": mix.graph_dirichlet,
            "graph_disagreement": mix.graph_disagreement,
        }
    spec = spectral.summarize_multiclass(k, spec_labels)
    mix = mixing.summarize_multiclass(d2, spec_labels, k=k_neighbors)
    return {
        "tail_at_10pct": spec.tail_at_10pct,
        "tail_auc": spec.tail_auc,
        "alignment": spec.alignment,
        "mixing": mix.knn_disagreement_ratio,
        "local_entropy": mix.local_label_entropy_normalized,
        "graph_dirichlet": mix.graph_dirichlet,
        "graph_disagreement": mix.graph_disagreement,
    }


def ridge_diagnostics(
    k_train: np.ndarray,
    k_test_train: np.ndarray,
    train_y: np.ndarray,
    test_y: np.ndarray,
    binary: bool,
    ridge: float,
) -> dict:
    if binary:
        train_sign = label_for_spectral(train_y, binary=True)
        test_sign = label_for_spectral(test_y, binary=True)
        result = kernel_ridge.fit_binary_kernel_ridge(k_train, train_sign, k_test_train, test_sign, ridge=ridge)
    else:
        result = kernel_ridge.fit_multiclass_kernel_ridge(k_train, train_y, k_test_train, test_y, ridge=ridge)
    return kernel_ridge.to_jsonable(result)


def extract_features_logits(
    model: ResNet18MetricModel,
    x_np: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    feats = []
    logits = []
    with torch.no_grad():
        for start in range(0, len(x_np), batch_size):
            raw = torch.tensor(x_np[start : start + batch_size], dtype=torch.float32)
            batch = preprocess_batch(raw, device=device)
            feats.append(model.features(batch).detach().cpu().numpy())
            logits.append(model(batch).detach().cpu().numpy())
    return np.concatenate(feats, axis=0), np.concatenate(logits, axis=0)


def record_checkpoint(
    model: ResNet18MetricModel,
    task: dict,
    regime: str,
    epoch: int,
    device: torch.device,
    args: argparse.Namespace,
    k0_train: np.ndarray | None,
    k0_test: np.ndarray | None,
) -> tuple[dict, np.ndarray, np.ndarray]:
    train_z, train_logits = extract_features_logits(model, task["train_x"], device=device, batch_size=args.eval_batch_size)
    test_z, test_logits = extract_features_logits(model, task["test_x"], device=device, batch_size=args.eval_batch_size)
    k_train, k_test, k_test_train = linear_gram_blocks(train_z, test_z)
    train_diag = split_diagnostics(k_train, task["train_y"], task["binary"], args.k_neighbors)
    test_diag = split_diagnostics(k_test, task["test_y"], task["binary"], args.k_neighbors)
    ridge = ridge_diagnostics(k_train, k_test_train, task["train_y"], task["test_y"], task["binary"], ridge=args.ridge)
    head_train_acc, head_train_margin_mean, head_train_margin_median = margin_stats_np(train_logits, task["train_y"])
    head_test_acc, head_test_margin_mean, head_test_margin_median = margin_stats_np(test_logits, task["test_y"])
    movement_train = None if k0_train is None else float(np.linalg.norm(k_train - k0_train, "fro") / (np.linalg.norm(k0_train, "fro") + 1e-12))
    movement_test = None if k0_test is None else float(np.linalg.norm(k_test - k0_test, "fro") / (np.linalg.norm(k0_test, "fro") + 1e-12))
    row = {
        "dataset": task["name"],
        "regime": regime,
        "epoch": epoch,
        "classes": len(task["classes"]),
        "binary": bool(task["binary"]),
        "feature_dim": int(train_z.shape[1]),
        "kernel_movement_train": movement_train,
        "kernel_movement_test": movement_test,
        "train": train_diag,
        "test": test_diag,
        "head_train_accuracy": head_train_acc,
        "head_test_accuracy": head_test_acc,
        "head_train_margin_mean": head_train_margin_mean,
        "head_train_margin_median": head_train_margin_median,
        "head_test_margin_mean": head_test_margin_mean,
        "head_test_margin_median": head_test_margin_median,
        "kernel_ridge": ridge,
    }
    return row, k_train, k_test


def train_regime(task: dict, regime: str, args: argparse.Namespace, device: torch.device, seed: int) -> list[dict]:
    models.set_seed(seed)
    model = ResNet18MetricModel(num_classes=len(task["classes"])).to(device)
    set_trainable(model, regime)
    optimizer = make_optimizer(model, regime, args)
    loss_fn = nn.CrossEntropyLoss()
    checkpoints = sorted(set([0, 1, 3, args.epochs]))
    trace = []
    first, k0_train, k0_test = record_checkpoint(model, task, regime, 0, device, args, None, None)
    first["kernel_movement_train"] = 0.0
    first["kernel_movement_test"] = 0.0
    trace.append(first)

    x_train = torch.tensor(task["train_x"], dtype=torch.float32)
    y_train = torch.tensor(task["train_y"], dtype=torch.long)
    n = len(y_train)
    for epoch in range(1, args.epochs + 1):
        set_epoch_train_mode(model, regime)
        order = torch.randperm(n)
        for start in range(0, n, args.batch_size):
            idx = order[start : start + args.batch_size]
            raw = x_train[idx]
            if args.augment:
                raw = random_crop_flip(raw)
            batch = preprocess_batch(raw, device=device)
            target = y_train[idx].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch), target)
            loss.backward()
            optimizer.step()
        if epoch in checkpoints:
            row, _, _ = record_checkpoint(model, task, regime, epoch, device, args, k0_train, k0_test)
            trace.append(row)
    return trace


def safe_corr(xs: list[float], ys: list[float]) -> float:
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def final_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["dataset"], row["regime"]), []).append(row)
    finals = []
    for group_rows in grouped.values():
        group_rows = sorted(group_rows, key=lambda item: item["epoch"])
        first = group_rows[0]
        last = group_rows[-1]
        out = dict(last)
        out["test_tail_delta"] = last["test"]["tail_at_10pct"] - first["test"]["tail_at_10pct"]
        out["test_tail_decrease"] = first["test"]["tail_at_10pct"] - last["test"]["tail_at_10pct"]
        out["test_graph_delta"] = last["test"]["graph_dirichlet"] - first["test"]["graph_dirichlet"]
        out["test_mixing_delta"] = last["test"]["mixing"] - first["test"]["mixing"]
        out["head_test_accuracy_delta"] = last["head_test_accuracy"] - first["head_test_accuracy"]
        out["ridge_test_accuracy_delta"] = last["kernel_ridge"]["test_accuracy"] - first["kernel_ridge"]["test_accuracy"]
        finals.append(out)
    return finals


def plot_over_time(rows: list[dict], key_path: tuple[str, ...], ylabel: str, out: Path) -> None:
    plotting.setup()
    plt.figure(figsize=(7.8, 4.8))
    for dataset in sorted({row["dataset"] for row in rows}):
        for regime in sorted({row["regime"] for row in rows if row["dataset"] == dataset}):
            rs = sorted([row for row in rows if row["dataset"] == dataset and row["regime"] == regime], key=lambda item: item["epoch"])
            values = []
            for row in rs:
                value = row
                for key in key_path:
                    value = value[key]
                values.append(value)
            plt.plot([row["epoch"] for row in rs], values, marker="o", linewidth=1.4, markersize=3, label=f"{dataset}/{regime}")
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.legend(fontsize=5.8)
    plotting.savefig(out)


def plot_movement_vs_tail(finals: list[dict]) -> None:
    plotting.setup()
    plt.figure(figsize=(6.8, 4.8))
    for regime in sorted({row["regime"] for row in finals}):
        rs = [row for row in finals if row["regime"] == regime]
        plt.scatter([row["kernel_movement_test"] for row in rs], [row["test_tail_decrease"] for row in rs], s=48, alpha=0.84, label=regime)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xlabel("test kernel movement")
    plt.ylabel("test tail@10% decrease")
    plt.legend(fontsize=7)
    plotting.savefig(EXP_DIR / "figures" / "kernel_movement_vs_tail_decrease.png")


def plot_results(rows: list[dict]) -> None:
    plot_over_time(rows, ("test", "tail_at_10pct"), "test tail@10%", EXP_DIR / "figures" / "test_tail_over_time.png")
    plot_over_time(rows, ("test", "graph_dirichlet"), "test graph Dirichlet", EXP_DIR / "figures" / "test_graph_dirichlet_over_time.png")
    plot_over_time(rows, ("test", "mixing"), "test mixing/disagreement", EXP_DIR / "figures" / "test_mixing_over_time.png")
    plot_over_time(rows, ("head_test_margin_median",), "test head margin median", EXP_DIR / "figures" / "test_head_margin_over_time.png")
    plot_over_time(rows, ("kernel_ridge", "test_margin_median"), "test ridge margin median", EXP_DIR / "figures" / "test_ridge_margin_over_time.png")
    plot_movement_vs_tail(final_rows(rows))


def write_result_md(rows: list[dict], command: str) -> None:
    finals = final_rows(rows)
    regime_means = {}
    for regime in sorted({row["regime"] for row in finals}):
        rs = [row for row in finals if row["regime"] == regime]
        regime_means[regime] = {
            "movement": float(np.mean([row["kernel_movement_test"] for row in rs])),
            "tail_delta": float(np.mean([row["test_tail_delta"] for row in rs])),
            "graph_delta": float(np.mean([row["test_graph_delta"] for row in rs])),
            "head_acc_delta": float(np.mean([row["head_test_accuracy_delta"] for row in rs])),
            "ridge_acc_delta": float(np.mean([row["ridge_test_accuracy_delta"] for row in rs])),
        }
    movement = [row["kernel_movement_test"] for row in finals]
    tail_decrease = [row["test_tail_decrease"] for row in finals]
    ridge_margin = [row["kernel_ridge"]["test_margin_median"] for row in finals]
    tail = [row["test"]["tail_at_10pct"] for row in finals]
    head_acc = [row["head_test_accuracy"] for row in finals]
    ridge_acc = [row["kernel_ridge"]["test_accuracy"] for row in finals]
    lines = [
        "# Result",
        "",
        "## Run",
        "",
        f"Command: `{command}`",
        "",
        "## Summary",
        "",
        f"- corr(final test movement, test tail decrease) = `{safe_corr(movement, tail_decrease):.3f}`",
        f"- corr(final test tail, ridge test margin) = `{safe_corr(tail, ridge_margin):.3f}`",
        f"- corr(final test tail, head test accuracy) = `{safe_corr(tail, head_acc):.3f}`",
        f"- corr(final test tail, ridge test accuracy) = `{safe_corr(tail, ridge_acc):.3f}`",
        "",
        "Regime means over tasks:",
        "",
        "| regime | movement | tail delta | graph delta | head acc delta | ridge acc delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for regime, values in regime_means.items():
        lines.append(
            "| {regime} | {movement:.3f} | {tail_delta:.3f} | {graph_delta:.3f} | {head_acc_delta:.3f} | {ridge_acc_delta:.3f} |".format(
                regime=regime,
                **values,
            )
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- `frozen_head` is the control: head margin can change while the feature",
            "  metric should not move.",
            "- A fine-tuning regime supports metric adaptation only if held-out",
            "  movement is paired with lower test tail/graph roughness and better",
            "  margins/accuracy.",
            "- In the default run, `finetune_layer4` is the cleanest metric-repair",
            "  regime: test movement is moderate and mean tail/graph deltas are",
            "  negative.",
            "- `finetune_all` is an important negative control: it moves the metric",
            "  more, but mean held-out tail and graph roughness increase.",
            "- If movement is large but test tail does not decrease, treat it as weak",
            "  transfer rather than a positive feature-learning result.",
            "",
            "| dataset | regime | movement | test tail final | test tail delta | graph delta | mix delta | head acc final | ridge acc final | ridge margin final |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(finals, key=lambda item: (item["dataset"], item["regime"])):
        lines.append(
            "| {dataset} | {regime} | {move:.3f} | {tail:.3f} | {tdelta:.3f} | {gdelta:.3f} | {mdelta:.3f} | {hacc:.3f} | {racc:.3f} | {rmargin:.3f} |".format(
                dataset=row["dataset"],
                regime=row["regime"],
                move=row["kernel_movement_test"],
                tail=row["test"]["tail_at_10pct"],
                tdelta=row["test_tail_delta"],
                gdelta=row["test_graph_delta"],
                mdelta=row["test_mixing_delta"],
                hacc=row["head_test_accuracy"],
                racc=row["kernel_ridge"]["test_accuracy"],
                rmargin=row["kernel_ridge"]["test_margin_median"],
            )
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `metrics.json`",
            "- `figures/test_tail_over_time.png`",
            "- `figures/test_graph_dirichlet_over_time.png`",
            "- `figures/test_mixing_over_time.png`",
            "- `figures/test_head_margin_over_time.png`",
            "- `figures/test_ridge_margin_over_time.png`",
            "- `figures/kernel_movement_vs_tail_decrease.png`",
        ]
    )
    (EXP_DIR / "result.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--binary-n-per-class", type=int, default=50)
    parser.add_argument("--multi-n-per-class", type=int, default=25)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=3e-5)
    parser.add_argument("--full-backbone-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--k-neighbors", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.binary_n_per_class = 8
        args.multi_n_per_class = 5
        args.epochs = 1
        args.batch_size = 8
        args.eval_batch_size = 16

    device = choose_device(args.device)
    tasks = build_tasks(args.data_root, args.binary_n_per_class, args.multi_n_per_class, args.seed)
    regimes = ["frozen_head", "finetune_layer4", "finetune_all"]
    rows = []
    for task_idx, task in enumerate(tasks):
        for regime_idx, regime in enumerate(regimes):
            print(f"Running {task['name']} / {regime} on {device}", flush=True)
            rows.extend(train_regime(task, regime, args, device, seed=args.seed + 1000 * task_idx + regime_idx))

    payload = {
        "config": {
            "binary_n_per_class": args.binary_n_per_class,
            "multi_n_per_class": args.multi_n_per_class,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "head_lr": args.head_lr,
            "backbone_lr": args.backbone_lr,
            "full_backbone_lr": args.full_backbone_lr,
            "weight_decay": args.weight_decay,
            "ridge": args.ridge,
            "k_neighbors": args.k_neighbors,
            "seed": args.seed,
            "device": str(device),
            "augment": bool(args.augment),
            "torchvision_weights": "ResNet18_Weights.IMAGENET1K_V1",
        },
        "results": rows,
    }
    (EXP_DIR / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot_results(rows)
    command = "python3 experiments/013-pretrained-finetune-metric-dynamics/scripts/run_finetune_dynamics.py"
    if args.quick:
        command += " --quick"
    else:
        command += f" --binary-n-per-class {args.binary_n_per_class} --multi-n-per-class {args.multi_n_per_class}"
        command += f" --epochs {args.epochs} --batch-size {args.batch_size} --eval-batch-size {args.eval_batch_size}"
    command += f" --device {args.device}"
    if args.augment:
        command += " --augment"
    write_result_md(rows, command)
    print(f"Wrote {EXP_DIR / 'metrics.json'}")
    print(f"Wrote {EXP_DIR / 'result.md'}")


if __name__ == "__main__":
    main()
