#!/usr/bin/env python3
"""Frozen self-supervised representation sweep for experiment 011."""

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

from src import datasets, kernel_ridge, kernels, mixing, plotting, spectral


EXP_DIR = ROOT / "experiments" / "011-self-supervised-fixed-representation-sweep"
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)


def class_id(cls: str | int) -> int:
    if isinstance(cls, int):
        return cls
    return datasets.CIFAR10_LABELS.index(cls)


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


def select_cifar_subset(
    root: Path,
    classes: tuple[str | int, ...],
    n_per_class: int,
    split: str,
    seed: int,
    binary_sign: bool,
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
        if binary_sign:
            sign = 1.0 if out_label == 0 else -1.0
            ys.append(np.full(n_per_class, sign, dtype=np.float64))
        else:
            ys.append(np.full(n_per_class, out_label, dtype=np.int64))
    out_x = np.concatenate(xs, axis=0).reshape(-1, 3, 32, 32)
    out_y = np.concatenate(ys, axis=0)
    perm = rng.permutation(len(out_y))
    return out_x[perm], out_y[perm], names


def build_tasks(root: Path, binary_n_per_class: int, multi_n_per_class: int, seed: int) -> list[dict]:
    specs: list[tuple[str, tuple[str | int, ...], int, bool]] = [
        ("cifar10_catvsdog", ("cat", "dog"), binary_n_per_class, True),
        ("cifar10_automobilevstruck", ("automobile", "truck"), binary_n_per_class, True),
        ("cifar10_all10", tuple(range(10)), multi_n_per_class, False),
        ("cifar10_animals6", ("bird", "cat", "deer", "dog", "frog", "horse"), multi_n_per_class, False),
        ("cifar10_vehicles4", ("airplane", "automobile", "ship", "truck"), multi_n_per_class, False),
    ]
    tasks = []
    for offset, (name, classes, n_per_class, binary) in enumerate(specs):
        train_x, train_y, names = select_cifar_subset(root, classes, n_per_class, "train", seed + offset, binary_sign=binary)
        test_x, test_y, _ = select_cifar_subset(root, classes, n_per_class, "test", seed + 100 + offset, binary_sign=binary)
        tasks.append(
            {
                "name": name,
                "class_names": names,
                "binary": binary,
                "train_x": train_x,
                "train_y": train_y,
                "test_x": test_x,
                "test_y": test_y,
            }
        )
    return tasks


def make_resnet_feature_extractor(pretrained: bool, device: torch.device) -> nn.Module:
    weights = tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = tv_models.resnet18(weights=weights)
    feature_extractor = nn.Sequential(*list(model.children())[:-1], nn.Flatten()).to(device)
    feature_extractor.eval()
    return feature_extractor


def make_dino_feature_extractor(device: torch.device) -> nn.Module:
    model = torch.hub.load("facebookresearch/dino:main", "dino_vits16", pretrained=True)
    model.to(device)
    model.eval()
    return model


def preprocess_batch(x: np.ndarray, device: torch.device) -> torch.Tensor:
    batch = torch.tensor(x, dtype=torch.float32, device=device)
    batch = F.interpolate(batch, size=(224, 224), mode="bilinear", align_corners=False)
    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)
    return (batch - mean) / std


def extract_resnet_features(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    feats = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = preprocess_batch(x[start : start + batch_size], device=device)
            feats.append(model(batch).detach().cpu().numpy())
    return np.concatenate(feats, axis=0)


def raw_pixel_features(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return train_x.reshape(train_x.shape[0], -1), test_x.reshape(test_x.shape[0], -1)


def diagnostics(task: dict, representation: str, train_z: np.ndarray, test_z: np.ndarray, ridge: float) -> dict:
    k_train, k_test, k_test_train = linear_gram_blocks(train_z, test_z)
    d2 = kernels.kernel_metric_squared(k_test)
    if task["binary"]:
        spec = spectral.summarize(k_test, task["test_y"])
        mix = mixing.summarize(d2, task["test_y"], k=10)
        clf = kernel_ridge.fit_binary_kernel_ridge(k_train, task["train_y"], k_test_train, task["test_y"], ridge=ridge)
        mix_value = mix.knn_opposite_ratio
        entropy = mix.local_label_entropy
    else:
        spec = spectral.summarize_multiclass(k_test, task["test_y"])
        mix = mixing.summarize_multiclass(d2, task["test_y"], k=10)
        clf = kernel_ridge.fit_multiclass_kernel_ridge(k_train, task["train_y"], k_test_train, task["test_y"], ridge=ridge)
        mix_value = mix.knn_disagreement_ratio
        entropy = mix.local_label_entropy_normalized
    return {
        "dataset": task["name"],
        "representation": representation,
        "classes": len(task["class_names"]),
        "binary": task["binary"],
        "feature_dim": int(train_z.shape[1]),
        "test_tail_at_10pct": spec.tail_at_10pct,
        "test_tail_auc": spec.tail_auc,
        "test_alignment": spec.alignment,
        "test_mixing": mix_value,
        "test_local_entropy": entropy,
        "test_graph_dirichlet": mix.graph_dirichlet,
        "test_graph_disagreement": mix.graph_disagreement,
        "kernel_ridge": kernel_ridge.to_jsonable(clf),
    }


def safe_corr(xs: list[float], ys: list[float]) -> float:
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def plot_scatter(rows: list[dict], x_key: str, y_getter, xlabel: str, ylabel: str, out: Path) -> None:
    plotting.setup()
    plt.figure(figsize=(6.8, 4.8))
    for rep in sorted({r["representation"] for r in rows}):
        rs = [r for r in rows if r["representation"] == rep]
        plt.scatter([r[x_key] for r in rs], [y_getter(r) for r in rs], s=44, alpha=0.84, label=rep)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(fontsize=7)
    plotting.savefig(out)


def plot_results(rows: list[dict]) -> None:
    plot_scatter(
        rows,
        "test_mixing",
        lambda r: r["test_tail_at_10pct"],
        "test local mixing/disagreement",
        "test tail@10%",
        EXP_DIR / "figures" / "mixing_vs_tail.png",
    )
    plot_scatter(
        rows,
        "test_tail_at_10pct",
        lambda r: r["kernel_ridge"]["test_accuracy"],
        "test tail@10%",
        "kernel ridge test accuracy",
        EXP_DIR / "figures" / "tail_vs_accuracy.png",
    )
    plot_scatter(
        rows,
        "test_tail_at_10pct",
        lambda r: r["kernel_ridge"]["test_margin_median"],
        "test tail@10%",
        "kernel ridge test margin median",
        EXP_DIR / "figures" / "tail_vs_margin.png",
    )


def write_result_md(rows: list[dict], command: str) -> None:
    tail = [r["test_tail_at_10pct"] for r in rows]
    mix = [r["test_mixing"] for r in rows]
    graph = [r["test_graph_dirichlet"] for r in rows]
    acc = [r["kernel_ridge"]["test_accuracy"] for r in rows]
    margin = [r["kernel_ridge"]["test_margin_median"] for r in rows]
    lines = [
        "# Result",
        "",
        "## Run",
        "",
        f"Command: `{command}`",
        "",
        "## Summary",
        "",
        f"- corr(test local mixing/disagreement, test tail@10%) = `{safe_corr(mix, tail):.3f}`",
        f"- corr(test graph Dirichlet, test tail@10%) = `{safe_corr(graph, tail):.3f}`",
        f"- corr(test tail@10%, kernel ridge test accuracy) = `{safe_corr(tail, acc):.3f}`",
        f"- corr(test tail@10%, kernel ridge test margin median) = `{safe_corr(tail, margin):.3f}`",
        "",
        "Interpretation:",
        "",
        "- DINO ViT-S/16 features are a self-supervised frozen representation",
        "  baseline, not a CNTK or toy-network kernel.",
        "- If self-supervised features lower local mixing/tail and improve ridge",
        "  margin, that supports the broader fixed-representation obstruction",
        "  framing.",
        "",
        "| dataset | representation | classes | dim | tail@10% | mixing | graph dir | align | ridge acc | ridge margin | source norm |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in sorted(rows, key=lambda item: (item["dataset"], item["representation"])):
        kr = r["kernel_ridge"]
        lines.append(
            "| {dataset} | {rep} | {classes} | {dim} | {tail:.3f} | {mix:.3f} | {graph:.3f} | {align:.3f} | {acc:.3f} | {margin:.3f} | {norm:.2f} |".format(
                dataset=r["dataset"],
                rep=r["representation"],
                classes=r["classes"],
                dim=r["feature_dim"],
                tail=r["test_tail_at_10pct"],
                mix=r["test_mixing"],
                graph=r["test_graph_dirichlet"],
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
            "- `figures/mixing_vs_tail.png`",
            "- `figures/tail_vs_accuracy.png`",
            "- `figures/tail_vs_margin.png`",
        ]
    )
    (EXP_DIR / "result.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--binary-n-per-class", type=int, default=60)
    parser.add_argument("--multi-n-per-class", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    tasks = build_tasks(args.data_root, args.binary_n_per_class, args.multi_n_per_class, args.seed)
    pretrained_resnet = make_resnet_feature_extractor(pretrained=True, device=device)
    dino = make_dino_feature_extractor(device=device)

    rows = []
    for task in tasks:
        print(f"Running {task['name']}", flush=True)
        train_raw, test_raw = raw_pixel_features(task["train_x"], task["test_x"])
        rows.append(diagnostics(task, "raw_pixels", train_raw, test_raw, ridge=args.ridge))
        train_pre = extract_resnet_features(pretrained_resnet, task["train_x"], device=device, batch_size=args.batch_size)
        test_pre = extract_resnet_features(pretrained_resnet, task["test_x"], device=device, batch_size=args.batch_size)
        rows.append(diagnostics(task, "imagenet_resnet18", train_pre, test_pre, ridge=args.ridge))
        train_dino = extract_resnet_features(dino, task["train_x"], device=device, batch_size=args.batch_size)
        test_dino = extract_resnet_features(dino, task["test_x"], device=device, batch_size=args.batch_size)
        rows.append(diagnostics(task, "dino_vits16", train_dino, test_dino, ridge=args.ridge))

    payload = {
        "config": {
            "binary_n_per_class": args.binary_n_per_class,
            "multi_n_per_class": args.multi_n_per_class,
            "batch_size": args.batch_size,
            "ridge": args.ridge,
            "seed": args.seed,
            "device": args.device,
            "supervised_weights": "torchvision ResNet18_Weights.IMAGENET1K_V1",
            "self_supervised_weights": "torch.hub facebookresearch/dino dino_vits16",
        },
        "results": rows,
    }
    (EXP_DIR / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot_results(rows)
    command = "python3 experiments/011-self-supervised-fixed-representation-sweep/scripts/run_selfsup.py"
    command += f" --binary-n-per-class {args.binary_n_per_class} --multi-n-per-class {args.multi_n_per_class}"
    command += f" --batch-size {args.batch_size} --ridge {args.ridge} --seed {args.seed} --device {args.device}"
    write_result_md(rows, command)
    print(f"Wrote {EXP_DIR / 'metrics.json'}")
    print(f"Wrote {EXP_DIR / 'result.md'}")


if __name__ == "__main__":
    main()
