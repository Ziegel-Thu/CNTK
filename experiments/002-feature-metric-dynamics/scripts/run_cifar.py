#!/usr/bin/env python3
"""CIFAR-10 feature-metric dynamics for experiment 002."""

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


EXP_DIR = ROOT / "experiments" / "002-feature-metric-dynamics"


def binary_subset(
    x: np.ndarray,
    labels: np.ndarray,
    classes: tuple[str, str],
    n_per_class: int,
    seed: int,
    name: str,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> tuple[datasets.Dataset, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    class_ids = (datasets.CIFAR10_LABELS.index(classes[0]), datasets.CIFAR10_LABELS.index(classes[1]))
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
    return datasets.Dataset(name=name, x=out_x, y=out_y), mean, std


def make_cifar_train_test(
    root: Path,
    classes: tuple[str, str],
    n_per_class: int,
    seed: int,
) -> tuple[datasets.Dataset, datasets.Dataset]:
    x_train, y_train = datasets.load_cifar10(root, split="train")
    x_test, y_test = datasets.load_cifar10(root, split="test")
    label = f"{classes[0]}vs{classes[1]}_n{n_per_class}"
    train, mean, std = binary_subset(
        x_train,
        y_train,
        classes,
        n_per_class=n_per_class,
        seed=seed,
        name=f"cifar10_{label}_train",
    )
    test, _, _ = binary_subset(
        x_test,
        y_test,
        classes,
        n_per_class=n_per_class,
        seed=seed + 100,
        name=f"cifar10_{label}_test",
        mean=mean,
        std=std,
    )
    return train, test


def feature_diagnostics(
    model: models.MLP,
    x_np: np.ndarray,
    y_sign: np.ndarray,
    k0: np.ndarray | None,
    device: torch.device,
) -> dict:
    model.eval()
    with torch.no_grad():
        x = torch.tensor(x_np, dtype=torch.float32, device=device)
        feats = model.features(x).detach().cpu().numpy()
        logits = model(x).detach().cpu().numpy()
    k = kernels.feature_gram(feats, normalize=True)
    d2 = kernels.kernel_metric_squared(k)
    spec = spectral.summarize(k, y_sign)
    mix = mixing.summarize(d2, y_sign, k=10)
    movement = None
    if k0 is not None:
        movement = float(np.linalg.norm(k - k0, ord="fro") / (np.linalg.norm(k0, ord="fro") + 1e-12))
    margin = y_sign * logits
    return {
        "tail_at_10pct": spec.tail_at_10pct,
        "tail_auc": spec.tail_auc,
        "alignment": spec.alignment,
        "knn_opposite_ratio": mix.knn_opposite_ratio,
        "local_label_entropy": mix.local_label_entropy,
        "opposite_nn_mean": mix.opposite_nn_mean,
        "collision_rate": mix.collision_rate,
        "kernel_movement": movement,
        "margin_mean": float(np.mean(margin)),
        "margin_median": float(np.median(margin)),
    }


def make_model(regime: str, input_dim: int) -> tuple[models.MLP, str, float]:
    if regime == "frozen_random":
        model = models.MLP(input_dim=input_dim, width=768, depth=2)
        model.freeze_features()
        return model, "adam", 1e-2
    if regime == "feature_learning":
        return models.MLP(input_dim=input_dim, width=512, depth=2), "adam", 1e-3
    if regime == "lazy_wide_small_lr":
        return models.MLP(input_dim=input_dim, width=2048, depth=2), "sgd", 1e-3
    raise ValueError(f"unknown regime: {regime}")


def train_regime(
    train: datasets.Dataset,
    test: datasets.Dataset,
    regime: str,
    epochs: int,
    seed: int,
    device: torch.device,
) -> dict:
    models.set_seed(seed)
    x_train = torch.tensor(train.x, dtype=torch.float32, device=device)
    y_train01 = torch.tensor((train.y > 0).astype(np.float32), device=device)
    x_test = torch.tensor(test.x, dtype=torch.float32, device=device)
    y_test01 = torch.tensor((test.y > 0).astype(np.float32), device=device)

    model, opt_kind, lr = make_model(regime, input_dim=train.x.shape[1])
    model.to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr) if opt_kind == "adam" else torch.optim.SGD(trainable_params, lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    checkpoints = sorted(set([0, 1, 5, 20, 60, 100, epochs]))
    with torch.no_grad():
        k0_train = kernels.feature_gram(model.features(x_train).detach().cpu().numpy(), normalize=True)
        k0_test = kernels.feature_gram(model.features(x_test).detach().cpu().numpy(), normalize=True)

    trace = []

    def record(epoch: int, loss_value: float | None = None) -> None:
        model.eval()
        with torch.no_grad():
            train_logits = model(x_train)
            test_logits = model(x_test)
            train_acc = models.binary_accuracy(train_logits, y_train01)
            test_acc = models.binary_accuracy(test_logits, y_test01)
            if loss_value is None:
                loss_value = float(loss_fn(train_logits, y_train01).item())
        train_diag = feature_diagnostics(model, train.x, train.y, k0_train, device=device)
        test_diag = feature_diagnostics(model, test.x, test.y, k0_test, device=device)
        test_diag = {f"test_{key}": value for key, value in test_diag.items()}
        trace.append(
            {
                "epoch": epoch,
                "loss": loss_value,
                "train_acc": train_acc,
                "test_acc": test_acc,
                **train_diag,
                **test_diag,
            }
        )

    record(0)
    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_train)
        loss = loss_fn(logits, y_train01)
        loss.backward()
        optimizer.step()
        if epoch in checkpoints:
            record(epoch, float(loss.item()))
            model.train()
    return {"regime": regime, "optimizer": opt_kind, "lr": lr, "trace": trace}


def plot_metric(results: list[dict], metric: str, ylabel: str, out: Path) -> None:
    plotting.setup()
    plt.figure(figsize=(7.4, 4.6))
    for run in results:
        for regime in run["regimes"]:
            epochs = [p["epoch"] for p in regime["trace"]]
            values = [p[metric] if p[metric] is not None else 0.0 for p in regime["trace"]]
            plt.plot(epochs, values, marker="o", label=f"{run['dataset']} / {regime['regime']}")
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.title(f"CIFAR-10 {ylabel}")
    plt.legend(fontsize=6)
    plotting.savefig(out)


def write_result_md(results: list[dict], command: str, out: Path) -> None:
    lines = [
        "# CIFAR-10 Feature-Dynamics Result",
        "",
        "## Run",
        "",
        f"Command: `{command}`",
        "",
        "## Summary",
        "",
        "Main observations to check after each run:",
        "",
        "- If train tail collapses but test tail stays high, the run is mostly memorization.",
        "- Raw-pixel MLPs may lack the inductive bias needed for CIFAR metric adaptation.",
        "- A small CNN should be the next CIFAR feature-dynamics control if raw MLPs do not transfer.",
        "",
        "| dataset | regime | train tail start | train tail final | test tail start | test tail final | train mix final | test mix final | movement final | test acc |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in results:
        for regime in run["regimes"]:
            first = regime["trace"][0]
            last = regime["trace"][-1]
            lines.append(
                "| {dataset} | {regime} | {tail0:.3f} | {tail1:.3f} | {ttail0:.3f} | {ttail1:.3f} | {mix1:.3f} | {tmix1:.3f} | {move:.3f} | {te:.3f} |".format(
                    dataset=run["dataset"],
                    regime=regime["regime"],
                    tail0=first["tail_at_10pct"],
                    tail1=last["tail_at_10pct"],
                    ttail0=first["test_tail_at_10pct"],
                    ttail1=last["test_tail_at_10pct"],
                    mix1=last["knn_opposite_ratio"],
                    tmix1=last["test_knn_opposite_ratio"],
                    move=last["kernel_movement"] or 0.0,
                    te=last["test_acc"],
                )
            )
    lines.extend(
        [
            "",
            "## Interpretation Prompts",
            "",
            "- Does feature learning reduce test tail on CIFAR, or mainly memorize train geometry?",
            "- Is `cat vs dog` harder than `automobile vs truck` in metric-dynamics terms?",
            "- Does a raw-pixel MLP provide enough inductive bias, or should the next CIFAR run use a small CNN?",
            "",
            "## Artifacts",
            "",
            "- `metrics_cifar_over_time.json`",
            "- `figures/cifar_tail_over_time.png`",
            "- `figures/cifar_test_tail_over_time.png`",
            "- `figures/cifar_mixing_over_time.png`",
            "- `figures/cifar_test_mixing_over_time.png`",
            "- `figures/cifar_kernel_movement_over_time.png`",
            "- `figures/cifar_accuracy_over_time.png`",
        ]
    )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--n-per-class", type=int, default=120)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    tasks = [("cat", "dog"), ("automobile", "truck")]
    regimes = ["frozen_random", "feature_learning", "lazy_wide_small_lr"]
    results = []
    for task_idx, task in enumerate(tasks):
        train, test = make_cifar_train_test(args.data_root, task, args.n_per_class, seed=args.seed + task_idx)
        run = {"dataset": train.name, "regimes": []}
        for regime_idx, regime in enumerate(regimes):
            run["regimes"].append(
                train_regime(
                    train=train,
                    test=test,
                    regime=regime,
                    epochs=args.epochs,
                    seed=args.seed + 1000 * task_idx + regime_idx,
                    device=device,
                )
            )
        results.append(run)

    metrics_path = EXP_DIR / "metrics_cifar_over_time.json"
    metrics_path.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
    plot_metric(results, "tail_at_10pct", "tail@10% over training", EXP_DIR / "figures" / "cifar_tail_over_time.png")
    plot_metric(results, "test_tail_at_10pct", "test tail@10% over training", EXP_DIR / "figures" / "cifar_test_tail_over_time.png")
    plot_metric(results, "knn_opposite_ratio", "opposite-label kNN ratio over training", EXP_DIR / "figures" / "cifar_mixing_over_time.png")
    plot_metric(results, "test_knn_opposite_ratio", "test opposite-label kNN ratio over training", EXP_DIR / "figures" / "cifar_test_mixing_over_time.png")
    plot_metric(results, "kernel_movement", "feature Gram movement over training", EXP_DIR / "figures" / "cifar_kernel_movement_over_time.png")
    plot_metric(results, "test_acc", "test accuracy over training", EXP_DIR / "figures" / "cifar_accuracy_over_time.png")

    command = "python experiments/002-feature-metric-dynamics/scripts/run_cifar.py"
    command += f" --n-per-class {args.n_per_class} --epochs {args.epochs} --seed {args.seed} --device {args.device}"
    write_result_md(results, command, EXP_DIR / "result_cifar.md")
    print(f"Wrote {metrics_path}")
    print(f"Wrote {EXP_DIR / 'result_cifar.md'}")


if __name__ == "__main__":
    main()
