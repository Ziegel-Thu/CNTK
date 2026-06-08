#!/usr/bin/env python3
"""Toy feature-metric dynamics for experiment 002."""

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


def matching_test_dataset(train: datasets.Dataset, n: int, seed: int) -> datasets.Dataset:
    if train.name.startswith("two_moons"):
        noise = float(train.name.split("noise", maxsplit=1)[1])
        return datasets.make_two_moons(n=n, noise=noise, seed=seed)
    if train.name.startswith("collision_pairs"):
        sep = float(train.name.split("sep", maxsplit=1)[1])
        return datasets.make_collision_pairs(n_pairs=n // 2, separation=sep, seed=seed)
    raise ValueError(f"no matching test generator for {train.name}")


def feature_diagnostics(
    model: models.MLP,
    x_np: np.ndarray,
    y_sign: np.ndarray,
    k0: np.ndarray | None,
) -> dict:
    model.eval()
    with torch.no_grad():
        x = torch.tensor(x_np, dtype=torch.float32)
        feats = model.features(x).cpu().numpy()
        logits = model(x).cpu().numpy()

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


def train_regime(
    train: datasets.Dataset,
    test: datasets.Dataset,
    regime: str,
    epochs: int,
    seed: int,
) -> dict:
    models.set_seed(seed)
    x_train = torch.tensor(train.x, dtype=torch.float32)
    y_train01 = torch.tensor((train.y > 0).astype(np.float32))
    x_test = torch.tensor(test.x, dtype=torch.float32)
    y_test01 = torch.tensor((test.y > 0).astype(np.float32))

    if regime == "frozen_random":
        model = models.MLP(input_dim=train.x.shape[1], width=128, depth=2)
        model.freeze_features()
        lr = 1e-2
    elif regime == "feature_learning":
        model = models.MLP(input_dim=train.x.shape[1], width=128, depth=2)
        lr = 1e-2
        optimizer_kind = "adam"
    elif regime == "lazy_wide_small_lr":
        model = models.MLP(input_dim=train.x.shape[1], width=2048, depth=2)
        lr = 5e-3
        optimizer_kind = "sgd"
    else:
        raise ValueError(f"unknown regime: {regime}")

    if regime == "frozen_random":
        optimizer_kind = "adam"

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if optimizer_kind == "adam":
        optimizer = torch.optim.Adam(trainable_params, lr=lr)
    elif optimizer_kind == "sgd":
        optimizer = torch.optim.SGD(trainable_params, lr=lr)
    else:
        raise ValueError(f"unknown optimizer: {optimizer_kind}")
    loss_fn = nn.BCEWithLogitsLoss()
    checkpoints = sorted(set([0, 1, 5, 20, 60, epochs]))

    with torch.no_grad():
        k0 = kernels.feature_gram(model.features(x_train).cpu().numpy(), normalize=True)

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
        diag = feature_diagnostics(model, train.x, train.y, k0)
        trace.append(
            {
                "epoch": epoch,
                "loss": loss_value,
                "train_acc": train_acc,
                "test_acc": test_acc,
                **diag,
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

    return {"regime": regime, "trace": trace}


def plot_metric(results: list[dict], metric: str, ylabel: str, out: Path) -> None:
    plotting.setup()
    plt.figure(figsize=(7.2, 4.4))
    for run in results:
        for regime in run["regimes"]:
            epochs = [p["epoch"] for p in regime["trace"]]
            values = [p[metric] if p[metric] is not None else 0.0 for p in regime["trace"]]
            plt.plot(epochs, values, marker="o", label=f"{run['dataset']} / {regime['regime']}")
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.title(ylabel)
    plt.legend(fontsize=6)
    plotting.savefig(out)


def write_result_md(results: list[dict], command: str, out: Path) -> None:
    lines = [
        "# Result",
        "",
        "## Run",
        "",
        f"Command: `{command}`",
        "",
        "## Summary",
        "",
        "Toy feature-metric dynamics completed for frozen random features, feature",
        "learning, and a lazy-ish wide/small-LR control.",
        "",
        "Interpretation checklist:",
        "",
        "- Feature learning should reduce tail/mixing on correctable metric-mismatch tasks.",
        "- Frozen features should keep feature Gram movement at zero.",
        "- The lazy-ish control now uses a wide MLP with SGD and small learning rate,",
        "  so it should move the feature Gram much less than the feature-learning run.",
        "- Synthetic opposite-label collision pairs are an intrinsic-ambiguity control:",
        "  movement alone should not make them separable.",
        "",
        "| dataset | regime | tail@10% start | tail@10% final | mixing start | mixing final | movement final | train acc | test acc |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in results:
        for regime in run["regimes"]:
            first = regime["trace"][0]
            last = regime["trace"][-1]
            lines.append(
                "| {dataset} | {regime} | {tail0:.3f} | {tail1:.3f} | {mix0:.3f} | {mix1:.3f} | {move:.3f} | {tr:.3f} | {te:.3f} |".format(
                    dataset=run["dataset"],
                    regime=regime["regime"],
                    tail0=first["tail_at_10pct"],
                    tail1=last["tail_at_10pct"],
                    mix0=first["knn_opposite_ratio"],
                    mix1=last["knn_opposite_ratio"],
                    move=last["kernel_movement"] or 0.0,
                    tr=last["train_acc"],
                    te=last["test_acc"],
                )
            )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `metrics_over_time.json`",
            "- `figures/tail_over_time.png`",
            "- `figures/mixing_over_time.png`",
            "- `figures/kernel_movement_over_time.png`",
            "",
            "## Next",
            "",
            "- Compare these feature-Gram dynamics with empirical NTK dynamics on tiny subsets.",
            "- Run the same trace on MNIST binary subsets once data utilities are ready.",
        ]
    )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    n = min(args.n, 300) if args.quick else args.n
    epochs = min(args.epochs, 120) if args.quick else args.epochs
    train_sets = [
        datasets.make_two_moons(n=n, noise=0.2, seed=args.seed),
        datasets.make_collision_pairs(n_pairs=n // 2, separation=0.03, seed=args.seed + 10),
    ]
    regimes = ["frozen_random", "feature_learning", "lazy_wide_small_lr"]

    results = []
    for ds_idx, train in enumerate(train_sets):
        test = matching_test_dataset(train, n=n, seed=args.seed + 100 + ds_idx)
        run = {"dataset": train.name, "regimes": []}
        for regime_idx, regime in enumerate(regimes):
            run["regimes"].append(
                train_regime(
                    train=train,
                    test=test,
                    regime=regime,
                    epochs=epochs,
                    seed=args.seed + 1000 * ds_idx + regime_idx,
                )
            )
        results.append(run)

    metrics_path = EXP_DIR / "metrics_over_time.json"
    metrics_path.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
    plot_metric(results, "tail_at_10pct", "tail@10% over training", EXP_DIR / "figures" / "tail_over_time.png")
    plot_metric(results, "knn_opposite_ratio", "opposite-label kNN ratio over training", EXP_DIR / "figures" / "mixing_over_time.png")
    plot_metric(results, "kernel_movement", "feature Gram movement over training", EXP_DIR / "figures" / "kernel_movement_over_time.png")

    command = "python experiments/002-feature-metric-dynamics/scripts/run_toy.py"
    if args.quick:
        command += " --quick"
    command += f" --n {args.n} --epochs {args.epochs} --seed {args.seed}"
    write_result_md(results, command, EXP_DIR / "result.md")
    print(f"Wrote {metrics_path}")
    print(f"Wrote {EXP_DIR / 'result.md'}")


if __name__ == "__main__":
    main()
