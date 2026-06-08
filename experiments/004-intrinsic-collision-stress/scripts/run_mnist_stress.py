#!/usr/bin/env python3
"""MNIST intrinsic-collision and label-noise stress test."""

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


EXP_DIR = ROOT / "experiments" / "004-intrinsic-collision-stress"


def perturb_dataset(base: datasets.Dataset, condition: str, seed: int) -> datasets.Dataset:
    rng = np.random.default_rng(seed)
    x = base.x.copy()
    y = base.y.copy()
    n = len(y)
    if condition == "clean":
        return datasets.Dataset(name=f"{base.name}_clean", x=x, y=y)
    if condition.startswith("flip"):
        frac = float(condition.replace("flip", ""))
        q = int(round(frac * n))
        idx = rng.choice(n, size=q, replace=False)
        y[idx] *= -1.0
        return datasets.Dataset(name=f"{base.name}_label_flip_{frac:g}", x=x, y=y)
    if condition.startswith("advflip"):
        frac = float(condition.replace("advflip", ""))
        q = int(round(frac * n))
        d2 = kernels.pairwise_sq_dists(x)
        np.fill_diagonal(d2, np.inf)
        nearest_opp = np.full(n, np.inf)
        for i in range(n):
            mask = y != y[i]
            nearest_opp[i] = np.min(d2[i, mask])
        idx = np.argsort(nearest_opp)[:q]
        y[idx] *= -1.0
        return datasets.Dataset(name=f"{base.name}_adversarial_label_flip_{frac:g}", x=x, y=y)
    if condition.startswith("duplicate"):
        frac = float(condition.replace("duplicate", ""))
        q = int(round(frac * n))
        idx = rng.choice(n, size=q, replace=False)
        x = np.concatenate([x, x[idx]], axis=0)
        y = np.concatenate([y, -y[idx]], axis=0)
        perm = rng.permutation(len(y))
        return datasets.Dataset(name=f"{base.name}_opposite_duplicates_{frac:g}", x=x[perm], y=y[perm])
    raise ValueError(f"unknown condition: {condition}")


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
        model = models.MLP(input_dim=input_dim, width=512, depth=2)
        model.freeze_features()
        return model, "adam", 1e-2
    if regime == "feature_learning":
        return models.MLP(input_dim=input_dim, width=256, depth=2), "adam", 1e-3
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

    model, _, lr = make_model(regime, input_dim=train.x.shape[1])
    model.to(device)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    checkpoints = sorted(set([0, 1, 5, 20, 60, 120, epochs]))
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
        trace.append(
            {
                "epoch": epoch,
                "loss": loss_value,
                "train_acc_noisy_labels": train_acc,
                "test_acc_clean_labels": test_acc,
                **train_diag,
                **{f"test_{key}": value for key, value in test_diag.items()},
            }
        )

    record(0)
    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x_train), y_train01)
        loss.backward()
        optimizer.step()
        if epoch in checkpoints:
            record(epoch, float(loss.item()))
            model.train()
    return {"regime": regime, "trace": trace}


def plot_metric(results: list[dict], metric: str, ylabel: str, out: Path) -> None:
    plotting.setup()
    plt.figure(figsize=(7.4, 4.6))
    for run in results:
        epochs = [p["epoch"] for p in run["trace"]]
        values = [p[metric] if p[metric] is not None else 0.0 for p in run["trace"]]
        plt.plot(epochs, values, marker="o", label=f"{run['condition']} / {run['regime']}")
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.title(ylabel)
    plt.legend(fontsize=6)
    plotting.savefig(out)


def write_result_md(results: list[dict], command: str, out: Path) -> None:
    lines = [
        "# MNIST Stress-Test Result",
        "",
        "## Run",
        "",
        f"Command: `{command}`",
        "",
        "## Summary",
        "",
        "Main observations to check after each run:",
        "",
        "- Clean data should reduce train and test tail under feature learning.",
        "- Heavy random or adversarial label noise can let train tail collapse while",
        "  clean test tail worsens, indicating memorization.",
        "- Exact opposite-label duplicates should cap train accuracy because a",
        "  deterministic classifier cannot satisfy both labels at identical inputs.",
        "",
        "| condition | regime | train tail start | train tail final | test tail final | train mix final | test mix final | movement | train acc noisy | test acc clean |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in results:
        first = run["trace"][0]
        last = run["trace"][-1]
        lines.append(
            "| {condition} | {regime} | {tail0:.3f} | {tail1:.3f} | {ttail:.3f} | {mix:.3f} | {tmix:.3f} | {move:.3f} | {tr:.3f} | {te:.3f} |".format(
                condition=run["condition"],
                regime=run["regime"],
                tail0=first["tail_at_10pct"],
                tail1=last["tail_at_10pct"],
                ttail=last["test_tail_at_10pct"],
                mix=last["knn_opposite_ratio"],
                tmix=last["test_knn_opposite_ratio"],
                move=last["kernel_movement"] or 0.0,
                tr=last["train_acc_noisy_labels"],
                te=last["test_acc_clean_labels"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Prompts",
            "",
            "- Clean labels should allow feature learning to lower train and test tail.",
            "- Label noise may lower train tail but should hurt clean test accuracy.",
            "- Exact opposite-label duplicates are intrinsic contradictions; a deterministic",
            "  network cannot classify both copies correctly.",
            "",
            "## Artifacts",
            "",
            "- `metrics_mnist_stress.json`",
            "- `figures/train_tail_over_time.png`",
            "- `figures/test_tail_over_time.png`",
            "- `figures/test_accuracy_over_time.png`",
        ]
    )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--n-per-class", type=int, default=150)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    clean_train, clean_test = datasets.make_mnist_binary_train_test(
        args.data_root,
        classes=(3, 8),
        n_per_class=args.n_per_class,
        seed=args.seed,
    )
    conditions = ["clean", "flip0.1", "flip0.3", "advflip0.1", "advflip0.3", "duplicate0.2"]
    regimes = ["frozen_random", "feature_learning"]
    results = []
    for condition_idx, condition in enumerate(conditions):
        train = perturb_dataset(clean_train, condition=condition, seed=args.seed + condition_idx)
        for regime_idx, regime in enumerate(regimes):
            run = train_regime(
                train=train,
                test=clean_test,
                regime=regime,
                epochs=args.epochs,
                seed=args.seed + 1000 * condition_idx + regime_idx,
                device=device,
            )
            run["condition"] = condition
            run["train_dataset"] = train.name
            results.append(run)

    metrics_path = EXP_DIR / "metrics_mnist_stress.json"
    metrics_path.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
    plot_metric(results, "tail_at_10pct", "train tail@10% over training", EXP_DIR / "figures" / "train_tail_over_time.png")
    plot_metric(results, "test_tail_at_10pct", "clean test tail@10% over training", EXP_DIR / "figures" / "test_tail_over_time.png")
    plot_metric(results, "test_acc_clean_labels", "clean test accuracy over training", EXP_DIR / "figures" / "test_accuracy_over_time.png")

    command = "python experiments/004-intrinsic-collision-stress/scripts/run_mnist_stress.py"
    command += f" --n-per-class {args.n_per_class} --epochs {args.epochs} --seed {args.seed} --device {args.device}"
    write_result_md(results, command, EXP_DIR / "result_mnist_stress.md")
    print(f"Wrote {metrics_path}")
    print(f"Wrote {EXP_DIR / 'result_mnist_stress.md'}")


if __name__ == "__main__":
    main()
