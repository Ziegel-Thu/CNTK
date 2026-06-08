#!/usr/bin/env python3
"""Tail-to-training-time consequence diagnostics for experiment 009."""

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

from src import datasets, gradient_flow, kernel_ridge, kernels, mixing, plotting, spectral


EXP_DIR = ROOT / "experiments" / "009-tail-training-time-consequence"


def flatten(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], -1)


def linear_gram(x: np.ndarray) -> np.ndarray:
    z = np.asarray(x, dtype=np.float64)
    z = (z - z.mean(axis=0, keepdims=True)) / (z.std(axis=0, keepdims=True) + 1e-12)
    z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
    return z @ z.T


def kernel_suite(x: np.ndarray, seed: int) -> list[tuple[str, np.ndarray]]:
    return [
        ("linear", linear_gram(x)),
        ("rbf", kernels.rbf_kernel(x)),
        ("laplace", kernels.laplace_kernel(x)),
        ("rff_512", kernels.feature_gram(kernels.random_fourier_features(x, n_features=512, seed=seed), normalize=True)),
    ]


def build_tasks(root: Path, n: int, n_per_class: int, seed: int) -> list[datasets.Dataset]:
    tasks: list[datasets.Dataset] = []
    tasks.extend(datasets.toy_suite(n=n, seed=seed))
    for idx, classes in enumerate([(3, 8), (4, 9)]):
        tasks.append(datasets.make_mnist_binary(root, classes=classes, n_per_class=n_per_class, split="train", seed=seed + 20 + idx))
    for idx, classes in enumerate([("cat", "dog"), ("automobile", "truck")]):
        tasks.append(datasets.make_cifar10_binary(root, classes=classes, n_per_class=n_per_class, split="train", seed=seed + 30 + idx))
    return tasks


def interpolated_time_to_fraction(times: np.ndarray, curve: np.ndarray, fraction: float) -> tuple[float, bool]:
    hit = np.flatnonzero(curve <= fraction)
    if hit.size == 0:
        return float(times[-1]), False
    idx = int(hit[0])
    if idx == 0:
        return float(times[0]), True
    # Interpolate in log-time/log-residual space for smoother estimates.
    t0, t1 = float(times[idx - 1]), float(times[idx])
    c0, c1 = max(float(curve[idx - 1]), 1e-300), max(float(curve[idx]), 1e-300)
    if c0 == c1:
        return t1, True
    log_t0, log_t1 = np.log(t0), np.log(t1)
    log_c0, log_c1 = np.log(c0), np.log(c1)
    alpha = (np.log(fraction) - log_c0) / (log_c1 - log_c0)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return float(np.exp(log_t0 + alpha * (log_t1 - log_t0))), True


def summarize(dataset: datasets.Dataset, kernel_name: str, gram: np.ndarray, times: np.ndarray, residual_fraction: float) -> dict:
    y = dataset.y
    eigvals, eigvecs = spectral.eigendecompose(gram)
    spec = spectral.summarize(gram, y)
    d2 = kernels.kernel_metric_squared(gram)
    mix = mixing.summarize(d2, y, k=10)
    curve = gradient_flow.residual_norm_curve(eigvals, eigvecs, y, times, normalize=True)
    t_frac, hit = interpolated_time_to_fraction(times, curve, residual_fraction)
    source_norm = kernel_ridge.source_norm_binary(gram, y, ridge=1e-6)
    lambda_10pct = float(eigvals[max(0, min(len(eigvals) - 1, int(np.ceil(0.10 * len(eigvals))) - 1))])
    return {
        "dataset": dataset.name,
        "kernel": kernel_name,
        "n": int(len(y)),
        "tail_at_10pct": spec.tail_at_10pct,
        "tail_auc": spec.tail_auc,
        "alignment": spec.alignment,
        "knn_opposite_ratio": mix.knn_opposite_ratio,
        "graph_dirichlet": mix.graph_dirichlet,
        "graph_disagreement": mix.graph_disagreement,
        "source_norm": source_norm,
        "lambda_10pct": lambda_10pct,
        "time_to_residual_fraction": t_frac,
        "hit_residual_fraction": bool(hit),
        "log10_time_to_residual_fraction": float(np.log10(t_frac)),
        "residual_final": float(curve[-1]),
        "residual_curve": curve.tolist(),
    }


def safe_corr(xs: list[float], ys: list[float]) -> float:
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def plot_scatter(rows: list[dict], x_key: str, y_key: str, xlabel: str, ylabel: str, out: Path) -> None:
    plotting.setup()
    plt.figure(figsize=(6.8, 4.8))
    for kernel in sorted({r["kernel"] for r in rows}):
        rs = [r for r in rows if r["kernel"] == kernel]
        plt.scatter([r[x_key] for r in rs], [r[y_key] for r in rs], s=38, alpha=0.82, label=kernel)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(fontsize=7)
    plotting.savefig(out)


def plot_residual_curves(rows: list[dict], times: np.ndarray) -> None:
    plotting.setup()
    plt.figure(figsize=(7.4, 4.9))
    selected = sorted(rows, key=lambda r: r["tail_at_10pct"])
    keep = selected[:4] + selected[-4:]
    for row in keep:
        label = f"{row['dataset']} / {row['kernel']}"
        plt.plot(times, row["residual_curve"], label=label, alpha=0.86)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("gradient-flow time")
    plt.ylabel("normalized residual")
    plt.legend(fontsize=5)
    plotting.savefig(EXP_DIR / "figures" / "residual_curves.png")


def plot_results(rows: list[dict], times: np.ndarray) -> None:
    plot_scatter(
        rows,
        x_key="tail_at_10pct",
        y_key="log10_time_to_residual_fraction",
        xlabel="tail@10%",
        ylabel="log10 time to residual <= 0.1",
        out=EXP_DIR / "figures" / "tail_vs_time.png",
    )
    plot_scatter(
        rows,
        x_key="tail_at_10pct",
        y_key="source_norm",
        xlabel="tail@10%",
        ylabel="source norm proxy",
        out=EXP_DIR / "figures" / "tail_vs_source_norm.png",
    )
    plot_scatter(
        rows,
        x_key="knn_opposite_ratio",
        y_key="log10_time_to_residual_fraction",
        xlabel="kNN opposite-label ratio",
        ylabel="log10 time to residual <= 0.1",
        out=EXP_DIR / "figures" / "mixing_vs_time.png",
    )
    plot_residual_curves(rows, times)


def write_result_md(rows: list[dict], command: str, residual_fraction: float) -> None:
    tail = [r["tail_at_10pct"] for r in rows]
    time = [r["log10_time_to_residual_fraction"] for r in rows]
    source = [r["source_norm"] for r in rows]
    mix = [r["knn_opposite_ratio"] for r in rows]
    graph = [r["graph_dirichlet"] for r in rows]
    align = [r["alignment"] for r in rows]
    hit_count = sum(1 for r in rows if r["hit_residual_fraction"])
    lines = [
        "# Result",
        "",
        "## Run",
        "",
        f"Command: `{command}`",
        "",
        "## Summary",
        "",
        f"- residual target = `{residual_fraction}`",
        f"- rows hitting residual target within grid = `{hit_count}/{len(rows)}`",
        f"- corr(tail@10%, log10 training time) = `{safe_corr(tail, time):.3f}`",
        f"- corr(kNN opposite ratio, log10 training time) = `{safe_corr(mix, time):.3f}`",
        f"- corr(graph Dirichlet, log10 training time) = `{safe_corr(graph, time):.3f}`",
        f"- corr(alignment, log10 training time) = `{safe_corr(align, time):.3f}`",
        f"- corr(tail@10%, source norm proxy) = `{safe_corr(tail, source):.3f}`",
        "",
        "Interpretation:",
        "",
        "- This directly tests the consequence link from spectral tail to static",
        "  kernel gradient-flow time.",
        "- Source norm is included as a proxy, but should be read by kernel family",
        "  because mixed kernel scales can compress or inflate it.",
        "",
        "| dataset | kernel | tail@10% | mix | graph dir | alignment | source norm | log10 time | hit | final residual |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for r in sorted(rows, key=lambda item: (item["dataset"], item["kernel"])):
        lines.append(
            "| {dataset} | {kernel} | {tail:.3f} | {mix:.3f} | {graph:.3f} | {align:.3f} | {source:.2f} | {time:.2f} | {hit} | {resid:.3g} |".format(
                dataset=r["dataset"],
                kernel=r["kernel"],
                tail=r["tail_at_10pct"],
                mix=r["knn_opposite_ratio"],
                graph=r["graph_dirichlet"],
                align=r["alignment"],
                source=r["source_norm"],
                time=r["log10_time_to_residual_fraction"],
                hit="yes" if r["hit_residual_fraction"] else "no",
                resid=r["residual_final"],
            )
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `metrics.json`",
            "- `figures/tail_vs_time.png`",
            "- `figures/tail_vs_source_norm.png`",
            "- `figures/mixing_vs_time.png`",
            "- `figures/residual_curves.png`",
        ]
    )
    (EXP_DIR / "result.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--n", type=int, default=160)
    parser.add_argument("--n-per-class", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--residual-fraction", type=float, default=0.1)
    parser.add_argument("--time-min", type=float, default=1e-2)
    parser.add_argument("--time-max", type=float, default=1e6)
    parser.add_argument("--time-steps", type=int, default=320)
    args = parser.parse_args()

    times = np.logspace(np.log10(args.time_min), np.log10(args.time_max), args.time_steps)
    tasks = build_tasks(args.data_root, n=args.n, n_per_class=args.n_per_class, seed=args.seed)
    rows = []
    for task_idx, task in enumerate(tasks):
        x = flatten(task.x)
        print(f"Running {task.name}", flush=True)
        for kernel_name, gram in kernel_suite(x, seed=args.seed + task_idx):
            rows.append(summarize(task, kernel_name, gram, times=times, residual_fraction=args.residual_fraction))

    payload = {
        "config": {
            "n": args.n,
            "n_per_class": args.n_per_class,
            "seed": args.seed,
            "residual_fraction": args.residual_fraction,
            "time_min": args.time_min,
            "time_max": args.time_max,
            "time_steps": args.time_steps,
        },
        "times": times.tolist(),
        "results": rows,
    }
    (EXP_DIR / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot_results(rows, times)
    command = "python3 experiments/009-tail-training-time-consequence/scripts/run_tail_time.py"
    command += f" --n {args.n} --n-per-class {args.n_per_class} --seed {args.seed}"
    command += f" --residual-fraction {args.residual_fraction}"
    command += f" --time-min {args.time_min} --time-max {args.time_max} --time-steps {args.time_steps}"
    write_result_md(rows, command, residual_fraction=args.residual_fraction)
    print(f"Wrote {EXP_DIR / 'metrics.json'}")
    print(f"Wrote {EXP_DIR / 'result.md'}")


if __name__ == "__main__":
    main()
