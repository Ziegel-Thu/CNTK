#!/usr/bin/env python3
"""Controlled same-kernel source-norm sweep for experiment 012."""

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

from src import datasets, kernel_ridge, kernels, mixing, plotting, spectral


EXP_DIR = ROOT / "experiments" / "012-source-norm-controlled-sweep"


def flatten(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], -1)


def normalized_linear_gram(x: np.ndarray) -> np.ndarray:
    z = np.asarray(x, dtype=np.float64)
    z = (z - z.mean(axis=0, keepdims=True)) / (z.std(axis=0, keepdims=True) + 1e-12)
    z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
    return z @ z.T


def kernel_suite(x: np.ndarray, seed: int) -> list[tuple[str, np.ndarray]]:
    return [
        ("linear", normalized_linear_gram(x)),
        ("rbf", kernels.rbf_kernel(x)),
        ("laplace", kernels.laplace_kernel(x)),
        ("rff_512", kernels.feature_gram(kernels.random_fourier_features(x, n_features=512, seed=seed), normalize=True)),
    ]


def build_tasks(root: Path, n: int, n_per_class: int, seed: int) -> list[datasets.Dataset]:
    tasks: list[datasets.Dataset] = []
    for idx, sep in enumerate([0.02, 0.04, 0.08, 0.16, 0.32, 0.64]):
        tasks.append(datasets.make_collision_pairs(n_pairs=n // 2, separation=sep, seed=seed + idx))
    for idx, noise in enumerate([0.02, 0.08, 0.15, 0.25, 0.35]):
        tasks.append(datasets.make_two_moons(n=n, noise=noise, seed=seed + 20 + idx))
    for idx, noise in enumerate([0.05, 0.15, 0.25, 0.4]):
        tasks.append(datasets.make_xor(n=n, noise=noise, seed=seed + 40 + idx))
    for idx, classes in enumerate([(3, 8), (4, 9)]):
        tasks.append(datasets.make_mnist_binary(root, classes=classes, n_per_class=n_per_class, split="train", seed=seed + 60 + idx))
    for idx, classes in enumerate([("cat", "dog"), ("automobile", "truck")]):
        tasks.append(datasets.make_cifar10_binary(root, classes=classes, n_per_class=n_per_class, split="train", seed=seed + 70 + idx))
    return tasks


def summarize(dataset: datasets.Dataset, kernel_name: str, gram: np.ndarray, ridge: float, norm_ridge: float) -> dict:
    y = dataset.y
    spec = spectral.summarize(gram, y)
    d2 = kernels.kernel_metric_squared(gram)
    mix = mixing.summarize(d2, y, k=10)
    clf = kernel_ridge.fit_binary_kernel_ridge(gram, y, gram, y, ridge=ridge, norm_ridge=norm_ridge)
    return {
        "dataset": dataset.name,
        "kernel": kernel_name,
        "n": int(len(y)),
        "tail_at_10pct": spec.tail_at_10pct,
        "tail_auc": spec.tail_auc,
        "alignment": spec.alignment,
        "knn_opposite_ratio": mix.knn_opposite_ratio,
        "graph_disagreement": mix.graph_disagreement,
        "graph_dirichlet": mix.graph_dirichlet,
        "source_norm": clf.source_norm,
        "ridge_train_accuracy": clf.train_accuracy,
        "ridge_margin_median": clf.train_margin_median,
        "ridge_rkhs_norm": clf.rkhs_norm,
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


def kernel_summaries(rows: list[dict]) -> list[dict]:
    out = []
    for kernel in sorted({row["kernel"] for row in rows}):
        rs = [row for row in rows if row["kernel"] == kernel]
        out.append(
            {
                "kernel": kernel,
                "n": len(rs),
                "corr_tail_source_norm": safe_corr([r["tail_at_10pct"] for r in rs], [r["source_norm"] for r in rs]),
                "corr_mixing_source_norm": safe_corr([r["knn_opposite_ratio"] for r in rs], [r["source_norm"] for r in rs]),
                "corr_graph_source_norm": safe_corr([r["graph_dirichlet"] for r in rs], [r["source_norm"] for r in rs]),
                "corr_source_norm_margin": safe_corr([r["source_norm"] for r in rs], [r["ridge_margin_median"] for r in rs]),
                "corr_tail_margin": safe_corr([r["tail_at_10pct"] for r in rs], [r["ridge_margin_median"] for r in rs]),
                "source_norm_mean": float(np.mean([r["source_norm"] for r in rs])),
            }
        )
    return out


def plot_scatter(rows: list[dict], x_key: str, y_key: str, xlabel: str, ylabel: str, out: Path) -> None:
    plotting.setup()
    plt.figure(figsize=(6.8, 4.8))
    for kernel in sorted({r["kernel"] for r in rows}):
        rs = [r for r in rows if r["kernel"] == kernel]
        plt.scatter([r[x_key] for r in rs], [r[y_key] for r in rs], s=36, alpha=0.82, label=kernel)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(fontsize=7)
    plotting.savefig(out)


def plot_results(rows: list[dict]) -> None:
    plot_scatter(
        rows,
        "tail_at_10pct",
        "source_norm",
        "tail@10%",
        "source norm proxy",
        EXP_DIR / "figures" / "tail_vs_source_norm.png",
    )
    plot_scatter(
        rows,
        "knn_opposite_ratio",
        "source_norm",
        "kNN opposite-label ratio",
        "source norm proxy",
        EXP_DIR / "figures" / "mixing_vs_source_norm.png",
    )
    plot_scatter(
        rows,
        "source_norm",
        "ridge_margin_median",
        "source norm proxy",
        "kernel ridge train margin median",
        EXP_DIR / "figures" / "source_norm_vs_margin.png",
    )


def write_result_md(rows: list[dict], summaries: list[dict], command: str) -> None:
    lines = [
        "# Result",
        "",
        "## Run",
        "",
        f"Command: `{command}`",
        "",
        "## Summary",
        "",
        "Correlations are reported within each kernel family to avoid mixing kernel",
        "scales.",
        "",
        "| kernel | rows | corr tail/source | corr mix/source | corr graph/source | corr source/margin | corr tail/margin | source mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summaries:
        lines.append(
            "| {kernel} | {n} | {cts:.3f} | {cms:.3f} | {cgs:.3f} | {csm:.3f} | {ctm:.3f} | {mean:.2f} |".format(
                kernel=item["kernel"],
                n=item["n"],
                cts=item["corr_tail_source_norm"],
                cms=item["corr_mixing_source_norm"],
                cgs=item["corr_graph_source_norm"],
                csm=item["corr_source_norm_margin"],
                ctm=item["corr_tail_margin"],
                mean=item["source_norm_mean"],
            )
        )
    tail = [r["tail_at_10pct"] for r in rows]
    source = [r["source_norm"] for r in rows]
    margin = [r["ridge_margin_median"] for r in rows]
    lines.extend(
        [
            "",
            "Global mixed-kernel correlations, for comparison only:",
            "",
            f"- corr(tail@10%, source norm) = `{safe_corr(tail, source):.3f}`",
            f"- corr(source norm, ridge margin) = `{safe_corr(source, margin):.3f}`",
            "",
            "Interpretation:",
            "",
            "- Source norm is meaningful when read within a fixed kernel family.",
            "- Local mixing/tail are still cleaner cross-family diagnostics, while",
            "  source norm should be reported with kernel/regularization context.",
            "",
            "| dataset | kernel | tail@10% | mix | graph dir | align | source norm | ridge margin | train acc |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(rows, key=lambda item: (item["kernel"], item["dataset"])):
        lines.append(
            "| {dataset} | {kernel} | {tail:.3f} | {mix:.3f} | {graph:.3f} | {align:.3f} | {source:.2f} | {margin:.3f} | {acc:.3f} |".format(
                dataset=row["dataset"],
                kernel=row["kernel"],
                tail=row["tail_at_10pct"],
                mix=row["knn_opposite_ratio"],
                graph=row["graph_dirichlet"],
                align=row["alignment"],
                source=row["source_norm"],
                margin=row["ridge_margin_median"],
                acc=row["ridge_train_accuracy"],
            )
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `metrics.json`",
            "- `figures/tail_vs_source_norm.png`",
            "- `figures/mixing_vs_source_norm.png`",
            "- `figures/source_norm_vs_margin.png`",
        ]
    )
    (EXP_DIR / "result.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--n", type=int, default=160)
    parser.add_argument("--n-per-class", type=int, default=80)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--norm-ridge", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    tasks = build_tasks(args.data_root, n=args.n, n_per_class=args.n_per_class, seed=args.seed)
    rows = []
    for task_idx, task in enumerate(tasks):
        x = flatten(task.x)
        print(f"Running {task.name}", flush=True)
        for kernel_name, gram in kernel_suite(x, seed=args.seed + task_idx):
            rows.append(summarize(task, kernel_name, gram, ridge=args.ridge, norm_ridge=args.norm_ridge))
    summaries = kernel_summaries(rows)
    payload = {
        "config": {
            "n": args.n,
            "n_per_class": args.n_per_class,
            "ridge": args.ridge,
            "norm_ridge": args.norm_ridge,
            "seed": args.seed,
        },
        "kernel_summaries": summaries,
        "results": rows,
    }
    (EXP_DIR / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot_results(rows)
    command = "python3 experiments/012-source-norm-controlled-sweep/scripts/run_source_norm.py"
    command += f" --n {args.n} --n-per-class {args.n_per_class}"
    command += f" --ridge {args.ridge} --norm-ridge {args.norm_ridge} --seed {args.seed}"
    write_result_md(rows, summaries, command)
    print(f"Wrote {EXP_DIR / 'metrics.json'}")
    print(f"Wrote {EXP_DIR / 'result.md'}")


if __name__ == "__main__":
    main()
