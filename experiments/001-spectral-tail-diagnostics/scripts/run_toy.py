#!/usr/bin/env python3
"""Run toy fixed-kernel spectral-tail diagnostics for experiment 001."""

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

from src import datasets, gradient_flow, kernels, mixing, plotting, spectral


EXP_DIR = ROOT / "experiments" / "001-spectral-tail-diagnostics"


def make_kernel(name: str, x: np.ndarray, seed: int) -> np.ndarray:
    if name == "linear":
        return kernels.diagonal_normalize(kernels.linear_kernel(x))
    if name == "rbf_median":
        return kernels.rbf_kernel(x)
    if name == "rbf_narrow":
        return kernels.rbf_kernel(x, bandwidth=0.45 * kernels.median_bandwidth(x))
    if name == "laplace_median":
        return kernels.laplace_kernel(x)
    if name == "rff_512":
        feats = kernels.random_fourier_features(x, n_features=512, seed=seed)
        return kernels.feature_gram(feats)
    raise ValueError(f"unknown kernel: {name}")


def run_case(dataset: datasets.Dataset, kernel_name: str, seed: int) -> dict:
    k = make_kernel(kernel_name, dataset.x, seed)
    d2 = kernels.kernel_metric_squared(k)

    spec = spectral.summarize(k, dataset.y)
    mix = mixing.summarize(d2, dataset.y, k=10)
    eigvals, eigvecs = spectral.eigendecompose(k)
    times = np.logspace(-2, 4, 160)
    gf_curve = gradient_flow.residual_norm_curve(eigvals, eigvecs, dataset.y, times)
    t10 = gradient_flow.time_to_fraction(times, gf_curve, fraction=0.10)

    m_idx = max(0, int(np.ceil(0.10 * len(dataset.y))) - 1)
    lambda_m = float(eigvals[m_idx])
    bound10 = mixing.disjoint_collision_tail_bound(
        n=len(dataset.y),
        q_rho=mix.collision_count,
        rho=mix.rho,
        lambda_m=lambda_m,
    )

    return {
        "dataset": dataset.name,
        "kernel": kernel_name,
        "n": int(len(dataset.y)),
        "spectral": {
            "tail_auc": spec.tail_auc,
            "tail_at_10pct": spec.tail_at_10pct,
            "alignment": spec.alignment,
        },
        "mixing": mixing.to_jsonable(mix),
        "gradient_flow": {
            "time_to_10pct_residual": t10,
            "curve_times": times.tolist(),
            "curve": gf_curve.tolist(),
        },
        "bound_proxy": {
            "lambda_at_10pct": lambda_m,
            "disjoint_collision_tail_bound_at_10pct": bound10,
        },
        "tail_curve": spec.tail.tolist(),
        "energy_curve": spec.energy.tolist(),
    }


def plot_tail_curves(results: list[dict], out: Path) -> None:
    plotting.setup()
    plt.figure(figsize=(7.2, 4.4))
    selected = [
        r
        for r in results
        if r["kernel"] == "rbf_median"
        and (r["dataset"].startswith("two_moons") or r["dataset"].startswith("collision"))
    ]
    for r in selected:
        tail = np.asarray(r["tail_curve"])
        x = np.arange(1, len(tail) + 1) / len(tail)
        plt.plot(x, tail, label=r["dataset"])
    plt.xlabel("fraction of leading eigendirections")
    plt.ylabel("spectral label tail T_y(m)")
    plt.title("Toy spectral tail under RBF median kernel")
    plt.legend(fontsize=7)
    plotting.savefig(out)


def plot_mixing_vs_tail(results: list[dict], out: Path) -> None:
    plotting.setup()
    plt.figure(figsize=(6.2, 4.4))
    kernels_seen = sorted({r["kernel"] for r in results})
    for kernel_name in kernels_seen:
        rs = [r for r in results if r["kernel"] == kernel_name]
        x = [r["mixing"]["knn_opposite_ratio"] for r in rs]
        y = [r["spectral"]["tail_at_10pct"] for r in rs]
        plt.scatter(x, y, label=kernel_name, s=34, alpha=0.8)
    plt.xlabel("opposite-label ratio among 10 nearest neighbors")
    plt.ylabel("T_y(m) at top 10% eigendirections")
    plt.title("Local mixing vs spectral tail")
    plt.legend(fontsize=7)
    plotting.savefig(out)


def plot_gradient_flow(results: list[dict], out: Path) -> None:
    plotting.setup()
    plt.figure(figsize=(7.2, 4.4))
    selected_names = {
        ("two_moons_noise0.05", "rbf_median"),
        ("two_moons_noise0.2", "rbf_median"),
        ("collision_pairs_sep0.03", "rbf_median"),
        ("collision_pairs_sep0.25", "rbf_median"),
    }
    for r in results:
        if (r["dataset"], r["kernel"]) not in selected_names:
            continue
        times = np.asarray(r["gradient_flow"]["curve_times"])
        curve = np.asarray(r["gradient_flow"]["curve"])
        plt.semilogx(times, curve, label=r["dataset"])
    plt.xlabel("gradient-flow time")
    plt.ylabel("normalized residual norm squared")
    plt.title("Predicted static-kernel gradient-flow decay")
    plt.legend(fontsize=7)
    plotting.savefig(out)


def write_result_md(results: list[dict], command: str, out: Path) -> None:
    sorted_by_tail = sorted(results, key=lambda r: r["spectral"]["tail_at_10pct"], reverse=True)
    top = sorted_by_tail[:8]
    lines = [
        "# Result",
        "",
        "## Run",
        "",
        f"Command: `{command}`",
        "",
        "## Summary",
        "",
        "Toy quick run completed. It computes fixed-kernel local mixing, spectral label tail,",
        "kernel-target alignment, a disjoint-collision lower-bound proxy, and exact static",
        "kernel gradient-flow residual curves.",
        "",
        "Highest `T_y(m)` at top 10% eigendirections:",
        "",
        "| dataset | kernel | tail@10% | tail_auc | opposite-kNN ratio | alignment | t_res<=0.1 | bound proxy |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in top:
        t10 = r["gradient_flow"]["time_to_10pct_residual"]
        t10_s = "NA" if t10 is None else f"{t10:.3g}"
        lines.append(
            "| {dataset} | {kernel} | {tail:.3f} | {auc:.3f} | {knn:.3f} | {align:.3f} | {t10} | {bound:.3f} |".format(
                dataset=r["dataset"],
                kernel=r["kernel"],
                tail=r["spectral"]["tail_at_10pct"],
                auc=r["spectral"]["tail_auc"],
                knn=r["mixing"]["knn_opposite_ratio"],
                align=r["spectral"]["alignment"],
                t10=t10_s,
                bound=r["bound_proxy"]["disjoint_collision_tail_bound_at_10pct"],
            )
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `metrics.json`",
            "- `figures/tail_curves.png`",
            "- `figures/mixing_vs_tail.png`",
            "- `figures/gradient_flow_decay.png`",
            "",
            "## Next",
            "",
            "- Inspect whether the mixing-vs-tail scatter remains clean on MNIST/CIFAR binary subsets.",
            "- Add the theorem-bound audit table across multiple `m` values.",
            "- Use the same diagnostics in experiment 002 for feature metric dynamics.",
        ]
    )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Use a small deterministic toy suite.")
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    n = args.n if not args.quick else min(args.n, 300)
    suite = datasets.toy_suite(n=n, seed=args.seed)
    kernel_names = ["linear", "rbf_median", "rbf_narrow", "laplace_median", "rff_512"]

    results = []
    for ds in suite:
        for kernel_name in kernel_names:
            results.append(run_case(ds, kernel_name, seed=args.seed))

    metrics_path = EXP_DIR / "metrics.json"
    metrics_path.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")

    plot_tail_curves(results, EXP_DIR / "figures" / "tail_curves.png")
    plot_mixing_vs_tail(results, EXP_DIR / "figures" / "mixing_vs_tail.png")
    plot_gradient_flow(results, EXP_DIR / "figures" / "gradient_flow_decay.png")

    command = "python experiments/001-spectral-tail-diagnostics/scripts/run_toy.py"
    if args.quick:
        command += " --quick"
    command += f" --n {args.n} --seed {args.seed}"
    write_result_md(results, command, EXP_DIR / "result.md")

    print(f"Wrote {metrics_path}")
    print(f"Wrote {EXP_DIR / 'result.md'}")


if __name__ == "__main__":
    main()
