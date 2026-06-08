#!/usr/bin/env python3
"""Audit whether disjoint-collision spectral-tail bounds are numerically useful."""

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

from src import datasets, kernels, mixing, plotting, spectral


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
    if name == "rff_1024":
        feats = kernels.random_fourier_features(x, n_features=1024, seed=seed)
        return kernels.feature_gram(feats)
    raise ValueError(f"unknown kernel: {name}")


def build_datasets(root: Path, n: int, n_per_class: int, seed: int, include_cifar: bool) -> list[datasets.Dataset]:
    out = [
        datasets.make_two_moons(n=n, noise=0.2, seed=seed),
        datasets.make_collision_pairs(n_pairs=n // 2, separation=0.03, seed=seed + 1),
        datasets.make_mnist_binary(root, classes=(3, 8), n_per_class=n_per_class, seed=seed + 2),
        datasets.make_mnist_binary(root, classes=(4, 9), n_per_class=n_per_class, seed=seed + 3),
    ]
    if include_cifar:
        out.extend(
            [
                datasets.make_cifar10_binary(root, classes=("cat", "dog"), n_per_class=n_per_class, seed=seed + 4),
                datasets.make_cifar10_binary(root, classes=("automobile", "truck"), n_per_class=n_per_class, seed=seed + 5),
            ]
        )
    return out


def audit_case(
    dataset: datasets.Dataset,
    kernel_name: str,
    seed: int,
    fractions: list[float],
    rho_quantiles: list[float],
) -> dict:
    k = make_kernel(kernel_name, dataset.x, seed)
    d2 = kernels.kernel_metric_squared(k)
    eigvals, eigvecs = spectral.eigendecompose(k)
    _, tail = spectral.label_energy_curve(dataset.y, eigvecs)
    same, opp = mixing.nearest_distances_by_label(d2, dataset.y)
    finite_opp = opp[np.isfinite(opp)]
    n = len(dataset.y)

    rows = []
    for frac in fractions:
        m_idx = max(0, min(n - 1, int(np.ceil(frac * n)) - 1))
        actual_tail = float(tail[m_idx])
        lambda_m = float(eigvals[m_idx])
        best = {
            "bound": 0.0,
            "collision_rate": 0.0,
            "q_rho": 0,
            "rho": 0.0,
            "rho_quantile": None,
        }
        for rho_q in rho_quantiles:
            rho = float(np.quantile(finite_opp, rho_q)) if finite_opp.size else 0.0
            q_rho = len(mixing.greedy_disjoint_opposite_pairs(d2, dataset.y, rho=rho))
            bound = mixing.disjoint_collision_tail_bound(n=n, q_rho=q_rho, rho=rho, lambda_m=lambda_m)
            if bound > best["bound"]:
                best = {
                    "bound": bound,
                    "collision_rate": float(q_rho / n),
                    "q_rho": q_rho,
                    "rho": rho,
                    "rho_quantile": rho_q,
                }
        rows.append(
            {
                "fraction": frac,
                "m": int(m_idx + 1),
                "lambda_m": lambda_m,
                "actual_tail": actual_tail,
                "best_bound": best["bound"],
                "bound_ratio": float(best["bound"] / actual_tail) if actual_tail > 0 else 0.0,
                "best_collision_rate": best["collision_rate"],
                "best_q_rho": best["q_rho"],
                "best_rho": best["rho"],
                "best_rho_quantile": best["rho_quantile"],
            }
        )

    return {
        "dataset": dataset.name,
        "kernel": kernel_name,
        "n": n,
        "rows": rows,
    }


def plot_bound_vs_actual(results: list[dict], out: Path) -> None:
    plotting.setup()
    plt.figure(figsize=(6.4, 4.6))
    for frac in sorted({row["fraction"] for r in results for row in r["rows"]}):
        xs = []
        ys = []
        for r in results:
            row = next(item for item in r["rows"] if item["fraction"] == frac)
            xs.append(row["actual_tail"])
            ys.append(row["best_bound"])
        plt.scatter(xs, ys, label=f"m/n={frac:g}", s=32, alpha=0.8)
    lim = max([row["actual_tail"] for r in results for row in r["rows"]] + [1.0])
    plt.plot([0, lim], [0, lim], color="black", lw=1.0, alpha=0.45)
    plt.xlabel("actual spectral tail")
    plt.ylabel("best disjoint-collision lower bound")
    plt.title("Theorem-bound audit")
    plt.legend(fontsize=7)
    plotting.savefig(out)


def write_result_md(results: list[dict], command: str, out: Path) -> None:
    all_rows = [
        (r["dataset"], r["kernel"], row)
        for r in results
        for row in r["rows"]
    ]
    nonzero = [item for item in all_rows if item[2]["best_bound"] > 1e-8]
    ratios = np.asarray([item[2]["bound_ratio"] for item in all_rows], dtype=np.float64)
    collision_rates = np.asarray([item[2]["best_collision_rate"] for item in all_rows], dtype=np.float64)
    actual_tails = np.asarray([item[2]["actual_tail"] for item in all_rows], dtype=np.float64)
    corr_collision_tail = float(np.corrcoef(collision_rates, actual_tails)[0, 1])
    top = sorted(all_rows, key=lambda item: item[2]["best_bound"], reverse=True)[:12]

    lines = [
        "# Theorem-Bound Audit Result",
        "",
        "## Run",
        "",
        f"Command: `{command}`",
        "",
        "## Summary",
        "",
        f"- total audited rows = `{len(all_rows)}`",
        f"- nonzero formal-bound rows = `{len(nonzero)}`",
        f"- max bound/actual-tail ratio = `{float(np.max(ratios)):.3f}`",
        f"- mean bound/actual-tail ratio = `{float(np.mean(ratios)):.3f}`",
        f"- corr(best collision rate, actual tail) = `{corr_collision_tail:.3f}`",
        "",
        "Interpretation:",
        "",
        "- The formal corollary-style lower bound is useful only when kernel-edge",
        "  lengths are extremely small relative to `lambda_m/n`; otherwise it is",
        "  conservative or zero.",
        "- This particular greedy disjoint-collision-rate proxy is weaker than the",
        "  opposite-kNN/local-entropy diagnostics used in the main experiments.",
        "- Future experiments should report the formal bound as a sufficient",
        "  obstruction check, but use richer local-mixing diagnostics for prediction.",
        "",
        "Top formal bounds:",
        "",
        "| dataset | kernel | m/n | actual tail | bound | bound/actual | collision rate | rho |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset_name, kernel_name, row in top:
        lines.append(
            "| {dataset} | {kernel} | {frac:.2f} | {tail:.3f} | {bound:.3f} | {ratio:.3f} | {rate:.3f} | {rho:.3f} |".format(
                dataset=dataset_name,
                kernel=kernel_name,
                frac=row["fraction"],
                tail=row["actual_tail"],
                bound=row["best_bound"],
                ratio=row["bound_ratio"],
                rate=row["best_collision_rate"],
                rho=row["best_rho"],
            )
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `metrics_bound_audit.json`",
            "- `figures/bound_vs_actual.png`",
        ]
    )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--n-per-class", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-cifar", action="store_true")
    args = parser.parse_args()

    fractions = [0.05, 0.10, 0.20, 0.30, 0.50]
    rho_quantiles = [0.02, 0.05, 0.10, 0.20, 0.30]
    kernel_names = ["linear", "rbf_median", "rbf_narrow", "laplace_median", "rff_1024"]

    ds_list = build_datasets(args.data_root, args.n, args.n_per_class, args.seed, args.include_cifar)
    results = []
    for ds in ds_list:
        for kernel_name in kernel_names:
            results.append(audit_case(ds, kernel_name, args.seed, fractions, rho_quantiles))

    metrics_path = EXP_DIR / "metrics_bound_audit.json"
    metrics_path.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
    plot_bound_vs_actual(results, EXP_DIR / "figures" / "bound_vs_actual.png")

    command = "python experiments/001-spectral-tail-diagnostics/scripts/run_bound_audit.py"
    command += f" --n {args.n} --n-per-class {args.n_per_class} --seed {args.seed}"
    if args.include_cifar:
        command += " --include-cifar"
    write_result_md(results, command, EXP_DIR / "result_bound_audit.md")
    print(f"Wrote {metrics_path}")
    print(f"Wrote {EXP_DIR / 'result_bound_audit.md'}")


if __name__ == "__main__":
    main()
