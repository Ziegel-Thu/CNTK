#!/usr/bin/env python3
"""Audit margin/tail/accuracy relationships from existing experiment metrics."""

from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import plotting


EXP_DIR = ROOT / "experiments" / "007-margin-tail-audit"


BINARY_SOURCES = [
    ("002_mnist", ROOT / "experiments/002-feature-metric-dynamics/metrics_mnist_over_time.json"),
    ("002_cifar_mlp", ROOT / "experiments/002-feature-metric-dynamics/metrics_cifar_over_time.json"),
    ("002_cifar_cnn", ROOT / "experiments/002-feature-metric-dynamics/metrics_cifar_cnn_over_time.json"),
]

STRESS_SOURCE = ("004_stress", ROOT / "experiments/004-intrinsic-collision-stress/metrics_mnist_stress.json")
MULTICLASS_SOURCE = ("006_cifar_schedule", ROOT / "experiments/006-cifar-multiclass-schedule-sweep/metrics.json")


def finite(value: float | int | None) -> bool:
    return value is not None and math.isfinite(float(value))


def safe_corr(xs: list[float], ys: list[float]) -> float:
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def add_binary_rows(source: str, path: Path, final_rows: list[dict], delta_rows: list[dict]) -> None:
    data = json.loads(path.read_text())
    for run in data["results"]:
        dataset = run.get("dataset", run.get("train_dataset", "unknown"))
        for regime in run["regimes"]:
            trace = regime["trace"]
            first = trace[0]
            last = trace[-1]
            test_acc = last.get("test_acc")
            if test_acc is None:
                test_acc = last.get("test_acc_clean_labels")
            first_acc = first.get("test_acc")
            if first_acc is None:
                first_acc = first.get("test_acc_clean_labels")
            values = {
                "tail": last.get("test_tail_at_10pct"),
                "mixing": last.get("test_knn_opposite_ratio"),
                "margin": last.get("test_margin_median"),
                "accuracy": test_acc,
            }
            if all(finite(value) for value in values.values()):
                final_rows.append(
                    {
                        "source": source,
                        "dataset": dataset,
                        "regime": regime["regime"],
                        "row_type": "binary_final",
                        **{key: float(value) for key, value in values.items()},
                    }
                )
            if all(
                finite(value)
                for value in [
                    first.get("test_tail_at_10pct"),
                    last.get("test_tail_at_10pct"),
                    first.get("test_margin_median"),
                    last.get("test_margin_median"),
                    first_acc,
                    test_acc,
                ]
            ):
                delta_rows.append(
                    {
                        "source": source,
                        "dataset": dataset,
                        "regime": regime["regime"],
                        "tail_decrease": float(first["test_tail_at_10pct"] - last["test_tail_at_10pct"]),
                        "margin_gain": float(last["test_margin_median"] - first["test_margin_median"]),
                        "accuracy_gain": float(test_acc - first_acc),
                    }
                )


def add_stress_rows(source: str, path: Path, final_rows: list[dict], delta_rows: list[dict]) -> None:
    data = json.loads(path.read_text())
    for run in data["results"]:
        trace = run["trace"]
        first = trace[0]
        last = trace[-1]
        dataset = run.get("condition", "stress")
        values = {
            "tail": last.get("test_tail_at_10pct"),
            "mixing": last.get("test_knn_opposite_ratio"),
            "margin": last.get("test_margin_median"),
            "accuracy": last.get("test_acc_clean_labels"),
        }
        if all(finite(value) for value in values.values()):
            final_rows.append(
                {
                    "source": source,
                    "dataset": dataset,
                    "regime": run["regime"],
                    "row_type": "binary_stress_final",
                    **{key: float(value) for key, value in values.items()},
                }
            )
        if all(
            finite(value)
            for value in [
                first.get("test_tail_at_10pct"),
                last.get("test_tail_at_10pct"),
                first.get("test_margin_median"),
                last.get("test_margin_median"),
                first.get("test_acc_clean_labels"),
                last.get("test_acc_clean_labels"),
            ]
        ):
            delta_rows.append(
                {
                    "source": source,
                    "dataset": dataset,
                    "regime": run["regime"],
                    "tail_decrease": float(first["test_tail_at_10pct"] - last["test_tail_at_10pct"]),
                    "margin_gain": float(last["test_margin_median"] - first["test_margin_median"]),
                    "accuracy_gain": float(last["test_acc_clean_labels"] - first["test_acc_clean_labels"]),
                }
            )


def add_multiclass_rows(source: str, path: Path, final_rows: list[dict]) -> None:
    data = json.loads(path.read_text())
    for run in data["results"]:
        values = {
            "tail": run["test"].get("tail_at_10pct"),
            "mixing": run["test"].get("knn_disagreement_ratio"),
            "margin": run.get("probe_test_margin_median"),
            "accuracy": run.get("probe_test_acc"),
        }
        if all(finite(value) for value in values.values()):
            final_rows.append(
                {
                    "source": source,
                    "dataset": run["dataset"],
                    "regime": run["regime"],
                    "seed": run["seed"],
                    "row_type": "multiclass_final",
                    **{key: float(value) for key, value in values.items()},
                }
            )


def source_summaries(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["source"]].append(row)
    out = []
    for source, rs in sorted(grouped.items()):
        out.append(
            {
                "source": source,
                "n": len(rs),
                "tail_mean": float(np.mean([r["tail"] for r in rs])),
                "margin_mean": float(np.mean([r["margin"] for r in rs])),
                "accuracy_mean": float(np.mean([r["accuracy"] for r in rs])),
                "corr_tail_accuracy": safe_corr([r["tail"] for r in rs], [r["accuracy"] for r in rs]),
                "corr_margin_accuracy": safe_corr([r["margin"] for r in rs], [r["accuracy"] for r in rs]),
            }
        )
    return out


def plot_scatter(rows: list[dict], x_key: str, y_key: str, xlabel: str, ylabel: str, out: Path) -> None:
    plotting.setup()
    plt.figure(figsize=(6.8, 4.8))
    for source in sorted({row["source"] for row in rows}):
        rs = [row for row in rows if row["source"] == source]
        plt.scatter([r[x_key] for r in rs], [r[y_key] for r in rs], s=38, alpha=0.82, label=source)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(fontsize=7)
    plotting.savefig(out)


def plot_deltas(rows: list[dict], out: Path) -> None:
    plotting.setup()
    plt.figure(figsize=(6.8, 4.8))
    for source in sorted({row["source"] for row in rows}):
        rs = [row for row in rows if row["source"] == source]
        plt.scatter(
            [r["tail_decrease"] for r in rs],
            [r["margin_gain"] for r in rs],
            s=38,
            alpha=0.82,
            label=source,
        )
    plt.axhline(0.0, color="black", lw=0.8, alpha=0.35)
    plt.axvline(0.0, color="black", lw=0.8, alpha=0.35)
    plt.xlabel("test tail decrease")
    plt.ylabel("test margin gain")
    plt.legend(fontsize=7)
    plotting.savefig(out)


def write_result_md(final_rows: list[dict], delta_rows: list[dict], summaries: list[dict], command: str) -> None:
    corr_tail_acc = safe_corr([r["tail"] for r in final_rows], [r["accuracy"] for r in final_rows])
    corr_margin_acc = safe_corr([r["margin"] for r in final_rows], [r["accuracy"] for r in final_rows])
    corr_tail_margin = safe_corr([r["tail"] for r in final_rows], [r["margin"] for r in final_rows])
    corr_mix_tail = safe_corr([r["mixing"] for r in final_rows], [r["tail"] for r in final_rows])
    corr_taildec_margingain = safe_corr(
        [r["tail_decrease"] for r in delta_rows],
        [r["margin_gain"] for r in delta_rows],
    )
    corr_taildec_accgain = safe_corr(
        [r["tail_decrease"] for r in delta_rows],
        [r["accuracy_gain"] for r in delta_rows],
    )

    lines = [
        "# Result",
        "",
        "## Run",
        "",
        f"Command: `{command}`",
        "",
        "## Summary",
        "",
        f"- final rows: `{len(final_rows)}`",
        f"- dynamics delta rows: `{len(delta_rows)}`",
        f"- corr(final test tail, final accuracy) = `{corr_tail_acc:.3f}`",
        f"- corr(final margin median, final accuracy) = `{corr_margin_acc:.3f}`",
        f"- corr(final test tail, final margin median) = `{corr_tail_margin:.3f}`",
        f"- corr(final local mixing, final test tail) = `{corr_mix_tail:.3f}`",
        f"- corr(test tail decrease, margin gain) = `{corr_taildec_margingain:.3f}`",
        f"- corr(test tail decrease, accuracy gain) = `{corr_taildec_accgain:.3f}`",
        "",
        "Interpretation:",
        "",
        "- Margin is not just a duplicate of tail: final tail has the stronger",
        "  overall accuracy correlation, while margin adds complementary dynamic",
        "  and within-CIFAR signal.",
        "- Tail/mixing remain geometric obstruction diagnostics; margin is the",
        "  performance-facing companion that catches optimization and confidence.",
        "- In later dynamics runs, report tail, mixing, accuracy, and margin together.",
        "",
        "| source | rows | tail mean | margin mean | accuracy mean | corr tail/acc | corr margin/acc |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summaries:
        lines.append(
            "| {source} | {n} | {tail:.3f} | {margin:.3f} | {acc:.3f} | {cta:.3f} | {cma:.3f} |".format(
                source=item["source"],
                n=item["n"],
                tail=item["tail_mean"],
                margin=item["margin_mean"],
                acc=item["accuracy_mean"],
                cta=item["corr_tail_accuracy"],
                cma=item["corr_margin_accuracy"],
            )
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `metrics.json`",
            "- `figures/tail_vs_accuracy.png`",
            "- `figures/margin_vs_accuracy.png`",
            "- `figures/tail_decrease_vs_margin_gain.png`",
        ]
    )
    (EXP_DIR / "result.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    final_rows: list[dict] = []
    delta_rows: list[dict] = []
    for source, path in BINARY_SOURCES:
        add_binary_rows(source, path, final_rows, delta_rows)
    add_stress_rows(*STRESS_SOURCE, final_rows=final_rows, delta_rows=delta_rows)
    add_multiclass_rows(*MULTICLASS_SOURCE, final_rows=final_rows)

    summaries = source_summaries(final_rows)
    payload = {
        "final_rows": final_rows,
        "delta_rows": delta_rows,
        "source_summaries": summaries,
    }
    (EXP_DIR / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot_scatter(
        final_rows,
        x_key="tail",
        y_key="accuracy",
        xlabel="final test tail@10%",
        ylabel="final test/probe accuracy",
        out=EXP_DIR / "figures" / "tail_vs_accuracy.png",
    )
    plot_scatter(
        final_rows,
        x_key="margin",
        y_key="accuracy",
        xlabel="final test/probe margin median",
        ylabel="final test/probe accuracy",
        out=EXP_DIR / "figures" / "margin_vs_accuracy.png",
    )
    plot_deltas(delta_rows, out=EXP_DIR / "figures" / "tail_decrease_vs_margin_gain.png")
    write_result_md(
        final_rows,
        delta_rows,
        summaries,
        command="python3 experiments/007-margin-tail-audit/scripts/run_audit.py",
    )
    print(f"Wrote {EXP_DIR / 'metrics.json'}")
    print(f"Wrote {EXP_DIR / 'result.md'}")


if __name__ == "__main__":
    main()
