#!/usr/bin/env python3
"""Plot margin-over-time diagnostics from existing experiment 002 metrics."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import plotting


EXP_DIR = ROOT / "experiments" / "002-feature-metric-dynamics"


SPECS = [
    ("toy", "metrics_over_time.json"),
    ("mnist", "metrics_mnist_over_time.json"),
    ("cifar_mlp", "metrics_cifar_over_time.json"),
    ("cifar_cnn", "metrics_cifar_cnn_over_time.json"),
]


def safe_corr(xs: list[float], ys: list[float]) -> float:
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def load_results(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["results"]


def plot_metric(results: list[dict], metric: str, title: str, out: Path) -> bool:
    plotting.setup()
    plt.figure(figsize=(7.6, 4.8))
    plotted = False
    for run in results:
        for regime in run["regimes"]:
            trace = regime["trace"]
            if not trace or metric not in trace[0]:
                continue
            epochs = [point["epoch"] for point in trace]
            values = [point[metric] for point in trace]
            plt.plot(epochs, values, marker="o", linewidth=1.4, markersize=3, label=f"{run['dataset']} / {regime['regime']}")
            plotted = True
    if not plotted:
        plt.close()
        return False
    plt.xlabel("epoch")
    plt.ylabel(metric)
    plt.title(title)
    plt.legend(fontsize=5.8, ncol=1)
    plotting.savefig(out)
    return True


def collect_final_rows(name: str, results: list[dict]) -> list[dict]:
    rows = []
    for run in results:
        for regime in run["regimes"]:
            first = regime["trace"][0]
            last = regime["trace"][-1]
            row = {
                "family": name,
                "dataset": run["dataset"],
                "regime": regime["regime"],
                "train_margin_start": first.get("margin_median"),
                "train_margin_final": last.get("margin_median"),
                "train_tail_final": last.get("tail_at_10pct"),
                "train_acc_final": last.get("train_acc"),
                "test_margin_start": first.get("test_margin_median"),
                "test_margin_final": last.get("test_margin_median"),
                "test_tail_final": last.get("test_tail_at_10pct"),
                "test_acc_final": last.get("test_acc"),
            }
            rows.append(row)
    return rows


def write_result(rows: list[dict], generated: list[str]) -> None:
    test_rows = [row for row in rows if row["test_margin_final"] is not None and row["test_tail_final"] is not None]
    train_rows = [row for row in rows if row["train_margin_final"] is not None and row["train_tail_final"] is not None]
    lines = [
        "# Margin-Curve Supplement",
        "",
        "## Run",
        "",
        "Command: `python3 experiments/002-feature-metric-dynamics/scripts/plot_margin_curves.py`",
        "",
        "## Summary",
        "",
        f"- Train rows with margin/tail: `{len(train_rows)}`",
        f"- Test rows with margin/tail: `{len(test_rows)}`",
        f"- corr(final train tail, final train margin) = `{safe_corr([r['train_tail_final'] for r in train_rows], [r['train_margin_final'] for r in train_rows]):.3f}`",
        f"- corr(final test tail, final test margin) = `{safe_corr([r['test_tail_final'] for r in test_rows], [r['test_margin_final'] for r in test_rows]):.3f}`",
        f"- corr(final test margin, final test acc) = `{safe_corr([r['test_margin_final'] for r in test_rows], [r['test_acc_final'] for r in test_rows]):.3f}`",
        "",
        "Interpretation:",
        "",
        "- Margin is now a first-class feature-dynamics curve alongside tail,",
        "  mixing, kernel movement, and accuracy.",
        "- Tail and mixing describe geometry; margin indicates whether the",
        "  geometry is becoming a usable classifier.",
        "- Negative train/test gaps should be read as memorization even when train",
        "  margin increases.",
        "",
        "| family | dataset | regime | train margin start | train margin final | test margin final | test tail final | test acc final |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {family} | {dataset} | {regime} | {tms:.3f} | {tmf:.3f} | {xmf} | {xtf} | {xaf} |".format(
                family=row["family"],
                dataset=row["dataset"],
                regime=row["regime"],
                tms=row["train_margin_start"] if row["train_margin_start"] is not None else 0.0,
                tmf=row["train_margin_final"] if row["train_margin_final"] is not None else 0.0,
                xmf="NA" if row["test_margin_final"] is None else f"{row['test_margin_final']:.3f}",
                xtf="NA" if row["test_tail_final"] is None else f"{row['test_tail_final']:.3f}",
                xaf="NA" if row["test_acc_final"] is None else f"{row['test_acc_final']:.3f}",
            )
        )
    lines.extend(["", "## Artifacts", ""])
    lines.extend([f"- `{path}`" for path in generated])
    (EXP_DIR / "result_margin_curves.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    all_rows = []
    generated = []
    for name, filename in SPECS:
        path = EXP_DIR / filename
        if not path.exists():
            continue
        results = load_results(path)
        train_out = EXP_DIR / "figures" / f"{name}_margin_over_time.png"
        if plot_metric(results, "margin_median", f"{name} train margin median over time", train_out):
            generated.append(str(train_out.relative_to(EXP_DIR)))
        test_out = EXP_DIR / "figures" / f"{name}_test_margin_over_time.png"
        if plot_metric(results, "test_margin_median", f"{name} test margin median over time", test_out):
            generated.append(str(test_out.relative_to(EXP_DIR)))
        all_rows.extend(collect_final_rows(name, results))
    write_result(all_rows, generated)
    print(f"Wrote {EXP_DIR / 'result_margin_curves.md'}")
    for path in generated:
        print(f"Wrote {EXP_DIR / path}")


if __name__ == "__main__":
    main()
