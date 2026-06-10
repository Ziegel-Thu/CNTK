#!/usr/bin/env python3
"""Controlled audit of local mixing versus global alignment."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import plotting


EXP_DIR = ROOT / "experiments" / "014-mixing-alignment-controlled-audit"


def finite(value) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def add_row(rows: list[dict], **kwargs) -> None:
    row = {key: finite(value) if key not in {"source", "dataset", "model", "group"} else value for key, value in kwargs.items()}
    if row.get("tail") is None or row.get("alignment") is None or row.get("mixing") is None:
        return
    rows.append(row)


def collect_001(rows: list[dict]) -> None:
    for filename in ["metrics.json", "metrics_image_subsets.json"]:
        payload = json.loads((ROOT / "experiments" / "001-spectral-tail-diagnostics" / filename).read_text())
        for item in payload["results"]:
            spec = item["spectral"]
            mix = item["mixing"]
            add_row(
                rows,
                source=f"001/{filename}",
                group="all_with_toy",
                dataset=item["dataset"],
                model=item["kernel"],
                tail=spec["tail_at_10pct"],
                mixing=mix["knn_opposite_ratio"],
                graph=None,
                alignment=spec["alignment"],
                accuracy=None,
                margin=None,
            )


def collect_003_005(rows: list[dict], exp: str, multiclass: bool) -> None:
    payload = json.loads((ROOT / "experiments" / exp / "metrics.json").read_text())
    for item in payload["results"]:
        test = item["test"]
        mixing_key = "knn_disagreement_ratio" if multiclass else "knn_opposite_ratio"
        add_row(
            rows,
            source=exp[:3],
            group="image_fixed_repr",
            dataset=item["dataset"],
            model=item["representation"],
            tail=test["tail_at_10pct"],
            mixing=test[mixing_key],
            graph=None,
            alignment=test["alignment"],
            accuracy=item.get("linear_probe_test_acc"),
            margin=None,
        )


def collect_kernel_ridge(rows: list[dict], exp: str) -> None:
    payload = json.loads((ROOT / "experiments" / exp / "metrics.json").read_text())
    for item in payload["results"]:
        kr = item.get("kernel_ridge", {})
        mixing_value = item.get("test_mixing", item.get("test_knn_opposite_ratio"))
        add_row(
            rows,
            source=exp[:3],
            group="image_fixed_repr",
            dataset=item["dataset"],
            model=item["representation"],
            tail=item["test_tail_at_10pct"],
            mixing=mixing_value,
            graph=item.get("test_graph_dirichlet"),
            alignment=item["test_alignment"],
            accuracy=kr.get("test_accuracy"),
            margin=kr.get("test_margin_median"),
        )


def collect_012(rows: list[dict]) -> None:
    payload = json.loads((ROOT / "experiments" / "012-source-norm-controlled-sweep" / "metrics.json").read_text())
    for item in payload["results"]:
        add_row(
            rows,
            source="012",
            group="all_with_toy",
            dataset=item["dataset"],
            model=item["kernel"],
            tail=item["tail_at_10pct"],
            mixing=item["knn_opposite_ratio"],
            graph=item["graph_dirichlet"],
            alignment=item["alignment"],
            accuracy=item.get("ridge_train_accuracy"),
            margin=item.get("ridge_margin_median"),
        )


def collect_rows() -> list[dict]:
    rows: list[dict] = []
    collect_001(rows)
    collect_003_005(rows, "003-fixed-representation-sweep", multiclass=False)
    collect_003_005(rows, "005-multiclass-obstruction-diagnostics", multiclass=True)
    collect_kernel_ridge(rows, "008-graph-energy-kernel-margin")
    collect_kernel_ridge(rows, "010-pretrained-fixed-representation-sweep")
    collect_kernel_ridge(rows, "011-self-supervised-fixed-representation-sweep")
    collect_012(rows)
    extra = []
    for row in rows:
        if row["group"] == "image_fixed_repr":
            both = dict(row)
            both["group"] = "all_with_toy"
            extra.append(both)
    rows.extend(extra)
    return rows


def arrays(rows: list[dict], keys: list[str]) -> tuple[np.ndarray, list[dict]]:
    kept = []
    values = []
    for row in rows:
        vals = [row.get(key) for key in keys]
        if all(value is not None and math.isfinite(float(value)) for value in vals):
            kept.append(row)
            values.append([float(value) for value in vals])
    if not values:
        return np.zeros((0, len(keys))), []
    return np.asarray(values, dtype=np.float64), kept


def corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def residualize(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    x_design = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(x_design, y, rcond=None)
    return y - x_design @ beta


def partial_corr(rows: list[dict], x_key: str, y_key: str, control_keys: list[str]) -> dict:
    mat, kept = arrays(rows, [x_key, y_key, *control_keys])
    if len(kept) < len(control_keys) + 3:
        return {"n": len(kept), "partial_corr": 0.0}
    x = mat[:, 0]
    y = mat[:, 1]
    controls = mat[:, 2:]
    return {"n": len(kept), "partial_corr": corr(residualize(x, controls), residualize(y, controls))}


def standardized_regression(rows: list[dict], y_key: str, x_keys: list[str]) -> dict:
    mat, kept = arrays(rows, [y_key, *x_keys])
    if len(kept) < len(x_keys) + 2:
        return {"n": len(kept), "r2": 0.0, "coefficients": {key: 0.0 for key in x_keys}}
    y = mat[:, 0]
    x = mat[:, 1:]
    y_std = (y - y.mean()) / (y.std() + 1e-12)
    x_std = (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1e-12)
    design = np.column_stack([np.ones(len(x_std)), x_std])
    beta, *_ = np.linalg.lstsq(design, y_std, rcond=None)
    pred = design @ beta
    ss_res = float(np.sum((y_std - pred) ** 2))
    ss_tot = float(np.sum((y_std - y_std.mean()) ** 2))
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)
    return {
        "n": len(kept),
        "r2": float(r2),
        "coefficients": {key: float(beta[idx + 1]) for idx, key in enumerate(x_keys)},
    }


def summarize_group(rows: list[dict], group: str) -> dict:
    group_rows = [row for row in rows if row["group"] == group]
    tail_mix, _ = arrays(group_rows, ["tail", "mixing"])
    tail_graph, _ = arrays(group_rows, ["tail", "graph"])
    tail_align, _ = arrays(group_rows, ["tail", "alignment"])
    acc_tail, _ = arrays(group_rows, ["accuracy", "tail"])
    margin_tail, _ = arrays(group_rows, ["margin", "tail"])
    return {
        "group": group,
        "n_rows": len(group_rows),
        "corr_tail_mixing": corr(tail_mix[:, 0], tail_mix[:, 1]) if len(tail_mix) else 0.0,
        "corr_tail_graph": corr(tail_graph[:, 0], tail_graph[:, 1]) if len(tail_graph) else 0.0,
        "corr_tail_alignment": corr(tail_align[:, 0], tail_align[:, 1]) if len(tail_align) else 0.0,
        "corr_accuracy_tail": corr(acc_tail[:, 0], acc_tail[:, 1]) if len(acc_tail) else 0.0,
        "corr_margin_tail": corr(margin_tail[:, 0], margin_tail[:, 1]) if len(margin_tail) else 0.0,
        "partial_tail_mixing_given_alignment": partial_corr(group_rows, "tail", "mixing", ["alignment"]),
        "partial_tail_graph_given_alignment": partial_corr(group_rows, "tail", "graph", ["alignment"]),
        "reg_tail_mixing_alignment": standardized_regression(group_rows, "tail", ["mixing", "alignment"]),
        "reg_tail_graph_alignment": standardized_regression(group_rows, "tail", ["graph", "alignment"]),
        "reg_accuracy_tail_mixing_alignment": standardized_regression(group_rows, "accuracy", ["tail", "mixing", "alignment"]),
        "reg_margin_tail_graph_alignment": standardized_regression(group_rows, "margin", ["tail", "graph", "alignment"]),
    }


def plot_scatter(rows: list[dict], group: str, x_key: str, y_key: str, color_key: str, out: Path, xlabel: str, ylabel: str) -> None:
    rs = [row for row in rows if row["group"] == group and row.get(x_key) is not None and row.get(y_key) is not None and row.get(color_key) is not None]
    plotting.setup()
    plt.figure(figsize=(6.8, 4.8))
    x = [row[x_key] for row in rs]
    y = [row[y_key] for row in rs]
    c = [row[color_key] for row in rs]
    plt.scatter(x, y, c=c, cmap="viridis", s=34, alpha=0.82)
    cb = plt.colorbar()
    cb.set_label(color_key)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(group)
    plotting.savefig(out)


def plot_results(rows: list[dict]) -> None:
    plot_scatter(
        rows,
        "image_fixed_repr",
        "mixing",
        "tail",
        "alignment",
        EXP_DIR / "figures" / "tail_mixing_alignment.png",
        "local mixing/disagreement",
        "tail@10%",
    )
    plot_scatter(
        rows,
        "image_fixed_repr",
        "graph",
        "tail",
        "alignment",
        EXP_DIR / "figures" / "tail_graph_alignment.png",
        "graph Dirichlet",
        "tail@10%",
    )
    plot_scatter(
        rows,
        "image_fixed_repr",
        "tail",
        "accuracy",
        "mixing",
        EXP_DIR / "figures" / "accuracy_tail_mixing.png",
        "tail@10%",
        "accuracy",
    )


def write_result(summaries: list[dict]) -> None:
    lines = [
        "# Result",
        "",
        "## Run",
        "",
        "Command: `python3 experiments/014-mixing-alignment-controlled-audit/scripts/run_audit.py`",
        "",
        "## Summary",
        "",
        "| group | rows | corr tail/mix | partial tail/mix | corr tail/graph | partial tail/graph | corr tail/align |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summaries:
        lines.append(
            "| {group} | {n} | {ctm:.3f} | {ptm:.3f} | {ctg:.3f} | {ptg:.3f} | {cta:.3f} |".format(
                group=item["group"],
                n=item["n_rows"],
                ctm=item["corr_tail_mixing"],
                ptm=item["partial_tail_mixing_given_alignment"]["partial_corr"],
                ctg=item["corr_tail_graph"],
                ptg=item["partial_tail_graph_given_alignment"]["partial_corr"],
                cta=item["corr_tail_alignment"],
            )
        )
    lines.extend(["", "## Standardized Regressions", ""])
    for item in summaries:
        lines.append(f"### {item['group']}")
        for key in [
            "reg_tail_mixing_alignment",
            "reg_tail_graph_alignment",
            "reg_accuracy_tail_mixing_alignment",
            "reg_margin_tail_graph_alignment",
        ]:
            reg = item[key]
            coeff = ", ".join(f"{name}={value:.3f}" for name, value in reg["coefficients"].items())
            lines.append(f"- `{key}`: n=`{reg['n']}`, R2=`{reg['r2']:.3f}`, {coeff}")
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            "- On image fixed-representation rows, local mixing and graph roughness",
            "  retain signal after controlling for global alignment.",
            "- The `all_with_toy` group is expected to be messier because XOR/global",
            "  mismatch creates high tail without local collisions.",
            "- Therefore local mixing should be described as one diagnosable source",
            "  of spectral tail, not as a replacement for alignment.",
            "",
            "## Artifacts",
            "",
            "- `metrics.json`",
            "- `figures/tail_mixing_alignment.png`",
            "- `figures/tail_graph_alignment.png`",
            "- `figures/accuracy_tail_mixing.png`",
        ]
    )
    (EXP_DIR / "result.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = collect_rows()
    summaries = [summarize_group(rows, "image_fixed_repr"), summarize_group(rows, "all_with_toy")]
    payload = {"rows": rows, "summaries": summaries}
    (EXP_DIR / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot_results(rows)
    write_result(summaries)
    print(f"Wrote {EXP_DIR / 'metrics.json'}")
    print(f"Wrote {EXP_DIR / 'result.md'}")


if __name__ == "__main__":
    main()
