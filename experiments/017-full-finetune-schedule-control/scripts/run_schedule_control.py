#!/usr/bin/env python3
"""Schedule and augmentation controls for full ResNet18 fine-tuning."""

from __future__ import annotations

import argparse
import copy
import importlib.util
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


EXP_DIR = ROOT / "experiments" / "017-full-finetune-schedule-control"
VARIANTS = [
    {"name": "layer4_base", "base_regime": "finetune_layer4", "augment": False, "low_full_lr": False},
    {"name": "all_base", "base_regime": "finetune_all", "augment": False, "low_full_lr": False},
    {"name": "all_aug", "base_regime": "finetune_all", "augment": True, "low_full_lr": False},
    {"name": "all_low_lr", "base_regime": "finetune_all", "augment": False, "low_full_lr": True},
    {"name": "all_aug_low_lr", "base_regime": "finetune_all", "augment": True, "low_full_lr": True},
]
VARIANT_ORDER = [item["name"] for item in VARIANTS]


def load_exp013():
    path = ROOT / "experiments" / "013-pretrained-finetune-metric-dynamics" / "scripts" / "run_finetune_dynamics.py"
    spec = importlib.util.spec_from_file_location("exp013_finetune", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def args_for_variant(args: argparse.Namespace, spec: dict) -> argparse.Namespace:
    out = copy.copy(args)
    out.augment = bool(spec["augment"])
    if spec["low_full_lr"]:
        out.full_backbone_lr = args.low_full_backbone_lr
    return out


def final_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[int, str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["seed"], row["dataset"], row["variant"]), []).append(row)
    finals = []
    for (seed, dataset, variant), group_rows in grouped.items():
        group_rows = sorted(group_rows, key=lambda item: item["epoch"])
        first = group_rows[0]
        last = group_rows[-1]
        out = {
            "seed": seed,
            "dataset": dataset,
            "variant": variant,
            "base_regime": last["base_regime"],
            "augment": bool(last["augment"]),
            "backbone_lr_effective": last["backbone_lr_effective"],
            "full_backbone_lr_effective": last["full_backbone_lr_effective"],
            "kernel_movement_test": last["kernel_movement_test"],
            "test_tail_start": first["test"]["tail_at_10pct"],
            "test_tail_final": last["test"]["tail_at_10pct"],
            "test_tail_delta": last["test"]["tail_at_10pct"] - first["test"]["tail_at_10pct"],
            "test_graph_delta": last["test"]["graph_dirichlet"] - first["test"]["graph_dirichlet"],
            "test_mixing_delta": last["test"]["mixing"] - first["test"]["mixing"],
            "head_test_accuracy_start": first["head_test_accuracy"],
            "head_test_accuracy_final": last["head_test_accuracy"],
            "head_test_accuracy_delta": last["head_test_accuracy"] - first["head_test_accuracy"],
            "ridge_test_accuracy_start": first["kernel_ridge"]["test_accuracy"],
            "ridge_test_accuracy_final": last["kernel_ridge"]["test_accuracy"],
            "ridge_test_accuracy_delta": last["kernel_ridge"]["test_accuracy"] - first["kernel_ridge"]["test_accuracy"],
            "ridge_test_margin_final": last["kernel_ridge"]["test_margin_median"],
        }
        out["metric_repair"] = bool(out["test_tail_delta"] < 0.0 and out["test_graph_delta"] < 0.0)
        out["overmove"] = bool(out["kernel_movement_test"] > 0.1 and (out["test_tail_delta"] > 0.0 or out["test_graph_delta"] > 0.0))
        finals.append(out)
    return finals


def mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return 0.0, 0.0
    return float(np.mean(arr)), float(np.std(arr, ddof=0))


def summarize(finals: list[dict]) -> list[dict]:
    summaries = []
    for variant in VARIANT_ORDER:
        rows = [row for row in finals if row["variant"] == variant]
        item = {"variant": variant, "n": len(rows)}
        for key in [
            "kernel_movement_test",
            "test_tail_delta",
            "test_graph_delta",
            "test_mixing_delta",
            "head_test_accuracy_delta",
            "ridge_test_accuracy_delta",
            "ridge_test_margin_final",
        ]:
            avg, std = mean_std([float(row[key]) for row in rows])
            item[f"{key}_mean"] = avg
            item[f"{key}_std"] = std
        item["metric_repair_rate"] = float(np.mean([row["metric_repair"] for row in rows])) if rows else 0.0
        item["overmove_rate"] = float(np.mean([row["overmove"] for row in rows])) if rows else 0.0
        summaries.append(item)
    return summaries


def plot_bar(finals: list[dict], key: str, ylabel: str, out: Path) -> None:
    plotting.setup()
    values = [[row[key] for row in finals if row["variant"] == variant] for variant in VARIANT_ORDER]
    means = [float(np.mean(v)) if v else 0.0 for v in values]
    stds = [float(np.std(v)) if v else 0.0 for v in values]
    x = np.arange(len(VARIANT_ORDER))
    plt.figure(figsize=(8.2, 4.6))
    plt.bar(x, means, yerr=stds, capsize=4)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xticks(x, VARIANT_ORDER, rotation=22, ha="right")
    plt.ylabel(ylabel)
    plotting.savefig(out)


def plot_movement_vs_tail(finals: list[dict]) -> None:
    plotting.setup()
    plt.figure(figsize=(7.2, 4.8))
    for variant in VARIANT_ORDER:
        rows = [row for row in finals if row["variant"] == variant]
        plt.scatter(
            [row["kernel_movement_test"] for row in rows],
            [row["test_tail_delta"] for row in rows],
            s=42,
            alpha=0.82,
            label=variant,
        )
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xlabel("test kernel movement")
    plt.ylabel("test tail delta")
    plt.legend(fontsize=7)
    plotting.savefig(EXP_DIR / "figures" / "movement_vs_tail_delta.png")


def plot_results(finals: list[dict]) -> None:
    plot_bar(finals, "test_tail_delta", "test tail@10% delta", EXP_DIR / "figures" / "tail_delta_by_variant.png")
    plot_bar(finals, "test_graph_delta", "test graph Dirichlet delta", EXP_DIR / "figures" / "graph_delta_by_variant.png")
    plot_movement_vs_tail(finals)


def write_result(command: str, summaries: list[dict], finals: list[dict]) -> None:
    lines = [
        "# Result",
        "",
        "## Run",
        "",
        "Command:",
        "",
        "```bash",
        command,
        "```",
        "",
        "## Summary",
        "",
        "| variant | n | movement | tail delta | graph delta | repair rate | overmove rate | head acc delta | ridge acc delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summaries:
        lines.append(
            "| {variant} | {n} | {move:.3f} +/- {move_s:.3f} | {tail:.3f} +/- {tail_s:.3f} | {graph:.3f} +/- {graph_s:.3f} | {repair:.2f} | {over:.2f} | {head:.3f} | {ridge:.3f} |".format(
                variant=item["variant"],
                n=item["n"],
                move=item["kernel_movement_test_mean"],
                move_s=item["kernel_movement_test_std"],
                tail=item["test_tail_delta_mean"],
                tail_s=item["test_tail_delta_std"],
                graph=item["test_graph_delta_mean"],
                graph_s=item["test_graph_delta_std"],
                repair=item["metric_repair_rate"],
                over=item["overmove_rate"],
                head=item["head_test_accuracy_delta_mean"],
                ridge=item["ridge_test_accuracy_delta_mean"],
            )
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- `layer4_base` is the positive control inherited from `016`.",
            "- `all_base` is the default full fine-tuning over-move control.",
            "- A full fine-tuning variant counts as schedule rescue only if it lowers",
            "  both held-out tail and graph roughness while reducing overmove rate.",
            "- If augmentation or lower LR mainly improves head accuracy without",
            "  lowering held-out tail/graph roughness, it should not be read as metric",
            "  repair.",
            "",
            "## Per-Seed Final Rows",
            "",
            "| seed | dataset | variant | movement | tail delta | graph delta | repair | overmove | head acc final | ridge acc final |",
            "| ---: | --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: |",
        ]
    )
    for row in sorted(finals, key=lambda item: (item["seed"], item["dataset"], VARIANT_ORDER.index(item["variant"]))):
        lines.append(
            "| {seed} | {dataset} | {variant} | {move:.3f} | {tail:.3f} | {graph:.3f} | {repair} | {over} | {head:.3f} | {ridge:.3f} |".format(
                seed=row["seed"],
                dataset=row["dataset"],
                variant=row["variant"],
                move=row["kernel_movement_test"],
                tail=row["test_tail_delta"],
                graph=row["test_graph_delta"],
                repair="yes" if row["metric_repair"] else "no",
                over="yes" if row["overmove"] else "no",
                head=row["head_test_accuracy_final"],
                ridge=row["ridge_test_accuracy_final"],
            )
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `metrics.json`",
            "- `figures/tail_delta_by_variant.png`",
            "- `figures/graph_delta_by_variant.png`",
            "- `figures/movement_vs_tail_delta.png`",
        ]
    )
    (EXP_DIR / "result.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--binary-n-per-class", type=int, default=120)
    parser.add_argument("--multi-n-per-class", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=3e-5)
    parser.add_argument("--full-backbone-lr", type=float, default=1e-5)
    parser.add_argument("--low-full-backbone-lr", type=float, default=3e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--k-neighbors", type=int, default=10)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.seeds = [0]
        args.binary_n_per_class = 8
        args.multi_n_per_class = 5
        args.epochs = 1
        args.batch_size = 8
        args.eval_batch_size = 16

    exp013 = load_exp013()
    device = exp013.choose_device(args.device)
    rows = []
    for seed in args.seeds:
        tasks = exp013.build_tasks(args.data_root, args.binary_n_per_class, args.multi_n_per_class, seed)
        for task_idx, task in enumerate(tasks):
            for variant_idx, spec in enumerate(VARIANTS):
                var_args = args_for_variant(args, spec)
                train_seed = seed + 1000 * task_idx + 100 * variant_idx
                print(f"Running seed={seed} {task['name']} / {spec['name']} on {device}", flush=True)
                trace = exp013.train_regime(task, spec["base_regime"], var_args, device, seed=train_seed)
                for row in trace:
                    row = dict(row)
                    row["seed"] = seed
                    row["variant"] = spec["name"]
                    row["base_regime"] = spec["base_regime"]
                    row["regime"] = spec["name"]
                    row["augment"] = bool(var_args.augment)
                    row["backbone_lr_effective"] = float(var_args.backbone_lr)
                    row["full_backbone_lr_effective"] = float(var_args.full_backbone_lr)
                    rows.append(row)

    finals = final_rows(rows)
    summaries = summarize(finals)
    payload = {
        "config": {
            "seeds": args.seeds,
            "binary_n_per_class": args.binary_n_per_class,
            "multi_n_per_class": args.multi_n_per_class,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "head_lr": args.head_lr,
            "backbone_lr": args.backbone_lr,
            "full_backbone_lr": args.full_backbone_lr,
            "low_full_backbone_lr": args.low_full_backbone_lr,
            "weight_decay": args.weight_decay,
            "ridge": args.ridge,
            "k_neighbors": args.k_neighbors,
            "device": str(device),
            "variants": VARIANTS,
        },
        "summaries": summaries,
        "finals": finals,
        "results": rows,
    }
    (EXP_DIR / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot_results(finals)
    command = "python3 experiments/017-full-finetune-schedule-control/scripts/run_schedule_control.py"
    if args.quick:
        command += " --quick"
    else:
        command += " --seeds " + " ".join(str(seed) for seed in args.seeds)
        command += f" --binary-n-per-class {args.binary_n_per_class} --multi-n-per-class {args.multi_n_per_class}"
        command += f" --epochs {args.epochs} --batch-size {args.batch_size} --eval-batch-size {args.eval_batch_size}"
    command += f" --device {args.device}"
    write_result(command, summaries, finals)
    print(f"Wrote {EXP_DIR / 'metrics.json'}")
    print(f"Wrote {EXP_DIR / 'result.md'}")


if __name__ == "__main__":
    main()
