#!/usr/bin/env python3
"""Robustness check for BN-frozen full ResNet18 fine-tuning."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


EXP_DIR = ROOT / "experiments" / "019-bn-frozen-robustness"
VARIANTS = [
    {"name": "layer4_base", "base_regime": "finetune_layer4", "bn_mode": "original", "augment": False},
    {"name": "all_bn_train", "base_regime": "finetune_all", "bn_mode": "original", "augment": False},
    {"name": "all_bn_eval", "base_regime": "finetune_all", "bn_mode": "all_eval", "augment": False},
    {"name": "all_bn_train_aug", "base_regime": "finetune_all", "bn_mode": "original", "augment": True},
    {"name": "all_bn_eval_aug", "base_regime": "finetune_all", "bn_mode": "all_eval", "augment": True},
]


def load_exp018():
    path = ROOT / "experiments" / "018-full-finetune-bn-control" / "scripts" / "run_bn_control.py"
    spec = importlib.util.spec_from_file_location("exp018_bn_control", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EXP_DIR = EXP_DIR
    module.VARIANTS = VARIANTS
    module.VARIANT_ORDER = [item["name"] for item in VARIANTS]
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--binary-n-per-class", type=int, default=240)
    parser.add_argument("--multi-n-per-class", type=int, default=120)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=3e-5)
    parser.add_argument("--full-backbone-lr", type=float, default=1e-5)
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

    exp018 = load_exp018()
    exp013 = exp018.load_exp013()
    original_epoch_mode = exp013.set_epoch_train_mode
    device = exp013.choose_device(args.device)
    rows = []
    for seed in args.seeds:
        tasks = exp013.build_tasks(args.data_root, args.binary_n_per_class, args.multi_n_per_class, seed)
        for task_idx, task in enumerate(tasks):
            for variant_idx, spec in enumerate(VARIANTS):
                var_args = copy.copy(args)
                var_args.augment = bool(spec["augment"])
                train_seed = seed + 1000 * task_idx + 100 * variant_idx
                print(f"Running seed={seed} {task['name']} / {spec['name']} on {device}", flush=True)
                exp013.set_epoch_train_mode = exp018.make_epoch_mode(exp013, spec["bn_mode"])
                try:
                    trace = exp013.train_regime(task, spec["base_regime"], var_args, device, seed=train_seed)
                finally:
                    exp013.set_epoch_train_mode = original_epoch_mode
                for row in trace:
                    row = dict(row)
                    row["seed"] = seed
                    row["variant"] = spec["name"]
                    row["base_regime"] = spec["base_regime"]
                    row["regime"] = spec["name"]
                    row["bn_mode"] = spec["bn_mode"]
                    row["augment"] = bool(spec["augment"])
                    rows.append(row)

    finals = exp018.final_rows(rows)
    summaries = exp018.summarize(finals)
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
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    (EXP_DIR / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    exp018.plot_results(finals)
    command = "python3 experiments/019-bn-frozen-robustness/scripts/run_bn_frozen_robustness.py"
    if args.quick:
        command += " --quick"
    else:
        command += " --seeds " + " ".join(str(seed) for seed in args.seeds)
        command += f" --binary-n-per-class {args.binary_n_per_class} --multi-n-per-class {args.multi_n_per_class}"
        command += f" --epochs {args.epochs} --batch-size {args.batch_size} --eval-batch-size {args.eval_batch_size}"
    command += f" --device {args.device}"
    exp018.write_result(command, summaries, finals)
    print(f"Wrote {EXP_DIR / 'metrics.json'}")
    print(f"Wrote {EXP_DIR / 'result.md'}")


if __name__ == "__main__":
    main()
