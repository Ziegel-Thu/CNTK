# Plan

## Question

Is the `finetune_all` over-move effect from `013`/`015`/`016` caused by full
backbone adaptation itself, or by a weak no-augmentation schedule?

This is a mechanism-control experiment. It should not be sold as another
benchmark; the useful result is whether schedule/augmentation can turn full
fine-tuning from harmful movement into held-out metric repair.

## Variants

- `layer4_base`: `finetune_layer4`, no augmentation, default layer4 LR.
- `all_base`: `finetune_all`, no augmentation, default full-backbone LR.
- `all_aug`: `finetune_all`, crop/flip augmentation, default full-backbone LR.
- `all_low_lr`: `finetune_all`, no augmentation, lower full-backbone LR.
- `all_aug_low_lr`: `finetune_all`, crop/flip augmentation, lower full-backbone LR.

## Tasks

- CIFAR `cat vs dog`
- CIFAR `automobile vs truck`
- CIFAR `vehicles4`

## Default Cloud Run

Run on one visible A40:

```bash
tmux new-session -d -s cntk-017-schedule-control \
  'cd /beegfs_hdd/data/nfs_share/users/lifanhong/nishome/CNTK && CUDA_VISIBLE_DEVICES=0 conda run -n cntk python experiments/017-full-finetune-schedule-control/scripts/run_schedule_control.py --seeds 0 1 2 --binary-n-per-class 120 --multi-n-per-class 60 --epochs 6 --batch-size 64 --eval-batch-size 64 --device cuda'
```

## Metrics

For every seed/task/variant:

- test kernel movement;
- test tail delta;
- test graph Dirichlet delta;
- test mixing delta;
- head accuracy delta;
- kernel-ridge accuracy delta;
- kernel-ridge margin;
- metric-repair and overmove rates by variant.

## Success / Failure Criteria

Full fine-tuning is rescued only if a full-backbone variant has:

```text
tail delta < 0
graph delta < 0
overmove rate clearly lower than all_base
repair rate close to layer4_base
```

If full variants still move more while keeping positive tail/graph deltas, the
over-move result is not merely a no-augmentation artifact.

## Artifacts

- `metrics.json`
- `result.md`
- `figures/tail_delta_by_variant.png`
- `figures/graph_delta_by_variant.png`
- `figures/movement_vs_tail_delta.png`
