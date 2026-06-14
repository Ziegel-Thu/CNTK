# Plan

## Question

Does the BatchNorm-frozen full fine-tuning repair signal from `018` survive a
larger subset and simple augmentation?

This experiment is meant to make the BatchNorm/stat-mode discovery reliable
enough to guide future runs. It is not just another ResNet18 benchmark.

## Variants

- `layer4_base`: positive partial fine-tune control.
- `all_bn_train`: default full fine-tune, all BatchNorm modules in train mode.
- `all_bn_eval`: full weight-gradient fine-tune with all BatchNorm running stats
  frozen.
- `all_bn_train_aug`: default full fine-tune with crop/flip augmentation.
- `all_bn_eval_aug`: BN-frozen full fine-tune with crop/flip augmentation.

## Tasks

- CIFAR `cat vs dog`
- CIFAR `automobile vs truck`
- CIFAR `vehicles4`

## Default Cloud Run

Run on one visible A40:

```bash
tmux new-session -d -s cntk-019-bn-robust \
  'cd /beegfs_hdd/data/nfs_share/users/lifanhong/nishome/CNTK && CUDA_VISIBLE_DEVICES=0 conda run -n cntk python experiments/019-bn-frozen-robustness/scripts/run_bn_frozen_robustness.py --seeds 0 1 2 --binary-n-per-class 240 --multi-n-per-class 120 --epochs 6 --batch-size 64 --eval-batch-size 64 --device cuda'
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

The BN-frozen finding is robust if:

```text
all_bn_eval repair rate remains high
all_bn_eval overmove rate remains low
all_bn_eval beats all_bn_train on held-out tail/graph deltas
```

Augmentation is compatible with the BN-frozen mechanism if `all_bn_eval_aug`
keeps negative held-out tail/graph deltas and does not reintroduce overmove.

## Artifacts

- `metrics.json`
- `result.md`
- `figures/tail_delta_by_variant.png`
- `figures/graph_delta_by_variant.png`
- `figures/movement_vs_tail_delta.png`
