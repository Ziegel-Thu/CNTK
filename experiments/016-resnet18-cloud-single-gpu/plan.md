# Plan

## Question

Does the ResNet18 fine-tuning metric-dynamics pattern from `013` and `015`
survive a larger single-GPU cloud run?

This is the first cloud run, not the final large-backbone experiment. The goal is
to scale local evidence modestly while preserving comparability with `015`.

## Regimes

- `frozen_head`
- `finetune_layer4`
- `finetune_all`

## Tasks

- CIFAR `cat vs dog`
- CIFAR `automobile vs truck`
- CIFAR `vehicles4`

## Default Cloud Run

Run on one visible A40:

```bash
tmux new-session -d -s cntk-016-cloud-single \
  'cd /beegfs_hdd/data/nfs_share/users/lifanhong/nishome/CNTK && CUDA_VISIBLE_DEVICES=0 conda run -n cntk python experiments/016-resnet18-cloud-single-gpu/scripts/run_cloud_single_gpu.py --seeds 0 1 2 3 4 --binary-n-per-class 120 --multi-n-per-class 60 --epochs 6 --batch-size 64 --eval-batch-size 64 --device cuda'
```

## Metrics

Reuse experiment `015` diagnostics:

- test kernel movement;
- test tail delta;
- test graph Dirichlet delta;
- test mixing delta;
- head accuracy delta;
- kernel-ridge accuracy delta;
- kernel-ridge margin;
- metric-repair and overmove rates by regime.

## Success / Failure Criteria

Positive metric repair:

```text
tail delta < 0
graph delta < 0
movement > frozen_head movement
```

Negative over-move:

```text
movement is large
tail delta > 0 or graph delta > 0
```

The key check is whether `finetune_layer4` remains a more stable repair regime
than `finetune_all` when sample size and seed count increase.

## Artifacts

- `metrics.json`
- `result.md`
- `figures/tail_delta_by_regime.png`
- `figures/graph_delta_by_regime.png`
- `figures/movement_vs_tail_delta.png`
