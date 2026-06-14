# Plan

## Question

Does the `013` pretrained ResNet18 fine-tuning metric-dynamics result survive a
small multi-seed local rerun?

This is the simple local/MPS run before using cloud compute. It is not meant to
be the final large-scale fine-tuning experiment.

## Regimes

- `frozen_head`
- `finetune_layer4`
- `finetune_all`

## Tasks

- CIFAR `cat vs dog`
- CIFAR `automobile vs truck`
- CIFAR `vehicles4`

## Default Local Run

```bash
tmux new-session -d -s cntk-015-resnet18-ms \
  'cd /Users/ziegel/Documents/CNTK && python3 experiments/015-resnet18-finetune-multiseed-simple/scripts/run_multiseed.py --seeds 0 1 2 --binary-n-per-class 60 --multi-n-per-class 30 --epochs 4 --batch-size 16 --eval-batch-size 32 --device auto'
```

## Metrics

For every seed/task/regime, reuse experiment `013` diagnostics:

- test kernel movement;
- test tail delta;
- test graph Dirichlet delta;
- test mixing delta;
- head accuracy delta;
- kernel-ridge accuracy delta;
- kernel-ridge margin.

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

This experiment should report repair rates, not just means.

## Artifacts

- `metrics.json`
- `result.md`
- `figures/tail_delta_by_regime.png`
- `figures/graph_delta_by_regime.png`
- `figures/movement_vs_tail_delta.png`
