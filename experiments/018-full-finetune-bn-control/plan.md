# Plan

## Question

Is full fine-tuning over-move driven by weight-gradient movement, BatchNorm
running-stat dynamics, or both?

Experiment `017` showed that simple crop/flip augmentation and lower
full-backbone LR do not rescue full fine-tuning. Lower LR also did not reduce
measured kernel movement, which makes BatchNorm/stat-mode dynamics a natural
next mechanism to isolate.

## Variants

- `layer4_base`: positive control from `016`/`017`.
- `all_bn_train`: default full fine-tuning, all BatchNorm modules in train mode.
- `all_bn_eval`: full fine-tuning weights, but all backbone BatchNorm modules in
  eval mode, so running stats are frozen.
- `all_layer4_bn_train`: full fine-tuning weights, but only layer4 BatchNorm
  modules update running stats.

## Tasks

- CIFAR `cat vs dog`
- CIFAR `automobile vs truck`
- CIFAR `vehicles4`

## Default Cloud Run

Run on one visible A40:

```bash
tmux new-session -d -s cntk-018-bn-control \
  'cd /beegfs_hdd/data/nfs_share/users/lifanhong/nishome/CNTK && CUDA_VISIBLE_DEVICES=0 conda run -n cntk python experiments/018-full-finetune-bn-control/scripts/run_bn_control.py --seeds 0 1 2 --binary-n-per-class 120 --multi-n-per-class 60 --epochs 6 --batch-size 64 --eval-batch-size 64 --device cuda'
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

BatchNorm/stat-mode dynamics are implicated if:

```text
all_bn_eval has lower movement and lower overmove than all_bn_train
```

Full-gradient weight movement remains implicated if:

```text
all_bn_eval still overmoves and fails to repair held-out tail/graph roughness
```

Layer-local stat dynamics are implicated if `all_layer4_bn_train` behaves closer
to `layer4_base` than to `all_bn_train`.

## Artifacts

- `metrics.json`
- `result.md`
- `figures/tail_delta_by_variant.png`
- `figures/graph_delta_by_variant.png`
- `figures/movement_vs_tail_delta.png`
