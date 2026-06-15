# Plan

## Question

Does the ResNet18 BatchNorm/stat-mode failure from `018`/`019` disappear when
the backbone has no BatchNorm?

This experiment switches to an ImageNet-pretrained ViT-B/16 backbone. It asks
whether full fine-tuning of a LayerNorm/Transformer representation repairs the
held-out metric, overmoves in a different way, or behaves mostly like a frozen
feature.

## Variants

- `frozen_head`: frozen ViT backbone plus trained linear head.
- `finetune_lastblock`: last Transformer encoder block plus head.
- `finetune_all`: full ViT backbone plus head.
- `finetune_all_aug`: full ViT backbone plus head with crop/flip augmentation.

## Tasks

- CIFAR `cat vs dog`
- CIFAR `automobile vs truck`
- CIFAR `vehicles4`

## Default Cloud Run

Run on one visible A40:

```bash
tmux new-session -d -s cntk-020-vit-dyn \
  'cd /beegfs_hdd/data/nfs_share/users/lifanhong/nishome/CNTK && CUDA_VISIBLE_DEVICES=0 conda run -n cntk python experiments/020-vit-finetune-metric-dynamics/scripts/run_vit_dynamics.py --seeds 0 1 2 --binary-n-per-class 120 --multi-n-per-class 60 --epochs 4 --batch-size 64 --eval-batch-size 64 --device cuda'
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

The BN mechanism from ResNet is supported if:

```text
ViT full fine-tuning does not reproduce the all-BN-train ResNet overmove pattern
```

The metric-dynamics story remains alive even if ViT overmoves, but then the
failure mechanism is not BatchNorm and needs a different control.

## Artifacts

- `metrics.json`
- `result.md`
- `figures/tail_delta_by_variant.png`
- `figures/graph_delta_by_variant.png`
- `figures/movement_vs_tail_delta.png`
