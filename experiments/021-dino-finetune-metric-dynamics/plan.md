# Plan

## Question

Self-supervised DINO ViT-S/16 fine-tuning 是否也表现出 no-BN backbone 的
useful full-finetune metric repair？

`020` 已经显示 supervised ImageNet ViT-B/16 full fine-tuning 不复现
ResNet18 的 BN/stat-state overmove，并且能降低 held-out tail/graph roughness。
本实验把 backbone 换成 `011` 使用过的 self-supervised DINO ViT-S/16，测试
这个结论是否依赖 supervised ImageNet label pretraining。

## Variants

- `frozen_head`: frozen DINO ViT-S/16 backbone + trained linear head。
- `finetune_lastblock`: last Transformer block + final norm + head。
- `finetune_all`: full DINO backbone + head。
- `finetune_all_aug`: full DINO backbone + head with crop/flip augmentation。

## Tasks

- CIFAR `cat vs dog`
- CIFAR `automobile vs truck`
- CIFAR `vehicles4`

## Default Cloud Run

Run on one visible A40 in `tmux`:

```bash
tmux new-session -d -s cntk-021-dino-dyn \
  'cd ~/CNTK && CUDA_VISIBLE_DEVICES=0 conda run -n cntk python experiments/021-dino-finetune-metric-dynamics/scripts/run_dino_dynamics.py --seeds 0 1 2 --binary-n-per-class 120 --multi-n-per-class 60 --epochs 4 --batch-size 64 --eval-batch-size 64 --device cuda'
```

## Implementation Notes

- Start from `experiments/020-vit-finetune-metric-dynamics/scripts/run_vit_dynamics.py`.
- Reuse task construction, checkpoint diagnostics, plotting, final-row summary,
  repair/overmove definitions, and result writer shape from `020`.
- Replace the model constructor with `torch.hub` DINO ViT-S/16 loading, matching
  the frozen feature baseline in
  `experiments/011-self-supervised-fixed-representation-sweep/scripts/run_selfsup.py`.
- Keep the first run scale identical to `020` so differences are interpretable
  as backbone/pretraining differences rather than schedule differences.
- Use the same initial learning-rate scale as `020` unless the DINO run is
  numerically unstable: head LR `1e-3`, last-block LR `1e-5`, full-backbone LR
  `3e-6`, weight decay `1e-4`.

## Metrics

For every seed/task/variant:

- test kernel movement;
- test tail delta;
- test graph Dirichlet delta;
- test mixing/disagreement delta;
- head accuracy delta;
- kernel-ridge accuracy delta;
- kernel-ridge margin;
- metric-repair and overmove rates by variant.

## Success / Failure Criteria

The no-BN useful-repair interpretation is strengthened if:

```text
DINO full fine-tuning lowers held-out tail and graph roughness in most seed-task
rows, with low overmove rate.
```

A supervised-pretraining caveat is introduced if:

```text
DINO full fine-tuning improves head accuracy but fails to lower held-out
tail/graph roughness, or overmoves despite no BatchNorm.
```

Either outcome is meaningful: the first extends `020` to self-supervised
pretraining, while the second separates no-BN architecture effects from
pretraining/objective effects.

## Artifacts

- `metrics.json`
- `result.md`
- `figures/tail_delta_by_variant.png`
- `figures/graph_delta_by_variant.png`
- `figures/movement_vs_tail_delta.png`
