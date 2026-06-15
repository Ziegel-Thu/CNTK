# Experiment Index

Compact result summaries live in `experiments/results.md`.

## Execution Status

| ID | Name | Priority | Status | Main Question | Resource |
| --- | --- | --- | --- | --- | --- |
| 001 | spectral-tail-diagnostics | P0 | toy + image subsets + bound audit complete | Does local mixing predict spectral tail and slow kernel flow? | CPU first, GPU optional |
| 002 | feature-metric-dynamics | P1 | toy + MNIST + CIFAR MLP/CNN complete | Does feature learning reduce mixing/tail by changing `K_t`? | GPU preferred |
| 003 | fixed-representation-sweep | P2 | quick sweep complete | Is this fixed-metric general, beyond CNTK? | GPU for feature extraction |
| 004 | intrinsic-collision-stress | P2 | MNIST stress complete | What mixing is correctable vs intrinsic/noisy? | CPU/GPU |
| 005 | multiclass-obstruction-diagnostics | P2 | MNIST/CIFAR multiclass complete | Do local mixing and spectral tail diagnostics extend beyond binary labels? | CPU/GPU |
| 006 | cifar-multiclass-schedule-sweep | P2 | CPU two-seed schedule sweep complete | Is weak CIFAR multiclass transfer a schedule artifact? | GPU preferred |
| 007 | margin-tail-audit | P2 | existing-metrics audit complete | Is margin redundant with tail, or complementary? | CPU |
| 008 | graph-energy-kernel-margin | P1 | binary MNIST/CIFAR complete | Do graph energy and kernel margin connect mixing to classifier consequences? | CPU |
| 009 | tail-training-time-consequence | P1 | toy + MNIST/CIFAR static kernels complete | Does spectral tail predict kernel gradient-flow time/source norm? | CPU |
| 010 | pretrained-fixed-representation-sweep | P1 | CIFAR ImageNet ResNet18 complete | Does the obstruction story hold for real pretrained frozen features? | CPU/GPU |
| 011 | self-supervised-fixed-representation-sweep | P1 | CIFAR DINO ViT-S/16 complete | Does the obstruction story hold for self-supervised frozen features? | CPU/GPU |
| 012 | source-norm-controlled-sweep | P1 | controlled same-kernel sweep complete | Do tail/mixing predict RKHS/source-norm proxies when kernel scale is controlled? | CPU |
| 013 | pretrained-finetune-metric-dynamics | P1 | CIFAR ResNet18 fine-tune dynamics complete | Does fine-tuning repair held-out metric geometry or merely train the head? | MPS/GPU preferred |
| 014 | mixing-alignment-controlled-audit | P1 | existing-results audit complete | Do local mixing/graph roughness add signal beyond global alignment? | CPU |
| 015 | resnet18-finetune-multiseed-simple | P1 | local 3-seed probe complete | Does the ResNet18 fine-tune dynamics result survive a simple local multi-seed rerun? | MPS/GPU |
| 016 | resnet18-cloud-single-gpu | P1 | cloud 5-seed single-GPU probe complete | Does the ResNet18 fine-tune dynamics result survive a larger single-GPU cloud rerun? | 1x A40 |
| 017 | full-finetune-schedule-control | P1 | cloud schedule-control complete | Is full fine-tune over-move caused by regime choice or weak schedule/no augmentation? | 1x A40 |
| 018 | full-finetune-bn-control | P1 | cloud BatchNorm-control complete | Is full fine-tune over-move driven by BatchNorm/stat-mode dynamics, weight gradients, or both? | 1x A40 |
| 019 | bn-frozen-robustness | P1 | cloud robustness run complete | Does BN-frozen full fine-tuning remain stable on larger subsets and with augmentation? | 1x A40 |
| 020 | vit-finetune-metric-dynamics | P1 | cloud ViT dynamics complete | Does a no-BatchNorm ViT backbone reproduce or avoid the ResNet18 BN-stat overmove mode? | 1x A40 |
| 021 | dino-finetune-metric-dynamics | P1 | cloud DINO dynamics complete | Does self-supervised DINO ViT-S/16 full fine-tuning also give useful no-BN metric repair? | 1x A40 |

## Standard Artifact Contract

Each experiment should produce:

- `plan.md`
- `result.md` as the canonical human entry point
- `metrics.json` or `metrics_over_time.json`
- `figures/*.png`
- exact command lines in `result.md`

Supplemental phase-specific files such as `result_cifar.md` are allowed when an
experiment has multiple sub-runs, but they should be linked from the canonical
entry point or `experiments/results.md`.

## Standard Result Questions

Every `result.md` should answer:

- What was run?
- What datasets/subsets/seeds were used?
- What diagnostics moved?
- Did local mixing explain anything beyond global alignment?
- Did the result support, weaken, or refine the project hypothesis?
- What should be run next?

## tmux Sessions

Use these names:

- `cntk-001-spectral-tail`
- `cntk-002-feature-dynamics`
- `cntk-003-fixed-repr`
- `cntk-004-collision-stress`
- `cntk-005-multiclass`
- `cntk-006-cifar-schedule`
- `cntk-007-margin-audit`
- `cntk-008-graph-margin`
- `cntk-009-tail-time`
- `cntk-010-pretrained`
- `cntk-011-selfsup`
- `cntk-012-source-norm`
- `cntk-013-finetune-dyn`
- `cntk-014-mix-align-audit`
- `cntk-015-resnet18-ms`
- `cntk-016-cloud-single`
- `cntk-017-schedule-control`
- `cntk-018-bn-control`
- `cntk-019-bn-robust`
- `cntk-020-vit-dyn`
- `cntk-021-dino-dyn`
