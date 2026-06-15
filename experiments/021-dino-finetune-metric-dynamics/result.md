# Result

## Run

Command:

```bash
python3 experiments/021-dino-finetune-metric-dynamics/scripts/run_dino_dynamics.py --seeds 0 1 2 --binary-n-per-class 120 --multi-n-per-class 60 --epochs 4 --batch-size 64 --eval-batch-size 64 --device cuda
```

## Summary

| variant | n | movement | tail delta | graph delta | repair rate | overmove rate | head acc delta | ridge acc delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen_head | 9 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.00 | 0.00 | 0.347 | 0.000 |
| finetune_lastblock | 9 | 0.030 +/- 0.004 | -0.002 +/- 0.001 | -0.005 +/- 0.005 | 0.89 | 0.00 | 0.436 | 0.000 |
| finetune_all | 9 | 0.326 +/- 0.127 | -0.030 +/- 0.034 | -0.067 +/- 0.066 | 0.78 | 0.22 | 0.446 | 0.012 |
| finetune_all_aug | 9 | 0.380 +/- 0.084 | -0.006 +/- 0.050 | -0.039 +/- 0.079 | 0.56 | 0.44 | 0.430 | 0.003 |

Interpretation:

- DINO ViT-S/16 is a self-supervised no-BatchNorm backbone, so this
  separates the no-BN ViT signal from supervised ImageNet pretraining.
- A full fine-tuning variant supports useful metric dynamics only if
  held-out tail and graph roughness fall, not merely if head accuracy
  improves.
- If DINO full fine-tuning overmoves, the next control should separate
  architecture/no-BN effects from pretraining-objective effects.

Direct read:

- DINO ViT-S/16 full fine-tuning gives a useful but less clean repair signal
  than supervised ImageNet ViT-B/16 in `020`.
- `finetune_all` lowers mean held-out tail/graph roughness by `-0.030/-0.067`
  with repair/overmove `0.78/0.22`, so self-supervised no-BN full fine-tuning
  can repair the metric, but not in every seed-task row.
- `finetune_lastblock` barely moves the metric (`0.030`) and gives only tiny
  tail/graph changes (`-0.002/-0.005`), matching the pattern that last-block
  ViT tuning is a weak metric-adaptation intervention here.
- `finetune_all_aug` is weaker and less stable than unaugmented full tuning:
  mean tail/graph deltas are only `-0.006/-0.039`, with overmove rate `0.44`.
- The result refines the no-BN story: removing BatchNorm avoids the ResNet18
  BN-stat failure mode, but self-supervised DINO full fine-tuning is more
  schedule/pretraining sensitive than supervised ViT-B/16.

## Per-Seed Final Rows

| seed | dataset | variant | movement | tail delta | graph delta | repair | overmove | head acc final | ridge acc final |
| ---: | --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| 0 | cifar10_automobilevstruck | frozen_head | 0.000 | 0.000 | 0.000 | no | no | 0.804 | 0.925 |
| 0 | cifar10_automobilevstruck | finetune_lastblock | 0.033 | -0.004 | 0.001 | no | no | 0.871 | 0.938 |
| 0 | cifar10_automobilevstruck | finetune_all | 0.455 | -0.069 | -0.138 | yes | no | 0.908 | 0.946 |
| 0 | cifar10_automobilevstruck | finetune_all_aug | 0.414 | -0.043 | -0.064 | yes | no | 0.879 | 0.938 |
| 0 | cifar10_catvsdog | frozen_head | 0.000 | 0.000 | 0.000 | no | no | 0.767 | 0.838 |
| 0 | cifar10_catvsdog | finetune_lastblock | 0.023 | -0.003 | -0.005 | yes | no | 0.829 | 0.829 |
| 0 | cifar10_catvsdog | finetune_all | 0.244 | 0.025 | 0.055 | no | yes | 0.762 | 0.833 |
| 0 | cifar10_catvsdog | finetune_all_aug | 0.390 | -0.046 | -0.101 | yes | no | 0.829 | 0.875 |
| 0 | cifar10_vehicles4 | frozen_head | 0.000 | 0.000 | 0.000 | no | no | 0.696 | 0.925 |
| 0 | cifar10_vehicles4 | finetune_lastblock | 0.033 | -0.001 | -0.010 | yes | no | 0.758 | 0.925 |
| 0 | cifar10_vehicles4 | finetune_all | 0.310 | -0.045 | -0.107 | yes | no | 0.883 | 0.929 |
| 0 | cifar10_vehicles4 | finetune_all_aug | 0.365 | 0.017 | 0.017 | no | yes | 0.758 | 0.892 |
| 1 | cifar10_automobilevstruck | frozen_head | 0.000 | 0.000 | 0.000 | no | no | 0.883 | 0.917 |
| 1 | cifar10_automobilevstruck | finetune_lastblock | 0.030 | -0.002 | -0.002 | yes | no | 0.825 | 0.912 |
| 1 | cifar10_automobilevstruck | finetune_all | 0.265 | -0.030 | -0.065 | yes | no | 0.879 | 0.921 |
| 1 | cifar10_automobilevstruck | finetune_all_aug | 0.595 | -0.090 | -0.199 | yes | no | 0.950 | 0.967 |
| 1 | cifar10_catvsdog | frozen_head | 0.000 | 0.000 | 0.000 | no | no | 0.787 | 0.796 |
| 1 | cifar10_catvsdog | finetune_lastblock | 0.025 | -0.000 | -0.002 | yes | no | 0.838 | 0.800 |
| 1 | cifar10_catvsdog | finetune_all | 0.366 | -0.065 | -0.078 | yes | no | 0.863 | 0.863 |
| 1 | cifar10_catvsdog | finetune_all_aug | 0.326 | 0.075 | 0.042 | no | yes | 0.746 | 0.762 |
| 1 | cifar10_vehicles4 | frozen_head | 0.000 | 0.000 | 0.000 | no | no | 0.654 | 0.921 |
| 1 | cifar10_vehicles4 | finetune_lastblock | 0.032 | -0.002 | -0.000 | yes | no | 0.846 | 0.921 |
| 1 | cifar10_vehicles4 | finetune_all | 0.232 | 0.005 | -0.012 | no | yes | 0.838 | 0.896 |
| 1 | cifar10_vehicles4 | finetune_all_aug | 0.315 | -0.027 | -0.055 | yes | no | 0.854 | 0.938 |
| 2 | cifar10_automobilevstruck | frozen_head | 0.000 | 0.000 | 0.000 | no | no | 0.863 | 0.946 |
| 2 | cifar10_automobilevstruck | finetune_lastblock | 0.035 | -0.003 | -0.013 | yes | no | 0.908 | 0.942 |
| 2 | cifar10_automobilevstruck | finetune_all | 0.618 | -0.075 | -0.180 | yes | no | 0.950 | 0.979 |
| 2 | cifar10_automobilevstruck | finetune_all_aug | 0.350 | -0.020 | -0.075 | yes | no | 0.904 | 0.946 |
| 2 | cifar10_catvsdog | frozen_head | 0.000 | 0.000 | 0.000 | no | no | 0.750 | 0.883 |
| 2 | cifar10_catvsdog | finetune_lastblock | 0.026 | -0.001 | -0.002 | yes | no | 0.804 | 0.883 |
| 2 | cifar10_catvsdog | finetune_all | 0.206 | -0.011 | -0.026 | yes | no | 0.812 | 0.871 |
| 2 | cifar10_catvsdog | finetune_all_aug | 0.378 | 0.058 | 0.072 | no | yes | 0.771 | 0.838 |
| 2 | cifar10_vehicles4 | frozen_head | 0.000 | 0.000 | 0.000 | no | no | 0.738 | 0.900 |
| 2 | cifar10_vehicles4 | finetune_lastblock | 0.033 | -0.002 | -0.011 | yes | no | 0.817 | 0.904 |
| 2 | cifar10_vehicles4 | finetune_all | 0.240 | -0.008 | -0.048 | yes | no | 0.850 | 0.917 |
| 2 | cifar10_vehicles4 | finetune_all_aug | 0.288 | 0.021 | 0.014 | no | yes | 0.821 | 0.921 |

## Artifacts

- `metrics.json`
- `figures/tail_delta_by_variant.png`
- `figures/graph_delta_by_variant.png`
- `figures/movement_vs_tail_delta.png`
