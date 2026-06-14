# Result

## Run

Command:

```bash
python3 experiments/017-full-finetune-schedule-control/scripts/run_schedule_control.py --seeds 0 1 2 --binary-n-per-class 120 --multi-n-per-class 60 --epochs 6 --batch-size 64 --eval-batch-size 64 --device cuda
```

## Summary

| variant | n | movement | tail delta | graph delta | repair rate | overmove rate | head acc delta | ridge acc delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| layer4_base | 9 | 0.486 +/- 0.034 | -0.029 +/- 0.015 | -0.093 +/- 0.023 | 1.00 | 0.00 | 0.477 | -0.005 |
| all_base | 9 | 0.684 +/- 0.021 | 0.040 +/- 0.045 | 0.074 +/- 0.052 | 0.11 | 0.89 | 0.382 | -0.043 |
| all_aug | 9 | 0.683 +/- 0.022 | 0.037 +/- 0.049 | 0.080 +/- 0.056 | 0.11 | 0.89 | 0.347 | -0.035 |
| all_low_lr | 9 | 0.697 +/- 0.018 | 0.056 +/- 0.049 | 0.123 +/- 0.057 | 0.00 | 1.00 | 0.317 | -0.051 |
| all_aug_low_lr | 9 | 0.694 +/- 0.016 | 0.054 +/- 0.050 | 0.117 +/- 0.057 | 0.00 | 1.00 | 0.319 | -0.053 |

Interpretation:

- `layer4_base` is the positive control inherited from `016`.
- `all_base` is the default full fine-tuning over-move control.
- A full fine-tuning variant counts as schedule rescue only if it lowers
  both held-out tail and graph roughness while reducing overmove rate.
- If augmentation or lower LR mainly improves head accuracy without
  lowering held-out tail/graph roughness, it should not be read as metric
  repair.

Direct read:

- `layer4_base` repairs held-out geometry in all `9/9` seed-task rows.
- Simple crop/flip augmentation does not rescue full fine-tuning:
  `all_aug` keeps the same repair rate as `all_base` (`0.11`) and the same
  overmove rate (`0.89`).
- Lowering the full-backbone LR to `3e-6` also does not rescue full fine-tuning;
  both low-LR variants have repair rate `0.00` and overmove rate `1.00`.
- Because lower LR does not reduce the measured movement, the next mechanism
  control should isolate BatchNorm/stat-mode dynamics from weight-gradient
  dynamics.

## Per-Seed Final Rows

| seed | dataset | variant | movement | tail delta | graph delta | repair | overmove | head acc final | ridge acc final |
| ---: | --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| 0 | cifar10_automobilevstruck | layer4_base | 0.534 | -0.023 | -0.116 | yes | no | 0.938 | 0.946 |
| 0 | cifar10_automobilevstruck | all_base | 0.707 | 0.064 | 0.104 | no | yes | 0.867 | 0.871 |
| 0 | cifar10_automobilevstruck | all_aug | 0.699 | 0.049 | 0.063 | no | yes | 0.879 | 0.908 |
| 0 | cifar10_automobilevstruck | all_low_lr | 0.721 | 0.075 | 0.156 | no | yes | 0.804 | 0.867 |
| 0 | cifar10_automobilevstruck | all_aug_low_lr | 0.716 | 0.071 | 0.152 | no | yes | 0.863 | 0.879 |
| 0 | cifar10_catvsdog | layer4_base | 0.470 | -0.060 | -0.090 | yes | no | 0.812 | 0.792 |
| 0 | cifar10_catvsdog | all_base | 0.697 | -0.058 | -0.023 | yes | no | 0.758 | 0.796 |
| 0 | cifar10_catvsdog | all_aug | 0.698 | -0.056 | -0.039 | yes | no | 0.771 | 0.825 |
| 0 | cifar10_catvsdog | all_low_lr | 0.698 | -0.033 | 0.012 | no | yes | 0.729 | 0.812 |
| 0 | cifar10_catvsdog | all_aug_low_lr | 0.700 | -0.040 | 0.017 | no | yes | 0.650 | 0.796 |
| 0 | cifar10_vehicles4 | layer4_base | 0.447 | -0.015 | -0.089 | yes | no | 0.879 | 0.871 |
| 0 | cifar10_vehicles4 | all_base | 0.648 | 0.077 | 0.088 | no | yes | 0.783 | 0.838 |
| 0 | cifar10_vehicles4 | all_aug | 0.656 | 0.088 | 0.120 | no | yes | 0.754 | 0.842 |
| 0 | cifar10_vehicles4 | all_low_lr | 0.663 | 0.091 | 0.123 | no | yes | 0.738 | 0.812 |
| 0 | cifar10_vehicles4 | all_aug_low_lr | 0.664 | 0.099 | 0.128 | no | yes | 0.679 | 0.808 |
| 1 | cifar10_automobilevstruck | layer4_base | 0.532 | -0.026 | -0.143 | yes | no | 0.921 | 0.925 |
| 1 | cifar10_automobilevstruck | all_base | 0.683 | 0.046 | 0.073 | no | yes | 0.892 | 0.883 |
| 1 | cifar10_automobilevstruck | all_aug | 0.686 | 0.048 | 0.100 | no | yes | 0.896 | 0.883 |
| 1 | cifar10_automobilevstruck | all_low_lr | 0.700 | 0.065 | 0.140 | no | yes | 0.754 | 0.892 |
| 1 | cifar10_automobilevstruck | all_aug_low_lr | 0.694 | 0.062 | 0.133 | no | yes | 0.812 | 0.871 |
| 1 | cifar10_catvsdog | layer4_base | 0.462 | -0.049 | -0.072 | yes | no | 0.829 | 0.817 |
| 1 | cifar10_catvsdog | all_base | 0.705 | -0.013 | 0.010 | no | yes | 0.721 | 0.792 |
| 1 | cifar10_catvsdog | all_aug | 0.707 | -0.011 | 0.024 | no | yes | 0.725 | 0.775 |
| 1 | cifar10_catvsdog | all_low_lr | 0.710 | -0.023 | 0.074 | no | yes | 0.750 | 0.787 |
| 1 | cifar10_catvsdog | all_aug_low_lr | 0.711 | -0.015 | 0.063 | no | yes | 0.688 | 0.775 |
| 1 | cifar10_vehicles4 | layer4_base | 0.450 | -0.021 | -0.077 | yes | no | 0.896 | 0.875 |
| 1 | cifar10_vehicles4 | all_base | 0.655 | 0.089 | 0.155 | no | yes | 0.742 | 0.821 |
| 1 | cifar10_vehicles4 | all_aug | 0.639 | 0.090 | 0.153 | no | yes | 0.787 | 0.804 |
| 1 | cifar10_vehicles4 | all_low_lr | 0.670 | 0.116 | 0.187 | no | yes | 0.713 | 0.796 |
| 1 | cifar10_vehicles4 | all_aug_low_lr | 0.673 | 0.123 | 0.208 | no | yes | 0.679 | 0.800 |
| 2 | cifar10_automobilevstruck | layer4_base | 0.531 | -0.008 | -0.105 | yes | no | 0.896 | 0.908 |
| 2 | cifar10_automobilevstruck | all_base | 0.691 | 0.059 | 0.102 | no | yes | 0.900 | 0.883 |
| 2 | cifar10_automobilevstruck | all_aug | 0.691 | 0.073 | 0.125 | no | yes | 0.817 | 0.875 |
| 2 | cifar10_automobilevstruck | all_low_lr | 0.705 | 0.086 | 0.181 | no | yes | 0.792 | 0.863 |
| 2 | cifar10_automobilevstruck | all_aug_low_lr | 0.701 | 0.081 | 0.151 | no | yes | 0.796 | 0.867 |
| 2 | cifar10_catvsdog | layer4_base | 0.473 | -0.034 | -0.072 | yes | no | 0.850 | 0.812 |
| 2 | cifar10_catvsdog | all_base | 0.703 | 0.032 | 0.044 | no | yes | 0.762 | 0.796 |
| 2 | cifar10_catvsdog | all_aug | 0.703 | -0.013 | 0.064 | no | yes | 0.779 | 0.796 |
| 2 | cifar10_catvsdog | all_low_lr | 0.708 | 0.035 | 0.065 | no | yes | 0.738 | 0.779 |
| 2 | cifar10_catvsdog | all_aug_low_lr | 0.705 | 0.027 | 0.050 | no | yes | 0.683 | 0.796 |
| 2 | cifar10_vehicles4 | layer4_base | 0.471 | -0.024 | -0.070 | yes | no | 0.867 | 0.846 |
| 2 | cifar10_vehicles4 | all_base | 0.669 | 0.067 | 0.116 | no | yes | 0.708 | 0.775 |
| 2 | cifar10_vehicles4 | all_aug | 0.665 | 0.068 | 0.114 | no | yes | 0.713 | 0.812 |
| 2 | cifar10_vehicles4 | all_low_lr | 0.693 | 0.088 | 0.168 | no | yes | 0.637 | 0.767 |
| 2 | cifar10_vehicles4 | all_aug_low_lr | 0.685 | 0.083 | 0.148 | no | yes | 0.704 | 0.771 |

## Artifacts

- `metrics.json`
- `figures/tail_delta_by_variant.png`
- `figures/graph_delta_by_variant.png`
- `figures/movement_vs_tail_delta.png`
