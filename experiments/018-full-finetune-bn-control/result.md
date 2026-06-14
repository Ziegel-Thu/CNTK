# Result

## Run

Command:

```bash
python3 experiments/018-full-finetune-bn-control/scripts/run_bn_control.py --seeds 0 1 2 --binary-n-per-class 120 --multi-n-per-class 60 --epochs 6 --batch-size 64 --eval-batch-size 64 --device cuda
```

## Summary

| variant | n | movement | tail delta | graph delta | repair rate | overmove rate | head acc delta | ridge acc delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| layer4_base | 9 | 0.486 +/- 0.034 | -0.029 +/- 0.015 | -0.093 +/- 0.023 | 1.00 | 0.00 | 0.477 | -0.005 |
| all_bn_train | 9 | 0.684 +/- 0.021 | 0.040 +/- 0.045 | 0.075 +/- 0.053 | 0.11 | 0.89 | 0.382 | -0.042 |
| all_bn_eval | 9 | 0.483 +/- 0.066 | -0.045 +/- 0.016 | -0.124 +/- 0.030 | 1.00 | 0.00 | 0.434 | 0.013 |
| all_layer4_bn_train | 9 | 0.481 +/- 0.025 | -0.018 +/- 0.020 | -0.057 +/- 0.019 | 0.67 | 0.33 | 0.434 | -0.012 |

Interpretation:

- `layer4_base` is the positive metric-repair control.
- `all_bn_train` is the default full fine-tuning over-move control.
- `all_bn_eval` isolates full weight-gradient movement while freezing
  BatchNorm running statistics.
- `all_layer4_bn_train` tests whether limiting stat updates to layer4
  makes full fine-tuning behave more like the partial fine-tune control.

## Per-Seed Final Rows

| seed | dataset | variant | movement | tail delta | graph delta | repair | overmove | head acc final | ridge acc final |
| ---: | --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| 0 | cifar10_automobilevstruck | layer4_base | 0.534 | -0.023 | -0.116 | yes | no | 0.938 | 0.946 |
| 0 | cifar10_automobilevstruck | all_bn_train | 0.707 | 0.064 | 0.105 | no | yes | 0.867 | 0.871 |
| 0 | cifar10_automobilevstruck | all_bn_eval | 0.605 | -0.053 | -0.171 | yes | no | 0.954 | 0.954 |
| 0 | cifar10_automobilevstruck | all_layer4_bn_train | 0.514 | -0.006 | -0.073 | yes | no | 0.921 | 0.942 |
| 0 | cifar10_catvsdog | layer4_base | 0.470 | -0.060 | -0.090 | yes | no | 0.812 | 0.792 |
| 0 | cifar10_catvsdog | all_bn_train | 0.697 | -0.058 | -0.024 | yes | no | 0.758 | 0.796 |
| 0 | cifar10_catvsdog | all_bn_eval | 0.437 | -0.060 | -0.112 | yes | no | 0.825 | 0.817 |
| 0 | cifar10_catvsdog | all_layer4_bn_train | 0.478 | -0.046 | -0.078 | yes | no | 0.792 | 0.796 |
| 0 | cifar10_vehicles4 | layer4_base | 0.447 | -0.015 | -0.089 | yes | no | 0.879 | 0.871 |
| 0 | cifar10_vehicles4 | all_bn_train | 0.648 | 0.077 | 0.090 | no | yes | 0.783 | 0.838 |
| 0 | cifar10_vehicles4 | all_bn_eval | 0.404 | -0.030 | -0.106 | yes | no | 0.863 | 0.879 |
| 0 | cifar10_vehicles4 | all_layer4_bn_train | 0.473 | -0.021 | -0.077 | yes | no | 0.879 | 0.887 |
| 1 | cifar10_automobilevstruck | layer4_base | 0.532 | -0.026 | -0.143 | yes | no | 0.921 | 0.925 |
| 1 | cifar10_automobilevstruck | all_bn_train | 0.683 | 0.046 | 0.073 | no | yes | 0.892 | 0.883 |
| 1 | cifar10_automobilevstruck | all_bn_eval | 0.471 | -0.029 | -0.151 | yes | no | 0.929 | 0.933 |
| 1 | cifar10_automobilevstruck | all_layer4_bn_train | 0.499 | 0.003 | -0.045 | no | yes | 0.942 | 0.912 |
| 1 | cifar10_catvsdog | layer4_base | 0.462 | -0.049 | -0.072 | yes | no | 0.829 | 0.817 |
| 1 | cifar10_catvsdog | all_bn_train | 0.705 | -0.013 | 0.010 | no | yes | 0.721 | 0.792 |
| 1 | cifar10_catvsdog | all_bn_eval | 0.459 | -0.061 | -0.110 | yes | no | 0.825 | 0.838 |
| 1 | cifar10_catvsdog | all_layer4_bn_train | 0.492 | -0.054 | -0.068 | yes | no | 0.800 | 0.812 |
| 1 | cifar10_vehicles4 | layer4_base | 0.450 | -0.021 | -0.077 | yes | no | 0.896 | 0.875 |
| 1 | cifar10_vehicles4 | all_bn_train | 0.655 | 0.089 | 0.156 | no | yes | 0.742 | 0.821 |
| 1 | cifar10_vehicles4 | all_bn_eval | 0.392 | -0.016 | -0.065 | yes | no | 0.892 | 0.879 |
| 1 | cifar10_vehicles4 | all_layer4_bn_train | 0.445 | -0.026 | -0.043 | yes | no | 0.858 | 0.825 |
| 2 | cifar10_automobilevstruck | layer4_base | 0.531 | -0.008 | -0.105 | yes | no | 0.896 | 0.908 |
| 2 | cifar10_automobilevstruck | all_bn_train | 0.691 | 0.060 | 0.102 | no | yes | 0.900 | 0.883 |
| 2 | cifar10_automobilevstruck | all_bn_eval | 0.512 | -0.040 | -0.151 | yes | no | 0.938 | 0.933 |
| 2 | cifar10_automobilevstruck | all_layer4_bn_train | 0.495 | 0.000 | -0.060 | no | yes | 0.896 | 0.917 |
| 2 | cifar10_catvsdog | layer4_base | 0.473 | -0.034 | -0.072 | yes | no | 0.850 | 0.812 |
| 2 | cifar10_catvsdog | all_bn_train | 0.703 | 0.032 | 0.041 | no | yes | 0.762 | 0.800 |
| 2 | cifar10_catvsdog | all_bn_eval | 0.549 | -0.065 | -0.111 | yes | no | 0.821 | 0.833 |
| 2 | cifar10_catvsdog | all_layer4_bn_train | 0.497 | -0.016 | -0.046 | yes | no | 0.808 | 0.842 |
| 2 | cifar10_vehicles4 | layer4_base | 0.471 | -0.024 | -0.070 | yes | no | 0.867 | 0.846 |
| 2 | cifar10_vehicles4 | all_bn_train | 0.669 | 0.067 | 0.118 | no | yes | 0.708 | 0.775 |
| 2 | cifar10_vehicles4 | all_bn_eval | 0.518 | -0.050 | -0.136 | yes | no | 0.858 | 0.883 |
| 2 | cifar10_vehicles4 | all_layer4_bn_train | 0.436 | 0.002 | -0.019 | no | yes | 0.817 | 0.800 |

## Artifacts

- `metrics.json`
- `figures/tail_delta_by_variant.png`
- `figures/graph_delta_by_variant.png`
- `figures/movement_vs_tail_delta.png`
