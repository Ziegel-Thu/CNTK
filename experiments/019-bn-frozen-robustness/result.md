# Result

## Run

Command:

```bash
python3 experiments/019-bn-frozen-robustness/scripts/run_bn_frozen_robustness.py --seeds 0 1 2 --binary-n-per-class 240 --multi-n-per-class 120 --epochs 6 --batch-size 64 --eval-batch-size 64 --device cuda
```

## Summary

| variant | n | movement | tail delta | graph delta | repair rate | overmove rate | head acc delta | ridge acc delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| layer4_base | 9 | 0.668 +/- 0.064 | -0.044 +/- 0.008 | -0.143 +/- 0.030 | 1.00 | 0.00 | 0.496 | 0.008 |
| all_bn_train | 9 | 0.790 +/- 0.043 | 0.032 +/- 0.022 | 0.026 +/- 0.049 | 0.00 | 1.00 | 0.442 | -0.038 |
| all_bn_eval | 9 | 0.738 +/- 0.051 | -0.057 +/- 0.011 | -0.153 +/- 0.022 | 1.00 | 0.00 | 0.471 | 0.023 |
| all_bn_train_aug | 9 | 0.791 +/- 0.046 | 0.023 +/- 0.021 | 0.011 +/- 0.051 | 0.11 | 0.89 | 0.427 | -0.027 |
| all_bn_eval_aug | 9 | 0.771 +/- 0.097 | -0.065 +/- 0.011 | -0.160 +/- 0.029 | 1.00 | 0.00 | 0.494 | 0.022 |

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
| 0 | cifar10_automobilevstruck | layer4_base | 0.762 | -0.056 | -0.177 | yes | no | 0.950 | 0.954 |
| 0 | cifar10_automobilevstruck | all_bn_train | 0.801 | 0.015 | 0.014 | no | yes | 0.919 | 0.910 |
| 0 | cifar10_automobilevstruck | all_bn_eval | 0.850 | -0.063 | -0.167 | yes | no | 0.942 | 0.948 |
| 0 | cifar10_automobilevstruck | all_bn_train_aug | 0.801 | 0.008 | -0.015 | no | yes | 0.925 | 0.919 |
| 0 | cifar10_automobilevstruck | all_bn_eval_aug | 0.901 | -0.061 | -0.181 | yes | no | 0.954 | 0.944 |
| 0 | cifar10_catvsdog | layer4_base | 0.654 | -0.052 | -0.132 | yes | no | 0.848 | 0.844 |
| 0 | cifar10_catvsdog | all_bn_train | 0.844 | 0.002 | 0.001 | no | yes | 0.821 | 0.817 |
| 0 | cifar10_catvsdog | all_bn_eval | 0.714 | -0.071 | -0.143 | yes | no | 0.858 | 0.867 |
| 0 | cifar10_catvsdog | all_bn_train_aug | 0.841 | 0.003 | -0.003 | no | yes | 0.775 | 0.831 |
| 0 | cifar10_catvsdog | all_bn_eval_aug | 0.690 | -0.087 | -0.168 | yes | no | 0.877 | 0.858 |
| 0 | cifar10_vehicles4 | layer4_base | 0.627 | -0.048 | -0.131 | yes | no | 0.915 | 0.910 |
| 0 | cifar10_vehicles4 | all_bn_train | 0.726 | 0.054 | 0.082 | no | yes | 0.840 | 0.842 |
| 0 | cifar10_vehicles4 | all_bn_eval | 0.719 | -0.053 | -0.134 | yes | no | 0.917 | 0.929 |
| 0 | cifar10_vehicles4 | all_bn_train_aug | 0.717 | 0.037 | 0.069 | no | yes | 0.842 | 0.865 |
| 0 | cifar10_vehicles4 | all_bn_eval_aug | 0.745 | -0.051 | -0.125 | yes | no | 0.923 | 0.925 |
| 1 | cifar10_automobilevstruck | layer4_base | 0.748 | -0.054 | -0.193 | yes | no | 0.950 | 0.952 |
| 1 | cifar10_automobilevstruck | all_bn_train | 0.792 | 0.034 | -0.019 | no | yes | 0.923 | 0.927 |
| 1 | cifar10_automobilevstruck | all_bn_eval | 0.735 | -0.051 | -0.187 | yes | no | 0.960 | 0.956 |
| 1 | cifar10_automobilevstruck | all_bn_train_aug | 0.804 | 0.030 | -0.052 | no | yes | 0.910 | 0.927 |
| 1 | cifar10_automobilevstruck | all_bn_eval_aug | 0.922 | -0.074 | -0.218 | yes | no | 0.965 | 0.963 |
| 1 | cifar10_catvsdog | layer4_base | 0.599 | -0.036 | -0.133 | yes | no | 0.835 | 0.833 |
| 1 | cifar10_catvsdog | all_bn_train | 0.845 | 0.024 | -0.027 | no | yes | 0.812 | 0.781 |
| 1 | cifar10_catvsdog | all_bn_eval | 0.686 | -0.075 | -0.189 | yes | no | 0.875 | 0.875 |
| 1 | cifar10_catvsdog | all_bn_train_aug | 0.840 | 0.010 | -0.034 | no | yes | 0.810 | 0.794 |
| 1 | cifar10_catvsdog | all_bn_eval_aug | 0.628 | -0.070 | -0.161 | yes | no | 0.873 | 0.867 |
| 1 | cifar10_vehicles4 | layer4_base | 0.652 | -0.039 | -0.119 | yes | no | 0.906 | 0.910 |
| 1 | cifar10_vehicles4 | all_bn_train | 0.740 | 0.065 | 0.120 | no | yes | 0.827 | 0.844 |
| 1 | cifar10_vehicles4 | all_bn_eval | 0.718 | -0.053 | -0.126 | yes | no | 0.904 | 0.917 |
| 1 | cifar10_vehicles4 | all_bn_train_aug | 0.740 | 0.047 | 0.099 | no | yes | 0.835 | 0.863 |
| 1 | cifar10_vehicles4 | all_bn_eval_aug | 0.697 | -0.056 | -0.115 | yes | no | 0.892 | 0.917 |
| 2 | cifar10_automobilevstruck | layer4_base | 0.748 | -0.041 | -0.180 | yes | no | 0.950 | 0.942 |
| 2 | cifar10_automobilevstruck | all_bn_train | 0.793 | 0.020 | -0.014 | no | yes | 0.910 | 0.894 |
| 2 | cifar10_automobilevstruck | all_bn_eval | 0.807 | -0.053 | -0.155 | yes | no | 0.942 | 0.944 |
| 2 | cifar10_automobilevstruck | all_bn_train_aug | 0.809 | 0.021 | -0.015 | no | yes | 0.912 | 0.892 |
| 2 | cifar10_automobilevstruck | all_bn_eval_aug | 0.874 | -0.057 | -0.169 | yes | no | 0.921 | 0.938 |
| 2 | cifar10_catvsdog | layer4_base | 0.581 | -0.040 | -0.111 | yes | no | 0.825 | 0.833 |
| 2 | cifar10_catvsdog | all_bn_train | 0.827 | 0.011 | 0.002 | no | yes | 0.804 | 0.808 |
| 2 | cifar10_catvsdog | all_bn_eval | 0.691 | -0.037 | -0.130 | yes | no | 0.840 | 0.842 |
| 2 | cifar10_catvsdog | all_bn_train_aug | 0.834 | -0.010 | -0.019 | yes | no | 0.798 | 0.821 |
| 2 | cifar10_catvsdog | all_bn_eval_aug | 0.737 | -0.075 | -0.153 | yes | no | 0.835 | 0.848 |
| 2 | cifar10_vehicles4 | layer4_base | 0.641 | -0.030 | -0.114 | yes | no | 0.925 | 0.900 |
| 2 | cifar10_vehicles4 | all_bn_train | 0.742 | 0.061 | 0.072 | no | yes | 0.850 | 0.844 |
| 2 | cifar10_vehicles4 | all_bn_eval | 0.722 | -0.057 | -0.148 | yes | no | 0.931 | 0.935 |
| 2 | cifar10_vehicles4 | all_bn_train_aug | 0.732 | 0.058 | 0.070 | no | yes | 0.848 | 0.856 |
| 2 | cifar10_vehicles4 | all_bn_eval_aug | 0.747 | -0.055 | -0.150 | yes | no | 0.921 | 0.948 |

## Artifacts

- `metrics.json`
- `figures/tail_delta_by_variant.png`
- `figures/graph_delta_by_variant.png`
- `figures/movement_vs_tail_delta.png`
