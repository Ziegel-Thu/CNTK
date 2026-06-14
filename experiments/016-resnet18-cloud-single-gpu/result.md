# Result

## Run

Command:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n cntk python experiments/016-resnet18-cloud-single-gpu/scripts/run_cloud_single_gpu.py --seeds 0 1 2 3 4 --binary-n-per-class 120 --multi-n-per-class 60 --epochs 6 --batch-size 64 --eval-batch-size 64 --device cuda
```

## Summary

| regime | n | movement | tail delta | graph delta | repair rate | overmove rate | head acc delta | ridge acc delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen_head | 15 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.00 | 0.00 | 0.428 | 0.000 |
| finetune_layer4 | 15 | 0.479 +/- 0.033 | -0.030 +/- 0.021 | -0.094 +/- 0.023 | 1.00 | 0.00 | 0.472 | 0.004 |
| finetune_all | 15 | 0.689 +/- 0.019 | 0.041 +/- 0.046 | 0.084 +/- 0.068 | 0.20 | 0.80 | 0.370 | -0.033 |

Interpretation:

- This is the first single-GPU cloud scale-up of the ResNet18 fine-tuning
  dynamics probe.
- `frozen_head` remains the no-metric-movement control.
- `finetune_layer4` gives a robust repair signal: every seed-task row has
  negative tail and graph deltas.
- `finetune_all` still moves more but usually worsens held-out geometry, so the
  over-move warning from `013` and `015` survives the larger run.
- The next useful cloud step is not merely more of the same; it should test
  stronger augmentation/schedules and non-ResNet18 backbones.

## Per-Seed Final Rows

| seed | dataset | regime | movement | tail delta | graph delta | repair | overmove | head acc final | ridge acc final |
| ---: | --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| 0 | cifar10_automobilevstruck | frozen_head | 0.000 | 0.000 | 0.000 | no | no | 0.908 | 0.950 |
| 0 | cifar10_automobilevstruck | finetune_layer4 | 0.544 | -0.022 | -0.129 | yes | no | 0.942 | 0.938 |
| 0 | cifar10_automobilevstruck | finetune_all | 0.709 | 0.061 | 0.085 | no | yes | 0.842 | 0.904 |
| 0 | cifar10_catvsdog | frozen_head | 0.000 | 0.000 | 0.000 | no | no | 0.762 | 0.825 |
| 0 | cifar10_catvsdog | finetune_layer4 | 0.472 | -0.048 | -0.072 | yes | no | 0.812 | 0.804 |
| 0 | cifar10_catvsdog | finetune_all | 0.701 | -0.053 | -0.034 | yes | no | 0.787 | 0.821 |
| 0 | cifar10_vehicles4 | frozen_head | 0.000 | 0.000 | 0.000 | no | no | 0.846 | 0.867 |
| 0 | cifar10_vehicles4 | finetune_layer4 | 0.459 | -0.015 | -0.103 | yes | no | 0.896 | 0.900 |
| 0 | cifar10_vehicles4 | finetune_all | 0.644 | 0.079 | 0.099 | no | yes | 0.750 | 0.833 |
| 1 | cifar10_automobilevstruck | frozen_head | 0.000 | 0.000 | 0.000 | no | no | 0.887 | 0.921 |
| 1 | cifar10_automobilevstruck | finetune_layer4 | 0.509 | -0.038 | -0.122 | yes | no | 0.938 | 0.938 |
| 1 | cifar10_automobilevstruck | finetune_all | 0.699 | 0.058 | 0.126 | no | yes | 0.846 | 0.871 |
| 1 | cifar10_catvsdog | frozen_head | 0.000 | 0.000 | 0.000 | no | no | 0.750 | 0.800 |
| 1 | cifar10_catvsdog | finetune_layer4 | 0.461 | -0.036 | -0.078 | yes | no | 0.821 | 0.792 |
| 1 | cifar10_catvsdog | finetune_all | 0.708 | -0.016 | 0.042 | no | yes | 0.700 | 0.796 |
| 1 | cifar10_vehicles4 | frozen_head | 0.000 | 0.000 | 0.000 | no | no | 0.842 | 0.879 |
| 1 | cifar10_vehicles4 | finetune_layer4 | 0.457 | -0.037 | -0.086 | yes | no | 0.900 | 0.867 |
| 1 | cifar10_vehicles4 | finetune_all | 0.652 | 0.101 | 0.168 | no | yes | 0.696 | 0.808 |
| 2 | cifar10_automobilevstruck | frozen_head | 0.000 | 0.000 | 0.000 | no | no | 0.854 | 0.950 |
| 2 | cifar10_automobilevstruck | finetune_layer4 | 0.521 | -0.008 | -0.115 | yes | no | 0.917 | 0.912 |
| 2 | cifar10_automobilevstruck | finetune_all | 0.698 | 0.069 | 0.141 | no | yes | 0.883 | 0.879 |
| 2 | cifar10_catvsdog | frozen_head | 0.000 | 0.000 | 0.000 | no | no | 0.808 | 0.812 |
| 2 | cifar10_catvsdog | finetune_layer4 | 0.457 | -0.026 | -0.054 | yes | no | 0.796 | 0.800 |
| 2 | cifar10_catvsdog | finetune_all | 0.707 | 0.021 | 0.044 | no | yes | 0.800 | 0.775 |
| 2 | cifar10_vehicles4 | frozen_head | 0.000 | 0.000 | 0.000 | no | no | 0.787 | 0.833 |
| 2 | cifar10_vehicles4 | finetune_layer4 | 0.458 | -0.019 | -0.078 | yes | no | 0.842 | 0.833 |
| 2 | cifar10_vehicles4 | finetune_all | 0.678 | 0.064 | 0.113 | no | yes | 0.746 | 0.792 |
| 3 | cifar10_automobilevstruck | frozen_head | 0.000 | 0.000 | 0.000 | no | no | 0.892 | 0.912 |
| 3 | cifar10_automobilevstruck | finetune_layer4 | 0.531 | -0.012 | -0.093 | yes | no | 0.908 | 0.908 |
| 3 | cifar10_automobilevstruck | finetune_all | 0.699 | 0.059 | 0.184 | no | yes | 0.829 | 0.879 |
| 3 | cifar10_catvsdog | frozen_head | 0.000 | 0.000 | 0.000 | no | no | 0.762 | 0.746 |
| 3 | cifar10_catvsdog | finetune_layer4 | 0.462 | -0.063 | -0.111 | yes | no | 0.800 | 0.796 |
| 3 | cifar10_catvsdog | finetune_all | 0.684 | -0.033 | -0.012 | yes | no | 0.725 | 0.804 |
| 3 | cifar10_vehicles4 | frozen_head | 0.000 | 0.000 | 0.000 | no | no | 0.796 | 0.850 |
| 3 | cifar10_vehicles4 | finetune_layer4 | 0.427 | -0.000 | -0.072 | yes | no | 0.854 | 0.858 |
| 3 | cifar10_vehicles4 | finetune_all | 0.682 | 0.101 | 0.171 | no | yes | 0.717 | 0.800 |
| 4 | cifar10_automobilevstruck | frozen_head | 0.000 | 0.000 | 0.000 | no | no | 0.804 | 0.892 |
| 4 | cifar10_automobilevstruck | finetune_layer4 | 0.508 | -0.036 | -0.121 | yes | no | 0.925 | 0.917 |
| 4 | cifar10_automobilevstruck | finetune_all | 0.709 | 0.055 | 0.044 | no | yes | 0.887 | 0.883 |
| 4 | cifar10_catvsdog | frozen_head | 0.000 | 0.000 | 0.000 | no | no | 0.783 | 0.800 |
| 4 | cifar10_catvsdog | finetune_layer4 | 0.458 | -0.080 | -0.111 | yes | no | 0.838 | 0.833 |
| 4 | cifar10_catvsdog | finetune_all | 0.686 | -0.004 | -0.018 | yes | no | 0.733 | 0.808 |
| 4 | cifar10_vehicles4 | frozen_head | 0.000 | 0.000 | 0.000 | no | no | 0.879 | 0.887 |
| 4 | cifar10_vehicles4 | finetune_layer4 | 0.457 | -0.012 | -0.062 | yes | no | 0.887 | 0.887 |
| 4 | cifar10_vehicles4 | finetune_all | 0.674 | 0.050 | 0.102 | no | yes | 0.750 | 0.775 |

## Artifacts

- `metrics.json`
- `figures/tail_delta_by_regime.png`
- `figures/graph_delta_by_regime.png`
- `figures/movement_vs_tail_delta.png`
