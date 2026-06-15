# Result

## Run

Command:

```bash
python3 experiments/020-vit-finetune-metric-dynamics/scripts/run_vit_dynamics.py --seeds 0 1 2 --binary-n-per-class 120 --multi-n-per-class 60 --epochs 4 --batch-size 64 --eval-batch-size 64 --device cuda
```

## Summary

| variant | n | movement | tail delta | graph delta | repair rate | overmove rate | head acc delta | ridge acc delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen_head | 9 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.00 | 0.00 | 0.522 | 0.000 |
| finetune_lastblock | 9 | 0.030 +/- 0.007 | -0.001 +/- 0.000 | -0.004 +/- 0.003 | 1.00 | 0.00 | 0.449 | 0.001 |
| finetune_all | 9 | 0.227 +/- 0.056 | -0.023 +/- 0.006 | -0.043 +/- 0.011 | 1.00 | 0.00 | 0.545 | 0.011 |
| finetune_all_aug | 9 | 0.225 +/- 0.053 | -0.026 +/- 0.005 | -0.043 +/- 0.012 | 1.00 | 0.00 | 0.551 | 0.010 |

Interpretation:

- ViT-B/16 has LayerNorm and no BatchNorm, so it should not reproduce
  ResNet18's BN running-stat failure mode directly.
- A full fine-tuning variant supports useful metric dynamics only if
  held-out tail and graph roughness fall, not merely if head accuracy
  improves.
- If ViT full fine-tuning still overmoves, the mechanism is not BN stats
  and needs a new control.

## Per-Seed Final Rows

| seed | dataset | variant | movement | tail delta | graph delta | repair | overmove | head acc final | ridge acc final |
| ---: | --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| 0 | cifar10_automobilevstruck | frozen_head | 0.000 | 0.000 | 0.000 | no | no | 0.971 | 0.975 |
| 0 | cifar10_automobilevstruck | finetune_lastblock | 0.041 | -0.001 | -0.005 | yes | no | 0.958 | 0.971 |
| 0 | cifar10_automobilevstruck | finetune_all | 0.278 | -0.021 | -0.037 | yes | no | 0.971 | 0.979 |
| 0 | cifar10_automobilevstruck | finetune_all_aug | 0.345 | -0.025 | -0.038 | yes | no | 0.983 | 0.979 |
| 0 | cifar10_catvsdog | frozen_head | 0.000 | 0.000 | 0.000 | no | no | 0.921 | 0.900 |
| 0 | cifar10_catvsdog | finetune_lastblock | 0.029 | -0.002 | -0.011 | yes | no | 0.921 | 0.900 |
| 0 | cifar10_catvsdog | finetune_all | 0.236 | -0.021 | -0.063 | yes | no | 0.921 | 0.921 |
| 0 | cifar10_catvsdog | finetune_all_aug | 0.220 | -0.028 | -0.068 | yes | no | 0.933 | 0.921 |
| 0 | cifar10_vehicles4 | frozen_head | 0.000 | 0.000 | 0.000 | no | no | 0.950 | 0.954 |
| 0 | cifar10_vehicles4 | finetune_lastblock | 0.027 | -0.001 | -0.003 | yes | no | 0.933 | 0.954 |
| 0 | cifar10_vehicles4 | finetune_all | 0.156 | -0.011 | -0.025 | yes | no | 0.938 | 0.942 |
| 0 | cifar10_vehicles4 | finetune_all_aug | 0.174 | -0.020 | -0.030 | yes | no | 0.946 | 0.954 |
| 1 | cifar10_automobilevstruck | frozen_head | 0.000 | 0.000 | 0.000 | no | no | 0.967 | 0.983 |
| 1 | cifar10_automobilevstruck | finetune_lastblock | 0.043 | -0.002 | -0.008 | yes | no | 0.975 | 0.983 |
| 1 | cifar10_automobilevstruck | finetune_all | 0.241 | -0.019 | -0.036 | yes | no | 0.988 | 0.975 |
| 1 | cifar10_automobilevstruck | finetune_all_aug | 0.258 | -0.032 | -0.043 | yes | no | 0.988 | 0.983 |
| 1 | cifar10_catvsdog | frozen_head | 0.000 | 0.000 | 0.000 | no | no | 0.938 | 0.904 |
| 1 | cifar10_catvsdog | finetune_lastblock | 0.028 | -0.001 | -0.000 | yes | no | 0.917 | 0.904 |
| 1 | cifar10_catvsdog | finetune_all | 0.220 | -0.025 | -0.049 | yes | no | 0.925 | 0.929 |
| 1 | cifar10_catvsdog | finetune_all_aug | 0.166 | -0.032 | -0.032 | yes | no | 0.933 | 0.912 |
| 1 | cifar10_vehicles4 | frozen_head | 0.000 | 0.000 | 0.000 | no | no | 0.975 | 0.963 |
| 1 | cifar10_vehicles4 | finetune_lastblock | 0.028 | -0.001 | -0.001 | yes | no | 0.958 | 0.971 |
| 1 | cifar10_vehicles4 | finetune_all | 0.185 | -0.019 | -0.035 | yes | no | 0.971 | 0.979 |
| 1 | cifar10_vehicles4 | finetune_all_aug | 0.181 | -0.020 | -0.040 | yes | no | 0.971 | 0.975 |
| 2 | cifar10_automobilevstruck | frozen_head | 0.000 | 0.000 | 0.000 | no | no | 0.971 | 0.975 |
| 2 | cifar10_automobilevstruck | finetune_lastblock | 0.026 | -0.002 | -0.002 | yes | no | 0.967 | 0.979 |
| 2 | cifar10_automobilevstruck | finetune_all | 0.351 | -0.034 | -0.057 | yes | no | 0.971 | 0.988 |
| 2 | cifar10_automobilevstruck | finetune_all_aug | 0.260 | -0.020 | -0.033 | yes | no | 0.958 | 0.988 |
| 2 | cifar10_catvsdog | frozen_head | 0.000 | 0.000 | 0.000 | no | no | 0.925 | 0.925 |
| 2 | cifar10_catvsdog | finetune_lastblock | 0.027 | -0.001 | -0.003 | yes | no | 0.954 | 0.925 |
| 2 | cifar10_catvsdog | finetune_all | 0.173 | -0.022 | -0.041 | yes | no | 0.942 | 0.942 |
| 2 | cifar10_catvsdog | finetune_all_aug | 0.201 | -0.028 | -0.048 | yes | no | 0.963 | 0.933 |
| 2 | cifar10_vehicles4 | frozen_head | 0.000 | 0.000 | 0.000 | no | no | 0.921 | 0.938 |
| 2 | cifar10_vehicles4 | finetune_lastblock | 0.021 | -0.001 | -0.001 | yes | no | 0.925 | 0.938 |
| 2 | cifar10_vehicles4 | finetune_all | 0.207 | -0.031 | -0.047 | yes | no | 0.946 | 0.963 |
| 2 | cifar10_vehicles4 | finetune_all_aug | 0.217 | -0.031 | -0.054 | yes | no | 0.942 | 0.958 |

## Artifacts

- `metrics.json`
- `figures/tail_delta_by_variant.png`
- `figures/graph_delta_by_variant.png`
- `figures/movement_vs_tail_delta.png`
