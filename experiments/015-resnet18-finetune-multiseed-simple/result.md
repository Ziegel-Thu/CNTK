# Result

## Run

Command: `python3 experiments/015-resnet18-finetune-multiseed-simple/scripts/run_multiseed.py --seeds 0 1 2 --binary-n-per-class 60 --multi-n-per-class 30 --epochs 4 --batch-size 16 --eval-batch-size 32 --device auto`

## Summary

| regime | n | movement | tail delta | graph delta | repair rate | overmove rate | head acc delta | ridge acc delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen_head | 9 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.00 | 0.00 | 0.407 | 0.000 |
| finetune_layer4 | 9 | 0.422 +/- 0.022 | -0.020 +/- 0.028 | -0.048 +/- 0.041 | 0.67 | 0.33 | 0.440 | -0.001 |
| finetune_all | 9 | 0.663 +/- 0.017 | 0.064 +/- 0.064 | 0.131 +/- 0.075 | 0.11 | 0.89 | 0.359 | -0.065 |

Interpretation:

- This is a simple local multi-seed probe, not the final cloud-scale
  fine-tuning experiment.
- `frozen_head` should remain the no-metric-movement control.
- A robust metric-repair signal should show negative mean tail/graph
  deltas and a nontrivial repair rate.
- An overmove signal means the backbone moved but held-out geometry got
  rougher.

Direct read:

- `finetune_layer4` preserves the qualitative `013` result: it moves the metric
  moderately and repairs held-out geometry in `6/9` seed-task rows.
- `finetune_all` is the sharper warning case: it moves more than `layer4` but
  worsens tail or graph roughness in `8/9` rows.
- This supports using cloud compute for scale and backbone diversity, not for
  simply checking whether the small local signal exists.

## Per-Seed Final Rows

| seed | dataset | regime | movement | tail delta | graph delta | repair | overmove | head acc final | ridge acc final |
| ---: | --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| 0 | cifar10_automobilevstruck | frozen_head | 0.000 | 0.000 | 0.000 | no | no | 0.908 | 0.917 |
| 0 | cifar10_automobilevstruck | finetune_layer4 | 0.458 | -0.020 | -0.122 | yes | no | 0.958 | 0.933 |
| 0 | cifar10_automobilevstruck | finetune_all | 0.677 | 0.121 | 0.229 | no | yes | 0.842 | 0.825 |
| 0 | cifar10_catvsdog | frozen_head | 0.000 | 0.000 | 0.000 | no | no | 0.783 | 0.833 |
| 0 | cifar10_catvsdog | finetune_layer4 | 0.416 | -0.057 | -0.118 | yes | no | 0.842 | 0.833 |
| 0 | cifar10_catvsdog | finetune_all | 0.667 | -0.079 | -0.033 | yes | no | 0.792 | 0.850 |
| 0 | cifar10_vehicles4 | frozen_head | 0.000 | 0.000 | 0.000 | no | no | 0.783 | 0.850 |
| 0 | cifar10_vehicles4 | finetune_layer4 | 0.399 | 0.006 | -0.020 | no | yes | 0.825 | 0.842 |
| 0 | cifar10_vehicles4 | finetune_all | 0.661 | 0.089 | 0.153 | no | yes | 0.667 | 0.750 |
| 1 | cifar10_automobilevstruck | frozen_head | 0.000 | 0.000 | 0.000 | no | no | 0.875 | 0.858 |
| 1 | cifar10_automobilevstruck | finetune_layer4 | 0.445 | -0.009 | -0.038 | yes | no | 0.867 | 0.858 |
| 1 | cifar10_automobilevstruck | finetune_all | 0.670 | 0.091 | 0.163 | no | yes | 0.800 | 0.792 |
| 1 | cifar10_catvsdog | frozen_head | 0.000 | 0.000 | 0.000 | no | no | 0.742 | 0.750 |
| 1 | cifar10_catvsdog | finetune_layer4 | 0.412 | -0.037 | -0.054 | yes | no | 0.808 | 0.808 |
| 1 | cifar10_catvsdog | finetune_all | 0.659 | 0.027 | 0.067 | no | yes | 0.767 | 0.733 |
| 1 | cifar10_vehicles4 | frozen_head | 0.000 | 0.000 | 0.000 | no | no | 0.783 | 0.900 |
| 1 | cifar10_vehicles4 | finetune_layer4 | 0.392 | -0.035 | -0.032 | yes | no | 0.817 | 0.850 |
| 1 | cifar10_vehicles4 | finetune_all | 0.625 | 0.154 | 0.163 | no | yes | 0.617 | 0.758 |
| 2 | cifar10_automobilevstruck | frozen_head | 0.000 | 0.000 | 0.000 | no | no | 0.942 | 0.933 |
| 2 | cifar10_automobilevstruck | finetune_layer4 | 0.442 | 0.024 | -0.005 | no | yes | 0.900 | 0.892 |
| 2 | cifar10_automobilevstruck | finetune_all | 0.683 | 0.086 | 0.212 | no | yes | 0.825 | 0.817 |
| 2 | cifar10_catvsdog | frozen_head | 0.000 | 0.000 | 0.000 | no | no | 0.800 | 0.767 |
| 2 | cifar10_catvsdog | finetune_layer4 | 0.432 | -0.062 | -0.033 | yes | no | 0.800 | 0.767 |
| 2 | cifar10_catvsdog | finetune_all | 0.679 | 0.020 | 0.111 | no | yes | 0.800 | 0.750 |
| 2 | cifar10_vehicles4 | frozen_head | 0.000 | 0.000 | 0.000 | no | no | 0.775 | 0.833 |
| 2 | cifar10_vehicles4 | finetune_layer4 | 0.404 | 0.007 | -0.011 | no | yes | 0.825 | 0.850 |
| 2 | cifar10_vehicles4 | finetune_all | 0.644 | 0.064 | 0.116 | no | yes | 0.725 | 0.783 |

## Artifacts

- `metrics.json`
- `figures/tail_delta_by_regime.png`
- `figures/graph_delta_by_regime.png`
- `figures/movement_vs_tail_delta.png`
