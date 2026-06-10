# Result

## Run

Command: `python3 experiments/013-pretrained-finetune-metric-dynamics/scripts/run_finetune_dynamics.py --binary-n-per-class 50 --multi-n-per-class 25 --epochs 6 --batch-size 16 --eval-batch-size 32 --device auto`

## Summary

- corr(final test movement, test tail decrease) = `-0.516`
- corr(final test tail, ridge test margin) = `-0.802`
- corr(final test tail, head test accuracy) = `-0.901`
- corr(final test tail, ridge test accuracy) = `-0.888`

Regime means over tasks:

| regime | movement | tail delta | graph delta | head acc delta | ridge acc delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| finetune_all | 0.649 | 0.121 | 0.144 | 0.343 | -0.147 |
| finetune_layer4 | 0.431 | -0.024 | -0.072 | 0.420 | -0.030 |
| frozen_head | 0.000 | 0.000 | 0.000 | 0.407 | 0.000 |

Interpretation:

- `frozen_head` is the control: head margin can change while the feature
  metric should not move.
- A fine-tuning regime supports metric adaptation only if held-out
  movement is paired with lower test tail/graph roughness and better
  margins/accuracy.
- In the default run, `finetune_layer4` is the cleanest metric-repair
  regime: test movement is moderate and mean tail/graph deltas are
  negative.
- `finetune_all` is an important negative control: it moves the metric
  more, but mean held-out tail and graph roughness increase.
- If movement is large but test tail does not decrease, treat it as weak
  transfer rather than a positive feature-learning result.

| dataset | regime | movement | test tail final | test tail delta | graph delta | mix delta | head acc final | ridge acc final | ridge margin final |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cifar10_automobilevstruck | finetune_all | 0.656 | 0.330 | 0.037 | 0.172 | 0.080 | 0.860 | 0.880 | 0.467 |
| cifar10_automobilevstruck | finetune_layer4 | 0.456 | 0.255 | -0.038 | -0.056 | -0.026 | 0.910 | 0.900 | 0.560 |
| cifar10_automobilevstruck | frozen_head | 0.000 | 0.293 | 0.000 | 0.000 | 0.000 | 0.860 | 0.910 | 0.535 |
| cifar10_catvsdog | finetune_all | 0.649 | 0.607 | 0.227 | 0.133 | 0.065 | 0.680 | 0.680 | 0.251 |
| cifar10_catvsdog | finetune_layer4 | 0.401 | 0.370 | -0.009 | -0.065 | -0.041 | 0.750 | 0.810 | 0.373 |
| cifar10_catvsdog | frozen_head | 0.000 | 0.380 | 0.000 | 0.000 | 0.000 | 0.770 | 0.870 | 0.549 |
| cifar10_vehicles4 | finetune_all | 0.642 | 0.507 | 0.098 | 0.125 | 0.089 | 0.640 | 0.620 | 0.178 |
| cifar10_vehicles4 | finetune_layer4 | 0.437 | 0.384 | -0.025 | -0.094 | -0.069 | 0.810 | 0.820 | 0.290 |
| cifar10_vehicles4 | frozen_head | 0.000 | 0.409 | 0.000 | 0.000 | 0.000 | 0.780 | 0.840 | 0.295 |

## Artifacts

- `metrics.json`
- `figures/test_tail_over_time.png`
- `figures/test_graph_dirichlet_over_time.png`
- `figures/test_mixing_over_time.png`
- `figures/test_head_margin_over_time.png`
- `figures/test_ridge_margin_over_time.png`
- `figures/kernel_movement_vs_tail_decrease.png`
