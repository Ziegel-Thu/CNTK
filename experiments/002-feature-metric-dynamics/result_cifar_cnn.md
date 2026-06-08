# CIFAR-10 Small-CNN Feature-Dynamics Result

## Run

Command: `python experiments/002-feature-metric-dynamics/scripts/run_cifar_cnn.py --n-per-class 120 --epochs 80 --seed 0 --device cpu`

## Summary

Main observations:

- Small-CNN feature learning transfers better than raw-pixel MLP feature learning.
- `cat vs dog` remains hard, but feature learning improves test geometry and
  accuracy:
  - test `tail@10%`: `0.861 -> 0.834`
  - test opposite-kNN ratio: final `0.456`
  - test accuracy: `0.604`
- `automobile vs truck` shows a much cleaner feature-learning signal:
  - test `tail@10%`: `0.751 -> 0.631`
  - test opposite-kNN ratio: final `0.401`
  - test accuracy: `0.729`
- Lazy small-LR CNN has near-zero feature movement and does not improve geometry.
- Compared with the raw MLP run, CNN inductive bias appears necessary for
  transferable CIFAR metric adaptation.

| dataset | regime | train tail start | train tail final | test tail start | test tail final | train mix final | test mix final | movement final | test acc |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cifar10_catvsdog_n120_train | frozen_random | 0.818 | 0.818 | 0.874 | 0.874 | 0.473 | 0.503 | 0.000 | 0.492 |
| cifar10_catvsdog_n120_train | feature_learning | 0.795 | 0.521 | 0.861 | 0.834 | 0.320 | 0.456 | 0.097 | 0.604 |
| cifar10_catvsdog_n120_train | lazy_small_lr | 0.807 | 0.808 | 0.837 | 0.837 | 0.463 | 0.490 | 0.000 | 0.488 |
| cifar10_automobilevstruck_n120_train | frozen_random | 0.707 | 0.707 | 0.726 | 0.726 | 0.462 | 0.482 | 0.000 | 0.562 |
| cifar10_automobilevstruck_n120_train | feature_learning | 0.704 | 0.421 | 0.751 | 0.631 | 0.251 | 0.401 | 0.058 | 0.729 |
| cifar10_automobilevstruck_n120_train | lazy_small_lr | 0.671 | 0.671 | 0.735 | 0.735 | 0.477 | 0.463 | 0.000 | 0.500 |

## Artifacts

- `metrics_cifar_cnn_over_time.json`
- `figures/cifar_cnn_tail_over_time.png`
- `figures/cifar_cnn_test_tail_over_time.png`
- `figures/cifar_cnn_mixing_over_time.png`
- `figures/cifar_cnn_test_mixing_over_time.png`
- `figures/cifar_cnn_kernel_movement_over_time.png`
- `figures/cifar_cnn_accuracy_over_time.png`
