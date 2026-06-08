# CIFAR-10 Feature-Dynamics Result

## Run

Command: `python experiments/002-feature-metric-dynamics/scripts/run_cifar.py --n-per-class 120 --epochs 120 --seed 0 --device cpu`

## Summary

Main observations:

- Raw-pixel MLP feature learning collapses train tail but mostly fails to improve
  test geometry.
  - `cat vs dog`: train `tail@10%` drops from `0.830` to `0.041`, but test tail
    stays `0.886 -> 0.881`, and test mixing remains high.
  - `automobile vs truck`: train `tail@10%` drops from `0.753` to `0.024`, while
    test tail only partially improves `0.785 -> 0.727`.
- This looks like memorization on CIFAR raw pixels rather than transferable
  metric adaptation.
- The next CIFAR feature-dynamics run should use a small CNN, not only an MLP on
  flattened pixels.

| dataset | regime | train tail start | train tail final | test tail start | test tail final | train mix final | test mix final | movement final | test acc |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cifar10_catvsdog_n120_train | frozen_random | 0.852 | 0.852 | 0.882 | 0.882 | 0.458 | 0.479 | 0.000 | 0.642 |
| cifar10_catvsdog_n120_train | feature_learning | 0.830 | 0.041 | 0.886 | 0.881 | 0.000 | 0.492 | 0.442 | 0.575 |
| cifar10_catvsdog_n120_train | lazy_wide_small_lr | 0.781 | 0.776 | 0.858 | 0.855 | 0.440 | 0.477 | 0.002 | 0.550 |
| cifar10_automobilevstruck_n120_train | frozen_random | 0.723 | 0.723 | 0.802 | 0.802 | 0.437 | 0.432 | 0.000 | 0.633 |
| cifar10_automobilevstruck_n120_train | feature_learning | 0.753 | 0.024 | 0.785 | 0.727 | 0.000 | 0.420 | 0.521 | 0.600 |
| cifar10_automobilevstruck_n120_train | lazy_wide_small_lr | 0.704 | 0.695 | 0.755 | 0.754 | 0.423 | 0.431 | 0.005 | 0.642 |

## Interpretation Prompts

- Does feature learning reduce test tail on CIFAR, or mainly memorize train geometry?
- Is `cat vs dog` harder than `automobile vs truck` in metric-dynamics terms?
- Does a raw-pixel MLP provide enough inductive bias, or should the next CIFAR run use a small CNN?

## Artifacts

- `metrics_cifar_over_time.json`
- `figures/cifar_tail_over_time.png`
- `figures/cifar_test_tail_over_time.png`
- `figures/cifar_mixing_over_time.png`
- `figures/cifar_test_mixing_over_time.png`
- `figures/cifar_kernel_movement_over_time.png`
- `figures/cifar_accuracy_over_time.png`
