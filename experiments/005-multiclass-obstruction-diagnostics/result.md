# Result

## Run

Command: `python3 experiments/005-multiclass-obstruction-diagnostics/scripts/run_multiclass.py --n-per-class 40 --mlp-epochs 80 --cnn-epochs 60 --probe-epochs 250 --k-neighbors 10 --seed 0 --device cpu`

## Summary

- corr(test kNN disagreement, test multiclass tail@10%) = `0.960`
- corr(test normalized local entropy, test multiclass tail@10%) = `0.911`
- corr(test multiclass tail@10%, linear probe test acc) = `-0.925`

Interpretation:

- The binary obstruction signal extends if disagreement/entropy remain
  positively correlated with multiclass label-subspace tail.
- A representation should be treated as transferable only when the test
  tail/entropy improve together with probe accuracy.
- A train-only improvement is a memorization signal, not evidence that the
  fixed test metric improved.

Run-specific observations:

- MNIST all-10 and hard-5 both show transferable improvement from trained MLP
  features: all-10 test tail drops from raw `0.410` to `0.282`, and hard-5 test
  tail drops from raw `0.465` to `0.324`.
- CIFAR multiclass remains high-tail/high-entropy for these small CPU-trained
  models. Trained CNN features improve CIFAR all-10 tail (`0.816 -> 0.727` vs
  raw pixels) but test accuracy remains low (`0.363`).
- CIFAR vehicles expose a useful caveat: random CNN probe accuracy (`0.494`) is
  higher than the short trained-CNN run (`0.444`). Tail/mixing are useful
  diagnostics, but short multiclass CNN training needs stronger schedules/seeds
  before architecture-level claims.

| dataset | representation | classes | dim | test tail@10% | test disagreement | test entropy | alignment | probe train acc | probe test acc |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cifar10_all10 | random_cnn | 10 | 128 | 0.758 | 0.812 | 0.678 | 0.070 | 0.945 | 0.335 |
| cifar10_all10 | random_mlp | 10 | 256 | 0.809 | 0.810 | 0.655 | 0.113 | 1.000 | 0.225 |
| cifar10_all10 | raw_pixels | 10 | 3072 | 0.816 | 0.799 | 0.627 | 0.106 | 1.000 | 0.237 |
| cifar10_all10 | rff_512 | 10 | 512 | 0.805 | 0.819 | 0.593 | 0.114 | 1.000 | 0.295 |
| cifar10_all10 | trained_cnn | 10 | 128 | 0.727 | 0.766 | 0.611 | 0.190 | 0.663 | 0.363 |
| cifar10_all10 | trained_mlp | 10 | 256 | 0.775 | 0.793 | 0.634 | 0.145 | 1.000 | 0.327 |
| cifar10_animals6 | random_cnn | 6 | 128 | 0.782 | 0.779 | 0.780 | 0.060 | 0.917 | 0.363 |
| cifar10_animals6 | random_mlp | 6 | 256 | 0.842 | 0.781 | 0.762 | 0.089 | 1.000 | 0.283 |
| cifar10_animals6 | raw_pixels | 6 | 3072 | 0.821 | 0.766 | 0.735 | 0.071 | 1.000 | 0.279 |
| cifar10_animals6 | rff_512 | 6 | 512 | 0.838 | 0.799 | 0.765 | 0.086 | 1.000 | 0.275 |
| cifar10_animals6 | trained_cnn | 6 | 128 | 0.750 | 0.747 | 0.719 | 0.139 | 0.750 | 0.458 |
| cifar10_animals6 | trained_mlp | 6 | 256 | 0.810 | 0.775 | 0.746 | 0.091 | 1.000 | 0.325 |
| cifar10_vehicles4 | random_cnn | 4 | 128 | 0.770 | 0.686 | 0.811 | 0.082 | 1.000 | 0.494 |
| cifar10_vehicles4 | random_mlp | 4 | 256 | 0.814 | 0.678 | 0.764 | 0.118 | 1.000 | 0.463 |
| cifar10_vehicles4 | raw_pixels | 4 | 3072 | 0.777 | 0.647 | 0.749 | 0.131 | 1.000 | 0.463 |
| cifar10_vehicles4 | rff_512 | 4 | 512 | 0.770 | 0.649 | 0.768 | 0.135 | 1.000 | 0.481 |
| cifar10_vehicles4 | trained_cnn | 4 | 128 | 0.759 | 0.671 | 0.780 | 0.157 | 0.837 | 0.444 |
| cifar10_vehicles4 | trained_mlp | 4 | 256 | 0.740 | 0.649 | 0.730 | 0.174 | 1.000 | 0.469 |
| mnist_all10 | random_mlp | 10 | 256 | 0.487 | 0.433 | 0.361 | 0.377 | 1.000 | 0.728 |
| mnist_all10 | raw_pixels | 10 | 784 | 0.410 | 0.363 | 0.302 | 0.434 | 1.000 | 0.795 |
| mnist_all10 | rff_512 | 10 | 512 | 0.451 | 0.455 | 0.366 | 0.320 | 1.000 | 0.808 |
| mnist_all10 | trained_mlp | 10 | 256 | 0.282 | 0.219 | 0.151 | 0.567 | 1.000 | 0.830 |
| mnist_hard5 | random_mlp | 5 | 256 | 0.618 | 0.531 | 0.643 | 0.243 | 1.000 | 0.635 |
| mnist_hard5 | raw_pixels | 5 | 784 | 0.465 | 0.440 | 0.546 | 0.306 | 1.000 | 0.720 |
| mnist_hard5 | rff_512 | 5 | 512 | 0.536 | 0.503 | 0.600 | 0.199 | 1.000 | 0.800 |
| mnist_hard5 | trained_mlp | 5 | 256 | 0.324 | 0.277 | 0.266 | 0.549 | 1.000 | 0.770 |

## Artifacts

- `metrics.json`
- `figures/mixing_vs_tail.png`
- `figures/tail_vs_accuracy.png`
- `figures/entropy_vs_tail.png`
