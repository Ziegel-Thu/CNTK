# MNIST Stress-Test Result

## Run

Command: `python experiments/004-intrinsic-collision-stress/scripts/run_mnist_stress.py --n-per-class 150 --epochs 160 --seed 0 --device cpu`

## Summary

Main observations:

- Clean feature learning behaves as expected: train tail collapses and clean test
  tail drops to `0.085`, with test accuracy `0.947`.
- With `10%` label flips, feature learning still partially transfers: test tail
  drops to `0.151` and clean test accuracy remains `0.933`.
- With `30%` label flips, feature learning memorizes noisy labels: train tail
  drops from `0.807` to `0.008`, but clean test tail worsens to `0.288` and
  clean test accuracy falls to `0.670`.
- With exact opposite-label duplicates, train accuracy is capped near `0.833`
  and train tail remains high (`0.337`), because identical inputs with opposite
  labels are intrinsic contradictions. Still, clean test tail improves to
  `0.114` and clean test accuracy reaches `0.940`, so the model can learn the
  underlying clean structure while failing to satisfy contradictions.

| condition | regime | train tail start | train tail final | test tail final | train mix final | test mix final | movement | train acc noisy | test acc clean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | frozen_random | 0.246 | 0.246 | 0.284 | 0.155 | 0.141 | 0.000 | 0.997 | 0.920 |
| clean | feature_learning | 0.319 | 0.001 | 0.085 | 0.000 | 0.057 | 0.885 | 1.000 | 0.947 |
| flip0.1 | frozen_random | 0.498 | 0.498 | 0.246 | 0.268 | 0.131 | 0.000 | 0.970 | 0.883 |
| flip0.1 | feature_learning | 0.580 | 0.007 | 0.151 | 0.000 | 0.080 | 0.790 | 1.000 | 0.933 |
| flip0.3 | frozen_random | 0.807 | 0.807 | 0.242 | 0.436 | 0.156 | 0.000 | 0.930 | 0.707 |
| flip0.3 | feature_learning | 0.807 | 0.008 | 0.288 | 0.000 | 0.222 | 0.691 | 1.000 | 0.670 |
| duplicate0.2 | frozen_random | 0.641 | 0.641 | 0.290 | 0.367 | 0.157 | 0.000 | 0.831 | 0.853 |
| duplicate0.2 | feature_learning | 0.691 | 0.337 | 0.114 | 0.183 | 0.074 | 0.575 | 0.833 | 0.940 |

## Interpretation Prompts

- Clean labels should allow feature learning to lower train and test tail.
- Label noise may lower train tail but should hurt clean test accuracy.
- Exact opposite-label duplicates are intrinsic contradictions; a deterministic
  network cannot classify both copies correctly.

## Artifacts

- `metrics_mnist_stress.json`
- `figures/train_tail_over_time.png`
- `figures/test_tail_over_time.png`
- `figures/test_accuracy_over_time.png`
