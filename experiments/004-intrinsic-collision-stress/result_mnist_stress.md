# MNIST Stress-Test Result

## Run

Command: `python experiments/004-intrinsic-collision-stress/scripts/run_mnist_stress.py --n-per-class 150 --epochs 160 --seed 0 --device cpu`

## Summary

Main observations:

- Clean feature learning transfers: clean test tail drops to `0.085`, test acc
  reaches `0.947`.
- Random `30%` label flips are a clear memorization case: train tail drops
  `0.807 -> 0.008`, but clean test tail worsens to `0.288` and test acc falls to
  `0.670`.
- Adversarial local flips also expose a train/test gap. At `30%`, feature
  learning drops train tail to `0.005` and clean test tail to `0.169`, but clean
  test accuracy is only `0.663`. Tail/mixing must therefore be read together
  with clean accuracy/margin.
- Exact opposite-label duplicates cap train accuracy near `0.833` and prevent
  full train-tail collapse (`0.336` final), as expected for identical inputs with
  contradictory labels. Clean test geometry still improves, so the model can
  learn the underlying clean structure while failing on contradictions.

Main observations to check after each run:

- Clean data should reduce train and test tail under feature learning.
- Heavy random or adversarial label noise can let train tail collapse while
  clean test tail worsens, indicating memorization.
- Exact opposite-label duplicates should cap train accuracy because a
  deterministic classifier cannot satisfy both labels at identical inputs.

| condition | regime | train tail start | train tail final | test tail final | train mix final | test mix final | movement | train acc noisy | test acc clean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | frozen_random | 0.246 | 0.246 | 0.284 | 0.155 | 0.141 | 0.000 | 0.997 | 0.920 |
| clean | feature_learning | 0.319 | 0.001 | 0.085 | 0.000 | 0.057 | 0.885 | 1.000 | 0.947 |
| flip0.1 | frozen_random | 0.498 | 0.498 | 0.246 | 0.268 | 0.131 | 0.000 | 0.970 | 0.883 |
| flip0.1 | feature_learning | 0.580 | 0.007 | 0.151 | 0.000 | 0.080 | 0.790 | 1.000 | 0.933 |
| flip0.3 | frozen_random | 0.807 | 0.807 | 0.242 | 0.436 | 0.156 | 0.000 | 0.930 | 0.707 |
| flip0.3 | feature_learning | 0.807 | 0.008 | 0.288 | 0.000 | 0.222 | 0.691 | 1.000 | 0.670 |
| advflip0.1 | frozen_random | 0.401 | 0.401 | 0.290 | 0.253 | 0.157 | 0.000 | 0.970 | 0.883 |
| advflip0.1 | feature_learning | 0.453 | 0.008 | 0.117 | 0.000 | 0.099 | 0.807 | 1.000 | 0.880 |
| advflip0.3 | frozen_random | 0.576 | 0.576 | 0.261 | 0.308 | 0.146 | 0.000 | 0.973 | 0.707 |
| advflip0.3 | feature_learning | 0.607 | 0.005 | 0.169 | 0.000 | 0.078 | 0.799 | 1.000 | 0.663 |
| duplicate0.2 | frozen_random | 0.619 | 0.619 | 0.251 | 0.356 | 0.148 | 0.000 | 0.833 | 0.853 |
| duplicate0.2 | feature_learning | 0.661 | 0.336 | 0.128 | 0.184 | 0.093 | 0.561 | 0.833 | 0.917 |

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
