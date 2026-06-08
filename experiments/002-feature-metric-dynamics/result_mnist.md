# MNIST Feature-Dynamics Result

## Run

Command: `python experiments/002-feature-metric-dynamics/scripts/run_mnist.py --n-per-class 150 --epochs 160 --seed 0 --device cpu`

## Summary

Main observations:

- Feature learning reduces both train and test feature-Gram spectral tail.
  - `3 vs 8`: test `tail@10%` drops from `0.350` to `0.085`.
  - `4 vs 9`: test `tail@10%` drops from `0.402` to `0.142`.
- Test local mixing also improves under feature learning:
  - `3 vs 8`: test opposite-kNN ratio ends at `0.057`.
  - `4 vs 9`: test opposite-kNN ratio ends at `0.105`.
- Frozen features keep geometry fixed, as expected.
- Lazy small-LR training has tiny feature movement (`0.014` and `0.006`) and
  only small tail/mixing changes.
- This is stronger than the toy result because the metric adaptation partially
  transfers to held-out MNIST subsets, not only the training samples.

| dataset | regime | train tail start | train tail final | test tail start | test tail final | train mix final | test mix final | movement final | test acc |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mnist_3vs8_n150_train | frozen_random | 0.246 | 0.246 | 0.284 | 0.284 | 0.155 | 0.141 | 0.000 | 0.920 |
| mnist_3vs8_n150_train | feature_learning | 0.319 | 0.001 | 0.350 | 0.085 | 0.000 | 0.057 | 0.885 | 0.947 |
| mnist_3vs8_n150_train | lazy_wide_small_lr | 0.190 | 0.178 | 0.228 | 0.218 | 0.119 | 0.123 | 0.014 | 0.883 |
| mnist_4vs9_n150_train | frozen_random | 0.282 | 0.282 | 0.313 | 0.313 | 0.207 | 0.204 | 0.000 | 0.880 |
| mnist_4vs9_n150_train | feature_learning | 0.419 | 0.003 | 0.402 | 0.142 | 0.000 | 0.105 | 0.847 | 0.897 |
| mnist_4vs9_n150_train | lazy_wide_small_lr | 0.265 | 0.247 | 0.260 | 0.258 | 0.211 | 0.178 | 0.006 | 0.860 |

## Interpretation Prompts

- Does feature learning reduce train `tail@10%` more than frozen/lazy controls?
- Does the test-subset feature Gram also improve, or is the change mostly train memorization?
- Does local mixing move together with spectral tail on train and test subsets?
- Is improvement larger on the harder `4 vs 9` task than on `3 vs 8`?
- Does margin improve even when tail movement is small?

## Artifacts

- `metrics_mnist_over_time.json`
- `figures/mnist_tail_over_time.png`
- `figures/mnist_test_tail_over_time.png`
- `figures/mnist_mixing_over_time.png`
- `figures/mnist_test_mixing_over_time.png`
- `figures/mnist_kernel_movement_over_time.png`
- `figures/mnist_accuracy_over_time.png`
