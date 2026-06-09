# Margin-Curve Supplement

## Run

Command: `python3 experiments/002-feature-metric-dynamics/scripts/plot_margin_curves.py`

## Summary

- Train rows with margin/tail: `24`
- Test rows with margin/tail: `18`
- corr(final train tail, final train margin) = `-0.686`
- corr(final test tail, final test margin) = `-0.623`
- corr(final test margin, final test acc) = `0.559`

Interpretation:

- Margin is now a first-class feature-dynamics curve alongside tail,
  mixing, kernel movement, and accuracy.
- Tail and mixing describe geometry; margin indicates whether the
  geometry is becoming a usable classifier.
- Negative train/test gaps should be read as memorization even when train
  margin increases.

| family | dataset | regime | train margin start | train margin final | test margin final | test tail final | test acc final |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| toy | two_moons_noise0.2 | frozen_random | -0.002 | 2.498 | NA | NA | 0.917 |
| toy | two_moons_noise0.2 | feature_learning | 0.024 | 10.374 | NA | NA | 0.970 |
| toy | two_moons_noise0.2 | lazy_wide_small_lr | -0.047 | 2.261 | NA | NA | 0.890 |
| toy | collision_pairs_sep0.03 | frozen_random | -0.000 | 0.000 | NA | NA | 0.507 |
| toy | collision_pairs_sep0.03 | feature_learning | -0.002 | 0.001 | NA | NA | 0.500 |
| toy | collision_pairs_sep0.03 | lazy_wide_small_lr | 0.000 | 0.000 | NA | NA | 0.507 |
| mnist | mnist_3vs8_n150_train | frozen_random | 0.005 | 3.303 | 3.227 | 0.284 | 0.920 |
| mnist | mnist_3vs8_n150_train | feature_learning | 0.012 | 17.936 | 17.703 | 0.085 | 0.947 |
| mnist | mnist_3vs8_n150_train | lazy_wide_small_lr | 0.003 | 0.231 | 0.226 | 0.218 | 0.883 |
| mnist | mnist_4vs9_n150_train | frozen_random | -0.008 | 3.055 | 2.560 | 0.313 | 0.880 |
| mnist | mnist_4vs9_n150_train | feature_learning | 0.004 | 16.684 | 15.633 | 0.142 | 0.897 |
| mnist | mnist_4vs9_n150_train | lazy_wide_small_lr | -0.012 | 0.117 | 0.107 | 0.258 | 0.860 |
| cifar_mlp | cifar10_catvsdog_n120_train | frozen_random | -0.000 | 2.146 | 0.604 | 0.882 | 0.642 |
| cifar_mlp | cifar10_catvsdog_n120_train | feature_learning | 0.006 | 15.348 | 2.427 | 0.881 | 0.575 |
| cifar_mlp | cifar10_catvsdog_n120_train | lazy_wide_small_lr | -0.012 | 0.046 | 0.012 | 0.855 | 0.550 |
| cifar_mlp | cifar10_automobilevstruck_n120_train | frozen_random | 0.002 | 2.445 | 0.907 | 0.802 | 0.633 |
| cifar_mlp | cifar10_automobilevstruck_n120_train | feature_learning | -0.003 | 20.178 | 5.959 | 0.727 | 0.600 |
| cifar_mlp | cifar10_automobilevstruck_n120_train | lazy_wide_small_lr | -0.003 | 0.088 | 0.050 | 0.754 | 0.642 |
| cifar_cnn | cifar10_catvsdog_n120_train | frozen_random | 0.004 | 0.058 | -0.008 | 0.874 | 0.492 |
| cifar_cnn | cifar10_catvsdog_n120_train | feature_learning | 0.001 | 0.709 | 0.361 | 0.834 | 0.604 |
| cifar_cnn | cifar10_catvsdog_n120_train | lazy_small_lr | -0.000 | 0.000 | -0.000 | 0.837 | 0.488 |
| cifar_cnn | cifar10_automobilevstruck_n120_train | frozen_random | 0.001 | 0.106 | 0.056 | 0.726 | 0.562 |
| cifar_cnn | cifar10_automobilevstruck_n120_train | feature_learning | 0.004 | 1.084 | 0.861 | 0.631 | 0.729 |
| cifar_cnn | cifar10_automobilevstruck_n120_train | lazy_small_lr | -0.003 | -0.003 | 0.001 | 0.735 | 0.500 |

## Artifacts

- `figures/toy_margin_over_time.png`
- `figures/mnist_margin_over_time.png`
- `figures/mnist_test_margin_over_time.png`
- `figures/cifar_mlp_margin_over_time.png`
- `figures/cifar_mlp_test_margin_over_time.png`
- `figures/cifar_cnn_margin_over_time.png`
- `figures/cifar_cnn_test_margin_over_time.png`
