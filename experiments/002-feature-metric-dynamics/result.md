# Result

## Run

Command: `python experiments/002-feature-metric-dynamics/scripts/run_toy.py --quick --n 300 --epochs 120 --seed 0`

## Summary

Toy feature-metric dynamics completed for frozen random features, feature
learning, and a lazy-ish wide/small-LR control.

Interpretation:

- On `two_moons_noise0.2`, feature learning reduces `tail@10%` from `0.082` to
  `0.053`, lowers opposite-label kNN ratio from `0.044` to `0.028`, and reaches
  `0.970` test accuracy.
- The stricter lazy control now has much smaller feature movement than the
  feature-learning run (`0.105` vs `0.305`) and does not reduce tail/mixing.
- On `collision_pairs_sep0.03`, feature learning moves the feature Gram
  (`0.462`) but leaves tail/mixing and accuracy essentially unchanged. This is a
  good negative control for intrinsic contradictory local collisions.

| dataset | regime | tail@10% start | tail@10% final | mixing start | mixing final | movement final | train acc | test acc |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| two_moons_noise0.2 | frozen_random | 0.085 | 0.085 | 0.050 | 0.050 | 0.000 | 0.887 | 0.917 |
| two_moons_noise0.2 | feature_learning | 0.082 | 0.053 | 0.044 | 0.028 | 0.305 | 0.987 | 0.970 |
| two_moons_noise0.2 | lazy_wide_small_lr | 0.079 | 0.080 | 0.048 | 0.049 | 0.105 | 0.870 | 0.890 |
| collision_pairs_sep0.03 | frozen_random | 0.999 | 0.999 | 0.547 | 0.547 | 0.000 | 0.507 | 0.507 |
| collision_pairs_sep0.03 | feature_learning | 1.000 | 0.997 | 0.548 | 0.550 | 0.462 | 0.517 | 0.500 |
| collision_pairs_sep0.03 | lazy_wide_small_lr | 1.000 | 1.000 | 0.552 | 0.552 | 0.000 | 0.510 | 0.507 |

## Artifacts

- `metrics_over_time.json`
- `figures/tail_over_time.png`
- `figures/mixing_over_time.png`
- `figures/kernel_movement_over_time.png`
- `result_margin_curves.md`
- `figures/toy_margin_over_time.png`
- `figures/mnist_test_margin_over_time.png`
- `figures/cifar_mlp_test_margin_over_time.png`
- `figures/cifar_cnn_test_margin_over_time.png`

## Next

- Compare these feature-Gram dynamics with empirical NTK dynamics on tiny subsets.
- Run the same trace on MNIST binary subsets once data utilities are ready.
