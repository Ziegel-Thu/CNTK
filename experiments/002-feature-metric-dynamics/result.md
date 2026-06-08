# Result

## Run

Command: `python experiments/002-feature-metric-dynamics/scripts/run_toy.py --quick --n 300 --epochs 120 --seed 0`

## Summary

Toy feature-metric dynamics completed for frozen random features, feature
learning, and a lazy-ish wide/small-LR control.

Main observations:

- On `two_moons_noise0.2`, feature learning moves the feature Gram matrix
  (`movement=0.305`), lowers `tail@10%` from `0.082` to `0.053`, lowers
  opposite-label kNN ratio from `0.044` to `0.028`, and improves train/test
  accuracy.
- On `collision_pairs_sep0.03`, feature learning moves the metric but does not
  solve the task: tail remains near `1.0`, mixing remains near `0.55`, and
  accuracy stays around chance. This is a useful negative control for intrinsic
  contradictory local collisions.
- The current lazy-ish control is not lazy enough: it still has nontrivial
  feature Gram movement on two moons (`0.227`). The next version should use SGD,
  smaller learning rate, and possibly wider networks.

| dataset | regime | tail@10% start | tail@10% final | mixing start | mixing final | movement final | train acc | test acc |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| two_moons_noise0.2 | frozen_random | 0.085 | 0.085 | 0.050 | 0.050 | 0.000 | 0.887 | 0.917 |
| two_moons_noise0.2 | feature_learning | 0.082 | 0.053 | 0.044 | 0.028 | 0.305 | 0.987 | 0.970 |
| two_moons_noise0.2 | lazy_wide_small_lr | 0.076 | 0.071 | 0.049 | 0.031 | 0.227 | 0.987 | 0.977 |
| collision_pairs_sep0.03 | frozen_random | 0.999 | 0.999 | 0.547 | 0.547 | 0.000 | 0.507 | 0.507 |
| collision_pairs_sep0.03 | feature_learning | 1.000 | 0.997 | 0.548 | 0.550 | 0.462 | 0.517 | 0.500 |
| collision_pairs_sep0.03 | lazy_wide_small_lr | 1.000 | 0.995 | 0.553 | 0.550 | 0.062 | 0.530 | 0.497 |

## Artifacts

- `metrics_over_time.json`
- `figures/tail_over_time.png`
- `figures/mixing_over_time.png`
- `figures/kernel_movement_over_time.png`

## Next

- Compare these feature-Gram dynamics with empirical NTK dynamics on tiny subsets.
- Run the same trace on MNIST binary subsets once data utilities are ready.
