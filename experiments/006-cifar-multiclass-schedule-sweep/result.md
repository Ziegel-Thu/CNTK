# Result

## Run

Command: `python3 experiments/006-cifar-multiclass-schedule-sweep/scripts/run_sweep.py --n-per-class 40 --epochs 50 --short-epochs 30 --probe-epochs 150 --seeds 0 1 --width 32 --feature-dim 128 --batch-size 64 --lr 0.001 --short-lr 0.001 --weight-decay 0.001 --label-smoothing 0.05 --crop-padding 4 --k-neighbors 10 --seed 0 --device cpu`

## Summary

- corr(test kNN disagreement, test multiclass tail@10%) = `0.500`
- corr(test normalized local entropy, test multiclass tail@10%) = `0.613`
- corr(test multiclass tail@10%, probe test acc) = `-0.634`
- corr(probe test margin median, probe test acc) = `0.826`

Aggregate rows report mean/std over seeds.

Run-specific observations:

- `strong_minibatch` improves all three CIFAR multiclass tasks relative to
  `random_cnn`: all-10 tail `0.772 -> 0.731`, animals6 tail `0.826 -> 0.777`,
  vehicles4 tail `0.755 -> 0.724`.
- Probe accuracy also improves under `strong_minibatch`: all-10 `0.272 ->
  0.392`, animals6 `0.319 -> 0.373`, vehicles4 `0.428 -> 0.497`.
- Correlations are weaker than experiment `005` because this run only compares
  CIFAR tasks/regimes with a narrower tail range, but the signs remain aligned.
- Probe margin median is strongly tied to probe accuracy (`0.826`), so margin
  should become a first-class diagnostic in later dynamics runs.

| dataset | regime | seeds | test tail mean | test tail std | disagreement mean | entropy mean | movement mean | probe acc mean | probe acc std | margin median mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cifar10_all10 | random_cnn | 2 | 0.772 | 0.002 | 0.815 | 0.686 | 0.000 | 0.272 | 0.010 | -1.956 |
| cifar10_all10 | short_fullbatch | 2 | 0.748 | 0.015 | 0.789 | 0.614 | 0.457 | 0.344 | 0.001 | -0.746 |
| cifar10_all10 | strong_minibatch | 2 | 0.731 | 0.009 | 0.754 | 0.579 | 0.479 | 0.392 | 0.020 | -0.716 |
| cifar10_animals6 | random_cnn | 2 | 0.826 | 0.001 | 0.781 | 0.798 | 0.000 | 0.319 | 0.002 | -2.280 |
| cifar10_animals6 | short_fullbatch | 2 | 0.800 | 0.008 | 0.761 | 0.741 | 0.333 | 0.352 | 0.006 | -0.894 |
| cifar10_animals6 | strong_minibatch | 2 | 0.777 | 0.003 | 0.744 | 0.732 | 0.378 | 0.373 | 0.015 | -1.200 |
| cifar10_vehicles4 | random_cnn | 2 | 0.755 | 0.010 | 0.652 | 0.784 | 0.000 | 0.428 | 0.041 | -0.640 |
| cifar10_vehicles4 | short_fullbatch | 2 | 0.747 | 0.000 | 0.637 | 0.733 | 0.265 | 0.397 | 0.009 | -0.795 |
| cifar10_vehicles4 | strong_minibatch | 2 | 0.724 | 0.021 | 0.615 | 0.680 | 0.347 | 0.497 | 0.016 | -0.124 |

## Single-Seed Rows

| dataset | regime | seed | test tail | test disagreement | test entropy | movement | head test acc | probe test acc | probe margin median |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cifar10_all10 | random_cnn | 0 | 0.773 | 0.818 | 0.697 | 0.000 | 0.112 | 0.282 | -1.784 |
| cifar10_all10 | random_cnn | 1 | 0.770 | 0.813 | 0.675 | 0.000 | 0.102 | 0.262 | -2.128 |
| cifar10_all10 | short_fullbatch | 0 | 0.734 | 0.786 | 0.609 | 0.432 | 0.317 | 0.343 | -0.761 |
| cifar10_all10 | short_fullbatch | 1 | 0.763 | 0.791 | 0.619 | 0.483 | 0.270 | 0.345 | -0.732 |
| cifar10_all10 | strong_minibatch | 0 | 0.722 | 0.749 | 0.572 | 0.481 | 0.343 | 0.412 | -0.654 |
| cifar10_all10 | strong_minibatch | 1 | 0.740 | 0.758 | 0.585 | 0.478 | 0.365 | 0.373 | -0.778 |
| cifar10_animals6 | random_cnn | 0 | 0.825 | 0.787 | 0.807 | 0.000 | 0.167 | 0.317 | -2.193 |
| cifar10_animals6 | random_cnn | 1 | 0.827 | 0.775 | 0.789 | 0.000 | 0.175 | 0.321 | -2.368 |
| cifar10_animals6 | short_fullbatch | 0 | 0.792 | 0.750 | 0.723 | 0.346 | 0.371 | 0.346 | -0.788 |
| cifar10_animals6 | short_fullbatch | 1 | 0.808 | 0.772 | 0.758 | 0.320 | 0.292 | 0.358 | -1.001 |
| cifar10_animals6 | strong_minibatch | 0 | 0.780 | 0.748 | 0.736 | 0.379 | 0.350 | 0.358 | -1.355 |
| cifar10_animals6 | strong_minibatch | 1 | 0.774 | 0.740 | 0.727 | 0.378 | 0.371 | 0.387 | -1.044 |
| cifar10_vehicles4 | random_cnn | 0 | 0.765 | 0.642 | 0.772 | 0.000 | 0.312 | 0.387 | -0.766 |
| cifar10_vehicles4 | random_cnn | 1 | 0.745 | 0.661 | 0.795 | 0.000 | 0.256 | 0.469 | -0.515 |
| cifar10_vehicles4 | short_fullbatch | 0 | 0.747 | 0.611 | 0.714 | 0.274 | 0.425 | 0.406 | -0.706 |
| cifar10_vehicles4 | short_fullbatch | 1 | 0.747 | 0.662 | 0.753 | 0.256 | 0.381 | 0.387 | -0.885 |
| cifar10_vehicles4 | strong_minibatch | 0 | 0.745 | 0.608 | 0.669 | 0.367 | 0.475 | 0.481 | -0.471 |
| cifar10_vehicles4 | strong_minibatch | 1 | 0.703 | 0.623 | 0.692 | 0.328 | 0.431 | 0.512 | 0.223 |

## Artifacts

- `metrics.json`
- `figures/tail_by_regime.png`
- `figures/accuracy_by_regime.png`
- `figures/tail_vs_accuracy.png`
- `figures/margin_vs_accuracy.png`
