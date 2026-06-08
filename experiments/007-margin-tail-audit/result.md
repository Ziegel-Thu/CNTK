# Result

## Run

Command: `python3 experiments/007-margin-tail-audit/scripts/run_audit.py`

## Summary

- final rows: `48`
- dynamics delta rows: `30`
- corr(final test tail, final accuracy) = `-0.855`
- corr(final margin median, final accuracy) = `0.689`
- corr(final test tail, final margin median) = `-0.723`
- corr(final local mixing, final test tail) = `0.885`
- corr(test tail decrease, margin gain) = `0.918`
- corr(test tail decrease, accuracy gain) = `0.480`

Interpretation:

- Margin is not just a duplicate of tail: final tail has the stronger
  overall accuracy correlation, while margin adds complementary dynamic
  and within-CIFAR signal.
- Tail/mixing remain geometric obstruction diagnostics; margin is the
  performance-facing companion that catches optimization and confidence.
- In later dynamics runs, report tail, mixing, accuracy, and margin together.

| source | rows | tail mean | margin mean | accuracy mean | corr tail/acc | corr margin/acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 002_cifar_cnn | 6 | 0.773 | 0.212 | 0.563 | -0.716 | 0.974 |
| 002_cifar_mlp | 6 | 0.817 | 1.660 | 0.607 | -0.305 | -0.166 |
| 002_mnist | 6 | 0.217 | 6.576 | 0.898 | -0.598 | 0.702 |
| 004_stress | 12 | 0.209 | 6.447 | 0.830 | -0.384 | 0.429 |
| 006_cifar_schedule | 18 | 0.764 | -1.039 | 0.375 | -0.634 | 0.826 |

## Artifacts

- `metrics.json`
- `figures/tail_vs_accuracy.png`
- `figures/margin_vs_accuracy.png`
- `figures/tail_decrease_vs_margin_gain.png`
