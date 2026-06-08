# Plan

## Question

Experiment `006` showed that probe margin median can correlate strongly with
probe accuracy even when CIFAR tail ranges are narrow. This audit asks:

> Across existing feature-dynamics and schedule-sweep results, is margin a
> redundant proxy for spectral tail, or a complementary diagnostic?

## Inputs

Existing metrics only; no new training:

- `002-feature-metric-dynamics/metrics_mnist_over_time.json`
- `002-feature-metric-dynamics/metrics_cifar_over_time.json`
- `002-feature-metric-dynamics/metrics_cifar_cnn_over_time.json`
- `004-intrinsic-collision-stress/metrics_mnist_stress.json`
- `006-cifar-multiclass-schedule-sweep/metrics.json`

## Metrics

Final-row correlations:

- test tail vs test/probe accuracy;
- test margin median vs test/probe accuracy;
- test tail vs test margin median;
- test local mixing vs test tail.

Dynamics correlations:

- test tail decrease vs test margin gain;
- test tail decrease vs test accuracy gain.

## Command

Run in tmux:

```bash
tmux new-session -d -s cntk-007-margin-audit \
  'cd /Users/ziegel/Documents/CNTK && python3 experiments/007-margin-tail-audit/scripts/run_audit.py'
```

## Artifacts

- `metrics.json`
- `result.md`
- `figures/tail_vs_accuracy.png`
- `figures/margin_vs_accuracy.png`
- `figures/tail_decrease_vs_margin_gain.png`
