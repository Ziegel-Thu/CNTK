# 002 - Feature Metric Dynamics

## Goal

Track how feature learning changes the sample metric and whether that change
reduces local label mixing and spectral label tail.

## Hypothesis

Feature-learning regimes reduce fixed-metric obstruction by changing `K_t` so
that:

- opposite-label nearest-neighbor distances increase
- same-label local purity increases
- `T_y^{K_t}(m)` decreases
- kernel-target alignment increases
- margins and accuracy improve

Lazy/frozen regimes should show much smaller movement in these diagnostics.

## Regimes

- Frozen random features + train linear head.
- Lazy-ish wide MLP/CNN + small learning rate.
- Feature-learning MLP/CNN + normal width/larger learning rate.
- Fine-tuning from pretrained features if available.

## Datasets

- Toy: two moons / XOR with tunable mixing.
- MNIST binary tasks.
- CIFAR binary tasks.

## Measurements Over Training

At initialization, selected epochs, and final checkpoint:

- feature Gram matrix `K_t`
- relative kernel movement `||K_t - K_0||_F / ||K_0||_F`
- `E_y^{K_t}(m)`, `T_y^{K_t}(m)`
- kernel-target alignment
- opposite-label kNN/mixing stats
- train/test accuracy
- margin distribution

## Outputs

- `metrics_over_time.json`
- `figures/tail_over_time.png`
- `figures/mixing_over_time.png`
- `figures/kernel_movement_vs_accuracy.png`
- `result.md`

## Run Protocol

Run inside tmux:

```bash
tmux new -s cntk-002-feature-dynamics
```

## Success Criteria

The experiment is useful if feature-learning runs visibly move the metric in a
way that static/frozen/lazy runs do not, and this movement predicts accuracy or
margin improvements.
