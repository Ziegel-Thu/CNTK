# 004 - Intrinsic Collision Stress Test

## Goal

Separate correctable metric mismatch from intrinsic label ambiguity/noise.

## Hypothesis

Feature learning can reduce local label mixing when the collision comes from a
bad initial metric, but it should not fully solve true contradictory labels or
duplicated opposite-label samples.

## Perturbations

- Random label noise within local clusters.
- Duplicated examples with opposite labels.
- Near-duplicate examples with opposite labels.
- Adversarially selected opposite-label neighbor collisions.

## Regimes

- Static kernel / kernel classifier.
- Frozen features + linear head.
- Feature-learning MLP/CNN.

## Metrics

- local mixing before/after training
- spectral tail before/after training
- memorization gap
- train/test accuracy
- margin distribution
- distance between duplicated opposite-label pairs

## Outputs

- `metrics.json`
- `figures/noise_level_vs_tail.png`
- `figures/noise_level_vs_accuracy.png`
- `result.md`

## Run Protocol

Run inside tmux:

```bash
tmux new -s cntk-004-collision-stress
```

## Success Criteria

The experiment is useful if it identifies a boundary:

- feature learning helps when mixing is metric mismatch
- feature learning fails or memorizes when mixing is intrinsic contradiction
