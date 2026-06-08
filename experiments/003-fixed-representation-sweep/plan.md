# 003 - Fixed Representation Sweep

## Goal

Test whether the obstruction is about fixed metrics generally, not only CNTK.

## Hypothesis

Across fixed representations, performance is explained better by local mixing
and spectral label complexity than by representation family name alone.

## Representations

- Raw pixels.
- Random features.
- Random CNN features.
- Frozen CNN trained on another task.
- Frozen self-supervised/pretrained features if available.
- Linear probe on frozen trained features.
- Fine-tuned features as a positive control.

## Datasets

- MNIST binary tasks.
- CIFAR binary tasks.
- Optional: small subsets to make eigendecompositions cheap and repeatable.

## Metrics

- `T_y(m)` and area under spectral-tail curve.
- opposite-label local mixing stats.
- kernel-target alignment.
- linear probe accuracy.
- kernel ridge / kernel classifier accuracy.
- margin distribution.

## Outputs

- `metrics.json`
- `figures/representation_sweep.png`
- `figures/tail_vs_accuracy.png`
- `result.md`

## Run Protocol

Run inside tmux:

```bash
tmux new -s cntk-003-fixed-repr
```

## Success Criteria

The experiment supports the broader project if fixed representations with low
mixing/low spectral tail perform well, while fixed representations with high
mixing/high tail perform poorly, independent of whether the representation is
called a kernel, CNN feature, or pretrained feature.
