# 001 - Spectral Tail Diagnostics

## Goal

Measure whether local opposite-label mixing under a fixed kernel metric predicts
large spectral label tail and slow kernel gradient-flow training.

## Hypothesis

For fixed kernels / fixed representations, higher opposite-label local mixing
correlates with:

- larger spectral tail `T_y(m)`
- worse kernel-target alignment
- slower predicted kernel gradient-flow loss decay
- poorer margin or kernel classifier performance

## Datasets

- Toy: two moons, XOR/checkerboard, synthetic paired opposite-label collisions.
- MNIST binary tasks: `3 vs 8`, `4 vs 9`.
- CIFAR binary tasks: `cat vs dog`, `automobile vs truck`.

## Kernels / Representations

- RBF kernel over input pixels/features.
- Laplace kernel over input pixels/features.
- Random Fourier features.
- Random MLP/CNN feature Gram.
- Initial CNN feature Gram.

## Metrics

- eigenvalue spectrum of `K`
- `E_y(m)` and `T_y(m)`
- kernel-target alignment
- opposite-label nearest-neighbor distance
- opposite-label kNN ratio
- local label entropy
- collision graph count `q_rho`
- predicted kernel gradient-flow loss curve

## Outputs

- `metrics.json`
- `figures/spectral_tail.png`
- `figures/mixing_vs_tail.png`
- `figures/gradient_flow_decay.png`
- `result.md`

## Run Protocol

Run inside tmux:

```bash
tmux new -s cntk-001-spectral-tail
```

## Success Criteria

The experiment is useful if it shows either:

- a clear positive relationship between local mixing and spectral tail / slow
  gradient flow, or
- a clear counterexample that identifies when the proposed diagnostic fails.
