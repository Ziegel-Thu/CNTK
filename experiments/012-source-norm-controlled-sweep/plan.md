# Plan

## Question

Can fixed-metric local mixing and spectral tail be related to RKHS/source norm
proxies in a controlled same-kernel setting?

Experiments `008` and `009` included source norm, but mixed different kernel
families and feature scales. This experiment reports correlations *within each
kernel family*.

## Tasks

Controlled toy grid:

- collision-pair separations;
- two-moons noise levels;
- XOR with linear/RBF contrast.

Image binary tasks:

- MNIST `3 vs 8`, `4 vs 9`;
- CIFAR-10 `cat vs dog`, `automobile vs truck`.

## Kernels

- linear normalized feature Gram;
- RBF;
- Laplace;
- RFF feature Gram.

## Metrics

- kNN opposite-label mixing;
- graph Dirichlet energy;
- spectral `tail@10%`;
- alignment;
- regularized source norm proxy;
- kernel ridge margin/accuracy.

Correlations are reported globally and within each kernel family.

## Command

Run in tmux:

```bash
tmux new-session -d -s cntk-012-source-norm \
  'cd /Users/ziegel/Documents/CNTK && python3 experiments/012-source-norm-controlled-sweep/scripts/run_source_norm.py --n 160 --n-per-class 80 --seed 0'
```

## Artifacts

- `metrics.json`
- `result.md`
- `figures/tail_vs_source_norm.png`
- `figures/mixing_vs_source_norm.png`
- `figures/source_norm_vs_margin.png`
