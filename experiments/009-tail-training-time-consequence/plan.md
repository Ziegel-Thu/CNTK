# Plan

## Question

What is the measurable consequence of spectral label tail under a static kernel?

This experiment connects three quantities:

```text
local label mixing / graph roughness
=> spectral label tail
=> kernel gradient-flow residual time and source/RKHS norm proxy
```

## Tasks

Datasets:

- toy suite: moons, XOR, collision pairs;
- MNIST `3 vs 8`, `4 vs 9`;
- CIFAR-10 `cat vs dog`, `automobile vs truck`.

Kernels:

- linear;
- RBF;
- Laplace;
- RFF feature Gram.

## Metrics

Per dataset/kernel:

- local mixing and graph Dirichlet energy;
- spectral `tail@10%`, alignment;
- exact kernel gradient-flow residual curve;
- time to normalized training residual `<= 0.1`;
- regularized source norm proxy `sqrt(y^T(K + eps*nI)^-1 y)`.

## Command

Run in tmux:

```bash
tmux new-session -d -s cntk-009-tail-time \
  'cd /Users/ziegel/Documents/CNTK && python3 experiments/009-tail-training-time-consequence/scripts/run_tail_time.py --n 160 --n-per-class 80 --seed 0'
```

## Artifacts

- `metrics.json`
- `result.md`
- `figures/tail_vs_time.png`
- `figures/tail_vs_source_norm.png`
- `figures/mixing_vs_time.png`
- `figures/residual_curves.png`
