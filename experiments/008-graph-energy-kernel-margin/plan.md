# Plan

## Question

Can local label mixing be turned into a richer diagnostic system that predicts
both spectral tail and classifier margin?

This experiment adds two missing pieces:

- signed/multiclass graph Dirichlet energy on the kNN metric graph;
- closed-form kernel ridge margin/source-norm diagnostics.

## Tasks

Balanced binary tasks:

- MNIST `3 vs 8`, `4 vs 9`;
- CIFAR-10 `cat vs dog`, `automobile vs truck`.

Representations/kernels:

- raw linear kernel;
- RBF kernel;
- Laplace kernel;
- random Fourier features;
- random MLP / trained MLP features;
- random CNN / trained CNN features for CIFAR.

## Metrics

For each task/representation:

- test spectral `tail@10%`;
- test kNN opposite-label ratio;
- test graph disagreement;
- test graph Dirichlet energy;
- kernel-target alignment;
- kernel ridge test accuracy;
- kernel ridge test margin median;
- RKHS/source norm proxy.

## Command

Run in tmux:

```bash
tmux new-session -d -s cntk-008-graph-margin \
  'cd /Users/ziegel/Documents/CNTK && python3 experiments/008-graph-energy-kernel-margin/scripts/run_graph_margin.py --n-per-class 80 --mlp-epochs 60 --cnn-epochs 50 --seed 0 --device cpu'
```

## Artifacts

- `metrics.json`
- `result.md`
- `figures/graph_energy_vs_tail.png`
- `figures/tail_vs_margin.png`
- `figures/source_norm_vs_margin.png`
