# Plan

## Question

Do the fixed-metric obstruction diagnostics remain useful beyond binary class
pairs?

The earlier experiments intentionally used binary tasks so that spectral label
tail, opposite-label kNN mixing, and binary margins were easy to interpret. This
experiment extends the measurement contract to multiclass label subspaces:

- centered one-hot label subspace tail under a fixed Gram matrix;
- kNN disagreement ratio and true multiclass local label entropy;
- multiclass linear-probe train/test accuracy.

## Tasks

Default quick tasks:

- MNIST all 10 classes, balanced subset.
- CIFAR-10 all 10 classes, balanced subset.

Representations:

- raw pixels;
- random Fourier features;
- random MLP features;
- trained MLP features;
- random CNN features for CIFAR-10;
- trained CNN features for CIFAR-10.

## Hypotheses

- Multiclass kNN disagreement should still correlate positively with multiclass
  spectral tail.
- Multiclass spectral tail should correlate negatively with linear-probe test
  accuracy.
- CIFAR-10 raw/random features should stay high-tail/high-entropy compared with
  MNIST.
- Trained small features may improve train diagnostics more than test
  diagnostics; that gap should be treated as memorization or weak transfer.

## Command

Run substantive jobs in tmux:

```bash
tmux new-session -d -s cntk-005-multiclass \
  'cd /Users/ziegel/Documents/CNTK && python3 experiments/005-multiclass-obstruction-diagnostics/scripts/run_multiclass.py --n-per-class 40 --mlp-epochs 80 --cnn-epochs 60 --seed 0 --device cpu'
```

## Artifacts

- `metrics.json`
- `result.md`
- `figures/mixing_vs_tail.png`
- `figures/tail_vs_accuracy.png`
- `figures/entropy_vs_tail.png`
