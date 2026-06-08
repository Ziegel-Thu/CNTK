# Plan

## Question

Experiment `005` showed that multiclass CIFAR remains high-tail/high-entropy
under small CPU-trained representations, and one short trained-CNN run did not
consistently beat a random CNN linear probe. This experiment asks:

> Is that a real fixed-metric/architecture limitation, or mainly a weak training
> schedule artifact?

## Tasks

Balanced CIFAR-10 multiclass subsets:

- all 10 classes;
- animals: bird, cat, deer, dog, frog, horse;
- vehicles: airplane, automobile, ship, truck.

Regimes:

- `random_cnn`: random convolutional features + trained linear probe;
- `short_fullbatch`: the 005-style short full-batch CNN schedule;
- `strong_minibatch`: wider CNN, mini-batches, AdamW, cosine schedule, random
  crop/flip augmentation.

## Metrics

Per task/regime/seed:

- train/test multiclass label-subspace `tail@10%`;
- train/test kNN disagreement and normalized local label entropy;
- train/test head or probe accuracy;
- multiclass margin median/mean;
- feature Gram movement relative to initialization for trained regimes.

Aggregate:

- mean/std across seeds;
- correlations across final runs:
  - test disagreement vs test tail;
  - test entropy vs test tail;
  - test tail vs test accuracy;
  - test margin median vs test accuracy.

## Command

Run substantive jobs in tmux:

```bash
tmux new-session -d -s cntk-006-cifar-schedule \
  'cd /Users/ziegel/Documents/CNTK && python3 experiments/006-cifar-multiclass-schedule-sweep/scripts/run_sweep.py --n-per-class 40 --epochs 50 --short-epochs 30 --probe-epochs 150 --seeds 0 1 --width 32 --feature-dim 128 --device cpu'
```

## Artifacts

- `metrics.json`
- `result.md`
- `figures/tail_by_regime.png`
- `figures/accuracy_by_regime.png`
- `figures/tail_vs_accuracy.png`
- `figures/margin_vs_accuracy.png`
