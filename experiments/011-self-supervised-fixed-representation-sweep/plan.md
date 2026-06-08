# Plan

## Question

Do self-supervised frozen features follow the same fixed-representation
obstruction pattern?

Experiment `010` handled supervised ImageNet ResNet18. This experiment adds a
self-supervised DINO ViT-S/16 backbone from `facebookresearch/dino`.

## Representations

- raw pixels;
- supervised ImageNet ResNet18 frozen features;
- self-supervised DINO ViT-S/16 frozen features.

## Tasks

CIFAR-10:

- binary `cat vs dog`;
- binary `automobile vs truck`;
- multiclass all10;
- multiclass animals6;
- multiclass vehicles4.

## Metrics

- local mixing/disagreement;
- graph Dirichlet energy;
- spectral tail;
- kernel ridge accuracy and margin.

## Command

Run in tmux:

```bash
tmux new-session -d -s cntk-011-selfsup \
  'cd /Users/ziegel/Documents/CNTK && python3 experiments/011-self-supervised-fixed-representation-sweep/scripts/run_selfsup.py --binary-n-per-class 60 --multi-n-per-class 30 --batch-size 48 --seed 0 --device cpu'
```

## Artifacts

- `metrics.json`
- `result.md`
- `figures/mixing_vs_tail.png`
- `figures/tail_vs_accuracy.png`
- `figures/tail_vs_margin.png`
