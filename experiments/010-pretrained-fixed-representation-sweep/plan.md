# Plan

## Question

Does the fixed-representation obstruction story survive when the fixed metric
comes from a real pretrained vision backbone rather than small randomly trained
models?

## Representations

CIFAR-10 tasks only:

- raw pixels;
- random ResNet18 frozen features;
- ImageNet-pretrained ResNet18 frozen features.

The pretrained run uses `torchvision==0.22.1` and
`ResNet18_Weights.IMAGENET1K_V1`.

## Tasks

- binary `cat vs dog`;
- binary `automobile vs truck`;
- multiclass CIFAR-10 all classes;
- multiclass animals6;
- multiclass vehicles4.

## Metrics

- binary or multiclass spectral tail;
- kNN disagreement/mixing;
- graph Dirichlet energy;
- alignment;
- closed-form kernel ridge train/test accuracy and margin.

## Command

Run in tmux:

```bash
tmux new-session -d -s cntk-010-pretrained \
  'cd /Users/ziegel/Documents/CNTK && python3 experiments/010-pretrained-fixed-representation-sweep/scripts/run_pretrained.py --binary-n-per-class 80 --multi-n-per-class 40 --seed 0 --device cpu'
```

## Artifacts

- `metrics.json`
- `result.md`
- `figures/mixing_vs_tail.png`
- `figures/tail_vs_accuracy.png`
- `figures/tail_vs_margin.png`
