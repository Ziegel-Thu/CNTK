# Plan

## Question

Do pretrained backbone fine-tuning regimes actually change the held-out metric
in the direction predicted by the fixed-metric obstruction diagnostics?

The previous pretrained experiments (`010`, `011`) evaluated frozen features.
This experiment tracks `K_t` during fine-tuning.

## Regimes

- `frozen_head`: ImageNet ResNet18 frozen backbone, train only the classifier
  head.
- `finetune_layer4`: train ResNet18 `layer4` plus classifier head.
- `finetune_all`: train the full ResNet18 plus classifier head with a smaller
  backbone learning rate.

## Tasks

Default small CIFAR tasks:

- `cifar10_catvsdog`
- `cifar10_automobilevstruck`
- `cifar10_vehicles4`

## Metrics

At checkpoints `0, 1, 3, epochs`, record train/test:

- feature Gram `K_t`;
- relative movement `||K_t - K_0||_F / ||K_0||_F`;
- spectral `tail@10%`;
- local mixing/disagreement;
- graph Dirichlet energy;
- kernel-target alignment;
- head accuracy and margin;
- kernel-ridge accuracy and margin on the current features.

## Non-Trivial Check

The important comparison is not head accuracy alone.

Evidence for metric adaptation requires:

```text
finetune regime:
  test K_t movement > frozen_head
  test tail/graph roughness decreases
  test margin/accuracy improves

frozen_head:
  head margin may improve
  but K_t, tail, graph roughness should stay fixed
```

If train metrics improve while test metrics do not, classify the result as
memorization or weak transfer.

## Commands

Quick smoke:

```bash
python3 experiments/013-pretrained-finetune-metric-dynamics/scripts/run_finetune_dynamics.py \
  --quick --device auto
```

Substantive tmux run:

```bash
tmux new-session -d -s cntk-013-finetune-dyn \
  'cd /Users/ziegel/Documents/CNTK && python3 experiments/013-pretrained-finetune-metric-dynamics/scripts/run_finetune_dynamics.py --binary-n-per-class 50 --multi-n-per-class 25 --epochs 6 --batch-size 16 --eval-batch-size 32 --device auto'
```

## Artifacts

- `metrics.json`
- `result.md`
- `figures/test_tail_over_time.png`
- `figures/test_graph_dirichlet_over_time.png`
- `figures/test_mixing_over_time.png`
- `figures/test_head_margin_over_time.png`
- `figures/test_ridge_margin_over_time.png`
- `figures/kernel_movement_vs_tail_decrease.png`
