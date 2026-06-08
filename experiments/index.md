# Experiment Index

## Execution Status

| ID | Name | Priority | Status | Main Question | Resource |
| --- | --- | --- | --- | --- | --- |
| 001 | spectral-tail-diagnostics | P0 | toy + image subsets + bound audit complete | Does local mixing predict spectral tail and slow kernel flow? | CPU first, GPU optional |
| 002 | feature-metric-dynamics | P1 | toy + MNIST + CIFAR MLP/CNN complete | Does feature learning reduce mixing/tail by changing `K_t`? | GPU preferred |
| 003 | fixed-representation-sweep | P2 | quick sweep complete | Is this fixed-metric general, beyond CNTK? | GPU for feature extraction |
| 004 | intrinsic-collision-stress | P2 | MNIST stress complete | What mixing is correctable vs intrinsic/noisy? | CPU/GPU |

## Standard Artifact Contract

Each experiment should produce:

- `plan.md`
- `result.md`
- `metrics.json` or `metrics_over_time.json`
- `figures/*.png`
- exact command lines in `result.md`

## Standard Result Questions

Every `result.md` should answer:

- What was run?
- What datasets/subsets/seeds were used?
- What diagnostics moved?
- Did local mixing explain anything beyond global alignment?
- Did the result support, weaken, or refine the project hypothesis?
- What should be run next?

## tmux Sessions

Use these names:

- `cntk-001-spectral-tail`
- `cntk-002-feature-dynamics`
- `cntk-003-fixed-repr`
- `cntk-004-collision-stress`
