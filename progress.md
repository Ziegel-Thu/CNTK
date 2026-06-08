# Progress

## 2026-06-08

- Formalized the CNTK/fixed-metric obstruction project structure.
- Added project-level agent instructions in `AGENTS.md` and `GPT.md`.
- Added `README.md`, `todo.md`, and directory-level README files.
- Created initial deep research folder `deepresearch/001-existing-work-survey/`.
- Created initial experiment plans:
  - `experiments/001-spectral-tail-diagnostics/`
  - `experiments/002-feature-metric-dynamics/`
  - `experiments/003-fixed-representation-sweep/`
  - `experiments/004-intrinsic-collision-stress/`
- Added `PLAN.md`, `experiments/index.md`, and `requirements.txt` to make the
  execution queue, artifact contract, tmux session names, and resource needs
  explicit.
- Initialized a local git repository. No remote is configured yet, so pushing is
  still unavailable.
- Implemented first shared diagnostics in `src/`: datasets, kernels, spectral
  metrics, local mixing metrics, gradient-flow simulator, and plotting helpers.
- Added unit tests for spectral and mixing diagnostics; `python3 -m unittest
  discover -s tests` passes.
- Ran `001-spectral-tail-diagnostics` toy quick run in tmux:
  - corr(`opposite-kNN ratio`, `tail@10%`) = `0.926`
  - corr(`alignment`, `tail@10%`) = `-0.871`
  - found a useful caveat: XOR + linear kernel has high tail without local
    mixing, suggesting a second obstruction type (global/nonlinear misalignment).
- Implemented `src/models.py` and `002-feature-metric-dynamics/scripts/run_toy.py`.
- Ran `002-feature-metric-dynamics` toy quick run in tmux:
  - on two moons, feature learning reduced `tail@10%` from `0.082` to `0.053`
    and opposite-label kNN ratio from `0.044` to `0.028`;
  - on synthetic opposite-label collision pairs, feature learning moved the
    metric but did not reduce tail/mixing or improve accuracy, supporting the
    intrinsic-ambiguity stress-test direction;
- Re-ran `002-feature-metric-dynamics` with a stricter lazy control (wide MLP +
  SGD small LR): two-moons lazy feature movement dropped to `0.105`, while
  collision-pair lazy movement is near zero (`0.0002`).
- Implemented custom MNIST/CIFAR-10 loaders without requiring `torchvision`.
- Ran `001-spectral-tail-diagnostics` on MNIST and CIFAR binary subsets:
  - combined image-subset corr(`opposite-kNN ratio`, `tail@10%`) = `0.991`;
  - combined image-subset corr(`alignment`, `tail@10%`) = `-0.905`;
  - CIFAR `cat vs dog` has the largest tail/mixing under raw-pixel kernels
    (`tail@10%` up to `0.863`, opposite-kNN around `0.47`).
- Ran `001b` theorem-bound audit:
  - formal disjoint-collision bound is nonzero in only `4/150` audited rows;
  - max formal bound / actual tail ratio is `0.182`;
  - conclusion: current theorem is a conservative sufficient obstruction check,
    while richer local-mixing diagnostics are better empirical predictors.
- Ran `002-feature-metric-dynamics` on MNIST binary tasks with train/test
  feature-Gram diagnostics:
  - `3 vs 8`: feature learning reduces test `tail@10%` from `0.350` to `0.085`;
  - `4 vs 9`: feature learning reduces test `tail@10%` from `0.402` to `0.142`;
  - frozen features keep geometry fixed; strict lazy control has very small
    feature movement (`0.014` and `0.006`);
  - metric adaptation partially transfers to held-out MNIST subsets.
- Ran `002-feature-metric-dynamics` on CIFAR-10 raw-pixel MLPs:
  - feature learning collapses train tail but mostly does not improve test tail;
  - `cat vs dog` test tail remains `0.886 -> 0.881`, indicating memorization;
  - `automobile vs truck` test tail partially improves `0.785 -> 0.727`;
  - next CIFAR dynamics should use a small CNN inductive bias.
- Push is not available until a remote is configured.

## Current Framing

The project should be treated as an engineering study of fixed metric failure
and feature-learning metric adaptation, not merely as a CNTK benchmark project.

Working chain:

```text
fixed metric local label mixing
=> spectral label tail / high RKHS complexity
=> slow kernel gradient flow and poor margin
=> feature learning changes metric
=> lower tail, better local purity, better margin/accuracy
```
