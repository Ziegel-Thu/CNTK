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
