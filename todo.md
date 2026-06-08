# TODO

## P0 - Project Setup

- [x] Create required directory structure.
- [x] Add GPT/Codex project instructions.
- [x] Add `progress.md` and `todo.md`.
- [x] Add initial experiment folders and plans.
- [x] Add initial deep research survey folder.
- [x] Initialize git repository if desired.
- [ ] Add remote and verify push permissions.
- [ ] Decide whether to move large data/results outside the repo.

## P0 - Prior Art Survey

- [x] Run first-pass web survey for close existing work.
- [x] Download arXiv TeX/source for priority papers into `paper/`.
- [ ] For each priority paper, add `README.md` and project-specific notes.
- [ ] DeepResearch pass: ask for a focused novelty map around fixed metric
  obstruction, local label mixing, spectral label complexity, and feature
  learning metric dynamics.

## P0 - Shared Code

- [x] Implement `src/datasets.py` for toy datasets. MNIST/CIFAR loaders still pending.
- [x] Implement `src/kernels.py` for RBF, Laplace, random features, empirical feature
  Gram, empirical NTK if feasible.
- [x] Implement `src/spectral.py`: eigenspectrum, `E_y(m)`, `T_y(m)`,
  kernel-target alignment.
- [x] Implement `src/mixing.py`: opposite-label kNN, local label entropy,
  collision graph, graph Dirichlet energy.
- [x] Implement `src/gradient_flow.py` from eigendecomposition.
- [x] Implement `src/models.py` for small MLP and feature hooks.
- [x] Implement `src/plotting.py` with consistent file naming.
- [x] Add tests for spectral and graph diagnostics on small synthetic matrices.

## P1 - Experiments

- [x] `001-spectral-tail-diagnostics`: toy-only quick run.
- [x] `001-spectral-tail-diagnostics`: MNIST/CIFAR subset fixed-kernel diagnostics.
- [x] `001-spectral-tail-diagnostics`: theorem-bound audit for collision lower bounds.
- [x] `002-feature-metric-dynamics`: toy quick run tracking feature Gram,
  `T_y^{K_t}(m)`, collisions, alignment, margin.
- [x] `002-feature-metric-dynamics`: replace lazy-ish control with stricter SGD
  small-LR/wider lazy control.
- [x] `002-feature-metric-dynamics`: MNIST run tracking `K_t`,
  `T_y^{K_t}(m)`, collisions, alignment, margin.
- [x] `002-feature-metric-dynamics`: CIFAR raw-pixel MLP run after MNIST dynamics are clean.
- [x] `002-feature-metric-dynamics`: CIFAR small-CNN run after raw-pixel MLP
  shows mostly memorization.
- [x] `003-fixed-representation-sweep`: quick sweep over raw pixels, RFF,
  random/trained MLP, and random/trained small CNN features.
- [ ] `003-fixed-representation-sweep`: add pretrained/self-supervised features
  and larger fine-tuned backbones.
- [ ] `004-intrinsic-collision-stress`: label noise / duplicated opposite-label
  samples / adversarial local collisions.

## P1 - Run Discipline

- [ ] Run substantive experiments in tmux sessions named in `experiments/index.md`.
- [ ] Record exact commands in each `result.md`.
- [ ] Commit and push meaningful results once git/remote are configured.

## P1 - Engineering Hygiene

- [ ] Add environment setup instructions once dependencies are chosen.
- [ ] Add a small smoke test command that runs without downloading full datasets.
- [ ] Add a result index linking each experiment result.
- [ ] Standardize artifact names: `metrics.json`, `figures/*.png`, `result.md`.

## P2 - Theory/Measurement Bridge

- [ ] Formalize how collision graph statistics lower-bound spectral tail in
  measurable finite-sample settings.
- [ ] Derive sharper/non-vacuous finite-sample bounds or explain why the current
  disjoint-pair corollary is only a conservative sufficient obstruction.
- [ ] Relate spectral tail to kernel gradient-flow time to reach a training loss.
- [ ] Relate fixed-metric collision/mixing to margin and RKHS norm proxies.
- [ ] Separate metric mismatch from intrinsic label ambiguity.
- [ ] Add fixed-metric obstruction taxonomy: local collision obstruction vs
  global/nonlinear spectral misalignment obstruction.

## P2 - Optional Extensions

- [ ] Empirical NTK for small networks/subsets if feature Gram results are strong.
- [ ] Pretrained/self-supervised feature backbones for experiment `003`.
- [ ] Multi-class versions of local mixing and spectral tail beyond binary tasks.
