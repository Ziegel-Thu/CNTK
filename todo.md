# TODO

## P0 - Project Setup

- [x] Create required directory structure.
- [x] Add GPT/Codex project instructions.
- [x] Add `progress.md` and `todo.md`.
- [x] Add initial experiment folders and plans.
- [x] Add initial deep research survey folder.
- [x] Initialize git repository if desired.
- [x] Add remote and verify push permissions.
- [x] Decide whether to move large data/results outside the repo.

## P0 - Prior Art Survey

- [x] Run first-pass web survey for close existing work.
- [x] Download arXiv TeX/source for priority papers into `paper/`.
- [x] For each priority paper, add `README.md` and project-specific notes.
- [x] DeepResearch pass: ask for a focused novelty map around fixed metric
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
- [x] `010-pretrained-fixed-representation-sweep`: add ImageNet-pretrained
  frozen ResNet18 features on CIFAR binary/multiclass tasks.
- [x] `011-self-supervised-fixed-representation-sweep`: add DINO ViT-S/16
  frozen features on CIFAR binary/multiclass tasks.
- [x] `013-pretrained-finetune-metric-dynamics`: add pretrained ResNet18
  frozen-head, layer4 fine-tune, and full fine-tune metric dynamics.
- [x] `015-resnet18-finetune-multiseed-simple`: run local 3-seed MPS probe for
  pretrained ResNet18 fine-tune dynamics before using cloud compute.
- [x] `016-resnet18-cloud-single-gpu`: run single-A40 5-seed cloud probe for
  pretrained ResNet18 fine-tune dynamics.
- [x] `017-full-finetune-schedule-control`: test whether simple augmentation or
  lower full-backbone LR rescues full fine-tuning over-move.
- [x] `018-full-finetune-bn-control`: isolate BatchNorm/stat-mode dynamics in
  full ResNet18 fine-tuning.
- [x] `019-bn-frozen-robustness`: test BN-frozen full fine-tuning on larger
  subsets and with augmentation.
- [x] `020-vit-finetune-metric-dynamics`: test whether a no-BatchNorm
  ImageNet-pretrained ViT reproduces or avoids the ResNet full-fine-tune
  overmove mode.
- [ ] Add larger fine-tuned backbones.
- [x] `004-intrinsic-collision-stress`: label noise and duplicated opposite-label
  samples.
- [x] `004-intrinsic-collision-stress`: adversarially selected local collisions.
- [x] `005-multiclass-obstruction-diagnostics`: MNIST/CIFAR multiclass
  label-subspace tail, kNN disagreement, local entropy, and linear probes.
- [x] `006-cifar-multiclass-schedule-sweep`: rerun CIFAR multiclass with
  stronger CNN schedules and two seeds on CPU.
- [ ] `006-cifar-multiclass-schedule-sweep`: rerun stronger schedules on GPU
  with more seeds, larger subsets, and longer training.
- [x] `007-margin-tail-audit`: aggregate existing binary/multiclass metrics to
  test whether margin is complementary to tail.
- [x] `008-graph-energy-kernel-margin`: add graph Dirichlet/disagreement and
  closed-form kernel ridge margin diagnostics on MNIST/CIFAR binary tasks.
- [x] `008-graph-energy-kernel-margin`: run same-kernel ridge/source-norm sweep
  before making source norm a headline consequence.
- [x] `009-tail-training-time-consequence`: exact static-kernel gradient-flow
  residual time and source-norm consequence audit.
- [x] `014-mixing-alignment-controlled-audit`: audit whether local mixing/graph
  roughness add signal beyond global alignment.
- [ ] Run mechanism-changing cloud controls: self-supervised DINO/ViT
  fine-tuning and intrinsic-ambiguity negatives.

## P1 - Run Discipline

- [x] Run substantive experiments in tmux sessions named in `experiments/index.md`.
- [x] Record exact commands in each `result.md`.
- [x] Commit and push meaningful results once git/remote are configured.

## P1 - Engineering Hygiene

- [x] Add environment setup instructions once dependencies are chosen.
- [x] Add a small smoke test command that runs without downloading full datasets.
- [x] Add a result index linking each experiment result.
- [x] Standardize artifact names: `metrics.json`, `figures/*.png`, `result.md`.

## P2 - Theory/Measurement Bridge

- [x] Formalize how collision graph statistics lower-bound spectral tail in
  measurable finite-sample settings.
- [x] Derive sharper/non-vacuous finite-sample bounds or explain why the current
  disjoint-pair corollary is only a conservative sufficient obstruction.
- [x] Relate spectral tail to kernel gradient-flow time to reach a training loss.
- [x] Relate fixed-metric collision/mixing to kernel ridge margin empirically.
- [x] Relate fixed-metric collision/mixing to RKHS/source norm proxies in a
  controlled same-kernel sweep.
- [x] Audit margin against tail/accuracy using existing feature-dynamics and
  schedule metrics.
- [x] Promote margin curves to first-class diagnostics in future
  feature-dynamics plots and result tables.
- [x] Separate metric mismatch from intrinsic label ambiguity.
- [x] Add fixed-metric obstruction taxonomy: local collision obstruction vs
  global/nonlinear spectral misalignment obstruction.

## P2 - Optional Extensions

- [ ] Empirical NTK for small networks/subsets if feature Gram results are strong.
- [x] Pretrained supervised feature backbone sweep.
- [x] Self-supervised feature backbone sweep.
- [x] Multi-class versions of local mixing and spectral tail beyond binary tasks.
- [x] Extend multiclass diagnostics to supervised ImageNet-pretrained features.
- [x] Extend multiclass diagnostics to self-supervised features.
