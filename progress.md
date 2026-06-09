# Progress

## 2026-06-09

- Ran `012-source-norm-controlled-sweep` in tmux to audit RKHS/source-norm
  proxies under controlled same-kernel comparisons:
  - Laplace: corr(`tail@10%`, source norm) = `0.794`, corr(source norm, ridge
    margin) = `-0.947`;
  - RBF: corr(`tail@10%`, source norm) = `0.769`, corr(source norm, ridge
    margin) = `-0.918`;
  - RFF-512: corr(`tail@10%`, source norm) = `0.757`, corr(source norm, ridge
    margin) = `-0.903`;
  - linear: tail still tracks margin (`-0.901`) but local mixing does not track
    source norm, preserving the XOR/global-misalignment caveat.
- Updated project indexes and TODOs so source norm is now positioned as a
  conditional consequence diagnostic: useful within a fixed kernel and
  regularization context, not scale-free across heterogeneous kernels.

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
- Ran `002-feature-metric-dynamics` on CIFAR-10 small CNNs:
  - CNN feature learning transfers better than raw MLP feature learning;
  - `cat vs dog` test tail improves slightly `0.861 -> 0.834`, test acc `0.604`;
  - `automobile vs truck` test tail improves `0.751 -> 0.631`, test acc `0.729`;
  - lazy CNN control has near-zero feature movement and no geometry improvement.
- Ran `003-fixed-representation-sweep` across raw pixels, RFF, random MLP,
  trained MLP, and CIFAR CNN features:
  - corr(test opposite-kNN ratio, test `tail@10%`) = `0.988`;
  - corr(test `tail@10%`, linear-probe test accuracy) = `-0.971`;
  - supports the broader fixed-representation obstruction framing beyond CNTK.
- Ran `004-intrinsic-collision-stress` on MNIST `3 vs 8`:
  - clean feature learning reduces clean test tail to `0.085`, test acc `0.947`;
  - `10%` label flips still partially transfer, test acc `0.933`;
  - `30%` label flips cause memorization: train tail `0.807 -> 0.008` but clean
    test tail worsens to `0.288`, test acc `0.670`;
  - adversarial local `30%` flips reduce clean test tail to `0.169` but clean
    test accuracy is only `0.663`, showing tail/mixing must be read with
    clean accuracy/margin;
  - exact opposite-label duplicates cap train acc near `0.833` and prevent full
    train-tail collapse, but clean test tail improves to `0.128`.
- Added `findings.md` as a project-level synthesis of current experimental
  evidence, caveats, taxonomy, and next experiments.
- Added shared multiclass diagnostics:
  - MNIST/CIFAR multiclass train/test subset builders;
  - centered one-hot label-subspace spectral tail/alignment;
  - multiclass kNN disagreement and normalized local label entropy;
  - multiclass MLP/CNN classifier heads.
- Ran `005-multiclass-obstruction-diagnostics` in tmux:
  - corr(test kNN disagreement, test multiclass `tail@10%`) = `0.960`;
  - corr(test normalized local entropy, test multiclass `tail@10%`) = `0.911`;
  - corr(test multiclass `tail@10%`, linear-probe test acc) = `-0.925`;
  - MNIST trained MLP features transfer (`all10` tail `0.410 -> 0.282`,
    `hard5` tail `0.465 -> 0.324`);
  - CIFAR multiclass remains high-tail/high-entropy for short small-model CPU
    runs, so the next scope-expansion run should use stronger schedules,
    multiple seeds, or pretrained features.
- Ran `006-cifar-multiclass-schedule-sweep` in tmux with two seeds on CIFAR
  all-10, animals6, and vehicles4:
  - `strong_minibatch` improves all-10 tail `0.772 -> 0.731` and probe acc
    `0.272 -> 0.392` relative to random CNN features;
  - animals6 tail improves `0.826 -> 0.777`, probe acc `0.319 -> 0.373`;
  - vehicles4 tail improves `0.755 -> 0.724`, probe acc `0.428 -> 0.497`;
  - within-CIFAR correlations are weaker than 005 because the tail range is
    narrower, but the signs stay aligned;
  - probe margin median correlates strongly with probe accuracy (`0.826`), so
    margin should be promoted in later dynamics runs.
- Added engineering hygiene artifacts:
  - `scripts/run_smoke_test.sh` for unit tests plus toy diagnostic sanity checks
    without dataset downloads or experiment artifact writes;
  - `experiments/results.md` as a compact completed-results index;
  - updated `README.md` setup/status notes.
- Ran `007-margin-tail-audit` on existing 002/004/006 metrics:
  - final-row corr(test tail, accuracy) = `-0.855`;
  - final-row corr(margin median, accuracy) = `0.689`;
  - final-row corr(local mixing, tail) = `0.885`;
  - corr(test tail decrease, margin gain) = `0.918`;
  - conclusion: margin is complementary rather than a replacement for tail;
    future dynamics results should report tail, mixing, accuracy, and margin
    together.
- Added graph-energy and kernel-ridge shared diagnostics:
  - kNN graph adjacency, graph disagreement, and graph Dirichlet energy;
  - binary/multiclass kernel ridge classifiers with margin and source-norm
    proxies.
- Ran `008-graph-energy-kernel-margin` on MNIST/CIFAR binary tasks:
  - corr(test graph disagreement, test `tail@10%`) = `0.955`;
  - corr(test graph Dirichlet, test `tail@10%`) = `0.955`;
  - corr(test `tail@10%`, kernel ridge test margin median) = `-0.964`;
  - corr(kernel ridge margin median, test accuracy) = `0.958`;
  - source-norm proxy was weak in the mixed-kernel comparison (`-0.267` with
    margin), so it needs a controlled same-kernel sweep.
- Ran `009-tail-training-time-consequence` on toy, MNIST, and CIFAR static
  kernels:
  - `27/36` rows hit normalized residual `<= 0.1` within the time grid;
  - corr(`tail@10%`, log10 training time) = `0.596`;
  - corr(kNN opposite ratio, log10 training time) = `0.398`;
  - corr(graph Dirichlet, log10 training time) = `0.382`;
  - corr(alignment, log10 training time) = `-0.193`;
  - corr(`tail@10%`, source norm proxy) = `0.514`;
  - interpretation: spectral tail is the cleaner consequence predictor, while
    local mixing is one route to high tail rather than the whole story.
- Installed `torchvision==0.22.1` in the user Python environment to match the
  existing `torch==2.7.1` runtime and enable pretrained backbone experiments.
- Ran `010-pretrained-fixed-representation-sweep` using frozen ImageNet
  ResNet18 features on CIFAR binary and multiclass tasks:
  - corr(test local mixing/disagreement, test `tail@10%`) = `0.729`;
  - corr(test graph Dirichlet, test `tail@10%`) = `0.952`;
  - corr(test `tail@10%`, kernel ridge test accuracy) = `-0.794`;
  - corr(test `tail@10%`, kernel ridge test margin median) = `-0.848`;
  - ImageNet ResNet18 improves every task family, e.g. CIFAR all-10 tail
    `0.788 -> 0.450` and ridge acc `0.310 -> 0.765` vs raw pixels.
- Downloaded DINO ViT-S/16 self-supervised weights through `torch.hub`
  (`facebookresearch/dino`) and ran
  `011-self-supervised-fixed-representation-sweep`:
  - corr(test local mixing/disagreement, test `tail@10%`) = `0.839`;
  - corr(test graph Dirichlet, test `tail@10%`) = `0.990`;
  - corr(test `tail@10%`, kernel ridge test accuracy) = `-0.947`;
  - corr(test `tail@10%`, kernel ridge test margin median) = `-0.885`;
  - DINO improves over raw pixels on every task, e.g. CIFAR all-10 tail
    `0.799 -> 0.432` and ridge acc `0.287 -> 0.807`.
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
