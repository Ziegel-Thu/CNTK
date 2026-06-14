# Claims and Non-Trivial Parts

Date: 2026-06-14

This file separates routine project support from claims that are genuinely
non-trivial and need to be defended.

## One-Sentence Discipline

Do not present the project as:

```text
we ran MNIST/CIFAR and CNTK was bad
```

Present it as:

```text
we built and stress-tested a diagnostic chain for fixed metrics:
local label roughness -> spectral tail -> kernel consequences,
and measured when learned or pretrained metrics reduce that obstruction.
```

## Routine Or Supporting Pieces

These are useful, but should not be sold as the main contribution by themselves:

- implementing RBF/Laplace/RFF/feature Gram kernels;
- computing eigenspectra, kernel-target alignment, and gradient-flow curves;
- running MNIST and CIFAR subsets;
- adding pretrained ResNet18 or DINO frozen features;
- reporting raw correlations without controls;
- adding project structure, artifact indexes, tmux discipline, or paper notes;
- showing that a better pretrained feature has better accuracy.

These pieces are infrastructure and evidence. They become important only when
they support a sharper claim.

## Non-Trivial Contribution Candidates

### 1. Local roughness as a diagnostic, not just global alignment

Non-trivial part:

- The project measures opposite-label local neighborhoods and graph roughness as
  a finite-sample cause/proxy for high spectral label tail.
- This is different from simply saying "kernel-target alignment is low."

Evidence:

- `001`, `003`, `005`, `008`, `010`, `011`, `014`.
- Graph Dirichlet energy tracks tail very strongly in several settings, e.g.
  `008` corr(graph Dirichlet, tail) = `0.955`.
- In `014`, image fixed-representation rows retain signal after controlling for
  alignment: partial corr(`tail`, `mixing` | `alignment`) = `0.661`, and partial
  corr(`tail`, graph Dirichlet | `alignment`) = `0.906`.

What must be said carefully:

- Local mixing is not the only obstruction. XOR with a linear kernel creates
  high tail without local collisions.
- The right claim is "local roughness is a major diagnosable obstruction mode",
  not "all spectral tail comes from local collisions."

### 2. The graph-to-tail measurement bridge

Non-trivial part:

- `theory.md` turns local graph roughness into a measurable finite-sample lower
  bound:

```text
sqrt(T_y(m)) >= max(0, sqrt(D_G(y)) - sqrt(beta_m(G,K))) / rho_G
```

where `beta_m(G,K) = ||B P_m||_op^2`.

Why this matters:

- It explains why graph Dirichlet energy is not just an arbitrary feature.
- It also explains why the earlier disjoint-collision theorem can be true but
  empirically conservative.

What must be said carefully:

- The inequality itself is an elementary projection/triangle-inequality bridge.
  The non-trivial project value is choosing the measurable quantities and using
  them to audit real fixed metrics.
- A future experiment should compute `beta_m` directly before making this a
  headline theorem claim.

### 3. Consequences beyond "tail is high"

Non-trivial part:

- The project connects spectral tail to three downstream consequences:
  gradient-flow time, kernel-ridge margin, and controlled source/RKHS norm.

Evidence:

- `009`: corr(tail, log gradient-flow time) = `0.596`.
- `008`: corr(tail, kernel ridge margin) = `-0.964`.
- `012`: within RBF/Laplace/RFF, corr(tail, source norm) = `0.757-0.794` and
  corr(source norm, margin) = `-0.903` to `-0.947`.

What must be said carefully:

- Source norm is not scale-free across kernels. It is usable only within a fixed
  kernel family and regularization context.
- Gradient-flow time is a static-kernel consequence, not automatically a deep
  network training-time theorem.

### 4. Metric dynamics, not vague feature learning

Non-trivial part:

- Feature learning is measured as a change in metric/Gram geometry `K_t`, not
  merely as improved accuracy.
- The project asks whether `K_t` lowers held-out mixing/tail and improves margin.

Evidence:

- `002` on MNIST shows transferable metric adaptation:
  `3 vs 8` test tail `0.350 -> 0.085`;
  `4 vs 9` test tail `0.402 -> 0.142`.
- CIFAR raw MLP shows a negative case: train tail collapses but test tail does
  not, so this is memorization rather than true metric repair.
- Margin curves in `002` now make margin a first-class dynamics diagnostic.
- `013` extends dynamics to ImageNet ResNet18 fine-tuning: `finetune_layer4`
  lowers mean test tail by `-0.024` and graph roughness by `-0.072`, while
  `finetune_all` moves more but worsens held-out tail by `+0.121`.
- `015` repeats the ResNet18 fine-tuning dynamics as a small local 3-seed probe:
  `finetune_layer4` has mean tail/graph deltas `-0.020/-0.048` and repair rate
  `0.67`, while `finetune_all` has mean deltas `+0.064/+0.131` and overmove
  rate `0.89`.

What must be said carefully:

- A moving metric is not automatically a useful metric.
- The non-trivial claim requires held-out improvement or a clean negative
  diagnosis.

### 5. Separating correctable mismatch from intrinsic ambiguity

Non-trivial part:

- The project distinguishes at least four failure modes:
  local collision obstruction, global nonlinear misalignment, memorization, and
  intrinsic contradiction.

Evidence:

- `004` label-noise and duplicate-opposite-label stress tests.
- XOR/linear cases in `001` and `012`.
- CIFAR MLP train/test gap in `002`.

What must be said carefully:

- This is a taxonomy supported by targeted controls, not a complete theory of
  all classification failures.

### 6. Fixed representation obstruction, broader than CNTK

Non-trivial part:

- The same diagnostic stack is applied to raw pixels, random features, random
  CNN features, trained small features, supervised pretrained features, and
  self-supervised features.

Evidence:

- `003`, `010`, `011`.
- DINO and ImageNet ResNet18 lower CIFAR multiclass tail and improve ridge
  accuracy without task-specific training.

What must be said carefully:

- "Better frozen features reduce the obstruction" is not surprising by itself.
  The contribution is that the same mixing/tail/graph/margin diagnostics explain
  the improvement across representation families.

## Not Yet Established

Do not claim these as done:

- a strong non-vacuous theorem that local mixing always lower-bounds spectral
  tail for arbitrary fixed metrics without a measured `beta_m` term;
- causal proof that reducing local mixing alone improves generalization;
- large-scale fine-tuning results;
- cloud-scale pretrained fine-tuning results on larger subsets/backbones;
- empirical NTK dynamics for trained networks;
- GPU-scale multi-seed CIFAR results;
- a complete novelty proof against all metric-learning literature.

## Claim Wording Rules

Use:

- "supports";
- "diagnoses";
- "separates";
- "is consistent with";
- "under controlled same-kernel comparisons";
- "one obstruction mode."

Avoid:

- "proves feature learning works";
- "local mixing is the cause of all spectral tail";
- "source norm is a universal cross-kernel metric";
- "CNTK obstruction" as the main scope;
- "pretrained features prove the theorem."

## Current Best Core Contribution

The current defensible contribution is:

```text
A finite-sample diagnostic framework for fixed metric classification:
local label graph roughness and spectral label tail identify when a fixed
representation makes the label function hard for kernel learning; the same
framework predicts margin/training-time/source-norm consequences, tracks metric
dynamics during feature learning, and separates correctable metric mismatch from
global mismatch, memorization, and intrinsic ambiguity.
```
