# Project Plan

## Project Objective

把当前想法组织成一个可复现实验项目：

```text
fixed metric local label mixing
=> spectral label tail / hard label function
=> slow static kernel learning and weak margin
=> feature learning changes metric
=> lower mixing/tail and better classification
```

重点不是再证明“CNTK 不如 CNN”，而是建立一套可以在 toy/MNIST/CIFAR/固定表征上复用的工程诊断。

## Execution Order

### P0 - Instrumentation First

先做共享代码，不直接跑大实验。

Required shared modules:

- `src/datasets.py`: toy, MNIST binary, CIFAR binary subset builders.
- `src/kernels.py`: RBF, Laplace, random feature, feature Gram, centered Gram.
- `src/spectral.py`: eigen decomposition, `E_y(m)`, `T_y(m)`, spectral AUC, alignment.
- `src/mixing.py`: opposite-label kNN, collision graph, local label entropy, `q_rho`.
- `src/gradient_flow.py`: exact kernel gradient-flow loss from spectrum.
- `src/models.py`: small MLP/CNN and feature extractor hooks.
- `src/plotting.py`: shared plotting style.

First smoke target:

```bash
pytest
python experiments/001-spectral-tail-diagnostics/scripts/run_toy.py --quick
```

### P1 - Static Metric Obstruction

#### Experiment 001: spectral-tail diagnostics

Folder: `experiments/001-spectral-tail-diagnostics/`

Question:

> Under fixed kernels, does local opposite-label mixing predict large spectral
> label tail and slow kernel gradient-flow loss decay?

Datasets:

- two moons / XOR / synthetic opposite-label collision pairs
- MNIST `3 vs 8`, `4 vs 9`
- CIFAR `cat vs dog`, `automobile vs truck`

Kernels:

- RBF
- Laplace
- random Fourier features
- random CNN feature Gram
- initial CNN feature Gram

Must produce:

- `metrics.json`
- `figures/mixing_vs_tail.png`
- `figures/tail_curves.png`
- `figures/gradient_flow_decay.png`
- `result.md`

Success signal:

- local mixing metrics correlate with spectral tail and predicted slow learning.

Failure signal:

- global alignment explains everything and local mixing adds no information.

#### Experiment 001b: theorem-bound audit

This is a sub-run inside experiment 001 unless it grows large.

Question:

> Is the collision lower bound numerically non-vacuous on toy/MNIST/CIFAR subsets?

Must produce:

- actual `T_y(m)`
- lower-bound/proxy curves from collision graph
- table of when the bound is nonzero/useful

### P2 - Feature-Learning Metric Dynamics

#### Experiment 002: feature metric dynamics

Folder: `experiments/002-feature-metric-dynamics/`

Question:

> Does feature learning reduce local mixing and spectral tail by changing the
> feature/kernel metric over training?

Regimes:

- frozen random features + train linear head
- lazy-ish wide network + small LR
- normal feature-learning network
- optional pretrained fine-tuning

Track at epoch 0, early, middle, final:

- `K_t`
- `||K_t - K_0||_F / ||K_0||_F`
- `T_y^{K_t}(m)`
- kernel-target alignment
- opposite-label kNN/collision stats
- margin distribution
- train/test accuracy

Must produce:

- `metrics_over_time.json`
- `figures/tail_over_time.png`
- `figures/mixing_over_time.png`
- `figures/kernel_movement_vs_accuracy.png`
- `result.md`

Success signal:

- feature-learning runs move the metric, reduce mixing/tail, and improve margin.
- frozen/lazy controls move much less.

### P3 - Scope Expansion and Controls

#### Experiment 003: fixed representation sweep

Folder: `experiments/003-fixed-representation-sweep/`

Question:

> Is this a fixed-metric phenomenon across representation families, not merely a
> CNTK phenomenon?

Representations:

- raw pixels
- random features
- random CNN features
- frozen trained CNN
- frozen pretrained/self-supervised features if available
- linear probe
- fine-tuned features

Must compare:

- local mixing
- spectral tail
- global alignment
- linear/kernel classifier performance
- margin

#### Experiment 004: intrinsic collision stress

Folder: `experiments/004-intrinsic-collision-stress/`

Question:

> When is local mixing correctable metric mismatch, and when is it intrinsic
> label ambiguity/noise?

Perturbations:

- random label noise
- duplicated samples with opposite labels
- near-duplicate opposite-label samples
- adversarially selected local collisions

Expected outcome:

- feature learning helps metric mismatch
- feature learning memorizes or fails under true contradictory labels

#### Experiment 005: multiclass obstruction diagnostics

Folder: `experiments/005-multiclass-obstruction-diagnostics/`

Question:

> Do local mixing and spectral tail diagnostics remain predictive for multiclass
> label subspaces, not only binary class pairs?

Diagnostics:

- centered one-hot label-subspace spectral tail;
- kNN disagreement ratio;
- normalized multiclass local label entropy;
- multiclass linear-probe train/test accuracy.

Expected outcome:

- disagreement/entropy correlate positively with label-subspace tail;
- label-subspace tail correlates negatively with linear-probe accuracy;
- CIFAR multiclass small-model runs reveal weak-transfer regimes that need
  stronger schedules or pretrained features.

#### Experiment 006: CIFAR multiclass schedule sweep

Folder: `experiments/006-cifar-multiclass-schedule-sweep/`

Question:

> Are the weak CIFAR multiclass results from experiment `005` a real
> architecture/metric limitation, or mostly a weak training-schedule artifact?

Regimes:

- random CNN features plus linear probe;
- short full-batch CNN training;
- stronger mini-batch CNN training with AdamW, cosine schedule, crop/flip
  augmentation, and multiple seeds.

Expected outcome:

- stronger schedules should lower test label-subspace tail and improve probe
  accuracy if weak transfer was partly an optimization artifact;
- if tail improves without accuracy or margin improvement, treat it as a
  diagnostic boundary and route to pretrained features.

#### Experiment 007: margin-tail audit

Folder: `experiments/007-margin-tail-audit/`

Question:

> Across existing dynamics and schedule metrics, is margin redundant with
> spectral tail, or a complementary diagnostic?

Expected outcome:

- final tail should remain a strong obstruction/accuracy correlate;
- margin should add performance-facing information, especially within narrower
  CIFAR-only comparisons and over training deltas.

#### Experiment 008: graph energy and kernel margin

Folder: `experiments/008-graph-energy-kernel-margin/`

Question:

> Can local label mixing be made into a metric-graph diagnostic that predicts
> both spectral tail and kernel classifier margin?

Diagnostics:

- kNN graph disagreement;
- signed/multiclass graph Dirichlet energy;
- closed-form kernel ridge margin;
- source/RKHS norm proxy.

Expected outcome:

- graph energy should track spectral tail;
- kernel ridge margin should decrease as tail increases;
- source-norm proxies need same-kernel audits before being used as a headline.

#### Experiment 009: tail to training-time consequence

Folder: `experiments/009-tail-training-time-consequence/`

Question:

> Does high spectral label tail under a static kernel predict slower exact
> kernel gradient-flow residual decay and larger source-norm proxies?

Diagnostics:

- exact residual curve `||exp(-tK/n)y||^2 / ||y||^2`;
- time to residual threshold;
- source norm proxy `sqrt(y^T(K + eps*nI)^-1y)`.

Expected outcome:

- tail should correlate positively with training time and source norm;
- local mixing should affect training time mostly through tail, while XOR-like
  global misalignment remains a separate obstruction.

#### Experiment 010: pretrained fixed representation sweep

Folder: `experiments/010-pretrained-fixed-representation-sweep/`

Question:

> Does the fixed-representation obstruction story hold for real frozen
> pretrained vision backbones?

Representations:

- raw pixels;
- random ResNet18 frozen features;
- ImageNet-pretrained ResNet18 frozen features.

Expected outcome:

- better frozen pretrained metrics should lower local mixing, graph energy, and
  spectral tail, with improved kernel ridge margin/accuracy.

#### Experiment 011: self-supervised fixed representation sweep

Folder: `experiments/011-self-supervised-fixed-representation-sweep/`

Question:

> Does the fixed-representation obstruction story hold for self-supervised
> frozen vision features?

Representations:

- raw pixels;
- supervised ImageNet ResNet18 frozen features;
- self-supervised DINO ViT-S/16 frozen features.

Expected outcome:

- self-supervised frozen features should fit the same diagnostic pattern:
  lower local mixing/graph energy/tail and higher kernel ridge margin/accuracy.

## Run Queue

1. Implement `src/spectral.py`, `src/mixing.py`, `src/kernels.py`.
2. Add tests for spectral/mixing metrics.
3. Run toy-only `001` locally.
4. Run MNIST/CIFAR subset static diagnostics.
5. Run theorem-bound audit.
6. Run `002` on toy and MNIST first.
7. Extend `002` to CIFAR only after MNIST dynamics are clean.
8. Run `003` fixed representation sweep.
9. Run `004` stress tests.
10. Run `005` multiclass diagnostics.
11. Run `006` CIFAR multiclass schedule sweep.
12. Run `007` margin-tail audit.
13. Run `008` graph energy/kernel margin diagnostics.
14. Run `009` tail-to-training-time consequence diagnostics.
15. Run `010` pretrained fixed representation sweep.
16. Run `011` self-supervised fixed representation sweep.

## Decision Rules

- If `001` fails to show any relation between mixing and tail, pause feature
  dynamics and debug the diagnostic.
- If `001` works but theorem lower bounds are vacuous, keep the theorem as
  motivation but emphasize empirical proxies.
- If `002` shows metric movement without tail reduction, add margin/local purity
  as primary outputs.
- If feature learning reduces tail only on train data but not test data, treat it
  as memorization and route into experiment `004`.

## Immediate Needs

- A git repo + remote if we want the documented commit/push rule to be active.
- Python environment with PyTorch, torchvision, numpy, scipy, scikit-learn,
  matplotlib, pandas, tqdm, pytest.
- Enough disk for MNIST/CIFAR cache.
- GPU for experiment `002` and later CIFAR runs; toy and static subset
  eigendecompositions can start on CPU.
- A user decision on whether pretrained/self-supervised features are in scope for
  experiment `003`.
