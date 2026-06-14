# Current Findings

Date: 2026-06-09

## One-Line State

The project now has evidence for a broader fixed-metric obstruction story:

```text
local label mixing under a fixed metric
=> high binary or multiclass spectral label tail
=> worse fixed-representation linear/kernel behavior
=> feature learning can lower tail/mixing when the structure is learnable
=> but noisy or contradictory labels create train/test gaps or irreducible train obstruction
```

Claim discipline:

- See `claims.md` before writing summaries or new experiment plans. The
  non-trivial part is the diagnostic chain and its controls, not merely running
  MNIST/CIFAR or showing pretrained features work.

## 001 - Static Diagnostics

Toy diagnostics:

- Result: `experiments/001-spectral-tail-diagnostics/result.md`
- corr(`opposite-kNN ratio`, `tail@10%`) = `0.926`
- corr(`alignment`, `tail@10%`) = `-0.871`
- Caveat: XOR with a linear kernel has high tail without local mixing, so local
  mixing is one obstruction type, not the only fixed-metric obstruction.

MNIST/CIFAR image subsets:

- Result: `experiments/001-spectral-tail-diagnostics/result_image_subsets.md`
- corr(`opposite-kNN ratio`, `tail@10%`) = `0.991`
- corr(`alignment`, `tail@10%`) = `-0.905`
- CIFAR `cat vs dog` is the highest-tail/highest-mixing case under raw-pixel
  fixed kernels.

Theorem-bound audit:

- Result: `experiments/001-spectral-tail-diagnostics/result_bound_audit.md`
- Formal disjoint-collision bound is nonzero in only `4/150` audited rows.
- Max formal bound / actual tail ratio is `0.182`.
- Conclusion: current theorem is a conservative sufficient obstruction check;
  empirical opposite-kNN/local entropy diagnostics are stronger predictors.

## 002 - Feature Metric Dynamics

Toy:

- Result: `experiments/002-feature-metric-dynamics/result.md`
- Margin supplement:
  `experiments/002-feature-metric-dynamics/result_margin_curves.md`
- Two moons: feature learning reduces tail/mixing and improves accuracy.
- Synthetic opposite-label collision pairs: feature movement alone does not solve
  intrinsic contradictions.
- Across existing 002 dynamics rows, corr(final test tail, final test margin) =
  `-0.623` and corr(final test margin, final test accuracy) = `0.559`, so
  margin is now a first-class dynamics curve rather than only a final audit.

MNIST:

- Result: `experiments/002-feature-metric-dynamics/result_mnist.md`
- `3 vs 8`: feature learning reduces test `tail@10%` from `0.350` to `0.085`.
- `4 vs 9`: feature learning reduces test `tail@10%` from `0.402` to `0.142`.
- Frozen features stay fixed; strict lazy control has tiny feature movement.
- This is the cleanest evidence that metric adaptation transfers beyond train
  samples.

CIFAR raw-pixel MLP:

- Result: `experiments/002-feature-metric-dynamics/result_cifar.md`
- Train tail collapses, but test tail mostly does not improve.
- `cat vs dog`: test tail remains `0.886 -> 0.881`.
- Interpretation: raw MLP feature learning mostly memorizes CIFAR subsets.

CIFAR small CNN:

- Result: `experiments/002-feature-metric-dynamics/result_cifar_cnn.md`
- `cat vs dog`: test tail improves slightly `0.861 -> 0.834`, test acc `0.604`.
- `automobile vs truck`: test tail improves `0.751 -> 0.631`, test acc `0.729`.
- Interpretation: CNN inductive bias improves transferable metric adaptation
  relative to raw MLP.

## 003 - Representation Sweep

- Result: `experiments/003-fixed-representation-sweep/result.md`
- corr(test opposite-kNN ratio, test `tail@10%`) = `0.988`
- corr(test `tail@10%`, linear-probe test accuracy) = `-0.971`

Interpretation:

- The diagnostic generalizes beyond CNTK/kernels to fixed representation
  families.
- Good fixed representations tend to have lower test local mixing, lower test
  spectral tail, and higher linear-probe test accuracy.
- CIFAR remains high-tail/high-mixing for small fixed representations.

## 004 - Stress Tests

- Result: `experiments/004-intrinsic-collision-stress/result_mnist_stress.md`

Key observations:

- Clean feature learning: clean test tail drops to `0.085`, test acc `0.947`.
- `10%` random flips: still partially transfers, test acc `0.933`.
- `30%` random flips: train tail collapses but clean test tail worsens to
  `0.288`, test acc `0.670`.
- `30%` adversarial local flips: clean test tail improves to `0.169`, but clean
  accuracy remains low at `0.663`; tail/mixing must be read with clean
  accuracy/margin.
- Exact opposite-label duplicates: train acc caps near `0.833`, train tail
  remains high, but clean test tail improves. This separates intrinsic
  contradiction from correctable metric mismatch.

## 005 - Multiclass Diagnostics

- Result: `experiments/005-multiclass-obstruction-diagnostics/result.md`
- corr(test kNN disagreement, test multiclass `tail@10%`) = `0.960`
- corr(test normalized local entropy, test multiclass `tail@10%`) = `0.911`
- corr(test multiclass `tail@10%`, linear-probe test accuracy) = `-0.925`

Interpretation:

- The fixed-metric obstruction signal extends beyond binary pairs when labels
  are represented as a centered one-hot subspace.
- MNIST multiclass trained MLP features reduce test tail and improve/maintain
  probe accuracy, which supports correctable metric mismatch.
- CIFAR multiclass remains high-tail/high-entropy for short small-model runs.
  Some trained-CNN results are not better than random-CNN probes, so the next
  engineering question is stronger multiclass training, seeds, and pretrained
  features rather than another binary pair.

## 006 - CIFAR Multiclass Schedule Sweep

- Result: `experiments/006-cifar-multiclass-schedule-sweep/result.md`
- `strong_minibatch` improves test tail and probe accuracy on all three CIFAR
  multiclass task families relative to random CNN features:
  - all-10: tail `0.772 -> 0.731`, probe acc `0.272 -> 0.392`;
  - animals6: tail `0.826 -> 0.777`, probe acc `0.319 -> 0.373`;
  - vehicles4: tail `0.755 -> 0.724`, probe acc `0.428 -> 0.497`.
- Within-CIFAR correlations are weaker than the broad 005 sweep:
  - corr(test disagreement, test multiclass `tail@10%`) = `0.500`;
  - corr(test entropy, test multiclass `tail@10%`) = `0.613`;
  - corr(test multiclass `tail@10%`, probe acc) = `-0.634`.
- Probe margin median correlates strongly with probe accuracy (`0.826`), making
  margin a useful companion diagnostic when tail ranges are narrow.

Interpretation:

- The weak CIFAR multiclass transfer in 005 was partly a schedule artifact:
  stronger optimization improves both geometry and probe accuracy.
- The improvement is still modest, so the next engineering step should be
  larger/more-seed GPU runs or pretrained/self-supervised features.

## 007 - Margin/Tail Audit

- Result: `experiments/007-margin-tail-audit/result.md`
- final-row corr(test tail, accuracy) = `-0.855`
- final-row corr(margin median, accuracy) = `0.689`
- final-row corr(test tail, margin median) = `-0.723`
- final-row corr(local mixing, test tail) = `0.885`
- corr(test tail decrease, margin gain) = `0.918`

Interpretation:

- Tail remains the stronger global obstruction/accuracy correlate across the
  mixed existing-metrics collection.
- Margin is still useful because it tracks confidence/optimization and aligns
  strongly with tail decrease over feature dynamics.
- Going forward, tail and local mixing should describe geometry; margin should
  describe whether that geometry is translating into a usable classifier.

## 008 - Graph Energy and Kernel Margin

- Result: `experiments/008-graph-energy-kernel-margin/result.md`
- corr(test kNN opposite ratio, test `tail@10%`) = `0.961`
- corr(test graph disagreement, test `tail@10%`) = `0.955`
- corr(test graph Dirichlet, test `tail@10%`) = `0.955`
- corr(test `tail@10%`, kernel ridge test margin median) = `-0.964`
- corr(kernel ridge margin median, test accuracy) = `0.958`

Interpretation:

- The local mixing metric family now includes graph-level disagreement and
  Dirichlet energy, not only nearest-neighbor ratios.
- Kernel ridge margin gives a direct classifier consequence for high spectral
  tail.
- Source/RKHS norm proxies are not ready as a headline: in the mixed-kernel
  comparison they were weaker, likely due to feature/kernel scale and
  regularization differences.

## 009 - Tail to Training-Time Consequence

- Result: `experiments/009-tail-training-time-consequence/result.md`
- residual target: normalized residual `<= 0.1`
- rows hitting target within grid: `27/36`
- corr(`tail@10%`, log10 training time) = `0.596`
- corr(kNN opposite ratio, log10 training time) = `0.398`
- corr(graph Dirichlet, log10 training time) = `0.382`
- corr(alignment, log10 training time) = `-0.193`
- corr(`tail@10%`, source norm proxy) = `0.514`

Interpretation:

- Spectral tail now has an explicit static-kernel consequence: slower exact
  gradient-flow residual decay.
- Local mixing and graph energy are routes into high tail, but tail itself is
  the cleaner consequence predictor because XOR-like global misalignment also
  creates slow learning without local mixing.
- Source norm moves in the expected direction here, but source-norm claims still
  need same-kernel/regularization-controlled sweeps.

## 010 - Pretrained Fixed Representation Sweep

- Result: `experiments/010-pretrained-fixed-representation-sweep/result.md`
- frozen backbone: ImageNet ResNet18 from `torchvision`
- corr(test local mixing/disagreement, test `tail@10%`) = `0.729`
- corr(test graph Dirichlet, test `tail@10%`) = `0.952`
- corr(test `tail@10%`, kernel ridge test accuracy) = `-0.794`
- corr(test `tail@10%`, kernel ridge test margin median) = `-0.848`

Representative improvements from raw pixels to ImageNet ResNet18:

- CIFAR all-10: tail `0.788 -> 0.450`, ridge acc `0.310 -> 0.765`.
- CIFAR animals6: tail `0.826 -> 0.475`, ridge acc `0.308 -> 0.662`.
- CIFAR vehicles4: tail `0.701 -> 0.353`, ridge acc `0.388 -> 0.838`.
- CIFAR automobile/truck: tail `0.719 -> 0.261`, ridge acc `0.606 -> 0.906`.
- CIFAR cat/dog: tail `0.900 -> 0.455`, ridge acc `0.575 -> 0.794`.

Interpretation:

- The project now has a real pretrained frozen-feature result, not only raw
  pixels, small CNNs, or random features.
- A better fixed metric lowers local mixing, graph energy, spectral tail, and
  improves margin/accuracy without any feature learning on the CIFAR subsets.
- This strongly supports the broader "fixed representation obstruction" framing.

## 011 - Self-Supervised Fixed Representation Sweep

- Result: `experiments/011-self-supervised-fixed-representation-sweep/result.md`
- frozen backbone: DINO ViT-S/16 from `facebookresearch/dino`
- corr(test local mixing/disagreement, test `tail@10%`) = `0.839`
- corr(test graph Dirichlet, test `tail@10%`) = `0.990`
- corr(test `tail@10%`, kernel ridge test accuracy) = `-0.947`
- corr(test `tail@10%`, kernel ridge test margin median) = `-0.885`

Representative improvements from raw pixels to DINO:

- CIFAR all-10: tail `0.799 -> 0.432`, ridge acc `0.287 -> 0.807`.
- CIFAR animals6: tail `0.825 -> 0.475`, ridge acc `0.317 -> 0.711`.
- CIFAR vehicles4: tail `0.731 -> 0.228`, ridge acc `0.500 -> 0.917`.
- CIFAR automobile/truck: tail `0.749 -> 0.244`, ridge acc `0.533 -> 0.950`.
- CIFAR cat/dog: tail `0.780 -> 0.385`, ridge acc `0.550 -> 0.867`.

Interpretation:

- The fixed-representation scope now includes self-supervised frozen features.
- DINO often improves over supervised ResNet18 in this run, suggesting the
  diagnostic is representation-family agnostic: good frozen metrics reduce
  local mixing, graph roughness, spectral tail, and improve margin/accuracy.

## 012 - Controlled Source-Norm Sweep

- Result: `experiments/012-source-norm-controlled-sweep/result.md`
- Laplace: corr(`tail@10%`, source norm) = `0.794`; corr(source norm, ridge
  margin median) = `-0.947`.
- RBF: corr(`tail@10%`, source norm) = `0.769`; corr(source norm, ridge margin
  median) = `-0.918`.
- RFF-512: corr(`tail@10%`, source norm) = `0.757`; corr(source norm, ridge
  margin median) = `-0.903`.
- Linear: corr(`tail@10%`, source norm) = `0.573`, but corr(local mixing,
  source norm) is approximately zero.

Interpretation:

- Source norm has become usable as a consequence diagnostic, but only with
  kernel family, normalization, and regularization stated.
- The nonlinear-kernel rows support the expected chain:
  local mixing/graph roughness -> high spectral tail -> high source norm ->
  lower ridge margin.
- The linear-kernel rows are useful rather than embarrassing: XOR-like global
  nonlinear misalignment raises tail/source norm without local collisions,
  keeping the obstruction taxonomy honest.

## 013 - Pretrained Fine-Tune Metric Dynamics

- Result: `experiments/013-pretrained-finetune-metric-dynamics/result.md`
- Backbone: ImageNet ResNet18.
- Tasks: CIFAR `cat vs dog`, `automobile vs truck`, and `vehicles4`.
- `frozen_head`: mean test movement `0.000`; mean head accuracy delta `+0.407`.
- `finetune_layer4`: mean test movement `0.431`; mean test tail delta `-0.024`;
  mean graph Dirichlet delta `-0.072`.
- `finetune_all`: mean test movement `0.649`; mean test tail delta `+0.121`;
  mean graph Dirichlet delta `+0.144`.

Interpretation:

- This is the first pretrained feature-learning metric-dynamics run, rather
  than a frozen-feature sweep.
- The important result is not "fine-tuning helps" in general. Partial
  fine-tuning (`layer4`) gives modest held-out metric repair, while full
  fine-tuning moves the metric more but worsens held-out tail/graph roughness.
- This supports the project rule that metric movement is not enough; useful
  feature learning means held-out tail/roughness falls and margin/accuracy stay
  healthy.

## 014 - Mixing Versus Alignment Controlled Audit

- Result: `experiments/014-mixing-alignment-controlled-audit/result.md`
- Image fixed-representation rows: corr(`tail`, `mixing`) = `0.770`;
  partial corr(`tail`, `mixing` | `alignment`) = `0.661`.
- Image fixed-representation rows: corr(`tail`, graph Dirichlet) = `0.963`;
  partial corr(`tail`, graph Dirichlet | `alignment`) = `0.906`.
- Standardized regression on image rows:
  `tail ~ mixing + alignment` has `R2=0.792`, with coefficients
  `mixing=0.479`, `alignment=-0.533`.

Interpretation:

- Local mixing and graph roughness are not merely global alignment under another
  name.
- The right claim remains bounded: local roughness is one diagnosable source of
  spectral tail, while XOR/global mismatch remains a separate obstruction.

## 015 - Simple ResNet18 Fine-Tune Multi-Seed Probe

- Result: `experiments/015-resnet18-finetune-multiseed-simple/result.md`
- Local MPS run with 3 seeds, CIFAR `cat vs dog`, `automobile vs truck`, and
  `vehicles4`.
- `frozen_head`: mean movement `0.000`, mean head accuracy delta `+0.407`.
- `finetune_layer4`: mean movement `0.422`, mean test tail delta `-0.020`,
  mean graph Dirichlet delta `-0.048`, metric-repair rate `0.67`.
- `finetune_all`: mean movement `0.663`, mean test tail delta `+0.064`, mean
  graph Dirichlet delta `+0.131`, overmove rate `0.89`.

Interpretation:

- This strengthens the qualitative `013` finding: partial fine-tuning is the
  more reliable metric-repair candidate, while full fine-tuning often moves the
  metric in a way that worsens held-out geometry on small subsets.
- The result is still a simple local probe, not a cloud-scale final answer.
  Larger subsets, more seeds, stronger augmentation, and additional backbones
  are the right use of cloud compute.

## 016 - Single-GPU Cloud ResNet18 Fine-Tune Probe

- Result: `experiments/016-resnet18-cloud-single-gpu/result.md`
- Cloud run on `jiagpu8` with one A40, 5 seeds, CIFAR `cat vs dog`,
  `automobile vs truck`, and `vehicles4`.
- `frozen_head`: mean movement `0.000`, mean head accuracy delta `+0.428`.
- `finetune_layer4`: mean movement `0.479`, mean test tail delta `-0.030`,
  mean graph Dirichlet delta `-0.094`, metric-repair rate `1.00`.
- `finetune_all`: mean movement `0.689`, mean test tail delta `+0.041`, mean
  graph Dirichlet delta `+0.084`, overmove rate `0.80`.

Interpretation:

- This is the strongest fine-tuning dynamics evidence so far: partial
  fine-tuning repairs held-out geometry in every seed-task row.
- Full fine-tuning remains valuable as a negative control, because it moves the
  representation more but usually worsens held-out tail/graph roughness.
- The next non-trivial cloud experiment should change mechanism, not just scale:
  augmentation/schedule controls, DINO/ViT backbones, or explicit cases where
  feature learning should fail.

## 017 - Full Fine-Tune Schedule Control

- Result: `experiments/017-full-finetune-schedule-control/result.md`
- Cloud run on `jiagpu8` with one A40, 3 seeds, CIFAR `cat vs dog`,
  `automobile vs truck`, and `vehicles4`.
- `layer4_base`: mean movement `0.486`, mean test tail delta `-0.029`, mean
  graph Dirichlet delta `-0.093`, repair rate `1.00`, overmove rate `0.00`.
- `all_base`: mean movement `0.684`, mean test tail delta `+0.040`, mean graph
  delta `+0.074`, repair rate `0.11`, overmove rate `0.89`.
- `all_aug`: mean movement `0.683`, mean test tail delta `+0.037`, mean graph
  delta `+0.080`, repair rate `0.11`, overmove rate `0.89`.
- `all_low_lr`: mean movement `0.697`, mean test tail delta `+0.056`, mean
  graph delta `+0.123`, repair rate `0.00`, overmove rate `1.00`.
- `all_aug_low_lr`: mean movement `0.694`, mean test tail delta `+0.054`, mean
  graph delta `+0.117`, repair rate `0.00`, overmove rate `1.00`.

Interpretation:

- The full fine-tuning over-move result is not rescued by simple crop/flip
  augmentation or a lower full-backbone LR.
- Lower LR does not reduce measured metric movement, which suggests the next
  mechanism control should isolate BatchNorm/stat-mode movement from
  weight-gradient movement.
- This strengthens the negative side of the metric-dynamics story: feature
  learning needs the right metric dynamics, not just more trainable parameters
  or ordinary augmentation.

## 018 - Full Fine-Tune BatchNorm Control

- Result: `experiments/018-full-finetune-bn-control/result.md`
- Cloud run on `jiagpu8` with one A40, 3 seeds, CIFAR `cat vs dog`,
  `automobile vs truck`, and `vehicles4`.
- `layer4_base`: mean movement `0.486`, mean test tail delta `-0.029`, graph
  delta `-0.093`, repair/overmove `1.00/0.00`.
- `all_bn_train`: default full fine-tune control, mean movement `0.684`, mean
  test tail delta `+0.040`, graph delta `+0.075`, repair/overmove `0.11/0.89`.
- `all_bn_eval`: full weight-gradient fine-tuning with frozen BN running stats,
  mean movement `0.483`, mean test tail delta `-0.045`, graph delta `-0.124`,
  repair/overmove `1.00/0.00`.
- `all_layer4_bn_train`: intermediate result, mean movement `0.481`, mean test
  tail delta `-0.018`, graph delta `-0.057`, repair/overmove `0.67/0.33`.

Interpretation:

- This is the sharpest mechanism result so far. Full fine-tuning is not
  intrinsically bad; default full fine-tuning was bad because all BatchNorm
  running statistics were allowed to drift in train mode.
- Freezing BN running stats keeps full weight-gradient adaptation and turns it
  into a clean held-out metric repair.
- This refines the metric-dynamics claim: feature learning succeeds when the
  learned metric moves in the right subspace/mode; uncontrolled state dynamics
  can dominate and produce harmful metric movement.

## Current Working Taxonomy

1. Local collision obstruction:
   different labels are close in the fixed metric, pushing binary labels or
   centered one-hot label subspaces into spectral tail directions.

2. Global/nonlinear misalignment obstruction:
   labels are high-tail even without local opposite-label mixing, as in XOR with
   linear features.

3. Correctable metric mismatch:
   feature learning lowers train and test tail/mixing and improves clean test
   performance. Experiment `013` refines this: partial fine-tuning can repair
   held-out geometry, while full fine-tuning may move the metric in a harmful
   direction. Experiment `015` shows the same pattern survives a small local
   multi-seed rerun, and experiment `016` strengthens it on a single-GPU cloud
   run. Experiment `017` shows that simple augmentation/lower-LR controls do not
   rescue default full fine-tuning. Experiment `018` refines the diagnosis:
   full weight-gradient adaptation can repair the metric when BatchNorm running
   stats are frozen.

4. Memorization:
   train tail collapses but test tail/accuracy does not improve.

5. Intrinsic contradiction:
   deterministic models cannot fit identical inputs with opposite labels; train
   accuracy/tail remain obstructed even if clean test structure improves.

6. Margin/optimization companion:
   tail and mixing describe representation geometry, while margin captures
   confidence and optimization progress inside that geometry.

7. Graph-diagnostic layer:
   local mixing can be measured as a kNN graph roughness/Dirichlet-energy
   statistic, giving a bridge from pointwise collisions to spectral tail. The
   finite-sample bridge is now written in `theory.md` using the measurable
   quantity `beta_m(G,K) = ||B P_m||_op^2`. Experiment `014` confirms that
   graph roughness retains signal after controlling for global alignment.

8. Consequence layer:
   high spectral tail predicts slower static-kernel gradient-flow convergence;
   local mixing matters because it often pushes labels into those slow spectral
   directions. Source norm is another consequence proxy when evaluated within a
   fixed kernel context.

9. Fixed-pretrained-representation layer:
   pretrained frozen features can remove much of the obstruction by supplying a
   better metric before any task-specific training occurs.

10. Self-supervised-representation layer:
    self-supervised frozen features follow the same obstruction diagnostics,
    confirming that the framework is broader than supervised ImageNet features.

11. Source-norm consequence layer:
    source/RKHS norm proxies are meaningful within a fixed kernel family and
    regularization context; they should not be compared naively across kernels.

12. State-dynamics layer:
    in ResNet18 fine-tuning, BatchNorm running-stat updates can dominate metric
    movement. Freezing BN stats separates useful full weight adaptation from
    harmful state/stat drift.

## Next Best Experiments

- Add a sharper finite-sample bound or explicitly position the current theorem as
  a conservative sufficient condition.
- Add pretrained/self-supervised features to experiment 003.
- Run CIFAR small-CNN dynamics with more seeds and slightly stronger training to
  separate architecture effects from sample noise.
- Rerun multiclass CIFAR diagnostics with stronger schedules, multiple seeds,
  and pretrained/self-supervised features.
- Add the graph lower-bound audit from `theory.md` to experiments `001`, `008`,
  and `012`.
- Test whether the BatchNorm/stat-mode finding persists on larger subsets,
  with stronger augmentation, and across non-BN backbones such as ViT/DINO.
