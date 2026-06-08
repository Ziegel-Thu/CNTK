# Current Findings

Date: 2026-06-08

## One-Line State

The project now has evidence for a broader fixed-metric obstruction story:

```text
local label mixing under a fixed metric
=> high binary or multiclass spectral label tail
=> worse fixed-representation linear/kernel behavior
=> feature learning can lower tail/mixing when the structure is learnable
=> but noisy or contradictory labels create train/test gaps or irreducible train obstruction
```

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
- Two moons: feature learning reduces tail/mixing and improves accuracy.
- Synthetic opposite-label collision pairs: feature movement alone does not solve
  intrinsic contradictions.

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

## Current Working Taxonomy

1. Local collision obstruction:
   different labels are close in the fixed metric, pushing binary labels or
   centered one-hot label subspaces into spectral tail directions.

2. Global/nonlinear misalignment obstruction:
   labels are high-tail even without local opposite-label mixing, as in XOR with
   linear features.

3. Correctable metric mismatch:
   feature learning lowers train and test tail/mixing and improves clean test
   performance.

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
   statistic, giving a bridge from pointwise collisions to spectral tail.

## Next Best Experiments

- Add a sharper finite-sample bound or explicitly position the current theorem as
  a conservative sufficient condition.
- Add pretrained/self-supervised features to experiment 003.
- Run CIFAR small-CNN dynamics with more seeds and slightly stronger training to
  separate architecture effects from sample noise.
- Rerun multiclass CIFAR diagnostics with stronger schedules, multiple seeds,
  and pretrained/self-supervised features.
- Add margin curves to all dynamics experiments, not only accuracy; 006 suggests
  margin is especially useful when tail differences are compressed.
