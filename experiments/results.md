# Result Index

Date: 2026-06-08

## Experiments

| ID | Result | Main Signal |
| --- | --- | --- |
| 001 | `001-spectral-tail-diagnostics/result.md` | Toy corr(opposite-kNN, tail@10%) = `0.926`; XOR caveat shows local mixing is not the only obstruction. |
| 001 | `001-spectral-tail-diagnostics/result_image_subsets.md` | MNIST/CIFAR binary corr(opposite-kNN, tail@10%) = `0.991`. |
| 001 | `001-spectral-tail-diagnostics/result_bound_audit.md` | Formal disjoint-collision bound is conservative; nonzero in only `4/150` audited rows. |
| 002 | `002-feature-metric-dynamics/result.md` | Toy feature learning helps two moons but not intrinsic collision pairs. |
| 002 | `002-feature-metric-dynamics/result_mnist.md` | MNIST feature learning transfers: `3 vs 8` test tail `0.350 -> 0.085`, `4 vs 9` `0.402 -> 0.142`. |
| 002 | `002-feature-metric-dynamics/result_cifar.md` | CIFAR raw MLP mostly memorizes: cat/dog test tail `0.886 -> 0.881`. |
| 002 | `002-feature-metric-dynamics/result_cifar_cnn.md` | CIFAR CNN features transfer better than raw MLP: automobile/truck test tail `0.751 -> 0.631`. |
| 003 | `003-fixed-representation-sweep/result.md` | Fixed-representation corr(test mix, test tail) = `0.988`; corr(test tail, probe acc) = `-0.971`. |
| 004 | `004-intrinsic-collision-stress/result_mnist_stress.md` | Label noise separates train-tail collapse from clean transfer; exact contradictory duplicates cap train accuracy. |
| 005 | `005-multiclass-obstruction-diagnostics/result.md` | Multiclass extension works: corr(test disagreement, multiclass tail) = `0.960`; corr(tail, probe acc) = `-0.925`. |
| 006 | `006-cifar-multiclass-schedule-sweep/result.md` | Stronger CIFAR schedule lowers multiclass tail and improves probe accuracy across all-10, animals6, and vehicles4. |
| 007 | `007-margin-tail-audit/result.md` | Existing-metrics audit: corr(test tail, accuracy) = `-0.855`; corr(tail decrease, margin gain) = `0.918`. |
| 008 | `008-graph-energy-kernel-margin/result.md` | Graph diagnostics and kernel margin: corr(graph Dirichlet, tail) = `0.955`; corr(tail, kernel ridge margin) = `-0.964`. |

## Current Read

- The obstruction story is no longer just CNTK or binary MNIST/CIFAR pairs; it
  now covers fixed representations and multiclass label subspaces.
- Local mixing is a strong empirical diagnostic, but XOR and bound-audit results
  show that global spectral misalignment and theorem conservatism must be kept
  separate.
- Feature learning is useful when it improves held-out tail/mixing and accuracy;
  train-only tail collapse is memorization.
- Margin should be tracked alongside tail/mixing, especially when comparing
  nearby CIFAR representations where tail ranges are compressed.
- Graph Dirichlet energy is now a first-class local-mixing diagnostic; source
  norm still needs a controlled same-kernel audit.
