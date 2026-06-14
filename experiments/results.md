# Result Index

Date: 2026-06-10

## Experiments

| ID | Result | Main Signal |
| --- | --- | --- |
| 001 | `001-spectral-tail-diagnostics/result.md` | Toy corr(opposite-kNN, tail@10%) = `0.926`; XOR caveat shows local mixing is not the only obstruction. |
| 001 | `001-spectral-tail-diagnostics/result_image_subsets.md` | MNIST/CIFAR binary corr(opposite-kNN, tail@10%) = `0.991`. |
| 001 | `001-spectral-tail-diagnostics/result_bound_audit.md` | Formal disjoint-collision bound is conservative; nonzero in only `4/150` audited rows. |
| 002 | `002-feature-metric-dynamics/result.md` | Toy feature learning helps two moons but not intrinsic collision pairs. |
| 002 | `002-feature-metric-dynamics/result_margin_curves.md` | Margin curves promoted for dynamics: corr(final test tail, final test margin) = `-0.623`; corr(test margin, test acc) = `0.559`. |
| 002 | `002-feature-metric-dynamics/result_mnist.md` | MNIST feature learning transfers: `3 vs 8` test tail `0.350 -> 0.085`, `4 vs 9` `0.402 -> 0.142`. |
| 002 | `002-feature-metric-dynamics/result_cifar.md` | CIFAR raw MLP mostly memorizes: cat/dog test tail `0.886 -> 0.881`. |
| 002 | `002-feature-metric-dynamics/result_cifar_cnn.md` | CIFAR CNN features transfer better than raw MLP: automobile/truck test tail `0.751 -> 0.631`. |
| 003 | `003-fixed-representation-sweep/result.md` | Fixed-representation corr(test mix, test tail) = `0.988`; corr(test tail, probe acc) = `-0.971`. |
| 004 | `004-intrinsic-collision-stress/result.md` | Label noise separates train-tail collapse from clean transfer; exact contradictory duplicates cap train accuracy. |
| 005 | `005-multiclass-obstruction-diagnostics/result.md` | Multiclass extension works: corr(test disagreement, multiclass tail) = `0.960`; corr(tail, probe acc) = `-0.925`. |
| 006 | `006-cifar-multiclass-schedule-sweep/result.md` | Stronger CIFAR schedule lowers multiclass tail and improves probe accuracy across all-10, animals6, and vehicles4. |
| 007 | `007-margin-tail-audit/result.md` | Existing-metrics audit: corr(test tail, accuracy) = `-0.855`; corr(tail decrease, margin gain) = `0.918`. |
| 008 | `008-graph-energy-kernel-margin/result.md` | Graph diagnostics and kernel margin: corr(graph Dirichlet, tail) = `0.955`; corr(tail, kernel ridge margin) = `-0.964`. |
| 009 | `009-tail-training-time-consequence/result.md` | Consequence audit: corr(tail, log training time) = `0.596`; corr(tail, source norm proxy) = `0.514`. |
| 010 | `010-pretrained-fixed-representation-sweep/result.md` | Frozen ImageNet ResNet18 lowers CIFAR all-10 tail `0.788 -> 0.450` and raises ridge acc `0.310 -> 0.765` vs raw pixels. |
| 011 | `011-self-supervised-fixed-representation-sweep/result.md` | DINO ViT-S/16 lowers CIFAR all-10 tail `0.799 -> 0.432` and raises ridge acc `0.287 -> 0.807` vs raw pixels. |
| 012 | `012-source-norm-controlled-sweep/result.md` | Same-kernel source-norm sweep: RBF/Laplace/RFF corr(tail, source norm) = `0.757-0.794`; corr(source norm, ridge margin) = `-0.903` to `-0.947`. |
| 013 | `013-pretrained-finetune-metric-dynamics/result.md` | ResNet18 dynamics: `finetune_layer4` lowers mean test tail `-0.024` and graph `-0.072`; `finetune_all` moves more but worsens tail `+0.121`. |
| 014 | `014-mixing-alignment-controlled-audit/result.md` | Local diagnostics are not just alignment: image rows partial corr(tail, mixing \| alignment) = `0.661`; partial corr(tail, graph \| alignment) = `0.906`. |
| 015 | `015-resnet18-finetune-multiseed-simple/result.md` | Local 3-seed ResNet18 probe: `finetune_layer4` mean tail/graph delta `-0.020/-0.048`, repair rate `0.67`; `finetune_all` mean `+0.064/+0.131`, overmove rate `0.89`. |
| 016 | `016-resnet18-cloud-single-gpu/result.md` | Single-A40 5-seed ResNet18 probe: `finetune_layer4` mean tail/graph delta `-0.030/-0.094`, repair rate `1.00`; `finetune_all` mean `+0.041/+0.084`, overmove rate `0.80`. |

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
  norm is useful when reported within a fixed kernel/regularization context.
- Spectral tail now connects to exact static-kernel gradient-flow time; local
  mixing is best read as one cause of high tail, not the only one.
- The fixed-representation claim now includes a real supervised pretrained
  backbone and a self-supervised DINO backbone.
- Fine-tuning dynamics now distinguish useful metric repair from excessive
  metric movement: partial ResNet18 fine-tuning helps held-out geometry, while
  full fine-tuning can worsen it on small subsets.
- The simple 3-seed local rerun preserves the same qualitative fine-tuning
  story: partial `layer4` tuning is the stable candidate, while full fine-tuning
  is an over-move risk.
- The first single-GPU cloud rerun strengthens the fine-tuning dynamics result:
  `layer4` repair holds in every seed-task row, while full fine-tuning remains
  an over-move risk.
- Local mixing/graph roughness retain signal after controlling for global
  alignment, so they are not merely alignment under another name.
