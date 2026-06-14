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
| 017 | `017-full-finetune-schedule-control/result.md` | Schedule control: `layer4_base` repairs `9/9`; `all_aug` does not improve over `all_base` repair/overmove (`0.11/0.89`); low-LR full variants repair `0.00` and overmove `1.00`. |
| 018 | `018-full-finetune-bn-control/result.md` | BatchNorm control: default `all_bn_train` overmoves (`0.11/0.89` repair/overmove), while `all_bn_eval` repairs `9/9` with mean tail/graph delta `-0.045/-0.124`. |
| 019 | `019-bn-frozen-robustness/result.md` | Larger-subset BN-frozen robustness: `all_bn_eval` repairs `9/9`; `all_bn_eval_aug` also repairs `9/9` and has mean tail/graph delta `-0.065/-0.160`; BN-train variants still overmove. |

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
- Simple schedule changes do not rescue full fine-tuning: crop/flip
  augmentation and a lower full-backbone LR fail to reduce held-out
  tail/graph roughness. The next mechanism control should isolate BatchNorm
  statistics from weight-gradient movement.
- BatchNorm/stat-mode dynamics explain the ResNet18 full-fine-tune failure in
  these runs: freezing BN running stats turns full fine-tuning from over-move
  into the strongest metric-repair variant.
- The BN-frozen full-fine-tune repair survives a larger subset and is compatible
  with crop/flip augmentation; augmentation alone still does not rescue
  BN-train full fine-tuning.
- Local mixing/graph roughness retain signal after controlling for global
  alignment, so they are not merely alignment under another name.
