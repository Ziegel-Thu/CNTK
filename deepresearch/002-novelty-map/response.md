# Focused Novelty Map

Date: 2026-06-09

## Short Answer

I do not see the full chain as already packaged in the priority literature:

```text
finite-sample local opposite-label mixing under a fixed metric
=> spectral label tail
=> training-time/source-norm/margin consequences
=> feature learning or better frozen features repair the metric
```

The closest works cover individual links. The project's strongest contribution
is therefore an engineering diagnostic framework that connects local metric
failure, spectral complexity, and feature-learning metric dynamics across
toy/MNIST/CIFAR/frozen pretrained features.

## Likely Prior Art

- Local elasticity: "The Local Elasticity of Neural Networks"
  (<https://arxiv.org/abs/1910.06943>) studies neural/local similarity through
  SGD response and NTK-like gradient inner products.
- Label-aware NTK: "Label-Aware Neural Tangent Kernel"
  (<https://arxiv.org/abs/2010.11775>) improves label-local behavior by adding
  label information to the kernel.
- Spectral/task-model alignment: "Spectral Bias and Task-Model Alignment in
  Kernel Regression" (<https://arxiv.org/abs/2006.13198>) gives the main
  language for kernel eigenspaces, target power, and generalization.
- Feature versus lazy training: "Disentangling Feature and Lazy Training in Deep
  Neural Networks" (<https://arxiv.org/abs/1906.08034>) supports the frozen/lazy
  versus feature-learning regime design.
- Silent alignment: "Neural Networks as Kernel Learners: The Silent Alignment
  Effect" (<https://arxiv.org/abs/2111.00034>) shows feature learning can alter
  the tangent kernel eigenstructure early.
- Data-adaptive kernels: "Training Neural Networks as Learning Data-Adaptive
  Kernels" (<https://arxiv.org/abs/1901.07114>) frames trained networks as
  adaptive RKHS/kernel learners.
- Function-space reshaping: "How Does Feature Learning Reshape the Function
  Space?" (<https://arxiv.org/abs/2605.17718>) is a close recent theory
  neighbor: feature learning induces target-dependent spectral deformation.
- Feature-learning lower bounds/advantages: "Backward Feature Correction"
  (<https://arxiv.org/abs/2001.04413>) supports the broad claim that trained
  features can overcome fixed-feature limitations.

## Closest Existing Claims

- Spectral alignment work already says target energy in low-eigenvalue kernel
  directions causes slow/poor kernel learning.
- Local elasticity and label-aware NTK already say neural/local label structure
  matters and static/random NTKs can miss it.
- Silent alignment and data-adaptive kernel work already say finite networks can
  learn an effective kernel rather than merely follow `K_0`.
- Recent function-space reshaping theory already says feature learning can
  deform eigenspaces toward target directions.

## Gaps Not Covered

- A finite-sample, computable local label mixing metric family is not the center
  of those works. The project's kNN opposite-label ratio, local entropy, graph
  disagreement, and graph Dirichlet energy fill this slot.
- The bridge from local collision/mixing to label spectral tail is not treated
  as the main empirical diagnostic.
- The consequence chain is not usually measured together:
  `T_y(m)`, exact kernel gradient-flow time, kernel-ridge margin, and
  source/RKHS norm under controlled same-kernel comparisons.
- Feature learning is often described through global alignment or theoretical
  model deformation. This project measures concrete metric dynamics:
  `d_{K_t}` changes, local opposite-label neighborhoods separate, tail drops,
  and margin rises.
- Negative cases are unusually important here. Experiments with label noise,
  contradictory duplicates, and XOR/global mismatch separate:
  correctable metric mismatch, intrinsic ambiguity, memorization, and
  global/nonlinear spectral misalignment.
- The pretrained/self-supervised fixed-representation sweeps turn the claim from
  "CNTK is weak" into "any fixed metric can be diagnosed; good fixed metrics can
  already remove much of the obstruction."

## Project Implications

- The project should not claim that local mixing is the only source of spectral
  tail. XOR with a linear kernel is the canonical caveat: global nonlinear
  mismatch can create tail/source norm without local collisions.
- The best headline is diagnostic, not benchmark:

  ```text
  fixed metrics can be audited by local mixing, graph roughness, spectral tail,
  gradient-flow time, source norm, and margin.
  ```

- The theory should present the collision theorem as a conservative sufficient
  obstruction, then use the graph-Dirichlet finite-sample bridge as the practical
  measurement layer.
- Source norm should be reported only within fixed kernel/regularization
  contexts; experiment `012` makes this credible.
- The feature-learning section should be written as metric dynamics:
  feature learning succeeds when held-out `K_t` has lower mixing/tail and better
  margin, and fails or memorizes when those improvements stay train-only.

## Papers To Read Next

- Original NTK/CNTK references for exact kernel baselines and convolutional NTK.
- Kernel-target alignment and centered alignment papers for comparison metrics.
- High-dimensional kernel limitation papers, especially random features or
  rotationally invariant kernels versus single-index tasks.
- Recent feature-learning theory around spiked covariance, one-step learning,
  and neural feature matrix spectra.
- Metric learning / representation geometry papers only if the project needs a
  broader ML audience; they are less central than the kernel/spectral papers.
