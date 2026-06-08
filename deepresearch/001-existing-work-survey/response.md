# Initial Existing-Work Survey

Date: 2026-06-08

## TL;DR

I did not find an exact prior-art match for the full engineering claim:

```text
opposite-label local collisions under a fixed kernel metric
=> lower bound / diagnostic for spectral label tail
=> slow static kernel learning or poor margin
=> feature learning reduces collisions/tail by changing the metric over time
```

But several nearby literatures cover important pieces. The project should be
positioned as an engineering diagnostic framework connecting those pieces, not
as a standalone observation that static NTKs underperform neural networks.

## Closest Prior-Art Clusters

### 1. Local Elasticity and Label-Aware NTK

Representative work:

- `he-2019-local-elasticity` / local elasticity of neural networks.
- `chen-2020-label-aware-ntk` / Label-Aware Neural Tangent Kernel.

What is close:

- Studies how neural predictors respond more strongly to semantically or
  label-related nearby examples.
- Introduces label-aware variants of NTK to improve local behavior.
- Directly adjacent to the question of local geometry and labels.

What seems missing:

- Does not appear to formalize opposite-label local collisions as a spectral-tail
  obstruction.
- Does not center `T_y(m)` / spectral label energy as the diagnostic.
- Label-aware kernels modify the kernel using labels; this project asks what
  feature learning does to the metric without directly injecting labels into the
  test-time kernel.

### 2. Spectral Bias, Kernel Alignment, and Task-Model Alignment

Representative work:

- `canatar-2020-spectral-bias-task-model-alignment`
- kernel-target alignment literature
- NTK eigenvalue/generalization analyses

What is close:

- Connects generalization and learning speed to alignment between target
  functions and kernel eigenspaces.
- Provides the right language for `E_y(m)`, `T_y(m)`, and gradient-flow speed.

What seems missing:

- Usually treats target spectral complexity as given.
- Does not isolate local opposite-label kernel collisions as a concrete
  geometric cause of label spectral-tail mass.

### 3. NTK Alignment During Training

Representative work:

- empirical/theoretical papers measuring how NTK changes or aligns during
  finite-width training.
- "silent alignment" style observations where networks improve alignment without
  obvious function movement.

What is close:

- Directly studies `K_t` rather than only `K_0`.
- Supports the idea that finite networks can move away from the lazy kernel.

What seems missing:

- The measured object is often global alignment, not local opposite-label
  collision removal.
- The project can add a mechanistic metric: `K_t` improves because it separates
  local opposite-label neighborhoods and reduces spectral tail.

### 4. Feature Learning vs Lazy Training

Representative work:

- `chizat-2019-lazy-training`
- papers separating kernel/lazy regimes from feature-learning regimes
- examples where neural networks beat kernel methods through learned features

What is close:

- Establishes that feature learning can be qualitatively different from static
  kernel learning.
- Provides experimental regimes: frozen, lazy, feature-learning.

What seems missing:

- A reusable diagnostic explaining when the difference should appear from sample
  geometry.
- Direct measurement of local label mixing and spectral-tail changes.

### 5. Kernel Methods Can Fail on Structured Classification

Representative work:

- high-dimensional Gaussian mixture / hidden-manifold examples where neural nets
  can outperform kernels.
- provable limitations of random features / kernel methods.

What is close:

- Gives negative results for fixed kernels and positive contrasts with learned
  features.

What seems missing:

- Often problem-specific and asymptotic.
- The current project can provide finite-sample diagnostics that can be computed
  on real datasets like MNIST/CIFAR subsets.

### 6. Metric Learning and Representation Geometry

Representative work:

- classical metric learning
- representation similarity / CKA
- neural collapse and class geometry

What is close:

- Studies learned representation geometry and class separation.

What seems missing:

- Usually not connected to NTK spectral tail or kernel gradient-flow speed.
- The project can translate representation geometry into kernel spectral
  complexity diagnostics.

## Current Novelty Hypothesis

The strongest project claim to test is:

> The key diagnostic is not just whether a kernel is globally aligned with
> labels, but whether its local metric creates opposite-label collisions. Those
> collisions force label energy into spectral-tail directions. Feature learning
> is valuable when it removes these collisions and thereby lowers the spectral
> complexity of the labels.

## Risk

The project may overlap substantially with:

- local elasticity if their measurements already imply opposite-label local
  separation,
- label-aware NTK if reviewers view this as another route to label-local kernels,
- spectral alignment work if the local-collision theorem is not empirically
  stronger than global alignment.

The first experiments should therefore compare local-mixing diagnostics against
plain kernel-target alignment and spectral tail, rather than reporting them in
isolation.

## Priority Papers to Download / Read

- Neural Tangent Kernel: convergence and generalization in neural networks.
- On exact computation with an infinitely wide neural net.
- Enhanced Convolutional Neural Tangent Kernels.
- Label-Aware Neural Tangent Kernel: Toward Better Generalization and Local
  Elasticity.
- Towards understanding the role of over-parametrization in generalization of
  neural networks / local elasticity related papers.
- Spectral bias and task-model alignment explain generalization in kernel
  regression and infinitely wide neural networks.
- The Neural Tangent Kernel in High Dimensions.
- Feature learning/lazy training separation papers.
- Provable kernel limitations / neural network feature-learning advantage papers.

## Engineering Implications

- Build local-mixing diagnostics as first-class shared code.
- Always report local mixing, spectral tail, and global alignment together.
- Track `K_t` over time in feature-learning experiments.
- Include negative controls with intrinsic label ambiguity.
