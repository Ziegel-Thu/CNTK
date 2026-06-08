# Prompt

Survey whether the following project has already been done:

> Fixed kernels / fixed representations fail classification when their metric
> locally mixes opposite labels; this forces the label vector into spectral-tail
> directions, slowing kernel gradient flow and harming margin/generalization.
> Feature learning escapes by dynamically changing the kernel/feature metric,
> reducing opposite-label local mixing and spectral label complexity.

Please map close prior art around:

- neural tangent kernels and lazy training
- feature learning vs static kernels
- local elasticity / label-aware NTK
- kernel-target alignment and spectral bias
- representation geometry and metric learning
- fixed representations, linear probing, and fine-tuning

For each close paper, identify:

- what it already proves/measures
- whether it measures local opposite-label mixing
- whether it links mixing to spectral label tail
- whether it tracks feature metric dynamics during training
- what gap remains for this project
