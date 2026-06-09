# Neural Networks as Kernel Learners: The Silent Alignment Effect

arXiv: <https://arxiv.org/abs/2111.00034>

## Files

- `paper.pdf`
- `source.tar`
- `source/`

## Notes

## Project Notes

Closest connection:

- This is a strong precedent for feature learning as kernel learning: `K_t`
  changes eigenstructure early, before much loss reduction, and later behaves
  like a data-dependent kernel machine.
- It supports the project's "metric dynamics" framing.

Gap relative to this project:

- It focuses on global kernel alignment/eigenstructure rather than local
  opposite-label collision removal.
- It does not connect the changing kernel to kNN label mixing, graph energy,
  label spectral tail, or kernel-ridge margin.

How to use it:

- Cite as evidence that moving beyond the lazy kernel can be understood as
  learning a better kernel.
- Our engineering contribution is the diagnostic layer on top of that: what
  exactly improved about the metric, and when did it fail to transfer?
