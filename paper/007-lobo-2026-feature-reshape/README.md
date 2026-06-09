# How Does Feature Learning Reshape the Function Space?

arXiv: <https://arxiv.org/abs/2605.17718>

## Files

- `paper.pdf`
- `source.tar`
- `source/`

## Notes

## Project Notes

Closest connection:

- This is a very close recent theoretical neighbor: early feature learning
  induces a target-dependent deformation of the kernel/function space and
  changes spectral structure.
- It strengthens the language of feature learning as metric/function-space
  reshaping rather than vague representation improvement.

Gap relative to this project:

- The setting is high-dimensional and model-specific; this project is an
  engineering diagnostic across toy/MNIST/CIFAR/fixed pretrained features.
- It studies target-aligned spectral deformation, but not local opposite-label
  mixing, graph Dirichlet energy, or finite-sample collision diagnostics.

How to use it:

- Cite as the closest "feature learning reshapes spectral geometry" theory.
- Position our work as the empirical diagnostic counterpart:
  fixed metric obstruction is visible before training, and feature learning is
  useful when it lowers measured mixing/tail/margin complexity.
