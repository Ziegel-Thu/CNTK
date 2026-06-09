# The Local Elasticity of Neural Networks

arXiv: <https://arxiv.org/abs/1910.06943>

## Files

- `paper.pdf`
- `source.tar`
- `source/`

## Notes

## Project Notes

Closest connection:

- This is the nearest prior work to the project's local-geometry language. It
  defines local elasticity through how an SGD update at one point changes
  predictions at nearby or distant points, and connects the induced similarity
  to NTK-style gradient inner products.
- It supports the idea that neural nets have local, semantic response structure
  that a static or label-agnostic kernel may miss.

Gap relative to this project:

- It does not formulate opposite-label nearest-neighbor collisions as an
  obstruction.
- It does not connect local mixing to spectral label tail `T_y(m)`, RKHS/source
  norm, kernel gradient-flow time, or margin.
- Its output is mainly a similarity/clustering story; this project turns local
  geometry into a diagnostic for fixed-metric classification failure.

How to use it:

- Cite as the main precedent for measuring local neural/kernel behavior.
- Contrast "local elasticity as a positive neural phenomenon" with this
  project's "local label mixing as a negative fixed-metric diagnostic."
- Use experiments `002`, `010`, and `011` to show that better learned or
  pretrained metrics produce the local separation that local elasticity would
  qualitatively predict.
