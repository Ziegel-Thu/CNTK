# Disentangling Feature and Lazy Training in Deep Neural Networks

arXiv: <https://arxiv.org/abs/1906.08034>

## Files

- `paper.pdf`
- `source.tar`
- `source/`

## Notes

## Project Notes

Closest connection:

- This paper gives an experimental and scaling vocabulary for lazy/kernel
  training versus feature-learning regimes.
- It motivates the project's regime controls: frozen features, lazy wide/small
  learning-rate models, and normal feature-learning models.

Gap relative to this project:

- It distinguishes regimes but does not provide a diagnostic for which data or
  class pairs should benefit from feature learning.
- It does not measure local label mixing, graph Dirichlet energy, spectral tail,
  source norm, or margin under `K_t`.

How to use it:

- Cite for the lazy-vs-feature-learning framing and for why `K_t` movement is a
  meaningful experimental object.
- Our addition is to measure what the moving kernel does to the label function:
  lower local mixing, lower tail, better margin, or train-only memorization.
