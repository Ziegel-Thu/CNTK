# Spectral Bias and Task-Model Alignment in Kernel Regression

arXiv: <https://arxiv.org/abs/2006.13198>

## Files

- `paper.pdf`
- `source.tar`
- `source/`

## Notes

## Project Notes

Closest connection:

- This is the main spectral-alignment precedent. It explains kernel regression
  generalization through eigenvalues/eigenfunctions and target power in the
  kernel basis.
- It justifies using cumulative label energy, spectral tail, and
  kernel-target alignment as first-class diagnostics.

Gap relative to this project:

- It largely treats the target's spectral decomposition as a property to
  analyze, while this project asks what geometric events create bad target
  decompositions.
- It does not isolate local opposite-label mixing or graph roughness as a
  finite-sample cause of tail energy.
- It does not track feature-learning metric dynamics `K_t` as a mechanism for
  reducing target spectral complexity.

How to use it:

- Cite as the conceptual backbone for `E_y(m)`, `T_y(m)`, and spectral learning
  speed.
- Make our novelty the bridge:
  local fixed-metric mixing -> spectral tail -> training/margin consequences.
- Keep the XOR caveat from experiments `001` and `012`: spectral tail can also
  arise from global nonlinear mismatch, not only local collisions.
