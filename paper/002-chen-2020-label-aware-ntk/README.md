# Label-Aware Neural Tangent Kernel

arXiv: <https://arxiv.org/abs/2010.11775>

## Files

- `paper.pdf`
- `source.tar`
- `source/`

## Notes

## Project Notes

Closest connection:

- This paper directly addresses the gap between label-agnostic NTKs and real
  neural networks by injecting label information into kernel construction.
- It is highly relevant because it treats local elasticity and generalization as
  consequences of label-aware kernel structure.

Gap relative to this project:

- The kernel is modified using labels; this project diagnoses a given fixed
  metric/representation before adding label information.
- It does not center opposite-label collision statistics, graph Dirichlet
  energy, or spectral tail as measurable obstruction variables.
- It is closer to "design a better kernel" than "detect when any fixed metric is
  structurally bad, and measure how feature learning changes that metric."

How to use it:

- Cite as the strongest adjacent work in label-aware/local NTK improvement.
- Position our contribution as label-free diagnostics plus metric-dynamics
  measurements, not as another label-aware kernel design.
- Useful comparison point if adding a future experiment that explicitly
  constructs label-aware kernels and checks whether they reduce `T_y(m)`.
