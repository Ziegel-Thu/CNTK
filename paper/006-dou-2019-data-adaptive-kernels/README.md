# Training Neural Networks as Learning Data-Adaptive Kernels

arXiv: <https://arxiv.org/abs/1901.07114>

## Files

- `paper.pdf`
- `source.tar`
- `source/`

## Notes

## Project Notes

Closest connection:

- This paper frames neural-network training as learning an adaptive RKHS/kernel,
  with representation and approximation benefits relative to fixed bases.
- It is directly aligned with the project's claim that feature learning changes
  the effective metric/function space.

Gap relative to this project:

- It is mostly an adaptive-RKHS approximation theory story, not a finite-sample
  diagnostic framework for real class pairs.
- It does not make local label mixing, graph roughness, or `T_y(m)` the
  measurable obstruction variables.

How to use it:

- Cite when explaining why source/RKHS norm and adaptive kernels are natural
  consequences to study.
- Experiments `009` and `012` are the empirical companion: source norm is useful
  only when kernel family and regularization are controlled.
