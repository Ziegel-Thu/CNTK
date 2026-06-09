# Backward Feature Correction

arXiv: <https://arxiv.org/abs/2001.04413>

## Files

- `paper.pdf`
- `source.tar`
- `source/`

## Notes

## Project Notes

Closest connection:

- This paper is relevant as a provable feature-learning advantage story:
  training deeper layers can correct lower-level features and solve tasks that
  fixed or shallow representations cannot.
- It supports the broad claim that learned features can escape fixed-feature
  limitations.

Gap relative to this project:

- The result is a constructive/provable deep-learning mechanism, not a
  diagnostic for arbitrary fixed metrics.
- It does not use local label mixing, kernel spectral tail, graph energy, or
  margin/source-norm consequences as the explanatory variables.

How to use it:

- Cite as background for why feature learning can do more than static kernels.
- Keep it separate from the main novelty claim: our project is not proving a
  new deep feature-learning algorithm; it is measuring when fixed metrics are
  obstructed and whether learned metrics actually repair the obstruction.
