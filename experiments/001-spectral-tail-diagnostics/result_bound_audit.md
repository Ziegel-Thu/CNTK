# Theorem-Bound Audit Result

## Run

Command: `python experiments/001-spectral-tail-diagnostics/scripts/run_bound_audit.py --n 300 --n-per-class 150 --seed 0 --include-cifar`

## Summary

- total audited rows = `150`
- nonzero formal-bound rows = `4`
- max bound/actual-tail ratio = `0.182`
- mean bound/actual-tail ratio = `0.004`
- corr(best collision rate, actual tail) = `0.295`

Interpretation:

- The formal corollary-style lower bound is useful only when kernel-edge
  lengths are extremely small relative to `lambda_m/n`; otherwise it is
  conservative or zero.
- This particular greedy disjoint-collision-rate proxy is weaker than the
  opposite-kNN/local-entropy diagnostics used in the main experiments.
- Future experiments should report the formal bound as a sufficient
  obstruction check, but use richer local-mixing diagnostics for prediction.

Top formal bounds:

| dataset | kernel | m/n | actual tail | bound | bound/actual | collision rate | rho |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| collision_pairs_sep0.03 | rbf_narrow | 0.05 | 1.000 | 0.182 | 0.182 | 0.150 | 0.045 |
| collision_pairs_sep0.03 | rff_1024 | 0.05 | 1.000 | 0.153 | 0.153 | 0.150 | 0.020 |
| collision_pairs_sep0.03 | rbf_median | 0.05 | 1.000 | 0.141 | 0.141 | 0.150 | 0.020 |
| collision_pairs_sep0.03 | rbf_narrow | 0.10 | 1.000 | 0.088 | 0.088 | 0.150 | 0.045 |
| two_moons_noise0.2 | linear | 0.05 | 0.461 | 0.000 | 0.000 | 0.000 | 0.000 |
| two_moons_noise0.2 | linear | 0.10 | 0.449 | 0.000 | 0.000 | 0.000 | 0.000 |
| two_moons_noise0.2 | linear | 0.20 | 0.428 | 0.000 | 0.000 | 0.000 | 0.000 |
| two_moons_noise0.2 | linear | 0.30 | 0.414 | 0.000 | 0.000 | 0.000 | 0.000 |
| two_moons_noise0.2 | linear | 0.50 | 0.378 | 0.000 | 0.000 | 0.000 | 0.000 |
| two_moons_noise0.2 | rbf_median | 0.05 | 0.208 | 0.000 | 0.000 | 0.000 | 0.000 |
| two_moons_noise0.2 | rbf_median | 0.10 | 0.116 | 0.000 | 0.000 | 0.000 | 0.000 |
| two_moons_noise0.2 | rbf_median | 0.20 | 0.069 | 0.000 | 0.000 | 0.000 | 0.000 |

## Artifacts

- `metrics_bound_audit.json`
- `figures/bound_vs_actual.png`
