# Result

## Run

Command: `python experiments/001-spectral-tail-diagnostics/scripts/run_toy.py --quick --n 300 --seed 0`

## Summary

Toy quick run completed. It computes fixed-kernel local mixing, spectral label tail,
kernel-target alignment, a disjoint-collision lower-bound proxy, and exact static
kernel gradient-flow residual curves.

Across the 25 toy dataset/kernel cases:

- corr(`opposite-kNN ratio`, `tail@10%`) = `0.926`
- corr(`alignment`, `tail@10%`) = `-0.871`

This is a strong first signal that local opposite-label mixing tracks spectral
tail in the synthetic collision cases.

Important caveat: XOR with a linear kernel has high spectral tail despite almost
zero local opposite-label mixing. This means local label mixing is a concrete
fixed-metric obstruction, not the only fixed-metric obstruction. The project
should keep a taxonomy: local collision obstruction vs global/nonlinear
misalignment obstruction.

Highest `T_y(m)` at top 10% eigendirections:

| dataset | kernel | tail@10% | tail_auc | opposite-kNN ratio | alignment | t_res<=0.1 | bound proxy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| collision_pairs_sep0.03 | laplace_median | 1.000 | 0.743 | 0.549 | 0.000 | NA | 0.000 |
| collision_pairs_sep0.03 | rbf_narrow | 1.000 | 0.664 | 0.549 | 0.000 | NA | 0.028 |
| collision_pairs_sep0.03 | rff_512 | 0.999 | 0.579 | 0.551 | 0.000 | NA | 0.000 |
| collision_pairs_sep0.03 | rbf_median | 0.999 | 0.607 | 0.549 | 0.000 | NA | 0.000 |
| collision_pairs_sep0.25 | rbf_narrow | 0.994 | 0.561 | 0.563 | 0.001 | NA | 0.002 |
| collision_pairs_sep0.25 | laplace_median | 0.992 | 0.568 | 0.563 | 0.003 | 7.06e+03 | 0.000 |
| collision_pairs_sep0.25 | rff_512 | 0.983 | 0.546 | 0.561 | 0.000 | NA | 0.000 |
| collision_pairs_sep0.25 | rbf_median | 0.982 | 0.541 | 0.563 | 0.000 | NA | 0.000 |

## Artifacts

- `metrics.json`
- `figures/tail_curves.png`
- `figures/mixing_vs_tail.png`
- `figures/gradient_flow_decay.png`

## Next

- Inspect whether the mixing-vs-tail scatter remains clean on MNIST/CIFAR binary subsets.
- Add the theorem-bound audit table across multiple `m` values.
- Use the same diagnostics in experiment 002 for feature metric dynamics.
