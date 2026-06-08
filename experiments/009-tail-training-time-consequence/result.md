# Result

## Run

Command: `python3 experiments/009-tail-training-time-consequence/scripts/run_tail_time.py --n 160 --n-per-class 80 --seed 0 --residual-fraction 0.1 --time-min 0.01 --time-max 1000000.0 --time-steps 320`

## Summary

- residual target = `0.1`
- rows hitting residual target within grid = `27/36`
- corr(tail@10%, log10 training time) = `0.596`
- corr(kNN opposite ratio, log10 training time) = `0.398`
- corr(graph Dirichlet, log10 training time) = `0.382`
- corr(alignment, log10 training time) = `-0.193`
- corr(tail@10%, source norm proxy) = `0.514`

Interpretation:

- This directly tests the consequence link from spectral tail to static
  kernel gradient-flow time.
- Source norm is included as a proxy, but should be read by kernel family
  because mixed kernel scales can compress or inflate it.
- Tail is the cleaner consequence predictor than local mixing alone: XOR with a
  linear kernel has almost no local mixing but still has high tail and fails to
  hit the residual target.
- Collision-pair tasks and linear two-moons show the slow-training failure mode
  most starkly: several runs remain above the residual target even at
  `time_max=1e6`.

| dataset | kernel | tail@10% | mix | graph dir | alignment | source norm | log10 time | hit | final residual |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| cifar10_automobilevstruck_train_n80 | laplace | 0.736 | 0.437 | 0.883 | 0.106 | 15.71 | 2.49 | yes | 0 |
| cifar10_automobilevstruck_train_n80 | linear | 0.662 | 0.431 | 0.878 | 0.096 | 28.70 | 3.04 | yes | 6.22e-05 |
| cifar10_automobilevstruck_train_n80 | rbf | 0.695 | 0.437 | 0.881 | 0.090 | 27.40 | 3.00 | yes | 8.11e-242 |
| cifar10_automobilevstruck_train_n80 | rff_512 | 0.804 | 0.446 | 0.910 | 0.087 | 33.86 | 3.18 | yes | 1.27e-121 |
| cifar10_catvsdog_train_n80 | laplace | 0.831 | 0.482 | 0.970 | 0.068 | 16.78 | 2.53 | yes | 0 |
| cifar10_catvsdog_train_n80 | linear | 0.780 | 0.476 | 0.946 | 0.027 | 57.86 | 3.21 | yes | 0.00211 |
| cifar10_catvsdog_train_n80 | rbf | 0.816 | 0.482 | 0.969 | 0.044 | 32.34 | 3.14 | yes | 5e-170 |
| cifar10_catvsdog_train_n80 | rff_512 | 0.885 | 0.497 | 1.002 | 0.041 | 41.43 | 3.34 | yes | 1.32e-89 |
| collision_pairs_sep0.03 | laplace | 1.000 | 0.550 | 1.103 | 0.001 | 98.81 | 4.06 | yes | 9.12e-48 |
| collision_pairs_sep0.03 | linear | 0.952 | 0.548 | 1.077 | 0.000 | 999.87 | 6.00 | no | 1 |
| collision_pairs_sep0.03 | rbf | 1.000 | 0.550 | 1.086 | 0.000 | 998.52 | 6.00 | no | 0.995 |
| collision_pairs_sep0.03 | rff_512 | 1.000 | 0.549 | 1.087 | 0.000 | 999.12 | 6.00 | no | 0.997 |
| collision_pairs_sep0.25 | laplace | 0.992 | 0.556 | 1.103 | 0.005 | 46.39 | 3.41 | yes | 2.53e-48 |
| collision_pairs_sep0.25 | linear | 0.909 | 0.542 | 1.075 | 0.000 | 999.88 | 6.00 | no | 1 |
| collision_pairs_sep0.25 | rbf | 0.986 | 0.556 | 1.102 | 0.000 | 971.30 | 6.00 | no | 0.925 |
| collision_pairs_sep0.25 | rff_512 | 0.987 | 0.555 | 1.091 | 0.000 | 980.18 | 6.00 | no | 0.951 |
| mnist_3vs8_train_n80 | laplace | 0.230 | 0.186 | 0.400 | 0.206 | 10.97 | 2.16 | yes | 0 |
| mnist_3vs8_train_n80 | linear | 0.227 | 0.163 | 0.349 | 0.310 | 15.51 | 2.25 | yes | 2.93e-05 |
| mnist_3vs8_train_n80 | rbf | 0.244 | 0.186 | 0.391 | 0.245 | 16.19 | 2.43 | yes | 6.74e-154 |
| mnist_3vs8_train_n80 | rff_512 | 0.282 | 0.217 | 0.433 | 0.239 | 19.73 | 2.54 | yes | 1.93e-76 |
| mnist_4vs9_train_n80 | laplace | 0.227 | 0.286 | 0.608 | 0.161 | 11.74 | 2.21 | yes | 0 |
| mnist_4vs9_train_n80 | linear | 0.204 | 0.252 | 0.535 | 0.196 | 58.85 | 2.21 | yes | 0.00325 |
| mnist_4vs9_train_n80 | rbf | 0.250 | 0.286 | 0.605 | 0.177 | 17.53 | 2.43 | yes | 4.48e-143 |
| mnist_4vs9_train_n80 | rff_512 | 0.322 | 0.307 | 0.635 | 0.156 | 20.64 | 2.55 | yes | 7.56e-78 |
| two_moons_noise0.05 | laplace | 0.012 | 0.001 | 0.002 | 0.518 | 5.79 | 1.61 | yes | 2.21e-28 |
| two_moons_noise0.05 | linear | 0.422 | 0.221 | 0.440 | 0.428 | 690.96 | 6.00 | no | 0.477 |
| two_moons_noise0.05 | rbf | 0.040 | 0.001 | 0.002 | 0.556 | 142.20 | 3.03 | yes | 0.00776 |
| two_moons_noise0.05 | rff_512 | 0.039 | 0.001 | 0.004 | 0.530 | 176.53 | 3.21 | yes | 0.0193 |
| two_moons_noise0.2 | laplace | 0.073 | 0.037 | 0.082 | 0.564 | 11.50 | 1.74 | yes | 5.26e-72 |
| two_moons_noise0.2 | linear | 0.331 | 0.175 | 0.344 | 0.549 | 585.14 | 6.00 | no | 0.342 |
| two_moons_noise0.2 | rbf | 0.131 | 0.037 | 0.085 | 0.603 | 279.65 | 4.47 | yes | 0.0713 |
| two_moons_noise0.2 | rff_512 | 0.138 | 0.040 | 0.092 | 0.607 | 313.04 | 5.36 | yes | 0.0814 |
| xor_noise0.25 | laplace | 0.012 | 0.000 | 0.000 | 0.351 | 4.81 | 1.21 | yes | 3.66e-47 |
| xor_noise0.25 | linear | 0.932 | 0.000 | 0.000 | 0.007 | 994.70 | 6.00 | no | 0.989 |
| xor_noise0.25 | rbf | 0.017 | 0.000 | 0.000 | 0.153 | 128.50 | 1.69 | yes | 0.0144 |
| xor_noise0.25 | rff_512 | 0.017 | 0.000 | 0.000 | 0.157 | 131.06 | 1.68 | yes | 0.0158 |

## Artifacts

- `metrics.json`
- `figures/tail_vs_time.png`
- `figures/tail_vs_source_norm.png`
- `figures/mixing_vs_time.png`
- `figures/residual_curves.png`
