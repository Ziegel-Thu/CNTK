# Result

## Run

Command: `python3 experiments/012-source-norm-controlled-sweep/scripts/run_source_norm.py --n 160 --n-per-class 80 --ridge 0.001 --norm-ridge 1e-06 --seed 0`

## Summary

Correlations are reported within each kernel family to avoid mixing kernel
scales.

| kernel | rows | corr tail/source | corr mix/source | corr graph/source | corr source/margin | corr tail/margin | source mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| laplace | 19 | 0.794 | 0.774 | 0.773 | -0.947 | -0.898 | 28.15 |
| linear | 19 | 0.573 | -0.000 | -0.019 | -0.807 | -0.901 | 711.50 |
| rbf | 19 | 0.769 | 0.722 | 0.714 | -0.918 | -0.936 | 411.15 |
| rff_512 | 19 | 0.757 | 0.712 | 0.706 | -0.903 | -0.943 | 424.23 |

Global mixed-kernel correlations, for comparison only:

- corr(tail@10%, source norm) = `0.597`
- corr(source norm, ridge margin) = `-0.805`

Interpretation:

- Source norm is meaningful when read within a fixed kernel family.
- Local mixing/tail are still cleaner cross-family diagnostics, while
  source norm should be reported with kernel/regularization context.
- For Laplace/RBF/RFF, source norm tracks tail strongly (`0.757-0.794`) and
  source norm tracks ridge margin negatively (`-0.903` to `-0.947`).
- The linear kernel is the expected caveat: tail still tracks margin, but local
  mixing does not track source norm because XOR-like global misalignment creates
  high source norm without local collisions.

| dataset | kernel | tail@10% | mix | graph dir | align | source norm | ridge margin | train acc |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cifar10_automobilevstruck_train_n80 | laplace | 0.793 | 0.435 | 0.884 | 0.096 | 15.84 | 0.808 | 1.000 |
| cifar10_catvsdog_train_n80 | laplace | 0.863 | 0.494 | 0.989 | 0.063 | 17.76 | 0.776 | 1.000 |
| collision_pairs_sep0.02 | laplace | 1.000 | 0.551 | 1.114 | 0.000 | 115.61 | 0.070 | 1.000 |
| collision_pairs_sep0.04 | laplace | 1.000 | 0.553 | 1.116 | 0.001 | 80.51 | 0.139 | 1.000 |
| collision_pairs_sep0.08 | laplace | 0.999 | 0.551 | 1.103 | 0.002 | 61.18 | 0.221 | 1.000 |
| collision_pairs_sep0.16 | laplace | 0.996 | 0.552 | 1.101 | 0.004 | 51.62 | 0.330 | 0.981 |
| collision_pairs_sep0.32 | laplace | 0.986 | 0.546 | 1.093 | 0.007 | 44.03 | 0.460 | 0.975 |
| collision_pairs_sep0.64 | laplace | 0.966 | 0.518 | 1.033 | 0.011 | 44.93 | 0.453 | 0.981 |
| mnist_3vs8_train_n80 | laplace | 0.252 | 0.183 | 0.388 | 0.218 | 11.00 | 0.899 | 1.000 |
| mnist_4vs9_train_n80 | laplace | 0.240 | 0.304 | 0.644 | 0.172 | 11.91 | 0.888 | 1.000 |
| two_moons_noise0.02 | laplace | 0.007 | 0.000 | 0.000 | 0.544 | 4.91 | 0.987 | 1.000 |
| two_moons_noise0.08 | laplace | 0.019 | 0.014 | 0.037 | 0.510 | 6.56 | 0.988 | 1.000 |
| two_moons_noise0.15 | laplace | 0.031 | 0.017 | 0.038 | 0.560 | 6.74 | 0.994 | 1.000 |
| two_moons_noise0.25 | laplace | 0.165 | 0.099 | 0.198 | 0.499 | 18.00 | 0.983 | 0.988 |
| two_moons_noise0.35 | laplace | 0.291 | 0.179 | 0.358 | 0.395 | 24.86 | 0.957 | 0.988 |
| xor_noise0.05 | laplace | 0.000 | 0.000 | 0.000 | 0.418 | 3.09 | 0.996 | 1.000 |
| xor_noise0.15 | laplace | 0.004 | 0.000 | 0.000 | 0.405 | 3.76 | 0.996 | 1.000 |
| xor_noise0.25 | laplace | 0.007 | 0.000 | 0.000 | 0.363 | 4.44 | 0.999 | 1.000 |
| xor_noise0.4 | laplace | 0.073 | 0.030 | 0.055 | 0.288 | 8.16 | 0.998 | 1.000 |
| cifar10_automobilevstruck_train_n80 | linear | 0.760 | 0.444 | 0.907 | 0.071 | 51.92 | 0.720 | 1.000 |
| cifar10_catvsdog_train_n80 | linear | 0.903 | 0.483 | 0.962 | 0.033 | 37.12 | 0.587 | 0.981 |
| collision_pairs_sep0.02 | linear | 0.898 | 0.554 | 1.092 | 0.000 | 1000.00 | -0.000 | 0.500 |
| collision_pairs_sep0.04 | linear | 0.917 | 0.544 | 1.082 | 0.000 | 999.98 | 0.000 | 0.506 |
| collision_pairs_sep0.08 | linear | 0.923 | 0.545 | 1.081 | 0.000 | 1000.00 | 0.000 | 0.506 |
| collision_pairs_sep0.16 | linear | 0.926 | 0.534 | 1.053 | 0.000 | 999.84 | 0.000 | 0.500 |
| collision_pairs_sep0.32 | linear | 0.920 | 0.529 | 1.046 | 0.000 | 999.79 | 0.000 | 0.500 |
| collision_pairs_sep0.64 | linear | 0.806 | 0.503 | 1.009 | 0.002 | 998.27 | 0.009 | 0.519 |
| mnist_3vs8_train_n80 | linear | 0.264 | 0.170 | 0.349 | 0.311 | 60.11 | 0.911 | 1.000 |
| mnist_4vs9_train_n80 | linear | 0.244 | 0.255 | 0.547 | 0.195 | 98.81 | 0.889 | 1.000 |
| two_moons_noise0.02 | linear | 0.330 | 0.196 | 0.396 | 0.500 | 605.91 | 0.856 | 0.856 |
| two_moons_noise0.08 | linear | 0.430 | 0.222 | 0.445 | 0.447 | 689.31 | 0.744 | 0.863 |
| two_moons_noise0.15 | linear | 0.372 | 0.206 | 0.414 | 0.507 | 631.46 | 0.820 | 0.844 |
| two_moons_noise0.25 | linear | 0.387 | 0.194 | 0.383 | 0.491 | 635.35 | 0.808 | 0.856 |
| two_moons_noise0.35 | linear | 0.519 | 0.265 | 0.521 | 0.365 | 758.45 | 0.652 | 0.800 |
| xor_noise0.05 | linear | 0.909 | 0.000 | 0.000 | 0.013 | 991.05 | 0.112 | 0.544 |
| xor_noise0.15 | linear | 0.818 | 0.000 | 0.000 | 0.045 | 967.67 | 0.182 | 0.581 |
| xor_noise0.25 | linear | 0.924 | 0.000 | 0.000 | 0.007 | 995.30 | 0.010 | 0.544 |
| xor_noise0.4 | linear | 0.962 | 0.018 | 0.046 | 0.003 | 998.07 | 0.015 | 0.531 |
| cifar10_automobilevstruck_train_n80 | rbf | 0.776 | 0.435 | 0.881 | 0.075 | 27.90 | 0.650 | 1.000 |
| cifar10_catvsdog_train_n80 | rbf | 0.897 | 0.494 | 0.987 | 0.037 | 36.28 | 0.544 | 0.988 |
| collision_pairs_sep0.02 | rbf | 1.000 | 0.551 | 1.095 | 0.000 | 999.72 | -0.000 | 0.500 |
| collision_pairs_sep0.04 | rbf | 1.000 | 0.553 | 1.099 | 0.000 | 994.95 | 0.000 | 0.512 |
| collision_pairs_sep0.08 | rbf | 0.999 | 0.551 | 1.090 | 0.000 | 990.89 | 0.001 | 0.525 |
| collision_pairs_sep0.16 | rbf | 0.997 | 0.552 | 1.093 | 0.000 | 981.01 | 0.002 | 0.537 |
| collision_pairs_sep0.32 | rbf | 0.977 | 0.546 | 1.096 | 0.001 | 962.29 | 0.006 | 0.569 |
| collision_pairs_sep0.64 | rbf | 0.932 | 0.518 | 1.033 | 0.002 | 907.21 | 0.006 | 0.562 |
| mnist_3vs8_train_n80 | rbf | 0.261 | 0.183 | 0.376 | 0.258 | 16.48 | 0.885 | 1.000 |
| mnist_4vs9_train_n80 | rbf | 0.258 | 0.304 | 0.633 | 0.186 | 17.28 | 0.861 | 1.000 |
| two_moons_noise0.02 | rbf | 0.006 | 0.000 | 0.000 | 0.565 | 69.38 | 0.949 | 1.000 |
| two_moons_noise0.08 | rbf | 0.070 | 0.014 | 0.042 | 0.538 | 177.69 | 0.858 | 0.963 |
| two_moons_noise0.15 | rbf | 0.109 | 0.017 | 0.042 | 0.599 | 211.46 | 0.899 | 0.950 |
| two_moons_noise0.25 | rbf | 0.209 | 0.099 | 0.201 | 0.536 | 402.41 | 0.886 | 0.925 |
| two_moons_noise0.35 | rbf | 0.361 | 0.179 | 0.355 | 0.438 | 532.79 | 0.760 | 0.869 |
| xor_noise0.05 | rbf | 0.000 | 0.000 | 0.000 | 0.174 | 26.70 | 0.976 | 1.000 |
| xor_noise0.15 | rbf | 0.003 | 0.000 | 0.000 | 0.185 | 60.89 | 0.938 | 1.000 |
| xor_noise0.25 | rbf | 0.014 | 0.000 | 0.000 | 0.160 | 117.62 | 0.922 | 1.000 |
| xor_noise0.4 | rbf | 0.098 | 0.030 | 0.061 | 0.134 | 278.80 | 0.827 | 0.994 |
| cifar10_automobilevstruck_train_n80 | rff_512 | 0.778 | 0.428 | 0.861 | 0.076 | 31.39 | 0.638 | 0.994 |
| cifar10_catvsdog_train_n80 | rff_512 | 0.889 | 0.488 | 0.982 | 0.044 | 44.60 | 0.492 | 0.988 |
| collision_pairs_sep0.02 | rff_512 | 1.000 | 0.552 | 1.094 | 0.000 | 999.88 | 0.000 | 0.500 |
| collision_pairs_sep0.04 | rff_512 | 1.000 | 0.549 | 1.082 | 0.000 | 996.92 | 0.000 | 0.512 |
| collision_pairs_sep0.08 | rff_512 | 0.999 | 0.552 | 1.094 | 0.000 | 993.65 | 0.001 | 0.506 |
| collision_pairs_sep0.16 | rff_512 | 0.997 | 0.553 | 1.088 | 0.000 | 986.92 | 0.001 | 0.512 |
| collision_pairs_sep0.32 | rff_512 | 0.980 | 0.547 | 1.098 | 0.001 | 973.34 | 0.009 | 0.550 |
| collision_pairs_sep0.64 | rff_512 | 0.952 | 0.512 | 1.029 | 0.002 | 936.32 | 0.007 | 0.556 |
| mnist_3vs8_train_n80 | rff_512 | 0.294 | 0.221 | 0.440 | 0.228 | 21.25 | 0.842 | 1.000 |
| mnist_4vs9_train_n80 | rff_512 | 0.307 | 0.287 | 0.597 | 0.187 | 20.92 | 0.832 | 1.000 |
| two_moons_noise0.02 | rff_512 | 0.006 | 0.000 | 0.000 | 0.551 | 74.21 | 0.950 | 1.000 |
| two_moons_noise0.08 | rff_512 | 0.069 | 0.015 | 0.044 | 0.550 | 225.05 | 0.844 | 0.919 |
| two_moons_noise0.15 | rff_512 | 0.103 | 0.019 | 0.046 | 0.585 | 241.67 | 0.888 | 0.944 |
| two_moons_noise0.25 | rff_512 | 0.210 | 0.098 | 0.202 | 0.534 | 420.70 | 0.874 | 0.931 |
| two_moons_noise0.35 | rff_512 | 0.375 | 0.179 | 0.356 | 0.426 | 552.77 | 0.760 | 0.844 |
| xor_noise0.05 | rff_512 | 0.000 | 0.000 | 0.000 | 0.139 | 30.77 | 0.973 | 1.000 |
| xor_noise0.15 | rff_512 | 0.003 | 0.000 | 0.000 | 0.171 | 74.53 | 0.934 | 1.000 |
| xor_noise0.25 | rff_512 | 0.017 | 0.000 | 0.000 | 0.149 | 130.38 | 0.918 | 1.000 |
| xor_noise0.4 | rff_512 | 0.104 | 0.031 | 0.064 | 0.143 | 305.05 | 0.827 | 0.994 |

## Artifacts

- `metrics.json`
- `figures/tail_vs_source_norm.png`
- `figures/mixing_vs_source_norm.png`
- `figures/source_norm_vs_margin.png`
