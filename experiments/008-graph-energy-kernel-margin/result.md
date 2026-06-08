# Result

## Run

Command: `python3 experiments/008-graph-energy-kernel-margin/scripts/run_graph_margin.py --n-per-class 80 --mlp-epochs 60 --cnn-epochs 50 --ridge 0.001 --seed 0 --device cpu`

## Summary

- corr(test kNN opposite ratio, test tail@10%) = `0.961`
- corr(test graph disagreement, test tail@10%) = `0.955`
- corr(test graph Dirichlet, test tail@10%) = `0.955`
- corr(test tail@10%, kernel ridge test margin median) = `-0.964`
- corr(kernel ridge test margin median, test accuracy) = `0.958`
- corr(source norm proxy, test margin median) = `-0.267`

Interpretation:

- Graph disagreement/Dirichlet energy give a graph-level version of local
  label mixing and should move with spectral tail.
- Kernel ridge margin/source norm connect the geometric obstruction to a
  classifier consequence rather than only a spectral statistic.
- In this run, kernel ridge margin is the clean consequence variable: it has
  strong negative correlation with tail and strong positive correlation with
  accuracy.
- The source-norm proxy is weaker (`-0.267` with margin), likely because the
  comparison mixes normalized feature kernels, RBF/Laplace kernels, and random
  CNN feature scales. It should be audited in a same-kernel ridge sweep before
  becoming a headline claim.

| dataset | representation | tail@10% | kNN mix | graph dis | graph dir | align | ridge acc | ridge margin | source norm |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cifar10_automobilevstruck_n80_train | laplace | 0.806 | 0.436 | 0.442 | 0.884 | 0.094 | 0.662 | 0.156 | 16.41 |
| cifar10_automobilevstruck_n80_train | random_cnn | 0.782 | 0.491 | 0.497 | 0.994 | 0.038 | 0.669 | 0.196 | 409.97 |
| cifar10_automobilevstruck_n80_train | random_mlp | 0.853 | 0.461 | 0.465 | 0.930 | 0.068 | 0.619 | 0.134 | 38.72 |
| cifar10_automobilevstruck_n80_train | raw_linear | 0.801 | 0.448 | 0.459 | 0.919 | 0.069 | 0.631 | 0.220 | 37.15 |
| cifar10_automobilevstruck_n80_train | rbf | 0.822 | 0.436 | 0.442 | 0.883 | 0.076 | 0.681 | 0.220 | 29.97 |
| cifar10_automobilevstruck_n80_train | rff_512 | 0.790 | 0.443 | 0.454 | 0.907 | 0.076 | 0.594 | 0.133 | 26.18 |
| cifar10_automobilevstruck_n80_train | trained_cnn | 0.700 | 0.438 | 0.437 | 0.874 | 0.121 | 0.700 | 0.367 | 427.45 |
| cifar10_automobilevstruck_n80_train | trained_mlp | 0.789 | 0.439 | 0.433 | 0.866 | 0.115 | 0.656 | 0.348 | 49.72 |
| cifar10_catvsdog_n80_train | laplace | 0.815 | 0.492 | 0.492 | 0.985 | 0.070 | 0.619 | 0.078 | 17.71 |
| cifar10_catvsdog_n80_train | random_cnn | 0.888 | 0.495 | 0.498 | 0.996 | 0.026 | 0.631 | 0.181 | 451.96 |
| cifar10_catvsdog_n80_train | random_mlp | 0.868 | 0.465 | 0.463 | 0.926 | 0.053 | 0.525 | 0.034 | 35.95 |
| cifar10_catvsdog_n80_train | raw_linear | 0.832 | 0.483 | 0.483 | 0.966 | 0.041 | 0.562 | 0.152 | 37.49 |
| cifar10_catvsdog_n80_train | rbf | 0.816 | 0.492 | 0.492 | 0.983 | 0.046 | 0.588 | 0.072 | 36.55 |
| cifar10_catvsdog_n80_train | rff_512 | 0.839 | 0.491 | 0.486 | 0.972 | 0.048 | 0.562 | 0.077 | 24.59 |
| cifar10_catvsdog_n80_train | trained_cnn | 0.892 | 0.500 | 0.503 | 1.007 | 0.020 | 0.537 | 0.122 | 420.69 |
| cifar10_catvsdog_n80_train | trained_mlp | 0.913 | 0.497 | 0.496 | 0.991 | 0.024 | 0.550 | 0.105 | 36.57 |
| mnist_3vs8_n80_train | laplace | 0.309 | 0.216 | 0.224 | 0.448 | 0.179 | 0.850 | 0.616 | 10.74 |
| mnist_3vs8_n80_train | random_mlp | 0.344 | 0.237 | 0.256 | 0.513 | 0.209 | 0.869 | 0.681 | 19.21 |
| mnist_3vs8_n80_train | raw_linear | 0.293 | 0.163 | 0.167 | 0.335 | 0.383 | 0.931 | 0.783 | 20.77 |
| mnist_3vs8_n80_train | rbf | 0.285 | 0.188 | 0.158 | 0.317 | 0.168 | 0.850 | 0.739 | 15.39 |
| mnist_3vs8_n80_train | rff_512 | 0.370 | 0.218 | 0.240 | 0.480 | 0.244 | 0.863 | 0.728 | 15.66 |
| mnist_3vs8_n80_train | trained_mlp | 0.211 | 0.114 | 0.104 | 0.208 | 0.697 | 0.887 | 0.993 | 22.03 |
| mnist_4vs9_n80_train | laplace | 0.306 | 0.296 | 0.305 | 0.609 | 0.146 | 0.838 | 0.500 | 12.05 |
| mnist_4vs9_n80_train | random_mlp | 0.412 | 0.333 | 0.340 | 0.680 | 0.149 | 0.850 | 0.714 | 60.40 |
| mnist_4vs9_n80_train | raw_linear | 0.271 | 0.263 | 0.276 | 0.553 | 0.211 | 0.969 | 0.826 | 62.93 |
| mnist_4vs9_n80_train | rbf | 0.317 | 0.274 | 0.270 | 0.541 | 0.132 | 0.850 | 0.659 | 17.75 |
| mnist_4vs9_n80_train | rff_512 | 0.288 | 0.268 | 0.277 | 0.553 | 0.197 | 0.900 | 0.756 | 54.78 |
| mnist_4vs9_n80_train | trained_mlp | 0.140 | 0.086 | 0.087 | 0.175 | 0.738 | 0.912 | 0.988 | 9.93 |

## Artifacts

- `metrics.json`
- `figures/graph_energy_vs_tail.png`
- `figures/tail_vs_margin.png`
- `figures/source_norm_vs_margin.png`
