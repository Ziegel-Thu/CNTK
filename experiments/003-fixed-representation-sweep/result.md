# Result

## Run

Command: `python experiments/003-fixed-representation-sweep/scripts/run_sweep.py --n-per-class 100 --mlp-epochs 80 --cnn-epochs 60 --seed 0 --device cpu`

## Summary

- corr(test opposite-kNN ratio, test tail@10%) = `0.988`
- corr(test tail@10%, linear probe test acc) = `-0.971`

Interpretation:

- This supports the broader fixed-representation framing: across raw pixels,
  random features, random networks, and trained features, lower test local
  mixing tracks lower spectral tail, and lower tail tracks better linear-probe
  accuracy.
- MNIST trained MLP features are the cleanest positive case: they lower test
  tail/mixing and improve probe accuracy.
- CIFAR stays high-tail/high-mixing across most fixed representations. The small
  CNN improves `automobile vs truck` geometry relative to random/raw features,
  but `cat vs dog` remains hard in this small local run.
- This experiment is a better scope-expander than adding more kernels: it shows
  the diagnostic applies across representation families.

| dataset | representation | dim | test tail@10% | test mix | alignment | probe train acc | probe test acc |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cifar10_automobilevstruck_n100_train | random_cnn | 128 | 0.803 | 0.459 | 0.015 | 0.930 | 0.660 |
| cifar10_automobilevstruck_n100_train | random_mlp | 256 | 0.823 | 0.428 | 0.066 | 1.000 | 0.545 |
| cifar10_automobilevstruck_n100_train | raw_pixels | 3072 | 0.784 | 0.439 | 0.070 | 1.000 | 0.625 |
| cifar10_automobilevstruck_n100_train | rff_512 | 512 | 0.811 | 0.463 | 0.059 | 1.000 | 0.540 |
| cifar10_automobilevstruck_n100_train | trained_cnn | 128 | 0.696 | 0.430 | 0.090 | 0.880 | 0.660 |
| cifar10_automobilevstruck_n100_train | trained_mlp | 256 | 0.762 | 0.402 | 0.144 | 1.000 | 0.640 |
| cifar10_catvsdog_n100_train | random_cnn | 128 | 0.826 | 0.453 | 0.014 | 0.850 | 0.550 |
| cifar10_catvsdog_n100_train | random_mlp | 256 | 0.842 | 0.480 | 0.039 | 1.000 | 0.540 |
| cifar10_catvsdog_n100_train | raw_pixels | 3072 | 0.873 | 0.480 | 0.023 | 1.000 | 0.575 |
| cifar10_catvsdog_n100_train | rff_512 | 512 | 0.865 | 0.498 | 0.030 | 1.000 | 0.575 |
| cifar10_catvsdog_n100_train | trained_cnn | 128 | 0.864 | 0.481 | 0.013 | 0.840 | 0.525 |
| cifar10_catvsdog_n100_train | trained_mlp | 256 | 0.858 | 0.481 | 0.035 | 1.000 | 0.605 |
| mnist_3vs8_n100_train | random_mlp | 256 | 0.450 | 0.273 | 0.184 | 1.000 | 0.845 |
| mnist_3vs8_n100_train | raw_pixels | 784 | 0.307 | 0.168 | 0.296 | 1.000 | 0.875 |
| mnist_3vs8_n100_train | rff_512 | 512 | 0.354 | 0.223 | 0.145 | 1.000 | 0.895 |
| mnist_3vs8_n100_train | trained_mlp | 256 | 0.176 | 0.115 | 0.743 | 1.000 | 0.910 |
| mnist_4vs9_n100_train | random_mlp | 256 | 0.442 | 0.285 | 0.145 | 1.000 | 0.800 |
| mnist_4vs9_n100_train | raw_pixels | 784 | 0.289 | 0.208 | 0.182 | 1.000 | 0.865 |
| mnist_4vs9_n100_train | rff_512 | 512 | 0.311 | 0.251 | 0.107 | 1.000 | 0.890 |
| mnist_4vs9_n100_train | trained_mlp | 256 | 0.201 | 0.124 | 0.710 | 1.000 | 0.885 |

## Artifacts

- `metrics.json`
- `figures/tail_vs_accuracy.png`
- `figures/mixing_vs_tail.png`
