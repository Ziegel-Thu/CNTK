# Image Subset Result

## Run

Command: `python experiments/001-spectral-tail-diagnostics/scripts/run_image_subsets.py --n-per-class 150 --seed 0 --include-cifar`

## Summary

- corr(`opposite-kNN ratio`, `tail@10%`) = `0.991`
- corr(`alignment`, `tail@10%`) = `-0.905`
- highest tail case: `cifar10_catvsdog_train_n150` / `rff_1024` with `tail@10%=0.863` and `opposite-kNN=0.472`

Interpretation notes:

- These fixed raw-pixel/RFF kernels show the expected ordering: MNIST has
  moderate tail/mixing, while CIFAR binary tasks have much larger local
  opposite-label mixing and much larger spectral tail.
- `cat vs dog` is harder than `automobile vs truck` under these metrics,
  matching the intuition that raw-pixel local neighborhoods mix semantic
  classes more heavily.
- This run strengthens experiment 001: local mixing is not only a toy
  construction; it is a computable diagnostic on image subsets.

| dataset | kernel | tail@10% | tail_auc | opposite-kNN ratio | alignment | t_res<=0.1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| cifar10_automobilevstruck_train_n150 | laplace_median | 0.768 | 0.382 | 0.438 | 0.092 | 646 |
| cifar10_automobilevstruck_train_n150 | linear | 0.744 | 0.404 | 0.426 | 0.084 | 3.57e+03 |
| cifar10_automobilevstruck_train_n150 | rbf_median | 0.771 | 0.385 | 0.438 | 0.082 | 2.49e+03 |
| cifar10_automobilevstruck_train_n150 | rff_1024 | 0.759 | 0.392 | 0.450 | 0.080 | 3.57e+03 |
| cifar10_catvsdog_train_n150 | laplace_median | 0.830 | 0.462 | 0.468 | 0.049 | 773 |
| cifar10_catvsdog_train_n150 | linear | 0.856 | 0.482 | 0.479 | 0.028 | 6.71e+03 |
| cifar10_catvsdog_train_n150 | rbf_median | 0.856 | 0.470 | 0.468 | 0.031 | 3.91e+03 |
| cifar10_catvsdog_train_n150 | rff_1024 | 0.863 | 0.477 | 0.472 | 0.031 | 5.61e+03 |
| mnist_3vs8_train_n150 | laplace_median | 0.164 | 0.090 | 0.132 | 0.211 | 219 |
| mnist_3vs8_train_n150 | linear | 0.184 | 0.097 | 0.120 | 0.306 | 344 |
| mnist_3vs8_train_n150 | rbf_median | 0.179 | 0.097 | 0.132 | 0.249 | 450 |
| mnist_3vs8_train_n150 | rff_1024 | 0.198 | 0.102 | 0.139 | 0.247 | 539 |
| mnist_4vs9_train_n150 | laplace_median | 0.243 | 0.105 | 0.217 | 0.144 | 287 |
| mnist_4vs9_train_n150 | linear | 0.244 | 0.118 | 0.208 | 0.175 | 450 |
| mnist_4vs9_train_n150 | rbf_median | 0.259 | 0.113 | 0.217 | 0.148 | 539 |
| mnist_4vs9_train_n150 | rff_1024 | 0.255 | 0.115 | 0.221 | 0.150 | 590 |

## Artifacts

- `metrics_image_subsets.json`
- `figures/image_mixing_vs_tail.png`
- `figures/image_tail_curves.png`
