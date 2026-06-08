# Result

## Run

Command: `python3 experiments/010-pretrained-fixed-representation-sweep/scripts/run_pretrained.py --binary-n-per-class 80 --multi-n-per-class 40 --batch-size 64 --ridge 0.001 --seed 0 --device cpu`

## Summary

- corr(test local mixing/disagreement, test tail@10%) = `0.729`
- corr(test graph Dirichlet, test tail@10%) = `0.952`
- corr(test tail@10%, kernel ridge test accuracy) = `-0.794`
- corr(test tail@10%, kernel ridge test margin median) = `-0.848`

Interpretation:

- ImageNet-pretrained ResNet18 features are a real frozen representation
  baseline, not a CNTK or toy-network kernel.
- If pretrained features lower local mixing/tail and improve ridge margin,
  that supports the broader fixed-representation obstruction framing.
- The improvement is large and consistent: ImageNet ResNet18 lowers tail and
  raises ridge accuracy on every task family compared with raw pixels and random
  ResNet18.
- This is the cleanest fixed-representation scope expansion so far: no feature
  learning is performed on these CIFAR subsets, but a better frozen metric has
  lower local mixing, lower graph energy, lower spectral tail, and better
  margins.

| dataset | representation | classes | dim | tail@10% | mixing | graph dir | align | ridge acc | ridge margin | source norm |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cifar10_all10 | imagenet_resnet18 | 10 | 512 | 0.450 | 0.468 | 0.540 | 0.395 | 0.765 | 0.251 | 51.06 |
| cifar10_all10 | random_resnet18 | 10 | 512 | 0.787 | 0.809 | 0.909 | 0.088 | 0.307 | -0.087 | 248.08 |
| cifar10_all10 | raw_pixels | 10 | 3072 | 0.788 | 0.794 | 0.888 | 0.120 | 0.310 | -0.079 | 87.77 |
| cifar10_animals6 | imagenet_resnet18 | 6 | 512 | 0.475 | 0.494 | 0.607 | 0.346 | 0.662 | 0.155 | 31.63 |
| cifar10_animals6 | random_resnet18 | 6 | 512 | 0.808 | 0.781 | 0.936 | 0.071 | 0.300 | -0.113 | 95.44 |
| cifar10_animals6 | raw_pixels | 6 | 3072 | 0.826 | 0.778 | 0.936 | 0.067 | 0.308 | -0.147 | 56.56 |
| cifar10_automobilevstruck | imagenet_resnet18 | 2 | 512 | 0.261 | 0.192 | 0.394 | 0.343 | 0.906 | 0.741 | 19.53 |
| cifar10_automobilevstruck | random_resnet18 | 2 | 512 | 0.666 | 0.473 | 0.997 | 0.043 | 0.613 | 0.157 | 62.07 |
| cifar10_automobilevstruck | raw_pixels | 2 | 3072 | 0.719 | 0.456 | 0.923 | 0.078 | 0.606 | 0.140 | 37.71 |
| cifar10_catvsdog | imagenet_resnet18 | 2 | 512 | 0.455 | 0.282 | 0.565 | 0.201 | 0.794 | 0.532 | 16.99 |
| cifar10_catvsdog | random_resnet18 | 2 | 512 | 0.892 | 0.486 | 0.980 | 0.027 | 0.625 | 0.101 | 76.52 |
| cifar10_catvsdog | raw_pixels | 2 | 3072 | 0.900 | 0.479 | 0.952 | 0.032 | 0.575 | 0.073 | 37.67 |
| cifar10_vehicles4 | imagenet_resnet18 | 4 | 512 | 0.353 | 0.334 | 0.464 | 0.420 | 0.838 | 0.431 | 32.00 |
| cifar10_vehicles4 | random_resnet18 | 4 | 512 | 0.772 | 0.689 | 0.936 | 0.047 | 0.369 | -0.076 | 65.44 |
| cifar10_vehicles4 | raw_pixels | 4 | 3072 | 0.701 | 0.616 | 0.822 | 0.144 | 0.388 | -0.091 | 34.97 |

## Artifacts

- `metrics.json`
- `figures/mixing_vs_tail.png`
- `figures/tail_vs_accuracy.png`
- `figures/tail_vs_margin.png`
