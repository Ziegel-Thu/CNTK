# Result

## Run

Command: `python3 experiments/011-self-supervised-fixed-representation-sweep/scripts/run_selfsup.py --binary-n-per-class 60 --multi-n-per-class 30 --batch-size 48 --ridge 0.001 --seed 0 --device cpu`

## Summary

- corr(test local mixing/disagreement, test tail@10%) = `0.839`
- corr(test graph Dirichlet, test tail@10%) = `0.990`
- corr(test tail@10%, kernel ridge test accuracy) = `-0.947`
- corr(test tail@10%, kernel ridge test margin median) = `-0.885`

Interpretation:

- DINO ViT-S/16 features are a self-supervised frozen representation
  baseline, not a CNTK or toy-network kernel.
- If self-supervised features lower local mixing/tail and improve ridge
  margin, that supports the broader fixed-representation obstruction
  framing.
- DINO improves over raw pixels on every task and is often stronger than the
  supervised ResNet18 baseline in this run.
- This closes the main fixed-representation scope gap: the obstruction signal
  holds for self-supervised frozen features, not only supervised pretrained
  features or small trained models.

| dataset | representation | classes | dim | tail@10% | mixing | graph dir | align | ridge acc | ridge margin | source norm |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cifar10_all10 | dino_vits16 | 10 | 384 | 0.432 | 0.454 | 0.516 | 0.415 | 0.807 | 0.306 | 36.68 |
| cifar10_all10 | imagenet_resnet18 | 10 | 512 | 0.456 | 0.468 | 0.532 | 0.421 | 0.770 | 0.263 | 32.57 |
| cifar10_all10 | raw_pixels | 10 | 3072 | 0.799 | 0.800 | 0.893 | 0.125 | 0.287 | -0.107 | 67.65 |
| cifar10_animals6 | dino_vits16 | 6 | 384 | 0.475 | 0.472 | 0.581 | 0.373 | 0.711 | 0.242 | 23.01 |
| cifar10_animals6 | imagenet_resnet18 | 6 | 512 | 0.531 | 0.499 | 0.633 | 0.342 | 0.639 | 0.157 | 29.99 |
| cifar10_animals6 | raw_pixels | 6 | 3072 | 0.825 | 0.765 | 0.924 | 0.086 | 0.317 | -0.106 | 51.14 |
| cifar10_automobilevstruck | dino_vits16 | 2 | 384 | 0.244 | 0.198 | 0.405 | 0.339 | 0.950 | 0.747 | 10.13 |
| cifar10_automobilevstruck | imagenet_resnet18 | 2 | 512 | 0.242 | 0.191 | 0.403 | 0.341 | 0.917 | 0.631 | 10.13 |
| cifar10_automobilevstruck | raw_pixels | 2 | 3072 | 0.749 | 0.421 | 0.831 | 0.139 | 0.533 | 0.072 | 34.58 |
| cifar10_catvsdog | dino_vits16 | 2 | 384 | 0.385 | 0.240 | 0.488 | 0.287 | 0.867 | 0.604 | 11.52 |
| cifar10_catvsdog | imagenet_resnet18 | 2 | 512 | 0.499 | 0.315 | 0.631 | 0.198 | 0.833 | 0.679 | 18.65 |
| cifar10_catvsdog | raw_pixels | 2 | 3072 | 0.780 | 0.463 | 0.943 | 0.060 | 0.550 | 0.115 | 39.62 |
| cifar10_vehicles4 | dino_vits16 | 4 | 384 | 0.228 | 0.271 | 0.384 | 0.484 | 0.917 | 0.578 | 27.85 |
| cifar10_vehicles4 | imagenet_resnet18 | 4 | 512 | 0.346 | 0.339 | 0.488 | 0.433 | 0.833 | 0.415 | 26.82 |
| cifar10_vehicles4 | raw_pixels | 4 | 3072 | 0.731 | 0.637 | 0.862 | 0.148 | 0.500 | -0.001 | 35.18 |

## Artifacts

- `metrics.json`
- `figures/mixing_vs_tail.png`
- `figures/tail_vs_accuracy.png`
- `figures/tail_vs_margin.png`
