# data

Dataset caches and large local artifacts should live outside git.

Recommended policy:

- Keep this README committed.
- Do not commit raw MNIST/CIFAR downloads.
- Prefer configurable cache paths.
- Record dataset subset definitions and random seeds in experiment configs.
