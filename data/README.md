# data

Dataset caches and large local artifacts should live outside git.

Recommended policy:

- Keep this README committed.
- Do not commit raw MNIST/CIFAR downloads.
- Prefer configurable cache paths.
- Record dataset subset definitions and random seeds in experiment configs.

Current decision:

- Keep the local dataset cache in `data/`; it is ignored by git except this
  README.
- Do not move the existing cache outside the workspace unless disk pressure
  becomes a problem. As of 2026-06-09 the local `data/` directory is about
  `351M`, which is acceptable for local reuse and not committed.
