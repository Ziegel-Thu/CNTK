# src

Shared reusable code lives here.

Expected modules:

- `datasets.py` - toy/MNIST/CIFAR binary and multiclass task builders.
- `kernels.py` - static kernels and feature Gram utilities.
- `spectral.py` - eigendecomposition, binary/multiclass label energy,
  spectral tail, alignment.
- `mixing.py` - opposite-label kNN, multiclass disagreement, collision graph,
  local entropy, graph disagreement, graph Dirichlet energy.
- `gradient_flow.py` - kernel gradient-flow simulation.
- `kernel_ridge.py` - closed-form kernel ridge classifiers, margins, and
  source-norm proxies.
- `models.py` - small MLP/CNN models for feature-learning experiments.
- `plotting.py` - shared plotting utilities.
