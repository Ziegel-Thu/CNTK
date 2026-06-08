#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python3 -m unittest discover -s tests

python3 - <<'PY'
import numpy as np
import torch

from src import datasets, kernels, mixing, models, spectral

toy = datasets.make_two_moons(n=80, noise=0.08, seed=123)
gram = kernels.rbf_kernel(toy.x)
d2 = kernels.kernel_metric_squared(gram)
spec = spectral.summarize(gram, toy.y)
mix = mixing.summarize(d2, toy.y, k=5)

assert 0.0 <= spec.tail_at_10pct <= 1.0
assert 0.0 <= mix.knn_opposite_ratio <= 1.0

labels = np.repeat(np.arange(4), 10)
features = np.eye(4)[labels] + 0.01 * np.random.default_rng(0).normal(size=(40, 4))
gram = kernels.feature_gram(features, normalize=True)
d2 = kernels.kernel_metric_squared(gram)
mc_spec = spectral.summarize_multiclass(gram, labels)
mc_mix = mixing.summarize_multiclass(d2, labels, k=3)

assert mc_spec.n_classes == 4
assert 0.0 <= mc_spec.tail_at_10pct <= 1.0
assert 0.0 <= mc_mix.knn_disagreement_ratio <= 1.0

model = models.ClassifierMLP(input_dim=4, num_classes=4, width=8, depth=1)
assert tuple(model.features(torch.zeros(2, 4)).shape) == (2, 8)

print("smoke ok")
PY
