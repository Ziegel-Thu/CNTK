# Result

Canonical result entry for experiment `004`.

Detailed result: `result_mnist_stress.md`

## Run

Command: `python experiments/004-intrinsic-collision-stress/scripts/run_mnist_stress.py --n-per-class 150 --epochs 160 --seed 0 --device cpu`

## Summary

- Clean MNIST `3 vs 8` feature learning reduces clean test tail to `0.085` and
  reaches clean test accuracy `0.947`.
- Heavy random label noise creates memorization: at `30%` flips, train tail
  drops `0.807 -> 0.008`, but clean test tail worsens to `0.288` and clean test
  accuracy falls to `0.670`.
- Exact opposite-label duplicates are an intrinsic contradiction: train accuracy
  is capped near `0.833`, while clean test geometry can still improve.

## Artifacts

- `metrics_mnist_stress.json`
- `figures/train_tail_over_time.png`
- `figures/test_tail_over_time.png`
- `figures/test_accuracy_over_time.png`
