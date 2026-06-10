# Plan

## Question

Is local mixing / graph roughness adding diagnostic information beyond global
kernel-target alignment, or is it just alignment under another name?

This is an experimental/statistical audit over existing result files. It does
not add theory.

## Inputs

Use compact metrics from:

- `001-spectral-tail-diagnostics`
- `003-fixed-representation-sweep`
- `005-multiclass-obstruction-diagnostics`
- `008-graph-energy-kernel-margin`
- `010-pretrained-fixed-representation-sweep`
- `011-self-supervised-fixed-representation-sweep`
- `012-source-norm-controlled-sweep`

## Diagnostics

For rows with available fields, collect:

- `tail@10%`
- local mixing/disagreement
- graph Dirichlet energy
- kernel-target alignment
- accuracy
- margin

Run:

- correlations;
- partial correlations controlling for alignment;
- standardized linear regressions:
  - `tail ~ mixing + alignment`
  - `tail ~ graph_dirichlet + alignment`
  - `accuracy ~ tail + mixing + alignment`
  - `margin ~ tail + graph_dirichlet + alignment`

Report two groups:

- `image_fixed_repr`: image/fixed-representation rows only;
- `all_with_toy`: includes toy/XOR/collision rows to preserve global-mismatch
  caveats.

## Command

```bash
python3 experiments/014-mixing-alignment-controlled-audit/scripts/run_audit.py
```

## Artifacts

- `metrics.json`
- `result.md`
- `figures/tail_mixing_alignment.png`
- `figures/tail_graph_alignment.png`
- `figures/accuracy_tail_mixing.png`
