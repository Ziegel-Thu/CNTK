# Experiment Artifact Audit

Date: 2026-06-10

## Standard

Every new experiment should have:

- `plan.md`
- `result.md`
- `metrics.json` or `metrics_over_time.json`
- `figures/*.png` for result-facing plots
- `scripts/*.py` for experiment-specific code

Supplemental runs may add suffixed files such as `result_cifar.md` or
`metrics_mnist_over_time.json`, but `result.md` should remain the canonical
human entry point.

## Current Status

| ID | Status | Notes |
| --- | --- | --- |
| 001 | OK with supplements | Canonical `result.md`; extra image/bound audit files are intentional. |
| 002 | OK with supplements | Canonical `result.md`; MNIST/CIFAR/CNN runs keep suffixed artifacts. |
| 003 | OK | Standard artifact names. |
| 004 | OK with canonical wrapper | Added `result.md`; detailed MNIST stress artifacts remain suffixed. |
| 005 | OK | Standard artifact names. |
| 006 | OK | Standard artifact names. |
| 007 | OK | Standard artifact names. |
| 008 | OK | Standard artifact names. |
| 009 | OK | Standard artifact names. |
| 010 | OK | Standard artifact names. |
| 011 | OK | Standard artifact names. |
| 012 | OK | Standard artifact names. |
| 013 | OK | Standard artifact names; tmux run completed. |
| 014 | OK | Standard artifact names; existing-results audit. |

## Policy

- Do not rename committed historical supplemental artifacts unless a future
  cleanup also updates `experiments/results.md`, `progress.md`, and any paper
  references.
- For experiment `013+`, create `plan.md` before runs and write `result.md`
  after runs even when the result is negative.
- Keep large raw logs in `outputs/`; keep only compact metrics and final plots
  in experiment folders.
