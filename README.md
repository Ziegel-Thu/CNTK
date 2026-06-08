# CNTK Fixed-Metric Obstruction Project

This project studies classification failures of static kernels / fixed
representations through the lens of local label mixing, spectral label
complexity, and feature-learned metric dynamics.

## Core Question

When a fixed kernel metric places nearby samples from opposite labels too close,
does the label function necessarily live in high-frequency / spectral-tail
directions, causing slow kernel learning and poor margin/generalization? If yes,
can feature learning be measured as a process that reshapes the metric and moves
the label function into lower-complexity directions?

## Directory Layout

- `src/` - shared reusable code.
- `experiments/` - one folder per experiment, named `001-expname`.
- `progress.md` - project-wide progress log.
- `todo.md` - project-wide TODO list.
- `paper/` - one folder per paper/draft, including downloaded arXiv source when available.
- `deepresearch/` - deep research prompts/responses, following the style of `~/LR`.
- `data/` - local data/cache notes; large datasets should not be committed.
- `outputs/` - local run outputs/logs; committed only when small and necessary.
- `scripts/` - project maintenance scripts.
- `tests/` - tests for shared code in `src/`.

## Current Status

The project has been formalized, with initial literature survey notes and four
experiment plans. The current workspace is not yet a git repository, so the
commit/push policy is documented but cannot be executed until git is initialized
and a remote is configured.
