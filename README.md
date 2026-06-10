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
- `experiments/results.md` - compact index of completed result files.
- `progress.md` - project-wide progress log.
- `todo.md` - project-wide TODO list.
- `claims.md` - what is genuinely non-trivial versus supporting work.
- `theory.md` - finite-sample bridge from graph label roughness to spectral tail.
- `paper/` - one folder per paper/draft, including downloaded arXiv source when available.
- `deepresearch/` - deep research prompts/responses, following the style of `~/LR`.
- `data/` - local data/cache notes; large datasets should not be committed.
- `outputs/` - local run outputs/logs; committed only when small and necessary.
- `scripts/` - project maintenance scripts.
- `tests/` - tests for shared code in `src/`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

The shared-code smoke test does not download MNIST/CIFAR or overwrite experiment
artifacts:

```bash
scripts/run_smoke_test.sh
```

Full experiment scripts download/cache MNIST and CIFAR-10 under `data/` when
needed.

## Current Status

The project has a local git repository on `main`. No remote is configured yet,
so push is unavailable.

Large dataset caches stay under `data/` and raw logs under `outputs/`; both are
ignored except for README policy files. Compact experiment metrics/figures live
inside each experiment folder when they are needed to understand a result.

Completed experiment families:

- `001`: static spectral-tail and local-mixing diagnostics.
- `002`: feature-metric dynamics on toy, MNIST, and CIFAR.
- `003`: fixed-representation sweep.
- `004`: intrinsic/noisy collision stress tests.
- `005`: multiclass obstruction diagnostics.
- `006`: CIFAR multiclass schedule sweep.
- `007`: margin-tail audit.
- `008`: graph energy and kernel margin diagnostics.
- `009`: tail-to-gradient-flow-time consequence audit.
- `010`: pretrained fixed-representation sweep.
- `011`: self-supervised fixed-representation sweep.
- `012`: controlled source-norm/RKHS proxy sweep.
- `013`: pretrained ResNet18 fine-tune metric dynamics.
- `014`: local mixing versus alignment controlled audit.
