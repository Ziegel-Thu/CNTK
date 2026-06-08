# Project Instructions for GPT/Codex Agents

## Project Identity

This is an engineering research project on static NTK/CNTK/fixed-representation
obstructions in classification and the role of feature learning as metric
adaptation.

Primary language for notes: Chinese, with technical terms in English when useful.

## Research Scope

Do not frame the project narrowly as "CNTK performs poorly on MNIST/CIFAR".
The working scope is:

> Fixed metrics can create local label mixing; local label mixing pushes labels
> into spectral-tail directions; feature learning may escape by changing the
> metric so labels become lower-complexity and better-margin objects.

Key measurable objects:

- static kernel / feature Gram matrix `K`
- kernel metric `d_K(x_i, x_j)`
- opposite-label nearest-neighbor/collision statistics
- spectral label energy `E_y(m)` and tail `T_y(m)`
- kernel-target alignment
- kernel gradient flow loss predicted from spectrum
- feature metric dynamics `K_t`
- margin, accuracy, and representation purity

## Required Directory Structure

- `PLAN.md` - current execution order, experiment queue, decision rules, and
  resource needs.
- `src/` - shared code only. Put reusable datasets, kernels, metrics, plotting,
  training utilities, and evaluation code here.
- `experiments/` - one folder per experiment. Use `001-expname`, `002-expname`,
  etc. Do not use `exp-001`.
- `experiments/index.md` - experiment table, status, artifact contract, and tmux
  session names.
- `progress.md` - project-wide progress log, newest first.
- `todo.md` - project-wide TODO list.
- `paper/` - one folder per paper/draft. For arXiv papers, download source via
  `https://arxiv.org/e-print/{id}` and keep TeX/source files when available.
- `deepresearch/` - one folder per research pass, following `~/LR` style:
  `001-topic/prompt.md` and `001-topic/response.md`.
- `data/` - local datasets/caches. Large files are not committed.
- `outputs/` - local logs, generated figures, and raw run outputs. Commit only
  small, meaningful artifacts.
- `scripts/` - maintenance scripts such as arXiv source download helpers.
- `tests/` - tests for shared code.

## Experiment Protocol

Every experiment must follow this lifecycle:

1. Read `PLAN.md`, `experiments/index.md`, `todo.md`, and the experiment's
   `plan.md`.
2. Create `experiments/NNN-expname/` if this is a new experiment.
3. Write `plan.md` before running anything.
4. Run substantive experiments inside `tmux`.
5. Save scripts/configs needed for the experiment inside that experiment folder,
   unless the code is reusable, in which case put it in `src/`.
6. Write `result.md` after the run, even if the result is negative.
7. Update `experiments/index.md`, `progress.md`, and `todo.md`.
8. If the result is meaningful and git remote is configured, immediately commit
   and push the relevant changes.

Suggested tmux naming:

```bash
tmux new -s cntk-001-spectral-tail
tmux new -s cntk-002-feature-dynamics
```

## Git Policy

- The current workspace may not yet be a git repo; check `git status` first.
- When git is available, commit small coherent changes.
- Commit only files touched for the current task.
- Push immediately after a meaningful experimental result or project-state update
  if a remote is configured.
- Never rewrite history or reset user changes without explicit permission.

## Paper Management

Paper folder naming:

```text
paper/001-author-year-keyword/
├── README.md
├── paper.pdf              # optional but preferred
├── source/                # extracted arXiv source, TeX, bib, sty, figures
└── notes.md               # project-specific notes
```

For local drafts, use `paper/000-...`.

## Deep Research Protocol

Deep research folder naming:

```text
deepresearch/001-topic/
├── prompt.md
└── response.md
```

The response should separate:

- likely prior art
- closest existing claims
- gaps not covered by prior art
- project implications
- papers to download/read next

## Coding Conventions

- Prefer Python + PyTorch for experiments.
- Keep reusable code in `src/`.
- Keep experiment-specific scripts inside the experiment folder.
- Use deterministic seeds where reasonable.
- Write compact JSON/CSV/Markdown summaries for results.
- Put large datasets and heavy outputs in ignored folders.

## Engineering Additions

This project also keeps:

- `data/README.md` for dataset/cache policy.
- `outputs/README.md` for log/artifact policy.
- `tests/README.md` for shared-code verification.
- `scripts/` for repeatable maintenance actions.
