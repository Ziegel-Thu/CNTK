# GPT/Codex Project Instructions

This file mirrors the GPT/Codex-facing project rules in `AGENTS.md`.

GPT/Codex agents should read `AGENTS.md` first. The short version:

- Current execution order lives in `PLAN.md`.
- Claim discipline lives in `claims.md`; distinguish non-trivial contributions
  from supporting experiments and infrastructure.
- Experiment status and artifact contract live in `experiments/index.md`.
- Shared code goes in `src/`.
- Experiments go in `experiments/NNN-expname/`.
- Every experiment needs `plan.md` before running and `result.md` after running.
- Substantive experiments must run in `tmux`.
- Project state lives in `progress.md` and `todo.md`.
- Papers live in `paper/`, one folder per paper, with arXiv TeX/source when available.
- Deep research lives in `deepresearch/NNN-topic/prompt.md` and `response.md`.
- Meaningful experiment results should be committed and pushed immediately when a
  git repo and remote are configured.

See `AGENTS.md` for the full rules.
