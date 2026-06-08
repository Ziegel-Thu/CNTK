# experiments

Each experiment gets one folder named `NNN-expname`.

See `results.md` for a compact index of completed result files and headline
signals.

Required lifecycle:

1. Write `plan.md`.
2. Run the experiment in `tmux`.
3. Save experiment-specific scripts/configs in the experiment folder.
4. Write `result.md`.
5. Update `progress.md` and `todo.md`.
6. Commit and push meaningful results if git remote is configured.

Suggested files:

```text
experiments/NNN-expname/
├── plan.md
├── result.md
├── scripts/
├── configs/
├── figures/
└── metrics.json
```
