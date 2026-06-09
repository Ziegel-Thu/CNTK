# outputs

Local logs, raw run outputs, and generated artifacts can live here.

Commit only small artifacts that are necessary to understand a result. Prefer
putting final figures and compact `metrics.json` files inside the corresponding
experiment folder.

Current decision:

- Keep raw logs in `outputs/`; the directory is ignored by git except this
  README.
- Commit only compact, result-facing artifacts from experiment folders.
