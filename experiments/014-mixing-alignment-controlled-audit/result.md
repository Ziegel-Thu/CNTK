# Result

## Run

Command: `python3 experiments/014-mixing-alignment-controlled-audit/scripts/run_audit.py`

## Summary

| group | rows | corr tail/mix | partial tail/mix | corr tail/graph | partial tail/graph | corr tail/align |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| image_fixed_repr | 104 | 0.770 | 0.661 | 0.963 | 0.906 | -0.794 |
| all_with_toy | 221 | 0.814 | 0.675 | 0.873 | 0.716 | -0.789 |

## Standardized Regressions

### image_fixed_repr
- `reg_tail_mixing_alignment`: n=`104`, R2=`0.792`, mixing=0.479, alignment=-0.533
- `reg_tail_graph_alignment`: n=`58`, R2=`0.934`, graph=1.120, alignment=0.181
- `reg_accuracy_tail_mixing_alignment`: n=`104`, R2=`0.926`, tail=-0.319, mixing=-0.746, alignment=-0.080
- `reg_margin_tail_graph_alignment`: n=`58`, R2=`0.787`, tail=-0.551, graph=-0.624, alignment=-0.353

### all_with_toy
- `reg_tail_mixing_alignment`: n=`221`, R2=`0.794`, mixing=0.528, alignment=-0.462
- `reg_tail_graph_alignment`: n=`134`, R2=`0.814`, graph=0.636, alignment=-0.330
- `reg_accuracy_tail_mixing_alignment`: n=`180`, R2=`0.624`, tail=-0.253, mixing=-0.611, alignment=-0.054
- `reg_margin_tail_graph_alignment`: n=`134`, R2=`0.748`, tail=-0.849, graph=-0.006, alignment=0.014

## Interpretation

- On image fixed-representation rows, local mixing and graph roughness
  retain signal after controlling for global alignment.
- The `all_with_toy` group is expected to be messier because XOR/global
  mismatch creates high tail without local collisions.
- Therefore local mixing should be described as one diagnosable source
  of spectral tail, not as a replacement for alignment.

## Artifacts

- `metrics.json`
- `figures/tail_mixing_alignment.png`
- `figures/tail_graph_alignment.png`
- `figures/accuracy_tail_mixing.png`
