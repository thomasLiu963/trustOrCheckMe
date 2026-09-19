# Task 012 paper-impact decision

Label: `POST_HOC_SYSTEMS_REANALYSIS`

## Decision

`MATERIAL_UPGRADE`

## Frozen gates (from task012_analysis_freeze.json)

A. Consequence: |Δ leakage 0.70→0.99| ≥ 0.05 and conservative CI excludes 0, in ≥2 task/model cells.
B. Operating range: ≥25% of q1 in [0.70, 0.99] OR IQR intersects that interval, in the relevant cells.
C. Systems: ≥3 consecutive λ-grid points with regret vs best observed fixed score ≥ 0.05.
D. Reproduction passes; leakage contrasts remain causally attributable under the frozen-output design.

## Evaluation

- A = True
- B = True
- C = True (max consecutive λ = 8)
- D = True

### Cell-level leakage 0.70→0.99

| Task | Model | Δ leakage | conservative CI | material |
|---|---|---|---|---|
| mmlu | GPT | +11.8pp | [+9.0, +14.6] | yes |
| mmlu | Claude | +3.8pp | [+2.2, +5.6] | no |
| code | GPT | +22.0pp | [+17.5, +26.9] | yes |
| code | Claude | +1.4pp | [+0.3, +2.8] | no |

### q1 overlap

| Task | Model | frac in [0.70,0.99] | IQR intersects | pass |
|---|---|---|---|---|
| mmlu | GPT | 92.0% | yes | yes |
| code | GPT | 71.3% | yes | yes |

## Action

Write proposed paperDirection replacement. Do not overwrite paperDirection.txt.
