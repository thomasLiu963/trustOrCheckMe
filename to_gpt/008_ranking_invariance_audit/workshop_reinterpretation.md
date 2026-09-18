# Workshop / Checkpoint A reinterpretation

Exploratory only. Not confirmatory. Distinguishes coverage, ranking, and calibration.

## Self-correction paragraph (use only if retained)

The original workshop observation that *visible* confidence reduced GPT’s error-catching is primarily an operating-point result, not a demonstration that ranking quality collapsed. On V2-B, showing GPT its number cut coverage by about -18.8 pp on average and agreed with the raw-confidence threshold on 99.2% of answers. Visible GPT therefore slides left along the confidence risk-coverage curve. At matched budget, hidden GPT’s ranking edge over confidence-only is only 7.1 pp and the pooled interval includes zero (Checkpoint A). Calibration still matters for *cost*: an overconfident scalar plus a faithful threshold leaves too many errors unchecked at high L. That is a control-variable problem, not a ranking-degradation result. Claude is different on ranking: hidden Claude beats confidence-only at matched budget by about 9.6 pp, while visibility *increases* Claude’s coverage (mean shift 18.9 pp). Do not collapse these into one law.

## Coverage vs ranking vs calibration

| model | mean coverage shift vis−hid (pp) | vis vs raw-threshold agree (pp) | hidden−conf catch at matched budget (pp) |
|---|---:|---:|---:|
| GPT | -18.8 | 99.2 | 7.1 |
| Claude | 18.9 | 76.3 | 9.6 |

Study 1 numeric GPT cells are near-saturated on each side of the instructed threshold, so errors-caught differences across manipulated scores are almost entirely coverage. GPT manipulated condition-cells in `workshop_study1_roc_points.csv`: 10 (each is 100 questions).

Do not force this paragraph if a later confirmatory study shows ranking collapse at matched coverage.
