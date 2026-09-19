# Task 015 validation

Label: `POST_HOC_NORMATIVE_REANALYSIS`

## Reproduction of Task 014 coverage/leakage

Recomputed from `stage3_offset_results.csv` joined to frozen q1/labels.

Mismatches vs `coverage_by_delta.csv` / `leakage_by_delta.csv` (tolerance 1e-12): 0
None.

## GPT-code unmanipulated escape

leakage = 0.297203
error_rate = 0.534965
escape = leakage/error_rate = 0.555556
bootstrap point = 0.555556
95% CI [0.472393, 0.634618]

Handoff target: ~0.297 / 0.535 ≈ 0.55.

## Threshold convention

VERIFY iff r(Z) > λ. No λ/(1+λ) in 012/014 artifacts.

## Freeze

`task015_analysis_freeze.json` was written before empirical CSVs/figures in this run.

## Zero-call

No model adapters invoked. No Stage-3 requests.
