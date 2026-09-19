# Task 012 reproduction check

Label: `POST_HOC_SYSTEMS_REANALYSIS`. Zero API calls. Frozen 009/011 artifacts only.

Status: **PASS**

Rounding tolerance: ±0.6 percentage points.

| Task | Model | Metric | Published | Reproduced | N | OK |
|---|---|---|---:|---:|---:|:---:|
| mmlu | openai_gpt56_sol | coverage_0.70_0.99 | 50.2 | 50.2 | 500 | yes |
| mmlu | anthropic_sonnet5 | coverage_0.70_0.99 | 35.4 | 35.4 | 500 | yes |
| code | openai_gpt56_sol | coverage_0.70_0.99 | 38.8 | 38.811 | 286 | yes |
| code | anthropic_sonnet5 | coverage_0.70_0.99 | 4.9 | 4.895 | 286 | yes |
| code | openai_gpt56_sol | pass_rate | 46.5 | 46.503 | 286 | yes |
| code | anthropic_sonnet5 | pass_rate | 43.0 | 43.007 | 286 | yes |

MMLU filter: `roster=primary`, `repeat_index=0`, GPT+Claude.
Code filter: confirmatory `sample_main.csv` IDs only, `repeat_index=0`.
Code pass rate: official hidden-suite `passed` on those IDs.

