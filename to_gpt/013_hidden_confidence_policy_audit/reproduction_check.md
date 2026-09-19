# Task 013 reproduction check

Label: `POST_HOC_POLICY_REANALYSIS`. Zero API calls. Frozen 009–012 artifacts only.

Status: **PASS**

Rounding tolerance: ±0.6 percentage points.

| Source | Task | Model | Metric | Target | Reproduced | OK |
|---|---|---|---|---|---|:---:|
| published_009_011 | mmlu | openai_gpt56_sol | hidden_coverage | 39.6 | 39.6 | yes |
| published_009_011 | mmlu | openai_gpt56_sol | true_q_visible_coverage | 12.4 | 12.4 | yes |
| published_009_011 | mmlu | anthropic_sonnet5 | hidden_coverage | 65.4 | 65.4 | yes |
| published_009_011 | mmlu | anthropic_sonnet5 | true_q_visible_coverage | 79.4 | 79.4 | yes |
| published_009_011 | code | openai_gpt56_sol | hidden_coverage | 32.5 | 32.517 | yes |
| published_009_011 | code | openai_gpt56_sol | true_q_visible_coverage | 27.3 | 27.273 | yes |
| published_009_011 | code | anthropic_sonnet5 | hidden_coverage | 98.6 | 98.601 | yes |
| published_009_011 | code | anthropic_sonnet5 | true_q_visible_coverage | 97.9 | 97.902 | yes |
| task012 | mmlu | openai_gpt56_sol | hidden_leakage | 3.8 | 3.8 | yes |
| task012 | mmlu | openai_gpt56_sol | hidden_coverage | 39.6 | 39.6 | yes |
| task012 | mmlu | openai_gpt56_sol | true_q_visible_leakage | 12.0 | 12.0 | yes |
| task012 | mmlu | openai_gpt56_sol | true_q_visible_coverage | 12.4 | 12.4 | yes |
| task012 | mmlu | openai_gpt56_sol | displayed_0.70_leakage | 2.0 | 2.0 | yes |
| task012 | mmlu | openai_gpt56_sol | displayed_0.70_coverage | 58.4 | 58.4 | yes |
| task012 | mmlu | openai_gpt56_sol | displayed_0.85_leakage | 8.4 | 8.4 | yes |
| task012 | mmlu | openai_gpt56_sol | displayed_0.85_coverage | 22.8 | 22.8 | yes |
| task012 | mmlu | openai_gpt56_sol | displayed_0.90_leakage | 10.2 | 10.2 | yes |
| task012 | mmlu | openai_gpt56_sol | displayed_0.90_coverage | 18.0 | 18.0 | yes |
| task012 | mmlu | openai_gpt56_sol | displayed_0.95_leakage | 11.4 | 11.4 | yes |
| task012 | mmlu | openai_gpt56_sol | displayed_0.95_coverage | 14.4 | 14.4 | yes |
| task012 | mmlu | openai_gpt56_sol | displayed_0.99_leakage | 13.8 | 13.8 | yes |
| task012 | mmlu | openai_gpt56_sol | displayed_0.99_coverage | 8.2 | 8.2 | yes |
| task012 | mmlu | anthropic_sonnet5 | hidden_leakage | 2.8 | 2.8 | yes |
| task012 | mmlu | anthropic_sonnet5 | hidden_coverage | 65.4 | 65.4 | yes |
| task012 | mmlu | anthropic_sonnet5 | true_q_visible_leakage | 1.0 | 1.0 | yes |
| task012 | mmlu | anthropic_sonnet5 | true_q_visible_coverage | 79.4 | 79.4 | yes |
| task012 | mmlu | anthropic_sonnet5 | displayed_0.70_leakage | 0.0 | 0.0 | yes |
| task012 | mmlu | anthropic_sonnet5 | displayed_0.70_coverage | 92.4 | 92.4 | yes |
| task012 | mmlu | anthropic_sonnet5 | displayed_0.85_leakage | 1.2 | 1.2 | yes |
| task012 | mmlu | anthropic_sonnet5 | displayed_0.85_coverage | 80.0 | 80.0 | yes |
| task012 | mmlu | anthropic_sonnet5 | displayed_0.90_leakage | 1.4 | 1.4 | yes |
| task012 | mmlu | anthropic_sonnet5 | displayed_0.90_coverage | 77.6 | 77.6 | yes |
| task012 | mmlu | anthropic_sonnet5 | displayed_0.95_leakage | 2.4 | 2.4 | yes |
| task012 | mmlu | anthropic_sonnet5 | displayed_0.95_coverage | 64.6 | 64.6 | yes |
| task012 | mmlu | anthropic_sonnet5 | displayed_0.99_leakage | 3.8 | 3.8 | yes |
| task012 | mmlu | anthropic_sonnet5 | displayed_0.99_coverage | 57.0 | 57.0 | yes |
| task012 | code | openai_gpt56_sol | hidden_leakage | 25.874 | 25.874 | yes |
| task012 | code | openai_gpt56_sol | hidden_coverage | 32.517 | 32.517 | yes |
| task012 | code | openai_gpt56_sol | true_q_visible_leakage | 29.72 | 29.72 | yes |
| task012 | code | openai_gpt56_sol | true_q_visible_coverage | 27.273 | 27.273 | yes |
| task012 | code | openai_gpt56_sol | displayed_0.70_leakage | 10.839 | 10.839 | yes |
| task012 | code | openai_gpt56_sol | displayed_0.70_coverage | 62.937 | 62.937 | yes |
| task012 | code | openai_gpt56_sol | displayed_0.85_leakage | 18.881 | 18.881 | yes |
| task012 | code | openai_gpt56_sol | displayed_0.85_coverage | 44.406 | 44.406 | yes |
| task012 | code | openai_gpt56_sol | displayed_0.90_leakage | 21.329 | 21.329 | yes |
| task012 | code | openai_gpt56_sol | displayed_0.90_coverage | 37.762 | 37.762 | yes |
| task012 | code | openai_gpt56_sol | displayed_0.95_leakage | 27.972 | 27.972 | yes |
| task012 | code | openai_gpt56_sol | displayed_0.95_coverage | 29.021 | 29.021 | yes |
| task012 | code | openai_gpt56_sol | displayed_0.99_leakage | 32.867 | 32.867 | yes |
| task012 | code | openai_gpt56_sol | displayed_0.99_coverage | 24.126 | 24.126 | yes |
| task012 | code | anthropic_sonnet5 | hidden_leakage | 0.35 | 0.35 | yes |
| task012 | code | anthropic_sonnet5 | hidden_coverage | 98.601 | 98.601 | yes |
| task012 | code | anthropic_sonnet5 | true_q_visible_leakage | 1.399 | 1.399 | yes |
| task012 | code | anthropic_sonnet5 | true_q_visible_coverage | 97.902 | 97.902 | yes |
| task012 | code | anthropic_sonnet5 | displayed_0.70_leakage | 0.0 | 0.0 | yes |
| task012 | code | anthropic_sonnet5 | displayed_0.70_coverage | 99.65 | 99.65 | yes |
| task012 | code | anthropic_sonnet5 | displayed_0.85_leakage | 0.35 | 0.35 | yes |
| task012 | code | anthropic_sonnet5 | displayed_0.85_coverage | 99.301 | 99.301 | yes |
| task012 | code | anthropic_sonnet5 | displayed_0.90_leakage | 1.049 | 1.049 | yes |
| task012 | code | anthropic_sonnet5 | displayed_0.90_coverage | 98.601 | 98.601 | yes |
| task012 | code | anthropic_sonnet5 | displayed_0.95_leakage | 1.399 | 1.399 | yes |
| task012 | code | anthropic_sonnet5 | displayed_0.95_coverage | 97.552 | 97.552 | yes |
| task012 | code | anthropic_sonnet5 | displayed_0.99_leakage | 1.399 | 1.399 | yes |
| task012 | code | anthropic_sonnet5 | displayed_0.99_coverage | 94.755 | 94.755 | yes |
| task012 | both | shared | lambda_grid | 012_grid | 0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1.0 | yes |
| task012 | both | shared | loss_function | leakage + lambda * coverage | leakage + lambda * coverage | yes |

Filters match Task 012: 009 primary repeat 0; 011 confirmatory IDs only.

