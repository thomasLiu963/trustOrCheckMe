# Task 014 transformation audit

Computed from frozen q1/correctness only. No new Stage-3 outcomes were inspected.

Exact q1 endpoints 0 or 1: 6 item-model rows.
Order reversals after `.12g` rendering: **0**.

| Task | Model | δ | mean | p50 | ≥0.90 | ≤0.70 | Spearman | reversals | AUROC | Brier | ECE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mmlu | openai_gpt56_sol | -1.5 | 0.835 | 0.957 | 65.6% | 15.2% | 1.000000 | 0 | 0.671 | 0.156 | 0.085 |
| mmlu | openai_gpt56_sol | -0.75 | 0.887 | 0.979 | 76.4% | 10.2% | 1.000000 | 0 | 0.671 | 0.160 | 0.124 |
| mmlu | openai_gpt56_sol | 0.0 | 0.920 | 0.990 | 86.4% | 5.4% | 1.000000 | 0 | 0.671 | 0.168 | 0.152 |
| mmlu | openai_gpt56_sol | 0.75 | 0.939 | 0.995 | 91.6% | 4.8% | 1.000000 | 0 | 0.671 | 0.173 | 0.168 |
| mmlu | openai_gpt56_sol | 1.5 | 0.952 | 0.998 | 94.8% | 4.6% | 1.000000 | 0 | 0.671 | 0.175 | 0.174 |
| mmlu | anthropic_sonnet5 | -1.5 | 0.450 | 0.365 | 2.0% | 86.6% | 1.000000 | 0 | 0.730 | 0.263 | 0.308 |
| mmlu | anthropic_sonnet5 | -0.75 | 0.601 | 0.548 | 5.4% | 59.8% | 1.000000 | 0 | 0.730 | 0.187 | 0.157 |
| mmlu | anthropic_sonnet5 | 0.0 | 0.739 | 0.720 | 25.0% | 34.8% | 1.000000 | 0 | 0.730 | 0.159 | 0.034 |
| mmlu | anthropic_sonnet5 | 0.75 | 0.845 | 0.845 | 40.2% | 9.4% | 1.000000 | 0 | 0.730 | 0.167 | 0.087 |
| mmlu | anthropic_sonnet5 | 1.5 | 0.914 | 0.920 | 69.8% | 2.2% | 1.000000 | 0 | 0.730 | 0.189 | 0.156 |
| code | openai_gpt56_sol | -1.5 | 0.801 | 0.957 | 79.4% | 16.1% | 1.000000 | 0 | 0.666 | 0.351 | 0.348 |
| code | openai_gpt56_sol | -0.75 | 0.822 | 0.979 | 83.9% | 16.1% | 1.000000 | 0 | 0.666 | 0.369 | 0.367 |
| code | openai_gpt56_sol | 0.0 | 0.834 | 0.990 | 83.9% | 16.1% | 1.000000 | 0 | 0.666 | 0.379 | 0.377 |
| code | openai_gpt56_sol | 0.75 | 0.843 | 0.995 | 83.9% | 16.1% | 1.000000 | 0 | 0.666 | 0.384 | 0.385 |
| code | openai_gpt56_sol | 1.5 | 0.852 | 0.998 | 83.9% | 16.1% | 1.000000 | 0 | 0.666 | 0.388 | 0.391 |
| code | anthropic_sonnet5 | -1.5 | 0.298 | 0.251 | 0.0% | 90.2% | 1.000000 | 0 | 0.768 | 0.227 | 0.136 |
| code | anthropic_sonnet5 | -0.75 | 0.391 | 0.415 | 0.0% | 65.0% | 1.000000 | 0 | 0.768 | 0.220 | 0.146 |
| code | anthropic_sonnet5 | 0.0 | 0.475 | 0.600 | 21.7% | 57.7% | 1.000000 | 0 | 0.768 | 0.227 | 0.166 |
| code | anthropic_sonnet5 | 0.75 | 0.549 | 0.761 | 35.0% | 46.9% | 1.000000 | 0 | 0.768 | 0.239 | 0.193 |
| code | anthropic_sonnet5 | 1.5 | 0.618 | 0.871 | 42.7% | 45.5% | 1.000000 | 0 | 0.768 | 0.255 | 0.223 |

## δ=0 reuse

Items compared: 1572
Byte-identical prompts: 1572
Decision: **REUSE_TRUE_Q_VISIBLE**

