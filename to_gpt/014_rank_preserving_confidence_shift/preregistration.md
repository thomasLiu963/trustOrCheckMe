# Task 014 preregistration / freeze

Timestamp: 2026-09-19T01:19:17.837539+00:00
Label: confirmatory for the rank-preserving offset; post-hoc relative to Tasks 009–013.

## Design (frozen before new Stage-3 calls)

- Transform: `q_delta = sigmoid(logit(q1) + delta)` with `delta ∈ [-1.5, -0.75, 0.0, 0.75, 1.5]`
- `delta = 0` returns the original q1 float (no logit round-trip)
- Exact q1 0 or 1 stays at that endpoint for every delta
- Display: `format(number, '.12g')` — same as Tasks 009/011
- Models/endpoints: {'openai_gpt56_sol': 'gpt-5.6-sol', 'anthropic_sonnet5': 'claude-sonnet-5'}
- δ=0 reuse: **REUSE_TRUE_Q_VISIBLE**
- Planned new scientific calls: **6288**
- SQLite: `/Users/thomas/Desktop/trustOrCheckMe/results/study014_offset/offset.sqlite3`
- Bootstrap: seed 20260927, 5000 item-clustered resamples
- Loss: Task 012 `leakage + λ * coverage` on the same λ grid

## Frozen item counts

- MMLU GPT/Claude: 500 each
- Code GPT/Claude: 286 confirmatory hard IDs each
- Exact q1 = 0: 6 (code only: 2 GPT, 4 Claude)
- Exact q1 = 1: 0

## Primary endpoints

A. `LEAKAGE(+1.5) − LEAKAGE(−1.5)`
B. `COVERAGE(+1.5) − COVERAGE(−1.5)`
C. Retention vs constant-score 0.70→0.99 leakage and coverage

## World buckets (GPT primary)

- `LEVEL_SENSITIVITY_STRONGLY_GENERALIZES`
- `LEVEL_SENSITIVITY_PARTIAL`
- `ITEM_INFORMATION_LARGELY_PROTECTS`
- `MIXED_BY_TASK`
- `TECHNICAL_OR_VERSION_FAILURE`

## Prompt rule

Stage-3 prompts are byte-identical to the corresponding 009/011 q-visible prompt except the displayed number. The model is not told that q was transformed.

## Hashes

- transform.py: `62239942abc6eba590e34505c7bfdf34f7636732c5391b306afb0af1037f4ba5`
- paperDirection.txt: `cc554e0502c960f2e51252e0fb511892c6c1f04afef43eb1bc9d90946ba3c452`
- 009 stage3: `9a4e9adc6c3a9aabba6015c9695db2dd9a5acb424cf52fbb89e008b2811d842f`
- 011 stage3: `76d775244594f98591386944b16785d6e407bbd1b61dfa8edd74c80f3a5a7c3a`

Do not inspect new VERIFY rates until the planned nonzero-delta cells complete or a technical stop is hit.
