# Task 003 parse-repair audit (zero-cost, before Task 004 paid calls)

This is an engineering diagnostic, not a new hypothesis test.
No new API calls were made to produce this audit.

- Distinct cells with a parse-repair attempt: **52**
- GPT repairs: **0**
- Claude repairs: **52**
- Missing from the frozen primary plan: **0**

## Counts by model

- `anthropic_sonnet5`: 52

## Counts by L

- L=10.0: 28
- L=20.0: 24

## Counts by condition

- `hidden`: 10
- `manipulated_1`: 3
- `manipulated_2`: 7
- `manipulated_3`: 10
- `manipulated_4`: 9
- `manipulated_5`: 8
- `true_confidence_visible`: 5

## Counts by displayed confidence (includes true-visible historical values)

- displayed=0.72: 2
- displayed=0.8: 1
- displayed=0.88: 2
- displayed=0.89: 7
- displayed=0.9: 3
- displayed=0.91: 6
- displayed=0.93: 5
- displayed=0.94: 3
- displayed=0.95: 1
- displayed=0.96: 3
- displayed=0.97: 1
- displayed=0.99: 8
- displayed=hidden: 10

## Claude repair share of all Claude primary cells, by condition

- `hidden`: 10/200 = 0.050
- `manipulated_1`: 3/200 = 0.015
- `manipulated_2`: 7/200 = 0.035
- `manipulated_3`: 10/200 = 0.050
- `manipulated_4`: 9/200 = 0.045
- `manipulated_5`: 8/200 = 0.040
- `true_confidence_visible`: 5/200 = 0.025

## Key threshold-pair repair-rate gaps (Claude, L-specific)

- L10_0.89_vs_0.91: 0.070 vs 0.060 (gap 0.010)
- L20_0.94_vs_0.96: 0.030 vs 0.030 (gap 0.000)
- L10_0.80_vs_0.99: 0.010 vs 0.040 (gap 0.030)
- L20_0.90_vs_0.99: 0.020 vs 0.040 (gap 0.020)

## Final parsed actions, repaired vs unrepaired

- GPT repaired: n=0, VERIFY rate=None
- GPT other: n=1400, VERIFY rate=0.5042857142857143
- Claude repaired: n=52, VERIFY rate=0.9615384615384616
- Claude other: n=1348, VERIFY rate=0.7314540059347181
- Repaired-cell final actions: {'VERIFY_FIRST': 50, 'USE_UNVERIFIED': 2}

## Interpretation

- Repairs are concentrated in Claude. GPT had zero parse-repair attempts, consistent with structured JSON decoding.
- Claude condition repair rates range from 0.015 to 0.050. That is a Claude JSON-reliability issue spread across conditions, not a spike in one scientific cell type.
- The largest L-specific key-contrast repair-rate gap is 0.030. Threshold pairs are not systematically repair-imbalanced.
- Repaired Claude cells end as VERIFY more often than unrepaired Claude cells. That can slightly inflate Claude's overall VERIFY rate (on the order of 1pp if every extra VERIFY is attributed to repair), but it is not localized to a displayed-confidence condition.
- One Claude cell in Task 003 required a same-key rerun after truncated JSON; it is already in the completed primary dataset.

## Paid-call gate

**PROCEED.** Repair attempts are not strongly condition-dependent in a way that would distort Study-1 confidence contrasts. Task 004 paid repeat-extra calls may proceed.
