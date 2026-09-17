# Task 002 report — Study 1 API smoke test

Engineering inspection only. n = 1 question. These 12 cells are not a scientific test.
Do not treat action changes across fake confidence values as evidence for or against the hypothesis.

## Required answers

- What question ID was used? `mmlu_pro:test:7552` (first ID in frozen `data/development/study1_ids.json`)
- GPT historical reported confidence: `0.78`
- Claude historical reported confidence: `0.6`
- Did pre-flight validation pass? **True**
- Were exactly 12 scientific cells attempted? **True** (initiated=12)
- How many completed successfully? **12**
- How many raw provider request attempts occurred? **12**
- Were any retries needed? **False** (retry attempts=0)
- Were any parse repairs needed? **False** (repair attempts=0)
- Did every valid response parse into the two-action schema? **True**
- Were reported_confidence and displayed_confidence stored correctly? **True**
- Was historical V2 untouched? **True**
- Exact provider model/version metadata returned: `{"openai_gpt56_sol": ["gpt-5.6-sol"], "anthropic_sonnet5": ["claude-sonnet-5"]}`
- Total measured/estimated API cost: **$0.020056**
- Wall-clock runtime: **22.775 seconds**
- Did the call caps work? Scientific cells initiated=12 (cap 12); provider attempts=12 (cap 20). 13th cell and 21st attempt are refused.
- Is the pipeline technically ready for the full 2,800-call primary pilot? **YES, engineering-only**. This does not authorize Task 003.
- Engineering issue GPT must resolve first: none observed in this smoke test

## Compact raw action table

| model | L | condition | reported | displayed | action | attempts | latency_ms | tokens | cost |
|---|---:|---|---:|---:|---|---:|---:|---:|---:|
| openai_gpt56_sol | 10 | hidden | 0.78 | null | VERIFY_FIRST | 1 | 2986 | 352 | 0.001632 |
| openai_gpt56_sol | 10 | true_confidence_visible | 0.78 | 0.78 | VERIFY_FIRST | 1 | 1956 | 369 | 0.001700 |
| openai_gpt56_sol | 10 | manipulated_3 | 0.78 | 0.89 | VERIFY_FIRST | 1 | 969 | 369 | 0.001700 |
| openai_gpt56_sol | 10 | manipulated_4 | 0.78 | 0.91 | USE_UNVERIFIED | 1 | 1349 | 371 | 0.001740 |
| openai_gpt56_sol | 20 | manipulated_3 | 0.78 | 0.94 | VERIFY_FIRST | 1 | 1326 | 369 | 0.001700 |
| openai_gpt56_sol | 20 | manipulated_4 | 0.78 | 0.96 | USE_UNVERIFIED | 1 | 1408 | 371 | 0.001740 |
| anthropic_sonnet5 | 10 | hidden | 0.6 | null | VERIFY_FIRST | 1 | 3521 | 725 | 0.001594 |
| anthropic_sonnet5 | 10 | true_confidence_visible | 0.6 | 0.6 | VERIFY_FIRST | 1 | 1604 | 753 | 0.001650 |
| anthropic_sonnet5 | 10 | manipulated_3 | 0.6 | 0.89 | VERIFY_FIRST | 1 | 2116 | 753 | 0.001650 |
| anthropic_sonnet5 | 10 | manipulated_4 | 0.6 | 0.91 | VERIFY_FIRST | 1 | 1737 | 753 | 0.001650 |
| anthropic_sonnet5 | 20 | manipulated_3 | 0.6 | 0.94 | VERIFY_FIRST | 1 | 1706 | 753 | 0.001650 |
| anthropic_sonnet5 | 20 | manipulated_4 | 0.6 | 0.96 | VERIFY_FIRST | 1 | 2015 | 753 | 0.001650 |

This table is for engineering inspection only. It is not evidence for or against the hypothesis.

## Caps

The runner refuses a 13th scientific cell and refuses a 21st provider attempt.
`study1-run --yes` remains blocked for the 2,800-call primary and 1,120-call repeats.

## Engineering readiness

The live GPT and Claude verification path, parser, Study-1 sqlite writes, and call caps worked on this 12-cell subset.

No post-run issues recorded.

READY_FOR_TASK_003 = YES
