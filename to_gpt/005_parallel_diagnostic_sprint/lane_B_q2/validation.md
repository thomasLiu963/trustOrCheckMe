# Lane B q2 validation / preflight

- Planned scientific calls: **1000** (target 1000)
- Questions: **500** (all 500 V2-B)
- Models: `['anthropic_sonnet5', 'openai_gpt56_sol']`
- Prompt version: `stage_2_confidence_v3_structured_compat`
- Experiment version / new sqlite: `study2_q2_diagnostic` → `/Users/thomas/Desktop/trustOrCheckMe/results/study2_q2_diagnostic/q2.sqlite3`
- GPT endpoint: `gpt-5.6-sol` reasoning.effort=`none`
- Claude endpoint: `claude-sonnet-5` thinking=disabled
- Historical V2 sha256: `8597e3f73226bf39770af99a877282c22428203a3e2c11521256bf5cb3db5eb6`
- Study 1 sha256: `f8e5ddbb91db360403ea519b9b3481575c6cbc5ccdb553c650dbe726025ecfbd`
- paperDirection.txt will not be changed: **True**
- Prompt audits failed: **0**

Prompt audit: every planned prompt is the exact historical Stage-2 template with the frozen Stage-1 answer. q1, correctness, hidden action, and Study-1 displayed scores are not inserted.

## Paid run

- Scientific successes: **1000 / 1000**
- Failures: **0**
- Provider attempts: **1000**
- Parse repairs: **0**
- Returned models: `gpt-5.6-sol` × 500, `claude-sonnet-5` × 500
- New sqlite: `results/study2_q2_diagnostic/q2.sqlite3`
- Historical V2 / Study 1 / paperDirection hashes unchanged

