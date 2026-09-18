# Confirmation infrastructure (not executed)

This is scaffolding for a later numbered confirmatory task. Task 006 does not freeze the confirmatory design and makes 0 confirmatory calls.

## Reusable pieces already in the repo

- Prompt freezing: `src/study1_prompts.py`, `src/task006_prompts.py`
- Model roster freeze: `config/models.yaml` + `create_adapter` ID locks
- Call cap / CheckpointStore: `src/checkpointing.py`, Study 1 / q2 / qualitative runners
- Parser: `parse_study1_verification_response` / qualitative wrapper
- Per-question paired design: one row per question × condition; GroupKFold by question ID
- Stage-1 generation (if a future protocol requires fresh answers): `src/prompts.py` Stage-1 template + `stage='answer'`
- Stage-2 q1 and optional q2: historical Stage-2 template in `src/prompts.py`; q2 runner pattern in `src/task005_q2.py`
- Verification decision: numeric Study-1 or qualitative Task-006 families
- Matched-budget routing: `src/task005_lane_a.py` fractional weights + `src/task006_lane_b.py`

## What a later confirmatory task still must freeze

1. Final N and whether to use one of the candidate lists in this folder
2. Numeric vs qualitative stakes (depends on Lane A gate)
3. Model roster (GPT/Claude only vs Gemini/Grok after Lane D)
4. Whether Stage-1 answers are reused or regenerated on unseen items
5. Scientific call cap
6. Confirmatory vs exploratory label, dated before any target-model calls on the fresh pool

Do not treat the candidate_N*.json files as the confirmatory sample.
