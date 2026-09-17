# D1 Decision Log

Maintain this file as specified in `docs/D1_EXPERIMENTAL_PROTOCOL_v1.0_DRAFT.md` §69.

Every meaningful scientific change is appended. Do not rewrite historical entries if a later decision changes. Record:

- date
- previous plan / previous state
- evidence that motivated the change
- new decision
- whether the affected study had already been frozen
- whether existing data are exploratory or confirmatory relative to the new decision
- current next engineering task
- status labels

---

## 2026-09-17 — Proceed to a small exploratory causal confidence pilot

**Date:** 2026-09-17

**Previous state:**  
Original V2 and Checkpoint A complete.

**Previous plan:**  
The original frozen V2 protocol (`trust_check_v2`, freeze date 2026-09-04) treated decision authority as the primary question and confidence visibility as a mechanism question. After V2 data were observed, the workshop manuscript emphasized confidence visibility. Checkpoint A then re-analyzed the frozen V2-B data with no new API calls.

**Evidence:**  
Checkpoint A showed that visible GPT nearly follows its raw confidence threshold, much of its high-stakes failure is explained by overconfidence/reduced verification, Claude shows a clear equal-budget hidden-policy advantage, GPT and Claude hidden actions contain information beyond verbal confidence, and the proposed simple cross-model law failed.

**Decision:**  
Proceed to a small exploratory causal confidence pilot before any full expansion or mechanistic work.

**Affected study already frozen?**  
No. Historical V2-B remains HISTORICAL-FROZEN and is not reopened. Checkpoint A remains an exploratory re-analysis of that frozen data. The Study 1 causal pilot is not frozen as confirmatory. No Study 1 generations have been collected.

**Exploratory vs confirmatory relative to this decision:**  
Historical V2 and Checkpoint A are exploratory relative to the rebuilt causal-control hypothesis. Study 1 will also be EXPLORATORY. Prospective confirmation requires a later dated protocol addendum before fresh confirmatory generations.

**Current next engineering task:**  
Build and audit the Study-1 pilot infrastructure; do not launch the full pilot yet.

**Status:**  
Study 1 = EXPLORATORY.  
Later studies = CONDITIONAL.

---

## 2026-09-17 — Freeze Study 1 wording and L=20 confidence grid

**Date:** 2026-09-17

**Previous state:**  
Task 000 workflow bootstrap complete. Study 1 infrastructure not yet built. Draft protocol still listed a 0.949/0.951 vs simpler-format L=20 grid ambiguity, and a wording example (“Your previously reported probability”) that differed from historical V2 visible wording.

**Evidence:**  
Task `from_gpt/001_build_study1_pilot.md` is the current numbered engineering instruction. Continuity with the historical visibility effect is the stated reason for keeping historical V2 AI-authority Stage-3 wording. The same task resolves the L=20 formatting ambiguity in favor of the simpler natural-looking grid.

**Decision:**  
For the Study 1 exploratory pilot:

1. Use historical V2 AI-authority primary wording from `src/v2_prompts.py:build_verification_prompt` (`v2_owner_match_v1`, paraphrase=False). Do not switch to the protocol’s “Your previously reported…” example.
2. Use L=10 grid `0.80, 0.88, 0.89, 0.91, 0.99` (local contrast 0.89 vs 0.91).
3. Use L=20 grid `0.90, 0.93, 0.94, 0.96, 0.99` (local contrast 0.94 vs 0.96).
4. Authority = AI-system only. Models = GPT-5.6 Sol and Claude Sonnet 5 only.

The 100-question sample and 20-question repeat subset were frozen before any Study 1 model calls.

**Affected study already frozen?**  
No. Study 1 remains exploratory / not confirmatory. Historical V2 remains HISTORICAL-FROZEN and was not modified.

**Exploratory vs confirmatory relative to this decision:**  
These freezes apply to the exploratory Study 1 pilot only. They do not freeze Study 2.

**Current next engineering task:**  
GPT inspects `to_gpt/001_study1_pilot_build/` and, if satisfied, may authorize a tiny real API smoke test as Task 002. The full 2,800/1,120 pilot is not authorized.

**Status:**  
Study 1 infrastructure = DEVELOPMENT / EXPLORATORY.  
Study 1 data collection = NOT STARTED.  
Later studies = CONDITIONAL.

---

## 2026-09-17 — Task 002 12-cell engineering smoke test (not a scientific result)

**Date:** 2026-09-17

**Previous state:**  
Study 1 infrastructure built and audited offline. No Study 1 model generations. Full 2,800/1,120 pilot not authorized.

**Evidence:**  
Numbered task `from_gpt/002_smoke_test_study1.md` explicitly authorizes a tiny real API smoke test of 12 scientific cells and at most 20 provider request attempts, including retries and parse repairs. It forbids treating n=1 as a scientific test and forbids inferring authorization for the full pilot.

**Decision:**  
Run only the 12 specified cells on the first frozen ID (`mmlu_pro:test:7552`) for GPT and Claude. Do not add questions, models, confidence values, repeats, or controls. Do not interpret action changes. Do not launch the 2,800-call primary.

**Affected study already frozen?**  
No. Study 1 remains exploratory / DEVELOPMENT. Historical V2 remains HISTORICAL-FROZEN.

**Exploratory vs confirmatory relative to this decision:**  
The smoke-test outputs are engineering diagnostics, not exploratory scientific evidence and not confirmatory data.

**Current next engineering task:**  
Return `to_gpt/002_study1_smoke_test/`. GPT decides whether the engineering pipeline is ready enough to consider authorizing the full primary pilot as a later numbered task.

**Status:**  
Study 1 infrastructure = DEVELOPMENT / EXPLORATORY.  
Study 1 data collection = SMOKE TEST ONLY (not a scientific sample).  
Full primary/repeats = NOT AUTHORIZED.

