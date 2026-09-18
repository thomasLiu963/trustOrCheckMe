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

---

## 2026-09-17 — Task 003 authorizes the 2,800-cell primary only

**Date:** 2026-09-17

**Previous state:**  
Task 002 12-cell engineering smoke test succeeded. Full primary and repeats were still unauthorized.

**Evidence:**  
Numbered task `from_gpt/003_run_study1_primary.md` authorizes the frozen 2,800 primary cells and requires analysis of that primary dataset. It withholds the 1,120 repeat-extra cells, micro-controls, and later studies. Task-002 cells for `mmlu_pro:test:7552` reuse identical primary request keys.

**Decision:**  
Run remaining primary cells after reusing the 12 valid smoke-test records. Do not run repeats or controls. Analyze the 2,800-cell primary dataset as exploratory evidence. Do not claim the effect exceeds ordinary rerun noise.

**Affected study already frozen?**  
No. Study 1 remains exploratory. Historical V2 remains HISTORICAL-FROZEN.

**Exploratory vs confirmatory relative to this decision:**  
The 2,800-cell primary is EXPLORATORY. Repeats, if later authorized, would still be exploratory stability evidence, not confirmatory.

**Current next engineering task:**  
Return `to_gpt/003_study1_primary_results/` for GPT review. Do not launch Task 004.

**Status:**  
Study 1 primary = AUTHORIZED / EXPLORATORY.  
Study 1 repeats = NOT AUTHORIZED.  
Later studies = CONDITIONAL.

---

## 2026-09-17 — Task 004 authorizes the frozen 1,120 repeat-extra cells only

**Date:** 2026-09-17

**Previous state:**  
Task 003 completed the 2,800-cell primary dataset. GPT showed a large near-threshold displayed-confidence effect; Claude showed a smaller same-direction effect. Repeats were still unauthorized. The 20-question repeat subset was frozen under seed 20260918 before Task-003 outcomes were observed.

**Evidence:**  
Numbered task `from_gpt/004_run_study1_stability.md` authorizes exactly 1,120 already-frozen repeat-extra scientific calls (20 IDs × 2 models × 2 L × 7 conditions × 2 additional repetitions). It forbids new scientific conditions, provenance/qualitative-stakes/contradiction controls, sample expansion, and rerunning primary observations. A zero-cost Task-003 parse-repair audit must precede paid calls.

**Decision:**  
Run only the frozen repeat-extra cells. Use independent generation as the only intended difference. Compare displayed-confidence action change with identical-prompt rerun noise. Apply the final exploratory Study-1 gate A/B/C/D. Do not run micro-controls, Deep Research, or confirmatory design.

**Affected study already frozen?**  
No. Study 1 remains exploratory. Historical V2 remains HISTORICAL-FROZEN. The 20 repeat IDs remain the Task-001 freeze (hash `45e06fe9daf40dd1cbd9cbf91b49cef09c5d924df17f2cf2bf2f203090c26d38`).

**Exploratory vs confirmatory relative to this decision:**  
The 1,120 extras and the resulting stability analysis are EXPLORATORY. They can support a PASS CAUSAL BEHAVIORAL PHENOMENON in this controlled setup; they do not freeze a confirmatory protocol.

**Current next engineering task:**  
Return `to_gpt/004_study1_stability_results/` for GPT review. Do not launch later studies.

**Status:**  
Study 1 repeats = AUTHORIZED / EXPLORATORY.  
Later studies = CONDITIONAL.

---

## 2026-09-17 — Task 005 parallel diagnostic sprint (exploratory, not confirmatory)

**Date:** 2026-09-17

**Previous state:**  
Study 1 primary (2,800) and stability repeats (1,120) complete. GPT showed a large, rerun-stable displayed-confidence cliff. Claude was a weaker/gradual contrast. Checkpoint A had shown hidden verification carrying extra correctness information beyond verbal confidence `q1`.

**Previous plan:**  
Do not scale blindly or write the paper. Resolve three uncertainties: useful item-specific verification at fixed displayed score; whether hidden-action residual is just a second confidence assessment; whether GPT's cliff is specific to low/no-reasoning mode. Prepare, but do not run, a qualitative-stakes experiment.

**Evidence:**  
Numbered task `from_gpt/005_parallel_diagnostic_sprint.md`.

**Decision:**

1. Lane A analyzes existing Study-1 and historical V2 data with zero API calls. Question identity is the unit for correctness analyses. GPT action saturation is labeled `ACTION_SATURATED__ITEM_DISCRIMINATION_NOT_IDENTIFIABLE` and is not translated into hidden-state suppression claims.
2. Lane B re-elicits an independent Stage-2 confidence `q2` on all 500 V2-B questions for the Study-1 GPT and Claude endpoints (1,000 scientific calls). Exact historical Stage-2 template. New sqlite `results/study2_q2_diagnostic/q2.sqlite3`. Do not write Study-1 or V2 databases.
3. Lane C is `BLOCKED_UNSUPPORTED_REASONING_SETTING`. The current OpenAI adapter/config only support `reasoning.effort=none` for `gpt-5.6-sol`. Zero Lane-C calls. No guessed nonzero effort values.
4. Lane D drafts two qualitative-stakes prompt families for GPT review. Zero calls. Prompts are not frozen.
5. Do not modify `paperDirection.txt`. Do not call any result confirmatory. Do not auto-scale after seeing q2.

**Affected study already frozen?**  
Historical V2 remains HISTORICAL-FROZEN. Study 1 remains exploratory. Task 003/004 data are not overwritten.

**Exploratory vs confirmatory relative to this decision:**  
All Task 005 outputs are EXPLORATORY diagnostics.

**Current next engineering task:**  
Return `to_gpt/005_parallel_diagnostic_sprint/` for GPT review. Do not execute the recommended next scientific action.

**Status:**  
Task 005 = AUTHORIZED / EXPLORATORY.  
Lane C = BLOCKED_UNSUPPORTED_REASONING_SETTING.  
Later studies = CONDITIONAL.  
Confirmatory study = NOT AUTHORIZED.

---

## 2026-09-17 — Task 006 qualitative 20-question pilot and parallel prep

**Date:** 2026-09-17

**Previous state:**  
Study 1 and Task 005 complete as exploratory. Task 005B difficulty-control audit ran on existing Claude cells (zero API). GPT's Study-1 cliff remains consistent with explicit numerical L/C arithmetic; that objection was untested.

**Evidence:**  
Numbered task `from_gpt/006_qualitative_pilot_parallel_prep.md`.

**Decision:**

1. Lane A runs a 20-question qualitative-stakes pilot (320 scientific calls) using the Task-006 frozen prompt skeleton, not the Task-005 Lane D drafts. No numerical L/C, expected-value language, `outweigh`, `justified`, or `not as a default`. Score grid: hidden / 0.70 / 0.90 / 0.99. Families: moderate vs stronger consequence sentences specified in the task. Frozen Stage-1 answers reused. New sqlite `results/study1_qualitative_pilot/qualitative_pilot.sqlite3`.
2. Lane B is a zero-call exploratory matched-budget routing analysis of q1, q2, a predeclared single hidden cell (`ai_system`, L=10, hidden), the hidden aggregate (labeled multi-elicitation / not deployment-fair), and a grouped-CV q1+q2+single-hidden combination. Question identity is the unit.
3. Lane C builds an unseen MMLU-Pro eligible pool excluding the 500 V2-B IDs and writes candidate stratified lists for N=200…1000. No confirmatory sample is frozen. No target-model calls.
4. Lane D audits Gemini/Grok/open-model feasibility in code/config only. No provider calls.
5. Do not modify `paperDirection.txt`, historical V2, Study 1, q2, Task 005, or Task 005B data. Do not auto-expand from 20 to 100 questions.

**Affected study already frozen?**  
Historical V2 remains HISTORICAL-FROZEN. Study 1 / Tasks 003–005 remain exploratory and are not overwritten.

**Exploratory vs confirmatory relative to this decision:**  
All Task 006 outputs are EXPLORATORY.

**Current next engineering task:**  
Return `to_gpt/006_qualitative_pilot_parallel_prep/` for GPT review together with Task 005B. Do not execute the recommended next scientific action.

**Status:**  
Task 006 = AUTHORIZED / EXPLORATORY.  
Confirmatory study = NOT AUTHORIZED.

---

## 2026-09-18 — Task 007 full qualitative 100 + four-model pilot + paper-direction packet

**Date:** 2026-09-18

**Previous state:**  
Task 006 qualitative 20-q GPT/Claude pilot gated A-GO-FULL-QUALITATIVE. Tasks 005B/005C showed that much of the natural-verification residual is empirical item difficulty. `paperDirection.txt` was still unmodified.

**Evidence:**  
Numbered task `from_gpt/007_full_qualitative_four_model_decision_packet.md`.

**Decision:**

1. Lane A completes the remaining 80 Study-1 questions for GPT/Claude with the frozen Task-006 qualitative prompts (1,280 new cells). Task-006's original 320 cells are reused, not rerun. Merged grid = 1,600 exploratory cells. New sqlite `results/study1_qualitative_full80/qualitative_full80.sqlite3`.
2. Lane B runs Gemini/Grok on the frozen 20-question subset with the same prompts (up to 320 cells) if adapters/historical Stage-1 pass. Separate sqlite `results/study1_qualitative_gemini_grok20/qualitative_gemini_grok20.sqlite3`. No q2. No endpoint substitution.
3. Lane C tests whether displayed score merely shifts the verification budget or changes difficulty-sensitive prioritization. Cross-model other-correct is an evaluation proxy, not a production feature.
4. Lane D prepares, and does not run, a prospective four-model pipeline and D1/D2/D3 cost matrix on the Task-006 unseen pool.
5. Lane E writes a paper-direction decision packet. **Do not edit `paperDirection.txt` in this task.**
6. Do not launch a confirmatory study. Do not infer hidden-state suppression.

**Lane C buckets / spine:** GPT/Claude buckets and recommended Spine C (GPT SCORE_SHIFTS_BUDGET_PRESERVES_DIFFICULTY_PRIORITY; Claude SCORE_SHARPENS_DIFFICULTY_PRIORITY) are in `to_gpt/007_full_qualitative_four_model_decision_packet/`. Hold revision: no.

**Affected study already frozen?**  
Historical V2 remains HISTORICAL-FROZEN. Study 1 / Tasks 003–006 remain exploratory and are not overwritten.

**Exploratory vs confirmatory relative to this decision:**  
All Task 007 outputs are EXPLORATORY.

**Current next engineering task:**  
Return `to_gpt/007_full_qualitative_four_model_decision_packet/` for GPT review. Do not edit `paperDirection.txt` until GPT accepts the packet. Do not launch the confirmatory study.

**Status:**  
Task 007 = AUTHORIZED / EXPLORATORY.  
Confirmatory study = NOT AUTHORIZED.  
paperDirection.txt = UNMODIFIED.

