# Task 009 preregistration — prospective ranking-invariance confirmation

Frozen at `2026-09-18T12:58:03.563552+00:00` before scientific VERIFY-rate inspection.
Commit: `9d377506df3cd2890071e22aa94e5460cb6d91d4`.
Experiment version: `study009_prospective_v1`.
Label: **confirmatory**.

This file must not be rewritten after freeze except via a timestamped note in `deviations.md`.

## Question

On the exact same frozen answer, when displayed confidence is counterfactually changed, does the model mostly change **how many** answers it sends for independent verification, or **which** answers it prioritizes?

## Sample

- Dataset: TIGER-Lab/MMLU-Pro test, revision `b189ec765aa7ed75c8acfea42df31fdae71f97be`.
- Eligible unseen pool SHA256: `53b11137658de739537a41032d92aada7f50742ef07e9615da10e070ab37c4fa` (n=11532).
- Historical V2-B 500 IDs are excluded. Tasks 003–008 development items are inside that 500.
- Primary N=500, seed `20260918`, SHA256 `4cc2f1dce283499fe40049b73900fb7862d7e153d76d8b75590109c0ce12d99f`.
- Secondary N=200 Gemini/Grok subset of those 500, seed `20260918`, SHA256 `9ffd1ca537d6f05e06a1c88e296a147dbede7a9382d052b6a60ea1156b5b441d`.
  This is **not** `candidate_N200.json`.
- Repeat N=100 GPT/Claude subset of the 500, seed `20260919`, SHA256 `61ca0b1e40931ff6d97dc111bdc2439c3b2dbc25a98e533cb209141bb6a66b9a`.
- Sampler: category-stratified `select_category_stratified`.
- Same questions across models within each roster.

## Models

Primary:
- GPT `gpt-5.6-sol`
- Claude `claude-sonnet-5`

Secondary:
- Gemini `gemini-3.8-flash`
- Grok `grok-4.20-0309-non-reasoning`

If an endpoint is unavailable: mark BLOCKED. Do not substitute.

## Protocol

- Fresh Stage-1 answers, then separate q1 confidence. No q2.
- Stage-3: Task-006/007 **moderate** qualitative wording only.
- Prompt version `task006_qualitative_pilot_v1`, family `qualitative_stakes_moderate_v1`, template hash `ec90530c644038dcee86accfc578c501533fdb69d5ea23caeae23dcc1d06dbad`.
- Conditions: hidden, true_q_visible, displayed_0.70, displayed_0.85, displayed_0.90, displayed_0.95, displayed_0.99.
- hidden = no confidence sentence; true_q_visible = that item's fresh q1; displayed_* = exact fixed score, formatted with `.12g`.
- Actions: USE_UNVERIFIED or VERIFY_FIRST only.
- Repeats: GPT/Claude, N=100, 3 total Stage-3 generations per condition; original primary cell is generation 0; two extras. Do not repeat Stage-1/q1.

## Call budget

- GPT+Claude Stage1+q1: 2000
- GPT+Claude Stage3: 7000
- Repeat extras: 2800
- Primary total: 11800
- Gemini+Grok Stage1+q1: 800
- Gemini+Grok Stage3: 2800
- Secondary total: 3600
- Grand scientific cap: 15400
- Provider attempt cap: 20020 (retries counted separately)
- Concurrency: 8

## Hypotheses and criteria

### H1 coverage control

Primary effect: VERIFY_rate(0.70) − VERIFY_rate(0.99), paired by question, question-bootstrap 95% CI.
Substantive floors (large-effect markers, not publication gates): GPT ≥ 20pp, Claude ≥ 15pp.

### H2 shared-ranking invariance

Shared transferable model: condition intercepts + leave-one-target-out other-primary Stage-1 correctness.
Richer model: shared + displayed-score × difficulty interaction.
Primary metric: 5-fold GroupKFold held-out log-loss improvement of richer vs shared.
Invariance if improvement < 0.01 **and** question-bootstrap 95% upper bound of that improvement is also < 0.01.
Evaluate GPT and Claude separately.
If fits are unstable, report INCONCLUSIVE rather than changing the threshold.
Item intercepts are in-sample descriptive only; they do not transfer to held-out questions.

### H3 repeat rank stability

On N=100, 3-generation mean VERIFY propensity for adjacent visible pairs (.70/.85, .85/.90, .90/.95, .95/.99):
Spearman, bootstrap CI, pairwise ordering agreement, reversals vs same-prompt rerun noise.

### H4 discrimination stability

AUROC of repeat-averaged VERIFY propensity for target wrongness, with pairwise ΔAUROC.
Full-N binary-action AUROC is coarse secondary evidence.
ΔAUROC is evaluated only on non-saturated cells.

## Saturation

A condition-cell is saturated if VERIFY rate ≤ 0.05 or ≥ 0.95.
Flag: `ITEM_ROUTING_NOT_IDENTIFIABLE_DUE_TO_ACTION_SATURATION`. Saturated cells are excluded from ranking/discrimination identity claims.

## Retry / exclusion

- Transient transport retries: 3
- Parse repairs: 1
- One additional retry_failed pass after the first paid pass
- After that, missing cells are failures, not imputed
- If > 5% of primary GPT/Claude planned scientific cells fail: INCONCLUSIVE

## Bootstrap

Seed 20260918, 5000 resamples, unit = question ID.

## Decision buckets (frozen)

- `PROSPECTIVE_INVARIANCE_SUPPORTED` if coverage response is large, shared-ranking equivalence-style test passes, repeat rank stability is strong, discrimination is stable on non-saturated conditions, and the richer model adds no material held-out value.
- `PROSPECTIVE_RESHAPING_SUPPORTED` if coverage changes and the richer model materially/reproducibly beats shared ranking, repeats show score-dependent reordering beyond rerun noise, and discrimination changes materially at non-saturated points.
- `MIXED_BY_MODEL` if GPT and Claude genuinely diverge under the same tests.
- `INCONCLUSIVE` if saturation, low error counts, or uncertainty prevent distinction.

Do not redefine these after seeing results.

## Secondary Gemini/Grok

Same Stage-1/q1 + 7 Stage-3 conditions on the frozen N=200. No repeats. Compatibility report only. Secondary oddities do not redefine the GPT/Claude hypothesis.

## Isolation

- New sqlite: `/Users/thomas/Desktop/trustOrCheckMe/results/study009_prospective/prospective.sqlite3`
- Outputs: `/Users/thomas/Desktop/trustOrCheckMe/to_gpt/009_prospective_ranking_invariance_confirmation`
- Do not modify `/Users/thomas/Desktop/trustOrCheckMe/paperDirection.txt`
- Do not inspect scientific VERIFY rates until all primary GPT/Claude cells complete
- Do not launch a second benchmark inside this task
