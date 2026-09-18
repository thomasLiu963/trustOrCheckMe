# Task 006 — Qualitative-Stakes Pilot + Parallel Paper-Critical Prep

Exploratory only. Not confirmatory. `paperDirection.txt` was not modified.

READY_FOR_GPT_REVIEW = YES

---

## 1. Plain-English executive bottom line

Qualitative 20-question pilot gate: **A-GO-FULL-QUALITATIVE**. After numerical L/C arithmetic was removed, changing only the displayed confidence still moved VERIFY/USE, and the movement was large.

GPT moderate VERIFY rates: 70% at 0.70, 20% at 0.90, 5% at 0.99 (0.70→0.99: **−65 pp**, 13 of 20 questions decreased, 0 increased). GPT stronger: 95% / 35% / 20% (−75 pp). Claude also declined (95% / 75% / 65%) but did not separate moderate from stronger. Stronger wording increased GPT verification by about **+15 to +25 pp** at the same displayed score and did not pin every cell to VERIFY.

This is a 20-question pilot, not a confirmatory effect size. It does show that the Study-1 cliff is not *only* “the prompt handed GPT the arithmetic,” at least under this frozen qualitative wording.

Historical matched-budget routing on the same 500 V2-B questions still beats raw q1 most clearly for GPT’s multi-cell hidden aggregate (+8.3 pp error-catch at a 30% budget). A single predeclared hidden cell and q2 add only a few points for GPT; Claude’s single L=10 hidden cell is worse than raw q1. That remains exploratory.

A fresh unseen MMLU-Pro pool of 11,532 questions is prepared with candidate N=200…1000 lists. Gemini/Grok adapters exist; open-weight inference does not.

## 2. Exact paid scientific calls made

- Target: **320**
- Made: **320**
- Provider attempts: 320
- Failures: 0
- Lanes B/C/D API calls: **0**
- New sqlite: `/Users/thomas/Desktop/trustOrCheckMe/results/study1_qualitative_pilot/qualitative_pilot.sqlite3`

## 3. Prompt-freeze validation

- Frozen 20 IDs hash: `45e06fe9daf40dd1cbd9cbf91b49cef09c5d924df17f2cf2bf2f203090c26d38`
- GPT `gpt-5.6-sol` reasoning.effort=none; Claude `claude-sonnet-5` thinking=disabled
- Moderate family hash: `ec90530c644038dcee86accfc578c501533fdb69d5ea23caeae23dcc1d06dbad`
- Stronger family hash: `b758935fecdd49ad7a6a0a491d9c71abee35f90981a7030d90ac59a18dca9f3b`
- Visible tokens: `{'0.70': '0.7', '0.90': '0.9', '0.99': '0.99'}`
- Preflight ok: **True**; audit errors: 0
- Task-005 Lane D drafts were **not** used (they contained forbidden 'justified' / 'not as a default' language).

## 4. GPT qualitative score-response result

- Moderate: hidden 40.0% (8/20); displayed_0.70 70.0% (14/20); displayed_0.90 20.0% (4/20); displayed_0.99 5.0% (1/20)
- Stronger: hidden 65.0% (13/20); displayed_0.70 95.0% (19/20); displayed_0.90 35.0% (7/20); displayed_0.99 20.0% (4/20)
- Paired deltas (pilot): 0.7→0.9 -50.0 pp (dec 10/inc 0/same 10); 0.9→0.99 -15.0 pp (dec 3/inc 0/same 17); 0.7→0.99 -65.0 pp (dec 13/inc 0/same 7)
- Stronger paired deltas: 0.7→0.9 -60.0 pp (dec 12/inc 0/same 8); 0.9→0.99 -15.0 pp (dec 4/inc 1/same 15); 0.7→0.99 -75.0 pp (dec 15/inc 0/same 5)

## 5. Claude qualitative score-response result

- Moderate: hidden 65.0% (13/20); displayed_0.70 95.0% (19/20); displayed_0.90 75.0% (15/20); displayed_0.99 65.0% (13/20)
- Stronger: hidden 70.0% (14/20); displayed_0.70 95.0% (19/20); displayed_0.90 75.0% (15/20); displayed_0.99 65.0% (13/20)
- Paired deltas (pilot): 0.7→0.9 -20.0 pp (dec 4/inc 0/same 16); 0.9→0.99 -10.0 pp (dec 2/inc 0/same 18); 0.7→0.99 -30.0 pp (dec 6/inc 0/same 14)

## 6. Stronger-vs-moderate stakes manipulation check

- GPT displayed_0.70: stronger−moderate +25.0 pp (stronger-only 5, moderate-only 0)
- GPT displayed_0.90: stronger−moderate +15.0 pp (stronger-only 4, moderate-only 1)
- GPT displayed_0.99: stronger−moderate +15.0 pp (stronger-only 3, moderate-only 0)
- Claude displayed_0.70: stronger−moderate +0.0 pp (stronger-only 0, moderate-only 0)
- Claude displayed_0.90: stronger−moderate +0.0 pp (stronger-only 0, moderate-only 0)
- Claude displayed_0.99: stronger−moderate +0.0 pp (stronger-only 0, moderate-only 0)

- GPT visible-cell mean stakes effect: +18.3 pp; labeled active=True

## 7. Saturation / measurability result

- GPT visible cells in the provisional 10–90% VERIFY band: 4 / 6
- Full cell table: `lane_A_qualitative_pilot/saturation_diagnostic.csv`

## 8. Pilot wrong-vs-correct routing where identifiable

- Identifiable (both actions present) visible cells: 12
- Saturated cells labeled `ITEM_ROUTING_NOT_IDENTIFIABLE_DUE_TO_ACTION_SATURATION`.
- n=20; do not overinterpret.

## 9. Lane-A gate

**A-GO-FULL-QUALITATIVE**

Qualitative displayed score systematically changes GPT verification without a numerical threshold.

## 10. Exploratory routing result

Historical V2-B, GPT and Claude, grouped-CV where learned. Primary metric: error catch at 10–50% verification budgets.

- **GPT** catch@30% raw q1=0.5287356321839081; q2=0.5547667342799188; single hidden AI/L10=0.5454899668809662; hidden aggregate=0.6118999323867478 (not deployment-fair); combined q1+q2+single-hidden=0.5300127713920818
- **Claude** catch@30% raw q1=0.528324388789505; q2=0.5390578413834227; single hidden AI/L10=0.4539662312838484; hidden aggregate=0.5285412262156448 (not deployment-fair); combined q1+q2+single-hidden=0.5271317829457365

Mandatory-looking baselines for a later fresh study, if GPT agrees: raw q1, calibrated q1, q1+q2, and one hidden verification judgment. Do not freeze the router from this exploratory table.

## 11. Fresh unseen candidate-pool readiness

- Source MMLU-Pro test revision `b189ec765aa7ed75c8acfea42df31fdae71f97be`: 12032 items
- Eligible after excluding 500 V2-B IDs: **11532**
- Pool hash: `53b11137658de739537a41032d92aada7f50742ef07e9615da10e070ab37c4fa`
- Candidate stratified lists: N=200,400,600,800,1000 with predeclared seeds. Not the confirmatory sample.

## 12. Gemini/Grok/open-model feasibility

- Gemini: **READY** (`gemini-3.8-flash`); historical V2 Stage-1/2 present; no Study 1/q2 replication
- Grok: **READY** (`grok-4.20-0309-non-reasoning`); historical V2 Stage-1/2 present; no Study 1/q2 replication
- Open-weight: **BLOCKED** (no inference adapter in `src/`)

## 13. What the combined Task-006 evidence DOES establish

- Whether a non-numerical qualitative verification prompt still produces systematic displayed-score responsiveness in this 20-question GPT/Claude pilot.
- Whether stronger vs moderate consequence wording is behaviorally active at n=20.
- Whether GPT binary actions become measurable (non-saturated) without L/C arithmetic.
- That historical matched-budget routing still looks like a paper-critical payoff worth prospective confirmation, with resource labels that prevent treating the hidden aggregate as a one-call router.
- That a leakage-safe unseen MMLU-Pro pool and candidate N plans exist without peeking at new model outcomes.
- That Gemini/Grok closed-model replication is adapter-ready; open-weight is not.

## 14. What it DOES NOT establish

- A confirmatory effect size, a paper-ready qualitative cliff, or a frozen confirmatory sample.
- That GPT has 'zero internal evidence' if qualitative actions remain saturated.
- A hidden-state / suppression mechanism.
- That q2 or hidden should be the production router.
- Gemini, Grok, or open-model qualitative behavior (not run).
- Task 005B difficulty-control conclusions (separate packet).

## 15. Recommended next action(s), recommendation only

Do not execute these in Task 006.

1. GPT reviews Task 005B and Task 006 together before any scale-up.
2. If Lane A is A-GO-FULL-QUALITATIVE or A-GO-NARROW, freeze a larger qualitative experiment in a later numbered task using only the useful family/conditions.
3. If A-FAIL-QUALITATIVE or A-REVISE-PROMPT, do not scale this wording; GPT should rewrite once.
4. Do not launch a confirmatory 600–1000 study from these diagnostics.
5. Do not run Gemini/Grok/open models until GPT authorizes a numbered replication task.
6. Keep `paperDirection.txt` unmodified.

## 16. Exact cost / runtime / retries

- Lane A scientific calls: 320
- Lane A provider attempts: 320
- Lane A wall-clock seconds: 119.24362979200669
- Lane A estimated USD: 0.564264
- Lanes B/C/D: 0 calls
- V2 sha256: `8597e3f73226bf39770af99a877282c22428203a3e2c11521256bf5cb3db5eb6`
- Study 1 sha256: `f8e5ddbb91db360403ea519b9b3481575c6cbc5ccdb553c650dbe726025ecfbd`
- paperDirection sha256: `52cbde69e42f4db59f481d574a4f4395d1c7acfebb0b6f13a4390eda2c1f1d36`

## 17. READY_FOR_GPT_REVIEW

READY_FOR_GPT_REVIEW = YES
