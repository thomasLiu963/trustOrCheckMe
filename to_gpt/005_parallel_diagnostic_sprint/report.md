# Task 005 — Parallel Diagnostic Sprint After Study 1

Exploratory diagnostics only. Not confirmatory. `paperDirection.txt` was not modified.

READY_FOR_GPT_REVIEW = YES

---

## 1. Executive plain-English bottom line

Study 1's GPT cliff is real, but it mostly **pins the binary action to the displayed number**, so you usually cannot tell whether GPT still routes *wrong answers* to verification once the score is held fixed. In the few GPT cells that are not fully pinned, any wrong-vs-correct gap is small and noisy.

Claude is different. At a fixed displayed score Claude still verifies wrong answers more often than correct ones. That pilot gap is largest at 0.99, and on the frozen 20-question repeats it stays in the same direction when you average three generations.

The hidden verification bit from historical V2 is **not** just a second confidence sample. An independent Stage-2 re-elicitation `q2` on all 500 questions is highly correlated with `q1` (especially Claude) and barely improves error prediction beyond `q1`. Adding the historical hidden-VERIFY fraction still improves grouped-CV log-loss for both models after `q1+q2`. Claude's AUROC gain from hidden after `q1+q2` has a 95% CI that excludes zero. GPT's AUROC gain is positive but the 95% CI just includes zero; GPT's log-loss gain does not.

A GPT reasoning-mode grid was **not run**. The current adapter only supports `reasoning.effort=none`. Two qualitative-stakes prompt families are drafted for GPT review and were not run.

---

## 2. Exact API calls made

| lane | target | made |
|---|---:|---:|
| A existing-data analysis | 0 | **0** |
| B q2 re-elicitation | 1,000 | **1,000** scientific successes (500 GPT + 500 Claude) |
| C GPT reasoning mini-grid | 800 if supported | **0** (`BLOCKED_UNSUPPORTED_REASONING_SETTING`) |
| D qualitative-stakes prep | 0 | **0** |
| **Total scientific** | | **1,000** |
| Provider attempts (Lane B) | | **1,000** |
| Retries / parse repairs | | **0 / 0** |
| Failed scientific cells | | **0** |

New sqlite only: `results/study2_q2_diagnostic/q2.sqlite3`.  
Historical V2 and Study-1 sqlite hashes are unchanged.

---

## 3. Lane A: Does fixed-score item variation exist, and is any of it useful?

**GPT.** At most manipulated scores the action is effectively constant, so item discrimination is not identifiable. The only clearly mixed GPT cells are near/above the mechanical threshold:

- L=10, displayed 0.91: 4 VERIFY / 96 USE. `P(VERIFY|wrong)=10.5%` vs `P(VERIFY|correct)=2.5%`, Δ **+8.1 pp**, 95% [−3.8, +23.9]. Includes zero.
- L=20, displayed 0.96: Δ **+4.0 pp** [−3.6, +15.9]. Includes zero.
- L=20, displayed 0.99: 2 wrong VERIFY, 0 correct VERIFY. Δ **+10.5 pp** [0.0, +26.7]. Sparse.

That is a **pilot hint**, not a usable GPT error-targeting estimate.

**Claude.** All 10 manipulated cells have both actions in the sample. Wrong answers are more likely to be verified than correct ones at every grid point. The largest identifiable gaps are at 0.99:

- L=10, 0.99: `P(V|wrong)=81.5%` vs `P(V|correct)=38.4%`, Δ **+43.1 pp** [24.0, 60.1]
- L=20, 0.99: `P(V|wrong)=92.6%` vs `P(V|correct)=50.7%`, Δ **+41.9 pp** [26.2, 56.6]

On the frozen 20-question three-generation subset, Claude's wrong-minus-correct gap stays **positive in all 10 manipulated cells** when using the 3-generation mean. GPT's 20-question subset is too saturated for this check to move.

Grouped CV (question ID folds), manipulated cells only: adding the action to the experimentally assigned displayed score does **not** help GPT (ΔAUROC ~0.00–0.01, CI includes 0). For Claude it does: ΔAUROC about **+0.16**, 95% CIs exclude 0.

This is a 100-question **pilot signal check**, not a publication-grade evidence-retention estimate.

---

## 4. Lane A: What is structurally unidentifiable because of GPT saturation?

GPT manipulated cells labeled `ACTION_SATURATED__ITEM_DISCRIMINATION_NOT_IDENTIFIABLE`: **5 / 10** (L=10: 0.80, 0.88, 0.89, 0.99; L=20: 0.90). Those cells are pinned to one action.

That does **not** mean internal evidence was destroyed, suppressed, or absent. It means the binary Stage-3 action has no remaining variation from which to estimate wrong-vs-correct routing.

Hidden-vs-fixed-score agreement for GPT is mostly mechanical: below threshold everyone VERIFY (agreement with hidden ≈ hidden rate); above threshold everyone USE (agreement ≈ 1 − hidden rate). Cohen's kappa was **not** reported there because both actions are not supported in the fixed-score margin.

Claude kappa is reportable and rises toward high displayed scores (L=10 0.99 κ=0.78; L=20 0.96 κ=0.75): the same questions Claude hides-VERIFY tend to remain VERIFY even when shown 0.99.

---

## 5. Lane A: Existing matched-budget routing baseline

Historical V2-B, 500 questions, GPT and Claude, hidden cells. Ranking verifies lowest `q` first with Checkpoint A fractional tie handling. Calibration is 5-fold isotonic, grouped by question.

At the hidden policy's own verification count, **hidden catch > raw-q catch** in every GPT and Claude owner×L cell. Approximate hidden−raw gaps:

| model | owner | L | hidden catch | raw-q catch | hidden−raw |
|---|---|---:|---:|---:|---:|
| GPT | AI | 10 | 64.4% | 58.6% | +5.7 pp |
| GPT | AI | 20 | 66.7% | 62.1% | +4.6 pp |
| Claude | AI | 10 | 88.4% | 80.5% | +7.9 pp |
| Claude | AI | 20 | 93.0% | 85.8% | +7.2 pp |

Cross-fitted calibration does **not** eat this ranking gap (calibrated catch is similar to or slightly worse than raw). This is the baseline later q2 routers should beat. It is not a claim about hidden state.

---

## 6. Lane A: Power / precision implications

Empirical Stage-1 error rates on the Study-1 100: GPT **19%**, Claude **27%**.

A future **fixed-score** wrong-vs-correct verification contrast, treating question as the unit:

- At **N=100**, expected GPT wrong answers ≈ 19. A 10 pp delta has ~49 pp CI width and simulated power ≈ 0.13. A 30 pp (Claude-at-0.99-like) delta has power ≈ 0.70.
- At **N=400**, 10 pp power ≈ 0.36; 20 pp ≈ 0.90 (GPT error rate).
- At **N=1000**, 10 pp power ≈ 0.72; 15 pp ≈ 0.97.

This does **not** say “600–1000 is required.” It says a 5–10 pp GPT residual at a non-saturated score is still poorly estimated at N=100, while Claude-sized 20–40 pp gaps are already visible in the pilot. No sample size is frozen here.

---

## 7. Lane B: Does q2 explain the historical hidden-action residual?

**No, not as a complete explanation.**

Design: all 500 V2-B questions; frozen Stage-1 answers; exact historical Stage-2 prompt (`stage_2_confidence_v3_structured_compat`); q1 / correctness / hidden action / Study-1 scores not inserted. Prompt audit: 0 failures. Endpoints: `gpt-5.6-sol` (`reasoning.effort=none`), `claude-sonnet-5` (`thinking disabled`). 1000/1000 parsed on the first try.

Secondary:

| | GPT | Claude |
|---|---:|---:|
| corr(q1, q2) | 0.61 | 0.94 |
| mean \|q2−q1\| | 0.043 | 0.031 |
| raw q1 AUROC | 0.684 | 0.737 |
| raw q2 AUROC | 0.681 | 0.741 |
| mean q2 if hidden tends VERIFY | 0.81 | 0.64 |
| mean q2 if hidden tends USE | 0.97 | 0.86 |

Grouped 5-fold CV by question, logistic, n=500:

| features | GPT AUROC | GPT log-loss | Claude AUROC | Claude log-loss |
|---|---:|---:|---:|---:|
| q1 | 0.640 | 0.451 | 0.723 | 0.509 |
| q1+q2 | 0.641 | 0.452 | 0.730 | 0.501 |
| q1+hidden fraction | 0.735 | 0.407 | 0.773 | 0.471 |
| q1+q2+hidden fraction | 0.722 | 0.410 | 0.775 | 0.468 |

Hidden after q1+q2 (question bootstrap 95% CI):

- GPT: ΔAUROC **+0.081** [−0.002, +0.163]; Δlog-loss **−0.042** [−0.067, −0.019]
- Claude: ΔAUROC **+0.045** [+0.016, +0.074]; Δlog-loss **−0.032** [−0.055, −0.009]

Interpretation buckets (exploratory, not auto-scale):

- Claude: **B-PASS-DEEPER-RESIDUAL**
- GPT: **B-PASS-DEEPER-RESIDUAL**, with the explicit caveat that the AUROC interval barely includes 0 while log-loss does not. q2 alone does **not** absorb the hidden-action advantage (`B-PASS-SECOND-READ` is not supported).

Matched-budget ranking: sorting by historical hidden-VERIFY fraction still catches more GPT errors than ranking by q1, q2, or their mean at 10–50% budgets. Claude's hidden ranking is worse than q at very low budgets and better at 40–50%. Ties: fractional cutoff, same as Checkpoint A.

This is **not** a hidden-state mechanism result.

---

## 8. Lane C: Does higher reasoning change GPT score dominance or error targeting?

**Not answered. Lane C blocked.**

The current `gpt-5.6-sol` adapter hardcodes `"reasoning": {"effort": "none"}`. Config validation and tests refuse any other value. Task 005 forbids guessing a nonzero setting or switching model family. Zero Lane-C calls. See `lane_C_reasoning/BLOCKED.md`.

---

## 9. Lane D: Qualitative-stakes prompt candidates for GPT review

Drafts only. Not frozen. Not run.

- Moderate: `lane_D_qualitative_prep/prompt_candidate_A.md`
- Stronger: `lane_D_qualitative_prep/prompt_candidate_B.md`
- Audit / confounds / schema: `prompt_audit.md`
- Possible later size: 100 × 2 models × 2 stakes × 4 conditions = **1,600** calls (`dry_run_call_plan.md`)

Both families delete numerical L, C, and expected-value formulas. Manipulated conditions differ only in the displayed probability token. GPT should inspect residual qualitative-threshold language (“justified,” “outweighs”) before any freeze.

---

## 10. What the combined evidence DOES establish

- Study 1 GPT score dominance remains the right description of the binary action under explicit numerical stakes.
- At fixed displayed score, **Claude** shows a reproducible-looking pilot tendency to verify wrong answers more than correct ones; GPT mostly does not have enough action variation to say.
- Independent `q2` is a second confidence number. It does not explain away the historical hidden-action residual.
- Historical hidden VERIFY/USE still carries extra correctness information after q1+q2 in this exploratory 500-question analysis.
- A supported nonzero GPT reasoning setting does not currently exist in this repo's adapter.

---

## 11. What it still DOES NOT establish

- Confirmatory effect sizes or a paper-ready evidence-retention curve.
- That GPT “has no internal evidence” at saturated scores.
- A hidden-state / suppression mechanism.
- That the GPT cliff is *only* explicit arithmetic (qualitative-stakes experiment not run).
- That reasoning mode changes the cliff (Lane C blocked).
- Self-provenance, agent generalization, or other models.
- That we should scale N next.

---

## 12. Which deeper D1 story is currently supported

| story | current status |
|---|---|
| Deeper residual beyond q2 | **Supported as an exploratory result** for Claude; GPT supported on log-loss, AUROC CI just includes 0 |
| Repeated-assessment story | **Not sufficient.** q2 does not absorb hidden advantage |
| Arithmetic-specific score execution | **Plausible for the GPT cliff, untested.** Needs Lane D if GPT freezes a prompt |
| Reasoning-mode moderation | **Unknown / blocked** |
| Ambiguous / too weak | GPT *fixed-score item discrimination* is mostly unidentifiable; that part is structurally weak, not a failed measurement of a 10 pp residual |

---

## 13. Recommended next scientific action

**Recommendation only. Do not execute it in this task.**

1. GPT reviews and possibly rewrites the Lane D qualitative-stakes families, then (later numbered task) runs a small frozen experiment to see whether GPT's cliff survives without numerical L/C arithmetic.
2. Do **not** launch a confirmatory 600–1000 study from these diagnostics.
3. Do **not** invent a GPT reasoning setting. If reasoning remains interesting, GPT must name an already-supported parameter/value for this endpoint, or accept that Lane C stays blocked.
4. Optional later analysis, not new calls: treat q1+q2 as a practical router vs hidden ranking (Lane B tables already start that). Do not add q3.

---

## 14. Exact cost / runtime and retry accounting

| | |
|---|---|
| Lane B scientific calls | 1000 |
| Provider attempts | 1000 |
| Transient retries | 0 |
| Parse repairs | 0 |
| Input tokens | 431,808 (GPT 141,982; Claude 289,826) |
| Output tokens | 15,002 (GPT 8,500; Claude 6,502) |
| GPT estimated USD | $0.738 |
| Claude estimated USD | $0.645 |
| **Total estimated USD** | **$1.383** |
| Lane B wall-clock | 419 s |
| Lane A wall-clock | ~75–80 s (local) |
| Returned model IDs | `gpt-5.6-sol` × 500, `claude-sonnet-5` × 500 |
| Historical V2 sha256 after | `8597e3f73226bf39770af99a877282c22428203a3e2c11521256bf5cb3db5eb6` (unchanged) |
| Study 1 sha256 after | `f8e5ddbb91db360403ea519b9b3481575c6cbc5ccdb553c650dbe726025ecfbd` (unchanged) |
| paperDirection.txt sha256 | `52cbde69e42f4db59f481d574a4f4395d1c7acfebb0b6f13a4390eda2c1f1d36` (unchanged) |
| Code commit at q2 run | `cfdb0917a55bb164e5116c400b2046e610ad3900` |

---

## 15. READY_FOR_GPT_REVIEW = YES
