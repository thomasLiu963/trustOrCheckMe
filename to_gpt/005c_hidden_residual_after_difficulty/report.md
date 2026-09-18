# Task 005C — Hidden-Action Residual After Difficulty Control

Exploratory analysis only. Not confirmatory. `paperDirection.txt`, historical V2, Study 1, Task 005 q2 data, and Task 005B data were not modified.

READY_FOR_GPT_REVIEW = YES

API calls made: **0**

---

## 1. Plain-English bottom line

GPT bucket **HIDDEN_PARTIALLY_ADDS**. Claude bucket **DIFFICULTY_LARGELY_EXPLAINS_HIDDEN_RESIDUAL**. The analysis asks whether historical hidden VERIFY still helps after q1, an independent q2, and leakage-safe other-model difficulty.

- GPT Contrast B (hidden after q1+q2+difficulty): ΔAUROC 0.015 [0.001, 0.031]; Δlog-loss -0.005 [-0.015, 0.004]
- Claude Contrast B: ΔAUROC 0.005 [-0.008, 0.017]; Δlog-loss -0.008 [-0.020, 0.005]

Difficulty is leakage-safe: GPT is scored with Claude/Gemini/Grok correctness; Claude with GPT/Gemini/Grok. The target model's own Stage-1 correctness is never inside its difficulty feature.

---

## 2. Validation / leakage audit

- Joins: **500 GPT + 500 Claude** question-level rows on the 500 V2-B IDs
- Hidden summary rebuilt from V2 primary hidden cells (8 cells/question = 2 owners × 4 L) and matched Task 005 `q2_results.csv`
- Canonical single hidden action predeclared as `ai_system`, L=10, hidden
- Validation OK: **True**
- V2 sha256 `8597e3f73226bf39770af99a877282c22428203a3e2c11521256bf5cb3db5eb6`
- Study 1 sha256 `f8e5ddbb91db360403ea519b9b3481575c6cbc5ccdb553c650dbe726025ecfbd`
- q2 sqlite sha256 `57be401c4f069d5541f3b785e3c51448e49e915727eef178062196ea95272e6a`
- paperDirection sha256 `52cbde69e42f4db59f481d574a4f4395d1c7acfebb0b6f13a4390eda2c1f1d36`
- Task 005 report sha256 `a536d304a7d2581087fa1daba5e9ed92979f488f2348925142b79e28d8f06c69`
- Task 005B report sha256 `356fd4fc35aaac473fcc5551eb3d3cd6dc517273cf62851ca7de68d14b41daf2`
- Grouped 5-fold CV by question ID. One-hot category and length/choice-count scaling fit on training folds only.
- See `validation.md`.

---

## 3. Replication of Task-005 q2 result

Contrast A is `q1+q2+hidden_fraction` vs `q1+q2`, same construction as Task 005 Lane B.

- GPT: ΔAUROC 0.081 [0.003, 0.163]; Δlog-loss -0.042 [-0.067, -0.019]
- Claude: ΔAUROC 0.045 [0.016, 0.075]; Δlog-loss -0.032 [-0.055, -0.009]

Task 005 reported GPT ΔAUROC +0.081 [−0.002, +0.163] / Δlog-loss −0.042, and Claude ΔAUROC +0.045 [+0.016, +0.074] / Δlog-loss −0.032. Contrast A point estimates match those nested models exactly. Bootstrap percentile intervals can differ slightly because this audit reseeds each contrast/metric instead of advancing one RNG across sequential metrics.

- GPT q1 AUROC 0.640, q1+q2 0.641, q1+q2+hidden 0.722
- Claude q1 AUROC 0.723, q1+q2 0.730, q1+q2+hidden 0.775

---

## 4. Strength of observable difficulty

- GPT: other-model n_correct AUROC for GPT correctness 0.871; Pearson(hidden fraction, other-frac-wrong) 0.508
- Claude: other-model n_correct AUROC for Claude correctness 0.809; Pearson(hidden fraction, other-frac-wrong) 0.394
- Adding difficulty after q1+q2: GPT ΔAUROC 0.216 [0.134, 0.297]; Claude ΔAUROC 0.119 [0.071, 0.168]
- Difficulty-only OOF AUROC: GPT 0.864; Claude 0.818

Bin means are in `difficulty_associations.csv`. Hidden VERIFY propensity rises as fewer of the other three models are correct. That is the Task-005B-style confounder, now on all 500 V2-B items.

---

## 5. PRIMARY: does hidden action add after q1+q2+difficulty?

Contrast B: `q1+q2+difficulty+hidden_fraction` vs `q1+q2+difficulty`.

- GPT: ΔAUROC 0.015 [0.001, 0.031]; Δlog-loss -0.005 [-0.015, 0.004]; ΔBrier -0.000 [-0.003, 0.003]
- Claude: ΔAUROC 0.005 [-0.008, 0.017]; Δlog-loss -0.008 [-0.020, 0.005]; ΔBrier 0.001 [-0.004, 0.006]

Single canonical hidden action (AI-system, L=10, one cell) after q1+q2+difficulty:

- GPT: ΔAUROC 0.006 [-0.003, 0.016]
- Claude: ΔAUROC 0.003 [-0.008, 0.013]

The 8-cell fraction is a multi-elicitation diagnostic / upper bound. The single-cell sensitivity is closer to a one-call router.

- GPT: Contrast A ΔAUROC 0.081 [0.003, 0.163]; Contrast B ΔAUROC 0.015 [0.001, 0.031], Δlog-loss -0.005 [-0.015, 0.004].
- Claude: Contrast A ΔAUROC 0.045 [0.016, 0.075]; Contrast B ΔAUROC 0.005 [-0.008, 0.017], Δlog-loss -0.008 [-0.020, 0.005].

---

## 6. Matched-budget routing after difficulty control

OOF predicted P(correct); lowest predicted correctness verified first; Checkpoint A fractional ties. Primary comparison is +hidden vs q1+q2+difficulty.

### GPT

- 10%: q1+q2+difficulty catch 0.379 (33.0 of 87); +hidden 0.391 (Δ 0.011 [-0.051, 0.043])
- 20%: q1+q2+difficulty catch 0.644 (56.0 of 87); +hidden 0.655 (Δ 0.011 [-0.025, 0.060])
- 30%: q1+q2+difficulty catch 0.770 (67.0 of 87); +hidden 0.793 (Δ 0.023 [-0.038, 0.054])
- 40%: q1+q2+difficulty catch 0.874 (76.0 of 87); +hidden 0.874 (Δ 0.000 [-0.057, 0.047])
- 50%: q1+q2+difficulty catch 0.920 (80.0 of 87); +hidden 0.943 (Δ 0.023 [-0.012, 0.076])

### Claude

- 10%: q1+q2+difficulty catch 0.287 (37.0 of 129); +hidden 0.264 (Δ -0.023 [-0.046, 0.023])
- 20%: q1+q2+difficulty catch 0.512 (66.0 of 129); +hidden 0.496 (Δ -0.016 [-0.047, 0.025])
- 30%: q1+q2+difficulty catch 0.698 (90.0 of 129); +hidden 0.667 (Δ -0.031 [-0.056, 0.017])
- 40%: q1+q2+difficulty catch 0.806 (104.0 of 129); +hidden 0.806 (Δ 0.000 [-0.053, 0.033])
- 50%: q1+q2+difficulty catch 0.884 (114.0 of 129); +hidden 0.884 (Δ 0.000 [-0.035, 0.048])

Full 10–50% tables, including the single-cell hidden router, are in `matched_budget_routing.csv`.

---

## 7. GPT interpretation bucket

**HIDDEN_PARTIALLY_ADDS**

Contrast A ΔAUROC 0.081 [0.003, 0.163]; Contrast B ΔAUROC 0.015 [0.001, 0.031], Δlog-loss -0.005 [-0.015, 0.004].

---

## 8. Claude interpretation bucket

**DIFFICULTY_LARGELY_EXPLAINS_HIDDEN_RESIDUAL**

Contrast A ΔAUROC 0.045 [0.016, 0.075]; Contrast B ΔAUROC 0.005 [-0.008, 0.017], Δlog-loss -0.008 [-0.020, 0.005].

---

## 9. What this DOES establish

- Whether Task 005's hidden-after-q1+q2 residual survives a leakage-safe other-model difficulty control on the same 500 questions.
- How strongly other-model difficulty predicts target correctness and hidden VERIFY propensity.
- Whether a one-cell canonical hidden action behaves like the 8-cell fraction.
- Model-specific exploratory buckets for GPT review.

---

## 10. What it DOES NOT establish

- It does **not** establish an internal hidden-state mechanism.
- If difficulty explains the signal, that does **not** make verification useless; it may be a cheap item-difficulty estimator.
- If hidden remains useful beyond difficulty, that still does **not** prove a hidden internal state; it is a behavioral residual.
- The 8-cell hidden fraction is **not** automatically a fair one-call production router.
- This is not confirmatory and does not rewrite paperDirection.txt.

---

## 11. Recommendation only — do not launch paid experiments

- Do not launch paid experiments from this audit.
- Keep paperDirection.txt unmodified.
- Treat the 8-cell hidden fraction as a multi-elicitation diagnostic / upper bound, not a one-call production router.
- If GPT review agrees, keep a deeper residual as an exploratory D1 finding for the model(s) where Contrast B remains material, still not as a hidden-state proof.

---

## 12. READY_FOR_GPT_REVIEW = YES

READY_FOR_GPT_REVIEW = YES

