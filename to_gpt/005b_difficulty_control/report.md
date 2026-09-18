# Task 005B — Zero-Cost Difficulty-Control Audit

Exploratory analysis only. Not confirmatory. `paperDirection.txt`, historical V2, Study 1, and Task 005 data were not modified.

READY_FOR_GPT_REVIEW = YES

API calls made: **0**

---

## 1. Plain-English bottom line

After controlling for other-model difficulty, category, and question length, Claude's VERIFY/USE action adds little leftover correctness information.

Unadjusted Task-005-style ΔAUROC mean 0.161; difficulty-adjusted cell-level ΔAUROC mean 0.003 (L10 0.001 [-0.012, 0.014]; L20 0.006 [-0.009, 0.022]); question-level ΔAUROC 0.007 [-0.008, 0.026].

This is a leakage-safe *other-model* difficulty control (GPT/Gemini/Grok Stage-1 correctness). Four-model difficulty is reported descriptively only because it includes Claude.

---

## 2. Validation

- API calls: **0**
- Frozen Study-1 IDs: 100 (hash `badd6938e5ded12c9dd62733426e1db26d9843bb6a2321a4e4c9eb7e3547fe94`); 20-question repeats hash `45e06fe9daf40dd1cbd9cbf91b49cef09c5d924df17f2cf2bf2f203090c26d38`
- Claude manipulated cells analyzed: **1000** (10 displayed-score cells × 100 questions)
- Claude Stage-1 error rate on these 100: **27%** (27 wrong / 73 correct)
- Other-model correct count distribution (0/1/2/3 of GPT, Gemini, Grok): `{2: 19, 1: 11, 0: 7, 3: 63}`
- Categories present: biology, business, chemistry, computer science, economics, engineering, health, history, law, math, other, philosophy, physics, psychology
- Choice count unique values: [4, 7, 8, 9, 10]. Included in adjusted models because it varies. Dropped: **False**
- Option-text length omitted: Study-1 frozen answers are option letters.
- V2 sqlite sha256: `8597e3f73226bf39770af99a877282c22428203a3e2c11521256bf5cb3db5eb6` (unchanged)
- Study 1 sqlite sha256: `f8e5ddbb91db360403ea519b9b3481575c6cbc5ccdb553c650dbe726025ecfbd` (unchanged)
- paperDirection.txt sha256: `52cbde69e42f4db59f481d574a4f4395d1c7acfebb0b6f13a4390eda2c1f1d36` (unchanged)
- Task 005 report sha256: `a536d304a7d2581087fa1daba5e9ed92979f488f2348925142b79e28d8f06c69` (unchanged)
- Question-bootstrap seed `20260917`, resamples `5000`
- Statistical unit: question ID. Repeated L/score cells of the same question share a GroupKFold fold.
- Read-only sqlite (`file:?mode=ro`). No CheckpointStore on V2 or Study 1.

---

## 3. Raw Task-005 replication

Claude only, every L × manipulated displayed score. Question-bootstrap 95% CIs. These should match Task 005 Lane A `useful_item_sensitivity.csv`.

- L=10, displayed 0.8: n_wrong=27, n_correct=73, P(V|wrong)=96.3%, P(V|correct)=82.2%, Δ 14.1 pp [2.2, 25.0]
- L=10, displayed 0.88: n_wrong=27, n_correct=73, P(V|wrong)=88.9%, P(V|correct)=76.7%, Δ 12.2 pp [-3.9, 27.0]
- L=10, displayed 0.89: n_wrong=27, n_correct=73, P(V|wrong)=92.6%, P(V|correct)=74.0%, Δ 18.6 pp [3.8, 32.0]
- L=10, displayed 0.91: n_wrong=27, n_correct=73, P(V|wrong)=85.2%, P(V|correct)=67.1%, Δ 18.1 pp [-0.1, 34.6]
- L=10, displayed 0.99: n_wrong=27, n_correct=73, P(V|wrong)=81.5%, P(V|correct)=38.4%, Δ 43.1 pp [24.0, 60.1]
- L=20, displayed 0.9: n_wrong=27, n_correct=73, P(V|wrong)=100.0%, P(V|correct)=86.3%, Δ 13.7 pp [6.2, 22.1]
- L=20, displayed 0.93: n_wrong=27, n_correct=73, P(V|wrong)=92.6%, P(V|correct)=75.3%, Δ 17.3 pp [2.8, 30.7]
- L=20, displayed 0.94: n_wrong=27, n_correct=73, P(V|wrong)=88.9%, P(V|correct)=72.6%, Δ 16.3 pp [-0.2, 31.3]
- L=20, displayed 0.96: n_wrong=27, n_correct=73, P(V|wrong)=88.9%, P(V|correct)=63.0%, Δ 25.9 pp [9.5, 41.6]
- L=20, displayed 0.99: n_wrong=27, n_correct=73, P(V|wrong)=92.6%, P(V|correct)=50.7%, Δ 41.9 pp [26.2, 56.6]

Largest cells remain L=10 0.99 Δ **+43.1 pp** [24.0, 60.1] and L=20 0.99 Δ **+41.9 pp** [26.2, 56.6].

Unadjusted grouped-CV increment (displayed score vs score+action), same design as Task 005:

- L=10: ΔAUROC 0.160 [0.077, 0.245]; Δlog-loss -0.025 [-0.055, 0.008]
- L=20: ΔAUROC 0.162 [0.085, 0.244]; Δlog-loss -0.030 [-0.063, 0.005]

Task 005 reported Claude ΔAUROC about +0.16 with CIs excluding 0. This replication is the comparison baseline for the difficulty-adjusted models.

---

## 4. How strongly observable difficulty predicts Claude action

At a fixed displayed score, VERIFY questions are on average those that fewer of GPT/Gemini/Grok got right. That association is real but incomplete.

- L=10 0.99: VERIFY n=50 mean other-correct 2.08 vs USE n=50 mean 2.68 (VERIFY−USE -0.60 [-0.95, -0.25]). Error rate VERIFY 44.0% vs USE 10.0%.
- L=20 0.99: VERIFY n=62 mean other-correct 2.15 vs USE n=38 mean 2.76 (VERIFY−USE -0.62 [-0.94, -0.30]). Error rate VERIFY 40.3% vs USE 5.3%.

Full cell-by-cell mean/median other-model correct counts, question lengths, and category counts are in `difficulty_vs_action.csv`.

Difficulty itself predicts Claude correctness well. Adding other-model n_correct + category + length + choice count to the displayed-score stratum raises grouped-CV AUROC by 0.443 at L=10 and 0.443 at L=20. Question-level difficulty-only AUROC is 0.830.

So yes: Claude is more likely to VERIFY harder items, and harder items are more often wrong. That is exactly the confounder this task tests.

---

## 5. Whether Claude action adds correctness information after difficulty control

Primary models (grouped 5-fold CV by question ID):

1. displayed-score stratum + other-model n_correct + question length + choice count + category
2. those features + Claude VERIFY/USE

Cell-level (500 rows per L, 100 questions, 5 displayed scores):

- L=10 incremental action after difficulty: ΔAUROC 0.001 [-0.012, 0.014]; Δlog-loss 0.003 [-0.007, 0.013]
- L=20 incremental action after difficulty: ΔAUROC 0.006 [-0.009, 0.022]; Δlog-loss -0.001 [-0.015, 0.013]
- Pooled L: ΔAUROC 0.000 [-0.012, 0.013]

Question-level (one row per question; action = mean VERIFY propensity across the 10 manipulated cells):

- difficulty-only AUROC 0.830, log-loss 0.462
- difficulty + mean VERIFY propensity AUROC 0.837, log-loss 0.458
- incremental ΔAUROC 0.007 [-0.008, 0.026]; Δlog-loss -0.004 [-0.015, 0.008]

Stratified 2×2 within other-model-correct bins × displayed score: 22 estimable cells, 8 sparse cells left unestimated.

- other-models-correct bin 0-1: n=18 (13 wrong / 5 correct); mean VERIFY propensity wrong 0.808 vs correct 1.000 (Δ -0.192)
- other-models-correct bin 2: n=19 (8 wrong / 11 correct); mean VERIFY propensity wrong 1.000 vs correct 0.918 (Δ 0.082)
- other-models-correct bin 3: n=63 (6 wrong / 57 correct); mean VERIFY propensity wrong 1.000 vs correct 0.614 (Δ 0.386)

Nearest-neighbor secondary check (match each VERIFY to a USE on other-model correct count, category, length): 10 estimable displayed-score cells; VERIFY still has higher error than its matched USE in 10/10 of those cells. That leftover is compatible with imperfect matching; it is not the primary grouped-CV test. Details in `stratified_or_matched_analysis.csv`.

The stratified tables are not in conflict with a near-zero CV increment. Other-model difficulty already ranks Claude errors well (question-level AUROC ~0.84). Some easy-item 2×2 leftovers remain (especially the 3/3-other-correct bin), but they are small-N and do not add material grouped-CV ranking information once difficulty, category, length, and choice count are in the model.

---

## 6. Repeat-subset robustness

Frozen 20-question, 3-generation subset. Mean VERIFY propensity per question. Difficulty control = subtract the bin mean propensity within other-model n_correct.

- Raw wrong-minus-correct propensity gap positive in **10 / 10** manipulated cells.
- Difficulty-bin-adjusted gap positive in **10 / 10** cells.

n=20 is qualitative robustness only. It cannot settle the bucket by itself.

---

## 7. What this DOES establish

- The Task 005 Claude fixed-score 2×2 tables replicate on the same frozen 100 questions.
- Other-model empirical difficulty is associated with both Claude error and Claude VERIFY.
- After grouping by question and controlling for that difficulty plus category and stem length, we can say whether leftover action-correctness association remains in this pilot.
- Interpretation bucket: **DIFFICULTY_LARGELY_EXPLAINS_SIGNAL**.

---

## 8. What it DOES NOT establish

- It does **not** prove an internal hidden-state mechanism, even if difficulty does not absorb the signal.
- It does **not** make the routing signal useless if difficulty largely explains it; difficulty itself may be a practical routing cue.
- It does **not** replace fresh prospective confirmation.
- It does **not** use Claude's own correctness inside the difficulty feature.
- It does **not** analyze GPT here; GPT's fixed-score action is mostly saturated.
- Four-model (including Claude) difficulty is descriptive only and was not used in the primary adjusted models.

---

## 9. Interpretation bucket

**DIFFICULTY_LARGELY_EXPLAINS_SIGNAL**

Unadjusted Task-005-style ΔAUROC mean 0.161; difficulty-adjusted cell-level ΔAUROC mean 0.003 (L10 0.001 [-0.012, 0.014]; L20 0.006 [-0.009, 0.022]); question-level ΔAUROC 0.007 [-0.008, 0.026].

Even DIFFICULTY_DOES_NOT_ABSORB_SIGNAL does not prove an internal hidden-state mechanism.
Even DIFFICULTY_LARGELY_EXPLAINS_SIGNAL does not make the routing signal useless; difficulty itself may be a practical routing cue.
This task is exploratory and cannot replace fresh prospective confirmation.

---

## 10. Recommendation only — do not launch new experiments

Do not launch new paid runs from this task. Keep paperDirection.txt unmodified.

- Treat other-model item difficulty as the leading simpler explanation of the Task-005 Claude fixed-score VERIFY/wrong gap in this 100-question pilot.
- Keep any within-bin leftovers (especially among items all three other models got right) as a small-N exploratory residual, not as a new paid follow-up trigger.
- Sample-size / confirmatory decisions stay with GPT review of the D1 program. This audit does not launch experiments.

---

## 11. READY_FOR_GPT_REVIEW = YES

READY_FOR_GPT_REVIEW = YES

