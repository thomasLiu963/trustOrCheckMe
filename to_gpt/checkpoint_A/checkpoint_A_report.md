# Checkpoint A Results

Exploratory re-analysis of the frozen V2-B experiment. No new model calls. No edits to raw experiment records. This is a scientific checkpoint, not a paper draft.

**Tie handling used throughout Part 1.** Reported confidence is very discrete (GPT puts 247/500 answers at exactly `q = 0.99`). The confidence-only policy ranks answers from lowest `q` to highest `q` and verifies the same *expected* number of answers as the hidden policy. All answers with `q` strictly below the cutoff are verified. Answers tied at the cutoff share equal fractional weight so that the expected number verified equals the hidden policy’s integer verification count. Error-catch rates are therefore expected rates under random inclusion of the cutoff tie group. This matches budget exactly even when many items share one confidence value.

**Uncertainty.** Question-level bootstrap, 5,000 resamples, percentile 95% intervals, seed `20260904`. The hidden policy’s verification count is recomputed inside each resample. Repeated `L` / authority cells that share the same questions are *averaged inside each replicate*; they are not treated as 32 independent samples.

---

## 1. Data validation

**Canonical primary dataset.** V2-B, 500 MMLU-Pro questions, prompt family `v2_owner_match_v1`.

| Item | Value |
|---|---|
| Raw checkpoint | `results/v2/raw/v2.sqlite3` (194 MB) |
| Question list | `data/samples/mmlu_pro_v2b.jsonl` (500 IDs; exact match to Stage-1 IDs) |
| Existing analysis pointer | `paper_outputs/v2/analysis_manifest.json` |
| Models | 4: `anthropic_sonnet5`, `google_gemini38_flash`, `openai_gpt56_sol`, `xai_grok420_nonreasoning` |
| Unique questions / model | 500 / 500 / 500 / 500 |
| Error costs `L` | 2, 5, 10, 20 |
| Authority | `human`, `ai_system` |
| Confidence visibility | `hidden`, `visible` |
| Verification cost `C` | 1 in every primary row |
| Primary Stage-3 rows | **32,000** unique cells |
| Expected grid | 500 × 4 models × 2 owners × 2 visibilities × 4 `L` = 32,000 |

**Stage 1 / Stage 2.** 2,000 answer records and 2,000 confidence records. Unique on `(example_id, model_id)`. Zero duplicate keys. Zero value conflicts.

**Factor completeness.** 2,000 `(question, model)` groups, each with all 16 Stage-3 cells. Completeness-issue count: **0**. Duplicate primary cells: **0**. Refusals: **0**. Join failures to Stage 1 or Stage 2: **0**.

**Frozen-field consistency.** Stage-3 `frozen_answer_label` matches Stage 1 on every primary row. Stage-3 `probability_correct` matches Stage 2 on every primary row. Across the 16 Stage-3 cells of each `(question, model)` pair, frozen answer and frozen confidence are constant.

**Intentionally excluded.** 6,400 paraphrase-robustness Stage-3 rows (`v2_owner_match_paraphrase_v1`) are in the same SQLite file. They are **not** mixed into Checkpoint A.

**Bookkeeping note, not a data defect.** Primary Stage-3 `run_id` counts are `v2b-core-20260905` = 19,201, `v2a-core-20260904` = 12,735, `v2a-smoke-20260904` = 64 (sum 32,000). V2-B reuses the overlapping 200-question V2-A cells. One leftover V2-A cell was completed under the V2-B run id (19,201 rather than 19,200). The scientific cell grid is complete and unique.

**Ambiguities documented rather than repaired.**

1. Confidence values are discrete and heavily tied. Analysis uses fractional cutoff inclusion rather than silently breaking ties.
2. Grok’s hidden policy verifies ~96% of answers, so an equal-budget comparison sits at a ceiling. That is reported as a ceiling, not as “confidence ranking is deeply better.”
3. Gemini has only 60 wrong answers, so catch-rate intervals are wide.
4. Owner-pooled cost summaries average the two owners’ costs per question. They are labeled `pooled_owners` in the calibration CSV and are not a third experimental condition.

Data are complete enough for the analyses below. Nothing was imputed.

---

## 2. Bottom line in plain English

- **The boring “they just check more” story is partly true for GPT, and it is the main thing that happens when confidence is *shown*.** Showing GPT its number makes it verify less, and the visible policy agrees with the raw-confidence threshold on **99.2%** of answers. Visible GPT points sit *on* the confidence-ranking curve. They are not a smarter use of the same budget; they are a smaller budget.
- **Hidden GPT is only weakly better at choosing *which* answers to check, once budget is matched.** All 8 GPT cells have a positive point estimate (hidden catch > confidence-only catch), mean **+7.1 percentage points**, but the question-level 95% CI is **[−2.4, +16.4]** and includes zero. Only 1 of 8 GPT cells has a CI that excludes zero. Do not build a paper on “GPT’s hidden judgment ranks errors better than the number.”
- **Claude is the one model where hidden checking clearly beats confidence ranking at the same budget.** Mean catch-rate gap **+9.6 pp, 95% CI [3.2, 16.0]**. All 8 Claude cells are positive and all 8 CIs exclude zero. On the risk/coverage plot, Claude’s hidden points sit above the confidence curve.
- **Gemini is inconclusive; Grok is a ceiling.** Gemini: **+2.8 pp [−3.4, +9.6]**. Grok verifies almost everything hidden, so confidence-only at that budget catches 100% of errors and hidden is trivially *slightly worse* (**−0.6 pp [−1.1, −0.2]**).
- **Calibration explains a large part of GPT’s *cost* disaster at high stakes, but it is a different issue from ranking.** GPT’s number is badly overconfident (10-bin ECE **0.136**; wrong answers still have mean `q = 0.86`). Cross-fitted isotonic calibration drops ECE to **0.033**. A calibrated threshold at `L = 10` or `20` verifies ~96% of GPT answers and has much lower realized cost than visible/raw GPT. Isotonic is monotone, so it **cannot** create Claude’s equal-budget ranking advantage. Discrimination (AUROC) is weakest for GPT (**0.68**) and is not improved by calibration.
- **The hidden check/no-check bit does contain extra correctness information beyond the reported number, for GPT and Claude.** Out-of-sample logistic log-loss improves in 7/8 GPT cells and 7/8 Claude cells (CI excludes zero). The blunt nonparametric check: among GPT’s 247 answers with `q = 0.99`, at `L = 10` / human owner, hidden-VERIFY items were wrong **26.2%** of the time vs **5.5%** for hidden-USE. Same reported percentage, very different actual error rate. This supports only the narrow claim: the hidden action is not a function of the reported number alone.
- **The proposed “whichever signal is better, visibility helps/hurts accordingly” pattern does not show up as a cross-model law.** Descriptive Pearson `r ≈ 0.11` across 32 non-independent cells. GPT visibility hurts at high `L` even though its equal-budget ranking edge is uncertain. Claude visibility *helps* at high `L` (it makes Claude verify more) even though hidden ranking is better. Do not cite a p-value here.
- **There is enough to run a small causal fake-confidence pilot, not enough to write an interpretability paper, and not enough to claim that showing confidence “suppresses a richer uncertainty signal” as the GPT headline.** The GPT visibility effect is mostly: become the mechanical overconfident threshold. The residual-information and Claude ranking results are real enough to justify a cheap causal follow-up, not a mechanism story.

---

## 3. Equal-budget results

Question: when confidence is hidden, is the model better at picking *which* answers to verify, or does it only catch more mistakes because it verifies more often?

For each model × owner × `L`, take the hidden verification rate as the checking budget. Build a confidence-only policy with the same budget (lowest `q` first, fractional ties). Main metric: **error catch rate** = fraction of actually wrong Stage-1 answers that get verified.

### Model-level averages (8 cells: 2 owners × 4 `L`)

These are means of cell-level gaps. The CI resamples *questions* and re-averages the cells, so the 8 cells are not treated as independent observations.

| Model | Hidden budget | Hidden catch | Confidence-only catch at same budget | Hidden − conf (pp) | 95% CI | Cells with CI excluding 0 |
|---|---:|---:|---:|---:|---|---:|
| Claude Sonnet 5 | 56.2% | 88.3% | 78.6% | **+9.6** | **[3.2, 16.0]** | 8 / 8 |
| GPT-5.6 Sol | 31.0% | 61.1% | 53.9% | +7.1 | [−2.4, 16.4] | 1 / 8 |
| Gemini 3.8 Flash | 8.3% | 38.8% | 36.0% | +2.8 | [−3.4, 9.6] | 1 / 8 |
| Grok 4.20 | 96.2% | 99.4% | 100.0% | **−0.6** | **[−1.1, −0.2]** | 1 / 8 (against hidden) |

Owner-separated averages are in `checkpoint_A_pooled_catch_summaries.csv`. They look like the pooled numbers: Claude’s gap is present for both owners; GPT’s pooled CI includes zero for both owners; Grok’s slight negative gap is concentrated in the AI-owner cells.

### What the risk/coverage plots show

x-axis = fraction verified. y-axis = fraction of wrong answers caught. The solid curve is “always check the current lowest-confidence answers.” Random checking is the diagonal. Hidden policies are filled markers; visible policies are open markers.

- **Claude (`figures/risk_coverage_anthropic_sonnet5.png`).** Hidden points sit **above** the curve. That is the cleanest evidence in this checkpoint that hidden checking is not just “the number, plus a different threshold.” Visible Claude points move right (more checking) and stay near or slightly above the curve; showing `q` makes Claude *more* cautious, which is the opposite of GPT.
- **GPT (`figures/risk_coverage_openai_gpt56_sol.png`).** Hidden points sit slightly above the curve at ~22–39% coverage. Visible points sit **on** the curve at a much smaller coverage (~4–22%). The GPT visibility effect is mostly a leftward slide along the confidence curve, not a collapse of ranking quality at a fixed budget.
- **Gemini.** Hidden and visible both hug the confidence curve. Hidden budget is tiny (~6–11%), so there is little room for a ranking story. Visible at high `L` moves right along the same curve (more checking, more errors caught).
- **Grok.** Hidden is jammed at the top-right corner (verify almost everything). Visible is a bit left of that, still high. Ranking comparisons at 96% coverage are not informative.

At matched budget, precision of verification and the correct-answer verification rate are determined by catch rate (same `n`, same budget, same `n_wrong`). They are in `checkpoint_A_cell_results.csv` and do not change the ranking of policies.

### Cell-level GPT and Claude catch gaps

GPT, hidden minus confidence-only catch (pp), with 95% CI:

| Owner | L=2 | L=5 | L=10 | L=20 |
|---|---|---|---|---|
| AI | +10.0 [0.0, 20.2] | **+12.2 [1.3, 23.4]** | +5.7 [−5.3, 17.3] | +4.6 [−6.7, 15.6] |
| Human | +4.8 [−5.8, 15.9] | +7.1 [−3.6, 17.5] | +5.6 [−6.6, 17.2] | +7.1 [−4.4, 19.1] |

Claude, same layout:

| Owner | L=2 | L=5 | L=10 | L=20 |
|---|---|---|---|---|
| AI | +11.1 [3.9, 18.4] | +10.4 [3.3, 17.1] | +7.9 [1.3, 14.7] | +7.2 [1.9, 12.8] |
| Human | +12.8 [6.1, 20.0] | +10.6 [3.7, 17.1] | +9.9 [3.0, 16.8] | +7.3 [0.9, 14.0] |

**Adversarial reading.** If the interesting GPT story were “hidden judgment ranks mistakes better than the number,” equal-budget CIs would exclude zero in more than one cell, and the pooled GPT interval would not cover zero. They do not. Claude *does* pass that test.

---

## 4. Calibration results

Two different questions, kept separate:

**Calibration.** When the model says 90%, is it right about 90% of the time?

**Discrimination.** Does a lower number correctly flag the answers that are more likely wrong? (AUROC, and the Part 1 equal-budget curve.)

Primary calibrator: **isotonic regression** (the same PAVA implementation as the original V2 code), because the protocol already committed to a monotone map from `q` to a probability, and because GPT’s `q` is a few mass points rather than a smooth score. Robustness: unregularized logistic / Platt scaling of raw `q`. Both are 5-fold cross-fitted by question, stratified on Stage-1 correctness, so a question’s own label is never used to calibrate that question.

### Score quality (all 500 questions, cross-fitted)

| Model | Raw Brier | Iso Brier | Raw AUROC | Iso AUROC | Raw ECE | Iso ECE | Mean `q` if correct | Mean `q` if wrong |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Claude | 0.163 | 0.166 | 0.737 | 0.712 | 0.038 | 0.033 | 0.77 | 0.62 |
| Gemini | 0.083 | 0.082 | 0.867 | 0.840 | 0.049 | 0.029 | 0.94 | 0.73 |
| **GPT** | **0.158** | **0.136** | **0.684** | **0.653** | **0.136** | **0.033** | **0.95** | **0.86** |
| Grok | 0.191 | 0.192 | 0.744 | 0.725 | 0.030 | 0.016 | 0.71 | 0.54 |

GPT is the miscalibrated one. Wrong answers are still reported at mean 86%. Calibration fixes the *percentages* (ECE 0.136 → 0.033) and slightly *hurts* ranking (AUROC 0.684 → 0.653) because isotonic flattens regions. That is expected: calibration is not a discrimination upgrade.

Reliability diagrams: `figures/calibration_<model>.png`. GPT’s raw curve is too high in the 0.8–1.0 mass; after isotonic the high end is closer to observed accuracy.

### Cost under the actual rule: verify if `(1 − p) × L > 1`

Owner-pooled mean realized cost. Hidden/visible are the real Stage-3 policies. Raw / isotonic / Platt are synthetic threshold policies on the same frozen answers.

**GPT**

| L | Hidden cost | Visible cost | Raw-q threshold | Isotonic threshold | Hidden verify | Visible verify | Raw verify | Iso verify |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.388 | 0.347 | 0.352 | 0.354 | 22.2% | 3.9% | 3.2% | 2.6% |
| 5 | **0.626** | 0.782 | 0.786 | 0.718 | 28.1% | 7.7% | 7.6% | 21.8% |
| 10 | 0.975 | 1.293 | 1.328 | **0.956** | 35.5% | 15.3% | 14.8% | **95.6%** |
| 20 | 1.523 | 2.139 | 2.076 | **0.956** | 38.3% | 21.9% | 23.6% | **95.6%** |

At high `L`, a calibrated GPT number says “these 0.99s are not actually 0.99,” so the Bayes threshold verifies almost everything. That crushes visible/raw cost. Hidden GPT at `L = 20` (cost 1.52) is **worse** than the calibrated threshold (0.96), because hidden still leaves about a third of wrong answers unchecked. So: **calibration does not make hidden GPT look like the smart policy. It makes the *number*, once believed and thresholded, want to verify far more than visible GPT does.**

Visible GPT ≈ raw threshold (see Part 5). The high-stakes GPT failure is: show an overconfident number, follow it, verify too little.

**Does calibration remove Claude’s hidden ranking advantage?** No. That advantage is a discrimination result (Part 1). Claude is already reasonably calibrated (ECE 0.038). At `L = 5` hidden Claude is cheaper than the isotonic threshold (0.71 vs 0.88); at `L = 20` the isotonic/raw thresholds are cheaper than hidden because they verify more. Different operating points, not “hidden was just miscalibrated `q`.”

**Gemini / Grok, short.** Gemini hidden under-verifies at high `L`; raw and isotonic thresholds catch more errors and have lower cost (`L = 20`: hidden 1.46 vs isotonic/raw 0.55). Grok hidden over-verifies at low `L`; raw/isotonic are cheaper at `L = 2` (0.66 vs hidden 0.96). Platt often collapses to “always verify” at `L = 20` (Claude, Gemini, GPT, Grok). That is a known quirk of logistic calibration near 1, which is why isotonic is the primary method.

**Answer to “is this just a calibration problem?”**

- For GPT’s *visible high-L cost*: mostly yes. The number is too high, and GPT follows it.
- For “hidden ranking beats the number at the same budget”: no. Calibration cannot explain Claude’s equal-budget gap, and GPT’s equal-budget gap was already uncertain.
- Hidden is not secretly implementing a well-calibrated threshold. If it were, GPT hidden at `L = 20` would look like 96% verify, and it does not (38%).

Full table: `checkpoint_A_calibration_results.csv`. Cost-by-`L` figure: `figures/cost_policies_by_L.png`.

---

## 5. Residual-information results

Question: once we already know the reported confidence percentage, does the hidden VERIFY/USE bit tell us anything more about whether the Stage-1 answer is wrong?

Method: 5-fold question-level logistic regression, out-of-sample.

- Model A: `P(wrong)` from `q` only.
- Model B: `P(wrong)` from `q` plus hidden `VERIFY_FIRST`.

Positive log-loss / Brier delta means B is better (hidden action reduces loss). Positive AUROC delta means B ranks wrong answers better. Log-loss and Brier CIs use 5,000 question-level resamples of the frozen out-of-sample predictions. AUROC CIs use every 5th resample (1,000) because the rank statistic is slower; the AUROC *point estimate* uses the full sample.

### Out-of-sample deltas (log-loss improvement from adding hidden action)

| Model | Cells where log-loss CI is entirely > 0 | Mean Δ log-loss | Typical AUROC Δ (point) |
|---|---:|---:|---:|
| Claude | **7 / 8** | +0.026 | ~+0.03 to +0.05 |
| GPT | **7 / 8** | +0.027 | ~+0.02 to +0.04 |
| Gemini | 0 / 8 (1 cell CI entirely < 0) | +0.005 | ~0 |
| Grok | 3 / 8 | ~0 | ~0 |

GPT example, AI owner `L = 5`: log-loss 0.450 → 0.411, Δ **+0.039 [0.013, 0.066]**; Brier 0.140 → 0.128. AUROC 0.652 → 0.694, but the AUROC CI includes zero. The loss metrics move; the ranking metric is noisier.

Gemini: adding hidden action does not help. Grok: hidden action is almost always VERIFY, so there is almost no leftover bit.

### Nonparametric check: same reported `q`, different hidden action

Exact-`q` groups with at least 10 answers and both actions present. Among those groups, answers the model chose to verify are, on average, more often wrong than answers it chose to use.

The important GPT cell is the giant `q = 0.99` pile (247 questions). At `L = 10`, human owner:

| | n | Actual error rate |
|---|---:|---:|
| Hidden VERIFY_FIRST | 65 | **26.2%** |
| Hidden USE_UNVERIFIED | 182 | **5.5%** |

Same spoken percentage, roughly a 5× difference in actual error rate. That is the least parametric evidence that hidden action is not a function of the reported number.

Claude shows the same pattern at common `q` values (e.g. `q = 0.85`, n = 81, `L = 10` human: verify error 37.5% vs use error 10.2%).

**Allowed claim.** The hidden verification decision contains correctness-relevant information not captured by the reported confidence score, for GPT and Claude.

**Not allowed.** Hidden states contain richer uncertainty; the model internally knows the answer is wrong; there is an information bottleneck. Those need future experiments.

Tables: `checkpoint_A_residual_results.csv`, `checkpoint_A_residual_bins.csv`. Figure: `figures/residual_logloss_delta.png`.

---

## 6. Cross-condition exploratory pattern

Proposed principle: showing confidence should *help* when the confidence score ranks errors better than hidden judgment, and *hurt* when hidden judgment is better.

For each of the 32 model × owner × `L` cells:

- **x = signal gap** = hidden catch − confidence-only catch at the same budget. Positive = hidden ranking is better.
- **y = visibility cost effect** = visible mean cost − hidden mean cost. Positive = showing confidence hurt.

Scatter: `figures/signal_gap_vs_visibility.png`.

**This is descriptive only.** The 32 points reuse the same 500 questions and four models. They are not 32 independent samples. No p-value is reported.

| | Pearson r | Spearman r |
|---|---:|---:|
| All 32 cells (invalid as a test) | 0.11 | 0.07 |
| Claude, 8 cells | 0.85 | 0.93 |
| GPT, 8 cells | −0.30 | −0.26 |
| Gemini, 8 cells | −0.22 | −0.26 |
| Grok, 8 cells | 0.29 | 0.14 |

The overall cloud is a blob. Claude’s high within-model r is an `L` confound: as stakes rise, Claude’s ranking gap shrinks a little *and* showing `q` makes Claude verify more, which helps cost at high `L`. That is not the hypothesized “better signal wins” law. GPT’s high-`L` points are the ones where showing confidence hurts most, while GPT’s equal-budget ranking edge is weakest/noisiest there — the opposite of the simple story.

**Plain statement.** The proposed pattern does not appear as a usable cross-model relationship in this existing sample.

Also reported in the cell CSV, for completeness:

- Visible minus hidden wrong-unverified rate
- Visible minus hidden verification rate

For GPT, visibility *lowers* verification by about 16–21 pp and *raises* the wrong-unverified rate by about 21–45 pp. For Claude, visibility *raises* verification by about 13–22 pp and *lowers* the wrong-unverified rate.

---

## 7. What this DOES and DOES NOT establish

**Does establish, in this frozen V2-B sample:**

1. The checkpoint is complete: 4 models, 500 questions, 32,000 primary Stage-3 cells, frozen answer/confidence consistent.
2. Showing GPT its confidence number turns Stage 3 into the raw-`q` threshold policy (~99% action agreement) with a smaller checking budget. That is enough to explain more unchecked errors at high `L` without any “richer signal” story.
3. Claude’s hidden policy catches more errors than a budget-matched confidence ranking, with CIs excluding zero.
4. GPT’s corresponding ranking edge is only a point-estimate, not a reliable interval.
5. GPT’s spoken percentages are miscalibrated in the overconfident direction. Recalibrating them changes the Bayes threshold a lot at high `L`. That is a calibration fact, not a ranking fact.
6. For GPT and Claude, hidden VERIFY vs USE is still associated with actual error after you condition on the spoken `q` (logistic OOS + the `q = 0.99` split).

**Does not establish:**

1. That showing confidence suppresses a richer internal uncertainty representation. Not measured.
2. That GPT “knows” which answers are wrong in any mechanistic sense. Residual association is not process.
3. A four-model law relating signal gap to visibility harm.
4. That hidden GPT is a well-calibrated decision-theoretic agent. At `L = 20` a calibrated threshold wants ~96% verify; hidden GPT verifies 38%.
5. Anything about paraphrase robustness, GPQA, or other datasets. Not used here.
6. Causal status of the number. Visibility is a bundle: a sentence is added *and* it contains `q`. A fake-`q` swap is required to separate “a number was shown” from “this particular number was used.”

---

## 8. Recommended next experimental step

**C. CONTINUE to causal confidence pilot.**

Not A: there is more here than “everyone is just more/less cautious,” and Claude’s equal-budget ranking gap plus the GPT `q = 0.99` split would be irresponsible to ignore.

Not B-only: a further purely behavioral pass on the same data will not answer whether the shown number is a causal sufficient statistic.

Not D: residual information is a predictive association. It does not warrant interpretability, probes, or bottleneck language.

**What the small pilot should do.** On a cheap existing-question subset, keep Stage 1 and Stage 2 frozen. In Stage 3 visible, replace the shown probability with a swapped / shifted / constant fake `q` (including values that contradict the model’s own Stage-2 number). Pre-register that GPT should track the *shown* number like the raw threshold if the mechanical-policy account is right. If hidden-like ranking survives fake `q`, that is the first causal evidence that something other than the displayed number is driving the check/use decision.

Keep the pilot small. Do not spend interpretability budget until fake-`q` says there is something left to interpret.

---

## 9. Exact numbers GPT should know

| Quantity | Claude | Gemini | GPT | Grok |
|---|---:|---:|---:|---:|
| Accuracy (500 Q) | 0.742 | 0.880 | 0.826 | 0.644 |
| n wrong | 129 | 60 | 87 | 178 |
| Raw AUROC (correctness) | 0.737 | 0.867 | 0.684 | 0.744 |
| Raw 10-bin ECE | 0.038 | 0.049 | **0.136** | 0.030 |
| Iso-CV ECE | 0.033 | 0.029 | 0.033 | 0.016 |
| Mean `q` \| correct | 0.774 | 0.944 | 0.946 | 0.708 |
| Mean `q` \| wrong | 0.617 | 0.728 | 0.856 | 0.538 |
| Mean hidden verify budget | 56.2% | 8.3% | 31.0% | 96.2% |
| Mean hidden error catch | 88.3% | 38.8% | 61.1% | 99.4% |
| Mean conf-only catch, same budget | 78.6% | 36.0% | 53.9% | 100.0% |
| Mean catch gap hidden−conf (pp) | **+9.6** | +2.8 | +7.1 | −0.6 |
| Catch-gap 95% CI | **[3.2, 16.0]** | [−3.4, 9.6] | **[−2.4, 16.4]** | [−1.1, −0.2] |
| Visible vs raw-threshold agreement | 76.3% | 94.3% | **99.2%** | 72.2% |
| Hidden vs raw-threshold agreement | 68.7% | 84.7% | 76.8% | 72.0% |
| Residual log-loss cells with CI > 0 | 7 / 8 | 0 / 8 | 7 / 8 | 3 / 8 |
| GPT `q=0.99` error \| hidden VERIFY vs USE (`L=10`, human) | — | — | **26.2% vs 5.5%** (65 vs 182) | — |
| GPT `L=20` pooled cost hidden / visible / raw / iso | — | — | **1.52 / 2.14 / 2.08 / 0.96** | — |
| GPT `L=20` pooled verify hidden / visible / raw / iso | — | — | **38.3% / 21.9% / 23.6% / 95.6%** | — |
| Descriptive r, signal gap vs visibility cost (32 cells) | 0.11 (not a test) |  |  |  |

**One-sentence GPT takeaway.** Showing GPT its confidence number makes it follow that number, which is overconfident, so it checks too little at high stakes; whether GPT’s *hidden* checker is actually better at ranking errors than the number is still unproven after a budget-matched test.

**One-sentence Claude takeaway.** Claude’s hidden checker does beat the number at the same budget, and showing the number makes Claude check *more*, not less.

---

## Files

- Script: `to_gpt/checkpoint_A/analyze_checkpoint_A.py`
- Cell table: `to_gpt/checkpoint_A/checkpoint_A_cell_results.csv`
- Model summary: `to_gpt/checkpoint_A/checkpoint_A_model_summary.csv`
- Calibration: `to_gpt/checkpoint_A/checkpoint_A_calibration_results.csv`
- Residual: `to_gpt/checkpoint_A/checkpoint_A_residual_results.csv`
- Extra: `checkpoint_A_residual_bins.csv`, `checkpoint_A_pooled_catch_summaries.csv`, `data_validation.json`, `run_metadata.json`
- Figures: `to_gpt/checkpoint_A/figures/`
