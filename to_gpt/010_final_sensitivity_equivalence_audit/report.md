# Task 010 — Final zero-call sensitivity, equivalence, and routing audit

## 1. Plain-English bottom line

Task 009's confirmatory **coverage** result is untouched: displayed 0.70 vs 0.99 still moves verification a great deal. This audit asks how strong the **ranking-stability** sentence may be. Predeclared wording category: **WEAK**.

Displayed confidence strongly shifts verification coverage, while our data do not support large score-dependent reranking. The study is not sufficiently sensitive to rule out smaller ranking changes.

The Task-009 H2 test is a detector of **score × other-primary-wrongness**, not of all reranking. It has a high detection floor even for that family, and it has **no power** against generic item–condition noise on the simulated grid. Post-hoc, own-q1 mismatch predictors improve VERIFY prediction, but that is not confirmatory H2 and is not the same as displayed score reshuffling items.

## 2. Validation / zero-call confirmation

- API calls: **0**
- paperDirection.txt sha256 before/after: `8546e2e8002d9c96c6be7c946a4d63705068adda133b7a7c0c10bee2b8542b36` / `8546e2e8002d9c96c6be7c946a4d63705068adda133b7a7c0c10bee2b8542b36` (unchanged)
- Task 009 `stage3_results.csv` sha256 unchanged
- Claim templates frozen in `claim_wording_preregistered.md` before simulations
- Labels kept separate: Task-009 H2 is confirmatory; positive-control / AUROC / latent are Task-010 sensitivity; mismatch is **POST_HOC_ROBUSTNESS**
- Isolation: no writes to paperDirection, 007–009 packets, or historical SQLite files

## 3. Positive-control simulation design

See `positive_control_design.md`. Seed `20261010`, 500 Monte Carlo draws per cell, 400 question-bootstrap resamples of frozen OOF predictions. Detector is the Task-009 H2 pipeline: 5-fold GroupKFold by question, shared = condition intercepts + other-primary wrongness, richer = shared + γ(displayed−0.90)×wrongness, fail invariance if held-out improvement ≥ 0.01 **or** bootstrap 95% UB ≥ 0.01.

Family 1 injects that same interaction at 0, 0.25, 0.5, 1.0, 1.5, 2.0 × the Task-008 Claude development γ = 5.157. Family 2 adds independent Gaussian logit noise; **report the realized adjacent Spearman**, not the Pearson target labels (binary wrongness makes Spearman far more fragile than the Pearson calibration).

## 4. Detection floor and power

False-positive rate at zero injection (full 009 rule): GPT **0.000**, Claude **0.000**.

**Family 1 (score × difficulty)**

| Model | 0× | 0.25× | 0.5× | 1.0× (γ=5.16) | 1.5× (γ=7.73) | 2.0× (γ=10.31) |
|---|---|---|---|---|---|---|
| GPT power | 0.000 | 0.000 | 0.000 | 0.214 | **0.966** | 1.000 |
| GPT mean Δll | −0.0002 | 0.0000 | 0.0009 | 0.0041 | 0.0104 | 0.0189 |
| GPT Spearman | 1.00 | 1.00 | 1.00 | 1.00 | 0.50 | 0.50 |
| Claude power | 0.000 | 0.000 | 0.000 | 0.004 | 0.090 | 0.518 |
| Claude mean Δll | −0.0003 | −0.0002 | −0.0001 | 0.0008 | 0.0025 | 0.0054 |
| Claude Spearman | 1.00 | 1.00 | 1.00 | 1.00 | 0.50 | 0.50 |

Because difficulty is binary, family 1 does not produce a smooth “small Spearman drop.” Adjacent Spearman stays 1.0 until the interaction flips the two-group order (between 1.0× and 1.5×), then jumps to 0.50 with 25% pairwise reversal — a **large** ranking change by the predeclared scale.

The Task-009 analysis could reliably detect reranking of approximately **1.5× the Claude-008 γ (γ≈7.73), which already induces Spearman 0.50**, for GPT. For Claude, 80% power was **not reached** even at 2.0× (power 0.518 at the same large Spearman-0.50 disruption).

**Family 2 (generic item–condition noise):** 80% power **not reached** on either model. Realized adjacent Spearman was already ~0.55 (GPT) / ~0.42 (Claude) at the smallest noise and still yielded power 0.000. H2 does not see generic reranking that is orthogonal to other-primary wrongness.

## 5. Which predeclared equivalence wording is justified

**WEAK**

Displayed confidence strongly shifts verification coverage, while our data do not support large score-dependent reranking. The study is not sufficiently sensitive to rule out smaller ranking changes.

This is the conservative shared template. GPT reaches 80% power only after family 1 has already become a large ranking rewrite. Claude never reaches 80% on the grid. Family 2 never reaches 80% at all.

## 6. Shared-ranking prediction of AUROC drift

RMSE of observed vs shared-model predicted binary AUROC: **0.077**.

GPT observed AUROC falls with score (0.680 at 0.70 → 0.564 at 0.99). The shared model predicts the same **direction** (0.586 → 0.537) but underpredicts the level. Residuals: +0.093, +0.092, +0.066, +0.055, +0.027.

Claude observed AUROC rises with score (0.550 at 0.70 → 0.680 at 0.99). The shared model again matches **direction** (0.515 → 0.569) and underpredicts. Residuals: +0.035, +0.061, +0.069, +0.106, +0.111.

Do not wave this away. Directional AUROC drift is compatible with a moving threshold on a shared ranking; the **magnitude** is not fully explained. Several conditions have |residual| > 0.05.

## 7. Hierarchical latent rank-stability result

Algorithmic MAP label if taken at face value: `LATENT_STABILITY_STRONG` (latent Spearman ≈ 0.999). **Do not put that label in the paper without the criticism below.**

N=100 repeat questions × 3 Bernoulli draws. MAP item variance dominates condition×item variance (GPT τ²=2.09, σ²=0.013; Claude τ²=5.62, σ²=0.065), which forces latent Spearman to ~1. That is mostly **unidentifiability / overshrinkage**, not a measured proof of identical rankings.

Posterior-predictive check: if the fitted latent propensities were right, 3-draw Spearman would be much lower than several observed raw values. GPT 0.85→0.90 raw Spearman is 0.857 vs PP mean 0.544. GPT 0.70→0.85 raw 0.541 is compatible with PP 0.481 [0.334, 0.616]. Claude 0.70→0.85 raw 0.368 vs PP 0.483.

Paper-facing label: **INSUFFICIENT_REPEAT_INFORMATION**. Three Bernoulli repeats cannot support a strong latent-invariance claim once shrinkage is this severe and the PP check misfits.

## 8. Targeted mismatch robustness

**POST_HOC_ROBUSTNESS — NOT PREREGISTERED TASK-009 CONFIRMATION.**

| Model | Predictor | Δ held-out log loss | 95% UB | material 0.01 |
|---|---|---|---|---|
| GPT | signed mismatch (disp−q1) | +0.069 | 0.099 | yes |
| GPT | abs mismatch | +0.037 | 0.059 | yes |
| GPT | logit mismatch | +0.003 | 0.011 | no |
| GPT | score × difficulty | +0.0008 | 0.003 | no |
| Claude | signed mismatch | +0.144 | 0.177 | yes |
| Claude | abs mismatch | +0.071 | 0.088 | yes |
| Claude | logit mismatch | +0.034 | 0.049 | yes |
| Claude | score × difficulty | −0.0000 | 0.001 | no |

Score × difficulty remaining null matches confirmatory H2.

**Do not read signed-mismatch as displayed-score reranking.** Displayed score is constant within a condition, so `disp − q1` is in the span of condition intercepts plus q1. The held-out gain says **own q1 predicts VERIFY after condition intercepts and other-primary wrongness**. That is an omitted stable item signal, not evidence that the displayed number reshuffles the ranking. Abs / logit mismatch are nonlinear in q1 and also improve Claude (and abs improves GPT); still post-hoc, still not H2.

Narrow the invariance sentence to the **preregistered score×difficulty alternative**, not to “no item feature predicts verification.”

## 9. Matched-budget routing

**MODEST_ROUTING_GAIN** — paragraph, not a main-result figure, unless used as a descriptive operating-point plot.

Judgment gain = best of {hidden, q1+hidden} minus best of {raw q1, calibrated q1} at 20–40% budget:

- GPT: −1.6pp at 20%, **+10.7pp** at 30%, **+15.9pp** at 40%
- Claude: +0.3pp, −1.4pp, −0.7pp

Material (≥10pp) in GPT at 30–40% only; not replicated in Claude. Cross-model difficulty is evaluation-only and is not allowed to decide MATERIAL. Random and hindsight oracle are not realistic baselines.

## 10. Any evidence against the current narrative

- H2 cannot rule out small or even moderate score×difficulty reranking; Claude power is only 0.52 at 2× the development anchor.
- H2 is blind to generic item–condition noise even when realized Spearman is ~0.4–0.55.
- Shared-ranking AUROC predictions match direction, not magnitude (|residual| up to 0.11).
- Repeat-based latent Spearman ≈1 is not credible; PP check misfits.
- Post-hoc q1/mismatch predictors beat the 0.01 scale. They do **not** overturn confirmatory H2, but they forbid wording that “nothing else predicts verification.”

Coverage control from Task 009 is not contradicted.

## 11. Final strongest defensible claim

Displayed confidence strongly shifts verification coverage, while our data do not support large score-dependent reranking. The study is not sufficiently sensitive to rule out smaller ranking changes.

More precise: the preregistered score×difficulty alternative adds essentially no held-out log-loss on confirmatory data, but Task 010 shows that this test would only have caught a **large** rewrite of that specific form, and would not have caught generic reranking.

## 12. Exact wording that should be removed from paperDirection.txt, if any

Task 010 does not edit `paperDirection.txt`. Remove or qualify any sentence that treats the H2 null as proof that small reranking is ruled out, including:

- Current core claim (lines 1828–1831): “allowing confidence to create score-specific reprioritization provides essentially no held-out predictive benefit” — keep the **held-out number**, drop the implication that ranking is shown invariant below a large detection floor.
- Abstract-scale lines 51 and 1631/1638 that pair the coverage effect with “essentially no held-out predictive value” without the detection-floor caveat.
- “the dominant role of counterfactual displayed confidence is coverage control rather than ranking rewrite” if that is stated as if small ranking rewrite were tested and rejected.

Keep: the 50.2pp / 35.4pp coverage shifts; the confirmatory H2 point estimates (GPT +0.0008, Claude ~0); the systems line that coverage can be tuned by displayed confidence.

## 13. Whether MMLU analysis should now stop

YES. Do not mine more MMLU-Pro interactions. Move to literature/methods positioning, one non-MCQ generalization experiment, and paper writing. The next paper-direction edit should **weaken** the ranking-invariance sentence to the WEAK template, not launch another MMLU grid.

## 14. Review flag

READY_FOR_GPT_REVIEW = YES
