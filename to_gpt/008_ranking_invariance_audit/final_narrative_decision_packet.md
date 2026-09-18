# Final narrative decision packet

Ignore how we got here. This packet is about the strongest FINAL paper the current exploratory evidence can support.

**Current decision bucket:** `INVARIANCE_SUPPORTED_EXPLORATORY`
**Favored world:** WORLD 1 (MODERATELY_FAVORS)

GPT exploratory class: `invariance` (invariance votes 5, reshape votes 0; mean held-out u AUROC 0.777).
Claude exploratory class: `invariance` (invariance votes 5, reshape votes 0; mean held-out u AUROC 0.881).

## WORLD 1 — Ranking invariance

**Strongest honest narrative.** Counterfactual confidence is a coverage knob. It moves how many answers the model verifies. It does not, on current evidence, rewrite a stable item ordering in a way that survives logit-scale and held-out-propensity tests. Probability-scale hard/easy gaps can still change as a sigmoid artifact.

**Exact novelty claim.** Displayed confidence causally sets a verification operating point, including without explicit L/C arithmetic, while item ranking remains approximately invariant across that operating point.

**Contributions (max 4).**
1. Causal coverage control from a displayed confidence scalar, including qualitative stakes.
2. Decomposition of selective oversight into operating point vs item ranking.
3. Evidence that nested VERIFY sets and transferable item propensity are compatible with a threshold shift.
4. Reinterpretation of the workshop visible-confidence catch drop as coverage/calibration, not ranking collapse.

**Figures 1–4.** Fig 1: qualitative VERIFY vs displayed score. Fig 2: ROC / risk-coverage overlay. Fig 3: nesting/reversals. Fig 4: matched-budget workshop curve with visible points sliding along it.

**Practical implication.** If you want more checking, lower the displayed score (or raise stakes). Do not expect that knob to retarget *which* items get checked, beyond the mechanical effect of a different threshold on a fixed ranking.

**Closest prior-work distinction.** Not calibration-only, not “models follow expected-cost arithmetic,” not hidden-state suppression. The instrument is an externalized score used as a policy input.

**Prospective result needed.** On a fresh sample, coverage must move while (a) reversals stay near rerun noise, (b) a shared item propensity predicts held-out scores, (c) condition-sensitive models add little, (d) matched-coverage catch stays similar.

**Likely second-task requirement.** Repeated generations on a subset to show the latent ranking is stable, not a one-draw artifact. Optional: a cheap routing evaluation at matched budget.

## WORLD 2 — Ranking reshaping

**Strongest honest narrative.** The displayed score does not only ration verification. At non-saturated operating points it changes which items are treated as worth checking, so risk-coverage points leave a common ranking curve.

**Exact novelty claim.** Counterfactual confidence feedback is an item-prioritization intervention, not merely a coverage control.

**Contributions (max 4).**
1. Causal displayed-score control without requiring L/C arithmetic.
2. Identification of ranking/prioritization change separate from coverage.
3. Logit-scale score × difficulty interaction and/or non-nested reversals as the statistical signature.
4. Error-catching consequences at matched coverage, not raw catch at different budgets.

**Figures 1–4.** Fig 1: score-response. Fig 2: ROC points off a common curve. Fig 3: switch-set / reversals. Fig 4: matched-coverage catch gap vs a shared-ranking baseline.

**Practical implication.** Feeding back a confidence number can retarget oversight, so a dashboard that only sets a checking rate is incomplete.

**Closest prior-work distinction.** Stronger than a threshold-shift paper. Must beat the link-function and saturation critiques.

**Prospective result needed.** Material condition-sensitive improvement and/or matched-coverage catch change at identifiable (non-saturated) points, predeclared.

**Likely second-task requirement.** More operating points plus item-level repeats so per-condition propensities exist.

## WORLD 3 — Mixed / model-specific

**Strongest honest narrative.** GPT and Claude do not share one ranking law. GPT looks closer to a coverage/threshold machine. Claude’s remaining checks may concentrate differently, but that pattern must be separated from saturation and probability-scale artifacts. Heterogeneity is a finding only if predeclared and replicated; it is not automatically a contribution.

**Exact novelty claim.** Selective-oversight control by displayed confidence is real, but ranking invariance is model-dependent.

**Contributions (max 4).**
1. Causal coverage control without explicit arithmetic.
2. A predeclared GPT vs Claude contrast on ranking invariance, not a post-hoc split.
3. Difficulty as an evaluation covariate, not a production feature.
4. Workshop coverage-vs-ranking split, which already differs by model.

**Figures 1–4.** Same geometry, faceted by model, with a predeclared “same law / different law” test.

**Practical implication.** Do not ship a universal “confidence slider” story. GPT-like systems may be coverage-controllable; Claude-like systems need a ranking check.

**Closest prior-work distinction.** Cross-model policy difference, not a hidden-state story and not a four-model fishing expedition.

**Prospective result needed.** Predeclare the GPT vs Claude invariance tests before seeing the fresh sample. Gemini/Grok are secondary.

**Likely second-task requirement.** Enough Claude non-saturated points, and repeats, so “sharpening” cannot hide in a ceiling.

## What existing evidence favors

- Bucket: `INVARIANCE_SUPPORTED_EXPLORATORY`
- Strength: `MODERATELY_FAVORS`
- This is exploratory N=100 with one generation per cell. It cannot freeze World 1 as a law.
