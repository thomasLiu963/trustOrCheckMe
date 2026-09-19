# Task 012 — Oversight sensitivity and error-leakage audit

Label: `POST_HOC_SYSTEMS_REANALYSIS`. Zero API calls. Frozen Tasks 009 and 011 only.

## 1. Plain-English bottom line

Displayed confidence with **no item-discriminative information within a condition** still moves how often GPT and Claude request independent verification, and those coverage moves change how many frozen incorrect outputs escape checking.

The 0.70→0.99 leakage change (unverified errors as a fraction of all outputs) is:

| Task | Model | Leakage @0.70 | Leakage @0.99 | Δ pp [95% CI] | Coverage |
|---|---|---|---|---|---|
| mmlu | GPT | 2.0 | 13.8 | +11.8 [+9.0, +14.6] | 58.4 → 8.2 |
| mmlu | Claude | 0.0 | 3.8 | +3.8 [+2.2, +5.6] | 92.4 → 57.0 |
| code | GPT | 10.8 | 32.9 | +22.0 [+17.5, +26.9] | 62.9 → 24.1 |
| code | Claude | 0.0 | 1.4 | +1.4 [+0.3, +2.8] | 99.7 → 94.8 |

Paper-impact gate: **MATERIAL_UPGRADE**.

## 2. Reproduction check

See `reproduction_check.md`. Status: **PASS**.

Published 009 coverage 50.2 / 35.4 pp and 011 coverage 38.8 / 4.9 pp plus 011 pass 46.5% / 43.0% all reproduced within ±0.6 pp.

## 3. Why the constant-score intervention is item-uninformative within condition

Within each fixed displayed-score condition, every item receives the same scalar. Variance of displayed score across items is exactly 0. The displayed value therefore cannot distinguish items *within that condition*. Any item-specific VERIFY vs USE_UNVERIFIED differences must come from the model interacting that constant scalar with the question, frozen output, or context, or from stochasticity.

Use: “no item-discriminative information within the condition.” Do not say the score contains no information whatsoever. It still carries a claimed correctness probability.

This is a conceptual observation, not a new empirical result.

## 4. Verification coverage by score

| Task | Model | hidden | true_q | 0.70 | 0.85 | 0.90 | 0.95 | 0.99 |
|---|---|---|---|---|---|---|---|---|
| mmlu | GPT | 39.6 | 12.4 | 58.4 | 22.8 | 18.0 | 14.4 | 8.2 |
| mmlu | Claude | 65.4 | 79.4 | 92.4 | 80.0 | 77.6 | 64.6 | 57.0 |
| code | GPT | 32.5 | 27.3 | 62.9 | 44.4 | 37.8 | 29.0 | 24.1 |
| code | Claude | 98.6 | 97.9 | 99.7 | 99.3 | 98.6 | 97.6 | 94.8 |

Hidden is a reference marker only (no x-coordinate). true_q_visible is not equivalent to the fixed score at mean(q1).

## 5. Unverified-error leakage by score

Primary quantity: `LEAKAGE(q) = P(incorrect AND USE_UNVERIFIED | displayed q)`.

Because frozen outputs and correctness are identical across fixed-score conditions, leakage differences are causally attributable to the routing intervention under this design.

| Task | Model | Leakage @0.70 | Leakage @0.99 | Δ pp [95% CI] | Coverage |
|---|---|---|---|---|---|
| mmlu | GPT | 2.0 | 13.8 | +11.8 [+9.0, +14.6] | 58.4 → 8.2 |
| mmlu | Claude | 0.0 | 3.8 | +3.8 [+2.2, +5.6] | 92.4 → 57.0 |
| code | GPT | 10.8 | 32.9 | +22.0 [+17.5, +26.9] | 62.9 → 24.1 |
| code | Claude | 0.0 | 1.4 | +1.4 [+0.3, +2.8] | 99.7 → 94.8 |

Error-catch rates (P(VERIFY | incorrect)) at 0.70 vs 0.99: GPT MMLU 88.2% → 18.8%; Claude MMLU 100% → 84.3%; GPT code 79.7% → 38.6%; Claude code 100% → 97.5%. Claude MMLU leakage +3.8 pp has a CI that excludes 0 but is below the predeclared 5 pp material gate, so it is not counted as a material cell.

Full condition-level counts are in `error_leakage_by_condition.csv`.

## 6. Coverage sensitivity per +0.10 displayed confidence

| Task | Model | Avg coverage pp / +0.10 | Leakage /100 / +0.10 | Shape |
|---|---|---|---|---|
| mmlu | GPT | -17.3 | +4.07 | threshold-like |
| mmlu | Claude | -12.2 | +1.31 | monotonic |
| code | GPT | -13.4 | +7.60 | monotonic |
| code | Claude | -1.7 | +0.48 | saturated |

Local adjacent finite differences are in `oversight_sensitivity.csv`. Sign is preserved: higher displayed confidence usually *reduces* coverage.

## 7. Error-leakage sensitivity per +0.10

Same table as Section 6, leakage column. The 0.70→0.99 average converts the total leakage change into an intuitive per-+0.10 unit. Do not call this “amplification.”

## 8. Natural q1 distribution and manipulated-range overlap

| Task | Model | Mean q1 | Median | IQR | % in [0.70,0.99] | IQR intersects grid |
|---|---|---|---|---|---|---|
| mmlu | GPT | 0.920 | 0.990 | 0.960–0.990 | 92.0% | yes |
| mmlu | Claude | 0.739 | 0.720 | 0.600–0.863 | 69.8% | yes |
| code | GPT | 0.834 | 0.990 | 0.980–0.990 | 71.3% | yes |
| code | Claude | 0.475 | 0.600 | 0.050–0.850 | 42.7% | yes |

Safe claim: the intervention probes a numerical range that **overlaps** the models’ naturally reported confidence. This does **not** prove that a constant overwrite simulates ordinary miscalibration or natural q1 drift.

## 9. q1 calibration diagnostics

| Task | Model | Brier | ECE (10 equal-width) | Mean q1 | incorrect | Mean q1 | correct |
|---|---|---|---|---|---|
| mmlu | GPT | 0.168 | 0.152 | 0.855 | 0.933 |
| mmlu | Claude | 0.159 | 0.034 | 0.629 | 0.774 |
| code | GPT | 0.379 | 0.377 | 0.711 | 0.976 |
| code | Claude | 0.227 | 0.166 | 0.330 | 0.666 |

Do not overinterpret ECE. Constant-score overwrite is not a calibration intervention.

## 10. Hidden / true-q-visible reference behavior

See `natural_vs_counterfactual.csv`. Hidden coverage is drawn as a horizontal marker; true_q_visible as a distinct marker. Do not plot hidden at an x. Do not treat true_q_visible as the fixed-score curve evaluated at mean(q1).

Descriptively, GPT MMLU hidden coverage (39.6%) and leakage (3.8%) sit between the 0.70 and 0.85 fixed-score points. GPT MMLU true_q_visible coverage (12.4%) and leakage (12.0%) sit near the high-score end of the fixed curve (mean natural q1 = 0.920). Claude MMLU true_q_visible coverage (79.4%) is close to displayed 0.85 (80.0%). Claude-code hidden/true_q/fixed-score policies are all near the VERIFY ceiling.

## 11. Cost/regret sweep

Normalized loss: `loss(λ) = leakage_rate + λ * verification_coverage`.

Oracle (unattainable): if λ<1 verify exactly the incorrect outputs; if λ≥1 verify nothing. Labeled ORACLE. Deployed systems cannot attain it. Do not pick one λ after seeing the data and call it realistic.

Gate C (pass): longest consecutive λ-grid run with regret vs best observed fixed score ≥0.05 is 8 points.

When verification is cheap relative to an escaped error (small λ), the high-score / low-coverage points have the highest regret: GPT MMLU displayed 0.99 regret vs best fixed is 0.12 at λ=0.001; GPT code displayed 0.99 regret is 0.22. When verification is expensive (λ near 1), the low-score / high-coverage points become costly, including Claude-code saturation. Do not treat any single λ as “the realistic cost.”

Full sweep: `cost_sweep.csv`. Figures: `cost_regret_sweep_mmlu.png`, `cost_regret_sweep_code.png`.

## 12. Marginal verification efficiency

Moving toward lower displayed confidence / higher verification, additional errors caught per additional VERIFY call are in `marginal_verification_efficiency.csv`. From displayed 0.99 to 0.70: GPT MMLU catches 59 extra errors with 251 extra VERIFY calls (0.24 errors per added call); GPT code catches 63 extra errors with 111 extra calls (0.57 per added call; ~260 hidden-suite seconds on newly verified items). Claude-code adds only 14 VERIFY calls and 4 extra catches. No dollar conversion.

Optional matched-budget decomposition (not a headline): `matched_budget_leakage.csv`. Observed 0.70→0.99 leakage increases are typically slightly *smaller* than a random down-sample of the higher-coverage VERIFY set. That is a supporting allocation note, not a new headline.

## 13. GPT vs Claude

GPT shows large confidence-to-oversight gain on both tasks. Claude shows a large MMLU coverage/leakage response but a near-saturated VERIFY_FIRST policy on hard code (small coverage move). Do not say GPT is irrational or Claude is normatively correct/safer.

## 14. MMLU vs code

MMLU (N=500) and confirmatory hard LiveCodeBench (N=286) agree that GPT’s coverage, and therefore leakage, moves with displayed score. Code adds an external executable verifier and a model/task boundary: Claude-code saturation.

## 15. What Task 012 does NOT establish

- Exact ranking invariance, or that ranking is unchanged.
- That GPT is irrational or Claude is the rational/safe policy.
- That optimal coverage equals the empirical error rate.
- That provider guardrails cause the Claude-code ceiling.
- That constant-score overwrite equals ordinary calibration.
- That natural q1 drift would produce the same slope.
- Mechanisms, hidden confidence states, or “false confidence causes X.”

## 16. Strongest systems interpretation

A counterfactual confidence value with no item-discriminative information within a condition can change verification resource use and the number of erroneous outputs that escape independent checking, while the frozen output population stays fixed. The downstream cost of that sensitivity depends on λ. GPT exhibits substantially higher confidence-to-oversight gain than Claude’s saturated hard-code policy.

## 17. Paper-impact gate result

**MATERIAL_UPGRADE**

- A consequence (material leakage Δ in ≥2 cells): True
- B operating-range overlap: True
- C systems regret over a λ interval: True
- D no contradiction / reproduction: True

Material cells: mmlu/GPT Δ=+11.8pp, code/GPT Δ=+22.0pp.

## 18. Exact recommended abstract sentence

A counterfactual confidence value with no item-discriminative information within a condition can move verification coverage by tens of percentage points while the frozen output population and its correctness remain fixed; those coverage shifts translate into measurable changes in the number of erroneous outputs that escape independent checking, over a numerical range that overlaps the models’ naturally reported confidence.

## 19. Exact claims to avoid

- “false confidence causes X” unless the score is defined as false
- “miscalibration causes X”
- “natural q1 drift would cause exactly the same slope”
- “optimal coverage equals error rate”
- “Claude is more rational/safe”
- “GPT irrationally trusts confidence”
- “guardrails explain Claude”
- “confidence contains zero information” without the within-condition qualifier
- exact ranking invariance; GPT irrationality; Claude normative correctness

## 20. Whether paperDirection should be rewritten

Yes — write `paperDirection_task012_proposed.txt` as a full replacement; do not overwrite `paperDirection.txt`.

## 21. READY_FOR_GPT_REVIEW = YES
