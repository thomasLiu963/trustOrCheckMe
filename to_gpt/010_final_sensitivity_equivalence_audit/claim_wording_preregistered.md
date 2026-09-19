# Task 010 — predeclared claim-wording templates

Frozen **before** positive-control simulations, AUROC residual inspection, latent-model fitting, mismatch robustness, and the Task-010 matched-budget recomputation.

These templates are not Task-009 preregistration. They are Task-010 sensitivity labels. After the simulation, only the numerical floor / human-readable effect size may be inserted. The logical category cannot be changed to chase a stronger paper sentence.

Question-ID is the resampling unit throughout. Confirmatory Task-009 evidence remains the H2 held-out log-loss test on real data. Task-010 does not redefine that test.

## Detection-floor categories

A **small scientifically meaningful disturbance** is defined here as injected reranking that induces either:

- latent rank correlation across adjacent visible scores of about **0.90 or lower**, or
- pairwise ordering reversal of about **5% or more**.

**Moderate** reranking: latent rank correlation about **0.80**, or reversals about **10%**.

**Large** reranking: latent rank correlation about **0.70 or lower**, or reversals about **20% or more**.

Power is the Monte Carlo probability that the **exact Task-009 H2 pipeline** fails invariance, i.e. that held-out log-loss improvement of the richer model is **not** both `< 0.01` and with question-bootstrap 95% upper bound `< 0.01`.

Use the **smallest** category whose 80% detection floor is actually achieved for **both** GPT and Claude under the more conservative of the two injection families (structured score×difficulty vs generic item noise). If the two models disagree, choose the weaker shared template and say so.

## TIGHT DETECTION FLOOR

Use if the existing Task-009 design has ≥80% power to detect reranking that would induce only a small scientifically meaningful disturbance, as defined above.

Template:

> Displayed confidence strongly shifts verification coverage, while any score-dependent reranking large enough to exceed [DETECTION FLOOR] is inconsistent with the observed data under our sensitivity model.

## MODERATE DETECTION FLOOR

Use if the design reliably detects moderate reranking but not small reranking.

Template:

> Displayed confidence strongly shifts verification coverage. We find no evidence for score-dependent reranking of moderate or larger magnitude; smaller changes remain below the study's detection floor.

## WEAK DETECTION FLOOR

Use if the design only reliably detects large reranking.

Template:

> Displayed confidence strongly shifts verification coverage, while our data do not support large score-dependent reranking. The study is not sufficiently sensitive to rule out smaller ranking changes.

## Routing figure decision (also frozen here)

Descriptive only, not a preregistered hypothesis.

- `MATERIAL_ROUTING_GAIN`: ≥10 percentage-point error-catch gain over the strongest realistic baseline (not random, not hindsight oracle) at 20–40% budget, replicated in both GPT and Claude.
- `MODEST_ROUTING_GAIN`: 3–10pp, or ≥10pp in only one model.
- `NO_MATERIAL_ROUTING_GAIN`: <3pp / CIs broadly overlap.

Realistic baselines: raw q1, cross-fitted calibrated q1, hidden judgment, q1+hidden. Cross-model difficulty is evaluation-only and is not allowed to decide MATERIAL.

## Development anchors (frozen for family 1)

Primary: Task-008 moderate Claude logit γ = 5.157.
Secondary: Task-008 moderate GPT logit γ = 0.316.
Between-model: |5.157 − 0.316| = 4.840.

Injection grid: 0, 0.25×, 0.5×, 1.0×, 1.5×, 2.0× the primary Claude anchor, applied as γ × (displayed − 0.90) × other-primary-wrongness in the Task-009 H2 DGP.
