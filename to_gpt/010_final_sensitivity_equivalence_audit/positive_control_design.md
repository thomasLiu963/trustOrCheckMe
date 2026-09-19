# Task 010 positive-control design

Label: **Task-010 sensitivity analysis**, not Task-009 confirmation.

## Pipeline reconstructed from Task 009 H2

- Visible fixed scores only: 0.70, 0.85, 0.90, 0.95, 0.99
- N=500 questions × 5 conditions
- Shared model: condition intercepts + other-primary wrongness
- Richer model: shared + γ(displayed−0.90)×wrongness
- 5-fold GroupKFold by question ID
- LogisticRegression C∈{inf, 1e6, 1.0} as in Task 009
- Invariance fails if held-out log-loss improvement ≥ 0.01 **or** question-bootstrap 95% upper bound ≥ 0.01
- Power bootstrap resamples of frozen OOF predictions: 400 (Task 009 used 5000 on the real data; the detector is otherwise identical)

Design matrices and CV folds are computed once. Each Monte Carlo draw only resamples VERIFY and refits the same logits. That is computationally faster than rebuilding row dicts; it is not a more powerful detector.

## DGP

For each primary model, fit the Task-009 shared model in-sample. That fitted condition intercept + difficulty slope is the true-zero generator. Synthetic VERIFY draws are Bernoulli given that generator plus an injected extra.

## Family 1 — structured score × difficulty

Task-008 moderate logit-scale score×difficulty gamma (C=1 ridge, difficulty = other-models-correct 0–3). Claude γ=5.157 is the primary development anchor because it was the largest development interaction people treated as possible reshaping; GPT γ=0.316 is the secondary own-model anchor; |Claude−GPT|=4.840 is the between-model anchor. Task-009 codes difficulty as other-primary wrongness in {0,1} and interacts (score-0.90)×wrongness. Injections use that 009 coding; 008 γ is treated as the logit interaction magnitude on 009's binary wrongness (same units as a one-model easiness contrast, opposite sign to 008's other-correct coding). Primary 1C grid uses Claude 5.157.

Primary grid: multipliers [0, 0.25, 0.5, 1.0, 1.5, 2.0] × Claude-008 γ=5.157.

Because wrongness is binary, item ranking has two tied groups until the interaction flips their order. Adjacent Spearman therefore stays at 1.0 for multipliers ≤1.0 and jumps to 0.50 at 1.5×. Family 1 does not interpolate a “small Spearman disturbance”; the human-readable scale is the realized Spearman/reversal in `positive_control_power_curve.csv`.

## Family 2 — generic item-condition noise

Independent Gaussian perturbations on each item–condition logit.

A Pearson calibration `σ² = var(β·wrongness)·(1−ρ)/ρ` was used to target adjacent correlations 0.99, 0.95, 0.90, 0.80, 0.70, 0.60 on that binary latent. **Do not treat those ρ labels as realized Spearman.** Binary ties plus independent noise make Spearman much lower (about 0.55 GPT / 0.42 Claude across most of the grid). The human-readable scale is the realized mean adjacent Spearman and pairwise reversal in the power curve.

This family is still a valid positive control for “reranking that is not aligned with other-primary wrongness.” The Task-009 richer model is a score×difficulty detector, so low power here is expected and scientifically load-bearing.

## Monte Carlo

- Seed `20261010`
- 500 replications per cell
- True-zero cell: family 1, multiplier 0

Do not treat a more powerful detector than Task 009 as the paper's sensitivity.
