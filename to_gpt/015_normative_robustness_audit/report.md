# Task 015 — Normative robustness audit

Label: `POST_HOC_NORMATIVE_REANALYSIS`. Zero new model calls. Frozen Tasks 009–014 only.

## 1. Plain-English bottom line

Claim gate: **NORMATIVE_RESCOPED_TO_INVARIANCE**.

Claude's strong Candidate-2 wording — that no verification/error cost ratio λ rationalizes the observed responsiveness, read as “no observed Task-014 operating point is best among the five for any λ” — is **false**. 14 of 20 observed (task, model, delta) policies have a nonempty positive-λ interval. GPT MMLU: all five. GPT code: only δ=0 and δ=+1.5 (coverage is non-monotone; δ=0 already has the lowest leakage). That still falsifies the global “no operating point is rationalizable” claim.

The defensible upgrade is narrower: a delta-aware cost-sensitive policy that uses only the scalar information in q1 is invariant to the Task-014 transform, but the deployed composed router is not. Rank-preserving level shifts move normalized system loss by a material amount on GPT, especially MMLU.

This is a systems-robustness fact, not a proof that the model is irrational.

## 2. Correct derivation of optimal verification threshold

Under the Task-012 loss `ℓ = Y(1−V) + λV`, so `E[ℓ] = leakage + λ·coverage`, the loss-minimizing action given information Z is

    VERIFY iff r(Z) = P(Y=1 | Z) > λ.

See `decision_theory_derivation.md`. The threshold is λ, not λ/(1+λ).

## 3. Whether λ/(1+λ) was a notation error under Task-012 convention

**No applied error found.** Searched paperDirection and Tasks 012–014 analysis/code. The only mentions of λ/(1+λ) are in this Task-015 specification, as a warning. Existing 012/014 code implements `loss = leakage + λ * coverage` and does not threshold at λ/(1+λ).

## 4. Ground-truth oracle vs q1-only baseline vs delta=0 router

| Label | Information | Attainable? | Role |
|---|---|---|---|
| `GROUND_TRUTH_ORACLE` | per-item Y | no | lower bound only; unused for rationality claims |
| `Q1_ONLY_CALIBRATED_BASELINE` | OOF r̂(q1) | in principle, as a scalar policy | delta-invariant benchmark |
| `UNSHIFTED_ROUTER_BASELINE` | actual δ=0 / true-q-visible actions | yes; it is the deployed unshifted policy | pass-through comparator |

The LLM router also sees question/output text. Beating or losing to the q1-only baseline is therefore not a rationality verdict.

## 5. Transformation-invariance proposition

On the interior, `q_δ = sigmoid(logit(q1)+δ)` is a known strictly monotone bijection, so σ(q_δ)=σ(q1) (Task 014: Spearman 1.0; six exact-zero endpoints stay 0). A **delta-aware** policy `VERIFY iff r̂(q1)>λ` does not depend on δ. The deployed router is not told δ. Non-invariance is a composed-system robustness failure. See `transformation_invariance_note.md`.

## 6. Observed-policy λ rationalization intervals

Exact intervals for “this observed delta policy minimizes L among the five Task-014 deltas.”

### GPT MMLU
- δ=-1.5: cov=0.230, leak=0.090, Pareto-dominated=False, envelope=True, weak interval `(0, 0.27027)`, unique `(0, 0.27027)`
- δ=-0.75: cov=0.156, leak=0.110, Pareto-dominated=False, envelope=True, weak interval `(0.27027, 0.3125)`, unique `(0.27027, 0.3125)`
- δ=0: cov=0.124, leak=0.120, Pareto-dominated=False, envelope=True, weak interval `(0.3125, 0.5)`, unique `(0.3125, 0.5)`
- δ=0.75: cov=0.080, leak=0.142, Pareto-dominated=False, envelope=True, weak interval `(0.5, 0.666667)`, unique `(0.5, 0.666667)`
- δ=1.5: cov=0.068, leak=0.150, Pareto-dominated=False, envelope=True, weak interval `(0.666667, inf)`, unique `(0.666667, inf)`

### GPT code
- δ=-1.5: cov=0.262, leak=0.311, Pareto-dominated=False, envelope=False, weak interval `none`, unique `none`
- δ=-0.75: cov=0.245, leak=0.322, Pareto-dominated=False, envelope=False, weak interval `none`, unique `none`
- δ=0: cov=0.273, leak=0.297, Pareto-dominated=False, envelope=True, weak interval `(0, 0.73913)`, unique `(0, 0.73913)`
- δ=0.75: cov=0.196, leak=0.357, Pareto-dominated=True, envelope=False, weak interval `none`, unique `none`
- δ=1.5: cov=0.192, leak=0.357, Pareto-dominated=False, envelope=True, weak interval `(0.73913, inf)`, unique `(0.73913, inf)`

### Claude MMLU
- δ=-1.5: cov=0.916, leak=0.000, Pareto-dominated=False, envelope=True, weak interval `(0, 0.0625)`, unique `(0, 0.0625)`
- δ=-0.75: cov=0.852, leak=0.004, Pareto-dominated=False, envelope=True, weak interval `(0.0625, 0.0921053)`, unique `(0.0625, 0.0921053)`
- δ=0: cov=0.794, leak=0.010, Pareto-dominated=False, envelope=False, weak interval `none`, unique `none`
- δ=0.75: cov=0.700, leak=0.018, Pareto-dominated=False, envelope=True, weak interval `(0.0921053, 0.166667)`, unique `(0.0921053, 0.166667)`
- δ=1.5: cov=0.604, leak=0.034, Pareto-dominated=False, envelope=True, weak interval `(0.166667, inf)`, unique `(0.166667, inf)`

### Claude code
- δ=-1.5: cov=0.997, leak=0.000, Pareto-dominated=False, envelope=True, weak interval `(0, 0.25)`, unique `(0, 0.25)`
- δ=-0.75: cov=0.983, leak=0.003, Pareto-dominated=False, envelope=True, weak interval `(0.25, 0.272727)`, unique `(0.25, 0.272727)`
- δ=0: cov=0.979, leak=0.014, Pareto-dominated=True, envelope=False, weak interval `none`, unique `none`
- δ=0.75: cov=0.955, leak=0.017, Pareto-dominated=True, envelope=False, weak interval `none`, unique `none`
- δ=1.5: cov=0.944, leak=0.014, Pareto-dominated=False, envelope=True, weak interval `(0.272727, inf)`, unique `(0.272727, inf)`

A single fixed λ that makes **all five** observed policies simultaneously loss-minimizing: **does not exist** in any cell (the five points are not iso-loss). That question is different from operating-point rationalization and must not be conflated with it.

## 7. Is Claude's exact “no λ rationalizes” statement true or false?

**False** under the operating-point definition frozen in `task015_analysis_freeze.json`:
“no observed Task-014 delta policy is loss-minimizing among the five on any positive-λ interval.”

Pessimistic (low) deltas are typically best when verification is cheap; optimistic (high) deltas are typically best when verification is expensive. That is ordinary cost-sensitive selection among observed operating points.

**Also false as an extra theorem beyond invariance** if “no λ rationalizes the *responsiveness*” is taken to mean “a delta-aware Bayes policy would not produce a coverage-vs-delta slope.” That statement is true but tautological: the invariant benchmark is flat in δ by construction. It is the invariance claim, not a separate over-responsiveness theorem.

The one-fixed-λ-for-the-entire-curve question is true in the trivial sense that five distinct losses cannot all be optimal at once. Do not promote that tautology to “the router is over-responsive for every λ.”

## 8. Rank-preserving loss spread across λ

How much normalized system loss can change solely because the displayed confidence level is shifted while outputs, labels, and q1 ordering stay fixed.

- GPT MMLU: max spread 0.1020; mean 0.0542; consecutive grid points with spread≥0.02: 10; material=True
- GPT code: max spread 0.0594; mean 0.0491; consecutive grid points with spread≥0.02: 10; material=True
- Claude MMLU: max spread 0.2780; mean 0.0611; consecutive grid points with spread≥0.02: 5; material=True
- Claude code: max spread 0.0385; mean 0.0181; consecutive grid points with spread≥0.02: 1; material=False

No single λ is declared realistic. See `rank_preserving_loss_spread.csv` and `figures/rank_preserving_loss_spread.png`.

## 9. Delta=0 pass-through regret

`R_{δ0} = L_δ − L_0`. Sign flips with λ: making the router more pessimistic reduces leakage and helps when λ is small; it hurts when verification is expensive. Positive regret is **not** universal harm.

See `delta0_pass_through_regret.csv` and `figures/delta0_regret.png`.

## 10. Q1-only calibrated invariant baseline

5-fold OOF isotonic regression of error on q1 (seed 20260928); logistic sensitivity. No routing actions enter the fit. The policy `VERIFY iff r̂(q1)>λ` is the same at every delta.

If q1 is weakly discriminative (mass near 0.99), r̂ is nearly constant and the baseline collapses toward always-verify or never-verify according as error_rate ≷ λ. On GPT MMLU the isotonic r̂ is ≈0.17, so the baseline is verify-all for λ<0.17 and verify-none afterward. That scalar policy can have lower Task-012 loss than the observed mixed router; this uses the label-derived error rate and is not a rationality verdict. It also does not restore invariance of the composed display+router map.

See `q1_invariant_baseline.csv` and `figures/observed_vs_invariant_baseline.png`.

## 11. GPT MMLU

Coverage 23.0% → 6.8% and leakage 9.0% → 15.0% from δ=−1.5 to +1.5 (frozen 014). The five points are increasing in leakage as coverage falls; they are Pareto-undominated and each is best observed on a λ interval. Loss spread is material under the freeze rule. Unmanipulated escape: 70.6% of incorrect answers left unverified (95% CI [61.1%, 80.0%]).

## 12. GPT code

Coverage 26.2% → 19.2% and leakage 31.1% → 35.7%. δ=0 has the lowest leakage (29.7%) among the five; δ=+0.75 is weakly dominated by δ=+1.5 (same leakage, slightly higher coverage). The implied unmanipulated escape rate is 55.6% (leakage 0.297 / error 0.535 = 0.556; 95% CI [47.2%, 63.5%]), matching the ~55% figure in the handoff.

## 13. Claude MMLU

Large coverage move (91.6% → 60.4%) with leakage 0 → 3.4%. High-coverage deltas minimize loss at small λ; lower-coverage deltas win at large λ. Material spread. Do not translate this into a model-quality ranking.

## 14. Claude code

Near VERIFY ceiling. Small coverage/leakage moves. Small loss spreads are a saturation fact, not normative robustness and not evidence against GPT.

## 15. Unmanipulated conditional escape rate

`P(USE_UNVERIFIED | incorrect) = leakage / error_rate` at δ=0 / true-q-visible.

| Cell | Error rate | Leakage | Escape | 95% CI |
|---|---|---|---|---|
| GPT MMLU | 17.0% | 12.0% | 70.6% | [61.1%, 80.0%] |
| GPT code | 53.5% | 29.7% | 55.6% | [47.2%, 63.5%] |
| Claude MMLU | 24.2% | 1.0% | 4.1% | [0.9%, 8.1%] |
| Claude code | 57.0% | 1.4% | 2.5% | [0.6%, 5.0%] |

Preferred language: “In the unmanipulated q1-visible condition, about 55.6% of incorrect programs are left unverified.”

Do **not** call this a real-world status quo, deployment failure rate, or industry defect escape rate.

## 16. Whether “over-responsiveness” is defensible

**No, not as a headline.** The phrase blurs (i) ordinary movement along a coverage/leakage frontier, (ii) the tautology that a delta-aware Bayes policy is flat in δ, and (iii) a claim that no λ rationalizes an operating point — which is false. Do not use “over-responsive” or “no λ rationalizes” prominently.

## 17. Whether “transformation non-invariance” is defensible

**Yes**, as a composed-system statement with the qualifications in `transformation_invariance_note.md`. GPT MMLU and GPT code (and Claude MMLU) show material loss variation under the predeclared freeze rule. This is the safe conceptual upgrade.

## 18. Strongest exact paper claim

The composed verification policy is not invariant to rank-preserving confidence-level shifts; the same scalar information can induce materially different oversight costs under fixed system objectives.

## 19. Claims to avoid

- the model is irrational / not Bayes
- no λ rationalizes the observed operating point
- Bayes-dominated for every λ
- the q1-only baseline is the Bayes-optimal router
- the ground-truth oracle is an attainable policy
- λ/(1+λ) is the Task-012 threshold
- over-responsiveness as a novel generic metric (Kumaran et al. already study confidence-response sensitivity)
- unmanipulated escape as an industry defect rate
- Task 015 justifies a new experiment

## 20. Paper-impact decision

**NORMATIVE_RESCOPED_TO_INVARIANCE**. See `paper_impact_decision.md`.

## 21. Whether paperDirection should be rewritten

Yes: add one main-text calibration-shift / invariance section. Do not replace the control-surface story or promote Candidate 2.

## 22. Whether any new experiment is justified

**No.** Task 015 is zero-call and cannot trigger new data collection. The experimental program remains closed.

## 23. READY_FOR_GPT_REVIEW = YES

Generated 2026-09-19T06:23:37.842117+00:00.
