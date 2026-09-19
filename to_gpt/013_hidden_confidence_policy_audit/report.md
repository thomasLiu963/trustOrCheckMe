# Task 013 — Hidden-confidence policy audit

Label: `POST_HOC_POLICY_REANALYSIS`. Zero API calls. Frozen Tasks 009–012 only.

## 1. Plain-English bottom line

Withholding the displayed confidence scalar is **not** a robust default across both GPT tasks under the predeclared gates.

GPT MMLU hidden is near-best on 80% of the λ grid but has worst-case regret 0.214 when verification is expensive (it verifies much more than q=0.99). GPT code hidden is near-best on only 20% of the grid and has worst-case regret 0.150 when verification is cheap (it leaks more than q=0.70). Hidden sits on the observed Pareto frontier in both GPT cells, but that is not enough for the robustness gate.

Cross-task class: **HIDDEN_NOT_ROBUST**. Paper-impact: **DIAGNOSTIC_ONLY**.

Hidden is an observed policy, not an x-value and not a preregistered mitigation. It can have lower leakage than true-q-visible while costing more verification, or the reverse. Do not call it optimal.

## 2. Reproduction check

See `reproduction_check.md`. Status: **PASS**.

Published hidden / true-q coverage and Task 012 leakage, λ grid, and loss function all reproduced within ±0.6 pp.

## 3. Hidden vs true-q-visible raw metrics

| Task | Model | Hidden cov % | True-q cov % | Δ cov pp [95% CI] | Hidden leak % | True-q leak % | Δ leak pp [95% CI] |
|---|---|---|---|---|---|---|---|
| mmlu | GPT | 39.6 | 12.4 | +27.2 [+23.2, +31.2] | 3.8 | 12.0 | -8.2 [-10.8, -6.0] |
| mmlu | Claude | 65.4 | 79.4 | -14.0 [-17.0, -10.8] | 2.8 | 1.0 | +1.8 [+0.8, +3.0] |
| code | GPT | 32.5 | 27.3 | +5.2 [+1.7, +9.1] | 25.9 | 29.7 | -3.8 [-7.0, -0.7] |
| code | Claude | 98.6 | 97.9 | +0.7 [-1.4, +2.8] | 0.3 | 1.4 | -1.0 [-2.8, +0.3] |

Lower leakage is not “better” without a λ. Hidden vs true-q can trade coverage against escaped errors.

## 4. Hidden vs fixed-score metrics

Coverage % / unverified errors per 100:

| Task | Model | hidden | q=0.70 | q=0.99 |
|---|---|---|---|---|
| mmlu | GPT | 39.6/3.8 | 58.4/2.0 | 8.2/13.8 |
| mmlu | Claude | 65.4/2.8 | 92.4/0.0 | 57.0/3.8 |
| code | GPT | 32.5/25.9 | 62.9/10.8 | 24.1/32.9 |
| code | Claude | 98.6/0.3 | 99.7/0.0 | 94.8/1.4 |

Full table: `policy_metrics.csv`.

## 5. Hidden loss/regret across λ

`loss(λ) = leakage + λ * coverage` on the frozen Task-012 grid. Best observed is only best among the seven collected policies.

| Task | Model | Exactly best | Within 0.02 | Within 0.05 | Worst-case regret | Mean regret | GPT-robust? |
|---|---|---|---|---|---|---|---|
| mmlu | GPT | 20% | 80% | 80% | 0.214 | 0.036 | no |
| mmlu | Claude | 0% | 30% | 90% | 0.074 | 0.026 | no |
| code | GPT | 0% | 20% | 20% | 0.150 | 0.112 | no |
| code | Claude | 0% | 90% | 100% | 0.028 | 0.006 | no |

## 6. Near-best breadth

Hidden is within 0.02 of best on 80% of the λ grid for GPT MMLU and 20% for GPT code. Claude MMLU 30%; Claude code 90% (near-saturation).

## 7. Minimax regret result

| Task | Model | Minimax policy | That worst regret | Hidden worst regret | Hidden is minimax? | Hidden on Pareto? |
|---|---|---|---|---|---|---|
| mmlu | GPT | displayed_0.90 | 0.082 | 0.214 | no | yes |
| mmlu | Claude | displayed_0.99 | 0.038 | 0.074 | no | no |
| code | GPT | displayed_0.85 | 0.080 | 0.150 | no | yes |
| code | Claude | displayed_0.99 | 0.014 | 0.028 | no | yes |

Minimax is among observed policies only.

## 8. Pareto-frontier result

A policy is efficient if no other observed policy has both lower coverage and lower leakage. Hidden on GPT MMLU: yes. Hidden on GPT code: yes. See `figures/policy_pareto_frontier.png`.

## 9. GPT MMLU result

Hidden coverage 39.6%, leakage 3.8% (19 unverified errors / 500). True-q coverage 12.4%, leakage 12.0%. Worst-case hidden regret 0.214; within 0.02 on 80% of λ. GPT-robust criteria: fail.

## 10. GPT code result

Hidden coverage 32.5%, leakage 25.9%. True-q coverage 27.3%, leakage 29.7%. Worst-case hidden regret 0.150; within 0.02 on 20% of λ. GPT-robust criteria: fail.

## 11. Claude MMLU result

Hidden coverage 65.4%, leakage 2.8%; true-q 79.4% / 1.0%. Hidden verifies less than true-q and leaks slightly more. Worst-case regret 0.074. Secondary for the mitigation gate.

## 12. Claude code saturation result

Hidden 98.6% coverage, leakage 0.3%; true-q 97.9% / 1.4%. Near VERIFY-all. Hidden cannot be judged as a general mitigation in this regime. Reported fully; not used to manufacture a win.

## 13. Cross-task hidden robustness classification

**HIDDEN_NOT_ROBUST**

Claude is secondary because code is action-saturated. Claude rows remain in every table.

## 14. Systems design matrix

| Task | Model | Brier | ECE | Cov sens pp/+0.10 | Hidden near-best 0.02 | Hidden max regret | Hidden Pareto |
|---|---|---|---|---|---|---|---|
| mmlu | GPT | 0.168 | 0.152 | -17.3 | 80% | 0.214 | yes |
| mmlu | Claude | 0.159 | 0.034 | -12.2 | 30% | 0.074 | no |
| code | GPT | 0.379 | 0.377 | -13.4 | 20% | 0.150 | yes |
| code | Claude | 0.227 | 0.166 | -1.7 | 90% | 0.028 | yes |

Confidence quality and confidence-to-policy gain are separate system properties. Whether to expose the scalar is a third, optional design choice — only if the policy audit supports it.

## 15. What hidden policy does NOT establish

- Optimality of hiding confidence
- That hiding is always safer
- That hidden routing removes internal confidence or equals item-only reasoning
- That visible confidence causes failures
- That Claude is normatively more robust
- That true-q-visible is generally bad
- That λ is known in deployment
- That this mitigation was preregistered or confirmatory

## 16. Strongest actionable claim, if any

Diagnosing confidence-metadata sensitivity does not imply that simply withholding the scalar is a universally good mitigation among the observed policies.

## 17. Paper-impact gate outcome

**DIAGNOSTIC_ONLY**

- A both GPT robust: False
- B Pareto both GPT: True
- C near-best both GPT: False
- D max regret ≤ 0.10 both GPT: False
- E honest reporting / full Claude: True

## 18. Exact recommended paper sentence

Diagnosing confidence-metadata sensitivity does not imply that simply withholding the scalar is a universally good mitigation among the observed policies.

## 19. Exact claims to avoid

- hidden confidence is optimal
- hiding confidence is always safer
- hiding confidence removes internal confidence
- hidden routing is equivalent to item-only reasoning
- visible confidence causes failures
- Claude is more robust in a normative sense
- true-q-visible is bad in general
- confidence should never be shown
- λ is known in deployment
- the hidden policy is preregistered as a mitigation
- Task 013 was confirmatory

## 20. Whether paperDirection should be rewritten

No. Keep the Task-012 paperDirection as source of truth.

## 21. READY_FOR_GPT_REVIEW = YES
