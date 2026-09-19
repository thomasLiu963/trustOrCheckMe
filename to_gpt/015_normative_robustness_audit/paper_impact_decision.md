# Task 015 paper-impact decision

Label: `POST_HOC_NORMATIVE_REANALYSIS`

## Decision

**NORMATIVE_RESCOPED_TO_INVARIANCE**

## Why

- Threshold algebra is `VERIFY iff r(Z) > λ` under Task-012 loss. No λ/(1+λ) misuse in prior D1 artifacts.
- Operating-point “no λ rationalizes” is false: 14 of 20 observed (task, model, delta) policies have a nonempty weak positive-λ interval.
- One fixed λ cannot make all five deltas simultaneously optimal. That is a different, near-tautological statement and is not Candidate 2.
- Transformation invariance of a delta-aware scalar policy is a theorem, not an empirical discovery.
- Empirical content: rank-preserving shifts produce material loss spread on at least one GPT cell (freeze rule: max spread ≥ 0.05 and ≥ 3 consecutive grid points with spread ≥ 0.02).
    - GPT MMLU-Pro: max=0.1020, mean=0.0542, material=True
  - Claude MMLU-Pro: max=0.2780, mean=0.0611, material=True
  - GPT LiveCodeBench: max=0.0594, mean=0.0491, material=True
  - Claude LiveCodeBench: max=0.0385, mean=0.0181, material=False

## Paper integration

Add one main-text section on calibration-shift robustness / transformation non-invariance. Use “invariance benchmark,” “pass-through regret,” and “fixed-cost robustness.” Do not say Bayes-dominated for every λ. Do not make Candidate 2 a main contribution.

## paperDirection

Rewrite authorized: proposed file written and live paperDirection updated with an invariance section. Control-surface + 012 stress test + 014 attenuation remain the spine.

## Experiments

None. Program closed.
