# Decision-theory derivation (frozen before empirical Task 015 results)

Label: `POST_HOC_NORMATIVE_REANALYSIS`.

## Loss

Task-012 / Task-014 convention:

- escaped-error cost = 1
- verification cost = λ
- verification is treated as perfect for this normalized analysis

Per item, with Y = 1[incorrect] and V = 1[VERIFY_FIRST]:

    ℓ = Y(1 − V) + λ V

So:

- USE_UNVERIFIED (V=0): ℓ = Y
- VERIFY_FIRST (V=1): ℓ = λ

Population loss is therefore

    E[ℓ] = P(Y=1 and V=0) + λ P(V=1)
         = leakage + λ · coverage

This is exactly the Task-012 normalized loss.

## Optimal action given information Z

Let r(Z) = P(Y=1 | Z). Then:

- expected loss of USE = r(Z)
- expected loss of VERIFY = λ

Loss-minimizing rule:

    VERIFY iff r(Z) > λ

(Indifferent if r(Z) = λ.)

There is **no** factor of the form λ/(1+λ) under this convention.

## Where λ/(1+λ) would have come from

If someone parameterized “verification cost / (verification cost + escaped-error cost)” as a *share*, or used a different loss such as

    ℓ = Y(1−V) + c V   with a probability-threshold rewrite under a 0-1 plus cost-of-checking model that absorbs a different scaling,

a threshold λ/(1+λ) can appear when λ is defined as a *ratio of utilities in a different parameterization* (for example, verify iff r > C/(C+L) when C is check cost and L is error cost, which is the same rule as VERIFY iff r > λ after setting λ = C/L).

Under the Task-012 definition **λ already is C/L** (check cost in escaped-error units). The threshold is λ, not λ/(1+λ).

Using λ/(1+λ) while claiming the Task-012 loss is a **notation mismatch**.

## Audit of prior D1 artifacts

Searched paperDirection, Tasks 012–014 reports/code, and analysis markdown for `λ/(1+λ)` and `lambda/(1+lambda)` as an *applied verification threshold under the Task-012 loss*.

**No such misuse was found.** Task 012/014 implement `loss = leakage + λ * coverage` and do not threshold at λ/(1+λ).

## What this rule is not

- It is not a license to plug in ground-truth Y and call the result an attainable Bayes policy.
- It is not a claim that the LLM router observes r(Z).
- It does not say the model is irrational if it does not implement VERIFY iff r > λ.
