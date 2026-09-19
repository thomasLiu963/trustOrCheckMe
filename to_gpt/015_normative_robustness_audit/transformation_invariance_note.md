# Transformation-invariance proposition

Label: `POST_HOC_NORMATIVE_REANALYSIS`.

## Setup

Let Q be the scalar q1. For each frozen delta,

    S_δ = T_δ(Q) = sigmoid(logit(Q) + δ)

on the interior (0,1), with exact 0/1 endpoints held fixed (Task 014 rule).

On the interior, T_δ is a known strictly increasing bijection. Up to the recorded ties and six exact-zero endpoints,

    σ(S_δ) = σ(Q).

The pair (correctness Y, scalar information in Q) is therefore the same for every delta.

## Benchmark property

A **delta-aware** cost-sensitive policy that uses only this scalar information and a fixed λ implements

    VERIFY iff r̂(Q) > λ

or equivalently r̂(T_δ^{-1}(S_δ)) > λ.

That action set does **not** depend on δ.

In words: a known invertible reparameterization of the same scalar information does not change the underlying cost-sensitive decision problem *for a policy that knows the reparameterization*.

This is a property of the **benchmark**, labeled `Q1_ONLY_CALIBRATED_BASELINE`.

## What the deployed router actually sees

The LLM router is not told δ. It is shown S_δ as if it were a correctness probability.

If its VERIFY rate and leakage change with δ, the *composed system* (display + black-box router + fixed λ) is **not invariant** to rank-preserving level shifts.

That is a robustness fact about the composed policy.

## What non-invariance does not prove

- The model is irrational.
- The model should have inferred the hidden transform.
- A naturally generated probability shift must be ignored.
- Ground-truth oracle actions are attainable.
- No λ rationalizes any single observed operating point.

Allowed wording if the empirical audit supports it:

> The composed verification policy is not invariant to rank-preserving confidence-level shifts; the same scalar information can induce materially different oversight costs under fixed system objectives.
