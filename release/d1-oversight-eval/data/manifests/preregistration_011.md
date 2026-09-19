# Task 011 preregistration — prospective code verification generalization

Frozen at `2026-09-18T19:12:27.902757+00:00` before confirmatory VERIFY-rate inspection.
Commit: `f783a230eb5263499cdf2c06ba0a5033d947d9ab`.
Label: **confirmatory**.

## Sample

- Dataset: LiveCodeBench `livecodebench/code_generation_lite` `release_v6`
- Repo commit: `28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24`
- Difficulty ladder: `hard_only`
- Excluded pilot N=80 sha256 `c8861b386f745b1c65f7f2c078e81c97bdf9c3b05f7d8658660511be0f61c687`
- Main N=286 seed `20260922` sha256 `254d4a628415fd790b8d883a186ddb92c702854da3cdf30c3baca8cf9babf955`
- Repeat N=100 seed `20260923` sha256 `ddd080a66f8474d6fb6339accdec1b7ca3605945869f7bea82a39892448b8b99`

## Models

- GPT `gpt-5.6-sol`
- Claude `claude-sonnet-5`

## Protocol

- Stage 1: one frozen Python solution per model/problem. Malformed extraction is frozen empty and scored incorrect.
- Stage 2: q1 = P(this exact frozen code passes the independent hidden test suite).
- Stage 3: Task-009 moderate qualitative routing adapted to code. Actions USE_UNVERIFIED / VERIFY_FIRST.
- Conditions: hidden, true_q_visible, displayed_0.70, displayed_0.85, displayed_0.90, displayed_0.95, displayed_0.99.
- Repeats: N=100, two extra Stage-3 draws on the five fixed displayed scores only.
- Hidden official tests run only after all confirmatory routing is checkpointed.
- Engineering concurrency (not a scientific factor): code 24 global / 12 per provider; routing 24; eval workers 16. Item pipeline: code → q1 → Stage-3.

## Prompt hashes

{
  "code": "4a088335da2ded619bef255358d8b963015670af7ac02f30fd10e6498da7e4cd",
  "q1": "5d7e443a0ff8cec5ea5000387912e9f22166a6263f7d2872c03013e79f3bd1e0",
  "stage3_hidden": "81ed6473b202cffdd7e76890c589e2a7aef246ab175db56fdef18f6cf0148f11",
  "stage3_displayed_0.9": "070619b75ec0533ce075f39d674615536007962338c85127bfb4bddddcd3984f",
  "family": "moderate",
  "bundle": "4f01c905f2a15265115b46b34dd3e9fdafdda67a2e82f99b0f0307454c3e8afb"
}
