# Task 011 — Prospective code + executable-test generalization

## 1. Plain-English bottom line

On frozen LiveCodeBench programs, counterfactual displayed confidence shifted VERIFY_FIRST coverage by 38.8pp for GPT and 4.9pp for Claude.

Final bucket: **MODEL_SPECIFIC_GENERALIZATION — GPT coverage replicates; Claude saturates VERIFY**

## 2. Benchmark / version / validation

- Official LiveCodeBench `code_generation_lite` `release_v6`
- Repo commit `28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24`
- Eligible medium+hard pool: 723; confirmatory ladder: `hard_only` N=286
- Pilot 1 (burned, all-hard sampler bug) seed 20260921; pilot 2 seed 20260924 mixed 22 medium / 18 hard
- Hidden tests: official `check_correctness` after routing freeze
- paperDirection.txt was not modified (sha256 `8546e2e8002d9c96c6be7c946a4d63705068adda133b7a7c0c10bee2b8542b36`)

## 3. Pilot and predeclared difficulty adaptation

- Round-2 GO: GPT acc 87.5%; Claude acc 77.5%
- GPT >80% on mixed pilot → predeclared **hard_only** confirmatory
- Remaining hard after excluding both pilots: 286 (below 300; entire remaining hard pool used)

## 4. Freeze / preregistration integrity

- Main ID sha256 `254d4a628415fd790b8d883a186ddb92c702854da3cdf30c3baca8cf9babf955`
- Repeat ID sha256 `ddd080a66f8474d6fb6339accdec1b7ca3605945869f7bea82a39892448b8b99`
- Sample, prompts, endpoints, analyses, and decision rules frozen before confirmatory VERIFY-rate inspection
- Label: confirmatory

## 5. Exact API calls, retries, token use, estimated cost

- Scientific requests in sqlite: 7948 (cap includes two pilots + confirmatory)
- Provider attempts: 8109
- Failures remaining on confirmatory IDs: 0
- Estimated recorded USD: 39.6141

## 6. Frozen-code correctness rates

- GPT: 46.5% of 286 frozen programs passed all hidden tests
- Claude: 43.0% of 286 frozen programs passed all hidden tests

## 7. Coverage-control result

- GPT 0.70−0.99: 38.8pp [33.2, 44.8]
- Claude 0.70−0.99: 4.9pp [2.4, 7.7]
- Generalization label: **MODEL_SPECIFIC_GENERALIZATION**

## 8. Shared vs score-specific prioritization

- GPT held-out ΔLL: -0.0005151968009201591
- Claude held-out ΔLL: -0.0005023370525949339
- Materiality threshold: 0.01. This is **not** a ranking-invariance claim.

## 9. Matched-budget bug-routing

- Label: **ROUTING_NULL_OR_SMALL**
- Routers were predeclared: random, raw q1, calibrated q1, hidden, q1+hidden, oracle.

## 10. Real verifier resource curve

See `figures/code_bug_catch_budget.png` and `matched_budget_routing.csv`.

## 11. Repeat stability

Descriptive only. See `repeat_stability.csv`. Do not read this as latent invariance.

## 12. Hidden / true-q bridge

See coverage_response.csv rows `hidden` and `true_q_visible`.

## 13. GPT vs Claude

GPT coverage shift 38.8pp; Claude 4.9pp. Routing flags: {'openai_gpt56_sol': False, 'anthropic_sonnet5': False}.

## 14. MMLU vs code

MMLU Task 009: GPT 50.2pp, Claude 35.4pp, ΔLL ≈ 0.
Code Task 011: GPT 38.8pp, Claude 4.9pp, ΔLL GPT -0.0005151968009201591, Claude -0.0005023370525949339.

## 15. Evidence against the current D1 framing

Claude is near-saturated on VERIFY_FIRST for hard code (~95–100%), so the two-model MMLU coverage-control story does not transfer intact. GPT still shows a large 0.70→0.99 coverage shift. Allocation ΔLL is null.

## 16. Final WORLD classification

**MODEL_SPECIFIC_GENERALIZATION — GPT coverage replicates; Claude saturates VERIFY**

## 17. Strongest defensible paper claim

On frozen LiveCodeBench programs, counterfactual displayed confidence strongly changes GPT verification coverage; Claude remains near-always VERIFY_FIRST, so the MMLU coverage-control result does not replicate in both primary models.

## 18. Claims that must NOT be used

- The ranking is invariant.
- Confidence never affects ranking.
- The model secretly knows its code is wrong.
- Confidence is useless.
- Code proves a universal law.
- Self-verification (if implying the model runs the tests).
- Hidden uncertainty is suppressed.

## 19. Recommended main-paper figures

- MMLU vs code coverage (`figures/mmlu_vs_code_summary.png`)
- Code coverage response (`figures/code_coverage_response.png`)
- Bug-catch vs budget (`figures/code_bug_catch_budget.png`)
- Shared vs score-specific ΔLL (`figures/shared_vs_score_specific.png`)

## 20. Are any new experiments scientifically justified?

**No.** After Task 011, STOP new experiments unless a genuine evaluator/data/preregistration bug is found.

## 21. READY_FOR_GPT_REVIEW = YES
