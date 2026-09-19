# Paper integration packet (not a manuscript)

## A. Final one-sentence result

On frozen LiveCodeBench programs, counterfactual displayed confidence strongly changes GPT verification coverage; Claude remains near-always VERIFY_FIRST, so the MMLU coverage-control result does not replicate in both primary models.

## B. Abstract paragraph

Large language models can request costly independent verification before an output is used. We hold the output fixed and counterfactually vary only the confidence shown back to GPT and Claude. On MMLU-Pro, displayed confidence caused very large changes in verification coverage while a richer score-specific reprioritization model added essentially no held-out predictive value. We then test whether that coverage-control result survives when verification is a real operation: executing an official hidden test suite on frozen LiveCodeBench programs. In code, the 0.70 vs 0.99 coverage shift is 38.8pp for GPT and 4.9pp for Claude. Score-specific reprioritization ΔLL is GPT -0.0005151968009201591 and Claude -0.0005023370525949339. At matched verifier-call budgets, predeclared simple routers are summarized as ROUTING_NULL_OR_SMALL. Quantity and allocation remain empirically distinct: confidence can tune how much code is sent to tests; catching bugs at a budget is a separate routing problem.

## C. Introduction logic

1. Costly selective verification
2. Confidence as a routing input
3. Quantity vs allocation
4. Fixed-output causal intervention
5. MMLU-Pro Task 009
6. Real code-verifier Task 011
7. Design implication: coverage control is not bug-ranking

## D. Contributions

1. Causal coverage-control result on frozen MMLU answers
2. Prospective generalization to frozen executable code
3. Claim-disciplined shared vs score-specific comparison
4. Matched-budget bug-catching with a real verifier

## E. Main-text result sequence

Task 009 coverage + allocation null; Task 011 code coverage; resource curve; systems implication.

## F. Appendix

Pilot ladder, hard-only N=286, repeat stochasticity, hidden/true-q bridge.

## G. Reviewer attacks remaining

1. Hard-only confirmatory after mixed-pilot accuracy
2. N=286 vs target 500
3. Claude high VERIFY rates
4. Other-model-correctness as the item-risk feature
5. LiveCodeBench contamination window

## H. STOP recommendation

The scientific package is complete. STOP new experiments.
