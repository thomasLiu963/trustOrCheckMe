# Proposed paperDirection.txt patch (NOT applied)

Do not edit paperDirection.txt until this patch is accepted.

Replace section 49 "WHAT REMAINS..." remaining-priority language with:

CURRENT REMAINING SCIENTIFIC PRIORITY:

    none. Task 011 tested the same coverage-control claim on executable
    LiveCodeBench code with an official hidden-test verifier.

TASK-011 RESULT:

    Frozen-code confirmatory N=286 hard-only LiveCodeBench.
    Coverage shift 0.70→0.99: GPT 38.8pp, Claude 4.9pp.
    Score-specific ΔLL: GPT -0.0005151968009201591, Claude -0.0005023370525949339.
    Matched-budget routing: ROUTING_NULL_OR_SMALL.
    World: MODEL_SPECIFIC_GENERALIZATION — GPT coverage replicates; Claude saturates VERIFY.

CURRENT CORE PAPER CLAIM:

    On frozen LiveCodeBench programs, counterfactual displayed confidence strongly changes GPT verification coverage; Claude remains near-always VERIFY_FIRST, so the MMLU coverage-control result does not replicate in both primary models.

FINAL WRITING RULE is unchanged: STOP new experiments; write the paper.
