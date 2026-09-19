# Task 010 final claim audit

## A. Positive-control sensitivity

Template justified: **WEAK**

False-positive rate at zero injection (full 009 rule): GPT 0.000; Claude 0.000.

What Task 009 could reliably detect:

- GPT family 1: 80% power at 1.5× Claude-008 γ (γ≈7.73), already a large ranking change (adjacent Spearman 0.50, 25% pairwise reversal). At the 1.0× development anchor, power is only 0.214 and Spearman is still 1.00.
- Claude family 1: 80% power **not reached**. At 2.0×, power=0.518 with the same Spearman-0.50 rewrite.
- Family 2 generic noise: 80% power **not reached**; power stayed 0.000 even with realized Spearman ~0.4–0.55.

Intuitive ranking terms: the confirmatory H2 test is not a small-reranking detector. It is, at best, a detector of a **large two-group flip** along other-primary wrongness, and only reliably so for GPT.

## B. AUROC drift

GPT's decreasing and Claude's increasing binary-action AUROC are **directionally** predicted by the shared-ranking / moving-threshold model. They are **not** quantitatively predicted. RMSE=0.077. Residuals are all positive and several exceed 0.05 (GPT +0.093 at 0.70; Claude +0.111 at 0.99). Treat residual condition dependence as real; do not say AUROC drift is fully mechanical.

## C. Latent rank stability

MAP hierarchical Spearman is ~0.999, which would be `LATENT_STABILITY_STRONG` if taken literally. The posterior-predictive check rejects that reading: several observed 3-draw Spearmans are far above what the fitted latent propensities generate (GPT 0.85→0.90: raw 0.857 vs PP 0.544). Item×condition variance is not identified with N=100×3. Paper-facing label: **INSUFFICIENT_REPEAT_INFORMATION**. The raw GPT 0.70→0.85 Spearman (0.541) is not, by itself, evidence of ranking collapse; it is also not evidence of latent identity.

## D. Targeted mismatch rivals

Status: POST_HOC_ROBUSTNESS — NOT PREREGISTERED TASK-009 CONFIRMATION.

Material 0.01 hits: signed and abs mismatch for both models; Claude logit mismatch. Score×difficulty remains null (GPT +0.0008, Claude ~0).

Interpretation: `displayed − q1` is collinear with q1 once condition intercepts are in the model, because displayed score is constant within condition. The hits say own q1 predicts VERIFY, not that displayed confidence reorders items. Narrow invariance wording to the preregistered score×difficulty alternative. Do not upgrade mismatch to confirmatory.

## E. Routing consequence

Label: **MODEST_ROUTING_GAIN**. Main figure is not justified; a paragraph or a descriptive operating-point plot is enough. Hidden/q1+hidden beats q1 by ≥10pp at 30–40% budget for GPT only; Claude shows no such gain. Cross-model difficulty remains evaluation-only.

## F. Final recommended wording

1. Headline: On confirmatory MMLU-Pro data, displayed confidence moves verification coverage and large ranking rewrites of the preregistered score×difficulty form are not supported, but the H2 test is not sensitive enough to rule out smaller reranking.
2. Abstract-level: Displayed confidence strongly shifts verification coverage, while our data do not support large score-dependent reranking. The study is not sufficiently sensitive to rule out smaller ranking changes.
3. Conservative reviewer-proof: Displayed confidence strongly shifts verification coverage, while our data do not support large score-dependent reranking. The study is not sufficiently sensitive to rule out smaller ranking changes.
4. Forbidden stronger phrasings:
- Displayed confidence never changes which items are verified.
- Ranking is proven identical across scores.
- The richer model is exactly equivalent to the shared model.
- AUROC changes prove (or disprove) ranking invariance by themselves.
- Cross-model difficulty is a deployable production router.
- Hidden-state uncertainty is being crowded out.
- Task 010 was preregistered as part of Task 009.
- Mismatch tests are confirmatory.
- Three Bernoulli repeats prove latent ranking identity.
- Generic reranking was tested and rejected by H2.

MMLU analysis should now stop: YES — stop further MMLU-Pro mining unless a concrete statistical error in 009/010 is found.
