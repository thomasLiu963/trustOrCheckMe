# Task 014 — Rank-preserving confidence-level shift

## 1. Plain-English bottom line

World: **LEVEL_SENSITIVITY_PARTIAL**. Item-level q1 information attenuates but does not remove confidence-level sensitivity.

## 2. Exact model endpoints

- GPT: `gpt-5.6-sol`
- Claude: `claude-sonnet-5`
Same requested endpoints as Tasks 009/011.

## 3. Preregistration/freeze integrity

See `preregistration.md` and `analysis_freeze.json`. Freeze timestamp precedes new routing-outcome inspection.

## 4. δ=0 reuse decision

**REUSE_TRUE_Q_VISIBLE**. All 1572 δ=0 prompts were byte-identical to existing `true_q_visible` prompts.

## 5. Transformation audit

See `transformation_audit.md`. Spearman(q1, q_delta) = 1.0. Zero order reversals after `.12g` rendering. Six exact-zero q1 values (code) stay at 0 for every delta.

## 6. Order-preservation audit

Pairwise order preserved. New ties only from rendering/endpoints, quantified in `transformation_audit.csv`.

## 7. Prompt-integrity audit

Prompts match the 009/011 q-visible templates except the displayed number. Models were not told the score was transformed.

## 8. API calls / retries / cost

- Planned new scientific calls: 6288
- Missing nonzero-delta records: 0
- Estimated new-call USD: 17.9035

## 9. Coverage by delta

VERIFY_FIRST rate at δ = −1.5 / −0.75 / 0 / +0.75 / +1.5:

- MMLU GPT: 23.0 / 15.6 / 12.4 / 8.0 / 6.8
- MMLU Claude: 91.6 / 85.2 / 79.4 / 70.0 / 60.4
- Code GPT: 26.2 / 24.5 / 27.3 / 19.6 / 19.2
- Code Claude: 99.7 / 98.3 / 97.9 / 95.5 / 94.4

## 10. Leakage by delta

Unverified-error rate at the same grid:

- MMLU GPT: 9.0 / 11.0 / 12.0 / 14.2 / 15.0
- MMLU Claude: 0.0 / 0.4 / 1.0 / 1.8 / 3.4
- Code GPT: 31.1 / 32.2 / 29.7 / 35.7 / 35.7
- Code Claude: 0.0 / 0.3 / 1.4 / 1.7 / 1.4

## 11. GPT MMLU primary result

coverage 23.0% → 6.8% (Δ -16.2 [-19.6, -13.0]); leakage 9.0% → 15.0% (Δ +6.0 [+4.0, +8.2]); retention leak 0.51, cov 0.32

## 12. GPT code primary result

coverage 26.2% → 19.2% (Δ -7.0 [-10.1, -3.8]); leakage 31.1% → 35.7% (Δ +4.5 [+2.1, +7.3]); retention leak 0.21, cov 0.18

## 13. Claude MMLU result

coverage 91.6% → 60.4% (Δ -31.2 [-35.4, -27.0]); leakage 0.0% → 3.4% (Δ +3.4 [+1.8, +5.0]); retention leak 0.89, cov 0.88

## 14. Claude code saturation result

coverage 99.7% → 94.4% (Δ -5.2 [-8.0, -2.8]); leakage 0.0% → 1.4% (Δ +1.4 [+0.3, +2.8]); retention leak 1.00, cov 1.07

A small Claude-code effect is not evidence against the GPT result. Do not call Claude robust or invoke guardrails.

## 15. Constant-vs-offset effect retention

See `effect_retention.csv`.

## 16. Calibration/discrimination diagnostics

The monotone transform changes level/calibration and preserves q1 item ordering/AUROC up to ties. It does not preserve “all information.”

## 17. Cost/regret sweep

`loss(λ) = leakage + λ * coverage` on the Task-012 λ grid. See `cost_sweep.csv`. Do not cherry-pick λ.

## 18. Marginal verification efficiency

See `marginal_efficiency.csv`. Moving from δ=+1.5 toward δ=−1.5.

## 19. WORLD classification

**LEVEL_SENSITIVITY_PARTIAL**

## 20. Evidence against the current Task-012 framing

Task 012 remains the constant-score stress test. Task 014 tests whether level sensitivity survives when q1 ordering is preserved. World LEVEL_SENSITIVITY_PARTIAL determines whether 012 is an extreme case, a general level effect, or a scope correction.

## 21. Strongest defensible paper claim

Item-level q1 information attenuates but does not remove confidence-level sensitivity.

## 22. Claims that must NOT be used

- ordinary production calibration drift necessarily causes identical effects
- transformed scores are naturally generated
- ranking is perfectly informative
- q1 contains all relevant uncertainty
- Claude is normatively robust
- guardrails cause saturation
- hide-confidence is optimal (Task 013 already rejected that as a headline)

## 23. Paper integration recommendation

**MODERATE_UPGRADE**

## 24. Whether paperDirection should be rewritten

Yes — write paperDirection_task014_proposed.txt as an update, not a replacement of the 012 leakage story.

## 25. Are any further experiments scientifically justified?

No. Task 014 is the final scientific experiment. Next: write the paper.

## 26. READY_FOR_GPT_REVIEW = YES
