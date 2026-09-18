# Task 009 — Prospective confirmation of confidence-as-coverage control

## 1. Plain-English bottom line

On this unseen confirmatory sample, counterfactual displayed confidence mostly changed how many frozen answers were sent for verification, not which ones.

## 2. Preregistration integrity

- Freeze file: `/Users/thomas/Desktop/trustOrCheckMe/to_gpt/009_prospective_ranking_invariance_confirmation/freeze_manifest.json`
- Primary 500 SHA256: `4cc2f1dce283499fe40049b73900fb7862d7e153d76d8b75590109c0ce12d99f`
- Secondary 200 SHA256: `9ffd1ca537d6f05e06a1c88e296a147dbede7a9382d052b6a60ea1156b5b441d`
- Repeat 100 SHA256: `61ca0b1e40931ff6d97dc111bdc2439c3b2dbc25a98e533cb209141bb6a66b9a`
- Sample, prompts, endpoints, metrics, and decision buckets were frozen before VERIFY-rate inspection.
- paperDirection.txt was not modified (sha256 `6496db58d51e9246f6ae7fca39104b39256888863022fff03966e0fa467c5bc0`).

## 3. Calls, failures, retries, cost

- Scientific requests registered: 15400 / cap 15400
- Provider attempts (including retries): 22117
- Failures remaining: 0
- Estimated Stage-3 USD (recorded): 20.8800

## 4. Coverage result (H1)

- GPT: 50.2pp [45.8, 54.6] (floor 20.0; meets=True)
- Claude: 35.4pp [31.2, 39.6] (floor 15.0; meets=True)

## 5. Shared-ranking vs condition-sensitive (H2)

- GPT: held-out improvement 0.0008 [-0.0011, 0.0026]; invariance pass=True
- Claude: held-out improvement -0.0000 [-0.0009, 0.0008]; invariance pass=True

## 6. Repeat rank stability (H3)

- GPT displayed_0.70->displayed_0.85: Spearman 0.477 [0.430, 0.523]; order-reversal 0.0
- GPT displayed_0.85->displayed_0.90: Spearman 0.737 [0.668, 0.805]; order-reversal 0.23529411764705882
- GPT displayed_0.90->displayed_0.95: Spearman 0.714 [0.631, 0.792]; order-reversal 0.3
- GPT displayed_0.95->displayed_0.99: Spearman 0.694 [0.604, 0.776]; order-reversal 0.043478260869565216
- Claude displayed_0.70->displayed_0.85: Spearman 0.553 [0.469, 0.628]; order-reversal 0.015151515151515152
- Claude displayed_0.85->displayed_0.90: Spearman 0.856 [0.801, 0.906]; order-reversal 0.25
- Claude displayed_0.90->displayed_0.95: Spearman 0.738 [0.687, 0.790]; order-reversal 0.0
- Claude displayed_0.95->displayed_0.99: Spearman 0.831 [0.785, 0.873]; order-reversal 0.06382978723404255

## 7. Discrimination stability (H4)

- GPT displayed_0.70: AUROC 0.685 (identifiable)
- GPT displayed_0.85: AUROC 0.669 (identifiable)
- GPT displayed_0.90: AUROC 0.629 (identifiable)
- GPT displayed_0.95: AUROC 0.624 (identifiable)
- GPT displayed_0.99: AUROC 0.574 (identifiable)
- Claude displayed_0.70: AUROC 0.550 (identifiable)
- Claude displayed_0.85: AUROC 0.602 (identifiable)
- Claude displayed_0.90: AUROC 0.612 (identifiable)
- Claude displayed_0.95: AUROC 0.669 (identifiable)
- Claude displayed_0.99: AUROC 0.680 (identifiable)

## 8. Nesting / reversals

See `nesting_reversals.csv`. Adjacent visible-score V→U is the coverage-shift pattern; U→V is the anti-nested residue.

## 9. Risk-coverage interpretation

Raw catch rates track coverage. Do not treat catch changes at different coverage as ranking changes. See `risk_coverage.csv` and `figures/risk_coverage.png`.

## 10. Hidden / true-q bridge

Hidden and true_q_visible are contextual bridges, not members of the fixed-score threshold family. See coverage_response.csv for those two cells.

## 11. Matched-budget routing

Consequence analysis only. Routers were predeclared: random, raw q1, hidden judgment, q1+hidden, hindsight oracle. See `matched_budget_routing.csv`.

## 12. Gemini / Grok secondary

Compatibility only; they do not redefine the GPT/Claude hypothesis. Gemini shows a smaller coverage shift (13.5pp on N=200). Grok is action-saturated (VERIFY ≈ 1) so item routing is not identifiable. See `secondary_models_summary.csv`.

## 13. Decision bucket

`PROSPECTIVE_INVARIANCE_SUPPORTED`

## 14. Which final-paper world won

ranking invariance (HOW MANY)

## 15. Narrative implication

The prospective data support treating displayed confidence as a coverage control on ranking-stable qualitative verification.

## 16. Whether to run a second task

Do not launch a second benchmark inside Task 009. Recommend a later non-MCQ generalization study.

## 17. Review flag

READY_FOR_GPT_REVIEW = YES

