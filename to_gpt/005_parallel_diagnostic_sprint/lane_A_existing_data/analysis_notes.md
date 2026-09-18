# Lane A analysis notes

Pilot signal check on the frozen 100-question Study-1 sample. Not a publication-grade evidence-retention estimate. Not confirmatory.

## Units

Question identity is the statistical unit whenever correctness is involved. Repeated L/condition cells of the same question are not treated as independent questions. The 20-question repeat subset keeps three generations as repeated observations of those questions.

## Saturation

GPT manipulated cells labeled `ACTION_SATURATED__ITEM_DISCRIMINATION_NOT_IDENTIFIABLE`: **5** / 10.

A constant binary action at a fixed displayed score is **not** evidence that internal evidence was destroyed or suppressed. It means item discrimination is not identifiable from that binary action in that cell.

## Claude identifiable cells

10 of 10 Claude manipulated cells have both actions and both correctness classes represented enough to form a 2×2 table without a zero action margin.

## Incremental prediction

Grouped 5-fold CV by question ID. Baseline predicts Stage-1 correctness from the experimentally assigned displayed score. Augmented adds the VERIFY/USE action. Displayed score is assigned independently of correctness in manipulated cells, so the baseline is a near-null. If action carries useful item-specific error information, log-loss should fall and AUROC should rise. Complete separation is reported as unstable rather than forced.

## Routing baseline

Historical V2-B, 500 questions, GPT and Claude only. Hidden verification vs raw-q ranking vs cross-fitted isotonic-q ranking at the hidden policy's verification count. Ties use Checkpoint A fractional cutoff inclusion. Repeated L/owner cells are reported separately and are not treated as independent questions.

## Power simulation

Uses empirical Study-1 error rates and labeled hypothetical deltas. It does not declare a required N because an earlier critique proposed 600–1000. The table is the N → precision/power relationship.

## 20-question directional persistence

See `directional_persistence_20q.csv`. Claude's positive wrong-vs-correct verification gap remains positive in all 10 manipulated cells when replacing the primary action with the 3-generation mean. GPT's 20-question subset is saturated, so this check is mostly zeros.


