# D1 Decision Log

Maintain this file as specified in `docs/D1_EXPERIMENTAL_PROTOCOL_v1.0_DRAFT.md` §69.

Every meaningful scientific change is appended. Do not rewrite historical entries if a later decision changes. Record:

- date
- previous plan / previous state
- evidence that motivated the change
- new decision
- whether the affected study had already been frozen
- whether existing data are exploratory or confirmatory relative to the new decision
- current next engineering task
- status labels

---

## 2026-09-17 — Proceed to a small exploratory causal confidence pilot

**Date:** 2026-09-17

**Previous state:**  
Original V2 and Checkpoint A complete.

**Previous plan:**  
The original frozen V2 protocol (`trust_check_v2`, freeze date 2026-09-04) treated decision authority as the primary question and confidence visibility as a mechanism question. After V2 data were observed, the workshop manuscript emphasized confidence visibility. Checkpoint A then re-analyzed the frozen V2-B data with no new API calls.

**Evidence:**  
Checkpoint A showed that visible GPT nearly follows its raw confidence threshold, much of its high-stakes failure is explained by overconfidence/reduced verification, Claude shows a clear equal-budget hidden-policy advantage, GPT and Claude hidden actions contain information beyond verbal confidence, and the proposed simple cross-model law failed.

**Decision:**  
Proceed to a small exploratory causal confidence pilot before any full expansion or mechanistic work.

**Affected study already frozen?**  
No. Historical V2-B remains HISTORICAL-FROZEN and is not reopened. Checkpoint A remains an exploratory re-analysis of that frozen data. The Study 1 causal pilot is not frozen as confirmatory. No Study 1 generations have been collected.

**Exploratory vs confirmatory relative to this decision:**  
Historical V2 and Checkpoint A are exploratory relative to the rebuilt causal-control hypothesis. Study 1 will also be EXPLORATORY. Prospective confirmation requires a later dated protocol addendum before fresh confirmatory generations.

**Current next engineering task:**  
Build and audit the Study-1 pilot infrastructure; do not launch the full pilot yet.

**Status:**  
Study 1 = EXPLORATORY.  
Later studies = CONDITIONAL.
