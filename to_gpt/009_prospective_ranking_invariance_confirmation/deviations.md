# Task 009 deviations

## 2026-09-18T13:34:15Z — Stage-3 runner interrupt and resume

The first paid process received SIGINT / `CancelledError` during primary GPT/Claude Stage-3 after Stage-1/q1 was complete (1000/1000 answers, 1000/1000 q1). Checkpointed verification cells at interrupt: 6493 success, 2 failed, 505 pending. No scientific keys were rewritten.

Engineering-only change after freeze: Stage-3 execution now gathers in chunks of 64 with progress logging so an interrupt cannot cancel thousands of in-flight coroutines at once. Prompts, IDs, endpoints, metrics, and retry policy are unchanged. The run is resumed from `CheckpointStore` reuse; retries of the 2 failed cells follow the predeclared retry_failed pass.

This is not a result-driven protocol change and was recorded before inspecting VERIFY rates.

## 2026-09-18T14:13:37Z — second SIGINT during secondary Stage-3

A second process interrupt (`exit 130`) hit while Gemini/Grok Stage-3 was running. At that point primary GPT/Claude Stage-1/q1, primary Stage-3, and GPT/Claude repeat extras were complete; secondary Stage-1/q1 was complete (200/200). Remaining work is checkpointed pending secondary verification cells. Resume again from `CheckpointStore`; no protocol fields changed.

## 2026-09-18T14:13:00Z — Gemini hang interrupt

Secondary Stage-3 stalled: SQLite still 620/2800 Gemini+Grok successes, last completion 14:05Z, process blocked on hung Vertex calls (`429 RESOURCE_EXHAUSTED`, truncated JSON, 89s calls, no client timeout). Primary GPT/Claude 7000/7000 and repeats 2800/2800 already checkpointed.

Engineering-only: 45s timeout on Gemini `_send`, treated as a transient retry. Resume from SQLite; completed cells are not redone.
