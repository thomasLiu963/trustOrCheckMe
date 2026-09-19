# Task 014 deviations

- Four nonzero-delta calls returned truncated JSON on the first pass. They were retried under the predeclared parse-repair / failed-request retry rule and all four succeeded. Final missing nonzero-delta records: 0.
- Scientific request counter ended at 6292 vs 6288 planned because those four retries were additional attempts after failed first registrations.
- No delta-grid changes, no new models, no new benchmarks, no hidden-test reruns.
- δ=0 was reused from existing `true_q_visible` responses after a byte-identical prompt check.
