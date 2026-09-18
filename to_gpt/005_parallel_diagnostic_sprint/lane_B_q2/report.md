# Lane B — independent second-confidence (q2)

- Scientific successes: 1000 / 1000
- Failures: 0
- New calls: 1000
- Provider attempts: 1000
- Parse repairs: 0
- Estimated USD: $1.383
- Interpretation buckets: GPT `B-PASS-DEEPER-RESIDUAL` (AUROC CI just includes 0; log-loss CI excludes 0); Claude `B-PASS-DEEPER-RESIDUAL` (both CIs exclude 0)

Primary question: after q1 and q2, does the historical hidden verification judgment still add useful correctness information?

**Short answer:** yes, as an exploratory residual. q2 does not absorb it.

| features | GPT AUROC | GPT log-loss | Claude AUROC | Claude log-loss |
|---|---:|---:|---:|---:|
| q1 | 0.640 | 0.451 | 0.723 | 0.509 |
| q1+q2 | 0.641 | 0.452 | 0.730 | 0.501 |
| q1+q2+hidden fraction | 0.722 | 0.410 | 0.775 | 0.468 |

Hidden after q1+q2: GPT ΔAUROC +0.081 [−0.002, +0.163], Δlog-loss −0.042 [−0.067, −0.019]. Claude ΔAUROC +0.045 [+0.016, +0.074], Δlog-loss −0.032 [−0.055, −0.009].

corr(q1,q2): GPT 0.61, Claude 0.94.

Do not treat this as a hidden-state mechanism result. Do not auto-scale.


