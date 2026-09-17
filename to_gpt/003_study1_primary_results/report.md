# Study 1 Primary Causal Pilot

## 1. Data validation

Exactly 2800 unique successful primary cells are represented (target 2,800).
Per-model counts: {'openai_gpt56_sol': 1400, 'anthropic_sonnet5': 1400}. Per-L counts: {10.0: 1400, 20.0: 1400}.
Each of the seven condition slots has 400 cells (100 questions × 2 models × 2 L).
Task-002 reuse: 12 successful identical primary cells. New unique primary cells collected in Task 003: 2788.
One Claude L=10 `manipulated_2` cell (`mmlu_pro:test:9601`) returned truncated JSON, failed the standardized repair, and was rerun once on the same request key. That is not an extra scientific condition.
Historical V2 unchanged: True. Frozen ID hash `badd6938e5ded12c9dd62733426e1db26d9843bb6a2321a4e4c9eb7e3547fe94`.
All valid parsed actions are VERIFY_FIRST or USE_UNVERIFIED. Repeat-extra cells were not run.


## 2. Bottom line in plain English

- Changing only displayed confidence did change verification in this exploratory 100-question within-item pilot.
- GPT extreme low-vs-high contrasts were about 100.0 pp at L=10 and 98.0 pp at L=20.
- Claude extreme contrasts were about 36.0 pp at L=10 and 28.0 pp at L=20.
- Near-threshold GPT: GPT L=10 0.89 vs 0.91: 100.0% → 4.0% (Δ 96.0 pp, 95% [92.0, 99.0]); VERIFY→USE 96, USE→VERIFY 0, no change 4; GPT L=20 0.94 vs 0.96: 97.0% → 2.0% (Δ 95.0 pp, 95% [90.0, 99.0]); VERIFY→USE 95, USE→VERIFY 0, no change 5.
- Dose-response is described below; it is not forced to be monotonic. GPT non-increasing trajectories: 196/200; reverse flips: 4/200.
- GPT and Claude should be read as separate systems, not as replicates of one law.
- Hidden/true-visible versus historical V2 is a sanity check only; see section 8.
- GPT extreme-score flips involved up to 100 questions, so the GPT signal is not a one-item artifact if that count is large.
- Repeated-generation stability has not been tested. This result cannot claim the effect exceeds ordinary rerun noise.
- Gate recommendation: **C. PASS PRIMARY — run stability repeats next** Repeats are not launched by this task.

## 3. GPT results

L=10 manipulated curve: 0.8=100.0% [100.0, 100.0], 0.88=100.0% [100.0, 100.0], 0.89=100.0% [100.0, 100.0], 0.91=4.0% [1.0, 8.0], 0.99=0.0% [0.0, 0.0]; hidden 32.0%; true-visible 14.0%.
L=20 manipulated curve: 0.9=100.0% [100.0, 100.0], 0.93=97.0% [93.0, 100.0], 0.94=97.0% [93.0, 100.0], 0.96=2.0% [0.0, 5.0], 0.99=2.0% [0.0, 5.0]; hidden 33.0%; true-visible 25.0%.
Intervals are question-level bootstrap 95% percentile intervals (seed 20260917, 5,000 resamples).

## 4. Claude results

L=10 manipulated curve: 0.8=86.0% [79.0, 92.0], 0.88=80.0% [72.0, 87.0], 0.89=79.0% [71.0, 87.0], 0.91=72.0% [63.0, 80.0], 0.99=50.0% [40.0, 60.0]; hidden 57.0%; true-visible 84.0%.
L=20 manipulated curve: 0.9=90.0% [84.0, 95.0], 0.93=80.0% [72.0, 87.0], 0.94=77.0% [69.0, 85.0], 0.96=70.0% [61.0, 79.0], 0.99=62.0% [53.0, 71.0]; hidden 65.0%; true-visible 84.0%.
Claude is a contrast, not a required replication of GPT.

## 5. Near-threshold causal effects

- GPT L=10 0.89 vs 0.91: 100.0% → 4.0% (Δ 96.0 pp, 95% [92.0, 99.0]); VERIFY→USE 96, USE→VERIFY 0, no change 4
- GPT L=20 0.94 vs 0.96: 97.0% → 2.0% (Δ 95.0 pp, 95% [90.0, 99.0]); VERIFY→USE 95, USE→VERIFY 0, no change 5
- Claude L=10 0.89 vs 0.91: 79.0% → 72.0% (Δ 7.0 pp, 95% [2.0, 12.0]); VERIFY→USE 7, USE→VERIFY 0, no change 93
- Claude L=20 0.94 vs 0.96: 77.0% → 70.0% (Δ 7.0 pp, 95% [3.0, 12.0]); VERIFY→USE 7, USE→VERIFY 0, no change 93

These are paired within-question contrasts. They are not independent two-sample tests.

## 6. Extreme-score causal effects

- GPT L=10 0.8 vs 0.99: 100.0% → 0.0% (Δ 100.0 pp, 95% [100.0, 100.0]); VERIFY→USE 100, USE→VERIFY 0, no change 0
- GPT L=20 0.9 vs 0.99: 100.0% → 2.0% (Δ 98.0 pp, 95% [95.0, 100.0]); VERIFY→USE 98, USE→VERIFY 0, no change 2
- Claude L=10 0.8 vs 0.99: 86.0% → 50.0% (Δ 36.0 pp, 95% [27.0, 45.0]); VERIFY→USE 36, USE→VERIFY 0, no change 64
- Claude L=20 0.9 vs 0.99: 90.0% → 62.0% (Δ 28.0 pp, 95% [19.0, 37.0]); VERIFY→USE 28, USE→VERIFY 0, no change 72

## 7. Within-question trajectories

- GPT: 196/200 perfectly non-increasing; 4/200 contain a reverse USE→VERIFY as displayed confidence increases.
- Claude: 192/200 perfectly non-increasing; 8/200 contain a reverse flip.
- Binary single-generation trajectories can look like a threshold or like noise; this is descriptive only.

## 8. Current vs historical sanity check

- GPT L=10 hidden: historical 33.0% vs current 32.0% (Δ -1.0 pp; agree 93/100)
- GPT L=10 true_confidence_visible: historical 13.0% vs current 14.0% (Δ 1.0 pp; agree 99/100)
- GPT L=20 hidden: historical 38.0% vs current 33.0% (Δ -5.0 pp; agree 89/100)
- GPT L=20 true_confidence_visible: historical 25.0% vs current 25.0% (Δ 0.0 pp; agree 98/100)
- Claude L=10 hidden: historical 56.0% vs current 57.0% (Δ 1.0 pp; agree 99/100)
- Claude L=10 true_confidence_visible: historical 82.0% vs current 84.0% (Δ 2.0 pp; agree 98/100)
- Claude L=20 hidden: historical 64.0% vs current 65.0% (Δ 1.0 pp; agree 95/100)
- Claude L=20 true_confidence_visible: historical 87.0% vs current 84.0% (Δ -3.0 pp; agree 97/100)

Disagreement here is compatible with ordinary stochastic reruns and does not by itself prove the endpoint identity changed.

## 9. Secondary correctness-conditioned analysis

These splits are exploratory and noisy. They do not replace the all-question causal estimates.
- GPT L=10 correct extreme Δ 100.0 pp [100.0, 100.0] n=81
- GPT L=10 incorrect extreme Δ 100.0 pp [100.0, 100.0] n=19
- GPT L=20 correct extreme Δ 100.0 pp [100.0, 100.0] n=81
- GPT L=20 incorrect extreme Δ 89.5 pp [73.7, 100.0] n=19
- Claude L=10 correct extreme Δ 43.8 pp [32.9, 54.8] n=73
- Claude L=10 incorrect extreme Δ 14.8 pp [3.7, 29.6] n=27
- Claude L=20 correct extreme Δ 35.6 pp [24.7, 46.6] n=73
- Claude L=20 incorrect extreme Δ 7.4 pp [0.0, 18.5] n=27

## 10. What this DOES establish

In this exploratory 100-question AI-authority pilot, holding question and frozen answer fixed, the displayed confidence number was the only intended prompt change across manipulated cells. Where verification rates moved with that number, the movement is a causal effect of the displayed number in this experimental setup. Repeated-sampling stability is still unknown.

## 11. What this DOES NOT establish

- This is not confirmatory evidence.
- It does not establish that “own” provenance matters.
- It does not establish real-world or agent generalization.
- It does not establish a hidden-state mechanism, nor that showing confidence suppresses internal uncertainty.
- It does not establish that the model loses information.
- It does not establish that the effect exceeds ordinary rerun noise until repeats are run.

## 12. Study-1 gate recommendation

**C. PASS PRIMARY — run stability repeats next**

GPT shows a large within-question displayed-confidence effect in the expected direction, it is not confined to a handful of items, and Claude is at least an informative contrast. Repeats are still required before claiming the effect exceeds ordinary rerun noise.

Protocol criterion 4 (effect larger than ordinary repeated-sampling noise) is **not evaluated**, because the 1,120 repeat-extra cells were not run. The strongest claim permitted here is STRONG_BEHAVIORAL_SIGNAL_PENDING_STABILITY if the primary contrasts are large; this task still does not launch repeats.

## 13. Exact numbers GPT should know

| contrast | GPT Δ pp [95%] | Claude Δ pp [95%] |
|---|---:|---:|
| L=10 0.89 vs 0.91 | 96.0 [92.0, 99.0] | 7.0 [2.0, 12.0] |
| L=20 0.94 vs 0.96 | 95.0 [90.0, 99.0] | 7.0 [3.0, 12.0] |
| L=10 0.8 vs 0.99 | 100.0 [100.0, 100.0] | 36.0 [27.0, 45.0] |
| L=20 0.9 vs 0.99 | 98.0 [95.0, 100.0] | 28.0 [19.0, 37.0] |

## 14. Cost and runtime

- Reused Task-002 cells: 12
- New unique Task-003 scientific cells: 2788
- Failed cells remaining: 0
- Provider attempts in Study-1 sqlite (includes 12 smoke attempts): 2853
- Extra attempts beyond 2,800 successful cells: 53
- Distinct cells with a parse-repair attempt: 52
- Input tokens: 1604920
- Output tokens: 48538
- GPT estimated USD: 2.502800
- Claude estimated USD: 2.443820
- Total estimated USD: 4.946620
- Wall-clock seconds (Task 003 invocations combined): 1240.682
- Summed provider latency seconds: 4635.136
- Stopped reason: None
