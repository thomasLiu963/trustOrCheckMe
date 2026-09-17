# Study 1 Repeated-Sampling Stability

## 1. Preflight parse-repair audit

Zero-cost audit of Task 003 ran before any Task-004 API call. See `preflight_parse_repair_audit.md`.

- Distinct Task-003 parse-repair cells: 52
- GPT repairs: 0
- Claude repairs: 52
- Stop before paid calls: False
- Proceed: True

Repairs were Claude-only and not concentrated on a key confidence contrast. They are treated as an engineering diagnostic, not a reason to withhold the frozen repeat budget.

After the paid repeat run, Task 004 itself produced additional Claude parse repairs and zero GPT repairs. Those in-run repairs were also spread across hidden and manipulated conditions rather than pinned to one side of a threshold pair.

## 2. Data validation

- Repeat-extra successes: 1120 (target 1120)
- Repeat IDs: 20 (target 20)
- Cells with primary + two extras: 560/560
- Historical V2 unchanged: True
- Validation ok: True
- Frozen repeat IDs: mmlu_pro:test:1731, mmlu_pro:test:6452, mmlu_pro:test:429, mmlu_pro:test:3333, mmlu_pro:test:774, mmlu_pro:test:4568, mmlu_pro:test:9646, mmlu_pro:test:11802, mmlu_pro:test:10497, mmlu_pro:test:2776, mmlu_pro:test:2849, mmlu_pro:test:1065, mmlu_pro:test:4290, mmlu_pro:test:3888, mmlu_pro:test:471, mmlu_pro:test:11138, mmlu_pro:test:10164, mmlu_pro:test:4773, mmlu_pro:test:8662, mmlu_pro:test:10842
- Stopped reason: None

- No validation issues.

## 3. Bottom line in plain English

- How often does an identical prompt naturally flip its decision? GPT about 2.7% of the time; Claude about 3.4%.
- Does GPT's huge displayed-confidence effect survive reruns? **Yes.** Crossing the near threshold changed GPT's action about 99.2% of the time.
- Does Claude's smaller effect survive reruns? **The extreme contrast does; the local 0.89/0.91 and 0.94/0.96 contrasts do not.** Crossing the near threshold changed Claude's action about 5.0% of the time, against rerun noise of 3.4%. On L=10 the 3-generation mean even flips sign (−1.7pp).
- Is the manipulation effect larger than ordinary rerun noise? **For GPT, yes, by a wide margin (~99% vs ~3%).** For Claude, the near-threshold manipulation is comparable to rerun noise; the extreme 0.80/0.90 vs 0.99 contrast remains larger than rerun noise.
- Does GPT still look like a near-threshold policy? **Yes.** The cliff remains at 0.89→0.91 and 0.94→0.96. Exception: GPT L=20 has a 1.7pp uptick from 0.93 (98.3%) to 0.94 (100%) before the 98pp drop. That is a tiny non-monotonicity, not a missing step.
- Does Claude still look gradual? **Yes, with one local exception.** Claude L=10 ticks up 1.7pp from 0.89 to 0.91, then down at 0.99. Claude L=20 is monotonically decreasing.
- Were there meaningful condition-dependent parse issues? **No.** Task-003 repairs were Claude-only and not concentrated on a confidence contrast. Task-004 added 18 Claude parse-repair cells, again spread across conditions.
- Does Study 1 pass its final exploratory gate? **C. PASS CAUSAL BEHAVIORAL PHENOMENON.** Changing displayed confidence robustly and causally changes verification behavior in this controlled exploratory setup, beyond ordinary repeated-prompt variation. GPT's near-threshold effect is far larger than identical-prompt rerun noise. Claude's near-threshold contrast on this 20-question subset is small and not stable across generations; it does not itself beat rerun noise. Claude remains an informative weaker/gradual contrast, especially at the extreme scores.

## 4. GPT stability

- L=10 identical-prompt agreement: all-3-agree 97.1%; 2–1 split 2.9%; mean pairwise disagreement 1.9%
- L=20 identical-prompt agreement: all-3-agree 93.6%; 2–1 split 6.4%; mean pairwise disagreement 4.3%
- L=10 near-threshold (0.89 vs 0.91): primary-only 100.0pp (100.0% → 0.0%); 3-gen aggregated 100.0pp [100.0, 100.0]; direction consistent across generations: True
- L=20 near-threshold (0.94 vs 0.96): primary-only 100.0pp (100.0% → 0.0%); 3-gen aggregated 98.3pp [95.0, 100.0]; direction consistent across generations: True
- L=10 extreme (0.80 vs 0.99): primary-only 100.0pp (100.0% → 0.0%); 3-gen aggregated 100.0pp [100.0, 100.0]; direction consistent across generations: True
- L=20 extreme (0.90 vs 0.99): primary-only 100.0pp (100.0% → 0.0%); 3-gen aggregated 100.0pp [100.0, 100.0]; direction consistent across generations: True

## 5. Claude stability

- L=10 identical-prompt agreement: all-3-agree 95.0%; 2–1 split 5.0%; mean pairwise disagreement 3.3%
- L=20 identical-prompt agreement: all-3-agree 95.7%; 2–1 split 4.3%; mean pairwise disagreement 2.9%
- L=10 near-threshold (0.89 vs 0.91): primary-only 5.0pp (75.0% → 70.0%); 3-gen aggregated -1.7pp [-5.0, 0.0]; direction consistent across generations: False
- L=20 near-threshold (0.94 vs 0.96): primary-only 5.0pp (70.0% → 65.0%); 3-gen aggregated 5.0pp [0.0, 13.3]; direction consistent across generations: False
- L=10 extreme (0.80 vs 0.99): primary-only 20.0pp (80.0% → 60.0%); 3-gen aggregated 23.3pp [8.3, 41.7]; direction consistent across generations: True
- L=20 extreme (0.90 vs 0.99): primary-only 20.0pp (85.0% → 65.0%); 3-gen aggregated 25.0pp [8.3, 43.3]; direction consistent across generations: True

## 6. Threshold contrasts

Question-resampled 95% CIs. The 20 questions are the unit; the three generations are repeated observations of those questions.

- GPT L=10 0.89 vs 0.91: primary-only 100.0pp (100.0% → 0.0%); 3-gen aggregated 100.0pp [100.0, 100.0]; direction consistent across generations: True
- GPT L=20 0.94 vs 0.96: primary-only 100.0pp (100.0% → 0.0%); 3-gen aggregated 98.3pp [95.0, 100.0]; direction consistent across generations: True
- Claude L=10 0.89 vs 0.91: primary-only 5.0pp (75.0% → 70.0%); 3-gen aggregated -1.7pp [-5.0, 0.0]; direction consistent across generations: False
- Claude L=20 0.94 vs 0.96: primary-only 5.0pp (70.0% → 65.0%); 3-gen aggregated 5.0pp [0.0, 13.3]; direction consistent across generations: False
- GPT L=10 0.80 vs 0.99: primary-only 100.0pp (100.0% → 0.0%); 3-gen aggregated 100.0pp [100.0, 100.0]; direction consistent across generations: True
- GPT L=20 0.90 vs 0.99: primary-only 100.0pp (100.0% → 0.0%); 3-gen aggregated 100.0pp [100.0, 100.0]; direction consistent across generations: True
- Claude L=10 0.80 vs 0.99: primary-only 20.0pp (80.0% → 60.0%); 3-gen aggregated 23.3pp [8.3, 41.7]; direction consistent across generations: True
- Claude L=20 0.90 vs 0.99: primary-only 20.0pp (85.0% → 65.0%); 3-gen aggregated 25.0pp [8.3, 43.3]; direction consistent across generations: True

## 7. Manipulation effect versus rerun noise

Repeating the same prompt changed GPT's L=10 decision 1.4% of the time, whereas crossing the displayed-confidence threshold (0.89 vs 0.91) changed it 100.0% of the time.
Repeating the same prompt changed GPT's L=20 decision 3.9% of the time, whereas crossing the displayed-confidence threshold (0.94 vs 0.96) changed it 98.3% of the time.
Repeating the same prompt changed Claude's L=10 decision 3.9% of the time, whereas crossing the displayed-confidence threshold (0.89 vs 0.91) changed it 5.0% of the time.
Repeating the same prompt changed Claude's L=20 decision 2.9% of the time, whereas crossing the displayed-confidence threshold (0.94 vs 0.96) changed it 5.0% of the time.

These comparisons are paired at the question. Generations are not treated as independent questions.

## 8. Response-curve stability

Figures compare Task-003 n=100 primary, the same 20 questions' primary-only curve, and the 20-question three-generation mean. CIs resample questions, not generations.

- GPT L=10 3-gen rates: 0.8:100.0%, 0.88:100.0%, 0.89:100.0%, 0.91:0.0%, 0.99:0.0%
- GPT L=20 3-gen rates: 0.9:100.0%, 0.93:98.3%, 0.94:100.0%, 0.96:1.7%, 0.99:0.0%
- Claude L=10 3-gen rates: 0.8:83.3%, 0.88:75.0%, 0.89:71.7%, 0.91:73.3%, 0.99:60.0%
- Claude L=20 3-gen rates: 0.9:88.3%, 0.93:75.0%, 0.94:71.7%, 0.96:66.7%, 0.99:63.3%

- GPT aggregated upticks (exceptions): 1
- Claude aggregated upticks (exceptions): 1

## 9. What Study 1 now DOES establish

- In this frozen exploratory setup, independently resampling the same prompt does **not** wash out GPT's displayed-confidence effect.
- Changing the displayed confidence, holding the frozen answer and reported confidence fixed, changes GPT verification more than rerunning the identical prompt.
- The GPT trajectory remains a near-step around the mechanical threshold. Claude remains smaller and more gradual.
- The repeat subset was frozen before Task-003 outcomes were observed.

## 10. What Study 1 still DOES NOT establish

- Self-provenance / 'the model's own confidence'.
- Practical agent generalization beyond this Stage-3 JSON decision.
- A hidden-state mechanism or suppression of internal uncertainty.
- Novelty relative to all prior literature.
- A confirmatory, pre-registered estimate. This remains exploratory.
- That Claude's near-threshold effect exceeds ordinary rerun noise.

## 11. Final Study-1 gate

**C. PASS CAUSAL BEHAVIORAL PHENOMENON**

Changing displayed confidence robustly and causally changes verification behavior in this controlled exploratory setup, beyond ordinary repeated-prompt variation. GPT's near-threshold effect is far larger than identical-prompt rerun noise. Claude's near-threshold contrast on this 20-question subset is small and not stable across generations; it does not itself beat rerun noise. Claude remains an informative weaker/gradual contrast, especially at the extreme scores.

## 12. Exact numbers GPT should know

- Frozen repeat n = 20 questions; extra scientific calls = 1120
- GPT rerun-vs-primary (mean L=10/20): 2.68%
- GPT near-threshold action change (mean L=10/20): 99.17%
- Claude rerun-vs-primary (mean L=10/20): 3.39%
- Claude near-threshold action change (mean L=10/20): 5.00%
- GPT L=10 aggregated contrast: 100.0pp
- GPT L=20 aggregated contrast: 98.3pp
- Claude L=10 aggregated contrast: -1.7pp
- Claude L=20 aggregated contrast: 5.0pp
- Gate: C PASS CAUSAL BEHAVIORAL PHENOMENON
- Code commit: 9a8f78fee611d7363bbb7a880797d0428987b4bd

## 13. Cost and runtime

- New scientific calls: 1120
- Provider attempts: 1139
- Retries / extra attempts: 19
- Parse-repair cells: 18
- Input tokens: 628544
- Output tokens: 19363
- GPT cost: $0.977608
- Claude cost: $0.961914
- Total cost: $1.939522
- Wall-clock seconds: 513.630

