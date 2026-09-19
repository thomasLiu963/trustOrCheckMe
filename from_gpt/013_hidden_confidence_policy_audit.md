# TASK 013 — Hidden-Confidence Mitigation Audit
## Zero-call test of whether withholding confidence is a robust routing baseline

### Mission

This is a **zero-new-model-call** reanalysis.

The goal is to answer one concrete design question using only frozen Tasks 009–012 data:

> **If an LLM routing policy is highly sensitive to confidence metadata, is simply withholding the scalar a competitive and robust baseline?**

This is NOT a new model experiment.

This is NOT a claim that hidden confidence is universally better.

This is a post-hoc policy audit using existing `hidden`, `true_q_visible`, and fixed-score routing conditions that were already collected prospectively.

The scientific value of this task is potential **actionability**.

Right now D1 shows:
- confidence metadata can have very large causal leverage over verification coverage;
- that leverage can translate into substantial error leakage;
- GPT code is a particularly high-sensitivity regime;
- Claude code is near VERIFY saturation.

What is still missing is a simple lever a system designer can actually pull.

The candidate lever is:

> **Do not expose the routing model to the scalar confidence value.**

Task 013 asks whether the already-existing `hidden` condition is a good **robust baseline** across plausible verification/error tradeoffs.

If yes, the paper gains a simple design recommendation.

If not, the current Task-012 framing remains intact and this task stays secondary.

---

# 0. HARD RULES

1. **ZERO API calls.**
2. **ZERO new model generations.**
3. Use only frozen existing data.
4. Do not modify Tasks 009–012.
5. Label Task 013:
   `POST_HOC_POLICY_REANALYSIS`.
6. Do not call hidden-confidence routing “optimal” unless mathematically established under a stated loss and restricted policy set.
7. Do not claim hidden routing is universally safer.
8. Do not claim hidden routing removes confidence internally.
9. Do not claim the model “ignores confidence” in the hidden condition.
10. Do not claim Claude is better because it verifies more.
11. Do not cherry-pick one λ after inspecting results.
12. Do not redefine the λ grid after looking at outcomes.
13. Do not add new router families.
14. Do not add new benchmarks.
15. Do not create an intervention that was not actually run.
16. If hidden performs poorly, report that.
17. If this task contradicts the current paper framing, flag it rather than suppressing it.
18. If this task does not materially improve the paper, DO NOT rewrite paperDirection.

---

# 1. INPUTS

Read:

- current `paperDirection.txt` or latest Task-012 proposed paperDirection
- Task 009 report/preregistration
- Task 011 report/preregistration/deviations
- Task 012 report and all generated CSVs
- raw or merged Stage-3 data needed to reconstruct:
  - `hidden`
  - `true_q_visible`
  - `displayed_0.70`
  - `displayed_0.85`
  - `displayed_0.90`
  - `displayed_0.95`
  - `displayed_0.99`
- correctness labels
- q1 values
- test outcomes for code

Primary task/model cells:

1. MMLU / GPT
2. MMLU / Claude
3. Code / GPT
4. Code / Claude

Use confirmatory IDs only.

Exclude:
- pilots
- development samples
- exploratory-only samples
- malformed rows unless prior protocol explicitly includes them

---

# 2. REPRODUCTION CHECK

Before any new analysis, reproduce from frozen data:

## MMLU
GPT:
- hidden coverage = 39.6%
- true_q_visible coverage = 12.4%

Claude:
- hidden coverage = 65.4%
- true_q_visible coverage = 79.4%

## Code
GPT:
- hidden coverage = 32.5%
- true_q_visible coverage = 27.3%

Claude:
- hidden coverage = 98.6%
- true_q_visible coverage = 97.9%

Also reproduce Task 012:
- leakage by fixed score
- correctness
- λ-grid definition
- cost function

If any fail beyond rounding tolerance:

STOP with:

`TASK013_REPRODUCTION_FAILURE`

Write:

`reproduction_check.md`

---

# 3. PRIMARY POLICY COMPARISON

For each task/model/policy condition, compute:

- N
- verification coverage
- incorrect count
- incorrect verified
- incorrect unverified
- leakage rate:
  `P(incorrect AND USE_UNVERIFIED)`
- error catch rate:
  `P(VERIFY | incorrect)`
- verification precision:
  `P(incorrect | VERIFY)`
- errors caught per 100 outputs
- errors left unverified per 100 outputs

Policies:

1. `hidden`
2. `true_q_visible`
3. `displayed_0.70`
4. `displayed_0.85`
5. `displayed_0.90`
6. `displayed_0.95`
7. `displayed_0.99`

Use paired item bootstrap CIs where meaningful.

Create:

`policy_metrics.csv`

---

# 4. PRIMARY QUESTION A — IS HIDDEN A COMPETITIVE POLICY?

Use the same normalized loss as Task 012:

`loss(λ) = leakage_rate + λ * verification_coverage`

Use the **exact same λ grid** as Task 012.

For each task/model:

- compute loss for every policy
- identify the best observed policy at each λ
- compute hidden regret:

`hidden_regret(λ) = loss_hidden(λ) - min_observed_policy loss(λ)`

Also compute:
- true_q_visible regret
- regret for each fixed-score policy

Do NOT treat the best observed policy as globally optimal.

This is only:

> best among the observed policy conditions.

Create:

`policy_cost_sweep.csv`

---

# 5. ROBUSTNESS METRICS FOR HIDDEN

For each task/model, summarize hidden policy robustness over the λ grid.

Compute:

## A. Best-policy frequency

Fraction of λ grid where hidden is:
- exactly best
- within 0.01 normalized loss of best
- within 0.02
- within 0.05

## B. Worst-case regret

`max_λ hidden_regret(λ)`

## C. Mean regret

Average hidden regret over λ grid.

Report both:
- unweighted average over the frozen grid
- log-λ weighted/integrated version if straightforward

Do not invent a prior over λ.

## D. Regret-area / integrated regret

If technically clean:
- approximate area under regret-vs-logλ curve

Use descriptively only.

## E. Dominance check

Does hidden strictly dominate any observed policy?

Policy A strictly dominates B if:
- leakage_A <= leakage_B
- coverage_A <= coverage_B
- and at least one strict inequality

Because lower coverage is lower cost only if verification has positive cost.

If hidden dominates:
- report exact dominated policies.

If not:
- do not force a Pareto claim.

Create:

`hidden_robustness_summary.csv`

---

# 6. CROSS-TASK ROBUSTNESS

The paper-level recommendation cannot rest on one cell only.

Evaluate hidden across all four task/model cells.

Define the following frozen labels:

## `HIDDEN_ROBUST_BOTH_GPT_TASKS`

Criteria:
- In both GPT MMLU and GPT code:
  - hidden is within 0.02 normalized loss of the best observed policy for at least 50% of the λ grid;
  - hidden worst-case regret <= 0.10;
  - hidden does not create >10pp more leakage than true_q_visible at any broad λ-independent descriptive comparison.

This is a designer-facing robustness criterion, not a proof of optimality.

## `HIDDEN_ROBUST_ONE_TASK`

Criteria:
- passes above for exactly one GPT task.

## `HIDDEN_NOT_ROBUST`

Criteria:
- fails on both GPT tasks.

## Claude

Claude is secondary for the mitigation gate because:
- code is near action saturation;
- hidden cannot be fairly judged as a general mitigation in an already-verify-all regime.

Still report Claude fully.

Do NOT exclude Claude from tables.

---

# 7. PRIMARY QUESTION B — DOES HIDING CONFIDENCE REDUCE LEAKAGE RELATIVE TO TRUE-q VISIBLE?

This is a direct observed-policy contrast.

For each task/model:

Compare:

`hidden` vs `true_q_visible`

Compute:
- Δ coverage
- Δ leakage
- Δ error catch
- Δ normalized loss across λ

Use paired item bootstrap 95% CIs for:
- coverage difference
- leakage difference

Report exact counts too.

Important:

A hidden policy can have:
- lower leakage but higher verification cost;
- higher leakage but lower verification cost.

Therefore do not call lower leakage “better” without λ qualification.

Create:

`hidden_vs_trueq.csv`

---

# 8. PRIMARY QUESTION C — DOES HIDDEN PROVIDE A ROBUST DEFAULT UNDER UNCERTAIN λ?

This is the most policy-relevant question.

Do NOT assume a prior over λ.

Instead use three robust-choice diagnostics:

## Diagnostic 1 — Minimax regret

For each task/model/policy:

`max_λ regret(policy, λ)`

Identify the observed policy with minimum worst-case regret.

Question:
- Is hidden the minimax-regret policy among observed conditions?

## Diagnostic 2 — Near-best breadth

Fraction of λ grid where each policy is within:
- 0.01
- 0.02
- 0.05
of best observed.

Question:
- Does hidden remain near-best over a broader range than true_q_visible or fixed-score policies?

## Diagnostic 3 — Pareto frontier

Plot policies in:
- x = coverage
- y = leakage

A policy is Pareto-efficient if no other observed policy has both lower coverage and lower leakage.

Question:
- Is hidden on the Pareto frontier?

Create:

`robust_policy_selection.csv`

and:

`figures/policy_pareto_frontier.png`

---

# 9. FIGURE — HIDDEN AS A POLICY, NOT AN X-VALUE

Important visualization rule:

Hidden has no displayed confidence x-coordinate.

Do NOT place hidden at x=0.5 or mean q1.

Recommended figure:

## Panel A
Fixed-score curve:
- x = displayed confidence
- y = coverage

Add hidden and true_q_visible as labeled side markers or horizontal callouts.

## Panel B
Coverage vs leakage policy frontier:
- each policy is a point
- hidden labeled prominently
- true_q_visible labeled
- fixed-score policies labeled by q

## Panel C
Regret vs λ:
- hidden
- true_q_visible
- best one or two fixed-score policies
- optional envelope of best observed policy

Avoid spaghetti.

Create:

`figures/hidden_policy_summary.png`

---

# 10. MAIN-TEXT CANDIDATE RESULT

If hidden performs well, the paper-facing result should be framed carefully.

Potential allowed sentence:

> “In post-hoc policy analysis, withholding the scalar confidence signal is a competitive observed routing baseline across a broad range of verification/error cost ratios for GPT, suggesting that high-sensitivity routers should at least evaluate a confidence-hidden policy.”

Stronger version ONLY if supported across both GPT tasks:

> “For GPT, a confidence-hidden routing policy remains near the best observed operating point across a broad range of cost ratios on both MMLU and code.”

Do NOT say:

> “Hide confidence is the optimal policy.”

Do NOT say:

> “Models should never see confidence.”

Do NOT say:

> “Confidence causes harm.”

---

# 11. NEGATIVE RESULT IS ALSO USEFUL

If hidden is not robust:

That does NOT hurt Task 012.

Then the paper's conclusion is:

> diagnosing confidence sensitivity does not imply that simply withholding the scalar is a universally good mitigation.

That is scientifically clean.

No paperDirection rewrite unless the policy result materially changes the paper.

---

# 12. CALIBRATION + GAIN JOINT SYSTEMS VIEW

Task 013 should also prepare a compact systems decomposition using only existing data:

For each task/model report:

1. q1 calibration quality
2. confidence-to-coverage sensitivity
3. hidden-policy robustness
4. true_q-visible performance
5. code verifier efficiency where applicable

The purpose is to support the design lesson:

> confidence quality and confidence-to-policy gain are separate system properties.

If hidden is strong, add:

> routing architectures can also choose whether to expose the scalar at all.

Create:

`systems_design_matrix.csv`

---

# 13. PAPER-IMPACT GATE

Freeze this decision rule before running Task-013 analyses.

Create:

`task013_analysis_freeze.json`

Possible outcomes:

## `ACTIONABLE_UPGRADE`

Use only if ALL are true:

A. Hidden is `HIDDEN_ROBUST_BOTH_GPT_TASKS`.

B. Hidden is on the observed Pareto frontier for both GPT MMLU and GPT code.

C. Hidden is within 0.02 normalized loss of the best observed policy for >=50% of λ values in both GPT tasks.

D. Hidden does not have catastrophic worst-case regret:
   `max regret <= 0.10` in both GPT tasks.

E. The recommendation remains honest:
   no need to suppress Claude or choose one narrow λ.

If A–E hold:
- paper gets a new actionable design result;
- generate full revised paperDirection.

## `ACTIONABLE_BUT_TASK_SPECIFIC`

If hidden is robust on exactly one GPT task.

Then:
- do NOT make it a headline recommendation;
- include one short policy-analysis result;
- no full paper reframe unless another major reason appears.

## `DIAGNOSTIC_ONLY`

If hidden is not robust.

Then:
- current Task-012 framing remains source of truth;
- Task 013 goes to secondary/appendix;
- do not rewrite paperDirection.

## `CONTRADICTS_CURRENT_STORY`

If hidden analysis exposes a contradiction in Task 012 or prior reported numbers.

Then:
- issue correction;
- rewrite paperDirection for accuracy.

---

# 14. CONDITIONAL PAPERDIRECTION UPDATE

If and only if outcome is:

`ACTIONABLE_UPGRADE`

create:

`paperDirection_task013_proposed.txt`

The new framing should NOT replace the Task-012 sensitivity result.

Instead it should become:

1. confidence metadata can have high oversight leverage;
2. this can create large error leakage;
3. the natural q1 range overlaps the high-sensitivity region;
4. **withholding the scalar can be a competitive robust baseline in GPT across both tasks**;
5. therefore system designers should:
   - measure confidence calibration;
   - measure confidence-to-policy sensitivity;
   - compare visible-confidence routing against a confidence-hidden baseline;
   - evaluate cost robustness.

This turns the paper from:
- diagnosis

into:
- diagnosis + simple mitigation baseline.

If result is `ACTIONABLE_BUT_TASK_SPECIFIC`:
- create `paperDirection_task013_recommendation.md`
- no full rewrite.

If result is `DIAGNOSTIC_ONLY`:
- create `paperDirection_task013_recommendation.md`
- explicitly state current Task-012 paperDirection should remain unchanged.

If result is `CONTRADICTS_CURRENT_STORY`:
- create `paperDirection_task013_correction_required.txt`

---

# 15. REQUIRED OUTPUTS

Write to:

`to_gpt/013_hidden_confidence_policy_audit/`

Required:

- `report.md`
- `reproduction_check.md`
- `task013_analysis_freeze.json`
- `paper_impact_decision.md`
- `validation.md`
- `changed_files.txt`

Data:

- `policy_metrics.csv`
- `policy_cost_sweep.csv`
- `hidden_robustness_summary.csv`
- `hidden_vs_trueq.csv`
- `robust_policy_selection.csv`
- `systems_design_matrix.csv`

Figures:

- `figures/hidden_policy_summary.png`
- `figures/policy_pareto_frontier.png`
- `figures/hidden_regret_vs_lambda.png`
- `figures/coverage_leakage_policy_map.png`

Analysis:

`analysis/task013/`

Suggested scripts:

- `reproduce.py`
- `policy_metrics.py`
- `cost_sweep.py`
- `robustness.py`
- `pareto.py`
- `figures.py`
- `run_all.py`

---

# 16. REPORT FORMAT

`report.md` must include:

1. Plain-English bottom line
2. Reproduction check
3. Hidden vs true-q-visible raw metrics
4. Hidden vs fixed-score metrics
5. Hidden loss/regret across λ
6. Near-best breadth
7. Minimax regret result
8. Pareto-frontier result
9. GPT MMLU result
10. GPT code result
11. Claude MMLU result
12. Claude code saturation result
13. Cross-task hidden robustness classification
14. Systems design matrix
15. What hidden policy does NOT establish
16. Strongest actionable claim, if any
17. Paper-impact gate outcome
18. Exact recommended paper sentence
19. Exact claims to avoid
20. Whether paperDirection should be rewritten
21. `READY_FOR_GPT_REVIEW = YES`

---

# 17. CLAIMS TO AVOID

Do NOT say:

- hidden confidence is optimal
- hiding confidence is always safer
- hiding confidence removes internal confidence
- hidden routing is equivalent to item-only reasoning
- visible confidence causes failures
- Claude is more robust in a normative sense
- true-q-visible is bad in general
- confidence should never be shown
- λ is known in deployment
- the hidden policy is preregistered as a mitigation
- Task 013 was confirmatory

Preferred language:

- confidence-hidden routing baseline
- withholding the scalar
- competitive observed policy
- robust across tested cost ratios
- post-hoc policy analysis
- observed Pareto frontier
- minimax regret among observed policies

---

# 18. STOP RULE

Task 013 is the final zero-call policy audit.

After Task 013:

Do NOT:
- collect new model data;
- add new mitigation prompts;
- tune a new router;
- ask the model to ignore confidence;
- invent a new confidence transformation;
- add another benchmark;
- add Gemini/Grok to code;
- search over λ to manufacture a win.

If hidden is strong:
- integrate it as a simple baseline recommendation;
- write the paper.

If hidden is weak:
- keep Task 012 as the final framing;
- write the paper.

No more experiment expansion.
