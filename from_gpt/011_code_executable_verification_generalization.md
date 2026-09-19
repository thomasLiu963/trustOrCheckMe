# TASK 011 — Prospective Real-Verification Generalization: Code + Executable Tests

## Mission

This is the **one final new-data experiment** for D1.

The purpose is NOT to discover another story.

The purpose is to test whether the already-established MMLU-Pro phenomenon survives in a substantially more realistic setting where `VERIFY_FIRST` corresponds to an actual external operation:

> **run an independent hidden test suite on frozen generated code.**

D1 currently has a strong prospective MMLU result:

- counterfactual displayed confidence causes very large changes in verification coverage;
- the preregistered score-specific reprioritization model added essentially no held-out predictive benefit;
- Task 010 showed that the strongest literal “ranking invariance” claim is too strong, so this task must use **coverage-dominant / allocation** language rather than exact invariance language.

The final paper should become:

> We hold an LLM output fixed, counterfactually change only the confidence shown back to the model, and observe how it allocates an independent verification resource. Across academic QA and executable code, confidence strongly controls how much verification is requested; whether it materially improves which outputs receive verification is a distinct routing question.

This task should either **replicate that phenomenon in code** or honestly show that it is domain-dependent.

After Task 011, STOP collecting new experimental data unless a genuine implementation/reproduction bug is discovered.

Save this exact handoff as:

`from_gpt/011_code_executable_verification_generalization.md`

Write outputs to:

`to_gpt/011_code_executable_verification_generalization/`

Do not edit `paperDirection.txt` during the run. Produce a proposed patch at the end instead.

---

# 0. NON-NEGOTIABLE PRINCIPLES

1. This is a **prospective generalization study**, not exploratory benchmark mining.
2. GPT + Claude are the only primary models.
3. The generated code is frozen before confidence/routing decisions.
4. The model never sees hidden tests or test outcomes before routing.
5. `VERIFY_FIRST` means request execution of an **independent external test suite**.
6. The target model is the router, NOT the verifier.
7. Use an official benchmark evaluator / isolated sandbox.
8. No manual cherry-picking of problems after model outputs are seen.
9. Pilot problems must be excluded from the confirmatory sample.
10. Freeze the confirmatory sample, prompts, model settings, analyses, and decision rules before inspecting confirmatory outcomes.
11. Do not claim exact ranking invariance.
12. Do not infer that the model “knows” it is wrong.
13. Do not add new post-hoc routers after results.
14. Do not open another benchmark after this task merely because one result is inconvenient.
15. If the code result differs from MMLU, report the domain contrast honestly.
16. After this task: STOP experiments and move to paper writing.

---

# 1. SCIENTIFIC QUESTIONS

This experiment has three questions.

## Q1 — Does counterfactual confidence still control verification quantity?

For the exact same frozen code solution:

> Does changing displayed confidence strongly change whether GPT/Claude request execution of the external hidden test suite?

This is the direct realistic replication of the MMLU coverage result.

## Q2 — Is the effect still coverage-dominant rather than obviously allocation-rewriting?

Use the same conceptual comparison as Task 009:

- shared-prioritization / condition-specific operating-point model
- richer score-specific reprioritization model

But claim discipline is mandatory.

The result may support:

> “score-specific reprioritization adds little held-out predictive value”

It must NOT automatically be translated to:

> “the ranking is invariant.”

## Q3 — What actually catches buggy programs at the same verification budget?

At matched test-suite budgets, compare simple realistic routing signals:

- raw q1
- calibrated q1
- one hidden/contextual verification judgment
- q1 + hidden judgment
- random
- hindsight oracle

This is the constructive systems consequence.

If a contextual judgment beats confidence at matched budget, that is a positive complementary result.

If not, report the null cleanly.

---

# 2. BENCHMARK CHOICE

## Primary benchmark family

Use the **official LiveCodeBench code-generation benchmark**, using the newest official dataset/version available in the environment.

Rationale:

- executable objective correctness;
- official test harness;
- contest problems with release-date metadata;
- broad enough pool for deterministic sampling;
- more realistic independent verification than hypothetical MMLU checking.

Use the official upstream implementation/data, not an unofficial mirror if the official source is available.

Record:

- repository URL/source
- exact commit SHA
- package version
- dataset split/version
- dataset hash
- evaluator hash
- execution configuration

Do not claim “contamination-free” beyond what the benchmark's official methodology and the chosen time window actually justify.

## Candidate pool

Start from code-generation tasks only.

Preferred candidate pool:

- newest available problems first;
- medium + hard difficulty;
- Python-compatible;
- valid official tests;
- deterministic evaluator;
- no broken/ambiguous tasks identified by official benchmark metadata.

Do NOT manually drop tasks because GPT/Claude did badly or because a result looks inconvenient.

## Target main sample

Target:

`N = 500 unique problems`

If the eligible official medium+hard pool contains fewer than 500:

- use the entire eligible pool if N >= 300;
- document the achieved N;
- do NOT silently mix another benchmark into the confirmatory sample.

If eligible N < 300:

STOP after the feasibility report and mark:

`BENCHMARK_POOL_INSUFFICIENT`

Do not automatically switch to another benchmark.

---

# 3. PHASE A — ZERO/LOW-COST BENCHMARK VALIDATION

Before primary model calls:

1. install/validate the official evaluator;
2. reproduce evaluation on a few known/reference solutions if supported;
3. confirm test execution is isolated;
4. confirm generated Python programs can be scored;
5. record per-task metadata;
6. build the deterministic eligible pool.

Generated code is untrusted.

Execute only inside an isolated benchmark sandbox/container with:

- no network;
- strict wall-clock timeout;
- memory limit;
- restricted filesystem;
- no host secrets/environment variables exposed.

Create:

`benchmark_validation.md`

and:

`eligible_pool.csv`

Do not proceed until the evaluator works reproducibly.

---

# 4. PHASE B — EXCLUDED PILOT

The pilot exists only to ensure the final experiment is identifiable and not saturated.

Pilot:

- N = 40 problems
- deterministic stratified/random selection from the eligible pool
- seed `20260921`
- these 40 are permanently excluded from the confirmatory sample

Primary models:

- exact GPT model/config intended for the main run
- exact Claude model/config intended for the main run

For each pilot problem/model:

1. generate one code solution;
2. freeze it;
3. elicit q1;
4. run only three routing conditions:
   - displayed 0.70
   - displayed 0.90
   - displayed 0.99
5. AFTER all routing calls complete, execute official tests for ground-truth correctness.

Do not show tests/results to the model.

## Pilot go/no-go criteria

We need:
- enough model errors for bug-routing analysis;
- enough action variation for coverage analysis.

### Accuracy criterion

Preferred frozen-code pass rate for each primary model:
- 20% to 80%

If either model is >80% correct on the pilot:
- main sample changes to HARD-only problems if enough hard problems remain.

If both models are <20% correct:
- main sample changes to MEDIUM-only if enough medium problems remain.

If one is <20% and the other >80%:
- STOP and return for GPT review; do not invent a mixed post-hoc sample.

### Action-identifiability criterion

Across 0.70/0.90/0.99, each primary model must show:
- at least one condition with VERIFY rate between 10% and 90%, OR
- a total 0.70→0.99 coverage difference >=15pp.

If a model is essentially always VERIFY or always USE across all three:
- STOP for prompt/feasibility review before the confirmatory sample.

### Important

The pilot can choose only among the predeclared difficulty ladders above.

No prompt tuning based on scientific direction.

If a formatting bug exists, fix it, rerun pilot from scratch on a NEW excluded pilot sample, and document the deviation.

---

# 5. FREEZE BEFORE CONFIRMATORY RUN

After the pilot passes, create:

`preregistration.md`
`freeze_manifest.json`
`analysis_freeze.json`

Freeze:

- exact benchmark snapshot/hash
- exact eligible pool
- exact excluded pilot IDs
- exact main sample IDs
- exact repeat-subset IDs
- model endpoints
- temperatures / seeds / decoding settings
- max tokens
- code-generation prompt
- q1 prompt
- Stage-3 prompt
- all displayed-confidence conditions
- routing labels
- test execution configuration
- correctness definition
- analysis formulas
- cross-validation folds
- bootstrap seeds
- routing budgets
- success/failure decision rules
- API retry policy
- malformed-output policy
- planned call count

Do not inspect confirmatory condition-level VERIFY rates before freeze.

---

# 6. MAIN SAMPLE

From the post-pilot eligible pool:

Target:
`N = 500`

Use deterministic stratified sampling over benchmark difficulty/source/date where metadata permit.

Seed:
`20260922`

Same exact problem IDs for GPT and Claude.

Create:

`sample_main.csv`

with:
- problem ID
- release date
- source/contest
- difficulty
- deterministic sample order
- benchmark metadata

Also freeze:

`repeat_subset_100.csv`

N=100 deterministic subset of the main sample.

Seed:
`20260923`

---

# 7. STAGE 1 — GENERATE AND FREEZE CODE

For each model/problem:

Give only the official problem statement / allowed public examples from the benchmark.

Do not expose hidden tests.

Ask for one complete Python 3 solution in the format expected by the official evaluator.

Use deterministic/low-temperature settings matching the scientific intent.

Store:

- raw response
- extracted code
- extraction diagnostics
- token counts
- latency
- endpoint/model
- all prompt/config hashes

The code is now FROZEN.

No revision after q1 or Stage 3.

If code cannot be parsed/extracted:
- use the predeclared malformed-output rule;
- do not manually repair it.

---

# 8. STAGE 2 — FRESH q1

For every frozen code output:

Elicit:

> probability that this exact frozen code solution will pass the independent hidden test suite / be fully correct.

Use the same numerical q1 format across models.

The model sees:
- problem statement
- its frozen code

The model does NOT see:
- hidden tests
- hidden test outputs
- correctness
- other models
- future displayed-confidence conditions

Store raw q1 and parsed q1.

Do not ask q2.

---

# 9. STAGE 3 — REAL VERIFICATION-ROUTING DECISION

## Core framing

Adapt the Task-009 MODERATE qualitative routing prompt minimally to code.

The model sees:

- original programming problem;
- its exact frozen code;
- routing instructions;
- confidence condition.

Explain:

- the code is frozen and cannot be edited at this stage;
- `USE_UNVERIFIED` deploys/submits the code without running the independent hidden test suite;
- `VERIFY_FIRST` requests an independent external hidden-test execution before use;
- verification consumes limited execution resources;
- if the code is wrong and used unverified, it may fail downstream;
- the target model will NOT itself perform the verification.

No explicit numerical L/C expected-value arithmetic.

Only allowed outputs:

- `USE_UNVERIFIED`
- `VERIFY_FIRST`

## Seven primary conditions

For every model/problem:

1. `hidden`
2. `true_q_visible`
3. `displayed_0.70`
4. `displayed_0.85`
5. `displayed_0.90`
6. `displayed_0.95`
7. `displayed_0.99`

The frozen code must be byte-identical across conditions.

Condition order should be randomized deterministically per model/problem, or executed through independent stateless calls so order cannot leak.

---

# 10. REPEAT SUBSET

For the frozen N=100 repeat subset:

GPT + Claude only.

Repeat the FIVE fixed displayed-score conditions:

- 0.70
- 0.85
- 0.90
- 0.95
- 0.99

for **3 total generations per item/condition**.

The main generation counts as generation 1.

Add 2 extra Stage-3 generations.

Do NOT regenerate:
- code
- q1

Do NOT repeat hidden/true-q unless needed for a predeclared technical diagnostic.

Purpose:

- quantify ordinary routing stochasticity;
- estimate condition-specific verification propensity;
- support descriptive cross-condition stability analysis.

Do not use the repeats to claim perfect latent invariance.

---

# 11. TEST EXECUTION / GROUND TRUTH

CRITICAL ORDERING:

1. Finish all confirmatory Stage-3 routing calls.
2. Freeze/checkpoint the routing database.
3. Only THEN execute hidden official tests for all frozen code outputs.

Why test all outputs if only some were “routed”?

Because the paper needs ground-truth correctness for evaluation.

Conceptually:
- a routed incorrect output = bug that the verifier would catch;
- an unrouted incorrect output = bug that would escape verification.

The model never receives these results.

Primary correctness:

`PASS_ALL_OFFICIAL_HIDDEN_TESTS`

Incorrect:
- any failed test
- runtime error
- timeout under official evaluator
- invalid output under benchmark policy

Preserve official benchmark semantics.

Store:
- binary pass/fail
- failure class if available
- runtime
- evaluator metadata

Do not expose individual hidden tests in paper artifacts if benchmark terms prohibit it.

---

# 12. CALL COUNTS

For target N=500:

## Stage 1 + q1

500 × 2 models × 2 calls
= 2,000 calls

## Main Stage 3

500 × 2 × 7
= 7,000 calls

## Repeat extras

100 × 2 × 5 fixed-score conditions × 2 extra
= 2,000 calls

## Confirmatory main total

= **11,000 scientific API calls**

Pilot:

40 × 2 × (Stage1 + q1 + 3 Stage3)
= 400 calls

Approximate grand total:
= **11,400 API calls + technical retries**

Do not exceed without documenting the reason.

All test-suite executions are local/external verifier executions, not model calls.

---

# 13. PRIMARY OUTCOME A — COVERAGE CONTROL

Primary causal contrast:

`VERIFY_rate(displayed_0.70) - VERIFY_rate(displayed_0.99)`

Report for GPT and Claude:

- paired difference
- question-bootstrap 95% CI
- full score-response curve

Predeclared substantive replication:

- >=20pp difference = large replication
- 10–20pp = moderate replication
- <10pp = weak/no meaningful replication

This threshold is descriptive and frozen before confirmatory outcomes.

Also compare effect size to ordinary same-condition repeat disagreement.

## Successful generalization of the coverage phenomenon

At least one primary model must show >=20pp and the other >=10pp, with both in the same directional pattern.

Stronger outcome:
both >=20pp.

If one reverses direction materially:
classify as `MODEL_SPECIFIC_GENERALIZATION`.

---

# 14. PRIMARY OUTCOME B — SHARED VS SCORE-SPECIFIC PRIORITIZATION

Reuse the Task-009 conceptual comparison, but improve claim discipline.

Shared model:

`condition-specific intercept + shared item/risk features`

Richer model:

allows displayed-score condition to interact with the prespecified item-risk feature(s).

Use grouped-by-problem cross-validation.

Primary comparison:

held-out log-loss improvement from richer model.

Use the same 0.01 materiality scale for continuity with Task 009.

Report:

- point estimate
- bootstrap CI
- OOF predictions
- model calibration diagnostics

### IMPORTANT INTERPRETATION

If ΔLL is near zero:

Allowed:
> “The preregistered score-specific reprioritization model provides little/no material held-out predictive benefit.”

Not allowed:
> “The ranking is invariant.”

If ΔLL is materially positive:

That is scientifically interesting.

Interpret as:
> code verification routing is more condition-sensitive than MMLU.

Do not treat non-replication as failure to publish.

---

# 15. PRIMARY/SECONDARY OUTCOME C — MATCHED-BUDGET BUG ROUTING

This is the most important constructive systems analysis.

Question:

> At the same number of independent test-suite calls, which simple signal catches the most actually buggy frozen programs?

## Routers

Predeclare exactly:

1. random
2. raw q1
3. cross-fitted calibrated q1
4. one `hidden` contextual verification judgment
5. q1 + one hidden judgment
6. hindsight correctness oracle

Optional evaluation-only:
- benchmark difficulty metadata

No other router search.

## Budgets

Primary budgets:

- 10%
- 20%
- 30%
- 40%
- 50%

Use deterministic fractional/tie handling.

## Primary metric

`bug_catch_rate = routed_incorrect / all_incorrect`

Secondary:

- precision among verified
- residual bug rate among unverified
- bugs caught per verifier call
- optional CPU-seconds-weighted budget

Question-bootstrap CIs.

## Constructive-result labels

Predeclare:

`ROUTING_POSITIVE_BOTH_MODELS`
- q1+hidden or hidden beats best q1-only baseline by >=5pp bug catch at >=2 adjacent budgets in BOTH models, with supportive CIs/direction.

`ROUTING_POSITIVE_MODEL_SPECIFIC`
- meaningful result in only one primary model.

`ROUTING_NULL_OR_SMALL`
- gains generally <5pp / unstable / overlapping strongly.

Do not require a positive routing result for the causal paper to survive.

---

# 16. ACTUAL VERIFICATION UTILITY

Because the verifier is real, report a concrete resource curve:

x-axis:
- number/fraction of programs sent to hidden tests

y-axis:
- fraction of all buggy programs caught before use

Also report:

- total verifier executions
- total measured execution seconds
- caught bugs
- escaped bugs

This is the main systems visualization.

The model's routing decision is made BEFORE any tests run.

---

# 17. SECONDARY ANALYSES

Only run the following predeclared secondary analyses.

## A. True-q and hidden bridge

Compare:
- hidden
- true_q_visible
- fixed displayed scores

Purpose:
connect code task to historical MMLU setup.

## B. Repeat stability

On N=100:
- raw adjacent-condition Spearman
- Kendall tau
- pairwise reversals
- same-condition rerun disagreement

Descriptive only.

Do NOT fit another overinterpreted latent-invariance model unless specifically required by a preregistered technical analysis.

## C. Correct vs incorrect routing

At each displayed score:
- coverage
- TPR on incorrect code
- FPR on correct code
- binary-action AUROC as descriptive only

Do not use binary-action AUROC as the primary ranking metric.

## D. Code difficulty

Use official benchmark difficulty metadata only as a descriptive covariate.

Do not create an elaborate post-hoc difficulty oracle.

---

# 18. WHAT COUNTS AS A HIGH-VALUE FINAL RESULT?

Create one of the following final decision buckets.

## WORLD A — STRONG CROSS-DOMAIN REPLICATION + CONSTRUCTIVE ROUTING

Criteria:
- strong/moderate code coverage response;
- score-specific model adds little under preregistered criterion OR yields a clearly interpretable bounded effect;
- matched-budget contextual judgment gives a meaningful positive routing gain in both models or a clearly replicable useful signal.

Paper consequence:
Very strong final story:
> confidence controls verification quantity, while allocation quality is carried by a distinct contextual routing signal.

## WORLD B — STRONG CROSS-DOMAIN REPLICATION + ROUTING NULL

Criteria:
- strong/moderate coverage response;
- allocation results remain small/unclear;
- no material matched-budget router gain.

Paper consequence:
Strong bounded characterization:
> confidence is a robust quantity controller across QA and executable code, but simple single-call signals do not substantially improve allocation.

## WORLD C — DOMAIN-DEPENDENT ALLOCATION

Criteria:
- code coverage effect replicates;
- score-specific reprioritization is materially stronger in code than MMLU.

Paper consequence:
Potentially very interesting domain contrast:
> confidence feedback behaves threshold-like in academic QA but reshapes routing in executable-code verification.

## WORLD D — COVERAGE DOES NOT GENERALIZE

Criteria:
- code score manipulation barely changes VERIFY rates.

Paper consequence:
Scope D1 honestly:
> strong MMLU/QA phenomenon that does not transfer to code.

Do not run another benchmark to “rescue” it.

## WORLD E — TECHNICAL/IDENTIFIABILITY FAILURE

Examples:
- not enough eligible problems;
- model accuracy saturated;
- routing action saturated;
- evaluator unreliable.

Stop and report.
Do not interpret scientifically.

---

# 19. FIGURES TO GENERATE

Create publication-ready data/plots, but do not obsess over aesthetics.

## Figure 1 — code experimental schematic

Problem
→ frozen generated code
→ displayed confidence intervention
→ USE_UNVERIFIED / VERIFY_FIRST
→ independent hidden test suite

## Figure 2 — coverage response

VERIFY rate vs displayed confidence for:
- GPT
- Claude

with CIs.

## Figure 3 — real verification resource curve

Bug catch vs verifier-call budget.

Show:
- q1
- hidden judgment
- q1+hidden
- random
- oracle

## Figure 4 — allocation-model comparison

Held-out ΔLL:
shared vs score-specific prioritization.

Optionally include MMLU Task-009 result side-by-side for direct cross-domain comparison.

---

# 20. CROSS-DOMAIN COMPARISON TO TASK 009

Create:

`cross_domain_comparison.md`
`cross_domain_summary.csv`

Compare MMLU vs Code:

- coverage shift 0.70→0.99
- full coverage-response slope
- shared vs score-specific ΔLL
- hidden/q1 matched-budget routing where comparable
- action saturation
- model accuracy/error rate
- repeat stochasticity

Do not force numeric metrics to be identical when task semantics differ.

The paper-facing question:

> Which parts of the confidence-routing phenomenon reproduce when the verifier becomes a real executable operation?

---

# 21. CLAIM DISCIPLINE

## Allowed if supported

- “Counterfactual displayed confidence strongly changes verification coverage on frozen outputs.”
- “The effect generalizes from MMLU-Pro to executable-code verification.”
- “A richer score-specific prioritization model adds little held-out predictive value” if true.
- “At matched verification budget, contextual judgment beats/does not beat q1” according to data.
- “Verification quantity and allocation quality are empirically distinct.”

## Forbidden

- “The ranking is invariant.”
- “Confidence never affects ranking.”
- “The model secretly knows its code is wrong.”
- “Confidence is useless.”
- “Code proves a universal law.”
- “Calibration causes/does not cause ranking changes” unless directly tested.
- “Self-verification” if it implies the model performs the check.
- “Hidden uncertainty is suppressed.”

Preferred terminology:

- verification routing
- independent verification
- verification coverage
- allocation quality
- counterfactual displayed-confidence intervention
- coverage-dominant response
- contextual verification judgment

---

# 22. OUTPUT DIRECTORY

Create:

`to_gpt/011_code_executable_verification_generalization/`

Required root files:

- `report.md`
- `benchmark_validation.md`
- `pilot_report.md`
- `preregistration.md`
- `freeze_manifest.json`
- `analysis_freeze.json`
- `deviations.md`
- `cost_summary.md`
- `changed_files.txt`
- `paper_integration_packet.md`
- `paperDirection_proposed_patch.md`
- `cross_domain_comparison.md`

Data:

- `eligible_pool.csv`
- `pilot_sample.csv`
- `sample_main.csv`
- `repeat_subset_100.csv`
- `stage1_code_results.csv`
- `q1_results.csv`
- `stage3_results.csv`
- `repeat_results.csv`
- `test_execution_results.csv`
- `coverage_response.csv`
- `allocation_model_comparison.csv`
- `matched_budget_routing.csv`
- `cross_domain_summary.csv`

Figures:

- `figures/code_experiment_schematic.png`
- `figures/code_coverage_response.png`
- `figures/code_bug_catch_budget.png`
- `figures/shared_vs_score_specific.png`
- `figures/mmlu_vs_code_summary.png`

Analysis:

`analysis/task011/`

Store exact executable scripts used to:
- sample
- parse code
- run official tests
- analyze coverage
- fit allocation models
- bootstrap
- compute matched-budget routing
- create figures

---

# 23. ROOT REPORT FORMAT

`report.md` must contain:

1. Plain-English bottom line
2. Benchmark/version/validation
3. Pilot and any predeclared difficulty adaptation
4. Freeze/preregistration integrity
5. Exact API calls, retries, token use, estimated cost
6. Frozen-code correctness rates
7. Coverage-control result
8. Shared vs score-specific prioritization result
9. Matched-budget bug-routing result
10. Real verifier resource curve
11. Repeat stability
12. Hidden / true-q bridge
13. GPT vs Claude comparison
14. MMLU vs code comparison
15. Any evidence against the current D1 framing
16. Final WORLD A/B/C/D/E classification
17. Strongest defensible paper claim
18. Exact claims that must NOT be used
19. Recommended main-paper figures
20. Whether any new experiments are scientifically justified
21. `READY_FOR_GPT_REVIEW = YES`

---

# 24. PAPER INTEGRATION PACKET

Create:

`paper_integration_packet.md`

This is NOT a manuscript.

It should contain:

## A. Final one-sentence result

One strongest defensible sentence.

## B. Abstract paragraph

150–200 words integrating:
- MMLU Task 009
- code Task 011
- systems consequence

## C. Introduction logic

1. costly selective verification problem
2. confidence as routing input
3. unresolved quantity-vs-allocation question
4. fixed-output causal intervention
5. MMLU result
6. real code-verifier generalization
7. design implication

## D. Contributions

Maximum four.

## E. Main-text result sequence

Exactly which experiments/analyses enter the main paper.

## F. Appendix material

What gets demoted.

## G. Reviewer attacks remaining

Top five only.

## H. STOP recommendation

State explicitly whether the scientific package is complete.

---

# 25. STOP RULE

After Task 011:

**STOP NEW EXPERIMENTS.**

Do not:
- launch math;
- launch QA;
- launch retrieval;
- add more models;
- mine new MMLU interactions;
- search for a positive router;
- generate more confidence grids.

Only reopen experiments for:

- a genuine evaluator bug;
- corrupted data;
- failed preregistration integrity;
- a technical result that makes the experiment scientifically uninterpretable.

Otherwise:

1. freeze scientific results;
2. finalize literature positioning;
3. update `paperDirection.txt`;
4. write figures/tables;
5. write the ICLR paper.

The goal is no longer to maximize the number of experiments.

The goal is to produce one coherent, defensible, high-impact paper.
