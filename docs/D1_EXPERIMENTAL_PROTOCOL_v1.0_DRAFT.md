# D1 EXPERIMENTAL PROTOCOL

**Working title:** *When Confidence Controls Oversight: Self-Reported Uncertainty in LLM Verification*
**Core construct:** Confidence feedback in selective verification
**Protocol version:** v1.0-DRAFT
**Date:** 2026-09-17
**Status:** Consolidated post-Checkpoint-A working protocol. This document preserves the completed historical V2 experiment and Checkpoint A unchanged, defines the next exploratory causal pilot in executable detail, and specifies conditional downstream studies. The causal pilot is exploratory. No future study may be described as confirmatory until its final design is frozen in a dated protocol addendum before fresh target-model generations are collected.

---

# 0. PROVENANCE AND VERSIONING

This protocol does not retroactively replace or rewrite the original V2 protocol, the original workshop experiment, the submitted workshop paper, or Checkpoint A.

Those artifacts remain historical records of what was planned, run, observed, and subsequently re-analyzed.

The project has three distinct evidence stages.

## Historical Stage 0A — Original V2 experiment

The original V2-B study used 500 MMLU-Pro questions and four model families. Each model first produced an answer, then reported a numerical probability that the frozen answer was correct, and only afterward decided whether to use the answer without verification or verify it first.

The Stage-3 experiment crossed:

* two authority framings;
* confidence hidden versus visible;
* four error costs `L ∈ {2, 5, 10, 20}`;
* verification cost fixed at `C = 1`.

The resulting primary grid contained 32,000 Stage-3 decisions.

This historical experiment is complete and must not be retroactively described as if it prospectively tested the hypotheses developed after its results were seen.

## Historical Stage 0B — Checkpoint A

Checkpoint A was an exploratory post-submission re-analysis of the frozen V2-B data.

It made no new API calls.

Its main purposes were to determine whether:

1. hidden verification decisions caught more errors merely because they verified more answers;
2. confidence calibration explained the GPT result;
3. hidden verification decisions contained correctness-relevant information beyond the reported confidence score;
4. a simple cross-model “better signal determines whether visibility helps” theory was supported.

Checkpoint A found that visible GPT behaves almost exactly like the policy obtained by mechanically applying its raw confidence score to the supplied cost rule. Much of GPT's high-stakes cost failure is therefore explained by overconfident scores combined with reduced verification.

Checkpoint A also found evidence that GPT and Claude hidden verification decisions contain correctness-relevant information not fully captured by the reported confidence score. Claude additionally showed a clear equal-budget error-ranking advantage over confidence alone.

Checkpoint A did **not** establish that hidden states encode a richer uncertainty representation, that the models internally know when they are wrong, or that displaying confidence causes an internal information bottleneck.

Those remain future hypotheses.

## Current Stage 1 — Exploratory causal confidence pilot

The next empirical artifact is a deliberately small causal pilot.

The purpose of the pilot is not to produce the final ICLR result.

Its purpose is to answer cheaply:

> **If we keep the question and frozen answer fixed but deliberately change the confidence number shown to the model, does the model's verification decision change?**

Only if this premise survives the pilot should the project spend substantially more compute on prospective confirmation and generalization.

## Future confirmatory work

The final confirmatory confidence-manipulation experiment is **not frozen in this document**.

After the exploratory pilot:

1. inspect the causal effect;
2. conduct the planned targeted literature/novelty review;
3. decide which controls are scientifically necessary;
4. choose the final model roster and sample size;
5. freeze prompts, models, data, manipulations, primary estimands, exclusions, statistical tests, and hashes in a dated protocol addendum;
6. only then collect fresh confirmatory generations.

No exploratory pilot result may be relabeled as confirmatory later.

---

# 1. PROJECT-LEVEL CLAIM DISCIPLINE

The project does **not** currently claim that:

* showing confidence universally hurts models;
* LLMs consciously know when they are wrong;
* hidden states contain a privileged “true confidence” variable;
* confidence visibility suppresses an internal uncertainty representation;
* verbal confidence is useless;
* calibration is irrelevant;
* production systems universally feed numerical self-confidence back into the same LLM;
* a mechanism observed in an open model automatically explains GPT or Claude;
* policy agreement with confidence is inherently undesirable.

The project asks a narrower causal and descriptive question:

> **When an LLM must decide whether its own frozen answer deserves verification, how does explicitly displaying a confidence estimate change that decision, and what information explains any difference between the displayed-confidence policy and the model's otherwise-natural verification judgment?**

The protocol separates four kinds of statement.

### Observed result

A statement directly computed from an experiment.

Example:

> Visible GPT agrees with the raw-confidence threshold policy on approximately 99% of historical V2-B decisions.

### Exploratory interpretation

A possible explanation generated after looking at results.

Example:

> Explicit confidence may reduce re-evaluation.

### Prospective hypothesis

A claim specified before the data used to test it are generated.

Example:

> Manipulating only the displayed confidence number will causally alter verification probability.

### Mechanistic claim

A claim about how internal model computation produces behavior.

Example:

> Explicit confidence changes the influence of an internal error-related representation.

Mechanistic claims require appropriate internal and causal evidence.

---

# 2. TERMINOLOGY

This project should use the following terms consistently.

## Frozen answer

The answer produced before the verification manipulation.

Once frozen, the answer may not change during the Stage-3 verification decision.

## Reported confidence

The numerical probability of correctness originally supplied by the model for its frozen answer.

Example:

> `0.91` means that the model reported a 91% chance that its frozen answer was correct.

In internal code this variable may be stored as `reported_confidence`.

Avoid using the symbol `q` in researcher-facing operational documents unless needed mathematically.

## Displayed confidence

The confidence number shown to the model during the verification decision.

In historical V2 visible conditions:

> `displayed_confidence = reported_confidence`.

In the new causal manipulation:

> `displayed_confidence` may deliberately differ from `reported_confidence`.

This distinction is essential.

## True-confidence visible condition

The model is shown its actual previously reported confidence.

## Manipulated-confidence condition

The model is shown an experimentally assigned confidence number while all other frozen information remains unchanged.

The manipulation must never silently modify the stored historical confidence record.

## Hidden-confidence condition

The verification prompt does not display a numerical confidence value.

## Natural verification judgment

The model's `VERIFY_FIRST` versus `USE_UNVERIFIED` decision when confidence is hidden.

The term “natural” does **not** imply that this action reveals a pure latent state. It means only that the explicit confidence number is absent from the Stage-3 prompt.

## Verification action

The model must choose one of exactly two actions:

`VERIFY_FIRST`

or

`USE_UNVERIFIED`.

## Verification budget

The fraction of answers receiving `VERIFY_FIRST`.

Example:

> A verification budget of 20% means that 20 of every 100 outputs receive independent checking.

## Error catch rate

Among answers that are actually wrong, the fraction sent to verification.

This is a major deployment-relevant metric.

## Correct-answer verification rate

Among answers that are actually correct, the fraction unnecessarily sent to verification.

## Equal-budget comparison

Two routing methods are compared while allowing them the same total number of checks.

This distinguishes “checking better” from simply “checking more.”

## Error cost `L`

The synthetic cost of using a wrong answer without verification.

Historical V2 used:

`L ∈ {2, 5, 10, 20}`.

## Verification cost `C`

The synthetic cost of verification.

Historical V2 fixed:

`C = 1`.

If a numerical confidence `p` is treated literally as the probability of correctness, the simple expected-cost rule is:

`VERIFY_FIRST` if `(1 - p) × L > C`.

Equivalently, with `C = 1`:

`VERIFY_FIRST` when `p < 1 - 1/L`.

For example:

* at `L = 10`, the threshold is `0.90`;
* at `L = 20`, the threshold is `0.95`.

The causal pilot deliberately manipulates displayed confidence around these thresholds.

## Confidence responsiveness

The extent to which changing only the displayed confidence alters verification behavior.

## Threshold flip

A within-question change from `VERIFY_FIRST` below the numerical threshold to `USE_UNVERIFIED` above the threshold.

A threshold flip is useful evidence that the displayed number causally controls action.

## Override

A verification decision that goes against the action mechanically implied by the displayed confidence and supplied cost.

Overrides are especially informative when the displayed score conflicts strongly with other information available to the model.

---

# 3. PRIMARY RESEARCH QUESTIONS

## RQ1 — Causal confidence control

When the frozen answer is held constant, does changing only the confidence number displayed to the model causally change whether it requests verification?

This is the immediate primary question.

## RQ1A — Dose response

Does verification probability change systematically as displayed confidence moves from low to high?

The expected mechanical direction is:

> lower displayed confidence → more verification.

## RQ1B — Threshold sensitivity

Does verification behavior change sharply when the displayed score crosses the decision threshold implied by the supplied costs?

For example:

* at `L = 10`, compare scores just below versus just above `0.90`;
* at `L = 20`, compare scores just below versus just above `0.95`.

## RQ1C — Score provenance

If the same numerical score is described as:

* the model's own earlier confidence;
* an external estimate;
* or a generic correctness estimate,

does the model respond differently?

This question determines whether the paper should emphasize **self-confidence feedback** specifically or **displayed uncertainty signals** more generally.

## RQ1D — Override under conflict

When displayed confidence strongly conflicts with the model's previous confidence or with other evidence available in the prompt, does the model still mechanically follow the displayed score?

This measures the model's willingness to override an explicit uncertainty signal.

## RQ2 — Prospective robustness

If a causal displayed-confidence effect exists in the pilot, does it replicate prospectively on fresh questions and appropriately chosen models?

## RQ3 — Generalization

Does the effect survive beyond the original MMLU-Pro closed-API setting, including:

* at least one open-weight model family;
* and ideally one objectively checkable non-multiple-choice task?

## RQ4 — Source of hidden-decision information

When a model's hidden verification decision predicts correctness beyond its verbal confidence, what explains that information?

Candidate explanations include:

* a second independent evaluation of the question;
* another verbal confidence sample;
* answer-token probability;
* ordinary task difficulty;
* other observable response features;
* richer hidden-state information.

## RQ5 — Conditional mechanistic question

For an open model that behaviorally reproduces the relevant phenomenon, is there error-relevant information in internal activations beyond observable uncertainty baselines, and does explicit confidence change how strongly that information affects verification?

## RQ6 — Better routing

Can a routing method catch more real errors at the same verification budget than raw verbal confidence alone?

---

# 4. HISTORICAL DATA SOURCES

The canonical historical V2-B data identified during Checkpoint A are:

* raw checkpoint: `results/v2/raw/v2.sqlite3`;
* question list: `data/samples/mmlu_pro_v2b.jsonl`;
* existing analysis manifest: `paper_outputs/v2/analysis_manifest.json`.

The historical primary grid contains:

* 500 questions;
* 4 models;
* 2 authority conditions;
* 2 confidence-visibility conditions;
* 4 error costs;
* 32,000 primary Stage-3 rows.

The original frozen answer and confidence values must never be overwritten by the new pilot.

The manipulated confidence used in future Stage-3 prompts must be written to a **new field**, for example:

`displayed_confidence`

and the original field must remain separately stored as:

`reported_confidence`.

Recommended additional provenance fields include:

* `reported_confidence_original`;
* `displayed_confidence`;
* `display_source_condition`;
* `study_id`;
* `pilot_or_confirmatory`;
* `question_id`;
* `model_id`;
* `model_endpoint`;
* `api_timestamp`;
* `L`;
* `C`;
* `prompt_hash`;
* `attempt_number`;
* `raw_response`;
* `parsed_action`;
* `parse_status`;
* `latency_ms`;
* `input_tokens`;
* `output_tokens`;
* `estimated_cost_usd`.

---

# 5. STUDY 1: EXPLORATORY CAUSAL CONFIDENCE PILOT

## 5.1 Purpose

Study 1 is deliberately small.

Its job is to determine whether the project should continue down the confidence-as-control-signal path.

It is not intended to estimate a final publication-quality effect size.

The central test is:

> **Hold the question and frozen answer constant. Change only the confidence number shown during Stage 3. Does the verification action change?**

---

# 6. STUDY 1 MODEL ROSTER

Use two model families.

## Model 1 — GPT-5.6 Sol

Use the exact current API model identifier corresponding to GPT-5.6 Sol and save the provider-returned model/version metadata.

Rationale:

Historical V2 showed that when GPT sees its actual confidence, its Stage-3 behavior becomes almost identical to the raw-confidence threshold rule.

Therefore GPT is the cleanest test of whether the **number itself** causally drives the behavior.

## Model 2 — Claude Sonnet 5

Use the exact current API identifier corresponding to Claude Sonnet 5 and save provider metadata.

Rationale:

Historical V2 and Checkpoint A provide a useful contrast.

Claude's hidden policy clearly outperformed confidence-only ranking at an equal verification budget, while historical confidence visibility moved Claude toward **more** rather than less verification.

The two models therefore represent meaningfully different historical behavior.

## Model version drift

The pilot must record exact endpoint/model metadata at runtime.

If the provider silently serves a materially different model version from the historical run, document this.

Do not pretend current pilot behavior is an exact repeat of the historical model if endpoint drift is known.

---

# 7. STUDY 1 QUESTION SAMPLE

## 7.1 Sample size

Primary pilot:

**100 question IDs.**

Use the same 100 IDs for both pilot models.

## 7.2 Source

Draw the 100 questions from the historical 500-question V2-B set.

This is permitted because Study 1 is explicitly exploratory.

The later confirmatory experiment must use fresh data not selected after inspecting these pilot outcomes.

## 7.3 Selection rule

The 100-question sample must be selected deterministically **without using historical Stage-3 actions as a selection criterion**.

Preferred rule:

1. sort the canonical 500 IDs in the frozen V2-B order;
2. use a predeclared random seed `20260917`;
3. sample 100 unique IDs without replacement;
4. store the selected IDs before making any new API calls;
5. hash the final pilot ID list.

Do not manually substitute questions because they appear interesting.

Do not choose questions based on whether the previous model answer was right or wrong.

Do not choose questions based on whether the original confidence was convenient for the desired hypothesis.

A secondary post-hoc analysis may later stratify results by original correctness or original confidence, but these variables must not determine inclusion in the primary pilot sample.

---

# 8. STUDY 1 FROZEN ANSWERS

For every selected `(question, model)` pair:

* reuse that model's existing Stage-1 frozen answer from historical V2-B;
* reuse historical Stage-1 correctness for analysis;
* reuse historical Stage-2 reported confidence as the true historical confidence;
* do not ask the model to answer the question again for the primary pilot.

This is important because the causal manipulation should occur **after answer generation**.

The manipulated-confidence comparison must not be confounded by changes in the answer itself.

---

# 9. STUDY 1 AUTHORITY CONDITION

The pilot should **not** reproduce the full historical human-versus-AI authority factorial design.

Use one standardized authority framing.

Preferred condition:

**AI-system verification authority.**

Rationale:

1. authority was a secondary and weaker historical effect;
2. the rebuilt project's practical framing concerns uncertainty-controlled system routing;
3. retaining both authority conditions would double pilot cost without answering the immediate causal question.

The exact Stage-3 wording should remain as close as practical to the historical AI-authority prompt.

The only intended changes should be those required for the new displayed-confidence manipulation.

Any other wording change must be documented.

---

# 10. STUDY 1 ERROR-COST CONDITIONS

Use two error-cost conditions:

`L = 10`

and

`L = 20`.

Keep:

`C = 1`.

These conditions are chosen because they:

* represent the high-stakes region where the original GPT result was strongest;
* generate clear confidence thresholds;
* allow the score manipulation to be centered around two different decision boundaries.

The corresponding confidence thresholds are:

### `L = 10`

Threshold:

`1 - 1/10 = 0.90`.

### `L = 20`

Threshold:

`1 - 1/20 = 0.95`.

---

# 11. STUDY 1 DISPLAYED-CONFIDENCE GRID

The pilot should manipulate displayed confidence around the relevant threshold.

## At `L = 10`

Use:

* `0.80`
* `0.88`
* `0.89`
* `0.91`
* `0.99`

The most important local contrast is:

`0.89` versus `0.91`.

The extreme contrast is:

`0.80` versus `0.99`.

`0.90` itself is deliberately omitted from the primary manipulation because equality creates a boundary-case convention. The causal question is cleaner just below and just above the threshold.

## At `L = 20`

Use:

* `0.90`
* `0.94`
* `0.949`
* `0.951`
* `0.99`

The most important local contrast is:

`0.949` versus `0.951`.

The extreme contrast is:

`0.90` versus `0.99`.

If provider prompt formatting makes three-decimal probabilities awkward or unnatural, use:

* `0.90`
* `0.93`
* `0.94`
* `0.96`
* `0.99`

and document the change **before** the first pilot call.

Do not alter the grid after looking at pilot results.

---

# 12. STUDY 1 BASELINE CONDITIONS

In addition to manipulated-score conditions, every `(question, model, L)` should receive two baselines.

## Baseline A — confidence hidden

The Stage-3 prompt does not include any explicit confidence score.

This provides a same-day/current-model hidden reference and avoids relying only on historical API behavior.

## Baseline B — true confidence visible

The prompt displays the model's **actual historical Stage-2 reported confidence**.

This checks whether the current model instance reproduces approximately the same qualitative visible-confidence behavior as historical V2.

The historical confidence itself must not be recomputed during the pilot.

---

# 13. PRIMARY PILOT CALL COUNT

The primary grid contains:

100 questions
× 2 models
× 2 error costs
× 7 Stage-3 conditions

where the seven conditions are:

* hidden;
* true-confidence visible;
* five manipulated-confidence values.

Total:

**2,800 primary Stage-3 calls.**

This is the intended first causal pilot.

Do not expand beyond this grid before inspecting the result.

---

# 14. REPEATED-SAMPLING STABILITY SUBSET

The original V2 used one Stage-3 generation per cell.

To determine whether pilot findings are dominated by stochastic action flips, run a small repeated subset.

## Subset size

Select **20 of the 100 pilot question IDs** using a second deterministic seed.

Recommended seed:

`20260918`.

Do not select these questions based on pilot outputs.

## Repeats

For these 20 questions, run **two additional independent repetitions** of every primary Stage-3 pilot condition.

This produces three total observations per condition for the stability subset.

Additional calls:

20 questions
× 2 models
× 2 error costs
× 7 conditions
× 2 extra repetitions

= **1,120 calls.**

Total pilot size with repeats:

**3,920 calls.**

If API cost or runtime becomes unexpectedly high, the repeated subset may be deferred until after the 2,800 primary calls are analyzed.

The decision to defer must be made because of operational cost or time, not because the primary results are inconvenient.

---

# 15. GENERATION SETTINGS

The pilot should reuse the historical Stage-3 generation style as closely as provider APIs permit.

For each provider:

* freeze the exact model endpoint;
* record temperature;
* record top-p;
* record max output tokens;
* record reasoning mode if applicable;
* record seed if supported;
* record system/developer/user prompt text;
* record structured-output settings if used;
* record API date and SDK/package version.

Preferred settings should minimize irrelevant response variation.

If the historical Stage-3 runs used deterministic or near-deterministic settings, retain them.

Do not silently change reasoning mode between conditions.

Do not enable an expensive reasoning setting only for selected conditions.

---

# 16. STAGE-3 PROMPT INVARIANCE

The manipulated-confidence prompt must be created from the existing Stage-3 visible-confidence template.

Across manipulated-confidence conditions, the prompt should be **byte-identical except for the displayed confidence value** wherever feasible.

The question text must remain identical.

The frozen answer must remain identical.

The costs must remain identical.

The authority wording must remain identical.

The instructions must remain identical.

Only the displayed confidence value changes.

This is what makes the experiment causal.

---

# 17. IMPORTANT WORDING RULE FOR MANIPULATED CONFIDENCE

The primary causal pilot should preserve the historical framing that the number is the model's own prior report.

For example, the prompt may state the equivalent of:

> “Your previously reported probability that the frozen answer is correct was 89%.”

Do **not** tell the model that the value was experimentally manipulated.

The research question is how the displayed signal affects behavior.

The experimental database must, however, separately and correctly record:

* the real historical confidence;
* the displayed manipulated confidence.

Researcher-facing artifacts must never confuse the two.

---

# 18. RESPONSE FORMAT

The Stage-3 output must contain one of exactly two actions:

`VERIFY_FIRST`

or

`USE_UNVERIFIED`.

Prefer the same output structure used successfully in historical V2.

If a structured JSON response is already implemented robustly across both providers, use it.

Example acceptable structure:

`{"action":"VERIFY_FIRST"}`

or

`{"action":"USE_UNVERIFIED"}`.

The output schema itself must not differ between confidence conditions.

---

# 19. PARSING AND RETRY POLICY

Do not silently rerun an output until the desired label appears.

Each request should have:

* `attempt_number`;
* provider status;
* raw response;
* parser status;
* parsed action if valid.

## Transport/provider failure

If the API request fails because of timeout, rate limit, connection error, or documented provider error:

* retry automatically up to three times using exponential backoff;
* retain logs of all failed attempts;
* mark the final successful attempt as a retry.

## Parse failure

If the provider returns a successful response that cannot be parsed into one of the two allowed actions:

* retain the original response;
* mark `parse_failed = true`;
* allow at most one standardized repair request if the historical pipeline already has a fixed repair mechanism;
* do not create custom repair wording based on the content of the malformed response.

Report parse-failure prevalence by model and condition.

If parse failures are nontrivial or condition-dependent, the pilot should not be interpreted until that issue is understood.

---

# 20. PRIMARY STUDY 1 ESTIMANDS

The pilot should emphasize within-question causal comparisons.

## 20.1 Verification-response curve

For each model and `L`, compute:

> probability of `VERIFY_FIRST` at each displayed confidence value.

Plot:

x-axis = displayed confidence
y-axis = verification rate.

Each model and `L` should have its own curve.

The main question is whether verification systematically decreases as displayed confidence increases.

## 20.2 Near-threshold causal effect

For `L = 10`:

compare:

`displayed_confidence = 0.89`

versus

`displayed_confidence = 0.91`.

Primary quantity:

`P(VERIFY | 0.89) - P(VERIFY | 0.91)`.

For `L = 20`:

compare the selected just-below versus just-above threshold pair.

This is the cleanest local test of whether a small change in displayed confidence around the decision boundary changes action.

## 20.3 Extreme-score causal effect

For each `L`, compare the lowest versus highest displayed confidence.

This asks whether the displayed number has a large global effect even if behavior is not a sharp mathematical threshold.

## 20.4 Within-question threshold-flip rate

For each question, determine whether action moves in the mechanically expected direction across the near-threshold pair.

Report at least:

* `VERIFY → USE`;
* no change;
* `USE → VERIFY`.

The causal signal is strongest when `VERIFY → USE` substantially exceeds the reverse direction.

## 20.5 Monotonicity across displayed confidence

For each question, inspect whether actions are broadly ordered as displayed confidence rises.

Because the output is binary, perfect monotonicity is not required.

Report:

* fraction of question trajectories with no reverse switch;
* fraction containing at least one `USE → VERIFY` reversal as confidence increases.

This is descriptive.

---

# 21. PILOT UNCERTAINTY ESTIMATION

Use paired question-level bootstrap.

Recommended:

* 5,000 bootstrap resamples;
* resample question IDs;
* preserve all within-question confidence conditions together;
* fixed seed `20260917`.

Report nominal 95% percentile intervals.

The pilot is exploratory, so these intervals are descriptive rather than a basis for declaring a preregistered hypothesis “confirmed.”

For the 20-question repeated subset, retain repetition as nested within question.

Do not treat repeats as independent questions.

---

# 22. REPEATED-SAMPLING ANALYSIS

For the 20-question stability subset:

1. compute the fraction of cells where all three runs agree;
2. compute within-cell action entropy or simple disagreement rate;
3. compare stochastic disagreement with the size of the confidence-manipulation effect;
4. report whether the confidence effect is substantially larger than ordinary repeated-sampling variation.

The goal is to answer:

> **Is changing displayed confidence producing more behavioral change than simply asking the same question again?**

If the manipulated-score effect is no larger than ordinary repetition noise, scaling the causal experiment is not justified.

---

# 23. STUDY 1 GO / STOP GATE

Study 1 is exploratory, so the gate is a scientific decision rule rather than a confirmatory significance test.

## Strong PASS

Proceed toward prospective confirmation if the data show all or most of the following:

1. verification changes materially across displayed confidence;
2. the direction is generally lower confidence → more verification;
3. the effect appears within the same frozen questions;
4. the near-threshold contrast is clearly larger than ordinary repeated-sampling noise;
5. the effect is not produced entirely by a tiny number of questions;
6. at least GPT shows a strong causal response, and Claude provides either a compatible causal response or a scientifically informative contrast.

A rough practical benchmark is that the low-versus-high displayed-confidence contrast should be on the order of at least **10–15 percentage points** for at least one model, rather than a 1–2 point movement.

This numerical value is a pilot decision heuristic, not a preregistered scientific threshold.

## Weak / ambiguous result

If displayed confidence changes behavior only weakly or inconsistently:

* inspect parser issues;
* inspect repeated-sampling variability;
* inspect whether the current models reproduce the historical true-visible condition;
* inspect whether prompt differences accidentally changed.

Do not immediately launch a full study.

## FAIL

Stop the current causal-control framing if:

* manipulated confidence has essentially no systematic effect;
* observed shifts are comparable to ordinary stochastic reruns;
* current true-visible versus hidden behavior does not resemble the historical phenomenon at all;
* or the effect depends on an obvious implementation artifact.

A failed pilot should trigger rethinking, not a larger run.

---

# 24. PILOT REPORT REQUIRED BEFORE ANY EXPANSION

Create:

`to_gpt/study1_causal_pilot/`

Required outputs:

* `study1_pilot_report.md`
* `study1_primary_results.csv`
* `study1_repeated_results.csv`
* `study1_call_manifest.csv`
* `study1_failures.csv`
* `study1_selected_question_ids.json`
* `study1_config.json`
* `study1_prompt_templates/`
* `figures/`
* exact analysis scripts.

The report must clearly answer:

1. Did changing only displayed confidence causally change verification?
2. How large was the effect for GPT?
3. How large was the effect for Claude?
4. Did behavior change sharply around the supplied threshold?
5. Was the effect monotonic?
6. Was it larger than repeated-generation noise?
7. Did the current true-visible condition resemble historical behavior?
8. Is the causal-confidence framing worth continuing?
9. What result was surprising or inconsistent?
10. What should be tested next?

The report must be adversarial toward the preferred hypothesis.

---

# 25. STUDY 1B — OPTIONAL MICRO-CONTROLS AFTER THE PRIMARY PILOT

Do not run these until the primary 2,800-call pilot has been inspected.

If the primary causal effect is strong, small controls may be run before designing the confirmatory experiment.

These controls remain exploratory.

---

# 26. STUDY 1B-A — SCORE PROVENANCE MICRO-CONTROL

## Question

Does the same displayed percentage have a different effect depending on who supposedly produced it?

## Minimal design

Use a deterministic subset of approximately 40 pilot questions per model.

Use one high-stakes condition, provisionally:

`L = 20`.

Use two displayed confidence values straddling the threshold.

For example:

* `0.94`;
* `0.96`.

Cross with three provenance conditions:

### SELF

> “Your previously reported confidence was 94%.”

### EXTERNAL

> “An external confidence estimator assigns a 94% probability that the frozen answer is correct.”

### UNLABELED / GENERIC

> “Estimated probability that the frozen answer is correct: 94%.”

Everything else remains matched.

Approximate calls:

40 questions
× 2 models
× 2 values
× 3 provenance conditions

= **480 calls.**

## Interpretation

If SELF behaves substantially differently from EXTERNAL and GENERIC:

> a self-feedback interpretation becomes more plausible.

If all three behave similarly:

> the safer framing is displayed-score conditioning rather than self-confidence specifically.

---

# 27. STUDY 1B-B — QUALITATIVE-STAKES MICRO-CONTROL

## Motivation

The historical numerical setup explicitly supplies `L` and `C`.

A reviewer could argue:

> the model is simply performing arithmetic on a confidence value and a formula-like cost structure.

A qualitative-stakes control removes the exact numerical decision boundary.

Example conceptual wording:

> “Using a wrong answer here would be very costly. Independent verification is available but also has a smaller cost.”

Do not include an explicit threshold formula.

Then manipulate displayed confidence.

If confidence still strongly changes verification, the effect cannot be reduced entirely to executing a supplied numerical threshold.

This control should be small and exploratory.

The exact qualitative prompt must be frozen before use.

---

# 28. STUDY 1B-C — SELF-CONTRADICTION / OVERRIDE CONTROL

A useful conflict condition deliberately displays a score very different from the model's historical reported confidence.

Examples:

* original reported confidence ≥ 95%, displayed score = 60%;
* original reported confidence ≤ 70%, displayed score = 99%.

The scientific question is:

> **Does the model follow the newly displayed score even when it contradicts what the model previously said about the same frozen answer?**

This does not prove that the earlier or later score is more “true.”

It measures the strength of displayed-score control.

If the historical sample contains too few examples in one confidence region, do not manufacture a balanced sample by cherry-picking Stage-3 outcomes.

Use whatever qualifying examples exist and label the analysis exploratory.

---

# 29. TARGETED LITERATURE / NOVELTY AUDIT AFTER STUDY 1

If the causal pilot succeeds, run a targeted research pass **before** freezing the final confirmatory study.

The literature audit should be given the exact empirical result.

Its purpose is to try to invalidate the novelty claim.

It should focus on:

* confidence-driven abstention;
* verbal confidence and commitment;
* confidence-conditioned routing;
* model cascading;
* escalation;
* self-conditioning on previous outputs;
* numerical anchoring;
* re-evaluation suppression;
* confidence calibration;
* selective prediction;
* model-generated score feedback;
* uncertainty-driven agents;
* causal confidence steering;
* internal correctness representations.

The research question should not be:

> “Find papers about uncertainty.”

It should be:

> **“Has prior work already shown this exact causal feedback phenomenon, or offered an explanation that subsumes it?”**

The output of this literature audit must be incorporated into the dated confirmatory protocol addendum.

---

# 30. STUDY 2 — PROSPECTIVE CAUSAL CONFIRMATION

**Status: DESIGN SKELETON ONLY. NOT FROZEN.**

Study 2 becomes active only if Study 1 passes.

No headline Study-2 data should be collected before a separate dated addendum freezes the final design.

---

# 31. STUDY 2 CORE HYPOTHESIS

The prospective core should be approximately:

> **Holding a model's frozen answer constant, experimentally changing the confidence value displayed during the verification decision causally changes its probability of requesting verification.**

The final version of this hypothesis may be narrowed after the Study-1 literature audit.

---

# 32. STUDY 2 REQUIRED FRESHNESS

The confirmatory question set must not be selected using Study-1 action outcomes.

Preferred options include:

* a deterministic fresh sample from MMLU-Pro excluding all historical V2-B questions;
* or another frozen objectively scored question set selected before target-model generation.

The final choice must be made before confirmatory outputs are collected.

If MMLU-Pro does not contain sufficient clean fresh items, another dataset should be selected prospectively.

---

# 33. STUDY 2 MODEL ROSTER

Do not automatically include fifteen models.

The intended confirmatory roster should be compact.

A likely design is:

* GPT family;
* Claude family;
* one additional frontier family selected for complementary historical behavior;
* possibly one open-weight family if the behavioral implementation is ready.

Exact models must be frozen after checking API availability.

The paper should prioritize:

> distinct scientific behavior

over:

> many nearly redundant same-vendor variants.

---

# 34. STUDY 2 SAMPLE SIZE

Do not freeze a final sample size before the pilot provides an empirical effect-size estimate.

After Study 1:

1. estimate the near-threshold paired effect;
2. estimate question-to-question variability;
3. estimate stochastic action variability from the repeat subset;
4. run a simulation/power or precision analysis;
5. choose a confirmatory sample large enough to estimate the effect with useful precision.

The final sample-size justification must be included in the Study-2 addendum.

Do not choose `N` because it produces significance on pilot data.

---

# 35. STUDY 2 PRIMARY ENDPOINT

The likely primary endpoint is the paired difference in verification probability between a just-below-threshold and just-above-threshold displayed confidence value.

The exact pair should be selected before confirmation.

The analysis should remain question-paired.

Secondary endpoints may include:

* full confidence-response curve;
* low-versus-high confidence contrast;
* threshold-flip rate;
* action monotonicity;
* override rate;
* variation by model.

---

# 36. STUDY 2 PROVENANCE CONTROL

The final confirmatory design should include a score-provenance control if the Study-1 micro-control suggests provenance is scientifically meaningful.

If provenance has little effect, do not unnecessarily inflate the factorial design.

Instead, accurately rename the phenomenon.

The paper should prefer the broader truthful claim:

> displayed confidence controls verification

over the narrower unsupported claim:

> models specially defer to their own confidence.

---

# 37. STUDY 2 QUALITATIVE-STAKES CONTROL

Include a qualitative-stakes control only if:

* it addresses a major reviewer alternative explanation;
* the pilot shows the effect survives well enough to measure;
* and the wording can be made genuinely comparable.

The control is valuable if it demonstrates that the result is not merely the model calculating a supplied mathematical threshold.

---

# 38. STUDY 2 ANALYSIS FREEZE

Before any confirmatory call, freeze:

* exact datasets and question IDs;
* exact model roster and endpoints;
* exact Stage-3 prompt templates;
* confidence-manipulation values;
* cost/stakes conditions;
* authority framing;
* inference settings;
* retry policy;
* parser;
* primary endpoint;
* secondary endpoints;
* bootstrap/statistical procedure;
* exclusions;
* treatment of malformed outputs;
* sample size;
* random seeds;
* dataset hashes;
* code commit;
* protocol addendum hash.

After this freeze:

> do not add or remove confirmatory questions because of target-model behavior.

---

# 39. STUDY 3 — GENERALIZATION

**Status: CONDITIONAL.**

The causal effect should eventually be tested beyond the original closed-model multiple-choice setting if time permits.

Study 3 is not allowed to jeopardize the quality of Study 2.

---

# 40. STUDY 3A — OPEN-WEIGHT BEHAVIORAL CONTINUITY

Before using an open model for interpretability, test whether it exhibits the relevant behavioral phenomenon.

Candidate model characteristics:

* instruction-tuned;
* strong enough to answer the tasks meaningfully;
* weights and hidden activations accessible;
* token/log probabilities accessible;
* tractable on available compute;
* preferably from a model family distinct from the closed providers.

The exact model should be frozen only after checking current availability.

One strong open model is sufficient for initial mechanistic work.

A second family is desirable for replication if compute permits.

## Continuity requirement

An open model earns mechanistic interpretation only if it shows a qualitatively relevant confidence-feedback effect.

Do not use an unrelated model's internals to explain GPT or Claude merely because its weights are accessible.

---

# 41. STUDY 3B — SECOND TASK FAMILY

Preferred first extension:

**objectively checkable mathematics.**

Reasons:

* it is not multiple-choice;
* correctness can be scored automatically;
* the frozen-answer design can still be used;
* verification is meaningful;
* infrastructure is lighter than full code-agent evaluation.

A code + unit-test setting is a valuable later extension but is not required for the core paper.

The second task must preserve the fundamental causal structure:

1. answer generated;
2. answer frozen;
3. confidence elicited;
4. confidence frozen;
5. Stage-3 verification decision manipulated.

---

# 42. STUDY 4 — SOURCE OF THE HIDDEN-DECISION ADVANTAGE

**Status: CONDITIONAL.**

Study 4 should occur only if fresh data continue to show that the hidden verification decision predicts correctness beyond the explicit confidence score.

The goal is to explain the extra information without prematurely invoking hidden representations.

---

# 43. STUDY 4A — SECOND CONFIDENCE SAMPLE

## Hypothesis

Perhaps the hidden verification decision is useful simply because Stage 3 gives the model another opportunity to inspect the question.

This would mean the system benefits from two noisy assessments rather than one.

## Test

For each frozen answer, independently re-elicit a second confidence estimate under a prompt that does not reveal the first score.

Then compare predictive performance using:

1. first confidence only;
2. first + second confidence;
3. first confidence + hidden verification action;
4. first + second confidence + hidden action.

If two verbal confidence samples absorb the predictive advantage of the hidden action:

> the phenomenon may largely reflect repeated assessment or ensembling.

That is a valid result.

Do not force a deeper interpretation.

---

# 44. STUDY 4B — RE-EVALUATION CONTROL

A related hypothesis is:

> hidden Stage 3 causes the model to reconsider the answer, while visible confidence gives it an easy number to follow and therefore reduces reconsideration.

Possible tests may compare prompts that:

* explicitly request independent reconsideration before the action;
* explicitly prohibit re-solving and ask only for a routing decision;
* or separate re-evaluation from final action.

These manipulations must be designed carefully because telling the model to reconsider changes the task.

This section remains DRAFT until the causal pilot is complete.

---

# 45. STUDY 4C — TOKEN-PROBABILITY BASELINE

For open models, collect a direct model-probability uncertainty signal where technically meaningful.

Potential variables include:

* answer-token log probability;
* sequence probability;
* normalized probability across answer choices for MCQ;
* another principled likelihood-based confidence measure.

The exact measure depends on the task.

Any claim that hidden activations add useful information should compare against this baseline.

---

# 46. STUDY 4D — SIMPLE DIFFICULTY BASELINES

Before interpreting hidden-state information, compare against cheap observables such as:

* reported confidence;
* calibrated reported confidence;
* second reported confidence;
* token probability;
* question category;
* prompt length;
* answer-option entropy where applicable;
* response length where applicable;
* simple task-difficulty measures.

These features are controls, not headline contributions.

They exist to prevent a probe from rediscovering an obvious surface property.

---

# 47. GATE BEFORE MECHANISTIC INTERPRETABILITY

Proceed to Study 5 only if all of the following are reasonably satisfied:

1. at least one open model behaviorally reproduces the relevant confidence-feedback phenomenon;
2. its hidden verification decision carries correctness information beyond verbal confidence;
3. a second confidence sample does not fully explain that information;
4. token-probability baselines do not fully explain it;
5. simple surface/difficulty variables do not fully explain it;
6. enough examples remain to evaluate an internal predictor out of sample.

If these conditions fail:

> stop the mechanistic branch.

Do not include interpretability merely to make the paper appear more technical.

---

# 48. STUDY 5 — CONDITIONAL INTERNAL-STATE ANALYSIS

## 5.1 Goal

Determine whether the open model's pre-decision internal representation contains correctness-relevant information beyond observable uncertainty baselines.

The goal is **not** to prove that the model “knows” the correct answer.

The goal is predictive and mechanistic evidence about information available to the downstream verification process.

---

# 49. STUDY 5A — REPRESENTATION EXTRACTION

Extract hidden representations at a clearly defined pre-action location.

Potential candidate:

> the final token position immediately before the model generates the verification action.

Also consider a small predefined set of middle-to-late layers.

Do not search every layer, token position, and pooling method on the full test set and report only the best one.

Use a development split to choose representation details.

Freeze them before final evaluation.

---

# 50. STUDY 5B — PROBE TARGET

Primary target:

> Stage-1 answer wrong versus correct.

Secondary possible target:

> model's hidden verification action.

The more scientifically important target is actual correctness.

The probe should remain low capacity.

Examples:

* logistic regression;
* linear probe.

A large nonlinear classifier is not preferred because it weakens interpretability and increases overfitting risk.

---

# 51. STUDY 5C — BASELINE HIERARCHY

Compare:

### Baseline 1

Verbal confidence only.

### Baseline 2

Calibrated verbal confidence.

### Baseline 3

Token probability.

### Baseline 4

Repeated verbal confidence.

### Baseline 5

Surface/difficulty variables.

### Baseline 6

All observable baselines combined.

### Model 7

Observable baselines + hidden-state probe.

The internal representation earns inclusion only if Model 7 improves genuinely out-of-sample prediction beyond Model 6.

---

# 52. STUDY 5D — SPLITTING AND LEAKAGE CONTROL

Never evaluate the probe on examples used to train it.

Use grouped splits.

Where possible, hold out:

* question IDs;
* task categories;
* and eventually an entire task family.

If the dataset includes repeated confidence manipulations for the same frozen question, **all variants of that question must remain in the same split**.

Otherwise the probe can memorize question identity.

---

# 53. STUDY 5E — CONDITIONAL CAUSAL STEERING

Only after a stable internal signal is established should causal intervention be considered.

The novel question is not merely:

> “Can steering uncertainty alter abstention?”

The stronger D1-specific question is:

> **Does manipulating the internal error-related signal have a different behavioral effect when explicit confidence is visible than when it is hidden?**

This creates an interaction:

`visibility × internal steering`.

A particularly interesting result would be:

* internal steering strongly affects verification when confidence is hidden;
* the same steering effect weakens when an explicit confidence number is visible.

That would support the interpretation that displayed confidence changes which information controls the action.

If the interaction does not appear:

> do not claim policy locking or internal-signal suppression.

---

# 54. STUDY 6 — ROUTING / MITIGATION

**Status: CONDITIONAL BUT PRACTICALLY VALUABLE.**

The final system-design question is:

> **What should determine verification if raw verbal confidence is insufficient?**

Candidate routing methods include:

1. raw reported confidence;
2. calibrated confidence;
3. hidden natural verification action;
4. repeated self-assessment;
5. combined verbal-confidence + hidden-action policy;
6. token-probability routing for open models;
7. observable signals + internal probe for open models.

All methods must be compared under the same verification budget whenever the comparison is intended to measure routing quality.

---

# 55. STUDY 6 PRIMARY METRIC

The main metric is:

> **fraction of actual errors caught at a fixed verification budget.**

For example:

> At 20% verification coverage, what percentage of wrong answers are sent to checking?

This should be shown as a curve across budgets rather than only at one arbitrary point.

Useful secondary metrics include:

* correct-answer verification rate;
* precision among verified items;
* wrong-unverified rate;
* realized synthetic cost.

---

# 56. HISTORICAL AUTHORITY MANIPULATION

The original V2 crossed human and AI verification authority.

The rebuilt project does not delete this result.

However, authority is no longer on the critical experimental path.

Historical authority findings should be retained for:

* provenance;
* appendix analysis;
* possible secondary discussion.

Do not spend major new API budget reproducing the authority factor unless later evidence provides a direct scientific reason.

---

# 57. CALIBRATION ANALYSIS

Calibration remains important but should not be confused with error ranking.

For each model where confidence is central, report:

* Brier score;
* reliability/calibration curve;
* ECE with clearly specified bins;
* AUROC for correctness or error discrimination;
* confidence distributions for correct and wrong answers.

Where calibrated confidence is used operationally:

* calibration must be cross-fitted;
* fitting and evaluation must use separate folds;
* question identity must define fold membership;
* repeated conditions for one question remain in one fold.

Calibration can change the numerical interpretation of a score.

A monotonic calibrator does not automatically improve ranking.

Keep these concepts separate.

---

# 58. MATCHED-BUDGET ANALYSIS

Whenever natural verification is compared with a confidence-based policy:

1. determine the natural policy's verification rate;
2. give the comparison policy exactly the same expected verification budget;
3. rank answers according to the comparison score;
4. handle confidence ties using principled fractional inclusion or another frozen method;
5. compare error catch rate.

Use the Checkpoint-A tie-handling rule unless a later addendum explicitly improves it:

> answers strictly below the cutoff are included; examples tied at the cutoff share equal fractional inclusion so expected coverage exactly matches the target budget.

This is important because reported confidence is highly discrete.

---

# 59. STATISTICAL PRINCIPLES

## Question is the primary resampling unit

Conditions sharing the same frozen question are paired.

Do not pretend each Stage-3 row is an independent observation.

## Bootstrap

Default:

* 5,000 question-level resamples;
* percentile 95% intervals;
* frozen random seed recorded.

## Multiple conditions

Do not generate dozens of cellwise p-values and selectively discuss favorable ones.

Primary estimands must be defined before confirmatory data collection.

## Model-level heterogeneity

Model families may behave differently.

Do not average away important sign reversals merely to produce a single pooled number.

## Pilot inference

Study 1 is exploratory.

Effect sizes, curves, paired differences, and uncertainty intervals are more useful than binary significance declarations.

---

# 60. HANDLING MODEL HETEROGENEITY

A central lesson of historical V2 is that models may react in opposite directions.

The rebuilt paper should not require every model to share one sign.

For each model report:

* confidence-response curve;
* baseline verification rate;
* threshold responsiveness;
* calibration;
* error-catching behavior.

Any cross-model generalization must be based on fresh evidence.

The failed Checkpoint-A “better signal wins” correlation should not be resurrected merely because one later subset happens to fit it.

---

# 61. PROMPT ROBUSTNESS

Historical V2 already included one paraphrase robustness check.

For the final confirmatory causal study, at least one prompt robustness test should be considered if the main effect appears strongly prompt-dependent.

However, the confirmatory core should not multiply prompt variants unnecessarily.

A useful strategy is:

1. freeze one primary wording;
2. reserve a small independent robustness subset;
3. apply a semantically equivalent paraphrase;
4. test whether effect direction and approximate magnitude persist.

Prompt variants should be created before inspecting their target-model outcomes.

---

# 62. DATA LEAKAGE AND ADAPTIVITY

The project has already used historical V2 outcomes to design the new hypothesis.

Therefore:

* historical V2 is exploratory for the rebuilt claim;
* Checkpoint A is exploratory;
* Study 1 pilot is exploratory.

The final paper must not hide this adaptivity.

Prospective evidence begins only after the final confirmatory protocol is frozen.

Development questions and confirmatory questions should be stored separately.

Where practical:

`data/development/`

and

`data/confirmatory/`.

Do not reuse a question in confirmatory analysis after tuning prompts or thresholds on its pilot behavior.

---

# 63. REQUIRED REPRODUCIBILITY ARTIFACTS

For every new run, store:

* dataset name;
* dataset revision;
* question IDs;
* SHA-256 of frozen question list;
* exact model endpoint;
* provider-returned model metadata;
* API date;
* system prompt;
* user prompt;
* full rendered prompt if possible;
* prompt hash;
* temperature;
* top-p;
* maximum output tokens;
* reasoning settings;
* structured-output settings;
* random seed if available;
* SDK/package versions;
* code commit;
* retry logs;
* raw outputs;
* parsed outputs;
* token usage;
* latency;
* estimated cost;
* run start/end timestamps.

For open models additionally store:

* model repository;
* model commit hash;
* tokenizer commit;
* chat template;
* dtype;
* quantization;
* GPU type;
* inference library and version;
* selected layer indices;
* representation extraction location;
* probe hyperparameters;
* train/validation/test question IDs.

---

# 64. COST AND LATENCY LOGGING

Every API call should record:

* input tokens;
* output tokens;
* provider latency;
* model/provider;
* estimated dollar cost using a versioned pricing table.

Cost summaries should distinguish:

* primary successful calls;
* provider retries;
* parse repairs;
* repeated-sampling runs;
* exploratory micro-controls;
* confirmatory runs.

Summed provider latency is not the same as wall-clock runtime under concurrency.

Report both where possible.

---

# 65. COMPUTE BUDGET PHILOSOPHY

The project should remain gate-based.

Approximate immediate pilot:

**3,920 calls maximum** including the repeat subset.

Do not run an 80,000-call expansion before inspecting this result.

If Study 1 fails, the saved compute is scientifically useful.

If Study 1 succeeds, later budget should be approved study by study.

Mechanistic GPU work should begin with the smallest open model that behaviorally reproduces the phenomenon.

---

# 66. REPOSITORY ORGANIZATION

Recommended research structure:

`docs/`

* `D1_MASTER_RESEARCH_PLAN.md`
* `D1_EXPERIMENTAL_PROTOCOL_v1.0_DRAFT.md`
* later dated confirmatory addenda

`historical/`

* original V2 frozen protocol
* original paper
* original analysis manifests
* Checkpoint A report

`data/`

* historical question lists
* pilot question list
* later confirmatory question list

`prompts/`

* historical Stage-3 templates
* Study-1 pilot templates
* later confirmatory templates

`results/`

* `v2/`
* `checkpoint_A/`
* `study1_causal_pilot/`
* later studies

`analysis/`

* historical analysis
* Checkpoint A
* Study-1 pilot
* future studies

`from_gpt/`

* task instructions created by GPT for Cursor

`to_gpt/`

* Cursor reports, manifests, result summaries, and requested artifacts for GPT review

Do not overwrite historical artifacts when creating new analyses.

---

# 67. FROM_GPT / TO_GPT WORKFLOW PRINCIPLE

The researcher acts as the transport layer between GPT and Cursor.

The workflow should be:

1. GPT creates a numbered task document in `from_gpt/`.
2. The user places that file into the repository.
3. Cursor executes the task using repository context.
4. Cursor creates a matching output folder in `to_gpt/`.
5. The user uploads the primary report to GPT.
6. GPT reviews the result and creates the next task only after the previous result is understood.

Example:

`from_gpt/001_build_study1_pilot.md`

Cursor returns:

`to_gpt/001_study1_pilot_build/`

Then:

`from_gpt/002_run_study1_pilot.md`

Cursor returns:

`to_gpt/002_study1_pilot_results/`.

Tasks should remain small enough that a failed step can be debugged without rerunning the whole research pipeline.

---

# 68. STUDY STATUS LABELS

Every study or artifact must use one of these statuses.

### HISTORICAL-FROZEN

Completed under an earlier frozen protocol.

### EXPLORATORY

Results are used for hypothesis development and design decisions.

### DEVELOPMENT

Used to build or validate measurement/infrastructure.

### CONFIRMATORY-FROZEN

Design was frozen before target data generation.

### CONDITIONAL

Study occurs only if an earlier gate is passed.

### DEFERRED

Potentially useful, but currently outside the critical paper path.

Current status:

* V2-B: `HISTORICAL-FROZEN`
* Checkpoint A: `EXPLORATORY`
* Study 1 causal pilot: `EXPLORATORY`
* Study 1 micro-controls: `CONDITIONAL / EXPLORATORY`
* Study 2: `CONDITIONAL / NOT YET FROZEN`
* Study 3: `CONDITIONAL`
* Study 4: `CONDITIONAL`
* Study 5: `CONDITIONAL`
* Study 6: `CONDITIONAL`

---

# 69. DECISION LOG

Maintain:

`docs/D1_DECISION_LOG.md`.

Every meaningful scientific change should record:

* date;
* previous plan;
* evidence that motivated the change;
* new decision;
* whether the affected study had already been frozen;
* whether existing data are exploratory or confirmatory relative to the new decision.

Examples include:

* changing the confidence grid;
* adding or removing a model;
* changing task family;
* modifying the primary endpoint;
* deciding whether provenance deserves confirmation;
* deciding whether interpretability is warranted.

This prevents later hindsight from rewriting the project's history.

---

# 70. CURRENT CRITICAL PATH

The ICLR-critical path is deliberately narrower than the full research vision.

## Phase 0 — complete

Historical V2 experiment.

## Phase 1 — complete

Checkpoint A.

## Phase 2 — immediate

Build and run the 100-question, two-model causal confidence pilot.

## Phase 3

Review pilot.

Do not scale automatically.

## Phase 4

If pilot succeeds, run the targeted literature/novelty audit.

## Phase 5

Design and freeze the prospective confirmatory causal experiment in a dated addendum.

## Phase 6

Run fresh confirmatory data.

## Phase 7

Add modest generalization if the core result is secure.

## Phase 8

Run Study 4 explanation controls if the residual hidden-decision information remains scientifically important.

## Phase 9

Only if Study 4 leaves a genuine unexplained signal, perform internal-state probing and possible causal steering.

## Phase 10

Add a routing/mitigation result if it provides a clean practical contribution.

The project should not jump directly from Phase 2 to Phase 9.

---

# 71. SUCCESS CRITERIA FOR THE PAPER

The project can produce several levels of paper.

## Behavioral core succeeds if

A prospective experiment cleanly shows that changing only the displayed confidence alters verification decisions.

## Strong behavioral paper if

The causal effect generalizes across more than one model/context and the paper clearly separates:

* calibration;
* checking budget;
* error ranking;
* causal score responsiveness.

## Strong explanatory paper if

The project also identifies why the natural hidden verification judgment contains information beyond the verbal confidence report.

## High-ceiling mechanistic paper if

An open model shows behavioral continuity, internal representations add predictive information beyond strong observable baselines, and causal intervention demonstrates that confidence visibility changes how the internal signal influences verification.

## Strong design paper if

A practical routing method catches more errors than raw verbal confidence at equal oversight cost.

The paper does not need every level to be scientifically worthwhile.

---

# 72. FAILURE CRITERIA

The research program should be willing to conclude that the deeper story is wrong.

Examples:

### Failure mode 1

Changing displayed confidence barely changes verification.

Consequence:

> confidence visibility may not be acting through the numerical score itself.

Revisit the original prompt difference.

### Failure mode 2

The causal effect exists only because the model mechanically executes an explicit mathematical threshold.

Consequence:

> narrow the claim and decide whether qualitative-stakes controls rescue practical relevance.

### Failure mode 3

Provenance does not matter.

Consequence:

> drop the “own confidence” emphasis and frame the result as displayed reliability-score conditioning.

### Failure mode 4

Second confidence sampling explains the hidden decision advantage.

Consequence:

> interpret the effect as repeated evaluation/ensembling rather than hidden richer uncertainty.

### Failure mode 5

Token probabilities explain all residual information.

Consequence:

> do not claim a novel hidden-state signal.

### Failure mode 6

Open models do not reproduce the behavior.

Consequence:

> do not use their internals as an explanation for closed-model results.

### Failure mode 7

Internal probes fail strong baselines.

Consequence:

> omit mechanistic interpretability from the paper.

These outcomes are scientifically informative and should not be hidden.

---

# 73. INTERPRETATION BOUNDARIES

The experimental manipulation studies model behavior under controlled synthetic decisions.

It does not directly measure:

* human trust;
* real-world harm;
* institutional oversight;
* moral responsibility;
* consciousness;
* whether an AI should normatively decide its own oversight;
* actual production-agent deployment.

The verification cost and error cost are stylized experimental variables.

They should not be interpreted as literal real-world monetary or safety harms.

The confidence value is an elicited model report.

It is not direct access to a latent probability in the network.

A hidden verification action is behavior.

It is not automatically an internal state.

An open-model probe establishes predictive information in that model.

It does not establish the same mechanism in closed models.

---

# 74. IMMEDIATE EXECUTION TASK

The next engineering action after this protocol is adopted is **not** to run the entire research program.

The next task is:

> **Build the Study-1 causal-pilot pipeline and dry-run it locally / on a tiny smoke test without starting the full 3,920-call experiment.**

The first Cursor task should therefore:

1. create the `from_gpt/` and `to_gpt/` structure if absent;
2. inspect and reuse the existing V2 Stage-3 pipeline;
3. create a new pilot configuration without modifying historical files;
4. deterministically select and freeze the 100 pilot question IDs;
5. implement separate `reported_confidence` and `displayed_confidence` fields;
6. implement the two error-cost conditions;
7. implement the five manipulated-score conditions plus hidden and true-visible baselines;
8. produce rendered prompt examples for manual audit;
9. run only a tiny smoke test;
10. return the implementation report to GPT before the actual pilot is launched.

No full pilot should begin until the rendered prompts and data schema have been manually inspected.

---

# 75. FINAL PROTOCOL PRINCIPLE

The project should optimize for **sequential evidence**, not experimental spectacle.

The research process should be:

> observe
> → identify the simplest alternative explanation
> → test it cheaply
> → scale only if the phenomenon survives
> → test another explanation
> → move inside the model only after observable explanations are exhausted.

The original V2 experiment discovered the phenomenon.

Checkpoint A explained part of it and prevented an unjustified mechanistic story.

Study 1 now tests whether the displayed number itself causally controls verification.

Everything downstream must earn its place from that result.

That is the execution philosophy for D1.
