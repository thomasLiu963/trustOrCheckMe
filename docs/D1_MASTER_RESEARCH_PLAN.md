D1 MASTER RESEARCH PLAN

Working project name: Confidence as a Control Signal for LLM Verification
Working title family: When Confidence Controls Oversight
Plan version: v1.1 — post-Checkpoint-A, pre-causal-pilot
Date: 2026-09-17
Status: Current strategic research plan for rebuilding the original workshop project into a stronger ICLR-oriented research program.

This document is NOT a preregistration. It records the scientific vision, the historical evidence that motivated the new direction, the current interpretation after Checkpoint A, the planned studies, the claims each study is allowed to support, and the conditions under which later studies should or should not be run.


================================================================================
0. PURPOSE OF THIS MASTER PLAN
================================================================================

This document is the strategic source of truth for D1.

It is intentionally broader than an experimental protocol. The future experimentalProtocol will specify exact prompts, model identifiers, datasets, sampling rules, statistical tests, file formats, and stopping criteria. This Master Plan instead explains what the project is trying to discover and why each part of the research program exists.

A reader returning to this project months later should be able to use this document to answer all of the following questions:

1. What was the original paper about?
2. What did the original experiment actually find?
3. What did we originally think those findings meant?
4. What did Checkpoint A later show?
5. Which earlier hypotheses survived, and which ones were weakened or rejected?
6. What is the larger scientific problem we now care about?
7. Why is that problem important outside this one benchmark?
8. What would be genuinely novel relative to existing work?
9. What contribution could the paper make if the future studies succeed?
10. Which experiments are essential, and which are optional?
11. What evidence would justify moving into mechanistic interpretability?
12. What evidence would tell us to stop or change direction?
13. How should the paper ultimately be framed and marketed to an ICLR audience?

The project should not accumulate experiments merely because they sound technical. Every study must earn its place by doing at least one of the following:

- establishing the central phenomenon more convincingly;
- ruling out a plausible simpler explanation;
- explaining why the phenomenon occurs;
- showing that the phenomenon generalizes beyond the original setup;
- connecting the phenomenon to a practically important AI-system design problem; or
- providing a useful alternative design or mitigation.

If an experiment does none of these, it should not be included.


================================================================================
1. THE CENTRAL PROBLEM
================================================================================

The simplest version of the question is:

“When an AI has to decide whether one of its own answers should be checked, should that decision depend on the confidence score the AI previously reported?”

Suppose an AI answers a question and then says:

“I am 95% confident this answer is correct.”

Later, before the answer is used, the same system must choose between:

USE THE ANSWER WITHOUT CHECKING

and

VERIFY THE ANSWER FIRST.

The most obvious design is to make the verification decision depend on the confidence score. Low confidence should lead to checking; high confidence should lead to trusting the answer.

D1 asks whether this seemingly natural design is actually reliable.

More precisely, the project studies:

“What changes when an LLM’s previously generated confidence report is explicitly fed back into a later decision about whether its already-generated answer should receive independent verification?”

The key conceptual distinction is between:

confidence as a measurement

and

confidence as a control signal.

A confidence score is a measurement when we merely observe it.

It becomes a control signal when that score is fed back into a system and helps determine what the system does next.

D1 is about the consequences of that transition.


================================================================================
2. WHY THIS PROBLEM MATTERS
================================================================================

2.1 AI systems increasingly have to decide when to seek additional scrutiny

The broader motivation is not simply that people like confidence scores.

Modern AI systems are increasingly used in workflows where the system must decide not only what answer to produce, but also whether that answer is good enough to act on.

An increasingly capable system may have to decide whether to:

- answer directly;
- search for more information;
- call a retrieval system;
- use an external tool;
- run a test;
- ask another model;
- spend additional reasoning compute;
- defer to a stronger model;
- escalate to a human;
- abstain;
- or independently verify its own answer.

This creates a second-order problem:

Who or what decides which outputs receive additional scrutiny?

It is rarely practical to verify everything. Verification consumes money, latency, compute, human attention, or other resources.

Therefore, the real system-design problem is selective:

Given a limited verification budget, which outputs deserve to be checked?

Uncertainty is a natural candidate for making that decision.

That is why the quality of an uncertainty signal matters not only as a descriptive property of the model, but as an input into resource allocation and oversight.


2.2 Selective oversight is different from ordinary answer accuracy

A model may be highly accurate overall while still being poor at knowing which individual answers deserve checking.

Similarly, a model may have a confidence score that looks reasonable in aggregate while still ranking its own mistakes poorly.

For a selective-oversight system, the important question is not merely:

“How often is the model correct?”

It is:

“When the model is wrong, does the system successfully route those mistakes toward verification?”

That is why D1 places major emphasis on error-catching under a fixed verification budget.

For example, suppose two policies are each permitted to verify exactly 20% of answers.

The useful policy is the one that puts more actual mistakes inside that 20%.

A system that verifies more answers may catch more errors simply because it spends more resources. That does not mean it is better at identifying which answers are risky.

The distinction between checking more and checking better became one of the main lessons of Checkpoint A.


================================================================================
3. WHAT IS ARTIFICIAL ABOUT OUR SETUP, AND WHAT IS PRACTICALLY REAL
================================================================================

The project should not exaggerate the prevalence of the exact implementation used in the original experiment.

We currently do not have evidence that real production systems commonly use the exact sequence:

1. ask an LLM for a numerical confidence percentage;
2. save that percentage;
3. paste it back into the same LLM;
4. ask the model to decide whether verification is needed.

The experimental setup is intentionally controlled.

Its purpose is to isolate a broader design question that is practically relevant:

What happens when a model-generated uncertainty estimate becomes an explicit downstream routing or oversight signal?

The broader problem is real because uncertainty is actively studied as a trigger for abstention, escalation, retrieval, additional reasoning, tool use, model cascading, and selective verification.

The exact self-feedback design is our controlled way of studying a fundamental issue inside that broader system architecture.

The paper should therefore say:

“We study a controlled instance of uncertainty-conditioned oversight.”

It should not say:

“This exact confidence-feedback architecture is already universally deployed.”


================================================================================
4. HISTORICAL FOUNDATION: THE ORIGINAL V2 EXPERIMENT
================================================================================

The rebuilt D1 project did not begin from a theoretical hypothesis created after reading the literature.

It began from an empirical result in the original workshop experiment.

Preserving this history matters for both scientific honesty and future bookkeeping.


4.1 Original research question

The original paper was titled:

“Trust Me or Check Me? Self-Reported Confidence, Verification, and Decision Authority in Language Models.”

Its broad question was:

When should users trust an AI answer, and when should that answer be independently verified?

The experiment studied whether an AI system’s own self-reported confidence could guide that choice.

The original design separated three stages.


Stage 1 — Answer

The model answered a multiple-choice question.

That answer was then frozen.


Stage 2 — Confidence

The model reported a numerical probability that the frozen answer was correct.

For example:

0.92 = 92% confidence.

That confidence value was also frozen.


Stage 3 — Verification decision

The model could no longer change the answer.

It had to choose between:

USE_UNVERIFIED

or

VERIFY_FIRST.

The experiment manipulated:

- whether the model could see its previously reported confidence;
- whether verification authority was framed as belonging to a human or an AI system;
- and how costly an unverified wrong answer was.

The primary dataset used 500 MMLU-Pro questions across four model families and produced 32,000 Stage-3 verification decisions.


================================================================================
5. ORIGINAL MODEL PANEL
================================================================================

The four model families were:

- Claude Sonnet 5;
- Gemini 3.8 Flash;
- GPT-5.6 Sol;
- Grok 4.20 non-reasoning.

Each model answered the same 500-question primary set.

Their Stage-1 answer accuracies differed substantially:

- Gemini: 88.0%;
- GPT-5.6 Sol: 82.6%;
- Claude Sonnet 5: 74.2%;
- Grok 4.20: 64.4%.

The corresponding numbers of wrong answers were:

- Gemini: 60;
- GPT: 87;
- Claude: 129;
- Grok: 178.

This mattered because a verification policy can only catch mistakes that actually occur, and models with different base error rates create different error-catching opportunities.


================================================================================
6. WHAT THE ORIGINAL PAPER FOUND BEFORE CHECKPOINT A
================================================================================

6.1 Showing confidence strongly changed verification behavior

The first major result was that simply showing a model the confidence value it had already generated could substantially change whether it requested verification.

However, the direction of the effect was very different across models.

Claude became more verification-prone when its confidence was visible.

Across the eight authority-by-stakes conditions, visible confidence increased Claude’s verification rate by roughly 12.8 to 22.4 percentage points.

GPT moved strongly in the opposite direction.

Showing GPT its confidence reduced verification by approximately 16.2 to 20.8 percentage points across all eight conditions.

Grok also became less verification-prone, with decreases of approximately 8.6 to 16.4 points.

Gemini behaved differently again. At low stakes it changed little or verified slightly less, while at larger error costs it became substantially more verification-prone.

This was already important because the intervention was extremely small:

the answer was unchanged;

the confidence value was unchanged;

the decision problem was unchanged;

only the visibility of the already-generated confidence differed.

Yet verification rates changed by around twenty percentage points in some cases.

This established the original observation:

A model’s own confidence report is not behaviorally neutral when it is surfaced during a later decision.


================================================================================
7. THE ORIGINAL GPT PARADOX
================================================================================

GPT produced the most striking original result.

When GPT’s confidence was hidden, its final verification decisions disagreed with the simple policy implied by its confidence score on roughly 19.6% to 27.2% of questions.

When the exact same frozen confidence value was shown to GPT, disagreement dropped to roughly 0.2% to 2.2%.

In other words:

GPT became almost perfectly consistent with the decision rule implied by its own stated confidence once that confidence was explicitly visible.

Its decisions also became much more orderly as stakes increased.

With confidence hidden, 95 out of every 1,000 owner-specific decision trajectories contained at least one strange reversal where GPT requested verification at a lower error cost but stopped requesting verification when the cost of being wrong increased.

With confidence visible, this dropped to roughly 1 in 1,000.

From the perspective of behavioral consistency, visible GPT looked dramatically better.


================================================================================
8. YET GPT BECAME WORSE AT CATCHING ACTUAL ERRORS
================================================================================

The strange part was that this dramatic increase in consistency did not produce better oversight.

At the highest tested error cost, among GPT’s answers that were actually wrong:

- under human authority, the fraction left unverified rose from 32.18% to 56.32%;
- under AI authority, it rose from 33.33% to 54.02%.

At the same time, GPT verified fewer answers that were actually correct.

So the visible-confidence policy was not simply random or broken. It became less conservative overall: fewer useful checks were performed, but fewer unnecessary checks were also performed.

Under the experiment’s synthetic cost model, the total decision cost worsened substantially at high stakes.

This produced the original workshop paper’s most important empirical statement:

A model can become much more consistent with its own stated confidence while becoming worse at catching its actual mistakes.

That result remains true.

Checkpoint A changed our explanation of it, not the original observation itself.


================================================================================
9. ORIGINAL FINDINGS FOR THE OTHER MODELS
================================================================================

The original experiment also showed that the GPT result was not a universal law.

At high stakes:

- visible confidence generally made Claude verify substantially more;
- visible confidence made Gemini verify more at higher error costs;
- visible confidence made GPT verify substantially less;
- visible confidence also reduced verification for Grok.

The high-stakes consequences therefore differed across models.

For example, at L = 20, visible confidence reduced the percentage of Claude and Gemini errors that went unchecked, while increasing unchecked errors for GPT and Grok.

This meant the original evidence did not support the simplistic claim:

“Showing confidence is harmful.”

A more accurate statement was:

Showing self-reported confidence can strongly alter verification policy, but the direction and consequences of that alteration are model-dependent.

This heterogeneity eventually became important when we began searching for a more general explanation.


================================================================================
10. ORIGINAL AUTHORITY RESULT
================================================================================

The original frozen protocol gave substantial attention to whether verification authority was framed as belonging to:

- a human;
- or an AI system.

That factor turned out to be much weaker and less universal than confidence visibility.

Claude showed the clearest directional authority effect.

The remaining model families showed smaller or mixed average effects.

There were sometimes meaningful question-level differences even when average verification rates were similar, but the authority manipulation did not produce the same clear cross-model story as confidence visibility.

This matters historically because the rebuilt project should not pretend authority was never important.

The correct history is:

Authority was part of the original central design, but the observed visibility effect turned out to be substantially stronger and more scientifically interesting.

The current D1 project therefore demotes authority to secondary historical evidence rather than continuing to spend new experimental budget on it.


================================================================================
11. ORIGINAL PROMPT-ROBUSTNESS EVIDENCE
================================================================================

The original project also included a 100-question paraphrase robustness study.

The main purpose was to test whether the visible-versus-hidden effect depended entirely on one particular Stage-3 wording.

For Claude, GPT, and Grok, the direction of the visibility effect remained the same across all tested cells.

Gemini matched in 7 out of 8 cells.

However, the magnitude of the effects changed under paraphrasing.

Therefore the correct interpretation is:

The major effect directions were not unique to one exact prompt wording, but the size of the effects was prompt-sensitive.

The robustness study does not justify claiming full prompt invariance.


================================================================================
12. ORIGINAL PAPER INTERPRETATION AT SUBMISSION TIME
================================================================================

The original paper’s central interpretation was approximately:

Self-reported confidence is not merely an uncertainty report; once explicitly surfaced, it can act as a downstream control signal that changes whether the model recommends verification.

The GPT result then created a particularly provocative observation:

better agreement with the confidence-implied policy did not necessarily mean better verification.

That is a valid empirical observation.

However, at workshop-submission time, we did not yet know exactly why visible GPT performed worse.

Several explanations were possible.

One tempting interpretation was:

perhaps the model had some richer natural judgment about suspicious answers, and showing its numerical confidence caused it to rely too heavily on the number instead.

This possibility motivated later discussion about information bottlenecks and interpretability.

But it was not directly established by the original experiment.


================================================================================
13. HISTORICAL NOTE ON HYPOTHESIS EVOLUTION
================================================================================

The original frozen protocol did not begin with confidence visibility as the sole primary question.

The protocol originally treated decision authority as the primary question and confidence visibility as a mechanism question.

After the data were observed, the manuscript shifted emphasis toward confidence visibility because that effect was substantially larger and more consistent than the authority effect.

The original paper explicitly disclosed this change in emphasis.

This history should remain visible in the rebuilt project.

The correct provenance is:

Original design
→ surprising visibility result
→ workshop manuscript emphasizes visibility
→ post-submission analysis asks what visibility actually did
→ new causal research program emerges.

We should not rewrite this sequence as though the final D1 hypothesis was preregistered from the beginning.


================================================================================
14. CHECKPOINT A: WHY IT WAS RUN
================================================================================

After the workshop submission, we asked a more demanding question:

Did hidden GPT actually know better which answers deserved checking, or did it simply verify more answers overall?

This distinction is crucial.

Suppose:

hidden GPT checks 40% of answers;

visible GPT checks 20%.

The hidden condition will naturally catch more mistakes simply because it spends twice as much verification.

That would not show that hidden GPT is better at identifying which answers are risky.

Therefore Checkpoint A compared different policies at the same verification budget.

It also asked whether GPT’s poor visible performance was mainly caused by bad calibration of the confidence score.

Finally, it tested whether the hidden CHECK/USE decision contained any correctness information after the numerical confidence was already known.


================================================================================
15. CHECKPOINT A DATA INTEGRITY
================================================================================

Checkpoint A reused the frozen V2-B experiment.

The primary data contained:

- 500 questions;
- 4 models;
- 2 authority conditions;
- 2 confidence-visibility conditions;
- 4 error-cost levels;
- 32,000 unique primary Stage-3 decisions.

The analysis confirmed:

- no missing primary cells;
- no duplicate primary cells;
- no Stage-1/Stage-2 join failures;
- consistent frozen answers;
- consistent frozen confidence values.

No missing values were imputed.


================================================================================
16. CHECKPOINT A FINDING 1:
VISIBLE GPT IS ALMOST A MECHANICAL CONFIDENCE RULE
================================================================================

The most important reinterpretation concerned GPT.

When GPT sees its confidence, its final action agrees with the simple threshold rule implied by the raw confidence on approximately 99.2% of decisions.

This means the original visible-GPT result does not require a complicated internal explanation.

The first-order story is:

show GPT its confidence
→ GPT almost mechanically follows that score.

GPT’s reported confidence is also strongly overconfident.

Its wrong answers still have an average reported confidence of approximately 0.86.

Therefore, at high stakes, a policy that faithfully follows the raw score can still fail because the score itself is too optimistic.

Checkpoint A concluded:

A large part of the GPT high-stakes failure is explained by following an overconfident score and therefore verifying too little.

This is simpler than the earlier speculative “rich signal suppression” explanation.


================================================================================
17. CHECKPOINT A FINDING 2:
EQUAL-BUDGET ERROR RANKING
================================================================================

Checkpoint A compared the hidden verification policy with a confidence-only policy while giving both the same number of checks.

This asks:

Which method is better at choosing the mistakes?


17.1 Claude

Claude produced the clearest result.

Its hidden policy caught approximately 88.3% of its wrong answers at its average checking budget.

A confidence-only ranking using the same budget caught approximately 78.6%.

The mean advantage was approximately 9.6 percentage points, with a 95% confidence interval of approximately [3.2, 16.0].

This is the cleanest existing evidence that Claude’s natural verification decisions contain useful ranking information that is not captured by verbal confidence alone.


17.2 GPT

GPT’s hidden policy also had a positive point estimate:

hidden error catch: approximately 61.1%;

confidence-only catch at the same budget: approximately 53.9%;

difference: approximately +7.1 percentage points.

However, the confidence interval was approximately [-2.4, 16.4], which includes zero.

Therefore we may not currently claim that GPT’s hidden verification policy is reliably better at ranking errors than its confidence score.


17.3 Gemini

Gemini’s estimate was small and inconclusive.


17.4 Grok

Grok’s hidden policy verified almost everything, which created a ceiling where equal-budget ranking comparisons were not informative.


================================================================================
18. CHECKPOINT A FINDING 3:
CALIBRATION EXPLAINS MUCH OF GPT’S COST FAILURE
================================================================================

GPT’s raw confidence values were substantially overconfident.

Cross-fitted calibration reduced the calibration error considerably.

At large error costs, the calibrated confidence policy would verify dramatically more answers than visible GPT.

For example, at L = 20:

- hidden GPT verified roughly 38%;
- visible GPT verified roughly 22%;
- the raw-confidence threshold verified roughly 24%;
- the calibrated-confidence rule wanted to verify roughly 96%.

The calibrated rule achieved much lower synthetic cost.

Therefore:

GPT’s visible high-stakes failure is largely a calibration-and-verification-budget problem.

This finding weakens the claim that GPT visible behavior is harmful because it discards a better ranking signal.

That stronger interpretation remains unproven for GPT.


================================================================================
19. CHECKPOINT A FINDING 4:
HIDDEN GPT AND CLAUDE DECISIONS STILL CONTAIN EXTRA INFORMATION
================================================================================

This is the most important surviving clue for the deeper research direction.

Checkpoint A asked:

If we already know the model’s numerical confidence, does knowing its hidden CHECK/USE decision tell us anything additional about whether the answer is actually wrong?

For GPT and Claude, the answer was yes.

Out-of-sample prediction improved when the hidden verification decision was added to the reported confidence.

The most intuitive example comes from GPT.

GPT gave 247 answers exactly 99% confidence.

In one representative condition:

- the answers hidden GPT chose to verify were wrong 26.2% of the time;
- the answers hidden GPT chose to use without verification were wrong 5.5% of the time.

Every one of those answers had the same spoken confidence:

99%.

Yet the actual error risk differed by roughly fivefold depending on the hidden verification action.

Claude showed the same general phenomenon.

The correct claim is therefore:

For GPT and Claude, the natural hidden verification decision contains correctness-relevant information not fully captured by the reported confidence number.

This does NOT establish why that information exists.

It may come from:

- re-evaluating the question;
- another noisy confidence estimate;
- token probabilities;
- richer internal representations;
- or some other feature of the Stage-3 reasoning process.


================================================================================
20. CHECKPOINT A FINDING 5:
ONE PROPOSED GENERAL LAW FAILED
================================================================================

We considered a simple theory:

If confidence is a better error signal than the model’s natural verification judgment, showing confidence should help; if the natural judgment is better, showing confidence should hurt.

Checkpoint A did not find a useful cross-model relationship of this form.

Across the 32 model × authority × stakes cells, the descriptive relationship was extremely weak.

The pattern also differed within individual model families.

This hypothesis should therefore NOT become the central paper story unless future independent data unexpectedly support a refined version.

The current project should treat this early theory as rejected by exploratory evidence.


================================================================================
21. WHAT WE BELIEVE AFTER CHECKPOINT A
================================================================================

The evidence currently supports the following statements.


SUPPORTED CLAIM 1

Showing a model its previously reported confidence can strongly change whether it requests verification.

This was established by the original experiment.


SUPPORTED CLAIM 2

Visible GPT almost perfectly follows the simple rule implied by its shown confidence.

This was clarified by Checkpoint A.


SUPPORTED CLAIM 3

GPT’s raw confidence is substantially overconfident, and this explains much of the high-stakes visible-cost failure.

This was clarified by Checkpoint A.


SUPPORTED CLAIM 4

Claude’s natural hidden verification policy catches more errors than verbal-confidence ranking at the same number of checks.

This is supported by the equal-budget analysis.


SUPPORTED CLAIM 5

GPT and Claude hidden CHECK/USE decisions contain correctness information not fully captured by the reported confidence number.

This is supported by out-of-sample prediction and exact-confidence comparisons.


================================================================================
22. WHAT WE DO NOT CURRENTLY KNOW
================================================================================

We do not yet know:

- whether the displayed confidence number itself causally controls verification;
- whether any displayed number would do the same thing;
- whether “this is your own confidence” matters;
- whether the phenomenon generalizes beyond the original task;
- whether it generalizes to open-weight models;
- whether the hidden extra information comes from re-evaluation;
- whether a second confidence sample explains it;
- whether token probabilities explain it;
- whether hidden activations contain additional information beyond those observables;
- whether showing confidence changes the use of internal information;
- whether a better routing policy can recover more errors under the same verification budget.

These unknowns define the future study program.


================================================================================
23. UPDATED CENTRAL SCIENTIFIC QUESTION
================================================================================

The current project should not ask merely:

“Is self-reported confidence calibrated?”

Nor should it ask:

“Does confidence influence behavior?”

Those questions already have substantial prior work.

The current central question is:

When a language model must decide whether its own frozen answer deserves verification, how does explicitly displaying its prior confidence report change that decision, and what information does the model use when confidence is hidden versus visible?

The project then asks a sequence of narrower questions:

1. Does changing the displayed confidence itself causally change verification?
2. Does the source or provenance of the confidence matter?
3. When confidence is hidden, what additional information guides the natural verification decision?
4. Does the phenomenon generalize?
5. Can we build a better routing rule?


================================================================================
24. SCIENTIFIC VISION
================================================================================

The project’s broader scientific concept is:

An uncertainty estimate is not necessarily passive once it is fed back into the system that produced it.

A model may produce a confidence score as a compressed summary of uncertainty.

That score may then be used as an input into the next decision.

Once that happens, the score becomes part of the system’s decision process.

The key possibility is that:

the act of making an uncertainty estimate explicit may change how the model itself allocates verification.

This is different from merely asking whether the score is statistically calibrated.

D1 is therefore about feedback from uncertainty measurement into downstream behavior.


================================================================================
25. THE PRACTICAL NARRATIVE
================================================================================

The project should be framed around a transition from answer generation to selective AI oversight.

A simple language model only needs to answer a question.

A more autonomous AI system must also decide:

“Can I proceed with this answer, or should I seek additional scrutiny?”

That is a fundamental requirement for systems that operate with limited human oversight.

Confidence is an attractive routing signal because it is cheap and available.

But D1 raises an important design warning:

Before using self-reported confidence to control oversight, we need to know what feeding that score back into the model does to the model’s decision process.

The original experiment already shows that the behavior can change drastically.

The rebuilt project aims to determine whether those changes are useful, harmful, or merely different—and why.


================================================================================
26. INTENDED NOVELTY
================================================================================

The project should not claim novelty for ideas that are already established.

The novelty must come from the combination of the following elements.


26.1 Confidence visibility as a causal intervention

The original experiment compared confidence hidden versus visible.

The next study will go further by manipulating the numerical value itself while holding the frozen answer constant.

This asks:

Does the displayed confidence score causally steer verification?


26.2 Separation between explicit confidence and natural verification behavior

Checkpoint A suggests that the model’s natural verification decision can carry information not present in the explicit confidence number.

The project will study the relationship between:

what the model SAYS about confidence

and

what the model NATURALLY DOES when deciding whether verification is needed.

That specific contrast is more interesting than confidence calibration alone.


26.3 Confidence as a feedback/control signal

Most uncertainty work treats confidence as an estimate to be evaluated.

D1 studies what happens when the estimate is fed back into a later decision.

The broader contribution is therefore about:

measurement becoming control.


26.4 Conditional mechanistic novelty

If later experiments justify interpretability, the possible mechanistic novelty is NOT:

“activations predict correctness.”

That is already known.

The stronger question is:

Does displaying an explicit confidence report change how strongly other error-relevant internal signals influence the verification decision?

That interaction would be a much more distinctive mechanistic contribution.


================================================================================
27. STUDY ARCHITECTURE
================================================================================

The rebuilt project should progress through incremental gates.


================================================================================
28. STUDY 0 — HISTORICAL V2 EXPERIMENT
================================================================================

Status: COMPLETE.

Study 0 is the original workshop experiment.

Its scientific role in the rebuilt project is to establish the discovery observation:

confidence visibility substantially changes verification policy, with large and model-specific effects.

Study 0 should not be represented as prospectively testing the final rebuilt D1 theory.

Its role is discovery and hypothesis generation.


================================================================================
29. STUDY 0B — CHECKPOINT A
================================================================================

Status: COMPLETE.

Checkpoint A is the post-submission exploratory analysis.

Its role is to separate:

- checking more from checking better;
- calibration from error ranking;
- the confidence score from the natural hidden action.

Its main consequence is that it weakens the simple “GPT loses a better signal” story while preserving a more careful research question about what information drives hidden verification.


================================================================================
30. STUDY 1 — SMALL CAUSAL CONFIDENCE PILOT
================================================================================

Status: NEXT.

This should be a deliberately small experiment.

The purpose is NOT to produce the final paper result.

The purpose is to answer one decisive question cheaply:

If we change only the confidence number shown to the model, does its verification decision change?

For example, take the same frozen answer and show:

70%;
80%;
90%;
95%;
99%.

If the model systematically changes CHECK/USE behavior with the displayed score, we have direct evidence that the number itself acts as a causal control input.

If it does not, our current framing is weakened substantially.


================================================================================
31. STUDY 1 PILOT PHILOSOPHY
================================================================================

The pilot should be intentionally small enough to finish quickly.

A provisional scale might be approximately:

- 100 questions;
- 2 models;
- 2 stakes settings;
- approximately 5 displayed confidence values.

The exact design belongs in the Experimental Protocol.

The guiding principle is:

Use the smallest experiment capable of telling us whether the central causal premise is worth pursuing.

We should not run the full study before learning this.


================================================================================
32. GATE 1 — CAUSAL PILOT DECISION
================================================================================

If changing the displayed number produces little or inconsistent behavioral change:

STOP and reconsider whether confidence itself is the causal driver.

If changing the displayed number strongly and systematically changes verification:

PROCEED to a fresh prospective confirmatory design.

The pilot is exploratory.

The later confirmation must use fresh data.


================================================================================
33. STUDY 2 — PROSPECTIVE CAUSAL CONFIRMATION
================================================================================

Status: CONDITIONAL ON STUDY 1.

Study 2 should prospectively test:

Changing displayed confidence while holding the answer fixed causally changes verification behavior.

The exact:

- models;
- confidence manipulations;
- stakes;
- question set;
- analysis;
- exclusions;
- stopping rules

should be frozen before fresh outputs are collected.

This study should provide the main confirmatory behavioral result for the rebuilt paper.


================================================================================
34. STUDY 2A — PROVENANCE OF THE DISPLAYED SCORE
================================================================================

A key secondary question is whether the effect depends on the number being framed as the model’s OWN previous confidence.

Possible conditions may include:

“Your earlier confidence was 90%.”

“Another system estimates 90%.”

“Estimated probability of correctness: 90%.”

The number itself should be matched.

This distinguishes two possible phenomena.


SELF-CONDITIONING

The model gives special weight to its own earlier report.


GENERIC SCORE CONDITIONING

The model reacts similarly to any explicit numerical reliability signal.

Either finding is scientifically useful.

But the paper’s language should depend on which one is observed.

If provenance does not matter, the project should stop emphasizing “own confidence.”


================================================================================
35. STUDY 2B — CONTRADICTION CONDITIONS
================================================================================

A particularly informative experiment is to show a confidence score that conflicts with the apparent difficulty or plausibility of the answer.

For example:

- an easy answer paired with artificially low displayed confidence;
- a suspicious answer paired with artificially high displayed confidence.

The scientific question is:

When the shown confidence conflicts with other evidence available to the model, which signal controls the verification decision?

This directly probes whether the model treats the score as authoritative or remains willing to override it.


================================================================================
36. STUDY 2C — QUALITATIVE-STAKES CONTROL
================================================================================

The original experiment gives numerical costs.

A reviewer could therefore argue that the model is merely following an arithmetic formula.

A useful control may instead describe stakes qualitatively:

“An unchecked wrong answer would be very costly.”

without giving the model an explicit numerical equation.

If displayed confidence still strongly controls verification, the result is less easily dismissed as arithmetic instruction-following.

This control should remain small unless the pilot shows it is informative.


================================================================================
37. STUDY 3 — GENERALIZATION
================================================================================

Status: CONDITIONAL ON A SUCCESSFUL CAUSAL RESULT.

The project should then test whether the phenomenon survives beyond:

- one task;
- one prompt structure;
- closed APIs.

This should remain lean.

The project does not need a twenty-model leaderboard.


================================================================================
38. STUDY 3A — OPEN-WEIGHT BEHAVIORAL CONTINUITY
================================================================================

Add approximately one or two open-weight model families.

Before inspecting internal states, run the same behavioral experiment.

The open model must first demonstrate sufficiently similar behavior to justify asking mechanistic questions about it.

If an open model does not show the relevant confidence-visibility effect, it may still serve as a contrastive case, but its internal mechanisms cannot be presented as an explanation for GPT or Claude.


================================================================================
39. STUDY 3B — SECOND TASK FAMILY
================================================================================

The current primary benchmark is multiple-choice.

Add approximately one second task family with objectively checkable outputs.

Mathematics is currently the preferred first choice because it is:

- not multiple-choice;
- relatively easy to score;
- inexpensive;
- compatible with frozen-answer designs.

Code with unit tests could become a later extension if time permits, but it should not delay the core project.


================================================================================
40. STUDY 4 — EXPLAIN THE EXTRA INFORMATION IN HIDDEN VERIFICATION
================================================================================

Status: CONDITIONAL.

Checkpoint A established that GPT and Claude’s hidden decisions contain some correctness information beyond the verbal confidence number.

That does not tell us where the information comes from.

Study 4 should test simple explanations before mechanistic interpretability.


================================================================================
41. STUDY 4A — SECOND-LOOK EXPLANATION
================================================================================

In Stage 3, the model sees the question again.

Perhaps the hidden verification decision is useful because the model simply performs a second assessment.

A second noisy assessment can contain information not captured by the first confidence estimate.

A control should therefore ask for a second independent confidence estimate.

Then compare:

- first confidence;
- second confidence;
- hidden CHECK/USE action.

If the second confidence estimate captures most of the extra information, the phenomenon may be repeated evaluation or ensembling rather than a deeper hidden-state effect.

That is still interesting, but the explanation changes.


================================================================================
42. STUDY 4B — TOKEN-PROBABILITY EXPLANATION
================================================================================

For open models, answer-token probability or another direct model probability may contain information that verbal confidence misses.

Therefore any deeper internal-analysis claim must first compare against:

- verbal confidence;
- calibrated verbal confidence;
- repeated verbal confidence;
- token probabilities;
- basic question features.

If those simple signals explain the hidden verification advantage, interpretability is unnecessary.


================================================================================
43. STUDY 4C — RE-EVALUATION SUPPRESSION HYPOTHESIS
================================================================================

Another possibility is:

hidden Stage 3 encourages the model to reconsider the answer, while visible confidence causes it to stop reconsidering and simply follow the displayed score.

This is different from an information-compression explanation.

It is closer to:

the displayed score anchors the decision and suppresses re-evaluation.

If the evidence points in this direction, the paper should describe the phenomenon accurately rather than forcing an “internal uncertainty bottleneck” story.


================================================================================
44. GATE 4 — SHOULD INTERPRETABILITY HAPPEN?
================================================================================

Interpretability work should occur only if the following remain true after Study 4:

1. Hidden verification still contains useful correctness information beyond verbal confidence.
2. Repeated verbal confidence does not fully explain it.
3. Token probabilities do not fully explain it.
4. Simple surface features do not fully explain it.
5. At least one open model reproduces the relevant behavioral phenomenon.

If these conditions are not satisfied:

DO NOT force a mechanistic section into the paper.

A rigorous behavioral paper is preferable to decorative interpretability.


================================================================================
45. STUDY 5 — MECHANISTIC INTERPRETABILITY
================================================================================

Status: OPTIONAL HIGH-CEILING EXTENSION.

The primary mechanistic question would be:

Does the open model contain an internal error-relevant signal beyond observable confidence measures, and does showing explicit confidence change how strongly that signal influences verification?

This is more specific than generic error probing.


================================================================================
46. STUDY 5A — INTERNAL PREDICTION
================================================================================

Extract a suitable pre-decision representation from an open model.

Test whether a low-capacity probe predicts:

- actual correctness;
- or verification need;

beyond:

- verbal confidence;
- calibrated confidence;
- token probability;
- repeated confidence;
- simple difficulty features.

Evaluation must be out of sample.

Family/task/group leakage should be avoided.

A probe only matters if it beats these strong baselines.


================================================================================
47. STUDY 5B — CAUSAL INTERACTION
================================================================================

If a stable internal signal exists, the most interesting mechanistic experiment is NOT merely:

“steering the signal changes verification.”

Prior work already makes similar claims for confidence and abstention.

The stronger experiment is:

Does manipulating the internal signal have a different effect when the external confidence score is visible than when it is hidden?

If internal steering strongly changes verification when confidence is hidden but matters less when confidence is visible, that would support the claim that displaying the score changes which information controls the decision.

That would materially elevate the paper.


================================================================================
48. STUDY 6 — BETTER ROUTING
================================================================================

The final practical question is:

If raw verbal confidence is not always the best way to decide what deserves checking, what should a real selective-oversight system use?

Candidate policies could include:

- raw confidence;
- calibrated confidence;
- natural hidden verification judgment;
- repeated self-assessment;
- combinations of these;
- internal-state predictors for open models.

Every method must be compared at the same verification budget wherever possible.

The primary metric should be:

How many real errors are caught when each method is allowed the same number of checks?

This avoids rewarding a method merely for spending more oversight resources.


================================================================================
49. THE PAPER’S LIKELY CONTRIBUTION LADDER
================================================================================

The final paper’s strength depends on how far the evidence progresses.


LEVEL 1 — CAUSAL BEHAVIORAL CONTRIBUTION

We establish that displayed confidence itself causally changes verification.


LEVEL 2 — SIGNAL MISMATCH CONTRIBUTION

We establish that natural verification behavior and explicit verbal confidence contain non-identical correctness information.


LEVEL 3 — EXPLANATORY CONTRIBUTION

We identify whether the extra signal comes from repeated evaluation, token probability, another observable signal, or deeper internal information.


LEVEL 4 — MECHANISTIC CONTRIBUTION

We show that confidence visibility changes how internal error-related representations influence action.


LEVEL 5 — DESIGN CONTRIBUTION

We demonstrate a better routing method that catches more errors under an equal verification budget.

The project does not require Level 5 to be strong.


================================================================================
50. CORE PAPER VERSUS OPTIONAL EXTENSIONS
================================================================================

The critical path should remain small enough to finish well.


CORE

- historical Study 0;
- Checkpoint A;
- causal pilot;
- prospective causal confirmation;
- modest generalization.


PREFERRED EXTENSION

- Study 4 explanation of the hidden residual information.


OPTIONAL HIGH-CEILING EXTENSION

- mechanistic Study 5.


OPTIONAL DEPLOYMENT/DESIGN EXTENSION

- routing remedy.

We should not jeopardize the quality of the causal core by rushing too many optional studies.


================================================================================
51. WHAT WOULD MAKE THE RESULT GENUINELY IMPORTANT?
================================================================================

The paper becomes more than a confidence-calibration study if it establishes some version of the following:

A confidence report can become behaviorally powerful once it is fed back into the system, and that feedback can alter how verification is allocated.

A stronger version would be:

The explicit report does not capture all of the information used by the model’s natural verification behavior.

A still stronger version would be:

Making the report explicit changes how other error-relevant signals influence the decision.

This progression is important because each level answers a more meaningful question.


================================================================================
52. THE PRACTICAL WARNING
================================================================================

A naive engineering instinct might be:

“If the AI has a confidence estimate, make sure its escalation or verification behavior follows that estimate consistently.”

D1 suggests that this is not enough.

The original GPT result already demonstrates that a system can become more orderly and more faithful to the confidence score while allowing more mistakes through unchecked.

Checkpoint A explains that much of this particular GPT failure comes from faithfully following an overconfident signal.

The broader lesson is therefore:

Consistency with an uncertainty score should not be treated as proof of good selective oversight.

The routing signal itself must be evaluated on whether it actually sends mistakes toward scrutiny.


================================================================================
53. THE “LOOKS BETTER WHILE BEING WORSE” NARRATIVE
================================================================================

This remains one of the strongest communication hooks.

For visible GPT:

- policy agreement improved dramatically;
- stake consistency improved dramatically;
- the behavior looked more mathematically orderly.

Yet high-stakes error filtering worsened.

That creates an intuitively powerful warning:

A system can improve on the metrics that make its oversight policy look disciplined while becoming worse at catching the mistakes oversight is supposed to catch.

This should remain part of the eventual introduction.

However, the explanation must now be more careful.

For GPT, Checkpoint A shows that the primary culprit is overconfident scoring plus highly faithful threshold-following.

The paper should not exaggerate this into a hidden-state claim unless later studies justify it.


================================================================================
54. WHAT THE PAPER SHOULD NOT CLAIM
================================================================================

Unless future studies provide stronger evidence, we should not say:

- “LLMs know when they are wrong.”
- “Showing confidence destroys internal uncertainty.”
- “Explicit confidence universally harms verification.”
- “The model’s hidden state contains the truth.”
- “We discovered that confidence affects abstention.”
- “We discovered that internal activations encode correctness.”
- “Our four models establish a universal law.”
- “All agentic systems currently use self-reported numerical confidence this way.”
- “The visible condition proves a mechanistic information bottleneck.”
- “Claude and GPT share the same mechanism.”
- “Better calibration solves every verification problem.”
- “Faithfulness to confidence is inherently bad.”

The project will be stronger if every claim remains at the level actually supported by the data.


================================================================================
55. CURRENT MARKETING FRAMING
================================================================================

The project should be described differently depending on audience.


ACCESSIBLE FRAMING

As AI systems become more autonomous, they increasingly need to decide when to trust their own work and when to ask for another check. Confidence seems like the obvious signal for making that choice. We study what happens when an AI’s confidence is explicitly fed back into its own verification decision.


RESEARCH FRAMING

We study self-reported confidence not only as an uncertainty estimate, but as an intervention on downstream selective verification.


STRONGEST CONDITIONAL FRAMING

If later mechanistic evidence succeeds:

Externalized self-confidence can change which uncertainty information governs an LLM’s verification behavior.

The third version should not be used as a definitive claim until earned.


================================================================================
56. LIKELY NOVELTY STATEMENT IF STUDIES 1–3 SUCCEED
================================================================================

A defensible future novelty statement may be:

We isolate the causal effect of explicitly displayed confidence on an LLM’s post-answer verification decision while keeping the underlying answer fixed. We show that explicit confidence can strongly reshape verification behavior, distinguish this effect from confidence calibration and overall verification rate, and test whether the model’s natural verification judgment contains error information not captured by its reported confidence.

If provenance experiments show that the model specifically treats its own confidence differently, the statement can emphasize self-feedback.

If provenance does not matter, the paper should instead emphasize displayed uncertainty signals generally.


================================================================================
57. LIKELY NOVELTY STATEMENT IF STUDY 5 SUCCEEDS
================================================================================

The stronger version could become:

We show that making an explicit uncertainty report visible does more than provide additional information: it changes how strongly other error-relevant internal signals influence downstream verification.

This would be the most distinctive mechanistic version of D1.


================================================================================
58. EVALUATION PHILOSOPHY
================================================================================

The project should distinguish several different properties.


ACCURACY

Is the original answer correct?


CALIBRATION

When the model says 90%, is it actually correct approximately 90% of the time?


DISCRIMINATION

Does the confidence score rank risky answers below safer answers?


VERIFICATION BUDGET

How many answers are checked?


ERROR CATCH RATE

Among answers that are actually wrong, how many receive verification?


VERIFICATION PRECISION

Among checked answers, how many were actually wrong?


POLICY AGREEMENT

How often does the model’s CHECK/USE decision agree with the simple confidence-implied rule?


REALIZED SYNTHETIC COST

How well does the decision perform under the experiment’s stylized verification/error-cost model?

These properties should not be conflated.

Checkpoint A exists precisely because the original paper did not fully separate them.


================================================================================
59. PRIMARY METRIC FOR THE REBUILT PROJECT
================================================================================

Whenever comparing two routing methods, the most intuitive headline metric should usually be:

ERROR CATCH RATE AT THE SAME VERIFICATION BUDGET.

This directly answers:

“If both systems are allowed the same amount of oversight, which one sends more real mistakes to verification?”

Other metrics remain useful, but this is likely the most deployment-relevant comparison.


================================================================================
60. MODEL STRATEGY
================================================================================

The project does not need fifteen or twenty models by default.

The existing closed-model panel already provides meaningful diversity.


CLOSED MODELS

Use a small number of strong frontier families to establish practical behavioral relevance.


OPEN MODELS

Add one or two families primarily to:

- test behavioral continuity;
- access token probabilities;
- access hidden representations;
- enable causal internal intervention if justified.

The project should prefer scientifically distinct model families over many nearly redundant variants.


================================================================================
61. TASK STRATEGY
================================================================================

The original MMLU-Pro setup remains useful because:

- correctness is objective;
- frozen answers are easy to preserve;
- confidence can be elicited cleanly;
- existing infrastructure already exists.

However, the final paper should ideally include at least one non-MCQ task.

Mathematics is currently the simplest candidate.

A later code-with-tests study could be especially attractive because verification is naturally meaningful, but it should be treated as optional unless implementation is easy.


================================================================================
62. INCREMENTAL-EXPERIMENT PHILOSOPHY
================================================================================

D1 should deliberately avoid:

“run the entire project and hope the story works.”

Every major study should begin with the cheapest meaningful pilot.

The general workflow is:

hypothesis
→ tiny pilot
→ inspect result
→ decide whether the next scale is justified
→ freeze confirmatory design
→ run fresh data.

This is both scientifically cleaner and computationally cheaper.


================================================================================
63. RESOURCE PLAN
================================================================================

COMPLETED:

Original V2:
already completed.

Checkpoint A:
no additional API cost.


NEXT CAUSAL PILOT:

Expected scale:
approximately 2,000–4,000 calls, depending on the final design.

Expected wall-clock runtime:
likely well under three hours with reasonable concurrency, subject to provider limits.

Expected API cost:
likely single-digit to low-tens of dollars for a small two-model pilot, depending on model pricing and output length.


LEAN FULL BEHAVIORAL EXPANSION:

If the pilot succeeds, a realistic planning range for the broader behavioral program is approximately:

80,000–100,000 additional API calls maximum

rather than millions.

Current rough cost planning:

approximately $100–400,

with uncertainty depending on:

- which models are used;
- reasoning mode;
- output lengths;
- retries;
- additional task families.

Interpretability should initially use a small tractable open model rather than assuming hundreds of GPU-hours.


================================================================================
64. REPRODUCIBILITY AND PROVENANCE
================================================================================

The rebuilt project has a complicated history.

That makes documentation especially important.

We must preserve clear distinctions among:


HISTORICAL / EXPLORATORY

- original V2 data;
- original workshop analysis;
- Checkpoint A;
- causal pilot.


PROSPECTIVE / CONFIRMATORY

- later frozen confidence-manipulation experiment;
- later generalization studies if their hypotheses are frozen before new data.

For each future experiment we should store:

- exact model identifier;
- API date;
- question IDs;
- prompt templates;
- confidence manipulations;
- inference parameters;
- retries;
- raw outputs;
- analysis code;
- random seeds;
- protocol version;
- dataset hash;
- code commit;
- output hashes where practical.

No confirmatory study should silently inherit analysis decisions created after viewing its results.


================================================================================
65. RESEARCH-CLAIM DISCIPLINE
================================================================================

Every major claim should be tagged mentally as one of:


OBSERVED

Directly measured in data.


EXPLORATORY INTERPRETATION

A plausible explanation generated after viewing data.


PROSPECTIVE HYPOTHESIS

Specified before a new test.


MECHANISTIC CLAIM

Requires causal/internal evidence.


Examples:

“Visible GPT agrees with the raw confidence rule 99.2% of the time.”

Observed.


“Showing confidence may suppress reconsideration.”

Exploratory interpretation.


“Manipulated confidence will causally shift verification.”

Prospective hypothesis for Study 1/2.


“An internal risk representation is overridden by explicit confidence.”

Mechanistic claim requiring Study 5.


================================================================================
66. REVIEWER ATTACKS WE SHOULD DESIGN AGAINST
================================================================================

ATTACK:
“This is just bad calibration.”

RESPONSE:
Checkpoint A shows this explains much of GPT’s visible cost failure. The paper should concede that and distinguish it from other questions rather than pretending calibration is irrelevant.


ATTACK:
“The hidden condition only checks more.”

RESPONSE:
Use equal-budget error-catching comparisons.


ATTACK:
“You just added a number and the model followed instructions.”

RESPONSE:
Manipulate the number causally and include qualitative-stakes / contradiction controls.


ATTACK:
“Any random score would do.”

RESPONSE:
Use provenance controls.


ATTACK:
“The natural policy only benefits from seeing the question twice.”

RESPONSE:
Re-elicit confidence / second-look controls.


ATTACK:
“Token probability already explains everything.”

RESPONSE:
Test it directly on open models.


ATTACK:
“Your mechanism only exists in an unrelated open model.”

RESPONSE:
Require behavioral continuity before mechanistic interpretation.


ATTACK:
“This is one prompt.”

RESPONSE:
Use controlled paraphrases and fresh confirmatory prompts.


ATTACK:
“You changed the hypothesis after seeing results.”

RESPONSE:
Explicitly distinguish historical exploratory evidence from prospective confirmatory studies.


================================================================================
67. DEEP RESEARCH TIMING
================================================================================

A new broad literature search should NOT happen before the causal pilot.

The next Deep Research pass should be triggered after we know whether manipulated confidence actually causally controls verification.

At that point, the research prompt should contain the exact pilot results and ask:

Find the closest prior work capable of subsuming or explaining these exact findings. Try to invalidate our novelty claim.

The targeted literature pass should examine:

- confidence-driven abstention;
- explicit uncertainty as a routing signal;
- numeric anchoring;
- self-conditioning on earlier outputs;
- re-evaluation suppression;
- verbal confidence versus token probabilities;
- internal correctness probes;
- uncertainty-triggered escalation/retrieval;
- selective prediction;
- decision calibration;
- causal steering interactions;
- feedback from model-generated scores into later model behavior.

The goal should be novelty stress-testing, not collecting more citations.


================================================================================
68. WORKING PAPER NARRATIVE
================================================================================

The eventual introduction should likely follow this logic.


PARAGRAPH 1

AI systems increasingly need to decide not only what answer to produce, but whether their own output deserves further checking.


PARAGRAPH 2

Confidence is a natural signal for such selective oversight.


PARAGRAPH 3

Existing work usually evaluates confidence as an estimate.

But once the estimate is shown back to the system and used in its next decision, it is no longer just a measurement.


PARAGRAPH 4

Our original controlled experiment discovered that making the same frozen confidence explicit can dramatically alter verification, sometimes in opposite directions across models.


PARAGRAPH 5

Most strikingly, GPT became almost perfectly faithful to its confidence-based decision rule while catching fewer high-stakes mistakes.


PARAGRAPH 6

Checkpoint A showed that much of this GPT failure comes from faithfully following an overconfident score, but also revealed that GPT and Claude’s natural verification behavior contains correctness information not fully captured by the explicit confidence report.


PARAGRAPH 7

This motivates the central causal question:

What happens when an externally observable uncertainty report becomes a control signal for the model’s own oversight?

That is the paper.


================================================================================
69. WORKING TITLE DIRECTIONS
================================================================================

Current strategic placeholders:

When Confidence Controls Oversight: Self-Reported Uncertainty in LLM Verification

When Confidence Becomes a Control Signal: LLM Self-Reports and Verification

Showing a Model Its Confidence Changes What It Checks

Confidence Is Not Just a Report: Feedback Effects in LLM Verification

When Should Models Trust Their Own Confidence?

The title should remain conservative until the causal pilot determines exactly what we have.


================================================================================
70. CURRENT PAPER VISION IN ONE PARAGRAPH
================================================================================

As language models become more autonomous, they increasingly need to decide which of their own outputs deserve additional scrutiny. Self-reported confidence is an attractive signal for selective verification because it is inexpensive and available even for black-box systems. Our original V2 experiment discovered that explicitly showing a model the confidence it had already reported can dramatically alter its verification behavior: Claude became more cautious, GPT and Grok became less cautious, and GPT became almost perfectly consistent with its confidence-implied decision rule while allowing substantially more high-stakes mistakes through unchecked. A post-submission re-analysis showed that much of GPT’s failure is explained by faithfully following an overconfident score, but also revealed that GPT and Claude’s natural hidden verification decisions contain correctness information not fully captured by their reported confidence. The rebuilt D1 project therefore studies a broader problem: what happens when an uncertainty report becomes a control signal for the model’s own oversight? Through causal manipulation of displayed confidence, prospective replication, signal controls, limited generalization, and—only if justified—mechanistic analysis, the project aims to determine when explicit confidence governs verification, what information is lost or retained, and how selective oversight should be evaluated and designed.


================================================================================
71. CURRENT EVIDENCE LEDGER
================================================================================

ESTABLISHED FROM ORIGINAL V2:

- Confidence visibility strongly changes verification behavior.
- The sign and size of the effect are model-dependent.
- GPT becomes dramatically more consistent with its confidence-implied rule when confidence is visible.
- GPT simultaneously leaves substantially more wrong answers unchecked at high stakes.
- Authority effects are smaller and less universal.
- Major visibility-effect directions mostly persist under one prompt paraphrase.


ESTABLISHED FROM CHECKPOINT A:

- Visible GPT nearly equals the raw confidence-threshold policy.
- GPT’s verbal confidence is strongly overconfident.
- Much of GPT’s high-stakes visible cost failure is therefore explained by calibration plus reduced verification.
- Claude’s hidden policy catches more errors than verbal-confidence ranking at the same budget.
- GPT and Claude hidden verification decisions contain correctness information not fully captured by verbal confidence.
- The proposed simple cross-model “better signal determines effect direction” law is not supported.


STILL UNKNOWN:

- Whether changing the shown number itself causally changes verification.
- Whether self-provenance matters.
- Whether the effect generalizes.
- Where the hidden extra information comes from.
- Whether internal representations add information beyond observable signals.
- Whether explicit confidence changes use of those internal representations.
- What routing method should replace raw confidence if raw confidence is inadequate.


================================================================================
72. IMMEDIATE NEXT STEPS
================================================================================

The project should now proceed in this order:

1. Freeze this Master Plan as the current strategic source of truth.

2. Write the experimentalProtocol covering the small causal pilot first and later conditional studies separately.

3. Establish the repository communication structure:
   from_gpt/
   to_gpt/

4. Give Cursor the exact Study-1 pilot task.

5. Run approximately 2,000–4,000 calls, not the full program.

6. Inspect the result.

7. If the pilot fails, revise the scientific direction before spending more compute.

8. If the pilot succeeds, run the targeted novelty-focused Deep Research pass.

9. Use that literature audit plus pilot evidence to freeze the prospective confirmatory Study 2 protocol.

10. Only after the confirmatory behavioral core is secure should we decide whether Study 4/5 interpretability work is justified.


================================================================================
73. FINAL STRATEGIC PRINCIPLE
================================================================================

The project should continually return to one rule:

Do not add a technical method because it makes the paper look sophisticated. Add it only if it answers a question the previous evidence genuinely leaves unresolved.

The original experiment discovered a surprising phenomenon.

Checkpoint A removed one overly ambitious explanation while exposing a narrower and more interesting unanswered question.

The causal pilot should now determine whether that question is real.

If it is, the project can deepen step by step.

If it is not, we should learn that cheaply and change direction rather than manufacturing a story.

That is the intended research process for D1.