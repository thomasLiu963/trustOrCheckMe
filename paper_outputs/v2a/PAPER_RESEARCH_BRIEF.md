# Trust Me or Check Me? Current Research Results and Paper-Writing Brief

## How to use this document

This document is a paper-writing brief based on the completed V1 pilot and the
current V2-A experiment. It is intentionally explicit about what has and has
not been run. A paper-writing model should use the numerical results below,
preserve the stated claim limits, and not describe V2-A as the planned
500-question final V2-B study.

## One-paragraph summary

This project studies whether a language model can decide when an already
generated answer should be used without checking and when it should first be
independently verified. The central V2 contribution is a matched manipulation
of who controls that verification decision: the human or the AI system. On a
deterministic 200-question MMLU-Pro sample, four model families first produced
a frozen answer and a frozen probability that the answer was correct. Each
answer was then evaluated under 16 Stage-3 conditions crossing decision owner
(human versus AI), confidence visibility (hidden versus visible), and
synthetic error cost (`L = 2, 5, 10, 20`) while verification cost remained
fixed at 1. Of 12,800 planned Stage-3 decisions, 12,799 were obtained. The
clearest directional owner effect was specific to Anthropic Claude Sonnet 5:
when frozen confidence was visible, it chose verification 3.5 to 7.5
percentage points more often when the AI system, rather than the human,
controlled the decision, with nominal paired 95% bootstrap intervals excluding
zero at all four error costs. The other three models had signed owner effects
near zero, although some still changed individual answers between owner
framings in opposing directions. Showing frozen confidence had much larger and
strongly model-dependent effects: it increased verification for Anthropic,
decreased it for OpenAI and xAI, and increased it mainly at high stakes for
Google. Explicit confidence did not reliably shrink the owner effect. These
results show heterogeneous verification policies that cannot be inferred from
answer accuracy alone, but they do not establish a universal owner asymmetry.
The preregistered 500-question V2-B run and prompt-paraphrase robustness run
remain necessary before presenting this as the final study.

## Research motivation and contribution

The outsider-facing question is:

> Can AI tell when users should trust its answers versus independently verify
> them, especially as mistakes become more costly?

The more precise V2 question is:

> Holding the question, frozen answer, uncertainty, verification mechanism,
> verification cost, and error cost constant, does an LLM change its
> verification policy depending on whether the human or the AI system controls
> the verification decision?

The mechanism question is whether explicitly showing the model its own
previously reported confidence reduces or eliminates any human-versus-AI
policy difference.

The novelty claim should be narrow. The project does not claim to invent
confidence calibration, abstention, selective prediction, or human-AI
reliance. The intended contribution is a controlled, paired test of
verification-decision ownership after the answer and uncertainty have already
been frozen, crossed with explicit access to that uncertainty.

## Current experimental status

The project has two distinct phases:

1. **V1 pilot — complete.** This used 200 MMLU-Pro questions and two models,
   OpenAI GPT-5.6 Sol and Anthropic Claude Sonnet 5. It tested only
   human-facing `RELY` versus `VERIFY` recommendations at four error costs.
2. **V2-A matched-design validation — substantially complete.** This used the
   same deterministic 200-question sample and four independent provider
   families. It collected 12,799 of 12,800 primary Stage-3 factorial
   decisions.

The following planned components are **not complete**:

- V2-B, the preregistered 500-question paper run;
- the 100-question independent prompt-paraphrase robustness run;
- GPQA or another secondary dataset.

Therefore, a paper written now must call the V2 findings a 200-question V2-A
study, validation study, or preliminary matched experiment. It must not claim
that the preregistered final 500-question experiment or prompt robustness check
has been completed.

## V1 pilot: motivation for V2

V1 established that answer accuracy and verification advice could diverge.
OpenAI GPT-5.6 Sol answered 84.5% of the 200 questions correctly, compared with
73.5% for Anthropic Claude Sonnet 5. Despite its higher answer accuracy,
OpenAI's verification rate changed only from 14.0% at `L=2` to 17.5% at
`L=20`. Anthropic's verification rate increased from 42.5% to 53.0%.

Among wrong answers at `L=20`, V1 unsafe reliance was 58.1% for OpenAI and
9.4% for Anthropic. OpenAI also had a higher rate of non-monotonic
stake-response trajectories: 9.5% versus 2.0% for Anthropic. On the held-out
calibration evaluation split at `L=20`, OpenAI's direct recommendation policy
incurred 0.963 more synthetic cost per decision than its
calibrated-confidence policy, with a paired 95% bootstrap interval from 0.119
to 1.856.

These V1 findings motivated a stronger design, but V1 should not be presented
as the paper's novelty. It used different Stage-3 wording and action labels and
did not manipulate decision ownership. V1 Stage-3 records were not reused in
V2.

## V2-A design and methods

### Dataset and models

V2-A used a deterministic 200-question sample from the MMLU-Pro test split,
pinned to repository revision
`b189ec765aa7ed75c8acfea42df31fdae71f97be`. The four model families were:

- OpenAI `gpt-5.6-sol`, reasoning effort `none`;
- Anthropic `claude-sonnet-5`, thinking disabled;
- Google Vertex AI `gemini-3.8-flash`, thinking level `low`;
- xAI `grok-4.20-0309-non-reasoning`.

Tools, browsing, search, retrieval, code execution, and external context were
disabled. Provider reasoning settings are not internally equivalent, so
cross-model comparisons are descriptive generalization evidence. The
controlled comparisons are within-model matched comparisons.

Google used a 512-token output ceiling because its mandatory low-thinking
tokens consumed a 64-token smoke-test ceiling before valid structured output.
The other providers used a 64-token ceiling. This amendment was recorded
before V2-A.

### Three-stage protocol

Stage 1 presented a multiple-choice question and elicited only an answer
label. That answer was frozen.

Stage 2 presented the same question and frozen answer and elicited a
probability that the answer was correct. That probability was also frozen.
There was no stake or verification language in Stages 1 or 2.

Stage 3 presented the question and frozen answer and requested one of two
actions:

- `USE_UNVERIFIED`: use or deliver the frozen answer without checking it;
- `VERIFY_FIRST`: pay for an independent verifier before the answer is used or
  delivered.

For OpenAI and Anthropic, compatible V1 Stage-1 and Stage-2 records were reused
only when the example, requested model, model configuration, prompt version,
and frozen-answer dependency matched exactly. Google and xAI Stage-1 and
Stage-2 outputs were generated for V2. All V2 Stage-3 decisions were new.

### Factorial manipulation

For each question-model pair, Stage 3 crossed:

- decision owner: human or AI system;
- confidence visibility: hidden or visible;
- error cost: 2, 5, 10, or 20.

This yielded 16 decisions per question-model pair. Human and AI prompts used
the same action labels, costs, answer, verifier guarantee, and neutral
numerical language. The owner wording was the intended difference. In visible
conditions, one sentence supplied the frozen Stage-2 probability; hidden
conditions did not expose it.

### Cost model

Verification cost was fixed at `C=1`. Verification was assumed to return the
correct answer. Unverified use cost zero when the frozen answer was correct and
`L` when it was wrong. For confidence `q`, the confidence-derived rule verifies
when:

`(1-q)L > 1`.

The corresponding confidence thresholds are `q<0.50` at `L=2`, `q<0.80` at
`L=5`, `q<0.90` at `L=10`, and `q<0.95` at `L=20`.

### Outcomes and statistics

Primary outcomes included verification rate, unsafe unverified use among wrong
answers, unnecessary verification among correct answers, realized synthetic
cost, regret relative to an answer-aware oracle, and monotonicity violations.

The signed owner effect was:

`P(VERIFY_FIRST | AI owner) - P(VERIFY_FIRST | human owner)`.

The analysis also measured paired owner disagreement, because equal marginal
rates can hide question-level action changes in opposite directions.

The confidence-visibility effect was:

`P(VERIFY_FIRST | confidence visible) - P(VERIFY_FIRST | confidence hidden)`.

The owner-by-confidence interaction was the visible owner gap minus the hidden
owner gap.

Confidence calibration used deterministic 20% calibration and 80% evaluation
splits with isotonic regression and no label leakage. The main paired
statistics used 5,000 question-level bootstrap resamples and nominal 95%
intervals, keeping each question's factor cells together.

## V2-A data quality and coverage

The run obtained 12,799 of 12,800 planned decisions, or 99.992% coverage.
There was one unresolved Anthropic response for
`mmlu_pro:test:4132`, human owner, confidence hidden, `L=2`. It repeatedly
exhausted the frozen 64-token output ceiling and was not imputed. That one
stratum therefore has 199 rather than 200 decisions. The omitted item was a
wrong Anthropic answer, so the wrong-answer denominator in that paired owner
comparison is 52 rather than 53. Every other primary factorial stratum has
complete coverage.

## V2-A result 1: answer accuracy and confidence quality

Frozen-answer performance differed substantially:

- Google Gemini 3.8 Flash: 89.5% accuracy, 21 wrong answers, Brier score 0.0657;
- OpenAI GPT-5.6 Sol: 84.5% accuracy, 31 wrong answers, Brier score 0.1477;
- Anthropic Claude Sonnet 5: 73.5% accuracy, 53 wrong answers, Brier score
  0.1681;
- xAI Grok 4.20 non-reasoning: 65.0% accuracy, 70 wrong answers, Brier score
  0.1916.

Lower Brier score indicates more accurate probability forecasts. These
differences matter because wrong-answer safety rates for Google and OpenAI are
estimated from only 21 and 31 errors, respectively, making those percentages
less precise than the corresponding Anthropic and xAI estimates.

Accuracy did not determine verification behavior. Google was most accurate but
often used wrong answers without verification. xAI was least accurate but
verified almost everything in the hidden-confidence condition. These patterns
support the preregistered distinction between answer capability and
verification-policy quality, but four models cannot establish a scaling law.

## V2-A result 2: the owner effect was concentrated in Anthropic

Anthropic showed the clearest systematic owner effect. With confidence hidden,
the signed AI-minus-human verification effects were:

- `L=2`: +0.5 percentage points, 95% CI [-3.52, 4.52];
- `L=5`: +4.0 points, CI [0.0, 8.0];
- `L=10`: +3.5 points, CI [0.0, 7.0];
- `L=20`: +4.0 points, CI [1.5, 7.0].

With confidence visible, Anthropic's effects were:

- `L=2`: +3.5 points, CI [1.0, 6.5];
- `L=5`: +4.0 points, CI [1.5, 7.0];
- `L=10`: +7.5 points, CI [4.0, 11.5];
- `L=20`: +4.0 points, CI [1.0, 7.0].

Thus, under the visible-confidence prompt, Anthropic consistently verified
more when the AI system controlled the decision. The average of the four
stake-specific point estimates was +4.75 points with confidence visible and
approximately +3.0 points with confidence hidden. These averages are
descriptive summaries, not separately bootstrapped pooled estimates.

The other model families did not show a consistent directional owner effect:

- Google's estimates ranged from -1.5 to +2.0 points when confidence was
  hidden and from -1.0 to +2.5 points when visible.
- OpenAI's estimates ranged from -3.0 to +3.0 points when hidden and from
  -0.5 to +1.0 points when visible.
- xAI's estimates ranged from -2.0 to 0.0 points when hidden and from -3.5 to
  +3.5 points when visible.

No OpenAI, Google, or xAI nominal owner-effect interval strictly excluded zero.
The defensible conclusion is therefore heterogeneity, not a universal owner
asymmetry.

## V2-A result 3: near-zero signed effects did not imply action invariance

The signed owner effect can be near zero even when the owner framing changes
many individual decisions, because switches in opposite directions cancel.

For example, OpenAI at hidden confidence and `L=10` had a signed owner effect
of -3.0 points, CI [-8.0, 2.0], but the paired human-versus-AI actions differed
on 14.0% of questions. OpenAI's hidden-confidence disagreement was between
6.0% and 14.0% across stakes, despite small net owner gaps.

xAI had a particularly clear cancellation pattern when confidence was visible.
Its signed effects remained between -3.5 and +3.5 points, while paired owner
disagreement ranged from 14.0% to 18.0%. By contrast, xAI's hidden-confidence
owner disagreement was only 3.5% to 6.0%.

This distinction should appear in the paper: three models were approximately
invariant in marginal verification rates, but some were not invariant at the
question level.

## V2-A result 4: confidence visibility had large, opposite effects by model

Showing the frozen probability produced larger changes than the owner
manipulation, but the direction depended on model family.

For Anthropic, visible confidence increased verification in every
owner-by-stake cell. Under AI ownership, the increase was +16.5 points at
`L=2`, +22.0 at `L=5`, +25.0 at `L=10`, and +25.0 at `L=20`. Under human
ownership, the increases were +13.57, +22.0, +21.0, and +25.0 points. Every
corresponding nominal paired interval excluded zero.

For OpenAI, visible confidence decreased verification in all eight
owner-by-stake cells, by 13.5 to 21.0 points. Every corresponding interval was
wholly non-positive. Moreover, in visible-confidence conditions OpenAI's
direct action disagreed with the mechanical raw-confidence threshold on only
0% to 2% of questions. This suggests that OpenAI nearly implemented the
displayed probability threshold under this prompt. It does not show that the
underlying probability was calibrated well enough to minimize cost.

For xAI, visible confidence also decreased verification in all eight cells, by
7 to 14 points, with all nominal intervals excluding zero. Unlike OpenAI, xAI
still disagreed substantially with the raw-confidence policy, especially at
lower stakes.

For Google, confidence visibility had little or inconsistent effect at low
stakes but increased verification at high stakes. At `L=10`, verification rose
by +10.0 points, CI [5.5, 15.0], under AI ownership and +8.0 points, CI [4.5,
12.0], under human ownership. At `L=20`, the increases were +15.0 points, CI
[10.0, 20.0], and +13.5 points, CI [9.0, 18.5], respectively.

The strongest mechanism conclusion is therefore not that confidence always
makes models safer or more cautious. Explicit confidence caused each family to
adopt a different policy response.

## V2-A result 5: explicit confidence did not close the owner gap

The preregistered interaction asked whether confidence visibility moved the
signed owner gap toward zero. The data do not support a general closing effect.
All 16 model-by-stake owner-gap-change intervals included zero. Point estimates
ranged from -3.5 to +4.0 percentage points.

For Anthropic, the model with the clearest owner asymmetry, the descriptive
average signed gap was larger with confidence visible (+4.75 points) than
hidden (approximately +3.0 points). This is the opposite of the proposed
gap-closing mechanism, although the interaction intervals do not support a
confident claim that visibility enlarged the gap.

The correct conclusion is that confidence visibility changed overall
verification behavior but did not reliably moderate the signed owner effect.

## V2-A result 6: stake sensitivity was present but heterogeneous

Averaging across the two owners, the verification-rate change from `L=2` to
`L=20` was:

- Anthropic: +11.4 points with confidence hidden and +21.3 points visible;
- Google: +2.75 points hidden and +18.75 points visible;
- OpenAI: +14.75 points in both visibility conditions;
- xAI: +0.75 points hidden and +5.5 points visible.

These are descriptive endpoint contrasts without separately reported
confidence intervals. They show that all models had at least some aggregate
stake response, but Google hidden and xAI were nearly flat. xAI's hidden rate
was already near a ceiling: approximately 95% verification at `L=2` and 95.75%
at `L=20`.

Question-level monotonicity tells a less reassuring story. Among complete
four-stake trajectories:

- Anthropic had violations in 14 of 799 trajectories (1.75%);
- Google had violations in 16 of 800 (2.0%);
- OpenAI had violations in 35 of 800 (4.38%);
- xAI had violations in 115 of 800 (14.38%).

For Anthropic, Google, and OpenAI, visible confidence generally reduced
monotonicity violations. xAI was the exception: 47 of 200 human-visible and 43
of 200 AI-visible trajectories violated monotonicity, compared with 12 and 13
in the hidden conditions.

The paper should distinguish monotonic aggregate curves from monotonic
question-level policies.

## V2-A result 7: wrong-answer behavior exposed different failure modes

At the highest error cost, `L=20`, unsafe unverified use among actually wrong
answers was:

- Anthropic hidden: 11.32% under human ownership and 9.43% under AI ownership;
- Anthropic visible: 1.89% under both owners;
- Google hidden: 57.14% human and 61.90% AI;
- Google visible: 33.33% human and 28.57% AI;
- OpenAI hidden: 32.26% human and 25.81% AI;
- OpenAI visible: 58.06% human and 51.61% AI;
- xAI hidden: 0% human and 1.43% AI;
- xAI visible: 4.29% human and 5.71% AI.

These results illustrate why confidence visibility cannot be labeled
universally beneficial. It sharply reduced high-stakes unsafe unverified use
for Anthropic and Google but increased it for OpenAI and modestly increased it
for xAI. For OpenAI, displaying an apparently high frozen confidence often
caused the model to stop verifying wrong answers.

Wrong-answer owner-gap estimates were not robust: their intervals included or
touched zero. This is especially important for Google and OpenAI, which had
only 21 and 31 wrong answers. The paper should emphasize denominators and avoid
strong owner-specific wrong-answer claims.

## V2-A result 8: verification burden and cost reveal different policy errors

xAI's hidden-confidence policy verified approximately 94% to 97.5% of all
answers. This nearly eliminated unsafe unverified wrong answers but imposed
very high verification burden on correct answers. At `L=20`, unnecessary
verification among correct answers was around 95% in hidden conditions and
83% to 85% with confidence visible.

Anthropic's visible-confidence policy also verified heavily, reaching 85% to
89% overall at `L=20`; unnecessary verification among correct answers was
80.3% for the human owner and 85.7% for the AI owner. This reduced unsafe use
but incurred verification cost on many correct answers.

Google's visible high-stakes policy was less burdensome but left a larger share
of wrong answers unchecked. OpenAI's visible policy closely followed its raw
confidence threshold and therefore inherited the consequences of its
confidence estimates.

At `L=20`, descriptive mean synthetic costs for the direct visible-confidence
policies were 0.950 to 0.990 for Anthropic, 0.845 to 0.920 for Google, 1.805 to
1.995 for OpenAI, and 1.175 to 1.285 for xAI. The corresponding raw-confidence
policy costs were 0.955, 0.550, 1.815, and 1.000. These comparisons are
descriptive; the current V2 policy table does not provide paired confidence
intervals for every contrast.

Calibrated-confidence means use the 160-question evaluation partition, whereas
most direct and raw means use all 200 questions. They must not be compared as
if they were evaluated on identical samples unless a paired evaluation-only
contrast is recomputed.

No current result supports a universal claim that direct decisions,
raw-confidence thresholds, or calibrated-confidence thresholds are always
best.

## Interpretation by preregistered hypothesis

**H1, stake sensitivity:** Partially supported. Verification generally rose
with error cost, but the strength varied substantially, hidden Google and xAI
were nearly flat, and individual monotonicity violations remained.

**H2, verification-owner effect:** Supported only as a heterogeneous,
model-dependent phenomenon. Anthropic showed a consistent positive signed
effect, especially with confidence visible. Other models had no clear
directional effect, though OpenAI and xAI sometimes had meaningful paired
action disagreement with cancellation.

**H3, confidence visibility changes behavior:** Strongly supported. The effect
was large and replicated across stakes for several models, but its direction
was not universal.

**H4, confidence visibility reduces the owner gap:** Not supported. Interaction
intervals included zero, and Anthropic's point estimates did not shrink.

**H5, cross-model heterogeneity:** Strongly supported descriptively. Models
differed in accuracy, confidence quality, baseline verification tendency,
stake sensitivity, owner response, confidence response, unsafe use, burden,
and monotonicity.

**H6, capability and verification quality are separate:** Supported
descriptively. The most accurate model did not have the lowest wrong-answer
unsafe-use rate, and the least accurate model was often the most conservative.
This does not establish a general capability-safety relationship.

## Strongest defensible paper claim

A defensible central result is:

> In a paired 200-question verification experiment with frozen answers and
> uncertainty, decision-owner framing did not produce a universal directional
> effect across four model families. Claude Sonnet 5 selected verification 3.5
> to 7.5 percentage points more often when the AI system rather than the human
> controlled the decision under visible-confidence prompts, whereas signed
> owner effects for the other models were near zero. However, near-zero
> marginal effects sometimes concealed substantial question-level owner
> disagreement. Showing frozen confidence produced larger, oppositely directed
> policy shifts across families and did not reliably close the owner gap.

An appropriate broader interpretation is:

> Verification recommendations are not determined by answer accuracy, stated
> confidence, and numerical stakes in a uniform way across model families.
> Models differ both in whether owner framing changes their decisions and in
> how they use explicitly supplied uncertainty.

## Claims that the evidence does not support

Do not claim:

- that all LLMs verify more when the AI controls the decision;
- that AI systems “push responsibility” onto humans;
- that confidence visibility consistently improves safety;
- that the models know when they are wrong;
- that real users will trust or follow these recommendations;
- that the synthetic cost represents real-world harm;
- that one vendor or model is broadly safer;
- that four models establish a capability or scaling law;
- that prompt robustness has been demonstrated;
- that the preregistered 500-question V2-B study is complete;
- that nominal cellwise intervals are multiplicity-adjusted;
- that near-zero marginal owner effects imply identical paired decisions.

## Statistical cautions

The reported 95% intervals are nominal question-bootstrap intervals. Many
cellwise effects were examined—32 owner effects, 32 confidence-visibility
effects, and 16 interactions—without multiplicity adjustment. The paper should
emphasize repeated within-model patterns and effect sizes rather than isolated
intervals.

The intervals capture variation over sampled questions, not API-generation
stochasticity, provider drift, repeated runs, or uncertainty over which models
were selected.

Wrong-answer rates have small denominators for Google and OpenAI. Bootstrap
intervals with endpoints at zero may simply reflect no observed discordant
cases in a small sample.

The isotonic calibrators were fitted on only 40 questions per model and
evaluated on 160. Calibration comparisons should be treated cautiously.

## General limitations

The experiment uses synthetic numerical costs, not real consequences. There
are no human participants. MMLU-Pro is a multiple-choice benchmark and may be
contaminated in closed-model training data. Self-reported confidence is not
guaranteed to equal an internal probability. Verification is modeled as
perfect and costs exactly one unit. The owner framing is binary even though
real responsibility is often shared. Closed-model APIs and serving behavior
may change. Provider reasoning settings are not comparable. Binary structured
actions do not measure explanation quality or actual delegation. One primary
cell is missing, and prompt-paraphrase robustness has not yet been run.

## What must happen before the intended final paper

1. Run the deterministic 500-question V2-B experiment, which includes these
   200 questions plus 300 questions chosen without reference to V2-A results.
2. Run the preregistered 100-question independent prompt-paraphrase robustness
   subset.
3. Regenerate all tables and figures using the V2-B data.
4. Recompute paired policy comparisons on matched evaluation samples and add
   paired intervals where policy superiority is discussed.
5. Treat V2-A as design validation or exploratory evidence and clearly
   distinguish confirmatory V2-B results.
6. Consider GPQA only after the primary MMLU-Pro and prompt-robustness analyses
   are complete.

## Suggested abstract for a paper describing the current V2-A evidence

Language models increasingly provide answers that users may accept without
independent checking. We study whether models recommend verification
consistently when an answer is frozen and only the owner of the verification
decision changes. On 200 MMLU-Pro questions, four model families first
generated an answer and a probability of correctness. We then elicited 12,799
binary verification decisions in a matched `2 × 2 × 4` design crossing
decision owner (human or AI system), visibility of the frozen confidence, and
error cost. Claude Sonnet 5 verified 3.5–7.5 percentage points more often under
AI ownership when confidence was visible, with nominal paired 95% bootstrap
intervals excluding zero at all four costs. The other models showed signed
owner effects near zero, although question-level owner disagreement reached
14–18% in some conditions. Confidence visibility produced larger but opposing
policy shifts: it increased verification for Claude, decreased it for GPT-5.6
Sol and Grok, and increased it mainly at high stakes for Gemini. Confidence
visibility did not reliably reduce the owner gap. These results suggest that
verification policy is model-specific and sensitive to how uncertainty and
decision ownership are presented, while providing no evidence for a universal
owner asymmetry. The present study is a 200-question validation; the
preregistered 500-question and prompt-paraphrase replications remain future
work.

## Suggested four-page paper structure

**Page 1: Motivation and contribution.** Explain why post-answer verification
guidance matters, distinguish the work from calibration and abstention, state
the matched owner manipulation, and briefly cite V1 as motivating evidence.

**Page 2: Method.** Describe the deterministic MMLU-Pro sample, four models,
frozen answer and confidence stages, `2 × 2 × 4` Stage-3 factorial, cost model,
matched prompt constraints, metrics, and paired bootstrap.

**Page 3: Results.** Lead with the Anthropic-specific owner effect and the lack
of a universal directional effect. Then show paired disagreement, the
model-dependent confidence-visibility response, wrong-answer unsafe use,
stake sensitivity, and monotonicity. Report exact denominators.

**Page 4: Discussion.** Emphasize heterogeneity and the distinction between
accuracy and verification policy. Explain that explicit confidence changed
policy without reliably closing the owner gap. Discuss synthetic stakes, no
human participants, small wrong-answer denominators, provider differences,
multiplicity, and unfinished V2-B/prompt robustness.

## Canonical result files

The canonical V2 analysis outputs are in `paper_outputs/v2/`. The most relevant
files for paper writing are:

- `model_summary.csv`;
- `factorial_metrics.csv`;
- `owner_effects.csv`;
- `confidence_visibility_effects.csv`;
- `owner_confidence_interactions.csv`;
- `policy_comparison.csv`;
- `monotonicity.csv`;
- `factor_completeness_issues.json`;
- the generated PDF and PNG figures.

The frozen protocol is `EXPERIMENT_V2.md`. The broader implementation and
interpretation specification is `readme2.md`. V1 is preserved separately as
pilot evidence and must not be silently pooled with V2.
