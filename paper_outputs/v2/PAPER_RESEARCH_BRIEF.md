# Trust Me or Check Me? V2-B Research Results and Paper-Writing Brief

## How to use this document

This document is a paper-writing brief based on the completed V1 pilot, the
200-question V2-A validation, and the completed 500-question V2-B paper run.
A paper-writing model should use the V2-B numbers below as the primary
results, treat V2-A as design validation, and not claim that prompt-paraphrase
robustness or a secondary dataset has been run.

Canonical machine-readable tables and figures live in `paper_outputs/v2/`.
The frozen V2-A snapshot is in `paper_outputs/v2a/` and must not be mixed
into V2-B numerics.

## One-paragraph summary

This project studies whether a language model can decide when an already
generated answer should be used without checking and when it should first be
independently verified. The central V2 contribution is a matched manipulation
of who controls that verification decision: the human or the AI system. On a
deterministic 500-question MMLU-Pro sample that includes the 200 V2-A items
plus 300 new items chosen without reference to V2-A results, four model
families first produced a frozen answer and a frozen probability that the
answer was correct. Each answer was then evaluated under 16 Stage-3 conditions
crossing decision owner (human versus AI), confidence visibility (hidden
versus visible), and synthetic error cost (`L = 2, 5, 10, 20`) while
verification cost remained fixed at 1. All 32,000 planned Stage-3 decisions
were obtained. The only consistent directional owner effect was again
specific to Anthropic Claude Sonnet 5: with frozen confidence visible, it
chose verification 3.8 to 5.0 percentage points more often when the AI
system, rather than the human, controlled the decision, with nominal paired
95% bootstrap intervals excluding zero at all four error costs. The other
three models had signed owner effects near zero in most cells, although
OpenAI and xAI still changed many individual decisions between owner
framings in opposing directions. Showing frozen confidence had much larger
and strongly model-dependent effects: it increased verification for
Anthropic, decreased it for OpenAI and xAI, and increased it mainly at high
stakes for Google. Explicit confidence did not reliably shrink the owner
effect. These results confirm V2-A's heterogeneous-policy conclusion on the
preregistered 500-question sample. The independent 100-question
prompt-paraphrase robustness run has not been collected.

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

The novelty claim should remain narrow. The project does not claim to invent
confidence calibration, abstention, selective prediction, or human-AI
reliance. The intended contribution is a controlled, paired test of
verification-decision ownership after the answer and uncertainty have already
been frozen, crossed with explicit access to that uncertainty.

## Current experimental status

1. **V1 pilot — complete.** 200 MMLU-Pro questions, OpenAI and Anthropic only,
   human-facing `RELY`/`VERIFY` recommendations. Motivating evidence only.
2. **V2-A matched-design validation — complete.** Same 200 questions, four
   families, 12,799 of 12,800 Stage-3 decisions. Archived in
   `paper_outputs/v2a/`.
3. **V2-B paper run — collection and primary analysis complete.** 500
   questions, four families, 32,000 of 32,000 Stage-3 decisions, 0
   factor-completeness issues.

Not complete:

- the 100-question independent prompt-paraphrase robustness run;
- GPQA or another secondary dataset.

A paper written now may present V2-B as the completed primary MMLU-Pro
experiment. It must not claim prompt robustness.

## V1 and V2-A in one paragraph

V1 showed that answer accuracy and verification advice can diverge: OpenAI
was more accurate than Anthropic but verified much less, including on wrong
answers. V2-A then introduced the matched owner × confidence-visibility ×
stake factorial on 200 questions. Its directional owner effect was
Anthropic-specific (about +3.5 to +7.5 points when confidence was visible).
The other models had near-zero signed owner gaps, sometimes with substantial
question-level cancellation. Confidence visibility produced larger, oppositely
signed policy shifts. V2-B asks whether those patterns hold on the
preregistered 500-question sample with larger wrong-answer denominators.

Do not pool V1 Stage-3 records with V2. Do not silently average V2-A and
V2-B as if they were independent samples: V2-B contains all V2-A questions.

## V2-B design and methods

### Dataset and models

V2-B used a deterministic 500-question sample from the MMLU-Pro test split,
pinned to repository revision
`b189ec765aa7ed75c8acfea42df31fdae71f97be`. The sample includes every V2-A
ID plus 300 category-stratified additions selected without using observed
results. Category counts range from 35 to 37.

The four model families were unchanged:

- OpenAI `gpt-5.6-sol`, reasoning effort `none`;
- Anthropic `claude-sonnet-5`, thinking disabled;
- Google Vertex AI `gemini-3.8-flash`, thinking level `low`;
- xAI `grok-4.20-0309-non-reasoning`.

Tools, browsing, search, retrieval, code execution, and external context were
disabled. Google used a 512-token output ceiling; the others used 64. That
amendment was recorded before V2-A and was not changed for V2-B.

### Protocol

Stage 1 elicited a frozen multiple-choice answer. Stage 2 elicited a frozen
probability that the answer was correct. Stage 3 requested
`USE_UNVERIFIED` or `VERIFY_FIRST` under the `2 × 2 × 4` factorial:
decision owner (human, AI system), confidence visibility (hidden, visible),
and error cost `L = 2, 5, 10, 20`, with verification cost `C = 1`.

Human and AI prompts used the same action labels, costs, frozen answer,
verifier guarantee, and numerical language. The owner wording was the
intended difference. Visible conditions added one sentence containing the
frozen Stage-2 probability.

Primary prompt family: `v2_owner_match_v1`.

V2-A Stage-1/Stage-2/Stage-3 records for the overlapping 200 questions were
reused from the shared checkpoint. New paid calls covered the 300 added
questions plus a late Anthropic retry after a temporary credit-balance
outage. After retry, coverage was complete.

### Cost model, outcomes, and statistics

Verification costs 1 and is assumed to return the correct answer. Unverified
use costs 0 if the frozen answer is correct and `L` if it is wrong. The
raw-confidence rule verifies when `(1-q)L > 1`.

Primary owner effect:

`P(VERIFY_FIRST | AI owner) - P(VERIFY_FIRST | human owner)`.

Also report paired owner disagreement, because equal marginal rates can hide
opposing question-level switches.

Confidence-visibility effect:

`P(VERIFY_FIRST | visible) - P(VERIFY_FIRST | hidden)`.

Interaction: visible owner gap minus hidden owner gap.

Question-level bootstrap: 5,000 resamples, nominal 95% intervals, keeping
each question's factor cells together. Calibration used deterministic 20%
calibration / 80% evaluation splits (100 / 400 questions per model) with
isotonic regression and no label leakage.

## V2-B data quality

All 2,000 Stage-1 answers, 2,000 Stage-2 confidences, and 32,000 Stage-3
decisions succeeded. Every primary factorial stratum has `n = 500`.

During the first Stage-3 pass, 323 Anthropic cells failed: 320 from an
Anthropic credit-balance error near the end of the run, and 3 from truncated
JSON on the 64-token ceiling. After credits were restored, all 323 were
retried and succeeded. OpenAI, Google, and xAI completed on the first pass.
No cells were imputed.

Run ID: `v2b-core-20260905`. Checkpoint: `results/v2/raw/v2.sqlite3`.

## V2-B result 1: answer accuracy and confidence quality

Frozen-answer performance on 500 questions:

- Google Gemini 3.8 Flash: 88.0% accuracy, 60 wrong, Brier 0.0834;
- OpenAI GPT-5.6 Sol: 82.6% accuracy, 87 wrong, Brier 0.1579;
- Anthropic Claude Sonnet 5: 74.2% accuracy, 129 wrong, Brier 0.1633;
- xAI Grok 4.20 non-reasoning: 64.4% accuracy, 178 wrong, Brier 0.1912.

Wrong-answer denominators are substantially larger than in V2-A (21, 31, 53,
and 70). They are still modest for Google and OpenAI. Accuracy rank and
Brier rank match V2-A. Accuracy still does not determine verification
policy: Google remains most accurate and often leaves wrong answers
unchecked; xAI remains least accurate and verifies almost everything when
confidence is hidden.

## V2-B result 2: the owner effect remains concentrated in Anthropic

Anthropic again showed the only consistent directional owner effect. With
confidence hidden:

- `L=2`: +1.2 percentage points, 95% CI [-0.8, 3.2];
- `L=5`: +1.4 points, CI [-0.6, 3.4];
- `L=10`: +4.0 points, CI [1.8, 6.2];
- `L=20`: +4.8 points, CI [2.8, 7.0].

With confidence visible:

- `L=2`: +3.8 points, CI [2.0, 5.6];
- `L=5`: +3.8 points, CI [2.0, 5.8];
- `L=10`: +4.4 points, CI [2.6, 6.4];
- `L=20`: +5.0 points, CI [3.0, 7.0].

The four visible point estimates average +4.25 points; the four hidden
estimates average +2.85 points. Those averages are descriptive, not
separately bootstrapped. Compared with V2-A, the visible Anthropic effect
is slightly smaller and tighter, and the hidden effect is clearer at high
stakes.

The other families remain near zero in most cells:

- Google ranged from -0.4 to +3.2 points hidden and from -0.8 to +1.4
  visible. Two isolated intervals excluded zero: hidden `L=5` (+3.2, CI
  [1.0, 5.4]) and visible `L=5` (-0.8, CI [-1.6, -0.2]). These are not a
  consistent directional pattern.
- OpenAI ranged from -0.2 to +1.4 points hidden and from -0.2 to +1.0
  visible. No OpenAI owner-effect interval excluded zero.
- xAI ranged from -3.0 to -0.8 points hidden and from +1.0 to +2.6 visible.
  The hidden `L=2` interval excluded zero (-3.0, CI [-5.2, -0.8]), meaning
  slightly *less* AI-owned verification at the lowest stake. The other xAI
  owner intervals included zero.

The defensible conclusion remains heterogeneity, not a universal owner
asymmetry. Isolated non-Anthropic intervals should not be promoted to a
second main effect; many cells were examined without multiplicity
adjustment.

## V2-B result 3: near-zero signed effects still hide action changes

OpenAI's hidden-confidence signed owner effects were between -0.2 and +1.4
points, but paired human-versus-AI actions differed on 11.6% to 15.4% of
questions. With confidence visible, OpenAI disagreement collapsed to 0.2%
to 2.2%.

xAI showed the opposite visibility pattern. Hidden-confidence disagreement
was only 2.8% to 6.2%. Visible-confidence disagreement was 13.4% to 18.2%,
while signed effects stayed between +1.0 and +2.6 points.

Anthropic disagreement was moderate and stable, 4.6% to 6.0%. Google
disagreement was 3.0% to 6.4% when confidence was hidden and usually below
1% when visible, except 3.8% at visible `L=20`.

The paper should keep this distinction: three models are approximately
invariant in marginal verification rates, but OpenAI (hidden) and xAI
(visible) are not invariant at the question level.

## V2-B result 4: confidence visibility still dominates, with opposite signs

Showing the frozen probability again moved verification more than owner
framing, and the direction remained model-specific. Every Anthropic,
OpenAI, and xAI visibility interval excluded zero.

Anthropic: visible confidence increased verification in all eight
owner-by-stake cells, by +12.8 to +22.4 points. The largest increases were
at `L=10` and `L=20` under AI ownership (+20.6 and +22.4).

OpenAI: visible confidence decreased verification in all eight cells, by
16.2 to 20.8 points.

xAI: visible confidence decreased verification in all eight cells, by 8.6
to 16.4 points.

Google: little or slightly negative effect at `L=2` and mixed at `L=5`;
clear increases at high stakes. At `L=10`, +13.0 points under AI ownership
(CI [10.0, 16.2]) and +12.4 under human ownership (CI [9.6, 15.4]). At
`L=20`, +16.4 (CI [13.0, 19.8]) and +16.6 (CI [13.4, 20.0]).

Explicit confidence does not have a universal safety direction.

## V2-B result 5: explicit confidence still does not close the owner gap

The preregistered interaction asked whether visibility moved the signed
owner gap toward zero. Fourteen of 16 model-by-stake intervals included
zero. Point estimates ranged from -4.0 to +4.0 percentage points.

The two intervals that excluded zero were isolated: Google at `L=5`
(-4.0, CI [-6.6, -1.4]) and xAI at `L=20` (+4.0, CI [0.2, 7.8]). They do
not support a general gap-closing mechanism.

For Anthropic, the model with the real owner asymmetry, the descriptive
average signed gap was larger with confidence visible (+4.25) than hidden
(+2.85). Interaction intervals for Anthropic all included zero.

## V2-B result 6: stake sensitivity remains heterogeneous

Averaging the two owners, the verification-rate change from `L=2` to
`L=20` was:

- Anthropic: +11.2 points hidden and +19.4 visible;
- Google: +3.7 points hidden and +22.3 visible;
- OpenAI: +16.1 points hidden and +18.0 visible;
- xAI: +1.6 points hidden and +5.4 visible.

These are descriptive endpoint contrasts. Hidden Google and xAI remain
nearly flat. xAI's hidden rate is already near a ceiling: 94.9% at `L=2`
and 96.5% at `L=20`.

Among complete four-stake trajectories (`n = 2000` per model):

- Anthropic: 32 violations (1.60%);
- Google: 61 (3.05%);
- OpenAI: 96 (4.80%);
- xAI: 277 (13.85%).

Visible confidence reduced violations for Anthropic (24 hidden vs 8
visible per 1000), Google (59 vs 2), and OpenAI (95 vs 1). xAI was again
the exception: 57 hidden vs 220 visible per 1000, or 20.8% and 23.2% of
AI-visible and human-visible trajectories.

## V2-B result 7: wrong-answer failure modes are stable

At `L=20`, unsafe unverified use among actually wrong answers was:

- Anthropic hidden: 10.85% human, 6.98% AI;
- Anthropic visible: 1.55% human, 0.78% AI;
- Google hidden: 55.00% human, 58.33% AI;
- Google visible: 25.00% human, 23.33% AI;
- OpenAI hidden: 32.18% human, 33.33% AI;
- OpenAI visible: 56.32% human, 54.02% AI;
- xAI hidden: 0.00% human, 1.12% AI;
- xAI visible: 4.49% human, 3.93% AI.

Denominators are 129, 60, 87, and 178. Visibility again reduced high-stakes
unsafe use for Anthropic and Google and increased it for OpenAI and, more
modestly, xAI. Wrong-answer owner-gap intervals were generally consistent
with no robust owner difference; the Anthropic hidden `L=20` unsafe gap
interval did exclude zero ([0.8, 7.8] percentage points), but this should
not be over-read given multiplicity.

## V2-B result 8: verification burden and cost

At `L=20`, unnecessary verification among correct answers was approximately:

- Anthropic hidden: 49.6% human, 54.7% AI;
- Anthropic visible: 76.3% human, 82.7% AI;
- Google hidden: 4.1% human, 6.4% AI;
- Google visible: 18.9% human, 20.2% AI;
- OpenAI hidden: 31.2% human, 33.2% AI;
- OpenAI visible: 16.7% human, 17.4% AI;
- xAI hidden: 95.7% human, 94.1% AI;
- xAI visible: 78.6% human, 82.3% AI.

xAI's hidden policy nearly eliminates unsafe wrong-answer use by verifying
almost everything. Anthropic's visible policy is safer on wrong answers and
expensive on correct ones. Google's visible high-stakes policy is cheaper
but still leaves about a quarter of wrong answers unchecked. OpenAI's
visible policy tracks its displayed confidence and inherits that
forecast's errors.

At `L=20`, descriptive mean synthetic costs for direct visible policies
were 0.900 to 0.910 for Anthropic, 0.830 to 0.856 for Google, 2.104 to
2.174 for OpenAI, and 1.152 to 1.166 for xAI. Corresponding raw-confidence
costs were 0.946, 0.552, 2.076, and 0.998. Calibrated-confidence means use
the 400-question evaluation partition and must not be compared with
full-sample direct means as if they were paired unless recomputed on the
same 400 questions.

No result supports a universal ranking of direct, raw-confidence, or
calibrated-confidence policies.

## Comparison with V2-A

The V2-B patterns replicate the V2-A qualitative conclusions:

- owner effect is Anthropic-specific and positive (AI verifies more);
- other models have small signed owner gaps with occasional cancellation;
- confidence visibility is large and opposite across families;
- visibility does not close the owner gap;
- capability and verification policy remain separate.

Quantitative shifts from V2-A to V2-B are modest. Anthropic's visible owner
effect is a bit smaller and more stable (3.8–5.0 vs 3.5–7.5 points).
OpenAI's hidden paired disagreement remains high. xAI's visible
cancellation pattern remains the clearest example of a near-zero signed
effect with large question-level churn. Wrong-answer rates are more
precisely estimated but tell the same safety story.

V2-A had one missing Anthropic cell. V2-B does not.

## Interpretation by preregistered hypothesis

**H1, stake sensitivity:** Partially supported. Verification generally rose
with error cost. Hidden Google and xAI were nearly flat. Question-level
monotonicity violations remained, especially for visible xAI.

**H2, verification-owner effect:** Supported only as a heterogeneous,
model-dependent phenomenon. Anthropic showed a consistent positive signed
effect, especially with confidence visible. Other models had no consistent
directional effect. OpenAI hidden and xAI visible had meaningful paired
disagreement with cancellation.

**H3, confidence visibility changes behavior:** Strongly supported. Large
and replicated, but not universal in direction.

**H4, confidence visibility reduces the owner gap:** Not supported.

**H5, cross-model heterogeneity:** Strongly supported.

**H6, capability and verification quality are separate:** Supported
descriptively. The most accurate model did not have the lowest unsafe-use
rate.

## Strongest defensible paper claim

> In a paired 500-question verification experiment with frozen answers and
> uncertainty, decision-owner framing did not produce a universal directional
> effect across four model families. Claude Sonnet 5 selected verification
> 3.8–5.0 percentage points more often when the AI system rather than the
> human controlled the decision under visible-confidence prompts, with
> nominal paired 95% bootstrap intervals excluding zero at all four error
> costs. Signed owner effects for the other models were near zero, although
> question-level owner disagreement reached 12–18% in some OpenAI hidden and
> xAI visible conditions. Showing frozen confidence produced larger,
> oppositely directed policy shifts and did not reliably close the owner gap.

## Claims the evidence does not support

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
- that nominal cellwise intervals are multiplicity-adjusted;
- that near-zero marginal owner effects imply identical paired decisions;
- that V2-A and V2-B are independent replications on disjoint samples.

## Statistical cautions

Reported 95% intervals are nominal question-bootstrap intervals. Many
cellwise effects were examined without multiplicity adjustment. Emphasize
repeated within-model patterns, especially Anthropic's visible owner effect
across all four stakes, rather than isolated intervals.

Intervals capture variation over sampled questions, not API-generation
stochasticity, provider drift, or model-selection uncertainty.

The 323 Anthropic retries after a billing outage are valid completed
records. They are not a second independent draw of those items.

Isotonic calibrators were fitted on 100 questions per model and evaluated
on 400. Treat calibration comparisons cautiously.

## General limitations

Synthetic numerical costs; no human participants; multiple-choice MMLU-Pro
may be contaminated in closed-model training data; self-reported confidence
is not an internal probability; verification is modeled as perfect and
costs exactly one; owner framing is binary; provider reasoning settings are
not comparable; binary structured actions do not measure explanation
quality; prompt-paraphrase robustness has not been run.

## What remains before the intended final paper package

1. Run the preregistered 100-question independent prompt-paraphrase
   robustness subset (`v2_owner_match_paraphrase_v1`, confidence hidden).
2. Recompute robustness tables and, if desired, paired policy contrasts on
   matched evaluation samples.
3. Consider GPQA only after prompt robustness.
4. Keep V2-A labeled as validation. Use V2-B as the confirmatory
   MMLU-Pro result.

## Suggested abstract for a paper describing V2-B

Language models increasingly provide answers that users may accept without
independent checking. We study whether models recommend verification
consistently when an answer is frozen and only the owner of the verification
decision changes. On 500 MMLU-Pro questions, four model families first
generated an answer and a probability of correctness. We then elicited
32,000 binary verification decisions in a matched `2 × 2 × 4` design
crossing decision owner (human or AI system), visibility of the frozen
confidence, and error cost. Claude Sonnet 5 verified 3.8–5.0 percentage
points more often under AI ownership when confidence was visible, with
nominal paired 95% bootstrap intervals excluding zero at all four costs.
The other models showed signed owner effects near zero, although
question-level owner disagreement reached 12–18% in some conditions.
Confidence visibility produced larger but opposing policy shifts: it
increased verification for Claude, decreased it for GPT-5.6 Sol and Grok,
and increased it mainly at high stakes for Gemini. Confidence visibility
did not reliably reduce the owner gap. Verification policy is
model-specific and sensitive to how uncertainty and decision ownership are
presented. These results provide no evidence for a universal owner
asymmetry. Prompt-paraphrase robustness remains future work.

## Canonical result files

Current V2-B outputs are in `paper_outputs/v2/`:

- `model_summary.csv`;
- `factorial_metrics.csv`;
- `owner_effects.csv`;
- `confidence_visibility_effects.csv`;
- `owner_confidence_interactions.csv`;
- `policy_comparison.csv`;
- `monotonicity.csv`;
- `factor_completeness_issues.json` (empty);
- generated PDF and PNG figures.

The V2-A archive is `paper_outputs/v2a/`. The frozen protocol is
`EXPERIMENT_V2.md`. V1 is pilot evidence only.
