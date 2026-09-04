# Pilot decision: GO

Decision date: 2026-09-04

The pilot provides sufficient non-obvious evidence to justify the planned main
run. This is a research planning decision, not a claim of statistical or
real-world generality.

## Primary evidence

- OpenAI GPT-5.6 Sol had higher frozen-answer accuracy than Anthropic Claude
  Sonnet 5 (84.5% versus 73.5%).
- Despite higher accuracy, OpenAI's direct advice was only weakly sensitive to
  error cost: its VERIFY rate was 14.0% at `L=2` and 17.5% at `L=20`.
- Anthropic's VERIFY rate increased from 42.5% to 53.0% over the same range.
- At `L=20`, unsafe reliance among wrong answers was 58.1% for OpenAI and 9.4%
  for Anthropic.
- Monotonicity violation rates were 9.5% for OpenAI and 2.0% for Anthropic.
- On the held-out calibration evaluation split at `L=20`, OpenAI direct advice
  incurred 0.963 more cost per decision than its calibrated-confidence policy
  (95% bootstrap CI: 0.119 to 1.856).

These results support the project's central possibility: answer capability and
human-facing trust-guidance quality need not move together.

## Required next step

Run the preregistered 1,400-question MMLU-Pro experiment with the same frozen
protocol and add a third independent model family. Do not alter the prompt or
cost grid based on these pilot outcomes.

Consider GPQA only after the main MMLU-Pro run is complete and only as a
separately identified robustness extension.

## Interpretation limits

The pilot uses synthetic costs, multiple-choice questions, self-reported
confidence, and no human participants. Results concern model recommendations
under this protocol; they do not establish how people behave or whether a model
is broadly trustworthy.
