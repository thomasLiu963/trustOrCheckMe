# Study 1 rendered-prompt audit

Offline audit only. No model API calls.

## Historical wording source

Study 1 clones the historical V2 **AI-system / primary** Stage-3 template.

- File: `src/v2_prompts.py`
- Function: `build_verification_prompt`
- Family: `v2_owner_match_v1` (not the paraphrase family)
- Owner: AI-system, `paraphrase=False`
- Intro: line 21
- Owner/options/unverified sentences: lines 69–73
- Visible confidence sentence: lines 105–111
- Cost/closing sentences: lines 122–126
- Full template: lines 128–159

Visible sentence, unchanged:

> The AI previously estimated a {p} probability that this frozen answer is correct.

Probability formatting reuses historical `_format_probability` (`format(value, ".12g")`). That is why grid value `0.80` appears in prompts as `0.8` and `0.90` as `0.9`. This is historical continuity, not a grid change.

Tests confirm Study 1 true-visible and hidden prompts are byte-identical to V2 AI-authority visible/hidden prompts when the displayed number equals the V2 `probability_correct` argument.

## Presentation-only example questions

These IDs were chosen from the already-frozen 100-question sample using GPT’s historical reported confidence only, so GPT can inspect a high-confidence and a lower-confidence prompt.

They were **not** chosen using correctness or historical Stage-3 actions.
They **do not** change the experimental sample.

| Role | Question ID | GPT reported confidence |
|---|---|---|
| High | `mmlu_pro:test:10539` | 0.999 |
| Lower | `mmlu_pro:test:774` | 0.03 |

Both GPT (`openai_gpt56_sol`) and Claude (`anthropic_sonnet5`) prompts were rendered for L=10 and L=20, all seven conditions.

## Hidden vs visible

Hidden prompts contain no “The AI previously estimated a …” sentence.

True-visible adds exactly that sentence, using the model’s historical reported confidence. Example GPT L=10 on `10539`:

> The AI previously estimated a 0.999 probability that this frozen answer is correct.

The hidden-vs-true-visible diffs are stored separately. That difference is expected.

## Manipulated prompts

Within a given question, model, and L, the five manipulated prompts differ **only** in the displayed numeric confidence value.

L=10 sequence: 0.80 → `0.8`, 0.88, 0.89, 0.91, 0.99  
L=20 sequence: 0.90 → `0.9`, 0.93, 0.94, 0.96, 0.99

Machine diffs are in `prompt_diffs/`. The builder raises if any non-confidence line changes.

## Authority / schema

All audited prompts use AI-system authority. None contain `HUMAN USER`. Output schema remains the two historical JSON actions.
