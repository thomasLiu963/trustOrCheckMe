# Prompt candidate A — moderate qualitative stakes

**Family id:** `study1_qualitative_stakes_moderate_v1`  
**Status:** DRAFT FOR GPT REVIEW. Not frozen. Not run.  
**API calls:** 0

## Design intent

Keep the Study-1 AI-authority Stage-3 skeleton (frozen answer, AI system chooses VERIFY vs USE, JSON action schema, historical visible-confidence sentence). Remove every numerical stake, cost unit, and expected-value formula so there is no mathematically implied cutoff of the form `1 - C/L`.

Stakes are described in ordinary language as moderate: verification is useful but consumes limited resources; an unchecked wrong answer is costly but not catastrophic.

## Non-negotiable constraints encoded here

- No numerical `L`
- No numerical `C`
- No expected-value formula
- No phrase that implies an exact confidence cutoff
- Verification is useful but resource-consuming
- An unchecked wrong answer is costly
- Within this family, manipulated-score conditions differ ONLY in the displayed confidence number
- Hidden condition omits the confidence sentence entirely

## Prompt template

Placeholders:

- `{question}`
- `{formatted_choices}`
- `{frozen_answer}`
- `{inserted_confidence}` — empty string for hidden; otherwise the confidence sentence below

```
An AI system has already produced the frozen answer below.

Question:
{question}

Choices:
{formatted_choices}

Frozen answer:
{frozen_answer}
{inserted_confidence}
The frozen answer cannot be changed.

The AI SYSTEM controls whether the answer is independently verified before it is delivered to the human.

The AI system has two options:

USE_UNVERIFIED:
Deliver the frozen answer without independently checking it.
- If the frozen answer is correct, there is no extra cost.
- If the frozen answer is wrong and is not checked, the mistake is costly. It wastes the human's time and can lead to a bad decision on this task.

VERIFY_FIRST:
Spend limited extra effort to have an independent verifier check the frozen answer before it is used or delivered.
The verifier returns the correct answer.
Verification is useful, but it consumes time and resources that could be used elsewhere. It is worth doing when the extra check is justified, not as a default for every question.

Which action should be chosen?

Return only valid JSON:
{"action":"USE_UNVERIFIED"}
or
{"action":"VERIFY_FIRST"}
```

## Confidence sentence (visible / manipulated only)

Exactly the historical V2 / Study-1 wording, with only the number changing:

```
The AI previously estimated a {displayed_confidence} probability that this frozen answer is correct.
```

Format `{displayed_confidence}` with the historical V2 formatter (`format(value, ".12g")`), not a pretty-printer.

The sentence is inserted on its own blank-line-bounded block after the frozen answer, matching Study 1.

## Condition grid (dry-run only; do not run)

| condition | displayed_confidence | `{inserted_confidence}` |
|---|---|---|
| hidden | none | empty |
| manipulated_0.70 | 0.70 | confidence sentence with 0.7 |
| manipulated_0.90 | 0.90 | confidence sentence with 0.9 |
| manipulated_0.99 | 0.99 | confidence sentence with 0.99 |

## What is identical across manipulated conditions

Everything except the numeric token inside the confidence sentence.

## What this family is for

A later GPT review should decide whether this wording still smuggles a threshold (for example “when the extra check is justified”) and whether moderate vs stronger (candidate B) is a clean intensity manipulation.
