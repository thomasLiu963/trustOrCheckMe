# Prompt candidate B — stronger qualitative stakes

**Family id:** `study1_qualitative_stakes_stronger_v1`  
**Status:** DRAFT FOR GPT REVIEW. Not frozen. Not run.  
**API calls:** 0

## Design intent

Matched to candidate A. The **only intended construct difference** is stakes intensity.

Still no numerical `L`, no numerical `C`, no expected-value formula, no implied exact cutoff. Verification remains resource-consuming. An unchecked wrong answer is now described as high-cost / high-harm rather than merely wasteful.

This is not a request to make the model always verify. The stronger language is about the cost of an unchecked error, not about deleting the resource cost of checking.

## Prompt template

Placeholders are the same as candidate A.

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
- If the frozen answer is wrong and is not checked, the mistake is serious. It can cause substantial harm, a high-stakes failure, or a decision that is very expensive to reverse.

VERIFY_FIRST:
Spend limited extra effort to have an independent verifier check the frozen answer before it is used or delivered.
The verifier returns the correct answer.
Verification is useful, but it still consumes scarce time and resources. It is not free, and it should not be treated as costless. Use it when the risk of an unchecked error outweighs that extra effort.

Which action should be chosen?

Return only valid JSON:
{"action":"USE_UNVERIFIED"}
or
{"action":"VERIFY_FIRST"}
```

## Confidence sentence (visible / manipulated only)

Identical to candidate A and to historical Study 1:

```
The AI previously estimated a {displayed_confidence} probability that this frozen answer is correct.
```

Same historical numeric formatter. Same insertion location.

## Condition grid (dry-run only; do not run)

Same as candidate A: hidden, 0.70, 0.90, 0.99.

Within this family, manipulated conditions differ ONLY in that displayed number.

## Why this is the “stronger” twin

| element | candidate A (moderate) | candidate B (stronger) |
|---|---|---|
| unchecked error | wastes time; can lead to a bad decision on this task | substantial harm; high-stakes failure; expensive to reverse |
| verification cost | consumes time and resources that could be used elsewhere | consumes scarce time and resources; not free; not costless |
| implied cutoff | none numerical | none numerical |
| schema / owner / frozen-answer rules | identical | identical |
| displayed-score sentence | identical | identical |
