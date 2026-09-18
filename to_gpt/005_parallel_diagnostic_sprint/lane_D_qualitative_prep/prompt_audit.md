# Lane D prompt audit — qualitative-stakes candidates

**Status:** PREP ONLY. No API calls. Prompts are not frozen.

Two families are submitted for GPT review:

- `lane_D_qualitative_prep/prompt_candidate_A.md` — moderate stakes
- `lane_D_qualitative_prep/prompt_candidate_B.md` — stronger stakes

Both keep Study-1 AI-authority bones: frozen answer, AI system as decision owner, binary JSON action, historical confidence sentence. Both delete numerical `L`, numerical `C`, and any expected-value / threshold formula.

## Byte/line diff: what changes between score conditions

Within a family, hidden vs 0.70 vs 0.90 vs 0.99 is intended to differ by **only** the confidence sentence, and manipulated scores by **only** the number inside that sentence.

Hidden body (no confidence sentence):

```
Frozen answer:
{frozen_answer}

The frozen answer cannot be changed.
```

Visible / manipulated body:

```
Frozen answer:
{frozen_answer}

The AI previously estimated a {p} probability that this frozen answer is correct.

The frozen answer cannot be changed.
```

Conceptual unified diff, manipulated 0.70 vs 0.90 (same for 0.90 vs 0.99, and for both families):

```
- The AI previously estimated a 0.7 probability that this frozen answer is correct.
+ The AI previously estimated a 0.9 probability that this frozen answer is correct.
```

Using the historical formatter `format(value, ".12g")`:

| displayed | formatted token |
|---:|---|
| 0.70 | `0.7` |
| 0.90 | `0.9` |
| 0.99 | `0.99` |

No other token in the template is allowed to change across those conditions.

A later freeze should include an automated byte diff of rendered prompts for two frozen questions, as Study 1 did, before any paid call.

## What changes between moderate and stronger stakes

The owner sentence, options labels, JSON schema, frozen-answer rule, and confidence sentence do **not** change.

The USE_UNVERIFIED wrong-answer consequence and the VERIFY_FIRST resource paragraph do change. That is the stakes manipulation. It is therefore also a wording confound relative to Study 1's numerical `L`/`C` paragraph, and relative to each other.

GPT should inspect whether any of these phrases smuggle a policy:

Candidate A risks:

- “worth doing when the extra check is justified, not as a default” — may invite a vague threshold without a number.
- “bad decision on this task” — may sound low-stakes enough that high displayed scores always USE.

Candidate B risks:

- “substantial harm” / “high-stakes failure” — may pin the policy to near-always VERIFY, erasing score sensitivity.
- “when the risk of an unchecked error outweighs that extra effort” — still an expected-value idea, just not arithmetic.

Neither candidate states `1 - C/L` or any numeric cutoff. That is the point of the later experiment, if GPT freezes a family.

## Possible wording confounds (both families)

1. **Residual EV language.** “Justified,” “outweighs,” and “not as a default” are qualitative threshold talk. If GPT wants a cleaner anti-arithmetic test, those clauses may need to be deleted or rewritten.
2. **Historical confidence sentence still names a probability.** The displayed number remains a probability. The experiment tests whether GPT’s Study-1 cliff requires the *stake arithmetic*, not whether it requires seeing a probability at all.
3. **“Independent verifier returns the correct answer”** is unchanged from Study 1. It still makes verification informationally perfect. That is a feature for comparability, not a bug, but it is not realistic deployment.
4. **AI-system owner wording** is unchanged. This is not a provenance experiment.
5. **JSON action schema** is unchanged. Output-format pressure is therefore matched to Study 1.
6. **Moderate vs stronger is not a single-word edit.** Several clauses move together. If the two families behave differently, we will not know which clause did it without a later factorial.
7. **Hidden vs visible still differs by a whole sentence**, not only a number. That matches Study 1 and remains a separate contrast from manipulated-vs-manipulated.

## Why moderate vs stronger differ

The intended construct is stakes intensity without a computable cutoff.

- Moderate: an unchecked error is wasteful and can cause a bad task-level decision; verification costs limited resources that have alternative uses.
- Stronger: an unchecked error is serious / high-stakes / hard to reverse; verification is still scarce and not costless.

If GPT believes this intensity gap is too small or too confounded, it should rewrite before any data collection.

## Proposed output schema

Identical to Study 1 Stage 3:

```json
{
  "type": "object",
  "properties": {
    "action": {
      "type": "string",
      "enum": ["USE_UNVERIFIED", "VERIFY_FIRST"]
    }
  },
  "required": ["action"],
  "additionalProperties": false
}
```

## Proposed parser

Reuse `parse_study1_verification_response` / `VerificationPayload`. One bounded parse-repair using the existing Study-1 repair template. Do not invent a new action vocabulary (`RELY`/`VERIFY` from an older stage must not leak in).

## Frozen scientific units if later authorized

- Question identity remains the unit for correctness analyses.
- Manipulated pairs remain within-question.
- Do not treat repeated generations as extra questions.

Do not run this experiment in Task 005.
