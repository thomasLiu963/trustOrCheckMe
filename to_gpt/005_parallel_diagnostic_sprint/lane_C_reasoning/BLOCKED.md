# Lane C — BLOCKED_UNSUPPORTED_REASONING_SETTING

**Status:** `BLOCKED_UNSUPPORTED_REASONING_SETTING`  
**Paid Lane-C calls made:** 0  
**Smoke-test calls made:** 0

## Scientific question (not answered)

Does a higher reasoning setting change GPT's extreme score-dominance behavior?

This lane was allowed to proceed **only if** a nonzero reasoning setting was already clearly supported for the exact Study-1 GPT endpoint (`gpt-5.6-sol` via the OpenAI Responses API).

## What was inspected

| Location | Finding |
|---|---|
| `config/models.yaml` | `openai_gpt56_sol.reasoning_effort: none` |
| `src/config.py` `ModelSpec.reasoning_effort` | Type is `Literal["none"] \| None`. Anything else fails validation. |
| `src/config.py` `validate_provider_settings` | `"OpenAI pilot models must use Responses with reasoning effort none"` |
| `src/model_adapters.py` `OpenAIAdapter.prepare_request` | Payload is hardcoded `"reasoning": {"effort": "none"}` |
| `src/cli.py` | Frozen-payload check: OpenAI reasoning effort must remain `none` |
| `tests/test_model_adapters.py` | Asserts `payload["reasoning"] == {"effort": "none"}` |
| Study-1 stored `model_settings` | GPT cells record `requested_model_id=gpt-5.6-sol` with no nonzero reasoning value |

No adapter path, config enum, or test fixture exposes a supported nonzero effort such as `low`, `medium`, or `high` for this endpoint.

## Why this is a stop, not a guess

Task 005 forbids:

- guessing a parameter name;
- cycling through undocumented settings;
- switching model family to manufacture a reasoning comparison.

The parameter **name** `reasoning.effort` is known, because the frozen Study-1 adapter already sends it. The only **supported value in this repository** is `none`. Adding `low`/`medium`/`high` would be inventing a setting the current adapter/config explicitly reject.

A 4-call technical smoke test is allowed only to verify an *already-supported* setting. That precondition is not met, so zero Lane-C calls were made.

## What is missing

A first-class, already-wired nonzero reasoning value for `gpt-5.6-sol` in:

1. `config/models.yaml` / `ModelSpec` (allowed effort enum including a nonzero value);
2. `OpenAIAdapter.prepare_request` (must read that value instead of hardcoding `none`);
3. tests that lock the nonzero setting rather than forbidding it.

GPT would also need to decide whether a nonzero-effort `gpt-5.6-sol` call is the same endpoint/version as Study 1 or a material version shift.

## What was not done

- No paid GPT reasoning grid (0 / 800).
- No model-family substitution.
- No undocumented parameter search.
- No write to `results/study1_causal_pilot/study1.sqlite3` or `results/v2/raw/v2.sqlite3`.
- `results/study1_reasoning_diagnostic/` was not created as a results database.

READY_FOR_GPT_REVIEW of this lane = YES (blocked, as specified).
