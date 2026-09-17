# Study 1 schema summary

Study 1 uses a **new** record type: `src.study1_schemas.Study1DecisionRecord`.

It does **not** modify or relax historical `src.schemas.VerificationDecisionRecord` (`experiment_version="v2"`).

## Separate confidence fields

| Field | Meaning |
|---|---|
| `reported_confidence` | Historical V2 Stage-2 value, originally stored as `probability_correct`. Copied, never overwritten. |
| `displayed_confidence` | Number actually inserted into the Study 1 Stage-3 prompt, or `null` if hidden. |

Hidden condition: `displayed_confidence` is JSON `null` / CSV empty. The prompt contains no “The AI previously estimated a …” sentence.

True-visible: `displayed_confidence = reported_confidence`.

Manipulated: `displayed_confidence` equals the frozen grid value for that `L`. `reported_confidence` still stores the historical value on the same row.

## Other required fields

`study_id`, `pilot_or_confirmatory`, `question_id`, `model_id` / `model_alias`, `model_endpoint`, `api_timestamp` (null until a later paid run), `L`, `C`, `display_condition`, `display_source_condition`, `frozen_answer`, `stage1_correct`, `prompt_hash`, `attempt_number`, `raw_response`, `parsed_action`, `parse_status`, `latency_ms`, `input_tokens`, `output_tokens`, `estimated_cost_usd`.

Provenance also stored: `prompt_version`, `prompt_family`, `selected_sample_hash`, `code_commit`, `repeat_index`, historical Stage-1/2 request keys, `model_settings`.

## Action schema

Unchanged binary JSON:

```json
{"action":"USE_UNVERIFIED"}
```

or

```json
{"action":"VERIFY_FIRST"}
```

## Write target

New path only: `results/study1_causal_pilot/study1.sqlite3`

Historical path is read-only: `results/v2/raw/v2.sqlite3`
