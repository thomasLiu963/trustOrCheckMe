# Task 005 parallel execution log

## Preflight (before paid calls)

Printed and recorded before Lane B paid execution.

| item | value |
|---|---|
| Historical V2 dataset | `results/v2/raw/v2.sqlite3` (sha256 `8597e3f73226bf39770af99a877282c22428203a3e2c11521256bf5cb3db5eb6`) |
| Study-1 database | `results/study1_causal_pilot/study1.sqlite3` (sha256 `f8e5ddbb91db360403ea519b9b3481575c6cbc5ccdb553c650dbe726025ecfbd`) |
| GPT model / endpoint | `openai_gpt56_sol` → `gpt-5.6-sol` (OpenAI Responses, `reasoning.effort=none`, max_output_tokens=64) |
| Claude model / endpoint | `anthropic_sonnet5` → `claude-sonnet-5` (Messages, `thinking.type=disabled`, max_output_tokens=64) |
| Lane B scientific cap | 1,000 (500 questions × 2 models) |
| Lane C | **not supported**; `BLOCKED_UNSUPPORTED_REASONING_SETTING`; 0 calls |
| `paperDirection.txt` | will not be changed (sha256 `52cbde69e42f4db59f481d574a4f4395d1c7acfebb0b6f13a4390eda2c1f1d36`) |
| API keys required from user | none; `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` already present |

Gemini/Grok keys exist in `.env` but were not used.

## Lane order

1. Lane A — zero-cost analysis of existing Study-1 and V2 data. Read-only sqlite (`mode=ro`). No CheckpointStore open on historical files.
2. Lane D — qualitative-stakes prompt prep. No calls.
3. Lane C — blocked after adapter/config inspection. No calls.
4. Lane B — dry-run prompt audit (0 calls), then paid 1,000 q2 re-elicitations into a new sqlite.

Lanes B and C were not run concurrently. Lane C made no GPT traffic.

## Isolation

| stream | path |
|---|---|
| Historical V2 | `results/v2/raw/v2.sqlite3` (read-only; not opened writable) |
| Study 1 | `results/study1_causal_pilot/study1.sqlite3` (read-only) |
| Lane B q2 | `results/study2_q2_diagnostic/q2.sqlite3` |
| Lane C | no database |

## Notes

Scientific cleanliness preferred over maximum concurrency. q2 used the same concurrency=4 pattern as Study 1, writing only the new q2 sqlite.

## Actual paid execution

- Lane B started after dry-run prompt audit (0 errors).
- 1000/1000 scientific successes. 0 failures. 0 retries. 0 parse repairs.
- Wall-clock 418.9 s. Estimated $1.383.
- V2, Study 1, and paperDirection hashes unchanged after the run.
