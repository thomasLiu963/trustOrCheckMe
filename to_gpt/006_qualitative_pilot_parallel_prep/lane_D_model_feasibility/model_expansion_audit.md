# Model-expansion feasibility audit

Zero provider calls. Code and historical V2 sqlite inspected only.

## Closed models

### Gemini (`google_gemini38_flash`)

- Status: **READY**
- Endpoint: `gemini-3.8-flash` via `google_genai_vertex`
- Adapter: `GoogleAdapter` exists
- Historical V2 Stage-1/2 rows: 500 / 500
- Study 1 / q2 replication: none
- Later 20-question qualitative grid if authorized: 160 scientific calls
- Notes: create_adapter already accepts this frozen ID; Study 1 / q2 / qualitative grids were never run for this model

### Grok (`xai_grok420_nonreasoning`)

- Status: **READY**
- Endpoint: `grok-4.20-0309-non-reasoning` via `responses`
- Adapter: `XAIAdapter` exists
- Historical V2 Stage-1/2 rows: 500 / 500
- Study 1 / q2 replication: none
- Later 20-question qualitative grid if authorized: 160 scientific calls
- Notes: create_adapter already accepts this frozen ID; Study 1 / q2 / qualitative grids were never run for this model

## Open models

No local/open-weight inference adapter. See `open_model_requirements.md`. Status: **BLOCKED** (missing hook).

Do not run Gemini, Grok, or open models in Task 006.
