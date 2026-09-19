# D1 data card

## Purpose

Release the item-level measurements needed to reproduce the D1 headline analyses and to evaluate alternative *offline* routing rules on the same frozen labels.

This artifact is **not a leaderboard** and **not a new benchmark**. It does not define a D1 score that others should optimize. The underlying tasks remain MMLU-Pro and LiveCodeBench.

## Prospective / preregistered analyses

| Task | Label | What was frozen before outcome inspection |
|---|---|---|
| 009 | confirmatory / preregistered | MMLU-Pro N=500 sample, Stage-1/q1/Stage-3 protocol, coverage 0.70 vs 0.99 |
| 011 | confirmatory / preregistered | LiveCodeBench hard N=286 sample, code/q1/Stage-3 protocol, hidden tests after routing freeze |
| 014 | confirmatory for the offset; post-hoc relative to 009–013 | rank-preserving \(\delta\) grid and transform, before new Stage-3 VERIFY rates |

Primary models: `openai_gpt56_sol` → `gpt-5.6-sol`; `anthropic_sonnet5` → `claude-sonnet-5`.

## Post-hoc systems analyses

| Task | Label | Uses only frozen 009/011/014 |
|---|---|---|
| 012 | `POST_HOC_SYSTEMS_REANALYSIS` | leakage, cost sweep \(L=\mathrm{leakage}+\lambda\cdot\mathrm{coverage}\) |
| 015 | `POST_HOC_NORMATIVE_REANALYSIS` | unmanipulated conditional escape, rank-preserving loss spread |

Do not present 012/015 as preregistered experiments.

Task 010 ranking-sensitivity audits and Task 013 hidden-confidence policy audits are **not** in this package.

## Task / model coverage in the release

| Task | Items | Conditions | Models |
|---|---|---|---|
| MMLU-Pro test, revision `b189ec765aa7ed75c8acfea42df31fdae71f97be` | 500 | hidden, true_q_visible, displayed {0.70,0.85,0.90,0.95,0.99}, 014 \(\delta\in\{-1.5,-0.75,0,+0.75,+1.5\}\) | GPT, Claude |
| LiveCodeBench `code_generation_lite` `release_v6`, commit `28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24`, hard_only | 286 | same | GPT, Claude |

Repeat-draw Stage-3 rows and the 009 Gemini/Grok secondary roster are omitted.

## Files

- `data/item_level_routing.csv` — 009/011 primary routing + labels
- `data/item_level_offset.csv` — 014 rank-preserving routing + labels
- `data/manifests/` — freeze summaries, ID lists, sanitized preregistrations
- `data/derived_*.csv` — written by `reproduce_main.py`
- `prompts/` — templates with `{question}` / `{frozen_*}` placeholders
- `analysis/` — loaders, metrics, figures
- `examples/evaluate_threshold_router.py` — offline substitute router

## Metrics

See README. Conditional escape is \(P(\texttt{USE_UNVERIFIED}\mid\text{incorrect})\). It is not a false-omission rate and not an industry defect rate.

Bootstrap: item-clustered, 5000 resamples. Seeds: 012 = 20260925, 014 = 20260927, 015 = 20260928.

## Known limitations

- Frozen generated code and benchmark text are omitted, so a new LLM router cannot be queried from this tarball alone.
- LCB pass/fail is a released *outcome*, not a redistributable test suite.
- GPT \(q_1\) is often piled near 0.99; a q1-only calibrated baseline can collapse to always/never verify.
- Claude code is near VERIFY saturation; small coverage moves are a regime boundary, not a safety ranking.
- 012/015 are post-hoc.
- Closed models; results are version-specific (`gpt-5.6-sol`, `claude-sonnet-5`).

## Excluded data

MMLU-Pro stems, options, and official answer text; LiveCodeBench problem statements, starter code, hidden tests, and generated programs; raw SQLite; filled prompts that contain benchmark content; `.env` and API keys; Task 013; historical V1/V2; Gemini/Grok 009 cells.

## Recommended citation posture

Cite D1 for the measurement procedure and reported causal results. Cite MMLU-Pro and LiveCodeBench for the underlying tasks. Do not describe this package as “the D1 benchmark.”
