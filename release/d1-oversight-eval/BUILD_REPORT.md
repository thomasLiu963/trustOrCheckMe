# D1 public package — build report

Packaging only. No model calls. No scientific-result edits. `paperDirection.txt` not modified.

## Files created

```
release/d1-oversight-eval/
  README.md
  DATA_CARD.md
  LICENSES.md
  BUILD_REPORT.md
  reproduce_main.py
  data/item_level_routing.csv          # 11,004 rows (500×2×7 + 286×2×7)
  data/item_level_offset.csv           # 7,860 rows (1,572 items × 5 deltas)
  data/derived_*.csv                   # written by reproduce_main.py
  data/manifests/
    freeze_009.json
    freeze_011.json
    freeze_014.json
    freeze_015.json
    id_lists.json
    preregistration_009.md
    preregistration_011.md
    preregistration_014.md             # SQLite path stripped
    reproduction_summary.json
  prompts/stage3_mmlu_moderate.txt
  prompts/stage3_code_moderate.txt
  prompts/hashes.json
  analysis/{__init__,constants,io_csv,metrics,figures}.py
  figures/
    fixed_score_coverage_leakage.{png,pdf}
    rank_preserving_response.{png,pdf}
    rank_preserving_loss_spread.{png,pdf}
    unmanipulated_escape_callout.{png,pdf}
  examples/evaluate_threshold_router.py
```

A one-time extractor (`_pack_from_repo.py`) was used to build the CSVs from frozen 009/011/014 tables and was **not** left in the package (it pointed at internal repo paths).

## Source files used

- `to_gpt/009_prospective_ranking_invariance_confirmation/{stage1_q1_results,stage3_results,preregistration,freeze_manifest}.csv/md/json`
- `to_gpt/011_code_executable_verification_generalization/{q1_results,stage3_results,hidden_test_results,sample_main,preregistration,freeze_manifest}`
- `to_gpt/014_rank_preserving_confidence_shift/{stage3_offset_results,preregistration}`
- Headline targets from 009/011/012/014/015 reports and `to_gpt/015_…/unmanipulated_escape_rate.csv` / `spread_summary.json`
- Prompt wording from `src/task006_prompts.py` and `src/task011_prompts.py` (placeholders only)
- Bootstrap seeds from `src/task012_common.py`, `src/task014_common.py`, `src/task015_common.py`

Not read into the package: SQLite files, `examples_500.jsonl`, LCB `main_problems.jsonl`, `.cache/livecodebench_data/`, `.env`, Task 013, V1/V2.

## Headline-value reproduction checks

`python reproduce_main.py` exited 0. Assertions (tolerance 0.6 pp or 0.006 rate / 0.002 loss):

| Check | Regenerated | Frozen target |
|---|---|---|
| 009 GPT coverage 0.70−0.99 | 50.2 pp [45.8, 54.6] | 50.2 pp |
| 009 Claude coverage | 35.4 pp [31.2, 39.6] | 35.4 pp |
| 011 GPT coverage | 38.8 pp | 38.8 pp |
| 011 Claude coverage | 4.9 pp | 4.9 pp |
| 011 GPT pass rate | 46.5% | 46.5% |
| 011 Claude pass rate | 43.0% | 43.0% |
| 012 GPT MMLU leakage 0.70→0.99 | +11.8 pp | +11.8 pp |
| 012 GPT code leakage | +22.0 pp | +22.0 pp |
| 012 cost sweep GPT MMLU displayed_0.99 λ=0.001 | 0.138082 | 0.138082 |
| 014 GPT MMLU δ +1.5 vs −1.5 | −16.2 / +6.0 pp | −16.2 / +6.0 |
| 014 GPT code | −7.0 / +4.5 pp | −7.0 / +4.5 |
| 014 Claude MMLU | −31.2 / +3.4 pp | −31.2 / +3.4 |
| 014 GPT MMLU retention cov/leak | 0.323 / 0.508 | 0.323 / 0.508 |
| 014 GPT code retention cov/leak | 0.180 / 0.206 | 0.180 / 0.206 |
| 015 GPT-code escape | 0.555556 [0.472, 0.635] | 55.6% [47.2%, 63.5%] |
| 015 max loss spread GPT MMLU / code / Claude MMLU / Claude code | 0.102 / 0.059 / 0.278 / 0.038 | same |

Offline example ran: observed GPT-code true-q router escape 0.556; threshold-on-displayed-q is a different policy (as expected).

## Excluded files and why

| Excluded | Why |
|---|---|
| MMLU-Pro stems / options / official answers | license-sensitive; HF MIT vs GitHub Apache-2.0 unresolved |
| LCB problem text, starter code | contest-sourced; HF `license: cc` unspecified |
| LCB hidden tests and `.cache/livecodebench_data/` | must not be redistributed |
| Frozen generated programs | provider output terms not reviewed in-repo |
| Raw SQLite / filled prompts | contain benchmark text and/or code |
| `.env`, API keys | credentials |
| Task 013 | paperDirection: internal only |
| 009 Gemini/Grok secondary + repeats | not on the 012–015 spine |
| Historical V1/V2 | superseded, not confirmatory D1 |

## Unresolved license questions

1. D1 analysis code has **no** repository license.
2. MMLU-Pro GitHub Apache-2.0 vs Hugging Face MIT.
3. LiveCodeBench MIT harness vs unspecified HF `cc` vs contest-site terms.
4. OpenAI / Anthropic output-redistribution terms (code omitted; MMLU letters included).

See `LICENSES.md`. Nothing here is a permission grant.

## Blockers

**Scientific packaging:** none. Reproduction passes from the released CSVs.

**Public upload / Zenodo / GitHub:** blocked until the authors (1) add a license grant for D1 code and measurements, and (2) accept or clear the unresolved upstream questions. The conservative release policy is already applied; the remaining blocker is legal/policy, not missing tables.

## Ready for public release?

**Ready as a rights-safe, internally reproducible paper supplement.**  
**Not ready to publish to the internet** until a D1 license is chosen and the unresolved questions in `LICENSES.md` are either cleared or explicitly accepted by the authors.

No manuscript or `paperDirection.txt` changes were made.

READY_FOR_GPT_REVIEW = YES
