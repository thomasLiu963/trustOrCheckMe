# Licenses and unresolved redistribution questions

This file inventories every upstream component we know about. **Do not treat anything marked unresolved as a grant of permission.** This document does not guess rights.

The originating D1 research repository has no root `LICENSE` and `pyproject.toml` has no `license` field. Analysis code in this package is therefore **unlicensed unless the authors add a grant**. Until they do, treat `analysis/`, `reproduce_main.py`, `examples/`, and the prompt templates as all-rights-reserved research artifacts: reproduction of the paper results is the intended use; further redistribution is the authors’ decision.

## D1-generated columns (this package)

| Component | What is released | Status |
|---|---|---|
| Item IDs, model aliases, endpoints | yes | D1 metadata |
| Natural \(q_1\), displayed scores, \(q_\delta\) | yes | D1 measurements |
| Routing actions / verify flags | yes | D1 measurements |
| Binary correctness / LCB pass-fail, `n_tests`, `n_passed` | yes | D1 evaluation outcomes, not the tests |
| Frozen MMLU *model letter* (A–J) | yes | model output letter only |
| Request keys | yes | pointers; no payload |
| Prompt templates with placeholders | yes | D1 wording; filled prompts omitted |
| Analysis / figure code | yes | see repo-license gap above |

## MMLU-Pro (TIGER-Lab / TIGER-AI-Lab)

| Fact | Source |
|---|---|
| Dataset pin in D1 | Hugging Face `TIGER-Lab/MMLU-Pro`, test split, revision `b189ec765aa7ed75c8acfea42df31fdae71f97be` |
| GitHub repo license | Apache-2.0 (`TIGER-AI-Lab/MMLU-Pro`) |
| Hugging Face card (current) | `license: mit` |

**Unresolved:** GitHub Apache-2.0 vs Hugging Face MIT metadata disagree. This package therefore **does not vendor** question stems, options, or official answer keys. Reconstruct from the official pin if you have determined that your use is allowed.

## LiveCodeBench

| Fact | Source |
|---|---|
| Dataset pin | `livecodebench/code_generation_lite` `release_v6` |
| Harness commit | `28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24` |
| Harness GitHub license | MIT (`LiveCodeBench/LiveCodeBench`) |
| Hugging Face card | `license: cc` (Creative Commons variant **not specified**) |
| Problem origin | LeetCode, AtCoder, Codeforces contest problems |

**Unresolved:** MIT covers the harness, not necessarily contest problem text or hidden tests. HF `license: cc` is incomplete. Contest-site terms were not reviewed in this repository.

**Not redistributed here:** problem statements, starter code, hidden/private tests, cached `test.jsonl` files, frozen generated programs.

Released LCB fields are IDs, difficulty/platform metadata is *not* included except via official IDs, plus our pass/fail counts.

## Model providers

| Provider / endpoint | Role |
|---|---|
| OpenAI `gpt-5.6-sol` | generator, \(q_1\), router |
| Anthropic `claude-sonnet-5` | generator, \(q_1\), router |

**Unresolved:** this repository contains no recorded review of OpenAI or Anthropic output-redistribution terms. Frozen generated **code is omitted**. Frozen MMLU letters are included as single-character model outputs needed to interpret routing rows; if provider terms forbid even that, drop the `frozen_output_letter` column.

## Python dependencies used to reproduce

`numpy` and `matplotlib` are third-party packages with their own licenses. Install them from PyPI; they are not vendored here.

## Explicitly not a license grant

Nothing in this file authorizes:

- republishing MMLU-Pro items;
- republishing LiveCodeBench problems or hidden tests;
- bypassing official benchmark channels;
- commercial reuse of model outputs;
- treating D1 as a hosted evaluation service.

If you need a rights-cleared full dump (stems + code + tests), obtain it from the official sources and join on the IDs in `data/manifests/id_lists.json`.
