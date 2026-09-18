"""Task 006 Lane D: Gemini/Grok/open-model feasibility audit. Zero API calls."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Any

from .config import DEFAULT_MODELS_CONFIG, PROJECT_ROOT, load_models_config
from .model_adapters import GoogleAdapter, XAIAdapter, create_adapter
from .study1_analysis import _write_csv
from .task005_common import V2_SQLITE
from .task005_lane_a import load_v2_answers, load_v2_confidence
from .task006_common import LANE_D_DIR, SCIENTIFIC_CAP


def _source_mentions_local_inference() -> dict[str, bool]:
    adapters = (PROJECT_ROOT / "src" / "model_adapters.py").read_text(encoding="utf-8")
    needles = ("vllm", "ollama", "huggingface", "transformers", "llama.cpp", "local_model")
    return {needle: needle.lower() in adapters.lower() for needle in needles}


def run_lane_d() -> dict[str, Any]:
    LANE_D_DIR.mkdir(parents=True, exist_ok=True)
    models = load_models_config(DEFAULT_MODELS_CONFIG)
    answers = load_v2_answers()
    conf = load_v2_confidence()
    closed = []
    for alias, label, endpoint in (
        ("google_gemini38_flash", "Gemini", "gemini-3.8-flash"),
        ("xai_grok420_nonreasoning", "Grok", "grok-4.20-0309-non-reasoning"),
    ):
        spec = models.models[alias]
        n_ans = sum(1 for (qid, model) in answers if model == alias)
        n_conf = sum(1 for (qid, model) in conf if model == alias)
        adapter_cls = GoogleAdapter if spec.provider == "google" else XAIAdapter
        status = "READY"
        notes = []
        if spec.api_model != endpoint:
            status = "BLOCKED"
            notes.append(f"configured api_model {spec.api_model} != expected {endpoint}")
        if n_ans < 500 or n_conf < 500:
            status = "NEEDS_MINOR_ENGINEERING"
            notes.append(f"historical Stage-1/2 rows answer={n_ans} confidence={n_conf}")
        if spec.provider == "google" and spec.thinking_level not in {None, "low"}:
            notes.append(f"thinking_level={spec.thinking_level} (V2 used low; not Study-1 none)")
        notes.append("create_adapter already accepts this frozen ID")
        notes.append("Study 1 / q2 / qualitative grids were never run for this model")
        later_qual_calls = SCIENTIFIC_CAP  # same 20q × 2 families × 4 scores if both GPT-role models
        closed.append(
            {
                "model_alias": alias,
                "label": label,
                "provider": spec.provider,
                "api_model": spec.api_model,
                "api_style": spec.api_style,
                "adapter_class": adapter_cls.__name__,
                "adapter_exists": True,
                "json_schema_stages": "answer,confidence,trust,verification via output_schema",
                "historical_v2_stage1_rows": n_ans,
                "historical_v2_stage2_rows": n_conf,
                "study1_causal_rows": 0,
                "q2_rows": 0,
                "endpoint_drift_risk": (
                    "IDs are frozen in create_adapter; a provider silent version shift is still possible"
                ),
                "estimated_calls_later_qual_20q_grid_one_model": 160,
                "estimated_calls_later_qual_both_closed_models": 320,
                "status": status,
                "notes": "; ".join(notes),
            }
        )
    _write_csv(LANE_D_DIR / "closed_model_matrix.csv", closed)
    local_hits = _source_mentions_local_inference()
    open_md = f"""# Open-model requirements (planning only)

No open-weight inference hook exists in `src/model_adapters.py`. `create_adapter` supports openai, anthropic, google, and xai only.

Scanner hits in the adapter file (should be false): `{local_hits}`

This lane did **not** install models or download weights.

## What would be required later

1. **One instruction-tuned open model**
   - New provider in `create_adapter` (vLLM, Hugging Face, or similar)
   - Frozen `api_model` string and tokenizer
   - Same Stage-1/2/3 JSON schema (`USE_UNVERIFIED` / `VERIFY_FIRST`, probability in [0,1])
   - Deterministic request keys and CheckpointStore compatibility
   - Cost/runtime accounting even if local GPU is "free"

2. **Log probability access**
   - Not available from the current GPT/Claude/Gemini/Grok adapters in this repo
   - An open model would need an explicit logprob API (token-level `logprob` of the frozen answer, or sequence logprob)
   - That is a new scientific construct, not a drop-in replacement for verbal q1

3. **Hidden activation access**
   - Not implemented. Would require a white-box runtime (hooks on residual streams), a stored activation policy, and a protocol addendum
   - Task 006 forbids jumping to interpretability

4. **Same frozen-answer / confidence / verification structure**
   - Reuse `src/prompts.py` and `src/task006_prompts.py`
   - Do not mix numeric Study-1 L/C prompts with qualitative families in one unfrozen comparison

Status: **BLOCKED** for open-weight execution in the current codebase (missing adapter). Engineering, not a scientific stop on GPT/Claude work.
"""
    (LANE_D_DIR / "open_model_requirements.md").write_text(open_md, encoding="utf-8")
    audit = [
        "# Model-expansion feasibility audit",
        "",
        "Zero provider calls. Code and historical V2 sqlite inspected only.",
        "",
        "## Closed models",
        "",
    ]
    for row in closed:
        audit += [
            f"### {row['label']} (`{row['model_alias']}`)",
            "",
            f"- Status: **{row['status']}**",
            f"- Endpoint: `{row['api_model']}` via `{row['api_style']}`",
            f"- Adapter: `{row['adapter_class']}` exists",
            f"- Historical V2 Stage-1/2 rows: {row['historical_v2_stage1_rows']} / {row['historical_v2_stage2_rows']}",
            f"- Study 1 / q2 replication: none",
            f"- Later 20-question qualitative grid if authorized: {row['estimated_calls_later_qual_20q_grid_one_model']} scientific calls",
            f"- Notes: {row['notes']}",
            "",
        ]
    audit += [
        "## Open models",
        "",
        "No local/open-weight inference adapter. See `open_model_requirements.md`. Status: **BLOCKED** (missing hook).",
        "",
        "Do not run Gemini, Grok, or open models in Task 006.",
        "",
    ]
    (LANE_D_DIR / "model_expansion_audit.md").write_text("\n".join(audit), encoding="utf-8")
    _ = inspect, ast, V2_SQLITE, create_adapter
    return {"closed": closed, "open_status": "BLOCKED"}
