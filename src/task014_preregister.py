"""Write Task 014 freeze artifacts before any new Stage-3 outcome inspection."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from .study1_smoke import _git_commit
from .task005_common import sha256_file
from .task014_common import (
    ANALYSIS_DIR,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    DELTAS,
    DISPLAY_FORMAT,
    ENDPOINTS,
    EXPERIMENT_VERSION,
    FAMILY,
    GPT_CLAUDE,
    LAMBDA_GRID,
    NEW_DELTAS,
    PAPER_DIRECTION,
    RETURN_DIR,
    ROUNDING,
    RUN_ID,
    SAMPLE_DIR,
    SCIENTIFIC_CAP,
    SQLITE_PATH,
    TASK009_DIR,
    TASK011_DIR,
    WORLD_A,
    WORLD_B,
    WORLD_C,
    WORLD_D,
    WORLD_E,
    json_dump,
    write_csv,
)
from .task014_data import FrozenItem, offset_prompt
from .task014_transform import render_q, transform_q


def write_freeze(items: list[FrozenItem], reuse_decision: str, audit_rows: list[dict]) -> None:
    RETURN_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    src = Path(__file__).resolve().parent
    transform_hash = sha256_file(src / "task014_transform.py")
    data_hash = sha256_file(src / "task014_data.py")
    cond_rows = []
    for item in items:
        for delta in DELTAS:
            value, token = transform_q(item.q1, delta), render_q(transform_q(item.q1, delta))
            cond_rows.append(
                {
                    "task": item.task,
                    "question_id": item.question_id,
                    "model_alias": item.model_alias,
                    "delta": delta,
                    "q1": item.q1,
                    "q_delta": value,
                    "rendered_q_delta": token,
                    "incorrect": item.incorrect,
                    "reuse_delta0": delta == 0.0 and reuse_decision == "REUSE_TRUE_Q_VISIBLE",
                    "prompt_sha256": __import__("hashlib").sha256(offset_prompt(item, delta).encode()).hexdigest(),
                }
            )
    write_csv(RETURN_DIR / "stage3_offset_conditions.csv", cond_rows)
    write_csv(SAMPLE_DIR / "stage3_offset_conditions.csv", cond_rows)

    endpoints_ok = ENDPOINTS == {
        "openai_gpt56_sol": "gpt-5.6-sol",
        "anthropic_sonnet5": "claude-sonnet-5",
    }
    if not endpoints_ok:
        raise RuntimeError("MODEL_VERSION_BLOCKER: Task 014 endpoints do not match 009/011")

    n_new = sum(1 for r in cond_rows if r["delta"] != 0.0)
    freeze = {
        "created_at": datetime.now(UTC).isoformat(),
        "code_commit": _git_commit(),
        "experiment_version": EXPERIMENT_VERSION,
        "run_id": RUN_ID,
        "family": FAMILY,
        "delta_grid": list(DELTAS),
        "new_deltas": list(NEW_DELTAS),
        "transform": "sigmoid(logit(q1)+delta); delta=0 returns exact q1; q1 in {0,1} stays at endpoint",
        "display_format": DISPLAY_FORMAT,
        "rounding": ROUNDING,
        "transform_code_sha256": transform_hash,
        "data_loader_sha256": data_hash,
        "endpoints": dict(ENDPOINTS),
        "delta0_reuse_decision": reuse_decision,
        "n_frozen_items": len(items),
        "n_planned_new_calls": n_new,
        "scientific_cap": SCIENTIFIC_CAP,
        "sqlite": str(SQLITE_PATH),
        "paper_sha256": sha256_file(PAPER_DIRECTION),
        "task009_stage3_sha256": sha256_file(TASK009_DIR / "stage3_results.csv"),
        "task011_stage3_sha256": sha256_file(TASK011_DIR / "stage3_results.csv"),
        "models": list(GPT_CLAUDE),
        "bootstrap": {"seed": BOOTSTRAP_SEED, "resamples": BOOTSTRAP_RESAMPLES},
        "lambda_grid": list(LAMBDA_GRID),
        "worlds": [WORLD_A, WORLD_B, WORLD_C, WORLD_D, WORLD_E],
        "audit_n_rows": len(audit_rows),
        "endpoint_q1_exact_0": sum(it.q1 == 0.0 for it in items),
        "endpoint_q1_exact_1": sum(it.q1 == 1.0 for it in items),
    }
    json_dump(RETURN_DIR / "freeze_manifest.json", freeze)
    json_dump(SAMPLE_DIR / "freeze_manifest.json", freeze)

    analysis = {
        "frozen_before_stage3_calls": True,
        "created_at": freeze["created_at"],
        "primary_endpoints": ["leakage(+1.5)-leakage(-1.5)", "coverage(+1.5)-coverage(-1.5)", "retention vs constant 0.70-0.99"],
        "bootstrap": freeze["bootstrap"],
        "lambda_grid": freeze["lambda_grid"],
        "world_A": {
            "name": WORLD_A,
            "gpt_mmlu": "leakage shift >= +5pp, CI lo > 0, |coverage shift| >= 15pp",
            "gpt_code": "leakage shift >= +10pp, CI lo > 0, |coverage shift| >= 15pp",
            "retention": "leakage retention >= 0.50 in both GPT tasks OR coverage retention >= 0.50 in both",
        },
        "world_B": WORLD_B,
        "world_C": WORLD_C,
        "world_D": WORLD_D,
        "world_E": WORLD_E,
        "malformed_policy": "one parse repair then fail the item; retry transient up to 3; do not invent actions",
        "no_delta_tuning": True,
        "no_repeats": True,
        "no_pilot": True,
    }
    json_dump(RETURN_DIR / "analysis_freeze.json", analysis)

    prereg = f"""# Task 014 preregistration / freeze

Timestamp: {freeze['created_at']}
Label: confirmatory for the rank-preserving offset; post-hoc relative to Tasks 009–013.

## Design (frozen before new Stage-3 calls)

- Transform: `q_delta = sigmoid(logit(q1) + delta)` with `delta ∈ {list(DELTAS)}`
- `delta = 0` returns the original q1 float (no logit round-trip)
- Exact q1 0 or 1 stays at that endpoint for every delta
- Display: `format(number, '.12g')` — same as Tasks 009/011
- Models/endpoints: {ENDPOINTS}
- δ=0 reuse: **{reuse_decision}**
- Planned new scientific calls: **{n_new}**
- SQLite: `{SQLITE_PATH}`
- Bootstrap: seed {BOOTSTRAP_SEED}, {BOOTSTRAP_RESAMPLES} item-clustered resamples
- Loss: Task 012 `leakage + λ * coverage` on the same λ grid

## Frozen item counts

- MMLU GPT/Claude: 500 each
- Code GPT/Claude: 286 confirmatory hard IDs each
- Exact q1 = 0: {freeze['endpoint_q1_exact_0']} (code only: 2 GPT, 4 Claude)
- Exact q1 = 1: {freeze['endpoint_q1_exact_1']}

## Primary endpoints

A. `LEAKAGE(+1.5) − LEAKAGE(−1.5)`
B. `COVERAGE(+1.5) − COVERAGE(−1.5)`
C. Retention vs constant-score 0.70→0.99 leakage and coverage

## World buckets (GPT primary)

- `{WORLD_A}`
- `{WORLD_B}`
- `{WORLD_C}`
- `{WORLD_D}`
- `{WORLD_E}`

## Prompt rule

Stage-3 prompts are byte-identical to the corresponding 009/011 q-visible prompt except the displayed number. The model is not told that q was transformed.

## Hashes

- transform.py: `{transform_hash}`
- paperDirection.txt: `{freeze['paper_sha256']}`
- 009 stage3: `{freeze['task009_stage3_sha256']}`
- 011 stage3: `{freeze['task011_stage3_sha256']}`

Do not inspect new VERIFY rates until the planned nonzero-delta cells complete or a technical stop is hit.
"""
    (RETURN_DIR / "preregistration.md").write_text(prereg, encoding="utf-8")
