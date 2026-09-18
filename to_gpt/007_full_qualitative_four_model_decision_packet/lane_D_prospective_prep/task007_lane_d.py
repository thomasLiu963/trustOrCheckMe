"""Task 007 Lane D: prospective four-model study prep. Zero API calls."""

from __future__ import annotations

from typing import Any

from .task006_lane_c import CANDIDATE_NS, CANDIDATE_SEEDS
from .task007_common import (
    CLAUDE_ENDPOINT,
    GEMINI_ENDPOINT,
    GPT_ENDPOINT,
    GROK_ENDPOINT,
    LANE_D_DIR,
    write_csv,
)

POOL_HASH = "53b11137658de739537a41032d92aada7f50742ef07e9615da10e070ab37c4fa"


def run_lane_d() -> dict[str, Any]:
    LANE_D_DIR.mkdir(parents=True, exist_ok=True)
    (LANE_D_DIR / "configs").mkdir(parents=True, exist_ok=True)
    pipeline = [
        "# Prospective four-model pipeline (prep only)",
        "",
        "Do not run this pipeline in Task 007. GPT must freeze N/conditions later.",
        "",
        "## Models",
        "",
        f"- GPT `{GPT_ENDPOINT}` reasoning.effort=none",
        f"- Claude `{CLAUDE_ENDPOINT}` thinking disabled",
        f"- Gemini `{GEMINI_ENDPOINT}` thinking_level=LOW (frozen V2 setting)",
        f"- Grok `{GROK_ENDPOINT}`",
        "",
        "## Unseen pool",
        "",
        f"- Task 006 eligible pool hash `{POOL_HASH}` (11,532 items after excluding 500 V2-B IDs)",
        "- Candidate stratified lists already exist for N=200/400/600/800/1000; none is the confirmatory sample",
        "",
        "## Stages after GPT review",
        "",
        "1. Fresh Stage-1 answers for all selected models on the same unseen questions",
        "2. Score correctness from benchmark labels",
        "3. Elicit q1",
        "4. Optionally elicit q2 only if retained as a baseline",
        "5. Hidden qualitative verification (Task-006 frozen wording if still the qualitative family)",
        "6. Selected manipulated displayed-score conditions",
        "7. Leave-one-model-out empirical difficulty for evaluation only",
        "8. Matched-budget routing with deployment-fair baselines",
        "",
        "Cross-model difficulty is an evaluation-side proxy, not a production router feature unless separately justified.",
        "",
        "## Isolation",
        "",
        "- New sqlite, not V2 / Study 1 / q2 / qualitative 006/007 databases",
        "- Do not reuse Task-007 100-question IDs as the confirmatory sample",
        "",
    ]
    (LANE_D_DIR / "four_model_pipeline.md").write_text("\n".join(pipeline) + "\n", encoding="utf-8")
    rows = []
    designs = {
        "D1_lean": {
            "stakes": 1,
            "stage3_conditions": 4,
            "note": "one stakes family; hidden + 0.70 + 0.90 + 0.99",
        },
        "D2_visibility_bridge": {
            "stakes": 1,
            "stage3_conditions": 5,
            "note": "one stakes family; hidden + true-q visible + 0.70 + 0.90 + 0.99",
        },
        "D3_stakes_replication": {
            "stakes": 2,
            "stage3_conditions": 4,
            "note": "moderate+stronger; hidden + 0.70 + 0.90 + 0.99",
        },
    }
    n_models = 4
    for n in (200, 400, 600):
        for name, spec in designs.items():
            stage3 = n * n_models * spec["stakes"] * spec["stage3_conditions"]
            stage1 = n * n_models
            q1 = n * n_models
            q2 = n * n_models
            rows.append(
                {
                    "design": name,
                    "n_questions": n,
                    "n_models": n_models,
                    "stakes_families": spec["stakes"],
                    "stage3_conditions_per_stakes": spec["stage3_conditions"],
                    "stage1_cells": stage1,
                    "q1_cells": q1,
                    "optional_q2_cells": q2,
                    "stage3_cells": stage3,
                    "stage1_plus_q1_plus_stage3": stage1 + q1 + stage3,
                    "with_optional_q2": stage1 + q1 + q2 + stage3,
                    "note": spec["note"],
                    "sample_not_frozen": True,
                    "candidate_seed_if_using_task006_list": CANDIDATE_SEEDS.get(n, ""),
                }
            )
    write_csv(LANE_D_DIR / "design_cost_matrix.csv", rows)
    routing = [
        "# Deployment-oriented routing analysis plan",
        "",
        "Keep the multi-cell historical hidden aggregate out of the fair primary comparison.",
        "",
        "## Fair primary baselines (to implement later)",
        "",
        "1. raw q1 (lowest confidence first; Checkpoint A fractional ties)",
        "2. cross-fitted calibrated q1 (grouped by question)",
        "3. q1+q2 if q2 is retained",
        "4. one hidden qualitative verification judgment (predeclare owner/L or qualitative-hidden cell)",
        "5. simple combined q1 + one hidden judgment (learned only with grouped CV)",
        "",
        "## Oracle / upper-bound diagnostic only",
        "",
        "- Multi-elicitation hidden-VERIFY fraction across several L/owner cells",
        "",
        "## Metrics",
        "",
        "- Error catch at 10/20/30/40/50% verification budgets",
        "- Question-bootstrap CIs",
        "- Leave-one-model-out difficulty as an evaluation covariate, not as a claimed production feature",
        "",
        "Do not freeze a winner from Task 005/006/007 historical tables.",
        "",
    ]
    (LANE_D_DIR / "routing_analysis_plan.md").write_text("\n".join(routing) + "\n", encoding="utf-8")
    (LANE_D_DIR / "configs" / "README.md").write_text(
        "Scaffolding only. No confirmatory sample, prompt freeze, or paid calls.\n",
        encoding="utf-8",
    )
    _ = CANDIDATE_NS
    return {"ok": True, "n_design_rows": len(rows), "pool_hash": POOL_HASH, "api_calls": 0}
