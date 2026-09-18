"""Task 006 Lane C: fresh unseen MMLU-Pro pool and candidate sampling plans.

Zero API calls. No target-model outcomes.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT, load_v2_experiment_config
from .datasets import load_mmlu_pro_source, select_category_stratified
from .study1_sample import hash_id_list, load_v2b_examples
from .task006_common import LANE_C_DIR, prompt_sha256, sha256_file


CANDIDATE_NS = (200, 400, 600, 800, 1000)
CANDIDATE_SEEDS = {
    200: 20260918,
    400: 20260919,
    600: 20260920,
    800: 20260921,
    1000: 20260922,
}


def run_lane_c() -> dict[str, Any]:
    LANE_C_DIR.mkdir(parents=True, exist_ok=True)
    config = load_v2_experiment_config()
    source = load_mmlu_pro_source(config)
    v2b = load_v2b_examples()
    exclude = {row.example_id for row in v2b}
    eligible = [row for row in source if row.example_id not in exclude]
    if not eligible:
        raise RuntimeError("eligible unseen pool is empty")
    eligible_ids = [row.example_id for row in eligible]
    pool_hash = hash_id_list(eligible_ids)
    revision = eligible[0].dataset_revision
    payload = {
        "dataset": "TIGER-Lab/MMLU-Pro",
        "split": "test",
        "revision": revision,
        "n_source": len(source),
        "n_excluded_v2b": len(exclude),
        "n_eligible": len(eligible),
        "eligible_ids": eligible_ids,
        "pool_id_list_sha256": pool_hash,
        "exclusion_rule": "all 500 historical V2-B IDs; no target-model behavior filter",
        "candidate_plans_are_not_final": True,
    }
    (LANE_C_DIR / "eligible_pool.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    meta_rows = [
        {
            "example_id": row.example_id,
            "category": row.category,
            "n_choices": len(row.choices),
            "question_char_len": len(row.question),
            "source_index": row.source_index,
            "dataset_revision": row.dataset_revision,
            "correct_label": row.correct_label,
        }
        for row in eligible
    ]
    from .study1_analysis import _write_csv

    _write_csv(LANE_C_DIR / "eligible_pool_metadata.csv", meta_rows)
    (LANE_C_DIR / "pool_hash.txt").write_text(pool_hash + "\n", encoding="utf-8")
    hash_lines = [f"eligible_pool {pool_hash} n={len(eligible)} revision={revision}"]
    category_counts = Counter(row.category for row in eligible)
    for n in CANDIDATE_NS:
        seed = CANDIDATE_SEEDS[n]
        selected, quotas = select_category_stratified(eligible, n, seed)
        ids = [row.example_id for row in selected]
        digest = hash_id_list(ids)
        plan = {
            "n": n,
            "seed": seed,
            "stratify_by": "category",
            "sampler": "src.datasets.select_category_stratified",
            "status": "candidate_plan_only",
            "ordered_ids": ids,
            "id_list_sha256": digest,
            "quotas": quotas,
            "category_counts": dict(Counter(row.category for row in selected)),
        }
        (LANE_C_DIR / f"candidate_N{n}.json").write_text(
            json.dumps(plan, indent=2) + "\n", encoding="utf-8"
        )
        hash_lines.append(f"N{n} seed={seed} sha256={digest}")
    (LANE_C_DIR / "candidate_hashes.txt").write_text("\n".join(hash_lines) + "\n", encoding="utf-8")
    infra = """# Confirmation infrastructure (not executed)

This is scaffolding for a later numbered confirmatory task. Task 006 does not freeze the confirmatory design and makes 0 confirmatory calls.

## Reusable pieces already in the repo

- Prompt freezing: `src/study1_prompts.py`, `src/task006_prompts.py`
- Model roster freeze: `config/models.yaml` + `create_adapter` ID locks
- Call cap / CheckpointStore: `src/checkpointing.py`, Study 1 / q2 / qualitative runners
- Parser: `parse_study1_verification_response` / qualitative wrapper
- Per-question paired design: one row per question × condition; GroupKFold by question ID
- Stage-1 generation (if a future protocol requires fresh answers): `src/prompts.py` Stage-1 template + `stage='answer'`
- Stage-2 q1 and optional q2: historical Stage-2 template in `src/prompts.py`; q2 runner pattern in `src/task005_q2.py`
- Verification decision: numeric Study-1 or qualitative Task-006 families
- Matched-budget routing: `src/task005_lane_a.py` fractional weights + `src/task006_lane_b.py`

## What a later confirmatory task still must freeze

1. Final N and whether to use one of the candidate lists in this folder
2. Numeric vs qualitative stakes (depends on Lane A gate)
3. Model roster (GPT/Claude only vs Gemini/Grok after Lane D)
4. Whether Stage-1 answers are reused or regenerated on unseen items
5. Scientific call cap
6. Confirmatory vs exploratory label, dated before any target-model calls on the fresh pool

Do not treat the candidate_N*.json files as the confirmatory sample.
"""
    (LANE_C_DIR / "confirmation_infrastructure.md").write_text(infra, encoding="utf-8")
    _ = PROJECT_ROOT, prompt_sha256, sha256_file
    return {
        "n_source": len(source),
        "n_eligible": len(eligible),
        "pool_hash": pool_hash,
        "revision": revision,
        "category_counts": dict(category_counts),
        "candidate_hashes": hash_lines,
    }
