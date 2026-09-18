"""Task 009 sample freeze and preregistration. Zero scientific API calls."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import load_models_config, load_v2_experiment_config
from .datasets import load_mmlu_pro_source, select_category_stratified
from .study1_sample import hash_id_list, load_v2b_examples
from .study1_smoke import _git_commit
from .task006_lane_a import family_template_hash
from .task006_prompts import PROMPT_FAMILY_MODERATE, build_qualitative_prompt, prompt_family_id
from .task009_common import (
    ADJACENT_PAIRS,
    ALL_MODELS,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    CONCURRENCY,
    CV_FOLDS,
    ENDPOINTS,
    EXPERIMENT_VERSION,
    FAILURE_INCONCLUSIVE_FRAC,
    FAMILY,
    FIGURES_DIR,
    FROM_GPT,
    GEMINI_GROK,
    GPT_CLAUDE,
    H1_FLOORS,
    H1_HIGH,
    H1_LOW,
    MATERIAL_AUROC,
    MATERIAL_LOGLOSS,
    MATCHED_BUDGETS,
    MAX_PARSE_REPAIRS,
    MAX_TRANSIENT_RETRIES,
    N_REPEAT_EXTRAS,
    N_REPEAT_GENERATIONS,
    PAPER_DIRECTION,
    PILOT_OR_CONFIRMATORY,
    PLANNED_PRIMARY_STAGE12,
    PLANNED_PRIMARY_STAGE3,
    PLANNED_PRIMARY_TOTAL,
    PLANNED_REPEAT_EXTRAS,
    PLANNED_SECONDARY_STAGE12,
    PLANNED_SECONDARY_STAGE3,
    PLANNED_SECONDARY_TOTAL,
    POOL_HASH,
    PRIMARY_N,
    PRIMARY_SEED,
    PROMPT_FAMILY,
    PROMPT_HASH_MODERATE,
    PROMPT_VERSION,
    PROVIDER_ATTEMPT_CAP,
    REPEAT_N,
    REPEAT_SEED,
    RETRY_FAILED_PASSES,
    RETURN_DIR,
    RUN_ID,
    SATURATION_HIGH,
    SATURATION_LABEL,
    SATURATION_LOW,
    SCIENTIFIC_CAP,
    SCORE_CONDITIONS,
    SECONDARY_N,
    SECONDARY_SEED,
    SQLITE_PATH,
    TASK_ID,
    VISIBLE_FIXED,
    VISIBLE_FIXED_CONDITIONS,
    assert_009_write_target,
    freeze_paths,
    frozen,
    json_dump,
    load_json,
    prompt_sha256,
    protected_fingerprints,
    write_csv,
)


def _example_row(example: Any, *, subset: str) -> dict[str, Any]:
    return {
        "example_id": example.example_id,
        "subset": subset,
        "category": example.category,
        "n_choices": len(example.choices),
        "question_char_len": len(example.question),
        "source_index": example.source_index,
        "dataset_name": example.dataset_name,
        "split": example.split,
        "dataset_revision": example.dataset_revision,
        "correct_label": example.correct_label,
    }


def _write_jsonl(path: Path, examples: list[Any]) -> None:
    assert_009_write_target(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(example.model_dump_json() + "\n" for example in examples),
        encoding="utf-8",
    )


def freeze_sample(*, force: bool = False) -> dict[str, Any]:
    paths = freeze_paths()
    if frozen() and not force:
        return load_json(paths["freeze_manifest"])

    fingerprints_before = protected_fingerprints()
    config = load_v2_experiment_config()
    models = load_models_config()
    source = load_mmlu_pro_source(config)
    v2b = load_v2b_examples()
    exclude = {row.example_id for row in v2b}
    eligible = [row for row in source if row.example_id not in exclude]
    eligible_ids = [row.example_id for row in eligible]
    pool_hash = hash_id_list(eligible_ids)
    if pool_hash != POOL_HASH:
        raise RuntimeError(
            f"unseen pool hash {pool_hash} != frozen Task-006/007 hash {POOL_HASH}"
        )
    if len(eligible) != 11532:
        raise RuntimeError(f"expected 11532 eligible items, got {len(eligible)}")
    if exclude & set(eligible_ids):
        raise RuntimeError("eligible pool overlaps historical V2-B IDs")

    primary, primary_quotas = select_category_stratified(eligible, PRIMARY_N, PRIMARY_SEED)
    secondary, secondary_quotas = select_category_stratified(
        primary, SECONDARY_N, SECONDARY_SEED
    )
    repeats, repeat_quotas = select_category_stratified(primary, REPEAT_N, REPEAT_SEED)
    primary_ids = [row.example_id for row in primary]
    secondary_ids = [row.example_id for row in secondary]
    repeat_ids = [row.example_id for row in repeats]
    if set(secondary_ids) - set(primary_ids):
        raise RuntimeError("secondary 200 is not a subset of primary 500")
    if set(repeat_ids) - set(primary_ids):
        raise RuntimeError("repeat 100 is not a subset of primary 500")

    observed_endpoints = {alias: models.models[alias].api_model for alias in ALL_MODELS}
    if observed_endpoints != dict(ENDPOINTS):
        raise RuntimeError(
            f"models.yaml endpoints {observed_endpoints} != frozen {dict(ENDPOINTS)}"
        )
    template_hash = family_template_hash(FAMILY)
    if template_hash != PROMPT_HASH_MODERATE:
        raise RuntimeError(
            f"moderate template hash {template_hash} != {PROMPT_HASH_MODERATE}"
        )
    if prompt_family_id(FAMILY) != PROMPT_FAMILY_MODERATE:
        raise RuntimeError("moderate family id drifted")

    hidden_example = build_qualitative_prompt(
        question="{question}",
        choices={"A": "{choice_a}", "B": "{choice_b}"},
        frozen_answer="A",
        family=FAMILY,
        displayed_confidence=None,
    )
    displayed_example = build_qualitative_prompt(
        question="{question}",
        choices={"A": "{choice_a}", "B": "{choice_b}"},
        frozen_answer="A",
        family=FAMILY,
        displayed_confidence=0.85,
    )
    created_at = datetime.now(UTC).isoformat()
    primary_hash = hash_id_list(primary_ids)
    secondary_hash = hash_id_list(secondary_ids)
    repeat_hash = hash_id_list(repeat_ids)
    revision = primary[0].dataset_revision

    manifest: dict[str, Any] = {
        "task_id": TASK_ID,
        "created_at": created_at,
        "code_commit": _git_commit(),
        "pilot_or_confirmatory": PILOT_OR_CONFIRMATORY,
        "experiment_version": EXPERIMENT_VERSION,
        "run_id": RUN_ID,
        "dataset": {
            "name": "TIGER-Lab/MMLU-Pro",
            "split": "test",
            "revision": revision,
            "n_source": len(source),
            "n_excluded_v2b": len(exclude),
            "n_eligible": len(eligible),
            "eligible_pool_sha256": pool_hash,
            "sampler": "src.datasets.select_category_stratified",
            "stratify_by": "category",
        },
        "sample": {
            "primary_n": PRIMARY_N,
            "primary_seed": PRIMARY_SEED,
            "primary_id_list_sha256": primary_hash,
            "primary_ids": primary_ids,
            "primary_category_quotas": primary_quotas,
            "primary_category_counts": dict(Counter(row.category for row in primary)),
            "secondary_n": SECONDARY_N,
            "secondary_seed": SECONDARY_SEED,
            "secondary_id_list_sha256": secondary_hash,
            "secondary_ids": secondary_ids,
            "secondary_category_quotas": secondary_quotas,
            "secondary_note": (
                "Stratified from the frozen primary 500 with seed 20260918. "
                "This is NOT the Task-006 candidate_N200.json list."
            ),
            "repeat_n": REPEAT_N,
            "repeat_seed": REPEAT_SEED,
            "repeat_id_list_sha256": repeat_hash,
            "repeat_ids": repeat_ids,
            "repeat_category_quotas": repeat_quotas,
            "n_repeat_generations": N_REPEAT_GENERATIONS,
            "n_repeat_extras": N_REPEAT_EXTRAS,
        },
        "models": {
            alias: {
                "api_model": models.models[alias].api_model,
                "provider": models.models[alias].provider,
                "api_style": models.models[alias].api_style,
                "max_output_tokens": models.models[alias].max_output_tokens,
                "reasoning_effort": models.models[alias].reasoning_effort,
                "thinking": (
                    models.models[alias].thinking.model_dump()
                    if getattr(models.models[alias], "thinking", None) is not None
                    else None
                ),
                "thinking_level": getattr(models.models[alias], "thinking_level", None),
            }
            for alias in ALL_MODELS
        },
        "endpoints": dict(ENDPOINTS),
        "primary_models": list(GPT_CLAUDE),
        "secondary_models": list(GEMINI_GROK),
        "prompts": {
            "stakes_family": FAMILY,
            "prompt_version": PROMPT_VERSION,
            "prompt_family": PROMPT_FAMILY,
            "template_hash_moderate": template_hash,
            "hidden_skeleton_sha256": prompt_sha256(hidden_example),
            "displayed_0.85_skeleton_sha256": prompt_sha256(displayed_example),
            "stage1_prompt_version": "stage_1_answer_v2_structured",
            "q1_prompt_version": "stage_2_confidence_v3_structured_compat",
            "no_q2": True,
            "no_stronger_family": True,
            "no_numerical_LC": True,
        },
        "conditions": list(SCORE_CONDITIONS),
        "visible_fixed_scores": list(VISIBLE_FIXED),
        "visible_fixed_conditions": list(VISIBLE_FIXED_CONDITIONS),
        "adjacent_pairs": [list(pair) for pair in ADJACENT_PAIRS],
        "h1": {
            "metric": "VERIFY_rate(0.70)-VERIFY_rate(0.99)",
            "low": H1_LOW,
            "high": H1_HIGH,
            "floors": dict(H1_FLOORS),
            "unit": "question_id",
            "ci": "question-bootstrap 95%",
        },
        "h2": {
            "shared": (
                "condition intercept + transferable shared item covariate "
                "(leave-one-target-out other-primary Stage-1 correctness)"
            ),
            "richer": "shared + score x difficulty interaction",
            "primary_metric": "held-out log-loss improvement of richer vs shared",
            "invariance_if": (
                "richer improves GroupKFold held-out log-loss by < 0.01 AND "
                "question-bootstrap 95% upper bound of that improvement is also < 0.01"
            ),
            "material_logloss": MATERIAL_LOGLOSS,
            "cv_folds": CV_FOLDS,
            "bootstrap_of": (
                "question-resampled mean OOF log-loss difference from the frozen "
                "5-fold predictions"
            ),
        },
        "h3": {
            "repeat_subset_n": REPEAT_N,
            "generations": N_REPEAT_GENERATIONS,
            "pairs": [list(pair) for pair in ADJACENT_PAIRS],
            "metrics": [
                "spearman of 3-generation mean VERIFY propensity",
                "pairwise ordering agreement",
                "reversals vs same-prompt rerun disagreement",
            ],
        },
        "h4": {
            "repeat_auc": "AUROC of repeat-averaged VERIFY propensity for target wrongness",
            "full_n_secondary": "binary-action AUROC on N=500 as coarse evidence",
            "material_auroc": MATERIAL_AUROC,
            "exclude_saturated": True,
        },
        "saturation": {
            "low": SATURATION_LOW,
            "high": SATURATION_HIGH,
            "label": SATURATION_LABEL,
        },
        "retry_policy": {
            "max_transient_retries": MAX_TRANSIENT_RETRIES,
            "max_parse_repairs": MAX_PARSE_REPAIRS,
            "retry_failed_passes": RETRY_FAILED_PASSES,
            "provider_attempt_cap": PROVIDER_ATTEMPT_CAP,
            "scientific_cap": SCIENTIFIC_CAP,
            "inconclusive_if_primary_failure_frac_gt": FAILURE_INCONCLUSIVE_FRAC,
            "exclusion": (
                "After retries, drop a question from paired 7-condition analyses if any "
                "required cell is missing. Report failures separately. Do not impute actions."
            ),
        },
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "n_resamples": BOOTSTRAP_RESAMPLES,
            "unit": "question_id",
        },
        "matched_budgets": list(MATCHED_BUDGETS),
        "concurrency": CONCURRENCY,
        "planned_calls": {
            "primary_stage12": PLANNED_PRIMARY_STAGE12,
            "primary_stage3": PLANNED_PRIMARY_STAGE3,
            "repeat_extras": PLANNED_REPEAT_EXTRAS,
            "primary_total": PLANNED_PRIMARY_TOTAL,
            "secondary_stage12": PLANNED_SECONDARY_STAGE12,
            "secondary_stage3": PLANNED_SECONDARY_STAGE3,
            "secondary_total": PLANNED_SECONDARY_TOTAL,
            "grand_scientific": SCIENTIFIC_CAP,
        },
        "sqlite": str(SQLITE_PATH),
        "from_gpt": str(FROM_GPT),
        "decision_buckets": [
            "PROSPECTIVE_INVARIANCE_SUPPORTED",
            "PROSPECTIVE_RESHAPING_SUPPORTED",
            "MIXED_BY_MODEL",
            "INCONCLUSIVE",
        ],
        "do_not_inspect_verify_rates_until_primary_complete": True,
        "fingerprints_at_freeze": fingerprints_before,
    }

    RETURN_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    csv_writes = (
        (paths["sample_500"], primary, "primary_500"),
        (paths["secondary_200"], secondary, "secondary_200"),
        (paths["repeat_100"], repeats, "repeat_100"),
        (paths["sample_500_local"], primary, "primary_500"),
        (paths["secondary_200_local"], secondary, "secondary_200"),
        (paths["repeat_100_local"], repeats, "repeat_100"),
    )
    for path, examples, subset in csv_writes:
        assert_009_write_target(path)
        write_csv(path, [_example_row(row, subset=subset) for row in examples])
    _write_jsonl(paths["examples_500"], primary)
    _write_jsonl(paths["examples_200"], secondary)
    _write_jsonl(paths["examples_100"], repeats)
    assert_009_write_target(paths["freeze_manifest"])
    json_dump(paths["freeze_manifest"], manifest)
    json_dump(paths["freeze_local"], manifest)
    assert_009_write_target(paths["preregistration"])
    paths["preregistration"].write_text(_preregistration_markdown(manifest), encoding="utf-8")

    after = protected_fingerprints()
    if after["paperDirection"]["sha256"] != fingerprints_before["paperDirection"]["sha256"]:
        raise RuntimeError("paperDirection.txt changed during freeze")
    for name in ("v2", "study1", "q2", "qual006", "qual007a", "qual007b"):
        if fingerprints_before[name].get("sha256") != after[name].get("sha256"):
            raise RuntimeError(f"{name} changed during Task 009 freeze")
    return manifest


def _preregistration_markdown(manifest: Mapping[str, Any]) -> str:
    sample = manifest["sample"]
    planned = manifest["planned_calls"]
    endpoints = manifest["endpoints"]
    return f"""# Task 009 preregistration — prospective ranking-invariance confirmation

Frozen at `{manifest["created_at"]}` before scientific VERIFY-rate inspection.
Commit: `{manifest.get("code_commit") or "unknown"}`.
Experiment version: `{manifest["experiment_version"]}`.
Label: **confirmatory**.

This file must not be rewritten after freeze except via a timestamped note in `deviations.md`.

## Question

On the exact same frozen answer, when displayed confidence is counterfactually changed, does the model mostly change **how many** answers it sends for independent verification, or **which** answers it prioritizes?

## Sample

- Dataset: TIGER-Lab/MMLU-Pro test, revision `{manifest["dataset"]["revision"]}`.
- Eligible unseen pool SHA256: `{manifest["dataset"]["eligible_pool_sha256"]}` (n={manifest["dataset"]["n_eligible"]}).
- Historical V2-B 500 IDs are excluded. Tasks 003–008 development items are inside that 500.
- Primary N=500, seed `{sample["primary_seed"]}`, SHA256 `{sample["primary_id_list_sha256"]}`.
- Secondary N=200 Gemini/Grok subset of those 500, seed `{sample["secondary_seed"]}`, SHA256 `{sample["secondary_id_list_sha256"]}`.
  This is **not** `candidate_N200.json`.
- Repeat N=100 GPT/Claude subset of the 500, seed `{sample["repeat_seed"]}`, SHA256 `{sample["repeat_id_list_sha256"]}`.
- Sampler: category-stratified `select_category_stratified`.
- Same questions across models within each roster.

## Models

Primary:
- GPT `{endpoints["openai_gpt56_sol"]}`
- Claude `{endpoints["anthropic_sonnet5"]}`

Secondary:
- Gemini `{endpoints["google_gemini38_flash"]}`
- Grok `{endpoints["xai_grok420_nonreasoning"]}`

If an endpoint is unavailable: mark BLOCKED. Do not substitute.

## Protocol

- Fresh Stage-1 answers, then separate q1 confidence. No q2.
- Stage-3: Task-006/007 **moderate** qualitative wording only.
- Prompt version `{PROMPT_VERSION}`, family `{PROMPT_FAMILY}`, template hash `{PROMPT_HASH_MODERATE}`.
- Conditions: {", ".join(SCORE_CONDITIONS)}.
- hidden = no confidence sentence; true_q_visible = that item's fresh q1; displayed_* = exact fixed score, formatted with `.12g`.
- Actions: USE_UNVERIFIED or VERIFY_FIRST only.
- Repeats: GPT/Claude, N=100, 3 total Stage-3 generations per condition; original primary cell is generation 0; two extras. Do not repeat Stage-1/q1.

## Call budget

- GPT+Claude Stage1+q1: {planned["primary_stage12"]}
- GPT+Claude Stage3: {planned["primary_stage3"]}
- Repeat extras: {planned["repeat_extras"]}
- Primary total: {planned["primary_total"]}
- Gemini+Grok Stage1+q1: {planned["secondary_stage12"]}
- Gemini+Grok Stage3: {planned["secondary_stage3"]}
- Secondary total: {planned["secondary_total"]}
- Grand scientific cap: {planned["grand_scientific"]}
- Provider attempt cap: {PROVIDER_ATTEMPT_CAP} (retries counted separately)
- Concurrency: {CONCURRENCY}

## Hypotheses and criteria

### H1 coverage control

Primary effect: VERIFY_rate(0.70) − VERIFY_rate(0.99), paired by question, question-bootstrap 95% CI.
Substantive floors (large-effect markers, not publication gates): GPT ≥ 20pp, Claude ≥ 15pp.

### H2 shared-ranking invariance

Shared transferable model: condition intercepts + leave-one-target-out other-primary Stage-1 correctness.
Richer model: shared + displayed-score × difficulty interaction.
Primary metric: 5-fold GroupKFold held-out log-loss improvement of richer vs shared.
Invariance if improvement < 0.01 **and** question-bootstrap 95% upper bound of that improvement is also < 0.01.
Evaluate GPT and Claude separately.
If fits are unstable, report INCONCLUSIVE rather than changing the threshold.
Item intercepts are in-sample descriptive only; they do not transfer to held-out questions.

### H3 repeat rank stability

On N=100, 3-generation mean VERIFY propensity for adjacent visible pairs (.70/.85, .85/.90, .90/.95, .95/.99):
Spearman, bootstrap CI, pairwise ordering agreement, reversals vs same-prompt rerun noise.

### H4 discrimination stability

AUROC of repeat-averaged VERIFY propensity for target wrongness, with pairwise ΔAUROC.
Full-N binary-action AUROC is coarse secondary evidence.
ΔAUROC is evaluated only on non-saturated cells.

## Saturation

A condition-cell is saturated if VERIFY rate ≤ {SATURATION_LOW} or ≥ {SATURATION_HIGH}.
Flag: `{SATURATION_LABEL}`. Saturated cells are excluded from ranking/discrimination identity claims.

## Retry / exclusion

- Transient transport retries: {MAX_TRANSIENT_RETRIES}
- Parse repairs: {MAX_PARSE_REPAIRS}
- One additional retry_failed pass after the first paid pass
- After that, missing cells are failures, not imputed
- If > {FAILURE_INCONCLUSIVE_FRAC:.0%} of primary GPT/Claude planned scientific cells fail: INCONCLUSIVE

## Bootstrap

Seed {BOOTSTRAP_SEED}, {BOOTSTRAP_RESAMPLES} resamples, unit = question ID.

## Decision buckets (frozen)

- `PROSPECTIVE_INVARIANCE_SUPPORTED` if coverage response is large, shared-ranking equivalence-style test passes, repeat rank stability is strong, discrimination is stable on non-saturated conditions, and the richer model adds no material held-out value.
- `PROSPECTIVE_RESHAPING_SUPPORTED` if coverage changes and the richer model materially/reproducibly beats shared ranking, repeats show score-dependent reordering beyond rerun noise, and discrimination changes materially at non-saturated points.
- `MIXED_BY_MODEL` if GPT and Claude genuinely diverge under the same tests.
- `INCONCLUSIVE` if saturation, low error counts, or uncertainty prevent distinction.

Do not redefine these after seeing results.

## Secondary Gemini/Grok

Same Stage-1/q1 + 7 Stage-3 conditions on the frozen N=200. No repeats. Compatibility report only. Secondary oddities do not redefine the GPT/Claude hypothesis.

## Isolation

- New sqlite: `{SQLITE_PATH}`
- Outputs: `{RETURN_DIR}`
- Do not modify `{PAPER_DIRECTION}`
- Do not inspect scientific VERIFY rates until all primary GPT/Claude cells complete
- Do not launch a second benchmark inside this task
"""


def load_frozen_ids() -> dict[str, Any]:
    paths = freeze_paths()
    if not paths["freeze_manifest"].exists():
        raise RuntimeError("Task 009 sample is not frozen")
    return load_json(paths["freeze_manifest"])
