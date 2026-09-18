"""Task 006 Lane A: qualitative-stakes 20-question paid pilot.

Writes only results/study1_qualitative_pilot/ and to_gpt/006.../lane_A_*.
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .checkpointing import CheckpointStore, deterministic_request_key
from .config import PROJECT_ROOT, load_models_config
from .model_adapters import create_adapter
from .study1_analysis import MODEL_LABELS, VERIFY, _write_csv, _is_verify
from .study1_sample import (
    assert_not_v2_write_target,
    hash_id_list,
    load_frozen_ids,
    load_historical_bundle,
    load_v2b_examples,
)
from .study1_schemas import STUDY1_MODEL_ALIASES, load_study1_config
from .study1_smoke import (
    ProviderAttemptBudget,
    ProviderAttemptCapReached,
    _git_commit,
    _install_attempt_cap,
    _response_value,
    estimate_cost_usd,
)
from .task005_lane_a import load_study1_primary_rows, load_v2_primary_verification
from .task006_common import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    CLAUDE_ENDPOINT,
    CONCURRENCY,
    EXPERIMENT_VERSION,
    EXPECTED_REPEAT_IDS,
    GPT_ENDPOINT,
    LANE_A_DIR,
    MAX_PARSE_REPAIRS,
    PAPER_DIRECTION,
    PROMPT_VERSION,
    PROVIDER_ATTEMPT_CAP,
    QUAL_SQLITE,
    REPEAT_HASH,
    RUN_ID,
    SATURATION_LABEL,
    SCIENTIFIC_CAP,
    SCORE_CONDITIONS,
    SCORE_VALUES,
    STAKES_FAMILIES,
    STUDY1_SQLITE,
    V2_SQLITE,
    assert_006_write_target,
    file_fingerprint,
    prompt_sha256,
    protected_fingerprints,
    sha256_file,
)
from .task006_prompts import (
    FORBIDDEN_SUBSTRINGS,
    MODERATE_CONSEQUENCE,
    STRONGER_CONSEQUENCE,
    audit_prompt_body,
    build_qualitative_prompt,
    build_qualitative_repair_prompt,
    format_displayed_probability,
    parse_qualitative_response,
    prompt_family_id,
    strip_confidence_number,
    unified_diff,
    wrapper_skeleton,
)


class QualAuthorizationError(PermissionError):
    pass


@dataclass(frozen=True)
class QualCall:
    question_id: str
    model_alias: str
    model_endpoint: str
    provider: str
    family: str
    score_condition: str
    displayed_confidence: float | None
    displayed_token: str | None
    frozen_answer: str
    stage1_correct: bool
    reported_confidence: float
    prompt: str
    prompt_hash: str
    prompt_family: str
    request_key: str
    historical_answer_request_key: str
    historical_confidence_request_key: str
    choices: dict[str, str]
    question: str
    selected_sample_hash: str


def _condition_displayed(condition: str) -> float | None:
    if condition == "hidden":
        return None
    if condition.startswith("displayed_"):
        return float(condition.split("_", 1)[1])
    raise ValueError(condition)


def family_template_hash(family: str) -> str:
    template = build_qualitative_prompt(
        question="{question}",
        choices={"A": "{choice_a}", "B": "{choice_b}"},
        frozen_answer="A",
        family=family,
        displayed_confidence=0.7,
    )
    hidden = build_qualitative_prompt(
        question="{question}",
        choices={"A": "{choice_a}", "B": "{choice_b}"},
        frozen_answer="A",
        family=family,
        displayed_confidence=None,
    )
    return prompt_sha256(template + "\n---\n" + hidden)


def build_qual_calls() -> list[QualCall]:
    config = load_study1_config()
    models = load_models_config(config.resolve_path(config.models_config_path))
    _selected, repeats, _sh, repeat_hash = load_frozen_ids(config)
    if repeat_hash != REPEAT_HASH:
        raise QualAuthorizationError(
            f"frozen ID hash {repeat_hash} != {REPEAT_HASH}; STOP Lane A"
        )
    if tuple(repeats) != EXPECTED_REPEAT_IDS:
        raise QualAuthorizationError("repeat IDs do not match the Task-006 frozen list")
    examples = load_v2b_examples(config)
    by_id = {row.example_id: row for row in examples}
    historical = load_historical_bundle(repeats, config)
    calls: list[QualCall] = []
    for qid in repeats:
        example = by_id[qid]
        for alias in STUDY1_MODEL_ALIASES:
            spec = models.models[alias]
            hist = historical[(qid, alias)]
            for family in STAKES_FAMILIES:
                for condition in SCORE_CONDITIONS:
                    displayed = _condition_displayed(condition)
                    prompt = build_qualitative_prompt(
                        question=example.question,
                        choices=example.choices,
                        frozen_answer=str(hist["frozen_answer"]),
                        family=family,
                        displayed_confidence=displayed,
                    )
                    request_key = deterministic_request_key(
                        stage="verification",
                        dataset="mmlu_pro",
                        example_id=qid,
                        model_id=alias,
                        prompt_version=PROMPT_VERSION,
                        experiment_version=EXPERIMENT_VERSION,
                        prompt_family=prompt_family_id(family),
                        stake={
                            "stakes_family": family,
                            "score_condition": condition,
                            "displayed_confidence": displayed,
                        },
                        dependency={
                            "frozen_answer": hist["frozen_answer"],
                            "historical_answer_request_key": hist[
                                "historical_answer_request_key"
                            ],
                        },
                    )
                    calls.append(
                        QualCall(
                            question_id=qid,
                            model_alias=alias,
                            model_endpoint=spec.api_model,
                            provider=spec.provider,
                            family=family,
                            score_condition=condition,
                            displayed_confidence=displayed,
                            displayed_token=(
                                None
                                if displayed is None
                                else format_displayed_probability(displayed)
                            ),
                            frozen_answer=str(hist["frozen_answer"]),
                            stage1_correct=bool(hist["stage1_correct"]),
                            reported_confidence=float(hist["reported_confidence"]),
                            prompt=prompt,
                            prompt_hash=prompt_sha256(prompt),
                            prompt_family=prompt_family_id(family),
                            request_key=request_key,
                            historical_answer_request_key=str(
                                hist["historical_answer_request_key"]
                            ),
                            historical_confidence_request_key=str(
                                hist["historical_confidence_request_key"]
                            ),
                            choices=dict(example.choices),
                            question=example.question,
                            selected_sample_hash=repeat_hash,
                        )
                    )
    return calls


def _audit_diffs(calls: Sequence[QualCall]) -> list[str]:
    errors: list[str] = []
    by_key: dict[tuple[str, str, str], dict[str, QualCall]] = defaultdict(dict)
    for row in calls:
        by_key[(row.question_id, row.model_alias, row.family)][row.score_condition] = row
    for key, by_cond in by_key.items():
        visible = [by_cond[c] for c in SCORE_CONDITIONS if c != "hidden"]
        stripped = [strip_confidence_number(row.prompt) for row in visible]
        if len(set(stripped)) != 1:
            errors.append(f"visible prompts differ by more than the number for {key}")
        hidden = by_cond["hidden"]
        vis = visible[0]
        hidden_cmp = hidden.prompt
        vis_minus = vis.prompt.replace(confidence_block(vis.displayed_confidence), "")
        # hidden vs first visible should differ only by confidence sentence
        if strip_confidence_number(vis.prompt).replace(
            "\nThe AI previously estimated a <DISPLAYED_CONFIDENCE> probability that this frozen answer is correct.\n",
            "",
        ) != hidden.prompt:
            # allow exact historical insertion (blank-line-bounded)
            reconstructed_hidden = vis.prompt.replace(
                confidence_block(vis.displayed_confidence), ""
            )
            if reconstructed_hidden != hidden.prompt:
                errors.append(f"hidden vs visible differs by more than confidence for {key}")
        _ = hidden_cmp, vis_minus
    by_ms: dict[tuple[str, str, str], dict[str, QualCall]] = defaultdict(dict)
    for row in calls:
        by_ms[(row.question_id, row.model_alias, row.score_condition)][row.family] = row
    for key, by_fam in by_ms.items():
        moderate = by_fam["moderate"].prompt
        stronger = by_fam["stronger"].prompt
        if moderate.replace(MODERATE_CONSEQUENCE, STRONGER_CONSEQUENCE) != stronger:
            errors.append(f"moderate vs stronger differs by more than consequence for {key}")
    return errors


def confidence_block(displayed: float | None) -> str:
    from .study1_prompts import confidence_sentence

    if displayed is None:
        return ""
    return confidence_sentence(displayed)


def validate_qual_plan(calls: Sequence[QualCall]) -> dict[str, Any]:
    errors: list[str] = []
    if len(calls) != SCIENTIFIC_CAP:
        errors.append(f"planned calls {len(calls)} != {SCIENTIFIC_CAP}")
    keys = [row.request_key for row in calls]
    if len(set(keys)) != len(keys):
        errors.append("duplicate request keys")
    qids = [row.question_id for row in calls]
    unique = []
    for qid in qids:
        if qid not in unique:
            unique.append(qid)
    if tuple(unique) != EXPECTED_REPEAT_IDS:
        errors.append("call plan question order does not match frozen 20 IDs")
    if hash_id_list(unique) != REPEAT_HASH:
        errors.append("frozen ID hash mismatch")
    models = {row.model_alias for row in calls}
    if models != set(STUDY1_MODEL_ALIASES):
        errors.append(f"unexpected models {models}")
    endpoints = {row.model_alias: row.model_endpoint for row in calls}
    if endpoints.get("openai_gpt56_sol") != GPT_ENDPOINT:
        errors.append("GPT endpoint is not gpt-5.6-sol")
    if endpoints.get("anthropic_sonnet5") != CLAUDE_ENDPOINT:
        errors.append("Claude endpoint is not claude-sonnet-5")
    for row in calls:
        errors.extend(
            f"{row.question_id}/{row.model_alias}/{row.family}/{row.score_condition}: {msg}"
            for msg in audit_prompt_body(
                row.prompt,
                question=row.question,
                choices=row.choices,
                frozen_answer=row.frozen_answer,
                family=row.family,
                displayed_confidence=row.displayed_confidence,
            )
        )
    errors.extend(_audit_diffs(calls))
    try:
        assert_006_write_target(QUAL_SQLITE)
        assert_not_v2_write_target(QUAL_SQLITE, load_study1_config())
    except RuntimeError as exc:
        errors.append(str(exc))
    fingerprints = protected_fingerprints()
    return {
        "ok": not errors,
        "errors": errors[:80],
        "n_errors": len(errors),
        "n_calls": len(calls),
        "n_questions": 20,
        "repeat_ids": list(EXPECTED_REPEAT_IDS),
        "repeat_hash": REPEAT_HASH,
        "gpt_endpoint": GPT_ENDPOINT,
        "claude_endpoint": CLAUDE_ENDPOINT,
        "gpt_reasoning_effort": "none",
        "claude_thinking": "disabled",
        "scientific_cap": SCIENTIFIC_CAP,
        "provider_attempt_cap": PROVIDER_ATTEMPT_CAP,
        "conditions_per_question_model": 8,
        "stakes_families": list(STAKES_FAMILIES),
        "score_conditions": list(SCORE_CONDITIONS),
        "visible_tokens": {
            "0.70": format_displayed_probability(0.70),
            "0.90": format_displayed_probability(0.90),
            "0.99": format_displayed_probability(0.99),
        },
        "prompt_hash_moderate": family_template_hash("moderate"),
        "prompt_hash_stronger": family_template_hash("stronger"),
        "prompt_version": PROMPT_VERSION,
        "qual_sqlite": str(QUAL_SQLITE),
        "will_not_modify_paperDirection": True,
        "paperDirection_sha256": fingerprints["paperDirection"]["sha256"],
        "v2_sha256": fingerprints["v2"]["sha256"],
        "study1_sha256": fingerprints["study1"]["sha256"],
        "q2_sha256": fingerprints["q2"].get("sha256"),
        "forbidden_substrings_checked": list(FORBIDDEN_SUBSTRINGS),
    }


def write_preflight_artifacts(calls: Sequence[QualCall], preflight: Mapping[str, Any]) -> None:
    LANE_A_DIR.mkdir(parents=True, exist_ok=True)
    rendered = LANE_A_DIR / "rendered_prompts"
    diffs = LANE_A_DIR / "prompt_diffs"
    rendered.mkdir(parents=True, exist_ok=True)
    diffs.mkdir(parents=True, exist_ok=True)
    audit_ids = list(EXPECTED_REPEAT_IDS[:2])
    for row in calls:
        if row.question_id not in audit_ids:
            continue
        name = (
            f"{row.question_id.replace(':', '_')}__{row.model_alias}__"
            f"{row.family}__{row.score_condition}.txt"
        )
        (rendered / name).write_text(row.prompt + "\n", encoding="utf-8")
    # representative diffs on first audit question / GPT
    sample = [
        row
        for row in calls
        if row.question_id == audit_ids[0] and row.model_alias == "openai_gpt56_sol"
    ]
    by = {(row.family, row.score_condition): row for row in sample}
    pairs = [
        (by[("moderate", "displayed_0.70")], by[("moderate", "displayed_0.90")], "moderate_0.70_vs_0.90.diff"),
        (by[("moderate", "displayed_0.90")], by[("moderate", "displayed_0.99")], "moderate_0.90_vs_0.99.diff"),
        (by[("stronger", "displayed_0.70")], by[("stronger", "displayed_0.90")], "stronger_0.70_vs_0.90.diff"),
        (by[("moderate", "hidden")], by[("moderate", "displayed_0.70")], "moderate_hidden_vs_0.70.diff"),
        (by[("moderate", "displayed_0.99")], by[("stronger", "displayed_0.99")], "moderate_vs_stronger_0.99.diff"),
    ]
    for left, right, filename in pairs:
        (diffs / filename).write_text(
            unified_diff(left.prompt, right.prompt, left.score_condition, right.score_condition)
            + "\n",
            encoding="utf-8",
        )
    _write_csv(
        LANE_A_DIR / "call_manifest.csv",
        [
            {
                "question_id": row.question_id,
                "model_alias": row.model_alias,
                "model_endpoint": row.model_endpoint,
                "family": row.family,
                "score_condition": row.score_condition,
                "displayed_confidence": row.displayed_confidence,
                "displayed_token": row.displayed_token,
                "frozen_answer": row.frozen_answer,
                "stage1_correct": row.stage1_correct,
                "prompt_hash": row.prompt_hash,
                "prompt_family": row.prompt_family,
                "request_key": row.request_key,
            }
            for row in calls
        ],
    )
    (LANE_A_DIR / "preflight.json").write_text(
        json.dumps(preflight, indent=2) + "\n", encoding="utf-8"
    )
    freeze_lines = [
        "# Lane A prompt freeze",
        "",
        "Paid templates are the Task-006 frozen qualitative families, not the Task-005 Lane D drafts.",
        "",
        f"- Prompt version: `{PROMPT_VERSION}`",
        f"- Moderate family: `{prompt_family_id('moderate')}` hash `{preflight['prompt_hash_moderate']}`",
        f"- Stronger family: `{prompt_family_id('stronger')}` hash `{preflight['prompt_hash_stronger']}`",
        f"- Visible tokens: `{preflight['visible_tokens']}` via `format(value, '.12g')`",
        f"- Frozen 20 IDs hash: `{REPEAT_HASH}`",
        "- Within a family, visible prompts differ only in the displayed confidence number.",
        "- Moderate vs stronger differs only in the exact wrong-answer-consequence sentence.",
        "- Hidden omits the entire confidence sentence.",
        "- Forbidden numerical/EV tokens were audited on the wrapper after stripping question/choice text.",
        "",
        "## Moderate consequence",
        "",
        f"`{MODERATE_CONSEQUENCE}`",
        "",
        "## Stronger consequence",
        "",
        f"`{STRONGER_CONSEQUENCE}`",
        "",
    ]
    (LANE_A_DIR / "prompt_freeze.md").write_text("\n".join(freeze_lines), encoding="utf-8")
    audit_lines = [
        "# Lane A prompt diff audit",
        "",
        f"- Planned scientific calls: **{preflight['n_calls']}**",
        f"- Audit errors: **{preflight['n_errors']}**",
        f"- GPT: `{preflight['gpt_endpoint']}` reasoning.effort=`none`",
        f"- Claude: `{preflight['claude_endpoint']}` thinking=disabled",
        f"- paperDirection.txt untouched: **{preflight['will_not_modify_paperDirection']}**",
        f"- Rendered examples: two frozen questions × 2 models × 8 conditions in `rendered_prompts/`",
        "",
    ]
    if preflight["errors"]:
        audit_lines += ["## Errors", *[f"- {e}" for e in preflight["errors"]], ""]
    else:
        audit_lines += [
            "All planned prompts match the frozen Task-006 skeleton.",
            "Visible-within-family diffs are the numeric confidence token only.",
            "Moderate vs stronger diffs are the consequence sentence only.",
            "",
        ]
    (LANE_A_DIR / "prompt_diff_audit.md").write_text("\n".join(audit_lines), encoding="utf-8")
    (LANE_A_DIR / "validation.md").write_text(
        "\n".join(
            [
                "# Lane A validation / preflight",
                "",
                f"- Exact 20 IDs: see `call_manifest.csv`; hash `{REPEAT_HASH}`",
                f"- GPT endpoint: `{GPT_ENDPOINT}`",
                f"- Claude endpoint: `{CLAUDE_ENDPOINT}`",
                f"- Moderate prompt hash: `{preflight['prompt_hash_moderate']}`",
                f"- Stronger prompt hash: `{preflight['prompt_hash_stronger']}`",
                "- Exact 8 conditions per question/model: hidden, 0.70, 0.90, 0.99 × moderate/stronger",
                f"- Scientific call cap: **{SCIENTIFIC_CAP}**",
                "- paperDirection.txt will not be changed",
                "- Numerical L, C, expected-value formula, outweighs, justified, and 'not as a default' are absent from paid prompt bodies",
                f"- New sqlite: `{QUAL_SQLITE}`",
                f"- Preflight ok: **{preflight['ok']}**",
                "",
            ]
        ),
        encoding="utf-8",
    )


async def _execute_qual_cell(
    *,
    row: QualCall,
    adapter: Any,
    spec: Any,
    checkpoint: CheckpointStore,
    budget: ProviderAttemptBudget,
    code_commit: str | None,
) -> dict[str, Any]:
    attempts_before = budget.used
    prompt = row.prompt
    last_raw = ""
    parse_repairs = 0
    base_record = {
        "study_id": EXPERIMENT_VERSION,
        "experiment_version": EXPERIMENT_VERSION,
        "pilot_or_confirmatory": "exploratory",
        "run_id": RUN_ID,
        "request_key": row.request_key,
        "question_id": row.question_id,
        "model_alias": row.model_alias,
        "model_endpoint": row.model_endpoint,
        "provider": row.provider,
        "stakes_family": row.family,
        "score_condition": row.score_condition,
        "displayed_confidence": row.displayed_confidence,
        "displayed_token": row.displayed_token,
        "reported_confidence": row.reported_confidence,
        "frozen_answer": row.frozen_answer,
        "stage1_correct": row.stage1_correct,
        "prompt_version": PROMPT_VERSION,
        "prompt_family": row.prompt_family,
        "prompt_hash": row.prompt_hash,
        "selected_sample_hash": row.selected_sample_hash,
        "historical_answer_request_key": row.historical_answer_request_key,
        "historical_confidence_request_key": row.historical_confidence_request_key,
        "code_commit": code_commit,
        "model_settings": {
            "requested_model_id": spec.api_model,
            "provider": spec.provider,
            "api_style": spec.api_style,
            "max_output_tokens": spec.max_output_tokens,
            "reasoning_effort": spec.reasoning_effort,
            "thinking": (
                spec.thinking.model_dump()
                if getattr(spec, "thinking", None) is not None
                else None
            ),
        },
    }
    for parse_attempt in range(MAX_PARSE_REPAIRS + 1):
        attempt_kind = "generation" if parse_attempt == 0 else "parse_repair"
        if parse_attempt:
            parse_repairs += 1
            prompt = build_qualitative_repair_prompt(row.prompt, last_raw)
        started = time.perf_counter()
        try:
            response = await adapter.generate(
                stage="verification",
                prompt=prompt,
                request_key=row.request_key,
                allow_paid=True,
                attempt_kind=attempt_kind,
            )
        except Exception as error:
            cause: BaseException | None = error
            seen: set[int] = set()
            while cause is not None and id(cause) not in seen:
                seen.add(id(cause))
                if isinstance(cause, ProviderAttemptCapReached):
                    checkpoint.record_attempt(
                        request_key=row.request_key,
                        attempt_kind=attempt_kind,
                        success=False,
                        provider=adapter.provider,
                        requested_model_id=adapter.api_model,
                        latency_seconds=time.perf_counter() - started,
                        error=cause,
                    )
                    checkpoint.mark_failed(row.request_key, cause)
                    raise cause
                cause = cause.__cause__ or getattr(cause, "__context__", None)
            checkpoint.record_attempt(
                request_key=row.request_key,
                attempt_kind=attempt_kind,
                success=False,
                provider=adapter.provider,
                requested_model_id=adapter.api_model,
                latency_seconds=time.perf_counter() - started,
                error=error,
                sanitized_payload=adapter.prepare_request(
                    stage="verification", prompt=prompt
                ).sanitized_payload(),
            )
            checkpoint.mark_failed(row.request_key, error)
            return {
                "ok": False,
                "row": row,
                "error": str(error),
                "parse_status": "failed",
                "parsed_action": None,
                "raw_response": None,
                "provider_attempts": budget.used - attempts_before,
                "parse_repairs": parse_repairs,
                "estimated_cost_usd": 0.0,
            }
        raw = str(_response_value(response, "raw_response", default="") or "")
        last_raw = raw
        refusal = bool(_response_value(response, "refused", "refusal", default=False))
        returned_model_id = _response_value(
            response, "provider_model_id", "returned_model_id"
        )
        input_tokens = _response_value(response, "input_tokens")
        output_tokens = _response_value(response, "output_tokens")
        latency_seconds = float(_response_value(response, "latency_seconds") or 0.0)
        try:
            if refusal:
                raise ValueError("Provider returned a refusal")
            payload = parse_qualitative_response(raw)
            action = payload.action.value
        except (TypeError, ValueError) as error:
            checkpoint.record_attempt(
                request_key=row.request_key,
                attempt_kind=attempt_kind,
                success=False,
                provider=adapter.provider,
                requested_model_id=adapter.api_model,
                returned_model_id=returned_model_id,
                latency_seconds=latency_seconds,
                raw_output=raw,
                parse_error=str(error),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=_response_value(response, "total_tokens"),
                finish_reason=_response_value(response, "finish_reason"),
                refusal=refusal,
                sanitized_payload=adapter.prepare_request(
                    stage="verification", prompt=prompt
                ).sanitized_payload(),
            )
            if parse_attempt < MAX_PARSE_REPAIRS:
                continue
            checkpoint.mark_failed(row.request_key, error)
            return {
                "ok": False,
                "row": row,
                "error": str(error),
                "parse_status": "parse_failed" if not refusal else "refused",
                "parsed_action": None,
                "raw_response": raw,
                "provider_attempts": budget.used - attempts_before,
                "parse_repairs": parse_repairs,
                "estimated_cost_usd": estimate_cost_usd(spec, input_tokens, output_tokens),
            }
        attempt_number = checkpoint.record_attempt(
            request_key=row.request_key,
            attempt_kind=attempt_kind,
            success=True,
            provider=adapter.provider,
            requested_model_id=adapter.api_model,
            returned_model_id=returned_model_id,
            latency_seconds=latency_seconds,
            raw_output=raw,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=_response_value(response, "total_tokens"),
            finish_reason=_response_value(response, "finish_reason"),
            refusal=False,
            sanitized_payload=adapter.prepare_request(
                stage="verification", prompt=prompt
            ).sanitized_payload(),
        )
        record = {
            **base_record,
            "parsed_action": action,
            "parse_status": "success",
            "raw_response": raw,
            "returned_model_id": returned_model_id,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": latency_seconds * 1000,
            "estimated_cost_usd": estimate_cost_usd(spec, input_tokens, output_tokens),
            "api_timestamp": datetime.now(UTC).isoformat(),
            "attempt_number": int(attempt_number),
            "parse_repairs": parse_repairs,
        }
        checkpoint.mark_success(row.request_key, record)
        return {
            "ok": True,
            "row": row,
            "parsed_action": action,
            "parse_status": "success",
            "raw_response": raw,
            "provider_attempts": budget.used - attempts_before,
            "parse_repairs": parse_repairs,
            "estimated_cost_usd": record["estimated_cost_usd"],
            "record": record,
        }
    raise RuntimeError("qualitative parse-repair loop ended without a result")


async def run_lane_a(*, allow_paid: bool, max_calls: int | None) -> dict[str, Any]:
    load_dotenv(PROJECT_ROOT / ".env")
    before = protected_fingerprints()
    calls = build_qual_calls()
    preflight = validate_qual_plan(calls)
    write_preflight_artifacts(calls, preflight)
    print(json.dumps({k: preflight[k] for k in (
        "ok", "n_calls", "repeat_hash", "gpt_endpoint", "claude_endpoint",
        "prompt_hash_moderate", "prompt_hash_stronger", "scientific_cap",
        "will_not_modify_paperDirection", "n_errors",
    )}, indent=2))
    if not preflight["ok"]:
        return {
            "ok": False,
            "preflight_ok": False,
            "errors": preflight["errors"],
            "new_scientific_calls": 0,
            "api_calls": 0,
            "preflight": preflight,
        }
    if not allow_paid:
        return {
            "ok": True,
            "preflight_ok": True,
            "dry_run": True,
            "planned": len(calls),
            "new_scientific_calls": 0,
            "api_calls": 0,
            "preflight": preflight,
        }
    if max_calls != SCIENTIFIC_CAP:
        raise QualAuthorizationError(
            f"paid Lane A requires --max-calls {SCIENTIFIC_CAP}"
        )
    assert_006_write_target(QUAL_SQLITE)
    QUAL_SQLITE.parent.mkdir(parents=True, exist_ok=True)
    config = load_study1_config()
    models = load_models_config(config.resolve_path(config.models_config_path))
    adapters = {
        alias: create_adapter(alias, models.models[alias])
        for alias in STUDY1_MODEL_ALIASES
    }
    budget = ProviderAttemptBudget(PROVIDER_ATTEMPT_CAP)
    for adapter in adapters.values():
        _install_attempt_cap(adapter, budget)
    code_commit = _git_commit()
    started = time.perf_counter()
    todo: list[QualCall] = []
    reused = 0
    results: list[dict[str, Any]] = []
    with CheckpointStore(QUAL_SQLITE) as checkpoint:
        for row in calls:
            checkpoint.register_request(
                request_key=row.request_key,
                run_id=RUN_ID,
                stage="verification",
                dataset="mmlu_pro",
                example_id=row.question_id,
                model_alias=row.model_alias,
                requested_model_id=row.model_endpoint,
                prompt_version=PROMPT_VERSION,
                stake={
                    "stakes_family": row.family,
                    "score_condition": row.score_condition,
                    "displayed_confidence": row.displayed_confidence,
                    "frozen_answer": row.frozen_answer,
                },
            )
            if checkpoint.request_status(row.request_key) == "success":
                existing = checkpoint.get_record(row.request_key) or {}
                reused += 1
                results.append(
                    {
                        "ok": True,
                        "row": row,
                        "parsed_action": existing.get("parsed_action"),
                        "parse_status": existing.get("parse_status", "success"),
                        "raw_response": existing.get("raw_response"),
                        "provider_attempts": 0,
                        "parse_repairs": existing.get("parse_repairs") or 0,
                        "estimated_cost_usd": float(existing.get("estimated_cost_usd") or 0),
                        "record": existing,
                        "reused": True,
                    }
                )
            else:
                todo.append(row)
        if len(todo) > SCIENTIFIC_CAP:
            raise QualAuthorizationError(
                f"todo {len(todo)} exceeds cap {SCIENTIFIC_CAP}; STOP"
            )
        semaphore = asyncio.Semaphore(CONCURRENCY)

        async def execute_one(row: QualCall) -> dict[str, Any]:
            async with semaphore:
                return await _execute_qual_cell(
                    row=row,
                    adapter=adapters[row.model_alias],
                    spec=models.models[row.model_alias],
                    checkpoint=checkpoint,
                    budget=budget,
                    code_commit=code_commit,
                )

        if todo:
            gathered = await asyncio.gather(*(execute_one(row) for row in todo))
            results.extend(gathered)
    wall = time.perf_counter() - started
    after = protected_fingerprints()
    for name in ("v2", "study1"):
        if before[name]["sha256"] != after[name]["sha256"]:
            raise RuntimeError(f"{name} sqlite changed during Task 006 Lane A")
    if before["q2"].get("sha256") != after["q2"].get("sha256"):
        raise RuntimeError("q2 sqlite changed during Task 006 Lane A")
    if before["paperDirection"]["sha256"] != after["paperDirection"]["sha256"]:
        raise RuntimeError("paperDirection.txt changed during Task 006 Lane A")
    successful = [item for item in results if item.get("ok")]
    failed = [item for item in results if not item.get("ok")]
    return {
        "ok": len(successful) == len(calls) and not failed,
        "preflight_ok": True,
        "dry_run": False,
        "planned": len(calls),
        "reused": reused,
        "new_scientific_calls": len(todo),
        "successful": len(successful),
        "failed": len(failed),
        "provider_attempts": budget.used,
        "wall_clock_seconds": wall,
        "results": results,
        "preflight": preflight,
        "code_commit": code_commit,
        "api_calls": len(todo),
    }


def _boot_mean(values: Sequence[float], n: int = BOOTSTRAP_RESAMPLES) -> tuple[float, float, float]:
    if not values:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(BOOTSTRAP_SEED)
    stats: list[float] = []
    arr = list(values)
    for _ in range(n):
        draw = [arr[rng.randrange(len(arr))] for _ in arr]
        stats.append(sum(draw) / len(draw))
    stats.sort()
    mean = sum(arr) / len(arr)
    return mean, stats[int(0.025 * (len(stats) - 1))], stats[int(0.975 * (len(stats) - 1))]


def analyze_lane_a(payload: Mapping[str, Any]) -> dict[str, Any]:
    results = payload.get("results") or []
    table: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for item in results:
        row: QualCall = item["row"]
        if not item.get("ok") or not item.get("parsed_action"):
            failures.append(
                {
                    "question_id": row.question_id,
                    "model_alias": row.model_alias,
                    "family": row.family,
                    "score_condition": row.score_condition,
                    "error": item.get("error"),
                    "parse_status": item.get("parse_status"),
                }
            )
            continue
        action = str(item["parsed_action"])
        rec = item.get("record") or {}
        table.append(
            {
                "question_id": row.question_id,
                "model_alias": row.model_alias,
                "model_endpoint": row.model_endpoint,
                "family": row.family,
                "score_condition": row.score_condition,
                "displayed_confidence": row.displayed_confidence,
                "displayed_token": row.displayed_token,
                "frozen_answer": row.frozen_answer,
                "stage1_correct": row.stage1_correct,
                "parsed_action": action,
                "verify": int(_is_verify(action)),
                "raw_response": item.get("raw_response"),
                "input_tokens": rec.get("input_tokens"),
                "output_tokens": rec.get("output_tokens"),
                "estimated_cost_usd": item.get("estimated_cost_usd"),
                "prompt_hash": row.prompt_hash,
                "request_key": row.request_key,
            }
        )
    _write_csv(LANE_A_DIR / "results.csv", table)
    _write_csv(LANE_A_DIR / "failures.csv", failures)

    rate_rows: list[dict[str, Any]] = []
    by_cell: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in table:
        by_cell[(row["model_alias"], row["family"], row["score_condition"])].append(row)
    for model in STUDY1_MODEL_ALIASES:
        for family in STAKES_FAMILIES:
            for condition in SCORE_CONDITIONS:
                cell = by_cell[(model, family, condition)]
                n_v = sum(int(r["verify"]) for r in cell)
                rate_rows.append(
                    {
                        "model_alias": model,
                        "model_label": MODEL_LABELS[model],
                        "family": family,
                        "score_condition": condition,
                        "n": len(cell),
                        "n_verify": n_v,
                        "n_use": len(cell) - n_v,
                        "verify_rate": n_v / len(cell) if cell else float("nan"),
                    }
                )

    score_rows: list[dict[str, Any]] = []
    for model in STUDY1_MODEL_ALIASES:
        for family in STAKES_FAMILIES:
            by_q: dict[str, dict[float, int]] = defaultdict(dict)
            for row in table:
                if row["model_alias"] != model or row["family"] != family:
                    continue
                if row["displayed_confidence"] is None:
                    continue
                by_q[row["question_id"]][float(row["displayed_confidence"])] = int(row["verify"])
            diffs = {pair: [] for pair in ((0.70, 0.90), (0.90, 0.99), (0.70, 0.99))}
            monotonic = []
            for qid, scores in by_q.items():
                if set(scores) != {0.70, 0.90, 0.99}:
                    continue
                diffs[(0.70, 0.90)].append(scores[0.90] - scores[0.70])
                diffs[(0.90, 0.99)].append(scores[0.99] - scores[0.90])
                diffs[(0.70, 0.99)].append(scores[0.99] - scores[0.70])
                monotonic.append(int(scores[0.70] >= scores[0.90] >= scores[0.99]))
                _ = qid
            for lo, hi in ((0.70, 0.90), (0.90, 0.99), (0.70, 0.99)):
                mean, lo_ci, hi_ci = _boot_mean(diffs[(lo, hi)])
                score_rows.append(
                    {
                        "model_alias": model,
                        "family": family,
                        "from_score": lo,
                        "to_score": hi,
                        "n_questions": len(diffs[(lo, hi)]),
                        "mean_delta_verify": mean,
                        "ci_lower": lo_ci,
                        "ci_upper": hi_ci,
                        "n_increased": sum(1 for v in diffs[(lo, hi)] if v > 0),
                        "n_decreased": sum(1 for v in diffs[(lo, hi)] if v < 0),
                        "n_unchanged": sum(1 for v in diffs[(lo, hi)] if v == 0),
                        "frac_weakly_decreasing_0.70_0.90_0.99": (
                            sum(monotonic) / len(monotonic) if monotonic else float("nan")
                        ),
                        "note": "pilot question-bootstrap CI; n=20",
                    }
                )

    stakes_rows: list[dict[str, Any]] = []
    for model in STUDY1_MODEL_ALIASES:
        for condition in SCORE_CONDITIONS:
            if condition == "hidden":
                displayed = None
            else:
                displayed = _condition_displayed(condition)
            paired = defaultdict(dict)
            for row in table:
                if row["model_alias"] != model or row["score_condition"] != condition:
                    continue
                paired[row["question_id"]][row["family"]] = int(row["verify"])
            deltas = []
            stronger_only = 0
            moderate_only = 0
            both = 0
            neither = 0
            for qid, fams in paired.items():
                if "moderate" not in fams or "stronger" not in fams:
                    continue
                d = fams["stronger"] - fams["moderate"]
                deltas.append(d)
                if fams["stronger"] and not fams["moderate"]:
                    stronger_only += 1
                elif fams["moderate"] and not fams["stronger"]:
                    moderate_only += 1
                elif fams["stronger"] and fams["moderate"]:
                    both += 1
                else:
                    neither += 1
                _ = qid
            mean, lo, hi = _boot_mean(deltas)
            stakes_rows.append(
                {
                    "model_alias": model,
                    "score_condition": condition,
                    "displayed_confidence": displayed,
                    "n": len(deltas),
                    "mean_stronger_minus_moderate": mean,
                    "ci_lower": lo,
                    "ci_upper": hi,
                    "n_stronger_only": stronger_only,
                    "n_moderate_only": moderate_only,
                    "n_both_verify": both,
                    "n_both_use": neither,
                    "note": "pilot paired question comparison",
                }
            )

    sat_rows: list[dict[str, Any]] = []
    for row in rate_rows:
        if row["score_condition"] == "hidden":
            continue
        rate = float(row["verify_rate"])
        sat_rows.append(
            {
                **row,
                "rate_0_or_100": rate in {0.0, 1.0},
                "rate_le_10_or_ge_90": rate <= 0.10 or rate >= 0.90,
                "useful_intermediate_10_90": 0.10 < rate < 0.90,
                "n_verify": row["n_verify"],
                "n_use": row["n_use"],
                "both_actions_present": row["n_verify"] not in {0, row["n"]} and row["n_use"] != 0,
            }
        )

    routing_rows: list[dict[str, Any]] = []
    for model in STUDY1_MODEL_ALIASES:
        for family in STAKES_FAMILIES:
            for condition in SCORE_CONDITIONS:
                if condition == "hidden":
                    continue
                cell = by_cell[(model, family, condition)]
                n_v = sum(int(r["verify"]) for r in cell)
                both_actions = 0 < n_v < len(cell)
                wrong = [r for r in cell if not r["stage1_correct"]]
                correct = [r for r in cell if r["stage1_correct"]]
                if not both_actions or not wrong or not correct:
                    routing_rows.append(
                        {
                            "model_alias": model,
                            "family": family,
                            "score_condition": condition,
                            "n": len(cell),
                            "n_wrong": len(wrong),
                            "n_correct": len(correct),
                            "p_verify_given_wrong": "",
                            "p_verify_given_correct": "",
                            "delta": "",
                            "label": SATURATION_LABEL,
                        }
                    )
                    continue
                p_w = sum(int(r["verify"]) for r in wrong) / len(wrong)
                p_c = sum(int(r["verify"]) for r in correct) / len(correct)
                routing_rows.append(
                    {
                        "model_alias": model,
                        "family": family,
                        "score_condition": condition,
                        "n": len(cell),
                        "n_wrong": len(wrong),
                        "n_correct": len(correct),
                        "p_verify_given_wrong": p_w,
                        "p_verify_given_correct": p_c,
                        "delta": p_w - p_c,
                        "label": "identifiable_pilot_only",
                    }
                )

    hidden_rows: list[dict[str, Any]] = []
    study1 = [
        r
        for r in load_study1_primary_rows()
        if r["question_id"] in set(EXPECTED_REPEAT_IDS)
        and r["display_condition"] == "hidden"
    ]
    v2_hidden = [
        r
        for r in load_v2_primary_verification()
        if r["question_id"] in set(EXPECTED_REPEAT_IDS)
        and r["confidence_visibility"] == "hidden"
        and r["decision_owner"] == "ai_system"
        and float(r["L"]) == 10.0
    ]
    s1_map = {(r["question_id"], r["model_alias"], float(r["L"])): r for r in study1}
    v2_map = {(r["question_id"], r["model_alias"]): r for r in v2_hidden}
    for model in STUDY1_MODEL_ALIASES:
        for family in STAKES_FAMILIES:
            hidden_cell = by_cell[(model, family, "hidden")]
            n_v = sum(int(r["verify"]) for r in hidden_cell)
            hidden_rate = n_v / len(hidden_cell) if hidden_cell else float("nan")
            for condition in SCORE_CONDITIONS:
                cell = by_cell[(model, family, condition)]
                cond_rate = (
                    sum(int(r["verify"]) for r in cell) / len(cell) if cell else float("nan")
                )
                agree_s1_10 = []
                agree_v2 = []
                for r in hidden_cell:
                    s1 = s1_map.get((r["question_id"], model, 10.0))
                    if s1:
                        agree_s1_10.append(
                            int(_is_verify(r["parsed_action"]) == _is_verify(s1["parsed_action"]))
                        )
                    v2 = v2_map.get((r["question_id"], model))
                    if v2:
                        agree_v2.append(
                            int(_is_verify(r["parsed_action"]) == _is_verify(v2["parsed_action"]))
                        )
                hidden_rows.append(
                    {
                        "model_alias": model,
                        "family": family,
                        "compared_condition": condition,
                        "qual_hidden_verify_rate": hidden_rate,
                        "qual_condition_verify_rate": cond_rate,
                        "hidden_minus_condition": (
                            hidden_rate - cond_rate
                            if math.isfinite(hidden_rate) and math.isfinite(cond_rate)
                            else float("nan")
                        ),
                        "agreement_with_study1_hidden_L10": (
                            sum(agree_s1_10) / len(agree_s1_10) if agree_s1_10 else ""
                        ),
                        "agreement_with_v2_ai_L10_hidden": (
                            sum(agree_v2) / len(agree_v2) if agree_v2 else ""
                        ),
                        "note": "descriptive continuity only; stakes wording changed",
                    }
                )

    _write_csv(LANE_A_DIR / "score_response.csv", score_rows)
    _write_csv(LANE_A_DIR / "stakes_manipulation.csv", stakes_rows)
    _write_csv(LANE_A_DIR / "saturation_diagnostic.csv", sat_rows)
    _write_csv(LANE_A_DIR / "pilot_item_routing.csv", routing_rows)
    _write_csv(LANE_A_DIR / "hidden_comparison.csv", hidden_rows)

    figures = LANE_A_DIR / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        for model in STUDY1_MODEL_ALIASES:
            fig, ax = plt.subplots(figsize=(6.2, 3.8))
            xs = [0.70, 0.90, 0.99]
            for family, style in (("moderate", "o-"), ("stronger", "s--")):
                ys = []
                for x in xs:
                    cell = [r for r in rate_rows if r["model_alias"] == model and r["family"] == family and r["score_condition"] == f"displayed_{x:.2f}"]
                    ys.append(cell[0]["verify_rate"] if cell else float("nan"))
                ax.plot(xs, ys, style, label=family)
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlabel("displayed confidence")
            ax.set_ylabel("P(VERIFY)")
            ax.set_title(f"{MODEL_LABELS[model]} qualitative score response (n=20)")
            ax.legend()
            fig.tight_layout()
            fig.savefig(figures / f"score_response_{model}.png", dpi=140)
            fig.savefig(figures / f"score_response_{model}.pdf")
            plt.close(fig)
    except Exception:
        pass

    gate = assign_gate(rate_rows, score_rows, stakes_rows, sat_rows)
    cost = sum(float(r.get("estimated_cost_usd") or 0) for r in table)
    (LANE_A_DIR / "cost_summary.md").write_text(
        "\n".join(
            [
                "# Lane A cost summary",
                "",
                f"- Scientific successes: {len(table)} / {SCIENTIFIC_CAP}",
                f"- Failures: {len(failures)}",
                f"- New scientific calls: {payload.get('new_scientific_calls')}",
                f"- Provider attempts: {payload.get('provider_attempts')}",
                f"- Reused successes: {payload.get('reused')}",
                f"- Estimated USD: {cost:.6f}",
                f"- Wall-clock seconds: {payload.get('wall_clock_seconds')}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "n_success": len(table),
        "n_fail": len(failures),
        "gate": gate,
        "rate_rows": rate_rows,
        "score_rows": score_rows,
        "stakes_rows": stakes_rows,
        "sat_rows": sat_rows,
        "routing_rows": routing_rows,
        "hidden_rows": hidden_rows,
        "cost_usd": cost,
        "table": table,
    }


def assign_gate(
    rate_rows: Sequence[Mapping[str, Any]],
    score_rows: Sequence[Mapping[str, Any]],
    stakes_rows: Sequence[Mapping[str, Any]],
    sat_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    def gpt_delta(family: str) -> float:
        rows = [
            r
            for r in score_rows
            if r["model_alias"] == "openai_gpt56_sol"
            and r["family"] == family
            and r["from_score"] == 0.70
            and r["to_score"] == 0.99
        ]
        return float(rows[0]["mean_delta_verify"]) if rows else float("nan")

    gpt_mod = gpt_delta("moderate")
    gpt_str = gpt_delta("stronger")
    claude_mod = [
        r
        for r in score_rows
        if r["model_alias"] == "anthropic_sonnet5"
        and r["family"] == "moderate"
        and r["from_score"] == 0.70
        and r["to_score"] == 0.99
    ]
    claude_d = float(claude_mod[0]["mean_delta_verify"]) if claude_mod else float("nan")
    gpt_responsive = (gpt_mod <= -0.15) or (gpt_str <= -0.15)
    claude_responsive = claude_d <= -0.15
    gpt_sat = [
        r
        for r in sat_rows
        if r["model_alias"] == "openai_gpt56_sol"
    ]
    gpt_intermediate = any(r["useful_intermediate_10_90"] for r in gpt_sat)
    gpt_all_corner = all(r["rate_0_or_100"] for r in gpt_sat) if gpt_sat else True
    stakes_gpt = [
        r
        for r in stakes_rows
        if r["model_alias"] == "openai_gpt56_sol" and r["score_condition"] != "hidden"
    ]
    stakes_mean = (
        sum(float(r["mean_stronger_minus_moderate"]) for r in stakes_gpt) / len(stakes_gpt)
        if stakes_gpt
        else 0.0
    )
    stakes_active = stakes_mean >= 0.05
    stakes_broken = stakes_mean <= -0.25
    stronger_pinned = all(
        r["verify_rate"] >= 0.95
        for r in rate_rows
        if r["model_alias"] == "openai_gpt56_sol"
        and r["family"] == "stronger"
        and r["score_condition"] != "hidden"
    )
    if gpt_all_corner and not gpt_responsive:
        gate = "A-FAIL-QUALITATIVE"
        reason = "GPT qualitative cells remain cornered and displayed score does not systematically move VERIFY/USE."
    elif (gpt_responsive or claude_responsive) and (gpt_all_corner or stakes_broken):
        gate = "A-REVISE-PROMPT"
        reason = "Score still moves some cells, but wording pins almost every GPT cell or the stakes manipulation is backwards/nonsensical."
    elif gpt_responsive and gpt_intermediate and not stronger_pinned:
        if stakes_active or not stronger_pinned:
            gate = "A-GO-FULL-QUALITATIVE" if (stakes_active and gpt_intermediate) else "A-GO-NARROW"
            reason = (
                "Qualitative displayed score systematically changes GPT verification without a numerical threshold."
            )
        else:
            gate = "A-GO-NARROW"
            reason = "Score responsiveness survives, but the design should be narrowed."
    elif gpt_responsive or claude_responsive:
        gate = "A-GO-NARROW"
        reason = "Score responsiveness survives qualitatively, but measurability is limited to a subset of families/models/cells."
    else:
        gate = "A-FAIL-QUALITATIVE"
        reason = "Displayed score has little systematic effect once numerical cost arithmetic is removed."
    if gate == "A-GO-FULL-QUALITATIVE" and (stronger_pinned or not gpt_intermediate):
        gate = "A-GO-NARROW"
        reason = "Score responsiveness survives, but one family saturates or GPT mixed cells are scarce."
    return {
        "gate": gate,
        "reason": reason,
        "gpt_moderate_0.70_to_0.99": gpt_mod,
        "gpt_stronger_0.70_to_0.99": gpt_str,
        "claude_moderate_0.70_to_0.99": claude_d,
        "gpt_has_intermediate_cell": gpt_intermediate,
        "stakes_mean_gpt_visible": stakes_mean,
        "stakes_active": stakes_active,
    }
