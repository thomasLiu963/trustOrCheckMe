"""End-to-end scientific analysis over successful staged experiment records.

This module deliberately depends only on the scientific analysis modules. It
accepts record dictionaries directly or discovers records in a SQLite store,
so it can operate with a ResultStore without coupling to its runtime code.
"""

from __future__ import annotations

import csv
import json
import math
import sqlite3
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .bootstrap import bootstrap_mean, paired_policy_bootstrap
from .calibration import (
    assign_calibrated_probabilities,
    brier_score,
    deterministic_model_splits,
    expected_calibration_error,
    fit_per_model_calibrators,
    persist_calibration_artifacts,
)
from .plotting import generate_core_figures
from .scoring import (
    ERROR_COSTS,
    RELY,
    VERIFICATION_COST,
    VERIFY,
    confidence_policy_action,
    monotonicity_from_records,
    normalize_action,
    oracle_action,
    score_decision,
    score_policy,
)


@dataclass
class AnalysisResult:
    direct_rows: list[dict[str, Any]]
    policy_rows: list[dict[str, Any]]
    model_summary: list[dict[str, Any]]
    decision_summary: list[dict[str, Any]]
    policy_summary: list[dict[str, Any]]
    policy_differences: list[dict[str, Any]]
    domain_summary: list[dict[str, Any]]
    request_coverage: list[dict[str, Any]]
    usage_summary: list[dict[str, Any]]
    output_paths: dict[str, Any]
    warnings: list[str]


_ALIASES: dict[str, tuple[str, ...]] = {
    "example_id": ("example_id", "question_id", "item_id", "sample_id"),
    "model_id": ("model_id", "model_alias", "model", "requested_model"),
    "run_id": ("run_id", "experiment_id"),
    "stage": ("stage", "stage_name", "request_stage"),
    "answer_label": ("answer_label", "answer", "selected_answer"),
    "correct_label": ("correct_label", "ground_truth", "target", "label"),
    "is_correct": ("is_correct", "correct", "answer_correct"),
    "probability_correct": (
        "probability_correct",
        "confidence",
        "probability",
        "stated_confidence",
    ),
    "recommendation": ("recommendation", "action", "decision"),
    "error_cost": ("error_cost", "wrong_answer_cost", "loss", "L"),
    "verification_cost": ("verification_cost", "check_cost", "C"),
    "category": ("category", "domain", "subject"),
    "question": ("question", "question_text", "prompt_question"),
    "status": ("status", "request_status"),
    "timestamp": ("timestamp", "created_at", "completed_at", "updated_at"),
}

_JSON_FIELDS = (
    "record_json",
    "result_json",
    "payload_json",
    "response_json",
    "parsed_response",
    "parsed",
    "result",
    "payload",
    "data",
    "response",
)


def _json_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", errors="replace")
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not (text.startswith("{") and text.endswith("}")):
        return None
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        return None
    return decoded if isinstance(decoded, Mapping) else None


def _merge_nested(record: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(record)
    for field in _JSON_FIELDS:
        nested = _json_mapping(record.get(field))
        if nested is None:
            continue
        # Outer persistence metadata wins; parsed scientific fields fill gaps.
        for key, value in _merge_nested(nested).items():
            if key not in merged or merged[key] is None:
                merged[key] = value
    return merged


def _first(record: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in record and record[name] is not None:
            return record[name]
    return None


def _bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1", "correct"}:
            return True
        if normalized in {"false", "no", "0", "incorrect", "wrong"}:
            return False
    return None


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _successful(record: Mapping[str, Any]) -> bool:
    explicit = _bool(record.get("successful", record.get("success")))
    if explicit is False:
        return False
    status = _first(record, _ALIASES["status"])
    if status is None:
        return True
    normalized = str(status).strip().lower()
    failures = {
        "failed",
        "failure",
        "error",
        "timeout",
        "refused",
        "parse_failed",
        "cancelled",
        "canceled",
    }
    return normalized not in failures


def _stage(
    value: Any, table_name: str = "", record: Mapping[str, Any] | None = None
) -> str | None:
    text = f"{value or ''} {table_name}".lower()
    if any(token in text for token in ("trust", "decision", "recommend", "stage3")):
        return "trust"
    if any(token in text for token in ("confidence", "stage2")):
        return "confidence"
    if any(token in text for token in ("answer", "stage1")):
        return "answer"
    record = record or {}
    if _first(record, _ALIASES["recommendation"]) is not None:
        return "trust"
    if _first(record, _ALIASES["probability_correct"]) is not None:
        return "confidence"
    if _first(record, _ALIASES["answer_label"]) is not None:
        return "answer"
    return None


def normalize_record(
    source: Mapping[str, Any], *, table_name: str = ""
) -> dict[str, Any] | None:
    """Normalize common flat/nested ResultStore record shapes."""
    merged = _merge_nested(source)
    if not _successful(merged):
        return None
    normalized = dict(merged)
    for canonical, names in _ALIASES.items():
        value = _first(merged, names)
        if value is not None:
            normalized[canonical] = value
    normalized["stage"] = _stage(
        normalized.get("stage"), table_name=table_name, record=normalized
    )
    for key in ("example_id", "model_id", "run_id"):
        if normalized.get(key) is not None:
            normalized[key] = str(normalized[key])
    if normalized.get("verification_cost") is None:
        normalized["verification_cost"] = VERIFICATION_COST
    for key in ("probability_correct", "error_cost", "verification_cost"):
        if normalized.get(key) is not None:
            normalized[key] = _number(normalized[key])
    normalized["is_correct"] = _bool(normalized.get("is_correct"))
    answer = normalized.get("answer_label")
    correct = normalized.get("correct_label")
    if normalized["is_correct"] is None and answer is not None and correct is not None:
        normalized["is_correct"] = (
            str(answer).strip().upper() == str(correct).strip().upper()
        )
    if normalized.get("recommendation") is not None:
        try:
            normalized["recommendation"] = normalize_action(
                normalized["recommendation"]
            )
        except (TypeError, ValueError):
            normalized["recommendation"] = None
    if not normalized.get("example_id") or not normalized.get("model_id"):
        return None
    return normalized


def load_sqlite_records(path: str | Path) -> list[dict[str, Any]]:
    """Read successful records from all user tables in a SQLite ResultStore."""
    database = Path(path)
    if not database.exists():
        raise FileNotFoundError(database)
    records: list[dict[str, Any]] = []
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        table_rows = connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        for table_row in table_rows:
            table = str(table_row["name"])
            quoted = '"' + table.replace('"', '""') + '"'
            try:
                rows = connection.execute(f"SELECT * FROM {quoted}")
            except sqlite3.DatabaseError:
                continue
            for row in rows:
                normalized = normalize_record(dict(row), table_name=table)
                if normalized is not None:
                    records.append(normalized)
    finally:
        connection.close()
    return records


def load_request_coverage(path: str | Path) -> list[dict[str, Any]]:
    """Summarize request completion and failures without exposing response text."""
    database = Path(path)
    if not database.exists():
        return []
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        has_requests = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='requests'"
        ).fetchone()
        if not has_requests:
            return []
        rows = connection.execute(
            """
            SELECT stage, model_alias AS model_id, status, COUNT(*) AS count
            FROM requests
            GROUP BY stage, model_alias, status
            ORDER BY stage, model_alias, status
            """
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()


def load_usage_summary(
    path: str | Path,
    pricing_per_million_tokens: Mapping[str, Mapping[str, float]] | None = None,
) -> list[dict[str, Any]]:
    """Aggregate billed token-bearing attempts, including malformed outputs."""
    database = Path(path)
    if not database.exists():
        return []
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        has_attempts = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='attempts'"
        ).fetchone()
        if not has_attempts:
            return []
        rows = connection.execute(
            """
            SELECT r.stage, r.model_alias AS model_id, a.provider,
                   COUNT(*) AS attempts,
                   SUM(COALESCE(a.input_tokens, 0)) AS input_tokens,
                   SUM(COALESCE(a.output_tokens, 0)) AS output_tokens,
                   SUM(COALESCE(a.total_tokens, 0)) AS total_tokens
            FROM attempts a
            JOIN requests r ON r.request_key = a.request_key
            WHERE a.input_tokens IS NOT NULL OR a.output_tokens IS NOT NULL
            GROUP BY r.stage, r.model_alias, a.provider
            ORDER BY r.stage, r.model_alias, a.provider
            """
        ).fetchall()
    finally:
        connection.close()
    output: list[dict[str, Any]] = []
    prices = pricing_per_million_tokens or {}
    for source in rows:
        row = dict(source)
        model_prices = prices.get(str(row["model_id"]), {})
        input_price = float(model_prices.get("input", 0))
        output_price = float(model_prices.get("output", 0))
        row["observed_cost_usd"] = (
            int(row["input_tokens"]) * input_price
            + int(row["output_tokens"]) * output_price
        ) / 1_000_000
        output.append(row)
    return output


def load_records(
    source: str | Path | Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Load SQLite or normalize an iterable of record dictionaries."""
    if isinstance(source, (str, Path)):
        return load_sqlite_records(source)
    records: list[dict[str, Any]] = []
    for record in source:
        normalized = normalize_record(record)
        if normalized is not None:
            records.append(normalized)
    return records


def _identity(record: Mapping[str, Any]) -> tuple[str | None, str, str]:
    return (
        str(record["run_id"]) if record.get("run_id") is not None else None,
        str(record["model_id"]),
        str(record["example_id"]),
    )


def _latest_map(
    records: Iterable[Mapping[str, Any]],
) -> tuple[
    dict[tuple[str | None, str, str], Mapping[str, Any]],
    dict[tuple[str, str], Mapping[str, Any]],
]:
    exact: dict[tuple[str | None, str, str], Mapping[str, Any]] = {}
    fallback_candidates: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(
        list
    )
    for record in records:
        key = _identity(record)
        exact[key] = record
        fallback_candidates[(key[1], key[2])].append(record)
    fallback = {
        key: max(values, key=lambda row: str(row.get("timestamp", "")))
        for key, values in fallback_candidates.items()
    }
    return exact, fallback


def _lookup(
    record: Mapping[str, Any],
    exact: Mapping[tuple[str | None, str, str], Mapping[str, Any]],
    fallback: Mapping[tuple[str, str], Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    key = _identity(record)
    matched = exact.get(key)
    if matched is not None:
        return matched
    if key[0] is not None:
        # A stage lacking run metadata may safely fill a run-aware record, but
        # never join a different explicit run merely because it is newer.
        return exact.get((None, key[1], key[2]))
    return fallback.get((key[1], key[2]))


def join_stages(
    records: Iterable[Mapping[str, Any]],
    *,
    seed: int = 2025,
    calibration_fraction: float = 0.20,
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    list[str],
]:
    """Join successful answer, confidence, and trust stages without leakage."""
    rows = list(records)
    answers = [row for row in rows if row.get("stage") == "answer"]
    confidences = [row for row in rows if row.get("stage") == "confidence"]
    decisions = [row for row in rows if row.get("stage") == "trust"]
    answer_exact, answer_fallback = _latest_map(answers)
    confidence_exact, confidence_fallback = _latest_map(confidences)
    warnings: list[str] = []

    calibration_rows: list[dict[str, Any]] = []
    seen_calibration: set[tuple[str, str]] = set()
    for confidence in confidences:
        answer = _lookup(confidence, answer_exact, answer_fallback)
        probability = _number(confidence.get("probability_correct"))
        is_correct = answer.get("is_correct") if answer else None
        key = (str(confidence["model_id"]), str(confidence["example_id"]))
        if probability is None or is_correct is None or key in seen_calibration:
            continue
        seen_calibration.add(key)
        calibration_rows.append(
            {
                "model_id": key[0],
                "example_id": key[1],
                "probability_correct": probability,
                "is_correct": bool(is_correct),
            }
        )

    splits = deterministic_model_splits(
        calibration_rows,
        seed=seed,
        calibration_fraction=calibration_fraction,
    )
    calibrators, metadata = fit_per_model_calibrators(calibration_rows, splits)
    calibrated_rows = assign_calibrated_probabilities(
        calibration_rows, splits, calibrators, evaluation_only=True
    )
    calibrated_by_key = {
        (row["model_id"], row["example_id"]): row for row in calibrated_rows
    }

    joined: list[dict[str, Any]] = []
    missing_answers = 0
    invalid_decisions = 0
    unsupported_stakes = 0
    nonfrozen_verification_costs = 0
    for decision in decisions:
        answer = _lookup(decision, answer_exact, answer_fallback)
        confidence = _lookup(decision, confidence_exact, confidence_fallback)
        action = decision.get("recommendation")
        error_cost = _number(decision.get("error_cost"))
        if answer is None or answer.get("is_correct") is None:
            missing_answers += 1
            continue
        if action not in {RELY, VERIFY} or error_cost is None:
            invalid_decisions += 1
            continue
        if error_cost not in ERROR_COSTS:
            unsupported_stakes += 1
            continue
        supplied_verification_cost = _number(decision.get("verification_cost"))
        if (
            supplied_verification_cost is not None
            and supplied_verification_cost != VERIFICATION_COST
        ):
            nonfrozen_verification_costs += 1
        model = str(decision["model_id"])
        example = str(decision["example_id"])
        calibrated = calibrated_by_key.get((model, example), {})
        raw_probability = (
            _number(confidence.get("probability_correct")) if confidence else None
        )
        joined.append(
            {
                "run_id": decision.get("run_id"),
                "model_id": model,
                "example_id": example,
                "category": answer.get("category", decision.get("category")),
                "question": answer.get("question", decision.get("question")),
                "answer_label": answer.get("answer_label"),
                "correct_label": answer.get("correct_label"),
                "is_correct": bool(answer["is_correct"]),
                "error_cost": error_cost,
                # Analysis always uses the preregistered, frozen C=1 model.
                "verification_cost": VERIFICATION_COST,
                "recommendation": action,
                "probability_correct": raw_probability,
                "calibrated_probability": calibrated.get("calibrated_probability"),
                "calibration_partition": calibrated.get(
                    "calibration_partition", "unassigned"
                ),
            }
        )
    if missing_answers:
        warnings.append(
            f"Skipped {missing_answers} trust records without a scored answer."
        )
    if invalid_decisions:
        warnings.append(
            f"Skipped {invalid_decisions} trust records with invalid action/cost."
        )
    if unsupported_stakes:
        warnings.append(
            f"Skipped {unsupported_stakes} trust records outside frozen "
            f"L={list(ERROR_COSTS)}."
        )
    if nonfrozen_verification_costs:
        warnings.append(
            f"Ignored non-frozen verification costs on "
            f"{nonfrozen_verification_costs} trust records and used C=1."
        )
    if not calibration_rows:
        warnings.append("No joinable confidence labels; calibration is unavailable.")
    return joined, splits, metadata, warnings


def score_joined_records(
    joined: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Score direct decisions and all available policies."""
    direct_rows: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []
    for source in joined:
        row = dict(source)
        direct = score_decision(
            is_correct=bool(row["is_correct"]),
            action=str(row["recommendation"]),
            error_cost=float(row["error_cost"]),
            verification_cost=float(row["verification_cost"]),
            raw_probability=row.get("probability_correct"),
            calibrated_probability=row.get("calibrated_probability"),
        )
        direct_row = {**row, **direct}
        direct_rows.append(direct_row)

        policies: list[tuple[str, str]] = [
            ("direct", str(row["recommendation"])),
            ("always_rely", RELY),
            ("always_verify", VERIFY),
            ("oracle", oracle_action(bool(row["is_correct"]))),
        ]
        if row.get("probability_correct") is not None:
            policies.append(
                (
                    "raw_confidence",
                    confidence_policy_action(
                        row["probability_correct"],
                        row["error_cost"],
                        row["verification_cost"],
                    ),
                )
            )
        if row.get("calibrated_probability") is not None:
            policies.append(
                (
                    "calibrated_confidence",
                    confidence_policy_action(
                        row["calibrated_probability"],
                        row["error_cost"],
                        row["verification_cost"],
                    ),
                )
            )
        for policy, action in policies:
            policy_rows.append(
                {
                    **{
                        key: row.get(key)
                        for key in (
                            "run_id",
                            "model_id",
                            "example_id",
                            "category",
                            "is_correct",
                            "error_cost",
                            "verification_cost",
                            "calibration_partition",
                        )
                    },
                    "policy": policy,
                    **score_policy(
                        is_correct=bool(row["is_correct"]),
                        action=action,
                        error_cost=float(row["error_cost"]),
                        verification_cost=float(row["verification_cost"]),
                    ),
                }
            )
    return direct_rows, policy_rows


def _metric_ci(
    rows: Sequence[Mapping[str, Any]],
    value_key: str,
    *,
    n_resamples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    result = bootstrap_mean(
        rows,
        value_key,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        seed=seed,
    )
    return {
        value_key: result.estimate,
        f"{value_key}_ci_lower": result.lower,
        f"{value_key}_ci_upper": result.upper,
        f"{value_key}_n": len(rows),
    }


def summarize_decisions(
    direct_rows: Iterable[Mapping[str, Any]],
    *,
    n_resamples: int = 5000,
    confidence_level: float = 0.95,
    seed: int = 2025,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for source in direct_rows:
        row = dict(source)
        row["verify_rate"] = float(bool(row.get("verify")))
        row["direct_vs_raw_disagreement"] = row.get(
            "direct_vs_raw_confidence_disagreement"
        )
        row["direct_vs_calibrated_disagreement"] = row.get(
            "direct_vs_calibrated_confidence_disagreement"
        )
        groups[(str(row["model_id"]), float(row["error_cost"]))].append(row)

    output: list[dict[str, Any]] = []
    for (model, error_cost), rows in sorted(groups.items()):
        summary: dict[str, Any] = {
            "model_id": model,
            "error_cost": error_cost,
            "n_decisions": len(rows),
            "n_questions": len({row["example_id"] for row in rows}),
        }
        metric_rows = {
            "verify_rate": rows,
            "unsafe_reliance": [row for row in rows if not bool(row["is_correct"])],
            "unnecessary_verification": [
                row for row in rows if bool(row["is_correct"])
            ],
            "realized_cost": rows,
            "regret": rows,
            "direct_vs_raw_disagreement": [
                row for row in rows if row.get("direct_vs_raw_disagreement") is not None
            ],
            "direct_vs_calibrated_disagreement": [
                row
                for row in rows
                if row.get("direct_vs_calibrated_disagreement") is not None
            ],
        }
        for index, (metric, eligible) in enumerate(metric_rows.items()):
            summary.update(
                _metric_ci(
                    eligible,
                    metric,
                    n_resamples=n_resamples,
                    confidence_level=confidence_level,
                    seed=seed + index,
                )
            )
        for direction in ("more_trusting", "more_cautious"):
            raw = [
                row
                for row in rows
                if row.get("direct_vs_raw_confidence_direction") == direction
            ]
            calibrated = [
                row
                for row in rows
                if row.get("direct_vs_calibrated_confidence_direction") == direction
            ]
            summary[f"direct_vs_raw_{direction}_rate"] = (
                len(raw)
                / sum(
                    row.get("direct_vs_raw_confidence_direction") is not None
                    for row in rows
                )
                if raw
                or any(
                    row.get("direct_vs_raw_confidence_direction") is not None
                    for row in rows
                )
                else float("nan")
            )
            summary[f"direct_vs_calibrated_{direction}_rate"] = (
                len(calibrated)
                / sum(
                    row.get("direct_vs_calibrated_confidence_direction") is not None
                    for row in rows
                )
                if calibrated
                or any(
                    row.get("direct_vs_calibrated_confidence_direction") is not None
                    for row in rows
                )
                else float("nan")
            )
        output.append(summary)
    return output


def summarize_policies(
    policy_rows: Iterable[Mapping[str, Any]],
    *,
    n_resamples: int = 5000,
    confidence_level: float = 0.95,
    seed: int = 2025,
) -> list[dict[str, Any]]:
    all_rows = [dict(row) for row in policy_rows]
    evaluation_ids: dict[tuple[str, float], set[str]] = defaultdict(set)
    for row in all_rows:
        if row.get("policy") == "calibrated_confidence":
            evaluation_ids[(str(row["model_id"]), float(row["error_cost"]))].add(
                str(row["example_id"])
            )
    groups: dict[tuple[str, str, float], list[Mapping[str, Any]]] = defaultdict(list)
    for row in all_rows:
        groups[
            (str(row["model_id"]), str(row["policy"]), float(row["error_cost"]))
        ].append(row)
    output: list[dict[str, Any]] = []
    for (model, policy, error_cost), rows in sorted(groups.items()):
        fair_ids = evaluation_ids.get((model, error_cost))
        comparison_scope = "evaluation" if fair_ids else "all"
        if fair_ids:
            rows = [row for row in rows if str(row.get("example_id")) in fair_ids]
        if not rows:
            continue
        summary: dict[str, Any] = {
            "model_id": model,
            "policy": policy,
            "error_cost": error_cost,
            "comparison_scope": comparison_scope,
            "n": len(rows),
            "n_questions": len({row["example_id"] for row in rows}),
        }
        aliases = {
            "realized_cost": "mean_cost",
            "regret": "regret",
            "unsafe_reliance": "unsafe_reliance",
            "verify": "verification_burden",
        }
        for index, (source_key, target_key) in enumerate(aliases.items()):
            eligible = (
                [row for row in rows if not bool(row["is_correct"])]
                if source_key == "unsafe_reliance"
                else rows
            )
            result = bootstrap_mean(
                eligible,
                source_key,
                n_resamples=n_resamples,
                confidence_level=confidence_level,
                seed=seed + index,
            )
            summary[target_key] = result.estimate
            summary[f"{target_key}_ci_lower"] = result.lower
            summary[f"{target_key}_ci_upper"] = result.upper
        output.append(summary)
    return output


def summarize_policy_differences(
    policy_rows: Iterable[Mapping[str, Any]],
    *,
    n_resamples: int = 5000,
    confidence_level: float = 0.95,
    seed: int = 2025,
) -> list[dict[str, Any]]:
    """Paired question-level cost differences for the primary comparisons."""
    rows = [dict(row) for row in policy_rows]
    output: list[dict[str, Any]] = []
    strata = sorted(
        {
            (str(row["model_id"]), float(row["error_cost"]))
            for row in rows
            if row.get("model_id") is not None and row.get("error_cost") is not None
        }
    )
    for model, error_cost in strata:
        subset = [
            row
            for row in rows
            if str(row.get("model_id")) == model
            and float(row.get("error_cost")) == error_cost
        ]
        for index, baseline in enumerate(("raw_confidence", "calibrated_confidence")):
            result = paired_policy_bootstrap(
                subset,
                left_policy="direct",
                right_policy=baseline,
                n_resamples=n_resamples,
                confidence_level=confidence_level,
                seed=seed + index,
            )
            if result.n_questions == 0:
                continue
            output.append(
                {
                    "model_id": model,
                    "error_cost": error_cost,
                    "comparison": f"direct_minus_{baseline}",
                    "mean_cost_difference": result.estimate,
                    "ci_lower": result.lower,
                    "ci_upper": result.upper,
                    "n_questions": result.n_questions,
                    "n_resamples": result.n_resamples,
                }
            )
    return output


def summarize_models(
    records: Iterable[Mapping[str, Any]],
    direct_rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    records_list = list(records)
    direct_list = list(direct_rows)
    answers_by_model: dict[str, dict[str, bool]] = defaultdict(dict)
    confidence_by_model: dict[str, dict[str, tuple[float, bool]]] = defaultdict(dict)
    for row in records_list:
        model = str(row.get("model_id"))
        example = str(row.get("example_id"))
        if row.get("stage") == "answer" and row.get("is_correct") is not None:
            answers_by_model[model][example] = bool(row["is_correct"])
    for row in direct_list:
        probability = _number(row.get("probability_correct"))
        if probability is not None:
            confidence_by_model[str(row["model_id"])][str(row["example_id"])] = (
                probability,
                bool(row["is_correct"]),
            )
    monotonic = monotonicity_from_records(
        direct_list,
        action_key="action",
        expected_error_costs=ERROR_COSTS,
    )
    monotonic_by_model: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in monotonic:
        monotonic_by_model[str(row["model_id"])].append(row)

    output: list[dict[str, Any]] = []
    models = sorted(
        set(answers_by_model) | {str(row["model_id"]) for row in direct_list}
    )
    for model in models:
        outcomes = list(answers_by_model[model].values())
        confidence = list(confidence_by_model[model].values())
        sequences = monotonic_by_model[model]
        output.append(
            {
                "model_id": model,
                "n_answers": len(outcomes),
                "answer_accuracy": (
                    sum(outcomes) / len(outcomes) if outcomes else float("nan")
                ),
                "n_confidence": len(confidence),
                "brier_score": brier_score(
                    [item[0] for item in confidence],
                    [item[1] for item in confidence],
                ),
                "ece_10_bin": expected_calibration_error(
                    [item[0] for item in confidence],
                    [item[1] for item in confidence],
                ),
                "n_monotonicity_sequences": len(sequences),
                "n_complete_monotonicity_sequences": sum(
                    bool(row["complete"]) for row in sequences
                ),
                "monotonicity_violation_rate": (
                    sum(bool(row["any_violation"]) for row in sequences)
                    / len(sequences)
                    if sequences
                    else float("nan")
                ),
                "adjacent_verify_to_rely_count": sum(
                    int(row["adjacent_verify_to_rely_count"]) for row in sequences
                ),
            }
        )
    return output


def summarize_domains(
    direct_rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, float], list[Mapping[str, Any]]] = defaultdict(list)
    for row in direct_rows:
        category = row.get("category")
        if category is None:
            continue
        groups[(str(row["model_id"]), str(category), float(row["error_cost"]))].append(
            row
        )
    output: list[dict[str, Any]] = []
    for (model, category, error_cost), rows in sorted(groups.items()):
        wrong = [row for row in rows if not bool(row["is_correct"])]
        output.append(
            {
                "model_id": model,
                "category": category,
                "error_cost": error_cost,
                "n": len(rows),
                "answer_accuracy": sum(bool(row["is_correct"]) for row in rows)
                / len(rows),
                "verify_rate": sum(bool(row["verify"]) for row in rows) / len(rows),
                "unsafe_reliance": (
                    sum(bool(row["unsafe_reliance"]) for row in wrong) / len(wrong)
                    if wrong
                    else float("nan")
                ),
                "mean_regret": sum(float(row["regret"]) for row in rows) / len(rows),
            }
        )
    return output


def build_diagnostics(
    direct_rows: Iterable[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Build internal diagnostics; these may contain benchmark question text."""
    rows = [dict(row) for row in direct_rows]
    monotonic = monotonicity_from_records(
        rows, action_key="action", expected_error_costs=ERROR_COSTS
    )
    violating = {
        (str(row["model_id"]), str(row["example_id"]))
        for row in monotonic
        if row["any_violation"]
    }
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model_id"]), str(row["example_id"]))].append(row)
    oververified_keys = {
        key
        for key, group in grouped.items()
        if group
        and all(bool(row["is_correct"]) and row["action"] == VERIFY for row in group)
    }
    confidence_examples: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        confidence_examples[(str(row["model_id"]), str(row["example_id"]))] = row
    return {
        "high_stakes_unsafe_reliance": [
            row
            for row in rows
            if not row["is_correct"]
            and row["action"] == RELY
            and float(row["error_cost"]) == max(ERROR_COSTS)
        ],
        "monotonicity_violations": [
            {
                **row,
                "question": next(
                    (
                        item.get("question")
                        for item in grouped[
                            (str(row["model_id"]), str(row["example_id"]))
                        ]
                        if item.get("question")
                    ),
                    None,
                ),
            }
            for row in monotonic
            if (str(row["model_id"]), str(row["example_id"])) in violating
        ],
        "direct_vs_confidence_disagreements": [
            row
            for row in rows
            if row.get("direct_vs_raw_confidence_disagreement")
            or row.get("direct_vs_calibrated_confidence_disagreement")
        ],
        "correct_verified_at_all_stakes": [
            row
            for key, group in grouped.items()
            if key in oververified_keys
            for row in group
        ],
        "confidence_extremes": [
            row
            for row in confidence_examples.values()
            if (
                not row["is_correct"]
                and _number(row.get("probability_correct")) is not None
                and float(row["probability_correct"]) >= 0.8
            )
            or (
                row["is_correct"]
                and _number(row.get("probability_correct")) is not None
                and float(row["probability_correct"]) <= 0.2
            )
        ],
    }


def _clean(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    return value


def _csv_value(value: Any) -> Any:
    value = _clean(value)
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True)
    if value is None:
        return ""
    return value


def write_csv(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = list(rows)
    fields: list[str] = []
    for row in data:
        for field in row:
            if field not in fields:
                fields.append(field)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in data:
                writer.writerow({field: _csv_value(row.get(field)) for field in fields})
    return destination


def write_json(path: str | Path, value: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(_clean(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def _format_cell(value: Any) -> str:
    cleaned = _clean(value)
    if cleaned is None:
        return "—"
    if isinstance(cleaned, float):
        return f"{cleaned:.3f}"
    return str(cleaned)


def write_markdown_table(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    columns: Sequence[str],
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| " + " | ".join(column.replace("_", " ") for column in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        cells = [
            _format_cell(row.get(column)).replace("|", "\\|") for column in columns
        ]
        lines.append("| " + " | ".join(cells) + " |")
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in value)


def write_latex_table(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    columns: Sequence[str],
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{tabular}{" + "l" * len(columns) + "}",
        r"\toprule",
        " & ".join(_latex_escape(column.replace("_", " ")) for column in columns)
        + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                _latex_escape(_format_cell(row.get(column))) for column in columns
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


def _public_direct_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    private_fields = {
        "question",
        "question_text",
        "choices",
        "raw_response",
        "prompt",
        "correct_label",
        "answer_label",
    }
    return [
        {key: value for key, value in row.items() if key not in private_fields}
        for row in rows
    ]


def _write_pilot_summary(
    path: Path,
    model_summary: Sequence[Mapping[str, Any]],
    decision_summary: Sequence[Mapping[str, Any]],
    policy_summary: Sequence[Mapping[str, Any]],
    policy_differences: Sequence[Mapping[str, Any]],
    request_coverage: Sequence[Mapping[str, Any]],
    usage_summary: Sequence[Mapping[str, Any]],
    warnings: Sequence[str],
) -> Path:
    lines = [
        "# Pilot evidence summary",
        "",
        (
            "This report presents observed evidence only; it does not assign an "
            "automatic GO/MODIFY/KILL conclusion."
        ),
        "",
        "## Coverage",
        "",
    ]
    if not model_summary:
        lines.append("- No complete model records were estimable.")
    for row in model_summary:
        lines.append(
            f"- {row['model_id']}: {row['n_answers']} scored answers; "
            f"accuracy {_format_cell(row['answer_accuracy'])}; "
            f"Brier {_format_cell(row['brier_score'])}; "
            f"monotonicity violation rate "
            f"{_format_cell(row['monotonicity_violation_rate'])}."
        )
    for row in request_coverage:
        lines.append(
            f"- Requests: {row['stage']} / {row['model_id']} / "
            f"{row['status']} = {row['count']}."
        )
    if usage_summary:
        lines.append(
            "- Observed provider cost from configured pricing metadata: "
            f"${sum(float(row['observed_cost_usd']) for row in usage_summary):.4f}."
        )
    lines.extend(["", "## Stake-sensitive evidence", ""])
    for row in decision_summary:
        lines.append(
            f"- {row['model_id']}, L={_format_cell(row['error_cost'])}: "
            f"VERIFY rate {_format_cell(row['verify_rate'])} "
            f"[{_format_cell(row['verify_rate_ci_lower'])}, "
            f"{_format_cell(row['verify_rate_ci_upper'])}]; unsafe reliance "
            f"{_format_cell(row['unsafe_reliance'])}; mean regret "
            f"{_format_cell(row['regret'])}."
        )
    lines.extend(
        [
            "",
            "## Policy evidence",
            "",
            (
                "Policy rows and paired examples are supplied in the machine-readable "
                "outputs. Calibrated-policy estimates use evaluation-partition "
                "examples only; their fitted labels come exclusively from calibration "
                "IDs."
            ),
        ]
    )
    if policy_summary:
        best_by_model_stake: dict[tuple[str, float], Mapping[str, Any]] = {}
        for row in policy_summary:
            key = (str(row["model_id"]), float(row["error_cost"]))
            cost = _number(row.get("mean_cost"))
            current = best_by_model_stake.get(key)
            if cost is not None and (
                current is None or cost < float(current["mean_cost"])
            ):
                best_by_model_stake[key] = row
        for (model, loss), row in sorted(best_by_model_stake.items()):
            lines.append(
                f"- Lowest observed mean cost for {model}, L={loss:g}: "
                f"{str(row['policy']).replace('_', ' ')} "
                f"({_format_cell(row['mean_cost'])}); this is descriptive, "
                "not a significance claim."
            )
    for row in policy_differences:
        lines.append(
            f"- {row['model_id']}, L={_format_cell(row['error_cost'])}, "
            f"{str(row['comparison']).replace('_', ' ')}: "
            f"{_format_cell(row['mean_cost_difference'])} "
            f"[{_format_cell(row['ci_lower'])}, "
            f"{_format_cell(row['ci_upper'])}], paired n="
            f"{row['n_questions']}."
        )
    if warnings:
        lines.extend(["", "## Data limitations", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    lines.extend(
        [
            "",
            "## Interpretation checklist",
            "",
            "- Examine effect sizes and confidence intervals, not only point estimates.",
            "- Check missingness and denominator counts before comparing policies.",
            "- Treat oracle results as non-deployable lower bounds.",
            "- Review internal diagnostics before making the research decision.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def analyze(
    source: str | Path | Iterable[Mapping[str, Any]],
    *,
    output_directory: str | Path = "results",
    paper_output_directory: str | Path | None = None,
    seed: int = 2025,
    calibration_fraction: float = 0.20,
    n_resamples: int = 5000,
    confidence_level: float = 0.95,
    make_plots: bool = True,
    pricing_per_million_tokens: Mapping[str, Mapping[str, float]] | None = None,
) -> AnalysisResult:
    """Run the full analysis and write diagnostics and paper-ready outputs."""
    records = load_records(source)
    request_coverage = (
        load_request_coverage(source) if isinstance(source, (str, Path)) else []
    )
    usage_summary = (
        load_usage_summary(source, pricing_per_million_tokens)
        if isinstance(source, (str, Path))
        else []
    )
    root = Path(output_directory)
    paper = (
        Path(paper_output_directory)
        if paper_output_directory is not None
        else root / "paper_outputs"
    )
    diagnostics_directory = root / "diagnostics"
    paper.mkdir(parents=True, exist_ok=True)
    diagnostics_directory.mkdir(parents=True, exist_ok=True)

    joined, splits, calibrator_metadata, warnings = join_stages(
        records,
        seed=seed,
        calibration_fraction=calibration_fraction,
    )
    direct_rows, policy_rows = score_joined_records(joined)
    decision_summary = summarize_decisions(
        direct_rows,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        seed=seed,
    )
    policy_summary = summarize_policies(
        policy_rows,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        seed=seed,
    )
    policy_differences = summarize_policy_differences(
        policy_rows,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        seed=seed,
    )
    model_summary = summarize_models(records, direct_rows)
    domain_summary = summarize_domains(direct_rows)
    diagnostics = build_diagnostics(direct_rows)

    calibration_directory = root / "calibration_splits"
    split_path, metadata_path = persist_calibration_artifacts(
        calibration_directory, splits, calibrator_metadata
    )
    paths: dict[str, Any] = {
        "calibration_splits": str(split_path),
        "calibrator_metadata": str(metadata_path),
        "diagnostics": {},
        "paper_outputs": {},
    }
    for name, rows in diagnostics.items():
        destination = write_csv(diagnostics_directory / f"{name}.csv", rows)
        paths["diagnostics"][name] = str(destination)

    public_direct = _public_direct_rows(direct_rows)
    public_files: dict[str, Any] = {
        "metrics_summary.csv": decision_summary,
        "metrics_summary.json": decision_summary,
        "policy_comparison.csv": policy_summary,
        "policy_comparison.json": policy_summary,
        "paired_policy_differences.csv": policy_differences,
        "paired_policy_differences.json": policy_differences,
        "model_summary.csv": model_summary,
        "model_summary.json": model_summary,
        "domain_summary.csv": domain_summary,
        "domain_summary.json": domain_summary,
        "request_coverage.csv": request_coverage,
        "request_coverage.json": request_coverage,
        "usage_summary.csv": usage_summary,
        "usage_summary.json": usage_summary,
        "scored_decisions.csv": public_direct,
        "scored_decisions.json": public_direct,
    }
    for filename, value in public_files.items():
        destination = paper / filename
        written = (
            write_json(destination, value)
            if destination.suffix == ".json"
            else write_csv(destination, value)
        )
        paths["paper_outputs"][filename] = str(written)

    table_specs = {
        "table_model_summary": (
            model_summary,
            (
                "model_id",
                "n_answers",
                "answer_accuracy",
                "brier_score",
                "monotonicity_violation_rate",
            ),
        ),
        "table_stake_results": (
            decision_summary,
            (
                "model_id",
                "error_cost",
                "verify_rate",
                "unsafe_reliance",
                "unnecessary_verification",
                "realized_cost",
                "regret",
                "direct_vs_calibrated_disagreement",
            ),
        ),
        "table_policy_comparison": (
            policy_summary,
            (
                "model_id",
                "policy",
                "error_cost",
                "mean_cost",
                "regret",
                "unsafe_reliance",
                "verification_burden",
            ),
        ),
        "table_paired_policy_differences": (
            policy_differences,
            (
                "model_id",
                "error_cost",
                "comparison",
                "mean_cost_difference",
                "ci_lower",
                "ci_upper",
                "n_questions",
            ),
        ),
    }
    for name, (rows, columns) in table_specs.items():
        markdown = write_markdown_table(paper / f"{name}.md", rows, columns)
        latex = write_latex_table(paper / f"{name}.tex", rows, columns)
        paths["paper_outputs"][f"{name}.md"] = str(markdown)
        paths["paper_outputs"][f"{name}.tex"] = str(latex)

    if make_plots:
        try:
            paths["figures"] = generate_core_figures(
                decision_summary, policy_summary, paper
            )
        except RuntimeError as exc:
            warnings.append(str(exc))
            paths["figures"] = {}
    summary_path = _write_pilot_summary(
        paper / "pilot_summary.md",
        model_summary,
        decision_summary,
        policy_summary,
        policy_differences,
        request_coverage,
        usage_summary,
        warnings,
    )
    paths["paper_outputs"]["pilot_summary.md"] = str(summary_path)
    write_json(
        paper / "analysis_manifest.json",
        {
            "seed": seed,
            "calibration_fraction": calibration_fraction,
            "n_resamples": n_resamples,
            "confidence_level": confidence_level,
            "verification_cost": VERIFICATION_COST,
            "error_costs": list(ERROR_COSTS),
            "n_input_records": len(records),
            "n_joined_decisions": len(joined),
            "warnings": warnings,
            "files": paths,
        },
    )
    paths["paper_outputs"]["analysis_manifest.json"] = str(
        paper / "analysis_manifest.json"
    )
    return AnalysisResult(
        direct_rows,
        policy_rows,
        model_summary,
        decision_summary,
        policy_summary,
        policy_differences,
        domain_summary,
        request_coverage,
        usage_summary,
        paths,
        warnings,
    )
