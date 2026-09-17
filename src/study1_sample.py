"""Deterministic Study 1 ID selection and read-only historical Stage-1/2 loading."""

from __future__ import annotations

import hashlib
import json
import random
import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT, load_models_config
from .datasets import load_local_sample
from .prompts import ANSWER_PROMPT_VERSION, CONFIDENCE_PROMPT_VERSION
from .schemas import BenchmarkExample
from .study1_schemas import (
    STUDY1_MODEL_ALIASES,
    Study1ExperimentConfig,
    load_study1_config,
)

PRIMARY_SEED = 20260917
REPEAT_SEED = 20260918
PRIMARY_SIZE = 100
REPEAT_SIZE = 20


def hash_id_list(ids: Sequence[str]) -> str:
    return hashlib.sha256(("\n".join(ids) + "\n").encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_v2b_examples(
    config: Study1ExperimentConfig | None = None,
) -> list[BenchmarkExample]:
    config = config or load_study1_config()
    sample_path = config.resolve_path(config.sample["source_sample_path"])
    manifest_path = config.resolve_path(config.sample["source_manifest_path"])
    examples = load_local_sample(
        sample_path,
        expected_size=int(config.sample["source_size"]),
        manifest_path=manifest_path,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    file_ids = [row.example_id for row in examples]
    if file_ids != manifest.get("selected_ids"):
        raise RuntimeError("V2-B JSONL order does not match manifest selected_ids")
    if manifest.get("sample_sha256") != file_sha256(sample_path):
        raise RuntimeError("V2-B sample hash does not match manifest")
    return examples


def select_ids(population: Sequence[str], *, seed: int, size: int) -> list[str]:
    if size > len(population):
        raise ValueError("cannot sample more IDs than the population")
    if len(set(population)) != len(population):
        raise ValueError("population contains duplicate IDs")
    return random.Random(seed).sample(list(population), size)


def select_study1_ids(
    config: Study1ExperimentConfig | None = None,
) -> tuple[list[str], list[str], dict[str, Any], dict[str, Any]]:
    """Sample 100 IDs then 20 repeats. Does not use outcomes or confidence."""
    config = config or load_study1_config()
    examples = load_v2b_examples(config)
    population = [row.example_id for row in examples]
    selected = select_ids(population, seed=PRIMARY_SEED, size=PRIMARY_SIZE)
    repeats = select_ids(selected, seed=REPEAT_SEED, size=REPEAT_SIZE)
    sample_path = config.resolve_path(config.sample["source_sample_path"])
    manifest_path = config.resolve_path(config.sample["source_manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    created_at = datetime.now(UTC).isoformat()
    primary_payload = {
        "study_id": config.study_id,
        "pilot_or_confirmatory": config.pilot_or_confirmatory,
        "selection_seed": PRIMARY_SEED,
        "sample_size": PRIMARY_SIZE,
        "source_sample_path": str(config.sample["source_sample_path"]),
        "source_manifest_path": str(config.sample["source_manifest_path"]),
        "source_sample_sha256": manifest.get("sample_sha256"),
        "source_file_sha256": file_sha256(sample_path),
        "canonical_order": "v2b_jsonl_file_order",
        "rng": "Python 3 random.Random(seed).sample(population, k)",
        "selection_uses_correctness": False,
        "selection_uses_historical_confidence": False,
        "selection_uses_historical_stage3": False,
        "selection_uses_category_performance": False,
        "ordered_selected_ids": selected,
        "selected_id_list_sha256": hash_id_list(selected),
        "repeat_subset_seed": REPEAT_SEED,
        "ordered_repeat_ids": repeats,
        "repeat_id_list_sha256": hash_id_list(repeats),
        "created_at": created_at,
    }
    repeat_payload = {
        "study_id": config.study_id,
        "parent_selected_id_list_sha256": primary_payload["selected_id_list_sha256"],
        "selection_seed": REPEAT_SEED,
        "sample_size": REPEAT_SIZE,
        "population": "the frozen 100 Study 1 primary IDs, in selected order",
        "rng": "Python 3 random.Random(seed).sample(population, k)",
        "selection_uses_study1_outcomes": False,
        "ordered_repeat_ids": repeats,
        "repeat_id_list_sha256": hash_id_list(repeats),
        "created_at": created_at,
    }
    return selected, repeats, primary_payload, repeat_payload


def write_study1_ids(
    config: Study1ExperimentConfig | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = config or load_study1_config()
    _, _, primary_payload, repeat_payload = select_study1_ids(config)
    primary_path = config.resolve_path(config.sample["selected_ids_path"])
    repeat_path = config.resolve_path(config.sample["repeat_ids_path"])
    primary_path.parent.mkdir(parents=True, exist_ok=True)
    primary_path.write_text(
        json.dumps(primary_payload, indent=2) + "\n", encoding="utf-8"
    )
    repeat_path.write_text(
        json.dumps(repeat_payload, indent=2) + "\n", encoding="utf-8"
    )
    return primary_payload, repeat_payload


def load_frozen_ids(
    config: Study1ExperimentConfig | None = None,
) -> tuple[list[str], list[str], str, str]:
    config = config or load_study1_config()
    primary = json.loads(
        config.resolve_path(config.sample["selected_ids_path"]).read_text(
            encoding="utf-8"
        )
    )
    repeat = json.loads(
        config.resolve_path(config.sample["repeat_ids_path"]).read_text(
            encoding="utf-8"
        )
    )
    selected = list(primary["ordered_selected_ids"])
    repeats = list(repeat["ordered_repeat_ids"])
    if len(selected) != PRIMARY_SIZE:
        raise RuntimeError(f"expected 100 primary IDs, found {len(selected)}")
    if len(repeats) != REPEAT_SIZE:
        raise RuntimeError(f"expected 20 repeat IDs, found {len(repeats)}")
    if hash_id_list(selected) != primary["selected_id_list_sha256"]:
        raise RuntimeError("primary ID list hash mismatch")
    if hash_id_list(repeats) != repeat["repeat_id_list_sha256"]:
        raise RuntimeError("repeat ID list hash mismatch")
    if not set(repeats).issubset(set(selected)):
        raise RuntimeError("repeat IDs are not a subset of primary IDs")
    recomputed, recomputed_repeats, _, _ = select_study1_ids(config)
    if selected != recomputed or repeats != recomputed_repeats:
        raise RuntimeError("frozen IDs do not match deterministic recompute")
    return (
        selected,
        repeats,
        primary["selected_id_list_sha256"],
        repeat["repeat_id_list_sha256"],
    )


def _open_historical_readonly(path: Path) -> sqlite3.Connection:
    if not path.exists():
        raise FileNotFoundError(f"historical V2 sqlite not found: {path}")
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _load_historical_stage12_on_connection(
    connection: sqlite3.Connection,
    *,
    example_id: str,
    model_alias: str,
    requested_model_id: str,
) -> dict[str, Any]:
    answer_rows = connection.execute(
        """
        SELECT record_json, request_key FROM requests
        WHERE stage = 'answer'
          AND dataset = 'mmlu_pro'
          AND example_id = ?
          AND model_alias = ?
          AND requested_model_id = ?
          AND prompt_version = ?
          AND status = 'success'
        """,
        (
            example_id,
            model_alias,
            requested_model_id,
            ANSWER_PROMPT_VERSION,
        ),
    ).fetchall()
    confidence_rows = connection.execute(
        """
        SELECT record_json, request_key FROM requests
        WHERE stage = 'confidence'
          AND dataset = 'mmlu_pro'
          AND example_id = ?
          AND model_alias = ?
          AND requested_model_id = ?
          AND prompt_version = ?
          AND status = 'success'
        """,
        (
            example_id,
            model_alias,
            requested_model_id,
            CONFIDENCE_PROMPT_VERSION,
        ),
    ).fetchall()
    if len(answer_rows) != 1:
        raise RuntimeError(
            f"expected 1 historical answer for {example_id}/{model_alias}, "
            f"found {len(answer_rows)}"
        )
    if len(confidence_rows) != 1:
        raise RuntimeError(
            f"expected 1 historical confidence for {example_id}/{model_alias}, "
            f"found {len(confidence_rows)}"
        )
    answer = json.loads(answer_rows[0]["record_json"])
    confidence = json.loads(confidence_rows[0]["record_json"])
    if confidence.get("frozen_answer_label") != answer.get("answer_label"):
        raise RuntimeError(
            f"historical Stage-2 answer mismatch for {example_id}/{model_alias}"
        )
    return {
        "question_id": example_id,
        "model_alias": model_alias,
        "frozen_answer": str(answer["answer_label"]),
        "stage1_correct": bool(answer["is_correct"]),
        "reported_confidence": float(confidence["probability_correct"]),
        "historical_answer_request_key": str(
            answer.get("request_key") or answer_rows[0]["request_key"]
        ),
        "historical_confidence_request_key": str(
            confidence.get("request_key") or confidence_rows[0]["request_key"]
        ),
        "historical_probability_correct_field": "probability_correct",
    }


def load_historical_stage12(
    *,
    example_id: str,
    model_alias: str,
    sqlite_path: Path,
    requested_model_id: str,
) -> dict[str, Any]:
    """Read frozen V2 Stage-1/2 fields. Never writes."""
    connection = _open_historical_readonly(sqlite_path)
    try:
        return _load_historical_stage12_on_connection(
            connection,
            example_id=example_id,
            model_alias=model_alias,
            requested_model_id=requested_model_id,
        )
    finally:
        connection.close()


def load_historical_bundle(
    question_ids: Sequence[str],
    config: Study1ExperimentConfig | None = None,
) -> dict[tuple[str, str], dict[str, Any]]:
    config = config or load_study1_config()
    models = load_models_config(config.resolve_path(config.models_config_path))
    sqlite_path = config.historical_sqlite()
    bundle: dict[tuple[str, str], dict[str, Any]] = {}
    connection = _open_historical_readonly(sqlite_path)
    try:
        for question_id in question_ids:
            for alias in STUDY1_MODEL_ALIASES:
                spec = models.models[alias]
                bundle[(question_id, alias)] = _load_historical_stage12_on_connection(
                    connection,
                    example_id=question_id,
                    model_alias=alias,
                    requested_model_id=spec.api_model,
                )
    finally:
        connection.close()
    return bundle


def assert_not_v2_write_target(path: Path, config: Study1ExperimentConfig) -> None:
    historical = config.historical_sqlite().resolve()
    target = path.resolve()
    if target == historical:
        raise RuntimeError("Study 1 refuses to write to historical V2 sqlite")
    if target.name == "v2.sqlite3":
        raise RuntimeError("Study 1 refuses any checkpoint named v2.sqlite3")
