"""Crash-safe SQLite checkpoint storage for experiment requests and attempts."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import sqlite3
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Self


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON encode {type(value).__name__}")


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        default=_jsonable,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def deterministic_request_key(
    *,
    stage: str,
    dataset: str,
    example_id: str,
    model_id: str,
    prompt_version: str,
    stake: Any = None,
) -> str:
    """Return a stable, unambiguous key for one scientific request."""
    identity = {
        "dataset": str(dataset),
        "example_id": str(example_id),
        "model_id": str(model_id),
        "prompt_version": str(prompt_version),
        "stage": str(stage),
        "stake": stake,
    }
    digest = hashlib.sha256(canonical_json(identity).encode("utf-8")).hexdigest()
    return f"{stage}:{digest}"


def redact_secrets(text: str) -> str:
    """Remove configured credentials and common API-key forms from text."""
    redacted = text
    for name, value in os.environ.items():
        normalized = name.upper()
        if value and any(
            marker in normalized
            for marker in ("API_KEY", "TOKEN", "SECRET", "PASSWORD")
        ):
            redacted = redacted.replace(value, "<redacted>")
    redacted = re.sub(
        r"(?i)(authorization|bearer|api[-_ ]?key)\s*[:=]?\s+\S+",
        r"\1 <redacted>",
        redacted,
    )
    redacted = re.sub(
        r"\b(?:sk-(?:proj-)?|sk-ant-)[A-Za-z0-9_-]{12,}\b",
        "<redacted>",
        redacted,
    )
    return redacted


def sanitize_payload(value: Any) -> Any:
    """Recursively redact fields that could contain credentials."""
    sensitive = {
        "api_key",
        "apikey",
        "authorization",
        "auth",
        "access_token",
        "token",
        "secret",
        "password",
    }
    if isinstance(value, Mapping):
        return {
            str(key): (
                "<redacted>"
                if (
                    str(key).lower().replace("-", "_") in sensitive
                    or str(key).lower().replace("-", "_").endswith("_api_key")
                    or "secret" in str(key).lower()
                    or "password" in str(key).lower()
                )
                else sanitize_payload(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [sanitize_payload(item) for item in value]
    if isinstance(value, str):
        return redact_secrets(value)
    return value


class CheckpointStore:
    """Thread-safe SQLite store with a commit after every state mutation."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            self.path,
            timeout=30.0,
            isolation_level=None,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        with self._lock:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=FULL")
            self._connection.execute("PRAGMA foreign_keys=ON")
            self._connection.execute("PRAGMA busy_timeout=30000")
            self._create_schema()

    def _create_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS requests (
                request_key TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                stage TEXT NOT NULL,
                dataset TEXT NOT NULL,
                example_id TEXT NOT NULL,
                model_alias TEXT NOT NULL,
                requested_model_id TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                stake_json TEXT,
                status TEXT NOT NULL
                    CHECK(status IN ('pending', 'success', 'failed')),
                record_json TEXT,
                error TEXT,
                attempt_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                completed_at TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_requests_lookup
                ON requests(stage, dataset, example_id, model_alias, status);
            CREATE INDEX IF NOT EXISTS idx_requests_run
                ON requests(run_id, stage, status);

            CREATE TABLE IF NOT EXISTS attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_key TEXT NOT NULL,
                attempt_number INTEGER NOT NULL,
                attempt_kind TEXT NOT NULL,
                provider TEXT,
                requested_model_id TEXT,
                returned_model_id TEXT,
                started_at TEXT NOT NULL,
                finished_at TEXT NOT NULL,
                latency_seconds REAL,
                success INTEGER NOT NULL,
                raw_output TEXT,
                error_type TEXT,
                error_message TEXT,
                parse_error TEXT,
                input_tokens INTEGER,
                output_tokens INTEGER,
                total_tokens INTEGER,
                finish_reason TEXT,
                refusal INTEGER,
                sanitized_payload_json TEXT,
                FOREIGN KEY(request_key) REFERENCES requests(request_key),
                UNIQUE(request_key, attempt_number)
            );

            CREATE INDEX IF NOT EXISTS idx_attempts_request
                ON attempts(request_key, attempt_number);

            CREATE TABLE IF NOT EXISTS manifests (
                run_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                manifest_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                yield self._connection
            except BaseException:
                self._connection.rollback()
                raise
            else:
                self._connection.commit()

    def register_request(
        self,
        *,
        request_key: str,
        run_id: str,
        stage: str,
        dataset: str,
        example_id: str,
        model_alias: str,
        requested_model_id: str,
        prompt_version: str,
        stake: Any = None,
    ) -> None:
        now = utc_now()
        with self._write() as connection:
            connection.execute(
                """
                INSERT INTO requests (
                    request_key, run_id, stage, dataset, example_id,
                    model_alias, requested_model_id, prompt_version,
                    stake_json, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)
                ON CONFLICT(request_key) DO NOTHING
                """,
                (
                    request_key,
                    run_id,
                    stage,
                    dataset,
                    example_id,
                    model_alias,
                    requested_model_id,
                    prompt_version,
                    canonical_json(stake) if stake is not None else None,
                    now,
                    now,
                ),
            )

    def request_status(self, request_key: str) -> str | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT status FROM requests WHERE request_key = ?",
                (request_key,),
            ).fetchone()
        return str(row["status"]) if row else None

    def should_run(self, request_key: str, *, retry_failed: bool = False) -> bool:
        status = self.request_status(request_key)
        if status == "success":
            return False
        if status == "failed":
            return retry_failed
        return True

    def record_attempt(
        self,
        *,
        request_key: str,
        attempt_kind: str,
        success: bool,
        started_at: str | None = None,
        finished_at: str | None = None,
        provider: str | None = None,
        requested_model_id: str | None = None,
        returned_model_id: str | None = None,
        latency_seconds: float | None = None,
        raw_output: str | None = None,
        error: BaseException | str | None = None,
        parse_error: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        total_tokens: int | None = None,
        finish_reason: str | None = None,
        refusal: bool | None = None,
        sanitized_payload: Mapping[str, Any] | None = None,
    ) -> int:
        error_type = type(error).__name__ if isinstance(error, BaseException) else None
        error_message = redact_secrets(str(error)) if error is not None else None
        with self._write() as connection:
            row = connection.execute(
                "SELECT COALESCE(MAX(attempt_number), 0) + 1 AS number "
                "FROM attempts WHERE request_key = ?",
                (request_key,),
            ).fetchone()
            attempt_number = int(row["number"])
            connection.execute(
                """
                INSERT INTO attempts (
                    request_key, attempt_number, attempt_kind, provider,
                    requested_model_id, returned_model_id, started_at,
                    finished_at, latency_seconds, success, raw_output,
                    error_type, error_message, parse_error, input_tokens,
                    output_tokens, total_tokens, finish_reason, refusal,
                    sanitized_payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request_key,
                    attempt_number,
                    attempt_kind,
                    provider,
                    requested_model_id,
                    returned_model_id,
                    started_at or utc_now(),
                    finished_at or utc_now(),
                    latency_seconds,
                    int(success),
                    redact_secrets(raw_output) if raw_output is not None else None,
                    error_type,
                    error_message,
                    parse_error,
                    input_tokens,
                    output_tokens,
                    total_tokens,
                    finish_reason,
                    None if refusal is None else int(refusal),
                    (
                        canonical_json(sanitize_payload(sanitized_payload))
                        if sanitized_payload is not None
                        else None
                    ),
                ),
            )
            connection.execute(
                """
                UPDATE requests
                SET attempt_count = attempt_count + 1, updated_at = ?
                WHERE request_key = ?
                """,
                (utc_now(), request_key),
            )
        return attempt_number

    def mark_success(self, request_key: str, record: Any) -> None:
        now = utc_now()
        with self._write() as connection:
            cursor = connection.execute(
                """
                UPDATE requests
                SET status = 'success', record_json = ?, error = NULL,
                    updated_at = ?, completed_at = ?
                WHERE request_key = ?
                """,
                (
                    canonical_json(
                        sanitize_payload(json.loads(canonical_json(record)))
                    ),
                    now,
                    now,
                    request_key,
                ),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Unknown request key: {request_key}")

    def mark_failed(self, request_key: str, error: BaseException | str) -> None:
        now = utc_now()
        with self._write() as connection:
            cursor = connection.execute(
                """
                UPDATE requests
                SET status = 'failed', error = ?, record_json = NULL,
                    updated_at = ?, completed_at = ?
                WHERE request_key = ?
                """,
                (redact_secrets(str(error)), now, now, request_key),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Unknown request key: {request_key}")

    def get_record(self, request_key: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT record_json FROM requests
                WHERE request_key = ? AND status = 'success'
                """,
                (request_key,),
            ).fetchone()
        return json.loads(row["record_json"]) if row and row["record_json"] else None

    def find_success(
        self,
        *,
        stage: str,
        dataset: str,
        example_id: str,
        model_alias: str,
        requested_model_id: str | None = None,
        prompt_version: str | None = None,
    ) -> dict[str, Any] | None:
        """Find the sole compatible upstream result, failing on ambiguity."""
        clauses = [
            "stage = ?",
            "dataset = ?",
            "example_id = ?",
            "model_alias = ?",
            "status = 'success'",
        ]
        parameters: list[Any] = [stage, dataset, example_id, model_alias]
        if requested_model_id is not None:
            clauses.append("requested_model_id = ?")
            parameters.append(requested_model_id)
        if prompt_version is not None:
            clauses.append("prompt_version = ?")
            parameters.append(prompt_version)
        with self._lock:
            rows = self._connection.execute(
                "SELECT record_json FROM requests WHERE "
                + " AND ".join(clauses)
                + " ORDER BY completed_at DESC",
                parameters,
            ).fetchall()
        if not rows:
            return None
        if len(rows) > 1:
            raise RuntimeError(
                "Multiple completed upstream records match; use a fresh store or "
                "remove incompatible prompt-version results."
            )
        return json.loads(rows[0]["record_json"])

    def upsert_manifest(
        self,
        run_id: str,
        manifest: Mapping[str, Any],
        *,
        status: str = "running",
    ) -> None:
        now = utc_now()
        safe_manifest = sanitize_payload(dict(manifest))
        with self._write() as connection:
            connection.execute(
                """
                INSERT INTO manifests (
                    run_id, status, manifest_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    status = excluded.status,
                    manifest_json = excluded.manifest_json,
                    updated_at = excluded.updated_at
                """,
                (run_id, status, canonical_json(safe_manifest), now, now),
            )

    def get_manifest(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT manifest_json FROM manifests WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        return json.loads(row["manifest_json"]) if row else None

    def counts(self, *, run_id: str | None = None) -> dict[str, int]:
        query = "SELECT status, COUNT(*) AS count FROM requests"
        parameters: Sequence[Any] = ()
        if run_id is not None:
            query += " WHERE run_id = ?"
            parameters = (run_id,)
        query += " GROUP BY status"
        with self._lock:
            rows = self._connection.execute(query, parameters).fetchall()
        counts = {"pending": 0, "success": 0, "failed": 0}
        counts.update({str(row["status"]): int(row["count"]) for row in rows})
        return counts

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
