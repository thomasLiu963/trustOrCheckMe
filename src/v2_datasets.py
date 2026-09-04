"""Deterministic V2-A, V2-B, and robustness sample construction."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

from .config import V2ExperimentConfig, load_v2_experiment_config
from .datasets import (
    load_local_sample,
    load_mmlu_pro_source,
    select_category_stratified,
)
from .schemas import BenchmarkExample


def _sample_bytes(examples: Sequence[BenchmarkExample]) -> bytes:
    return (
        "\n".join(example.model_dump_json() for example in examples) + "\n"
    ).encode("utf-8")


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_v2_sample(
    examples: Sequence[BenchmarkExample],
    *,
    sample_path: Path,
    manifest_path: Path,
    config: V2ExperimentConfig,
    sample_name: str,
    parent_ids: Sequence[str],
) -> None:
    content = _sample_bytes(examples)
    counts = dict(sorted(Counter(row.category for row in examples).items()))
    manifest = {
        "experiment_version": "v2",
        "sample_name": sample_name,
        "dataset_name": config.dataset.name,
        "hf_repo": config.dataset.hf_repo,
        "split": config.dataset.split,
        "revision": config.dataset.revision,
        "seed": config.seed,
        "sample_size": len(examples),
        "stratify_by": config.dataset.stratify_by,
        "category_counts": counts,
        "sample_sha256": hashlib.sha256(content).hexdigest(),
        "selected_ids": [row.example_id for row in examples],
        "required_parent_ids": list(parent_ids),
        "selection_uses_observed_results": False,
        "created_at": datetime.now(UTC).isoformat(),
    }
    _atomic_write(sample_path, content)
    _atomic_write(
        manifest_path,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def load_v2_sample(
    config: V2ExperimentConfig,
    sample: str,
    *,
    limit: int | None = None,
) -> list[BenchmarkExample]:
    paths = {
        "v2a": (config.dataset.v2a_sample_path, config.dataset.v2a_size),
        "v2b": (config.dataset.v2b_sample_path, config.dataset.v2b_size),
        "robustness": (
            config.dataset.robustness_sample_path,
            config.dataset.robustness_size,
        ),
    }
    if sample not in paths:
        raise ValueError("sample must be v2a, v2b, or robustness")
    relative, expected = paths[sample]
    path = config.resolve_path(relative)
    manifest = config.resolve_path(
        config.dataset.manifest_directory / f"mmlu_pro_{sample}_manifest.json"
    )
    if not path.exists():
        raise FileNotFoundError(f"V2 sample not found: {path}")
    rows = load_local_sample(
        path,
        expected_size=expected,
        expected_revision=config.dataset.revision,
        manifest_path=None,
    )
    if manifest.exists():
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
        if metadata.get("sample_sha256") != hashlib.sha256(
            path.read_bytes()
        ).hexdigest():
            raise ValueError("V2 sample checksum does not match its manifest")
    if limit is not None:
        if limit < 1 or limit > len(rows):
            raise ValueError(f"limit must be between 1 and {len(rows)}")
        return rows[:limit]
    return rows


def prepare_v2_sample(
    size: int,
    config: V2ExperimentConfig | None = None,
) -> list[BenchmarkExample]:
    """Prepare V2-A or V2-B while forcing inclusion of the V1 sample."""
    config = config or load_v2_experiment_config()
    if size not in {config.dataset.v2a_size, config.dataset.v2b_size}:
        raise ValueError(
            f"V2 sample size must be {config.dataset.v2a_size} or "
            f"{config.dataset.v2b_size}"
        )
    v1 = load_local_sample(
        config.resolve_path(config.dataset.v1_sample_path),
        expected_size=config.dataset.v2a_size,
        expected_revision=config.dataset.revision,
    )
    v1_ids = {row.example_id for row in v1}

    if size == config.dataset.v2a_size:
        selected = list(v1)
        sample_path = config.resolve_path(config.dataset.v2a_sample_path)
        sample_name = "v2a"
    else:
        source = load_mmlu_pro_source(config)  # type: ignore[arg-type]
        remaining = [row for row in source if row.example_id not in v1_ids]
        additions, _ = select_category_stratified(
            remaining, size - len(v1), config.seed + 1
        )
        selected = sorted([*v1, *additions], key=lambda row: row.source_index)
        sample_path = config.resolve_path(config.dataset.v2b_sample_path)
        sample_name = "v2b"

    manifest_path = config.resolve_path(
        config.dataset.manifest_directory
        / f"mmlu_pro_{sample_name}_manifest.json"
    )
    _write_v2_sample(
        selected,
        sample_path=sample_path,
        manifest_path=manifest_path,
        config=config,
        sample_name=sample_name,
        parent_ids=sorted(v1_ids),
    )
    if size == config.dataset.v2b_size:
        robustness, _ = select_category_stratified(
            selected, config.dataset.robustness_size, config.seed + 2
        )
        _write_v2_sample(
            robustness,
            sample_path=config.resolve_path(config.dataset.robustness_sample_path),
            manifest_path=config.resolve_path(
                config.dataset.manifest_directory
                / "mmlu_pro_robustness_manifest.json"
            ),
            config=config,
            sample_name="robustness",
            parent_ids=[],
        )
    return selected
