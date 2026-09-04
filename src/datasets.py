"""Deterministic MMLU-Pro loading and category-stratified sampling."""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT, ExperimentConfig, load_experiment_config
from .schemas import BenchmarkExample


def _answer_label(value: Any, number_of_choices: int) -> str:
    if isinstance(value, bool):
        raise TypeError("boolean answer values are invalid")
    if isinstance(value, int):
        index = value
    elif isinstance(value, str):
        stripped = value.strip().upper()
        if len(stripped) == 1 and "A" <= stripped <= "Z":
            index = ord(stripped) - ord("A")
        elif stripped.isdigit():
            index = int(stripped)
        else:
            raise ValueError(f"unsupported answer value: {value!r}")
    else:
        raise TypeError(f"unsupported answer value: {value!r}")
    if not 0 <= index < number_of_choices:
        raise ValueError(f"answer index {index} is outside the available choices")
    return chr(ord("A") + index)


def _normalise_choices(value: Any) -> dict[str, str]:
    if isinstance(value, Mapping):
        items = [(str(label).upper(), str(text)) for label, text in value.items()]
        items.sort(key=lambda item: item[0])
        choices = dict(items)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        choices = {chr(ord("A") + index): str(text) for index, text in enumerate(value)}
    else:
        raise TypeError("MMLU-Pro choices/options must be a mapping or sequence")
    if len(choices) > 26:
        raise ValueError("more than 26 choices are not supported")
    return choices


def normalise_mmlu_pro_row(
    row: Mapping[str, Any],
    *,
    source_index: int,
    revision: str,
    split: str = "test",
) -> BenchmarkExample:
    """Convert one source row while deliberately dropping `cot_content`."""
    choices = _normalise_choices(row.get("options", row.get("choices")))
    raw_answer = row.get("answer", row.get("answer_index"))
    if raw_answer is None:
        raise ValueError(f"source row {source_index} has no answer")

    source_id = next(
        (
            str(row[key])
            for key in ("example_id", "question_id", "id")
            if row.get(key) not in (None, "")
        ),
        f"{source_index:05d}",
    )
    return BenchmarkExample(
        example_id=f"mmlu_pro:{split}:{source_id}",
        dataset_name="mmlu_pro",
        category=str(row.get("category", "")).strip(),
        question=str(row.get("question", "")),
        choices=choices,
        correct_label=_answer_label(raw_answer, len(choices)),
        split=split,
        dataset_revision=revision,
        source_index=source_index,
    )


def load_mmlu_pro_source(config: ExperimentConfig) -> list[BenchmarkExample]:
    """Load the public pinned Hugging Face split; no provider API key is used."""
    dataset_config = config.dataset
    cache_dir = config.resolve_path(dataset_config.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Keep every downloaded artifact inside the project workspace. This also
    # makes the public-data step portable in restricted/CI environments.
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_dir / "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_dir / "datasets"))
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face datasets is required when no local sample exists"
        ) from exc

    source = load_dataset(
        dataset_config.hf_repo,
        revision=dataset_config.revision,
        split=dataset_config.split,
        cache_dir=str(cache_dir),
    )
    examples = [
        normalise_mmlu_pro_row(
            row,
            source_index=index,
            revision=dataset_config.revision,
            split=dataset_config.split,
        )
        for index, row in enumerate(source)
    ]
    ids = [example.example_id for example in examples]
    if len(ids) != len(set(ids)):
        raise ValueError("MMLU-Pro source produced duplicate stable example IDs")
    return examples


def allocate_category_quotas(
    category_sizes: Mapping[str, int], target_size: int, seed: int
) -> dict[str, int]:
    """Allocate a target as evenly as category capacities permit."""
    if target_size <= 0:
        raise ValueError("target_size must be positive")
    if any(size < 0 for size in category_sizes.values()):
        raise ValueError("category sizes cannot be negative")
    if target_size > sum(category_sizes.values()):
        raise ValueError("target_size exceeds the available examples")

    order = sorted(category for category, size in category_sizes.items() if size)
    random.Random(seed).shuffle(order)
    quotas = {category: 0 for category in sorted(category_sizes)}
    remaining = target_size
    while remaining:
        made_progress = False
        for category in order:
            if quotas[category] >= category_sizes[category]:
                continue
            quotas[category] += 1
            remaining -= 1
            made_progress = True
            if remaining == 0:
                break
        if not made_progress:
            raise RuntimeError("unable to allocate the requested sample")
    return quotas


def _category_seed(seed: int, category: str) -> int:
    digest = hashlib.sha256(f"{seed}\0{category}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def select_category_stratified(
    examples: Iterable[BenchmarkExample], target_size: int, seed: int
) -> tuple[list[BenchmarkExample], dict[str, int]]:
    """Select exact examples deterministically within balanced category quotas."""
    groups: dict[str, list[BenchmarkExample]] = defaultdict(list)
    for example in examples:
        groups[example.category].append(example)
    if not groups:
        raise ValueError("cannot sample from an empty dataset")

    quotas = allocate_category_quotas(
        {category: len(group) for category, group in groups.items()},
        target_size,
        seed,
    )
    selected: list[BenchmarkExample] = []
    for category in sorted(groups):
        group = sorted(groups[category], key=lambda example: example.example_id)
        rng = random.Random(_category_seed(seed, category))
        selected.extend(rng.sample(group, quotas[category]))
    selected.sort(key=lambda example: example.source_index)

    if len(selected) != target_size:
        raise RuntimeError("stratified sampler did not produce the exact target size")
    if len({example.example_id for example in selected}) != target_size:
        raise RuntimeError("stratified sampler selected duplicate IDs")
    return selected, quotas


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _sample_bytes(examples: Sequence[BenchmarkExample]) -> bytes:
    lines = [example.model_dump_json() for example in examples]
    return ("\n".join(lines) + "\n").encode("utf-8")


def write_pilot_sample(
    examples: Sequence[BenchmarkExample],
    *,
    sample_path: Path,
    manifest_path: Path,
    config: ExperimentConfig,
    quotas: Mapping[str, int],
) -> None:
    """Atomically replace the exact JSONL sample and its category manifest."""
    sample_content = _sample_bytes(examples)
    counts = Counter(example.category for example in examples)
    selected_ids = {
        category: [
            example.example_id for example in examples if example.category == category
        ]
        for category in sorted(counts)
    }
    manifest = {
        "dataset_name": config.dataset.name,
        "hf_repo": config.dataset.hf_repo,
        "split": config.dataset.split,
        "revision": config.dataset.revision,
        "seed": config.seed,
        "sample_size": len(examples),
        "stratify_by": config.dataset.stratify_by,
        "category_counts": dict(sorted(counts.items())),
        "category_quotas": dict(sorted(quotas.items())),
        "selected_ids_by_category": selected_ids,
        "sample_sha256": hashlib.sha256(sample_content).hexdigest(),
        "created_at": datetime.now(UTC).isoformat(),
    }
    manifest_content = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    _atomic_write(sample_path, sample_content)
    _atomic_write(manifest_path, manifest_content)


def load_local_sample(
    sample_path: Path,
    *,
    expected_size: int | None = None,
    expected_revision: str | None = None,
    manifest_path: Path | None = None,
) -> list[BenchmarkExample]:
    """Load and validate an already-persisted sample without network access."""
    content = sample_path.read_bytes()
    examples: list[BenchmarkExample] = []
    for line_number, line in enumerate(content.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            examples.append(BenchmarkExample.model_validate_json(line))
        except ValueError as exc:
            raise ValueError(
                f"invalid sample record at {sample_path}:{line_number}"
            ) from exc

    if expected_size is not None and len(examples) != expected_size:
        raise ValueError(
            f"local sample has {len(examples)} rows; expected {expected_size}"
        )
    if len({example.example_id for example in examples}) != len(examples):
        raise ValueError("local sample contains duplicate example IDs")
    if expected_revision is not None and any(
        example.dataset_revision != expected_revision for example in examples
    ):
        raise ValueError("local sample does not match the configured revision")

    if manifest_path is not None and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("sample_sha256") != hashlib.sha256(content).hexdigest():
            raise ValueError("local sample checksum does not match its manifest")
        if (
            expected_revision is not None
            and manifest.get("revision") != expected_revision
        ):
            raise ValueError("local manifest does not match the configured revision")
        actual_counts = dict(sorted(Counter(e.category for e in examples).items()))
        if manifest.get("category_counts") != actual_counts:
            raise ValueError("local sample category counts do not match its manifest")
    return examples


def load_or_create_pilot_sample(
    config: ExperimentConfig | None = None,
    *,
    project_root: Path = PROJECT_ROOT,
    force_resample: bool = False,
) -> list[BenchmarkExample]:
    """Prefer the frozen local sample; otherwise download, select, and persist it."""
    config = config or load_experiment_config()
    sample_path = config.resolve_path(config.dataset.pilot_sample_path, project_root)
    manifest_path = config.resolve_path(
        config.dataset.pilot_manifest_path, project_root
    )
    if sample_path.exists() and not force_resample:
        return load_local_sample(
            sample_path,
            expected_size=config.dataset.pilot_size,
            expected_revision=config.dataset.revision,
            manifest_path=manifest_path,
        )

    source = load_mmlu_pro_source(config)
    selected, quotas = select_category_stratified(
        source, config.dataset.pilot_size, config.seed
    )
    write_pilot_sample(
        selected,
        sample_path=sample_path,
        manifest_path=manifest_path,
        config=config,
        quotas=quotas,
    )
    return selected
