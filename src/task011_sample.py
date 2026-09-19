"""Deterministic Task 011 sampling: excluded pilot, main 500, repeat 100."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .study1_smoke import _git_commit
from .task011_benchmark import build_eligible_pool, lcb_repo_commit, load_release_v6_rows
from .task011_common import (
    DATASET_NAME,
    ENDPOINTS,
    LCB_COMMIT_PIN,
    LCB_HF_LITE,
    LCB_RELEASE,
    MAIN_N,
    MAIN_SEED,
    MIN_ELIGIBLE,
    MIN_HARD_ONLY,
    PILOT_N,
    PILOT_SEED,
    REPEAT_N,
    REPEAT_SEED,
    RETURN_DIR,
    SAMPLE_DIR,
    SCORE_CONDITIONS,
    json_dump,
    write_csv,
)
from .task011_prompts import template_hashes

DIFFICULTY_LADDERS = ("medium_hard", "hard_only", "medium_only")


def _id_hash(ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def _stratum_key(item: Mapping[str, Any]) -> str:
    return "|".join(
        [
            str(item.get("difficulty") or "").lower(),
            str(item.get("platform") or "").lower(),
            str(item.get("contest_date") or "")[:7],
        ]
    )


def _allocate(counts: Mapping[str, int], n: int) -> dict[str, int]:
    keys = sorted(counts)
    total = sum(counts.values())
    if total <= 0:
        raise ValueError("empty allocation counts")
    raw = {key: n * counts[key] / total for key in keys}
    alloc = {key: min(counts[key], int(raw[key])) for key in keys}
    leftover = n - sum(alloc.values())
    order = sorted(
        keys,
        key=lambda key: (raw[key] - alloc[key], counts[key], key),
        reverse=True,
    )
    guard = 0
    i = 0
    while leftover > 0 and guard < n * 20:
        key = order[i % len(order)]
        if alloc[key] < counts[key]:
            alloc[key] += 1
            leftover -= 1
        i += 1
        guard += 1
    if leftover != 0:
        raise RuntimeError(f"could not allocate {n} from counts {dict(counts)}")
    return alloc


def stratified_sample(
    items: Sequence[Mapping[str, Any]],
    n: int,
    seed: int,
) -> list[dict[str, Any]]:
    if n > len(items):
        raise ValueError(f"cannot sample {n} from {len(items)}")
    by_diff: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_diff[str(item.get("difficulty") or "").lower()].append(dict(item))
    alloc_diff = _allocate({key: len(rows) for key, rows in by_diff.items()}, n)
    rng = random.Random(seed)
    chosen: list[dict[str, Any]] = []
    for diff in sorted(by_diff):
        take_diff = int(alloc_diff.get(diff, 0))
        if take_diff <= 0:
            continue
        by_plat: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in by_diff[diff]:
            by_plat[str(item.get("platform") or "").lower()].append(item)
        alloc_plat = _allocate(
            {key: len(rows) for key, rows in by_plat.items()}, take_diff
        )
        for plat in sorted(by_plat):
            take = int(alloc_plat.get(plat, 0))
            if take <= 0:
                continue
            buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for item in by_plat[plat]:
                buckets[str(item.get("contest_date") or "")[:7]].append(item)
            for key in buckets:
                buckets[key].sort(key=lambda row: str(row["question_id"]))
                rng.shuffle(buckets[key])
            keys = sorted(buckets)
            index = {key: 0 for key in keys}
            picked: list[dict[str, Any]] = []
            while len(picked) < take:
                progressed = False
                for key in keys:
                    if index[key] < len(buckets[key]) and len(picked) < take:
                        picked.append(buckets[key][index[key]])
                        index[key] += 1
                        progressed = True
                if not progressed:
                    break
            if len(picked) != take:
                remainder = [
                    row
                    for row in by_plat[plat]
                    if row["question_id"] not in {item["question_id"] for item in picked}
                ]
                remainder.sort(key=lambda row: str(row["question_id"]))
                rng.shuffle(remainder)
                picked.extend(remainder[: take - len(picked)])
            if len(picked) != take:
                raise RuntimeError(
                    f"difficulty {diff} platform {plat} produced {len(picked)} != {take}"
                )
            chosen.extend(picked)
    chosen.sort(key=lambda row: str(row["question_id"]))
    if len(chosen) != n:
        raise RuntimeError(f"stratified sample produced {len(chosen)} != {n}")
    return chosen


def filter_ladder(
    items: Sequence[Mapping[str, Any]], ladder: str
) -> list[dict[str, Any]]:
    if ladder == "medium_hard":
        allowed = {"medium", "hard"}
    elif ladder == "hard_only":
        allowed = {"hard"}
    elif ladder == "medium_only":
        allowed = {"medium"}
    else:
        raise ValueError(ladder)
    return [
        dict(item)
        for item in items
        if str(item.get("difficulty") or "").lower() in allowed
    ]


def load_eligible_items() -> list[dict[str, Any]]:
    rows = load_release_v6_rows()
    _, eligible = build_eligible_pool(rows)
    return eligible


def sample_pilot(
    items: Sequence[Mapping[str, Any]] | None = None,
    *,
    exclude_ids: Sequence[str] = (),
    seed: int | None = None,
    label: str = "pilot",
) -> dict[str, Any]:
    pool = list(items) if items is not None else load_eligible_items()
    excluded = {str(qid) for qid in exclude_ids}
    if excluded:
        pool = [row for row in pool if str(row["question_id"]) not in excluded]
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    RETURN_DIR.mkdir(parents=True, exist_ok=True)
    if len(pool) < PILOT_N:
        raise RuntimeError(f"eligible pool {len(pool)} < pilot {PILOT_N}")
    used_seed = PILOT_SEED if seed is None else int(seed)
    selected = stratified_sample(pool, PILOT_N, used_seed)
    ids = [str(row["question_id"]) for row in selected]
    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "seed": used_seed,
        "n": PILOT_N,
        "label": label,
        "excluded_prior_ids": sorted(excluded),
        "question_ids": ids,
        "id_list_sha256": _id_hash(ids),
    }
    payload["difficulty"] = dict(
        Counter(str(row.get("difficulty")).lower() for row in selected)
    )
    payload["platform"] = dict(
        Counter(str(row.get("platform")) for row in selected)
    )
    write_csv(
        RETURN_DIR / f"{label}_sample.csv",
        [
            {
                "question_id": row["question_id"],
                "difficulty": row.get("difficulty"),
                "platform": row.get("platform"),
                "contest_date": row.get("contest_date"),
                "sample_order": index,
            }
            for index, row in enumerate(selected)
        ],
    )
    json_dump(SAMPLE_DIR / f"{label}.json", payload)
    json_dump(SAMPLE_DIR / "pilot.json", payload)
    _write_problems_jsonl(SAMPLE_DIR / "pilot_problems.jsonl", selected)
    _write_problems_jsonl(SAMPLE_DIR / f"{label}_problems.jsonl", selected)
    return payload


def freeze_main(
    *,
    excluded_ids: Sequence[str],
    ladder: str = "medium_hard",
    items: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    pool = filter_ladder(items if items is not None else load_eligible_items(), ladder)
    excluded = set(excluded_ids)
    remaining = [row for row in pool if str(row["question_id"]) not in excluded]
    min_n = MIN_HARD_ONLY if ladder == "hard_only" else MIN_ELIGIBLE
    if len(remaining) < min_n:
        raise RuntimeError(
            f"ladder {ladder} remaining {len(remaining)} < {min_n}"
        )
    n_main = min(MAIN_N, len(remaining))
    selected = stratified_sample(remaining, n_main, MAIN_SEED)
    main_ids = [str(row["question_id"]) for row in selected]
    leftover = [row for row in selected]
    repeats = stratified_sample(leftover, min(REPEAT_N, len(leftover)), REPEAT_SEED)
    repeat_ids = [str(row["question_id"]) for row in repeats]
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    RETURN_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(
        RETURN_DIR / "sample_main.csv",
        [
            {
                "question_id": row["question_id"],
                "difficulty": row.get("difficulty"),
                "platform": row.get("platform"),
                "contest_date": row.get("contest_date"),
                "contest_id": row.get("contest_id"),
                "sample_order": index,
                "release": LCB_RELEASE,
            }
            for index, row in enumerate(selected)
        ],
    )
    write_csv(
        RETURN_DIR / "repeat_subset_100.csv",
        [
            {
                "question_id": row["question_id"],
                "sample_order": index,
            }
            for index, row in enumerate(repeats)
        ],
    )
    _write_problems_jsonl(SAMPLE_DIR / "main_problems.jsonl", selected)
    hashes = template_hashes()
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "code_commit": _git_commit(),
        "dataset": DATASET_NAME,
        "hf_dataset": LCB_HF_LITE,
        "release": LCB_RELEASE,
        "lcb_commit": lcb_repo_commit(),
        "lcb_commit_pin": LCB_COMMIT_PIN,
        "ladder": ladder,
        "excluded_pilot_ids": list(excluded_ids),
        "pilot_id_list_sha256": _id_hash(list(excluded_ids)),
        "main_n": n_main,
        "main_seed": MAIN_SEED,
        "main_ids": main_ids,
        "main_id_list_sha256": _id_hash(main_ids),
        "repeat_n": len(repeat_ids),
        "repeat_seed": REPEAT_SEED,
        "repeat_ids": repeat_ids,
        "repeat_id_list_sha256": _id_hash(repeat_ids),
        "endpoints": dict(ENDPOINTS),
        "score_conditions": list(SCORE_CONDITIONS),
        "prompt_hashes": hashes,
        "planned_confirmatory_calls": (
            n_main * 2 * 2
            + n_main * 2 * len(SCORE_CONDITIONS)
            + len(repeat_ids) * 2 * 5 * 2
        ),
    }
    json_dump(SAMPLE_DIR / "freeze_manifest.json", manifest)
    json_dump(RETURN_DIR / "freeze_manifest.json", manifest)
    return manifest


def load_local_problems(name: str) -> list[dict[str, Any]]:
    path = SAMPLE_DIR / name
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_problems_jsonl(path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            out[str(item["question_id"])] = item
    return out


def _write_problems_jsonl(path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
