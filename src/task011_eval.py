"""Parallel official LiveCodeBench evaluation after routing is frozen."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

from .task011_benchmark import evaluation_sample
from .task011_common import EVAL_TIMEOUT_SEC, EVAL_WORKERS, RETURN_DIR, write_csv


def _strip_worker_env() -> None:
    blocked = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
    for key in list(os.environ):
        if any(token in key.upper() for token in blocked):
            del os.environ[key]
    os.environ.setdefault("PATH", "/usr/bin:/bin")
    os.environ.setdefault("LANG", "C")
    os.environ["OMP_NUM_THREADS"] = "1"


def _eval_one(payload: dict[str, Any]) -> dict[str, Any]:
    from .task011_benchmark import _ensure_lcb_on_path, evaluation_sample

    _ensure_lcb_on_path()
    from lcb_runner.evaluation.compute_code_generation_metrics import check_correctness

    item = payload["item"]
    code = str(payload.get("code") or "")
    timeout = int(payload.get("timeout") or EVAL_TIMEOUT_SEC)
    started = os.times()
    try:
        sample = evaluation_sample(item, public_only=False)
        raw, metadata = check_correctness(sample, code, timeout=timeout, debug=False)
    except Exception as error:
        raw, metadata = [], {"error": str(error)[:500]}
    flags = []
    for value in raw or []:
        flags.append(bool(value) if isinstance(value, bool) else False)
    passed = bool(flags) and all(flags)
    cpu = os.times()
    failure_class = "pass" if passed else "fail"
    meta_text = json.dumps(metadata, default=str).lower()
    if not passed:
        if "timeout" in meta_text or "timed out" in meta_text:
            failure_class = "timeout"
        elif not code.strip():
            failure_class = "unparseable"
        elif "syntax" in meta_text:
            failure_class = "syntax"
        elif "runtime" in meta_text or "exception" in meta_text:
            failure_class = "runtime"
    return {
        "question_id": payload["question_id"],
        "model_alias": payload["model_alias"],
        "passed": int(passed),
        "n_tests": len(flags),
        "n_passed": int(sum(1 for flag in flags if flag)),
        "failure_class": failure_class,
        "public_only": 0,
        "metadata": json.dumps(metadata, default=str)[:2000],
        "cpu_user": getattr(cpu, "user", 0) - getattr(started, "user", 0),
    }


def evaluate_frozen_code(
    jobs: Sequence[Mapping[str, Any]],
    *,
    workers: int = EVAL_WORKERS,
    timeout: int = EVAL_TIMEOUT_SEC,
) -> list[dict[str, Any]]:
    if not jobs:
        return []
    payloads = [
        {
            "question_id": str(job["question_id"]),
            "model_alias": str(job["model_alias"]),
            "item": job["item"],
            "code": str(job.get("code") or ""),
            "timeout": timeout,
        }
        for job in jobs
    ]
    results: list[dict[str, Any]] = []
    n_workers = max(1, min(int(workers), len(payloads)))
    print(
        f"eval: {len(payloads)} programs workers={n_workers} timeout={timeout}s",
        flush=True,
    )
    done = 0
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_strip_worker_env,
    ) as pool:
        futures = {pool.submit(_eval_one, payload): payload for payload in payloads}
        for future in as_completed(futures):
            results.append(future.result())
            done += 1
            if done % 25 == 0 or done == len(payloads):
                print(f"eval: {done}/{len(payloads)}", flush=True)
    results.sort(key=lambda row: (row["model_alias"], row["question_id"]))
    write_csv(RETURN_DIR / "hidden_test_results.csv", results)
    return results
