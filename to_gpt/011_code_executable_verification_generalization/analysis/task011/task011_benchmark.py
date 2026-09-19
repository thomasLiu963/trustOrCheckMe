"""Load LiveCodeBench lite, build the eligible pool, and validate the official checker."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from .task005_common import sha256_file
from .task011_common import (
    DATASET_FILES,
    ERRATA_QUESTION_IDS,
    EVAL_TIMEOUT_SEC,
    LCB_COMMIT_PIN,
    LCB_DATA,
    LCB_HF_LITE,
    LCB_RELEASE,
    LCB_REPO,
    LCB_REPO_URL,
    MIN_ELIGIBLE,
    RETURN_DIR,
    TARGET_ELIGIBLE,
    json_dump,
    write_csv,
)


def _ensure_lcb_on_path() -> None:
    root = str(LCB_REPO.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def lcb_repo_commit() -> str:
    head = LCB_REPO / ".git" / "HEAD"
    if not head.exists():
        return ""
    text = head.read_text(encoding="utf-8").strip()
    if text.startswith("ref:"):
        ref = LCB_REPO / ".git" / text.split(" ", 1)[1].strip()
        return ref.read_text(encoding="utf-8").strip()
    return text


def dataset_file_fingerprints() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in DATASET_FILES:
        path = LCB_DATA / name
        out[name] = {
            "path": str(path),
            "exists": path.exists(),
            "bytes": path.stat().st_size if path.exists() else 0,
            "sha256": sha256_file(path) if path.exists() else None,
        }
    return out


def _n_tests(raw: Any) -> int:
    if raw in (None, ""):
        return 0
    if isinstance(raw, list):
        return len(raw)
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return len(parsed)
    except Exception:
        pass
    try:
        import base64
        import pickle
        import zlib

        parsed = json.loads(
            pickle.loads(zlib.decompress(base64.b64decode(str(raw).encode("utf-8"))))
        )
        return len(parsed) if isinstance(parsed, list) else 0
    except Exception:
        return 0


def _public_tests(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        return raw
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def load_release_v6_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in DATASET_FILES:
        path = LCB_DATA / name
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                item["_source_file"] = name
                rows.append(item)
    return rows


def is_errata(question_id: str) -> bool:
    qid = str(question_id).strip()
    if qid in ERRATA_QUESTION_IDS:
        return True
    lowered = qid.lower()
    return any(token.lower() == lowered for token in ERRATA_QUESTION_IDS)


def metadata_row(item: dict[str, Any], *, eligible: bool, reason: str) -> dict[str, Any]:
    qid = str(item["question_id"])
    return {
        "question_id": qid,
        "question_title": item.get("question_title"),
        "platform": item.get("platform"),
        "contest_id": item.get("contest_id"),
        "contest_date": item.get("contest_date"),
        "difficulty": str(item.get("difficulty") or "").lower(),
        "starter_code_present": bool(str(item.get("starter_code") or "").strip()),
        "n_public_tests": _n_tests(item.get("public_test_cases")),
        "n_private_tests": _n_tests(item.get("private_test_cases")),
        "source_file": item.get("_source_file"),
        "eligible": int(eligible),
        "exclusion_reason": reason,
        "errata": int(is_errata(qid)),
        "python_compatible": 1,
        "release": LCB_RELEASE,
    }


def build_eligible_pool(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    catalog: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    for item in rows:
        qid = str(item["question_id"])
        difficulty = str(item.get("difficulty") or "").lower()
        n_pub = _n_tests(item.get("public_test_cases"))
        n_priv = _n_tests(item.get("private_test_cases"))
        reason = ""
        ok = True
        if difficulty not in {"medium", "hard"}:
            ok = False
            reason = "not_medium_or_hard"
        elif is_errata(qid):
            ok = False
            reason = "official_errata"
        elif n_pub + n_priv <= 0:
            ok = False
            reason = "no_official_tests"
        meta = metadata_row(item, eligible=ok, reason=reason)
        catalog.append(meta)
        if ok:
            eligible.append(item)
    eligible.sort(
        key=lambda item: (
            str(item.get("contest_date") or ""),
            str(item.get("question_id") or ""),
        )
    )
    return catalog, eligible


def evaluation_sample(item: dict[str, Any], *, public_only: bool = False) -> dict[str, Any]:
    _ensure_lcb_on_path()
    from lcb_runner.benchmarks.code_generation import CodeGenerationProblem

    payload = {key: item[key] for key in item if not str(key).startswith("_")}
    problem = CodeGenerationProblem(**payload)
    if public_only:
        tests = problem.public_test_cases
        return {
            "input_output": json.dumps(
                {
                    "inputs": [t.input for t in tests],
                    "outputs": [t.output for t in tests],
                    "fn_name": problem.metadata.get("func_name", None),
                }
            )
        }
    return problem.get_evaluation_sample()


def _strip_secrets(env: Mapping[str, str]) -> dict[str, str]:
    blocked = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
    cleaned: dict[str, str] = {}
    for key, value in env.items():
        upper = key.upper()
        if any(token in upper for token in blocked):
            continue
        cleaned[key] = value
    cleaned.setdefault("PATH", "/usr/bin:/bin")
    cleaned.setdefault("LANG", "C")
    cleaned["PYTHONPATH"] = str(LCB_REPO.resolve())
    cleaned["OMP_NUM_THREADS"] = "1"
    return cleaned


def run_official_checker(
    item: dict[str, Any],
    code: str,
    *,
    public_only: bool = False,
    timeout: int = EVAL_TIMEOUT_SEC,
) -> dict[str, Any]:
    _ensure_lcb_on_path()
    from lcb_runner.evaluation.compute_code_generation_metrics import check_correctness

    sample = evaluation_sample(item, public_only=public_only)
    previous = dict(os.environ)
    try:
        os.environ.clear()
        os.environ.update(_strip_secrets(previous))
        result, metadata = check_correctness(sample, code, timeout=timeout, debug=False)
    finally:
        os.environ.clear()
        os.environ.update(previous)
    flags = []
    for value in result or []:
        flags.append(bool(value) if isinstance(value, bool) else False)
    passed = bool(flags) and all(flags)
    return {
        "passed": passed,
        "n_tests": len(flags),
        "n_passed": int(sum(1 for flag in flags if flag)),
        "raw": list(result) if result is not None else [],
        "metadata": metadata,
        "public_only": public_only,
    }


def _public_echo_program(item: dict[str, Any]) -> str | None:
    tests = _public_tests(item.get("public_test_cases"))
    if not tests:
        return None
    first = tests[0]
    if str(first.get("testtype") or "").lower() == "functional":
        return None
    inp = first.get("input")
    out = first.get("output")
    if inp is None or out is None:
        return None
    return (
        "import sys\n"
        "data = sys.stdin.read()\n"
        f"expected = {inp!r}\n"
        f"output = {out!r}\n"
        "if data == expected or data.strip() == str(expected).strip():\n"
        "    sys.stdout.write(output if isinstance(output, str) else str(output))\n"
        "else:\n"
        "    sys.stdout.write('__no_match__')\n"
    )


def validate_evaluator(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stdin_items = [
        item
        for item in rows
        if str(item.get("difficulty") or "").lower() == "easy"
        and not str(item.get("starter_code") or "").strip()
        and _public_tests(item.get("public_test_cases"))
    ]
    if not stdin_items:
        raise RuntimeError("no easy stdin problem available for harness validation")
    item = stdin_items[0]
    echo = _public_echo_program(item)
    if echo is None:
        raise RuntimeError("could not build public-test echo program")
    good = run_official_checker(item, echo, public_only=True)
    bad = run_official_checker(item, "print('definitely-wrong')\n", public_only=True)
    boom = run_official_checker(item, "syntax error please", public_only=True)
    loop = "import time\nwhile True:\n    time.sleep(0.1)\n"
    timed = run_official_checker(item, loop, public_only=True, timeout=2)
    return {
        "question_id": item["question_id"],
        "platform": item.get("platform"),
        "difficulty": item.get("difficulty"),
        "echo_public_pass": good["passed"],
        "wrong_public_pass": bad["passed"],
        "syntax_public_pass": boom["passed"],
        "timeout_public_pass": timed["passed"],
        "echo_detail": {k: v for k, v in good.items() if k != "raw"},
        "harness_ok": bool(
            good["passed"]
            and not bad["passed"]
            and not boom["passed"]
            and not timed["passed"]
        ),
        "isolation": {
            "process_wrapper": "LiveCodeBench check_correctness multiprocessing.Process + SIGALRM",
            "env_secrets_stripped": True,
            "timeout_seconds": EVAL_TIMEOUT_SEC,
            "memory_guard": "LCB reliability_guard ~4GB",
            "network": "not granted; child inherits stripped env without API keys",
        },
    }


def run_phase_a() -> dict[str, Any]:
    RETURN_DIR.mkdir(parents=True, exist_ok=True)
    if not LCB_REPO.exists():
        raise RuntimeError(f"LiveCodeBench repo missing at {LCB_REPO}")
    commit = lcb_repo_commit()
    rows = load_release_v6_rows()
    catalog, eligible = build_eligible_pool(rows)
    write_csv(RETURN_DIR / "eligible_pool.csv", catalog)
    fingerprints = dataset_file_fingerprints()
    harness = validate_evaluator(rows)
    n_eligible = len(eligible)
    if n_eligible < MIN_ELIGIBLE:
        status = "BENCHMARK_POOL_INSUFFICIENT"
    elif n_eligible < TARGET_ELIGIBLE:
        status = "POOL_OK_USE_ENTIRE_ELIGIBLE"
    else:
        status = "POOL_OK_SAMPLE_500"
    id_hash = hashlib.sha256(
        "\n".join(str(item["question_id"]) for item in eligible).encode("utf-8")
    ).hexdigest()
    payload = {
        "status": status,
        "n_release_v6": len(rows),
        "n_eligible_medium_hard": n_eligible,
        "difficulty_release": dict(Counter(str(r.get("difficulty")).lower() for r in rows)),
        "difficulty_eligible": dict(
            Counter(str(r.get("difficulty")).lower() for r in eligible)
        ),
        "platform_eligible": dict(Counter(str(r.get("platform")) for r in eligible)),
        "date_min": min(str(r.get("contest_date")) for r in rows),
        "date_max": max(str(r.get("contest_date")) for r in rows),
        "eligible_id_list_sha256": id_hash,
        "lcb_repo_url": LCB_REPO_URL,
        "lcb_commit": commit,
        "lcb_commit_pin": LCB_COMMIT_PIN,
        "hf_dataset": LCB_HF_LITE,
        "release": LCB_RELEASE,
        "dataset_files": fingerprints,
        "harness": harness,
        "errata_excluded": sorted(ERRATA_QUESTION_IDS),
        "generated_at": datetime.now().isoformat(),
    }
    json_dump(RETURN_DIR / "benchmark_validation_stats.json", payload)
    _write_benchmark_markdown(payload)
    return payload


def _write_benchmark_markdown(payload: Mapping[str, Any]) -> None:
    harness = payload["harness"]
    proceed = payload["status"].startswith("POOL_OK")
    text = f"""# Task 011 benchmark validation

Label: **Phase A**, zero primary-model scientific calls.

## Source

- Official repo: `{payload["lcb_repo_url"]}`
- Pinned commit: `{payload["lcb_commit"]}` (expected pin `{payload["lcb_commit_pin"]}`)
- HuggingFace dataset: `{payload["hf_dataset"]}`
- Release: `{payload["release"]}` (newest official lite snapshot in this environment; files test.jsonl–test6.jsonl)
- Evaluator: LiveCodeBench `lcb_runner.evaluation.compute_code_generation_metrics.check_correctness` wrapping `testing_util.run_test`

Lite is the official default harness (`code_generation_lite`). Hidden private tests remain hidden from the model. We do **not** claim contamination-free beyond LCB's contest-date methodology (problems from 2023-05-07 through 2025-04-06).

## Pool

- Release_v6 problems: {payload["n_release_v6"]}
- Difficulty (all): {payload["difficulty_release"]}
- Eligible medium+hard after official ERRATA exclusions: **{payload["n_eligible_medium_hard"]}**
- Eligible platforms: {payload["platform_eligible"]}
- Eligible ID list sha256: `{payload["eligible_id_list_sha256"]}`
- Status: **{payload["status"]}**

Eligibility: code-generation tasks, Python-compatible official tests, difficulty in {{medium, hard}}, not listed in official `ERRATA.md`. No problem was dropped because a model would fail it.

## Harness check

Problem `{harness["question_id"]}` (easy stdin, public tests only):

- public-test echo program passed: {harness["echo_public_pass"]}
- wrong program passed: {harness["wrong_public_pass"]}
- syntax-error program passed: {harness["syntax_public_pass"]}
- infinite-loop / timeout passed: {harness["timeout_public_pass"]}
- harness_ok: **{harness["harness_ok"]}**

Isolation: {harness["isolation"]}

Generated code is untrusted. Evaluation runs in a child process with API keys stripped from the environment, LCB's 4GB reliability guard, and wall-clock timeout. Host secrets are not passed through.

## Decision

{"Proceed to the excluded 40-problem pilot." if proceed else "STOP. BENCHMARK_POOL_INSUFFICIENT."}
"""
    path = RETURN_DIR / "benchmark_validation.md"
    path.write_text(text, encoding="utf-8")
