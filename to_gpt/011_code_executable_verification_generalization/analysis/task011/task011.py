"""Task 011 orchestrator: optimized LiveCodeBench real-verification generalization."""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .checkpointing import CheckpointStore
from .config import PROJECT_ROOT
from .study1_smoke import _git_commit
from .task005_common import sha256_file
from .task011_benchmark import run_phase_a
from .task011_common import (
    ANALYSIS_DIR,
    CONCURRENCY_CODE,
    CONCURRENCY_PER_PROVIDER,
    CONCURRENCY_ROUTE,
    DATASET_NAME,
    EVAL_WORKERS,
    GPT_CLAUDE,
    MODEL_LABELS,
    PAPER_DIRECTION,
    PILOT_CONDITIONS,
    RETURN_DIR,
    SAMPLE_DIR,
    SECOND_PILOT_SEED,
    SQLITE_PATH,
    json_dump,
    protected_fingerprints,
    write_csv,
)
from .task011_eval import evaluate_frozen_code
from .task011_run import (
    frozen_code_jobs,
    run_main_paid,
    run_pilot_paid,
)
from .task011_sample import (
    freeze_main,
    load_local_problems,
    load_problems_jsonl,
    sample_pilot,
)


SRC_FILES = (
    Path(__file__),
    Path(__file__).with_name("task011_common.py"),
    Path(__file__).with_name("task011_benchmark.py"),
    Path(__file__).with_name("task011_prompts.py"),
    Path(__file__).with_name("task011_sample.py"),
    Path(__file__).with_name("task011_run.py"),
    Path(__file__).with_name("task011_eval.py"),
    Path(__file__).with_name("task011_analyze.py"),
)


def _copy_scripts() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    for src in SRC_FILES:
        shutil.copy2(src, ANALYSIS_DIR / src.name)


def cmd_validate() -> dict[str, Any]:
    before = protected_fingerprints()
    payload = run_phase_a()
    after = protected_fingerprints()
    if after["paperDirection"]["sha256"] != before["paperDirection"]["sha256"]:
        raise RuntimeError("paperDirection.txt changed during Task 011 validate")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "n_eligible_medium_hard": payload["n_eligible_medium_hard"],
                "lcb_commit": payload["lcb_commit"],
                "harness_ok": payload["harness"]["harness_ok"],
            },
            indent=2,
        ),
        flush=True,
    )
    return payload


def _pilot_decision(eval_rows: list[dict[str, Any]], route_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_model: dict[str, list[int]] = defaultdict(list)
    for row in eval_rows:
        by_model[row["model_alias"]].append(int(row["passed"]))
    accuracy = {
        alias: (sum(vals) / len(vals) if vals else 0.0) for alias, vals in by_model.items()
    }
    coverage: dict[str, dict[str, float]] = defaultdict(dict)
    for row in route_rows:
        alias = row["model_alias"]
        cond = row["score_condition"]
        action = row["parsed_action"]
        coverage[alias].setdefault(cond, [])
        coverage[alias][cond].append(int(action == "VERIFY_FIRST"))
    coverage_rate = {
        alias: {cond: (sum(vals) / len(vals) if vals else 0.0) for cond, vals in conds.items()}
        for alias, conds in coverage.items()
    }
    ladder = "medium_hard"
    stop = None
    for alias, acc in accuracy.items():
        if acc > 0.80:
            ladder = "hard_only"
        if acc < 0.20:
            ladder = "medium_only"
    accs = list(accuracy.values())
    if accs and min(accs) < 0.20 and max(accs) > 0.80:
        stop = "split_accuracy_pilot_STOP_for_GPT_review"
        ladder = "medium_hard"
    ident = True
    for alias, rates in coverage_rate.items():
        low = rates.get("displayed_0.70")
        high = rates.get("displayed_0.99")
        interior = any(0.10 <= rate <= 0.90 for rate in rates.values())
        delta = abs((low or 0) - (high or 0)) if low is not None and high is not None else 0
        if not interior and delta < 0.15:
            ident = False
    if not ident and stop is None:
        stop = "action_saturation_STOP_for_prompt_review"
    go = stop is None
    return {
        "go": go,
        "stop_reason": stop,
        "ladder": ladder if go else "medium_hard",
        "accuracy": accuracy,
        "coverage_rate": coverage_rate,
        "identifiable": ident,
    }


def _route_rows_from_sqlite(question_ids: list[str]) -> list[dict[str, Any]]:
    store = CheckpointStore(SQLITE_PATH)
    rows = []
    for qid in question_ids:
        for alias in GPT_CLAUDE:
            with store._lock:
                found = store._connection.execute(
                    """
                    SELECT record_json FROM requests
                    WHERE stage='verification' AND dataset=? AND example_id=?
                      AND model_alias=? AND status='success'
                    """,
                    (DATASET_NAME, qid, alias),
                ).fetchall()
            for rec in found:
                payload = json.loads(rec["record_json"])
                rows.append(
                    {
                        "question_id": qid,
                        "model_alias": alias,
                        "score_condition": payload.get("score_condition"),
                        "parsed_action": payload.get("parsed_action"),
                        "repeat_index": payload.get("repeat_index", 0),
                    }
                )
    return rows


def _write_pilot_report(sample: dict[str, Any], decision: dict[str, Any], path: Path) -> None:
    lines = [
        "# Task 011 excluded pilot",
        "",
        f"- Label: `{sample.get('label', 'pilot')}`",
        f"- N={sample['n']} seed `{sample['seed']}`",
        f"- IDs sha256 `{sample['id_list_sha256']}`",
        f"- Difficulty mix: {sample.get('difficulty')}",
        f"- Accuracy: {decision['accuracy']}",
        f"- Coverage rates: {decision['coverage_rate']}",
        f"- Identifiable: {decision['identifiable']}",
        f"- Predeclared ladder if GO: **{decision['ladder']}**",
        f"- GO: **{decision['go']}**"
        + (f" STOP: {decision['stop_reason']}" if decision["stop_reason"] else ""),
        "",
        "Pilot IDs are permanently excluded from the confirmatory sample.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _all_excluded_pilot_ids() -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for name in ("pilot_round1.json", "pilot_round2.json", "pilot.json"):
        path = SAMPLE_DIR / name
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for qid in payload.get("question_ids") or []:
            qid = str(qid)
            if qid not in seen:
                seen.add(qid)
                ids.append(qid)
        for qid in payload.get("excluded_prior_ids") or []:
            qid = str(qid)
            if qid not in seen:
                seen.add(qid)
                ids.append(qid)
    return ids


def _run_one_pilot(sample: dict[str, Any]) -> dict[str, Any]:
    items = load_local_problems("pilot_problems.jsonl")
    payload = asyncio.run(run_pilot_paid(items))
    jobs = frozen_code_jobs({str(row["question_id"]): row for row in items}, GPT_CLAUDE)
    eval_rows = evaluate_frozen_code(jobs, workers=EVAL_WORKERS)
    route_rows = _route_rows_from_sqlite(sample["question_ids"])
    decision = _pilot_decision(eval_rows, route_rows)
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "sample": sample,
        "run": {k: v for k, v in payload.items() if k != "results"},
        "decision": decision,
        "n_eval": len(eval_rows),
        "n_eval_passed": int(sum(int(row["passed"]) for row in eval_rows)),
        "concurrency": {
            "code": CONCURRENCY_CODE,
            "route": CONCURRENCY_ROUTE,
            "per_provider": CONCURRENCY_PER_PROVIDER,
            "eval_workers": EVAL_WORKERS,
        },
    }
    return report


def cmd_pilot() -> dict[str, Any]:
    RETURN_DIR.mkdir(parents=True, exist_ok=True)
    _copy_scripts()
    if not (RETURN_DIR / "benchmark_validation.md").exists():
        cmd_validate()
    round1_archive = RETURN_DIR / "pilot_round1_run.json"
    if not round1_archive.exists() and (RETURN_DIR / "pilot_run.json").exists():
        shutil.copy2(RETURN_DIR / "pilot_run.json", round1_archive)
        if (RETURN_DIR / "pilot_report.md").exists():
            shutil.copy2(RETURN_DIR / "pilot_report.md", RETURN_DIR / "pilot_round1_report.md")
        if (RETURN_DIR / "pilot_sample.csv").exists():
            shutil.copy2(RETURN_DIR / "pilot_sample.csv", RETURN_DIR / "pilot_round1_sample.csv")
        if (SAMPLE_DIR / "pilot.json").exists():
            shutil.copy2(SAMPLE_DIR / "pilot.json", SAMPLE_DIR / "pilot_round1.json")
        if (RETURN_DIR / "hidden_test_results.csv").exists():
            shutil.copy2(
                RETURN_DIR / "hidden_test_results.csv",
                RETURN_DIR / "pilot_round1_hidden_test_results.csv",
            )
    if round1_archive.exists():
        round1 = json.loads(round1_archive.read_text(encoding="utf-8"))
        burned = list(round1.get("sample", {}).get("question_ids") or [])
        sample = sample_pilot(
            exclude_ids=burned,
            seed=SECOND_PILOT_SEED,
            label="pilot_round2",
        )
        report = _run_one_pilot(sample)
        json_dump(RETURN_DIR / "pilot_round2_run.json", report)
        json_dump(RETURN_DIR / "pilot_run.json", report)
        _write_pilot_report(sample, report["decision"], RETURN_DIR / "pilot_round2_report.md")
        _write_pilot_report(sample, report["decision"], RETURN_DIR / "pilot_report.md")
        print(json.dumps({"ok": report["run"].get("ok"), **report["decision"]}, indent=2, default=str), flush=True)
        if not report["decision"]["go"]:
            raise SystemExit(f"pilot STOP: {report['decision']['stop_reason']}")
        return report
    sample = sample_pilot()
    report = _run_one_pilot(sample)
    json_dump(RETURN_DIR / "pilot_run.json", report)
    _write_pilot_report(sample, report["decision"], RETURN_DIR / "pilot_report.md")
    print(json.dumps({"ok": report["run"].get("ok"), **report["decision"]}, indent=2, default=str), flush=True)
    if not report["decision"]["go"]:
        raise SystemExit(f"pilot STOP: {report['decision']['stop_reason']}")
    return report


def _write_preregistration(manifest: dict[str, Any], ladder: str) -> None:
    text = f"""# Task 011 preregistration — prospective code verification generalization

Frozen at `{manifest["created_at"]}` before confirmatory VERIFY-rate inspection.
Commit: `{manifest.get("code_commit") or "unknown"}`.
Label: **confirmatory**.

## Sample

- Dataset: LiveCodeBench `{manifest["hf_dataset"]}` `{manifest["release"]}`
- Repo commit: `{manifest["lcb_commit"]}`
- Difficulty ladder: `{ladder}`
- Excluded pilot N={len(manifest["excluded_pilot_ids"])} sha256 `{manifest["pilot_id_list_sha256"]}`
- Main N={manifest["main_n"]} seed `{manifest["main_seed"]}` sha256 `{manifest["main_id_list_sha256"]}`
- Repeat N={manifest["repeat_n"]} seed `{manifest["repeat_seed"]}` sha256 `{manifest["repeat_id_list_sha256"]}`

## Models

- GPT `{manifest["endpoints"]["openai_gpt56_sol"]}`
- Claude `{manifest["endpoints"]["anthropic_sonnet5"]}`

## Protocol

- Stage 1: one frozen Python solution per model/problem. Malformed extraction is frozen empty and scored incorrect.
- Stage 2: q1 = P(this exact frozen code passes the independent hidden test suite).
- Stage 3: Task-009 moderate qualitative routing adapted to code. Actions USE_UNVERIFIED / VERIFY_FIRST.
- Conditions: {", ".join(manifest["score_conditions"])}.
- Repeats: N=100, two extra Stage-3 draws on the five fixed displayed scores only.
- Hidden official tests run only after all confirmatory routing is checkpointed.
- Engineering concurrency (not a scientific factor): code {CONCURRENCY_CODE} global / {CONCURRENCY_PER_PROVIDER} per provider; routing {CONCURRENCY_ROUTE}; eval workers {EVAL_WORKERS}. Item pipeline: code → q1 → Stage-3.

## Prompt hashes

{json.dumps(manifest["prompt_hashes"], indent=2)}
"""
    (RETURN_DIR / "preregistration.md").write_text(text, encoding="utf-8")
    (RETURN_DIR / "analysis_freeze.json").write_text(
        json.dumps(
            {
                "bootstrap_resamples": 5000,
                "cv_folds": 5,
                "material_logloss": 0.01,
                "h1_contrast": ["displayed_0.70", "displayed_0.99"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def cmd_freeze(ladder: str = "medium_hard") -> dict[str, Any]:
    pilot_path = SAMPLE_DIR / "pilot.json"
    if not pilot_path.exists():
        raise SystemExit("run pilot before freeze")
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    excluded = _all_excluded_pilot_ids() or list(pilot["question_ids"])
    if (RETURN_DIR / "pilot_run.json").exists():
        decision = json.loads((RETURN_DIR / "pilot_run.json").read_text(encoding="utf-8"))
        ladder = str(decision.get("decision", {}).get("ladder") or ladder)
    manifest = freeze_main(excluded_ids=excluded, ladder=ladder)
    _write_preregistration(manifest, ladder)
    (RETURN_DIR / "deviations.md").write_text(
        "# Task 011 deviations\n\n"
        "## Sampler bug and second excluded pilot\n\n"
        "The first excluded pilot (seed `20260921`) used a round-robin over "
        "`difficulty|platform|month` keys sorted lexicographically, so `hard` "
        "filled the entire N=40 before any `medium` item was drawn. That sample "
        "is permanently excluded. Stratification was corrected to allocate "
        "proportionally by difficulty and then by platform. A second excluded "
        "pilot (seed `20260924`) was drawn from the remaining pool. This is an "
        "implementation fix, not prompt tuning.\n\n"
        "## Predeclared hard-only confirmatory N\n\n"
        "GPT frozen-code accuracy on the mixed second pilot was 87.5% (>80%), "
        "so the predeclared ladder is hard-only. After excluding both pilots, "
        "286 hard problems remain (<300). The confirmatory sample uses the "
        "entire remaining hard pool rather than mixing medium back in.\n\n"
        "## Engineering\n\n"
        "Concurrency is code=24 / route=24 / 12 per provider; hidden tests use "
        "16 worker processes. Code generation uses unstructured 4096-token "
        "completions because models.yaml's 64-token JSON cap cannot emit "
        "programs. These choices do not change prompts, endpoints, or analyses.\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "ok": True,
                "ladder": ladder,
                "main_n": manifest["main_n"],
                "main_hash": manifest["main_id_list_sha256"],
            },
            indent=2,
        ),
        flush=True,
    )
    return manifest


def cmd_run() -> dict[str, Any]:
    if not (RETURN_DIR / "freeze_manifest.json").exists():
        cmd_freeze()
    manifest = json.loads((RETURN_DIR / "freeze_manifest.json").read_text(encoding="utf-8"))
    items = load_local_problems("main_problems.jsonl")
    payload = asyncio.run(run_main_paid(items, manifest["repeat_ids"]))
    json_dump(RETURN_DIR / "run_payload.json", {k: v for k, v in payload.items() if k != "results"})
    print(
        json.dumps(
            {
                "ok": payload.get("ok"),
                "pairs": payload.get("pairs"),
                "failed": payload.get("failed"),
                "scientific_used": payload.get("scientific_used"),
            },
            indent=2,
        ),
        flush=True,
    )
    return payload


def cmd_evaluate() -> list[dict[str, Any]]:
    items = load_problems_jsonl(SAMPLE_DIR / "main_problems.jsonl")
    jobs = frozen_code_jobs(items, GPT_CLAUDE)
    rows = evaluate_frozen_code(jobs, workers=EVAL_WORKERS)
    write_csv(RETURN_DIR / "test_execution_results.csv", rows)
    print(json.dumps({"n": len(rows), "passed": sum(int(r["passed"]) for r in rows)}, indent=2), flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        nargs="?",
        default="pilot",
        choices=("validate", "pilot", "freeze", "run", "evaluate", "analyze", "all"),
    )
    args = parser.parse_args()
    RETURN_DIR.mkdir(parents=True, exist_ok=True)
    before = protected_fingerprints()
    if args.command == "validate":
        cmd_validate()
    elif args.command == "pilot":
        cmd_pilot()
    elif args.command == "freeze":
        cmd_freeze()
    elif args.command == "run":
        cmd_run()
    elif args.command == "evaluate":
        cmd_evaluate()
    elif args.command == "analyze":
        from .task011_analyze import run_analysis

        run_analysis()
    else:
        cmd_validate()
        cmd_pilot()
        cmd_freeze()
        payload = cmd_run()
        if payload.get("ok"):
            cmd_evaluate()
    after = protected_fingerprints()
    if after["paperDirection"]["sha256"] != before["paperDirection"]["sha256"]:
        raise RuntimeError("paperDirection.txt changed during Task 011")
    _copy_scripts()


if __name__ == "__main__":
    main()
