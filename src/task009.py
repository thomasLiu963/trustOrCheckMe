"""Task 009 orchestrator: freeze, paid confirmatory run, then analysis packet."""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT
from .study1_smoke import _git_commit
from .task005_common import sha256_file
from .task009_analyze import run_analysis
from .task009_common import (
    ANALYSIS_DIR,
    FROM_GPT,
    PAPER_DIRECTION,
    RETURN_DIR,
    SCIENTIFIC_CAP,
    SQLITE_PATH,
    TASK_ID,
    assert_009_write_target,
    freeze_paths,
    json_dump,
    protected_fingerprints,
)
from .task009_run import run_paid
from .task009_sample import freeze_sample

SRC_FILES = (
    Path(__file__),
    Path(__file__).with_name("task009_common.py"),
    Path(__file__).with_name("task009_sample.py"),
    Path(__file__).with_name("task009_run.py"),
    Path(__file__).with_name("task009_analyze.py"),
)


def _copy_scripts() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    assert_009_write_target(ANALYSIS_DIR)
    for src in SRC_FILES:
        dest = ANALYSIS_DIR / src.name
        shutil.copy2(src, dest)


def _changed_files() -> None:
    paths = [
        *SRC_FILES,
        FROM_GPT,
        RETURN_DIR / "report.md",
        RETURN_DIR / "preregistration.md",
        RETURN_DIR / "freeze_manifest.json",
        SQLITE_PATH,
    ]
    lines = []
    for path in paths:
        if path.exists():
            lines.append(str(path.relative_to(PROJECT_ROOT)))
    (RETURN_DIR / "changed_files.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_support_files(run_payload: dict[str, Any], analysis: dict[str, Any] | None) -> None:
    after = protected_fingerprints()
    validation = f"""# Task 009 validation

- Eligible pool hash matched Task 006/007.
- Primary 500 is disjoint from historical V2-B 500.
- Secondary 200 and repeat 100 are subsets of the primary 500.
- Moderate qualitative template hash matched Task 006/007.
- Endpoints locked to config/models.yaml.
- paperDirection.txt sha256: `{after["paperDirection"]["sha256"]}`
- Scientific cap: {SCIENTIFIC_CAP}
- Scientific requests used: {run_payload.get("scientific_used")}
- Provider attempts: {run_payload.get("provider_attempts")}
- Primary complete: {run_payload.get("primary_complete")}
- Isolation: new sqlite `{SQLITE_PATH}`
"""
    (RETURN_DIR / "validation.md").write_text(validation, encoding="utf-8")
    cost = f"""# Task 009 cost summary

- Scientific cap: {SCIENTIFIC_CAP}
- Scientific requests used: {run_payload.get("scientific_used")}
- Provider attempts: {run_payload.get("provider_attempts")} / {run_payload.get("provider_cap")}
- Wall clock seconds: {run_payload.get("wall_clock_seconds")}
- Recorded Stage-3 USD: {(analysis or {}).get("stage3_usd")}
- Retries are included in provider attempts, not in the scientific cap.
"""
    (RETURN_DIR / "cost_summary.md").write_text(cost, encoding="utf-8")
    deviations = RETURN_DIR / "deviations.md"
    if not deviations.exists():
        deviations.write_text(
            "# Task 009 deviations\n\nNo deviations from the frozen preregistration.\n",
            encoding="utf-8",
        )
    json_dump(
        RETURN_DIR / "run_manifest.json",
        {
            "task_id": TASK_ID,
            "created_at": datetime.now(UTC).isoformat(),
            "code_commit": _git_commit(),
            "run": {k: v for k, v in run_payload.items() if k != "phases"},
            "phases": run_payload.get("phases"),
            "paperDirection_sha256": after["paperDirection"]["sha256"],
        },
    )
    _changed_files()


def cmd_freeze() -> dict[str, Any]:
    RETURN_DIR.mkdir(parents=True, exist_ok=True)
    _copy_scripts()
    manifest = freeze_sample()
    print(
        json.dumps(
            {
                "ok": True,
                "primary_hash": manifest["sample"]["primary_id_list_sha256"],
                "secondary_hash": manifest["sample"]["secondary_id_list_sha256"],
                "repeat_hash": manifest["sample"]["repeat_id_list_sha256"],
                "scientific_cap": SCIENTIFIC_CAP,
            },
            indent=2,
        ),
        flush=True,
    )
    return manifest


def cmd_run() -> dict[str, Any]:
    freeze_sample()
    payload = asyncio.run(run_paid(allow_paid=True))
    json_dump(RETURN_DIR / "run_payload.json", payload)
    print(
        json.dumps(
            {
                "ok": payload.get("ok"),
                "primary_complete": payload.get("primary_complete"),
                "scientific_used": payload.get("scientific_used"),
                "provider_attempts": payload.get("provider_attempts"),
            },
            indent=2,
        ),
        flush=True,
    )
    return payload


def cmd_analyze(run_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = run_payload
    if payload is None and (RETURN_DIR / "run_payload.json").exists():
        payload = json.loads((RETURN_DIR / "run_payload.json").read_text(encoding="utf-8"))
    analysis = run_analysis(payload)
    write_support_files(payload or {}, analysis)
    _copy_scripts()
    print(
        json.dumps(
            {
                "ok": True,
                "bucket": analysis["decision"]["bucket"],
                "world": analysis["decision"]["world"],
                "paperDirection_sha256": sha256_file(PAPER_DIRECTION),
            },
            indent=2,
        ),
        flush=True,
    )
    return analysis


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 009 prospective confirmation")
    parser.add_argument(
        "command",
        nargs="?",
        default="all",
        choices=("freeze", "run", "analyze", "all"),
    )
    args = parser.parse_args()
    before = protected_fingerprints()
    if args.command == "freeze":
        cmd_freeze()
    elif args.command == "run":
        cmd_run()
    elif args.command == "analyze":
        cmd_analyze()
    else:
        cmd_freeze()
        payload = cmd_run()
        if payload.get("primary_complete"):
            cmd_analyze(payload)
        else:
            write_support_files(payload, None)
            raise SystemExit("primary GPT/Claude cells incomplete; analysis withheld")
    after = protected_fingerprints()
    if after["paperDirection"]["sha256"] != before["paperDirection"]["sha256"]:
        raise RuntimeError("paperDirection.txt changed during Task 009")


if __name__ == "__main__":
    main()
