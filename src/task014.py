"""Task 014 orchestrator: freeze, then Stage-3, then analysis."""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from .task005_common import sha256_file
from .task014_audit import delta0_reuse, run_transform_audit, write_audit_md
from .task014_common import (
    ANALYSIS_DIR,
    PAPER_DIRECTION,
    RETURN_DIR,
    json_dump,
    write_csv,
)
from .task014_data import load_frozen_items
from .task014_preregister import write_freeze
from .task014_run import reused_delta0_record, run_stage3


def cmd_freeze() -> None:
    items = load_frozen_items()
    reuse = delta0_reuse(items)
    audit = run_transform_audit(items)
    write_csv(RETURN_DIR / "transformation_audit.csv", audit)
    write_audit_md(audit, reuse)
    write_freeze(items, reuse["decision"], audit)
    print(json.dumps({"status": "frozen", "reuse": reuse["decision"], "n": len(items)}, indent=2))


def cmd_run() -> None:
    if not (RETURN_DIR / "analysis_freeze.json").exists():
        raise RuntimeError("freeze before Stage-3 calls")
    items = load_frozen_items()
    reuse = delta0_reuse(items)
    if reuse["decision"] != "REUSE_TRUE_Q_VISIBLE":
        raise RuntimeError("δ=0 reuse unexpected; rerun decision must be recorded first")
    payload = asyncio.run(run_stage3(items))
    json_dump(RETURN_DIR / "run_progress.json", payload)
    print(json.dumps(payload, indent=2, default=str))


def cmd_export() -> None:
    from .checkpointing import CheckpointStore
    from .task014_common import NEW_DELTAS, SQLITE_PATH
    from .task014_run import request_key

    items = load_frozen_items()
    store = CheckpointStore(SQLITE_PATH)
    rows = []
    for item in items:
        rows.append(reused_delta0_record(item))
        for delta in NEW_DELTAS:
            rec = store.get_record(request_key(item, delta)) or {}
            rows.append(
                {
                    "task": item.task,
                    "question_id": item.question_id,
                    "model_alias": item.model_alias,
                    "delta": delta,
                    "q1": item.q1,
                    "q_delta": rec.get("q_delta"),
                    "parsed_action": rec.get("parsed_action"),
                    "verify": rec.get("verify"),
                    "incorrect": item.incorrect,
                    "reused_delta0": False,
                    "estimated_cost_usd": rec.get("estimated_cost_usd"),
                    "request_key": rec.get("request_key"),
                    "ok": bool(rec),
                }
            )
    write_csv(RETURN_DIR / "stage3_offset_results.csv", rows)
    print(f"exported {len(rows)} rows")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=("freeze", "run", "export", "analyze"))
    args = parser.parse_args()
    if args.cmd == "freeze":
        cmd_freeze()
    elif args.cmd == "run":
        cmd_run()
    elif args.cmd == "export":
        cmd_export()
    else:
        from .task014_analyze import analyze_all

        analyze_all()


if __name__ == "__main__":
    main()
