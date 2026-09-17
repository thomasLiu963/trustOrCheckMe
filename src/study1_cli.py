"""CLI commands for Study 1. Paid execution is gated and not used in Task 001."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from typing import Any

from .config import PROJECT_ROOT
from .study1_plan import prepare_study1_offline
from .study1_runner import Study1Runner
from .study1_schemas import load_study1_config

COMMANDS = (
    "study1-prepare",
    "study1-plan",
    "study1-run",
    "study1-smoke",
)


def register(subparsers: Any) -> None:
    prepare = subparsers.add_parser(
        "study1-prepare",
        help="Freeze Study 1 IDs, build offline call plans, and render prompt audits.",
    )
    prepare.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "experiment_study1.yaml"),
    )

    plan = subparsers.add_parser(
        "study1-plan",
        help="Print the offline Study 1 call-plan counts. No API calls.",
    )
    plan.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "experiment_study1.yaml"),
    )

    run = subparsers.add_parser(
        "study1-run",
        help="Study 1 execution. Dry-run unless explicitly authorized.",
    )
    run.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "experiment_study1.yaml"),
    )
    run.add_argument("--phase", choices=("primary", "repeats"), required=True)
    mode = run.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--yes", action="store_true")
    run.add_argument("--max-calls", type=int)

    smoke = subparsers.add_parser(
        "study1-smoke",
        help="Task 002 tiny paid smoke test: 12 scientific cells, 20 provider attempts.",
    )
    smoke.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "experiment_study1.yaml"),
    )
    smoke.add_argument(
        "--yes",
        action="store_true",
        help="Required. Confirms the 12-cell / 20-attempt paid smoke test.",
    )
    smoke.add_argument("--max-scientific-cells", type=int, required=True)
    smoke.add_argument("--max-provider-attempts", type=int, required=True)


def dispatch(args: argparse.Namespace) -> None:
    config = load_study1_config(args.config)
    if args.command in {"study1-prepare", "study1-plan"}:
        result = prepare_study1_offline(config)
        print(
            json.dumps(
                {
                    "primary_calls": result["primary_count"],
                    "repeat_extra_calls": result["repeat_extra_count"],
                    "total_calls": result["primary_count"] + result["repeat_extra_count"],
                    "selected_id_list_sha256": result["primary_payload"][
                        "selected_id_list_sha256"
                    ],
                    "repeat_id_list_sha256": result["repeat_payload"][
                        "repeat_id_list_sha256"
                    ],
                    "api_calls_made": 0,
                    "paid_calls_authorized": False,
                },
                indent=2,
            )
        )
        return
    if args.command == "study1-run":
        runner = Study1Runner(config)
        summary = runner.run(
            phase=args.phase,
            dry_run=bool(args.dry_run),
            allow_paid=bool(args.yes),
            max_calls=args.max_calls,
        )
        print(json.dumps(summary, indent=2))
        return
    if args.command == "study1-smoke":
        from .study1_runner import Study1AuthorizationError
        from .study1_smoke import (
            SMOKE_PROVIDER_ATTEMPT_CAP,
            SMOKE_SCIENTIFIC_CELL_CAP,
            execute_smoke_task,
        )

        if not args.yes:
            raise Study1AuthorizationError(
                "study1-smoke requires --yes for the authorized 12-cell paid test"
            )
        if args.max_scientific_cells != SMOKE_SCIENTIFIC_CELL_CAP:
            raise Study1AuthorizationError(
                "study1-smoke max-scientific-cells must be exactly 12"
            )
        if args.max_provider_attempts != SMOKE_PROVIDER_ATTEMPT_CAP:
            raise Study1AuthorizationError(
                "study1-smoke max-provider-attempts must be exactly 20"
            )
        execute_smoke_task()
        return
    raise ValueError(f"unknown Study 1 command: {args.command}")
