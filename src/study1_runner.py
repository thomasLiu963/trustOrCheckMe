"""Study 1 execution guards. Default is offline dry-run; no paid calls."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .config import load_models_config
from .study1_plan import Study1PlannedCall, build_call_plans, validate_call_plans
from .study1_sample import assert_not_v2_write_target
from .study1_schemas import (
    STUDY1_MODEL_ALIASES,
    Study1ExperimentConfig,
    Study1Phase,
    load_study1_config,
)

ALLOWED_PHASES = {Study1Phase.PRIMARY, Study1Phase.REPEATS}


class Study1AuthorizationError(PermissionError):
    """Raised when a Study 1 run is not explicitly authorized."""


def authorize_study1_execution(
    *,
    planned_calls: int,
    dry_run: bool = True,
    allow_paid: bool = False,
    max_calls: int | None = None,
    phase: Study1Phase | str | None = None,
    checkpoint_path: Path | None = None,
    config: Study1ExperimentConfig | None = None,
) -> None:
    """Refuse unpaid, uncapped, unphased, or historical-overwrite runs."""
    config = config or load_study1_config()
    if dry_run:
        if allow_paid:
            raise Study1AuthorizationError(
                "dry-run cannot be combined with allow_paid"
            )
        return
    if not allow_paid:
        raise Study1AuthorizationError(
            "Paid Study 1 execution requires allow_paid=True"
        )
    if max_calls is None:
        raise Study1AuthorizationError(
            "Paid Study 1 execution requires an explicit max_calls cap"
        )
    if int(max_calls) < 1:
        raise Study1AuthorizationError("max_calls must be a positive integer")
    if phase is None:
        raise Study1AuthorizationError(
            "Paid Study 1 execution requires phase='primary' or phase='repeats'"
        )
    resolved_phase = Study1Phase(phase)
    if resolved_phase not in ALLOWED_PHASES:
        raise Study1AuthorizationError(f"unsupported Study 1 phase: {resolved_phase}")
    if planned_calls > int(max_calls):
        raise Study1AuthorizationError(
            f"planned {planned_calls} calls exceed authorized cap {max_calls}; "
            "STOP before making any calls"
        )
    target = checkpoint_path or config.study1_sqlite()
    assert_not_v2_write_target(Path(target), config)


def filter_phase(
    primary: Sequence[Study1PlannedCall],
    extra: Sequence[Study1PlannedCall],
    phase: Study1Phase | str,
) -> list[Study1PlannedCall]:
    resolved = Study1Phase(phase)
    if resolved == Study1Phase.PRIMARY:
        return list(primary)
    if resolved == Study1Phase.REPEATS:
        return list(extra)
    raise Study1AuthorizationError(f"unsupported Study 1 phase: {resolved}")


class Study1Runner:
    """Plan and (in later tasks) execute Study 1 with hard paid-call gates."""

    def __init__(self, config: Study1ExperimentConfig | None = None) -> None:
        self.config = config or load_study1_config()
        self.models = load_models_config(
            self.config.resolve_path(self.config.models_config_path)
        )
        unknown = [
            alias
            for alias in self.config.models
            if alias not in STUDY1_MODEL_ALIASES
        ]
        if unknown:
            raise ValueError(f"Study 1 config contains non-roster models: {unknown}")
        extra_models = [
            alias
            for alias in self.models.models
            if alias not in STUDY1_MODEL_ALIASES
        ]
        # Extra models may exist in the shared historical models.yaml; they
        # must never be selected by this runner.
        self._forbidden_aliases = tuple(extra_models)

    def plan(self) -> tuple[list[Study1PlannedCall], list[Study1PlannedCall]]:
        primary, extra = build_call_plans(self.config)
        validate_call_plans(primary, extra, self.config)
        return primary, extra

    def run(
        self,
        *,
        phase: Study1Phase | str,
        dry_run: bool = True,
        allow_paid: bool = False,
        max_calls: int | None = None,
    ) -> dict[str, Any]:
        primary, extra = self.plan()
        selected = filter_phase(primary, extra, phase)
        authorize_study1_execution(
            planned_calls=len(selected),
            dry_run=dry_run,
            allow_paid=allow_paid,
            max_calls=max_calls,
            phase=phase,
            checkpoint_path=self.config.study1_sqlite(),
            config=self.config,
        )
        if not dry_run:
            resolved = Study1Phase(phase)
            if resolved == Study1Phase.REPEATS:
                raise Study1AuthorizationError(
                    "Task 003 does not authorize the 1,120 repeat-extra cells."
                )
            if resolved != Study1Phase.PRIMARY:
                raise Study1AuthorizationError(f"unsupported paid phase: {resolved}")
            if int(max_calls) != 2800:
                raise Study1AuthorizationError(
                    "Task 003 primary max_calls must be exactly 2800"
                )
            from .study1_primary import execute_primary_task

            payload = execute_primary_task()
            return {
                "task_id": "003_run_study1_primary",
                "dry_run": False,
                "allow_paid": True,
                "phase": "primary",
                "preflight_ok": bool(payload.get("preflight_ok")),
                "reused": payload.get("reused"),
                "new_initiated": payload.get("new_initiated"),
                "stopped_reason": payload.get("stopped_reason"),
                "wall_clock_seconds": payload.get("wall_clock_seconds"),
                "checkpoint_path": payload.get("checkpoint_path")
                or str(self.config.study1_sqlite()),
            }
        return {
            "dry_run": True,
            "allow_paid": False,
            "phase": Study1Phase(phase).value,
            "planned_calls": len(selected),
            "forbidden_model_aliases": list(self._forbidden_aliases),
            "checkpoint_path": str(self.config.study1_sqlite()),
            "historical_checkpoint_path": str(self.config.historical_sqlite()),
            "api_calls_made": 0,
        }
