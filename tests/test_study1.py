from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from src.cli import build_parser
from src.config import PROJECT_ROOT
from src.schemas import VerificationPayload
from src.study1_plan import (
    assert_manipulated_prompts_differ_only_by_number,
    build_call_plans,
    choose_audit_question_ids,
    validate_call_plans,
)
from src.study1_prompts import (
    build_study1_repair_prompt,
    build_study1_verification_prompt,
    format_displayed_probability,
    historical_visible_matches_v2_builder,
    parse_study1_verification_response,
)
from src.study1_runner import (
    Study1AuthorizationError,
    Study1Runner,
    authorize_study1_execution,
)
from src.study1_sample import (
    REPEAT_SEED,
    PRIMARY_SEED,
    hash_id_list,
    load_frozen_ids,
    load_historical_bundle,
    select_ids,
    select_study1_ids,
)
from src.study1_schemas import (
    HISTORICAL_V2_SQLITE,
    STUDY1_GRID_L10,
    STUDY1_GRID_L20,
    STUDY1_MODEL_ALIASES,
    DisplayCondition,
    Study1DecisionRecord,
    Study1Phase,
    grid_value,
    load_study1_config,
)
from src.v2_prompts import build_verification_prompt


def _example_choices() -> dict[str, str]:
    return {"A": "one", "B": "two", "C": "three"}


def test_study1_config_freezes_scientific_design() -> None:
    config = load_study1_config()
    assert config.models == list(STUDY1_MODEL_ALIASES)
    assert config.authority_condition == "ai_system"
    assert config.costs["error_costs"] == [10.0, 20.0]
    assert tuple(config.displayed_confidence_grids[10.0]) == STUDY1_GRID_L10
    assert tuple(config.displayed_confidence_grids[20.0]) == STUDY1_GRID_L20
    assert config.phases == ["primary", "repeats"]
    assert config.planned_calls["primary"] == 2800
    assert config.planned_calls["repeat_extra"] == 1120
    assert config.study1_sqlite().name != "v2.sqlite3"
    assert config.historical_sqlite() == PROJECT_ROOT / HISTORICAL_V2_SQLITE


def test_question_selection_is_deterministic() -> None:
    first = select_ids([f"id-{i}" for i in range(500)], seed=PRIMARY_SEED, size=100)
    second = select_ids([f"id-{i}" for i in range(500)], seed=PRIMARY_SEED, size=100)
    assert first == second
    assert len(set(first)) == 100
    other = select_ids([f"id-{i}" for i in range(500)], seed=REPEAT_SEED, size=100)
    assert first != other


def test_frozen_ids_match_recompute_and_hashes() -> None:
    selected, repeats, selected_hash, repeat_hash = load_frozen_ids()
    assert len(selected) == 100
    assert len(repeats) == 20
    assert set(repeats).issubset(set(selected))
    assert selected_hash == hash_id_list(selected)
    assert repeat_hash == hash_id_list(repeats)
    recomputed, recomputed_repeats, _, _ = select_study1_ids()
    assert selected == recomputed
    assert repeats == recomputed_repeats


def test_grids_are_frozen_by_l() -> None:
    assert STUDY1_GRID_L10 == (0.80, 0.88, 0.89, 0.91, 0.99)
    assert STUDY1_GRID_L20 == (0.90, 0.93, 0.94, 0.96, 0.99)
    assert grid_value(10.0, DisplayCondition.MANIPULATED_3) == 0.89
    assert grid_value(10.0, DisplayCondition.MANIPULATED_4) == 0.91
    assert grid_value(20.0, DisplayCondition.MANIPULATED_3) == 0.94
    assert grid_value(20.0, DisplayCondition.MANIPULATED_4) == 0.96


def test_true_visible_matches_historical_ai_authority_wording() -> None:
    choices = _example_choices()
    assert historical_visible_matches_v2_builder(
        question="What is one plus one?",
        choices=choices,
        frozen_answer="B",
        probability=0.73,
        verification_cost=1.0,
        error_cost=10.0,
    )
    historical_hidden = build_verification_prompt(
        question="What is one plus one?",
        choices=choices,
        answer_label="B",
        probability_correct=0.73,
        decision_owner="ai_system",
        confidence_visibility="hidden",
        verification_cost=1.0,
        error_cost=10.0,
    )
    study1_hidden = build_study1_verification_prompt(
        question="What is one plus one?",
        choices=choices,
        frozen_answer="B",
        verification_cost=1.0,
        error_cost=10.0,
        displayed_confidence=None,
    )
    assert study1_hidden == historical_hidden
    assert "The AI previously estimated a " not in study1_hidden


def test_hidden_prompt_has_no_confidence_sentence() -> None:
    prompt = build_study1_verification_prompt(
        question="Q?",
        choices=_example_choices(),
        frozen_answer="B",
        verification_cost=1.0,
        error_cost=20.0,
        displayed_confidence=None,
    )
    assert "The AI previously estimated a " not in prompt
    assert "HUMAN USER" not in prompt
    assert "The AI SYSTEM controls" in prompt


def test_manipulated_prompt_shows_assigned_not_reported() -> None:
    reported = 0.73
    displayed = 0.89
    prompt = build_study1_verification_prompt(
        question="Q?",
        choices=_example_choices(),
        frozen_answer="B",
        verification_cost=1.0,
        error_cost=10.0,
        displayed_confidence=displayed,
    )
    shown = format_displayed_probability(displayed)
    not_shown = format_displayed_probability(reported)
    assert (
        f"The AI previously estimated a {shown} probability that this frozen answer is correct."
        in prompt
    )
    assert (
        f"The AI previously estimated a {not_shown} probability that this frozen answer is correct."
        not in prompt
    )
    assert "Frozen answer:\nB" in prompt


def test_action_schema_remains_binary() -> None:
    parsed = parse_study1_verification_response('{"action":"VERIFY_FIRST"}')
    assert parsed.action.value == "VERIFY_FIRST"
    with pytest.raises(ValueError):
        parse_study1_verification_response('{"action":"RELY"}')
    with pytest.raises(ValueError):
        parse_study1_verification_response('{"action":"VERIFY_FIRST","why":"x"}')
    repair = build_study1_repair_prompt("PROMPT", "not-json")
    assert "Do not reconsider the action" in repair
    assert '{"action":"USE_UNVERIFIED"}' in repair


def test_schema_keeps_reported_and_displayed_separate() -> None:
    record = Study1DecisionRecord(
        phase=Study1Phase.PRIMARY,
        run_id="test",
        request_key="key",
        question_id="q",
        model_id="openai_gpt56_sol",
        model_alias="openai_gpt56_sol",
        model_endpoint="gpt-5.6-sol",
        provider="openai",
        L=10.0,
        reported_confidence=0.73,
        displayed_confidence=0.89,
        display_condition=DisplayCondition.MANIPULATED_3,
        display_source_condition="assigned_grid",
        frozen_answer="B",
        stage1_correct=True,
        prompt_version="v",
        prompt_family="f",
        prompt_hash="h",
        selected_sample_hash="s",
        repeat_index=0,
        attempt_number=1,
        historical_answer_request_key="a",
        historical_confidence_request_key="c",
    )
    dumped = record.model_dump()
    assert dumped["reported_confidence"] == 0.73
    assert dumped["displayed_confidence"] == 0.89
    assert dumped["reported_confidence"] != dumped["displayed_confidence"]


def test_hidden_schema_requires_null_displayed_confidence() -> None:
    with pytest.raises(ValueError):
        Study1DecisionRecord(
            phase=Study1Phase.PRIMARY,
            run_id="test",
            request_key="key",
            question_id="q",
            model_id="openai_gpt56_sol",
            model_alias="openai_gpt56_sol",
            model_endpoint="gpt-5.6-sol",
            provider="openai",
            L=10.0,
            reported_confidence=0.73,
            displayed_confidence=0.73,
            display_condition=DisplayCondition.HIDDEN,
            display_source_condition="none",
            frozen_answer="B",
            stage1_correct=True,
            prompt_version="v",
            prompt_family="f",
            prompt_hash="h",
            selected_sample_hash="s",
            repeat_index=0,
            attempt_number=1,
            historical_answer_request_key="a",
            historical_confidence_request_key="c",
        )


def test_call_plan_counts_and_uniqueness() -> None:
    primary, extra = build_call_plans()
    validate_call_plans(primary, extra)
    assert len(primary) == 2800
    assert len(extra) == 1120
    reported_by_cell = {}
    for row in primary:
        key = (row.question_id, row.model_alias)
        reported_by_cell.setdefault(key, row.reported_confidence)
        assert row.reported_confidence == reported_by_cell[key]
        assert row.frozen_answer
    assert_manipulated_prompts_differ_only_by_number(primary)
    high_id, low_id = choose_audit_question_ids(primary)
    assert high_id != low_id


def test_historical_confidence_loaded_from_v2_readonly() -> None:
    selected, _repeats, _, _ = load_frozen_ids()
    bundle = load_historical_bundle(selected[:3])
    assert len(bundle) == 6
    for (_qid, alias), row in bundle.items():
        assert alias in STUDY1_MODEL_ALIASES
        assert 0.0 <= row["reported_confidence"] <= 1.0
        assert row["historical_probability_correct_field"] == "probability_correct"
        assert row["frozen_answer"] in list("ABCDEFGHIJ")


def test_study1_writer_does_not_point_at_v2_sqlite() -> None:
    config = load_study1_config()
    assert config.study1_sqlite().resolve() != config.historical_sqlite().resolve()
    runner = Study1Runner(config)
    summary = runner.run(phase="primary", dry_run=True, allow_paid=False)
    assert summary["api_calls_made"] == 0
    assert "v2.sqlite3" not in summary["checkpoint_path"]


def test_paid_calls_disabled_by_default_and_cap_enforced() -> None:
    with pytest.raises(Study1AuthorizationError):
        authorize_study1_execution(
            planned_calls=10, dry_run=False, allow_paid=False, max_calls=10, phase="primary"
        )
    with pytest.raises(Study1AuthorizationError, match="max_calls"):
        authorize_study1_execution(
            planned_calls=10, dry_run=False, allow_paid=True, max_calls=None, phase="primary"
        )
    with pytest.raises(Study1AuthorizationError, match="exceed authorized cap"):
        authorize_study1_execution(
            planned_calls=2800, dry_run=False, allow_paid=True, max_calls=5, phase="primary"
        )
    runner = Study1Runner()
    with pytest.raises(Study1AuthorizationError, match="2800"):
        runner.run(phase="primary", dry_run=False, allow_paid=True, max_calls=5)
    with pytest.raises(Study1AuthorizationError, match="1120"):
        runner.run(phase="repeats", dry_run=False, allow_paid=True, max_calls=5)


def test_cli_study1_run_requires_explicit_mode() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["study1-run", "--phase", "primary"])
    args = parser.parse_args(["study1-run", "--phase", "repeats", "--dry-run"])
    assert args.dry_run is True
    assert args.yes is False
    smoke = parser.parse_args(
        [
            "study1-smoke",
            "--yes",
            "--max-scientific-cells",
            "12",
            "--max-provider-attempts",
            "20",
        ]
    )
    assert smoke.yes is True
    assert smoke.max_scientific_cells == 12
    assert smoke.max_provider_attempts == 20


def test_smoke_selects_first_frozen_id_and_twelve_cells() -> None:
    from src.study1_smoke import (
        SMOKE_PROVIDER_ATTEMPT_CAP,
        SMOKE_SCIENTIFIC_CELL_CAP,
        ProviderAttemptBudget,
        ProviderAttemptCapReached,
        select_smoke_calls,
        smoke_question_id,
        validate_preflight,
    )
    from src.study1_schemas import load_study1_config

    config = load_study1_config()
    question_id = smoke_question_id(config)
    assert question_id == "mmlu_pro:test:7552"
    selected, _repeats, _, _ = load_frozen_ids(config)
    assert question_id == selected[0]
    qid, calls = select_smoke_calls(config)
    assert qid == question_id
    assert len(calls) == 12
    assert {row.model_alias for row in calls} == {
        "openai_gpt56_sol",
        "anthropic_sonnet5",
    }
    assert [row.L for row in calls] == [10.0, 10.0, 10.0, 10.0, 20.0, 20.0] * 2
    preflight = validate_preflight(
        question_id=qid,
        calls=calls,
        config=config,
        scientific_cell_cap=SMOKE_SCIENTIFIC_CELL_CAP,
        provider_attempt_cap=SMOKE_PROVIDER_ATTEMPT_CAP,
    )
    assert preflight["ok"], preflight["errors"]
    thirteenth = validate_preflight(
        question_id=qid,
        calls=list(calls) + [calls[0]],
        config=config,
        scientific_cell_cap=12,
        provider_attempt_cap=20,
    )
    assert not thirteenth["ok"]
    budget = ProviderAttemptBudget(20)
    for _ in range(20):
        budget.consume()
    with pytest.raises(ProviderAttemptCapReached):
        budget.consume()
    assert budget.used == 20


def test_smoke_paid_path_requires_exact_caps() -> None:
    import asyncio

    from src.study1_smoke import run_smoke_test

    with pytest.raises(Study1AuthorizationError, match="allow_paid"):
        asyncio.run(run_smoke_test(allow_paid=False))
    with pytest.raises(Study1AuthorizationError, match="exactly 12"):
        asyncio.run(run_smoke_test(allow_paid=True, scientific_cell_cap=13))
    with pytest.raises(Study1AuthorizationError, match="exactly 20"):
        asyncio.run(
            run_smoke_test(
                allow_paid=True, scientific_cell_cap=12, provider_attempt_cap=21
            )
        )


def test_primary_preflight_reuses_identical_smoke_keys() -> None:
    from src.study1_primary import validate_primary_preflight

    preflight = validate_primary_preflight()
    assert preflight["ok"], preflight["errors"]
    assert preflight["planned_primary_cells"] == 2800
    assert preflight["repeat_extra_cells_excluded"] == 1120
    assert preflight["smoke_keys_are_primary_keys"] is True
    assert preflight["existing_smoke_cells_reused"] == 12
    assert preflight["existing_successes_reused"] >= 12
    assert preflight["new_scientific_calls"] == 2800 - preflight["existing_successes_reused"]
    assert preflight["selected_question_hash"] == (
        "badd6938e5ded12c9dd62733426e1db26d9843bb6a2321a4e4c9eb7e3547fe94"
    )
    assert preflight["smoke_question_id"] == "mmlu_pro:test:7552"
    assert preflight["provider_attempt_cap"] == math.ceil(
        preflight["new_scientific_calls"] * 1.10
    )


def test_repeats_preflight_uses_frozen_hash_and_excludes_primary() -> None:
    from src.study1_repeats import audit_task003_parse_repairs, validate_repeats_preflight

    audit = audit_task003_parse_repairs()
    assert audit["distinct_repair_cells"] == 52
    assert audit["by_model"].get("openai_gpt56_sol", 0) == 0
    assert audit["by_model"]["anthropic_sonnet5"] == 52
    assert audit["proceed"] is True
    assert audit["stop_before_paid"] is False
    preflight = validate_repeats_preflight()
    assert preflight["ok"], preflight["errors"]
    assert preflight["planned_repeat_extra_cells"] == 1120
    assert preflight["planned_primary_cells"] == 2800
    assert preflight["question_count"] == 20
    assert preflight["primary_successes_untouched_target"] == 2800
    assert preflight["new_scientific_calls"] == 1120 - preflight["existing_repeat_successes_reused"]
    assert preflight["repeat_question_hash"] == (
        "45e06fe9daf40dd1cbd9cbf91b49cef09c5d924df17f2cf2bf2f203090c26d38"
    )
    assert preflight["provider_attempt_cap"] == math.ceil(
        preflight["new_scientific_calls"] * 1.10
    )
