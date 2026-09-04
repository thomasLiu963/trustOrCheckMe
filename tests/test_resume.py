import sqlite3

from src.checkpointing import CheckpointStore, deterministic_request_key


def test_request_key_is_stable_and_stake_sensitive() -> None:
    common = {
        "stage": "trust",
        "dataset": "mmlu_pro",
        "example_id": "example-1",
        "model_id": "gpt-5.6-sol",
        "prompt_version": "trust-v1",
    }
    first = deterministic_request_key(**common, stake={"C": 1, "L": 2})
    second = deterministic_request_key(**common, stake={"L": 2, "C": 1})
    higher = deterministic_request_key(**common, stake={"C": 1, "L": 20})
    assert first == second
    assert first != higher


def test_success_is_immediately_persisted_and_skipped(tmp_path) -> None:
    path = tmp_path / "checkpoint.sqlite3"
    key = "answer:key"
    with CheckpointStore(path) as store:
        store.register_request(
            request_key=key,
            run_id="pilot",
            stage="answer",
            dataset="mmlu_pro",
            example_id="example-1",
            model_alias="model-a",
            requested_model_id="api-model",
            prompt_version="answer-v1",
        )
        store.record_attempt(
            request_key=key,
            attempt_kind="generation",
            success=True,
            raw_output='{"answer":"A"}',
        )
        store.mark_success(key, {"answer_label": "A", "is_correct": True})
        assert store.should_run(key) is False

    with CheckpointStore(path) as reopened:
        assert reopened.get_record(key) == {
            "answer_label": "A",
            "is_correct": True,
        }
        assert reopened.should_run(key) is False

    connection = sqlite3.connect(path)
    try:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        connection.close()


def test_failed_requests_only_retry_when_requested(tmp_path) -> None:
    with CheckpointStore(tmp_path / "checkpoint.sqlite3") as store:
        store.register_request(
            request_key="failed:key",
            run_id="pilot",
            stage="answer",
            dataset="mmlu_pro",
            example_id="example-1",
            model_alias="model-a",
            requested_model_id="api-model",
            prompt_version="answer-v1",
        )
        store.mark_failed("failed:key", "permanent")
        assert store.should_run("failed:key") is False
        assert store.should_run("failed:key", retry_failed=True) is True
