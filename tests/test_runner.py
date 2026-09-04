import asyncio

from src.analysis import analyze
from src.checkpointing import CheckpointStore
from src.config import load_experiment_config, load_models_config
from src.model_adapters import PreparedRequest
from src.runner import ExperimentRunner
from src.schemas import BenchmarkExample, ModelResponse


class FakeAdapter:
    provider = "openai"
    api_model = "gpt-5.6-sol"

    def prepare_request(self, *, prompt: str) -> PreparedRequest:
        return PreparedRequest(
            provider="openai",
            api_style="responses",
            requested_model_id=self.api_model,
            payload={"model": self.api_model, "input": prompt},
        )

    async def generate(
        self,
        *,
        stage: str,
        prompt: str,
        request_key: str,
        allow_paid: bool,
        **_: object,
    ) -> ModelResponse:
        assert allow_paid
        raw = {
            "answer": '{"answer":"B"}',
            "confidence": '{"probability_correct":0.7}',
            "trust": '{"recommendation":"VERIFY"}',
        }[stage]
        return ModelResponse(
            model_alias="openai_gpt56_sol",
            provider="openai",
            requested_model_id=self.api_model,
            provider_model_id="gpt-5.6-sol-snapshot",
            stage=stage,
            request_key=request_key,
            raw_response=raw,
            input_tokens=10,
            output_tokens=4,
            total_tokens=14,
            latency_seconds=0.01,
        )


def _example() -> BenchmarkExample:
    return BenchmarkExample(
        example_id="mmlu_pro:test:1",
        dataset_name="mmlu_pro",
        category="math",
        question="What is one plus one?",
        choices={"A": "1", "B": "2", "C": "3"},
        correct_label="B",
        split="test",
        dataset_revision="a" * 40,
        source_index=1,
    )


def test_three_stages_and_resume(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "src.runner.create_adapter", lambda *_args, **_kwargs: FakeAdapter()
    )
    checkpoint = tmp_path / "pilot.sqlite3"
    models = load_models_config()
    config = load_experiment_config()
    with ExperimentRunner(
        examples=[_example()],
        checkpoint=checkpoint,
        models_config=models,
        experiment_config=config,
        run_id="pilot",
        concurrency=2,
    ) as runner:
        aliases = ["openai_gpt56_sol"]
        answers = asyncio.run(
            runner.run_answers(model_aliases=aliases, dry_run=False, allow_paid=True)
        )
        confidence = asyncio.run(
            runner.run_confidence(model_aliases=aliases, dry_run=False, allow_paid=True)
        )
        trust = asyncio.run(
            runner.run_trust(model_aliases=aliases, dry_run=False, allow_paid=True)
        )
        resumed = asyncio.run(
            runner.run_answers(model_aliases=aliases, dry_run=False, allow_paid=True)
        )
    assert answers.completed == 1
    assert confidence.completed == 1
    assert trust.completed == 4
    assert resumed.calls_needed == 0
    assert resumed.skipped_successes == 1
    with CheckpointStore(checkpoint) as store:
        assert store.counts()["success"] == 6
    analysis = analyze(
        checkpoint,
        output_directory=tmp_path / "results",
        paper_output_directory=tmp_path / "paper",
        n_resamples=10,
        make_plots=False,
        pricing_per_million_tokens={"openai_gpt56_sol": {"input": 4, "output": 20}},
    )
    assert len(analysis.direct_rows) == 4
    assert analysis.request_coverage == [
        {
            "stage": "answer",
            "model_id": "openai_gpt56_sol",
            "status": "success",
            "count": 1,
        },
        {
            "stage": "confidence",
            "model_id": "openai_gpt56_sol",
            "status": "success",
            "count": 1,
        },
        {
            "stage": "trust",
            "model_id": "openai_gpt56_sol",
            "status": "success",
            "count": 4,
        },
    ]
