from src.analysis import analyze


def _records() -> list[dict]:
    rows: list[dict] = []
    for index in range(20):
        example_id = f"q-{index}"
        correct = index % 3 != 0
        rows.extend(
            [
                {
                    "stage": "answer",
                    "run_id": "pilot",
                    "model_id": "model-a",
                    "example_id": example_id,
                    "category": "math",
                    "question": f"Question {index}",
                    "answer_label": "A",
                    "correct_label": "A" if correct else "B",
                    "is_correct": correct,
                },
                {
                    "stage": "confidence",
                    "run_id": "pilot",
                    "model_id": "model-a",
                    "example_id": example_id,
                    "probability_correct": 0.8 if correct else 0.6,
                },
            ]
        )
        for loss in (2, 5, 10, 20):
            rows.append(
                {
                    "stage": "trust",
                    "run_id": "pilot",
                    "model_id": "model-a",
                    "example_id": example_id,
                    "verification_cost": 1,
                    "error_cost": loss,
                    "recommendation": "RELY" if loss == 2 else "VERIFY",
                }
            )
    return rows


def test_analysis_writes_private_diagnostics_and_public_outputs(tmp_path) -> None:
    result = analyze(
        _records(),
        output_directory=tmp_path / "results",
        paper_output_directory=tmp_path / "paper",
        seed=20260904,
        n_resamples=20,
        make_plots=False,
    )
    assert len(result.direct_rows) == 80
    assert result.model_summary[0]["answer_accuracy"] == 13 / 20
    assert result.policy_differences
    assert (tmp_path / "paper/model_summary.csv").exists()
    assert (tmp_path / "paper/paired_policy_differences.csv").exists()
    assert (tmp_path / "results/diagnostics/high_stakes_unsafe_reliance.csv").exists()
    scored = (tmp_path / "paper/scored_decisions.csv").read_text()
    assert "Question 1" not in scored
