from collections import Counter

from src.datasets import (
    allocate_category_quotas,
    normalise_mmlu_pro_row,
    select_category_stratified,
)


def _example(index: int, category: str):
    return normalise_mmlu_pro_row(
        {
            "question_id": index,
            "question": f"Question {index}",
            "options": ["x", "y", "z"],
            "answer": "B",
            "category": category,
            "cot_content": "must never be copied",
        },
        source_index=index,
        revision="a" * 40,
    )


def test_quota_allocation_is_balanced_and_exact() -> None:
    quotas = allocate_category_quotas({"a": 100, "b": 100, "c": 100}, 200, 7)
    assert sum(quotas.values()) == 200
    assert max(quotas.values()) - min(quotas.values()) <= 1


def test_sampling_is_deterministic_unique_and_stratified() -> None:
    source = [_example(index, ("a", "b", "c")[index % 3]) for index in range(300)]
    first, quotas = select_category_stratified(source, 200, 20260904)
    second, _ = select_category_stratified(source, 200, 20260904)
    assert [row.example_id for row in first] == [row.example_id for row in second]
    assert len({row.example_id for row in first}) == 200
    assert Counter(row.category for row in first) == Counter(quotas)
    assert "cot_content" not in first[0].model_dump()
