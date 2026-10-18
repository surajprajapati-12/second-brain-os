
import json
from pathlib import Path


DATASET_PATH = Path(__file__).parent / "retrieval_eval_cases.json"


def load_evaluation_cases():
    with DATASET_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def calculate_recall_at_k(cases, results_by_case, k=5):
    """
    Recall@K for this dataset:
    Fraction of questions with an expected source
    where that source appears in the top K results.
    """
    eligible_cases = [
        case for case in cases
        if case["expected_to_find_source"]
    ]

    if not eligible_cases:
        return 0.0

    hits = 0

    for case in eligible_cases:
        results = results_by_case.get(case["id"], [])[:k]
        expected_filename = case["expected_filename"]

        if any(
            result.get("filename") == expected_filename
            for result in results
        ):
            hits += 1

    return hits / len(eligible_cases)


def calculate_unsupported_source_rate(cases, results_by_case):
    """
    Fraction of unsupported questions that return any source.
    Lower is better.
    """
    unsupported_cases = [
        case for case in cases
        if not case["expected_to_find_source"]
    ]

    if not unsupported_cases:
        return 0.0

    questions_with_sources = sum(
        bool(results_by_case.get(case["id"], []))
        for case in unsupported_cases
    )

    return questions_with_sources / len(unsupported_cases)


def test_dataset_contains_four_cases():
    cases = load_evaluation_cases()

    assert len(cases) == 4


def test_dataset_has_unique_ids():
    cases = load_evaluation_cases()
    ids = [case["id"] for case in cases]

    assert len(ids) == len(set(ids))


def test_recall_at_k_is_one_for_correct_mock_results():
    cases = load_evaluation_cases()

    results = {
        "orion_exact": [
            {"filename": "neon-upload-test.txt"}
        ],
        "orion_paraphrase": [
            {"filename": "neon-upload-test.txt"}
        ],
        "resume_qualification": [
            {"filename": "Suraj_Prajapati_Resume.pdf"}
        ],
        "unsupported_general_knowledge": []
    }

    assert calculate_recall_at_k(cases, results, k=5) == 1.0


def test_unsupported_source_rate_is_zero_for_empty_results():
    cases = load_evaluation_cases()

    results = {
        "orion_exact": [
            {"filename": "neon-upload-test.txt"}
        ],
        "orion_paraphrase": [
            {"filename": "neon-upload-test.txt"}
        ],
        "resume_qualification": [
            {"filename": "Suraj_Prajapati_Resume.pdf"}
        ],
        "unsupported_general_knowledge": []
    }

    assert calculate_unsupported_source_rate(cases, results) == 0.0


def test_unsupported_source_rate_detects_irrelevant_results():
    cases = load_evaluation_cases()

    results = {
        "unsupported_general_knowledge": [
            {"filename": "Suraj_Prajapati_Resume.pdf"}
        ]
    }

    assert calculate_unsupported_source_rate(cases, results) == 1.0
