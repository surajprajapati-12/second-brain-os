
import json
from pathlib import Path

DATASET_PATH = (
    Path(__file__).parent
    / "fixtures"
    / "rag_eval_dataset.json"
)


def load_dataset():
    with DATASET_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def calculate_document_recall_at_k(cases, results_by_case, k=5):
    """Fraction of expected document IDs found in the top K."""
    expected_total = 0
    found_total = 0

    for case in cases:
        expected_ids = case["expected_document_ids"]
        if not expected_ids:
            continue

        expected_total += len(expected_ids)
        retrieved_ids = {
            result.get("document_id")
            for result in results_by_case.get(case["id"], [])[:k]
        }
        found_total += len(set(expected_ids) & retrieved_ids)

    return found_total / expected_total if expected_total else 0.0


def calculate_unsupported_source_rate(cases, results_by_case):
    """Fraction of unsupported questions returning any source."""
    cases_without_answers = [
        case for case in cases
        if not case["expected_document_ids"]
    ]

    if not cases_without_answers:
        return 0.0

    failures = sum(
        bool(results_by_case.get(case["id"], []))
        for case in cases_without_answers
    )
    return failures / len(cases_without_answers)


def test_generic_dataset_structure():
    dataset = load_dataset()
    documents = dataset["documents"]
    cases = dataset["cases"]

    document_ids = [doc["id"] for doc in documents]
    case_ids = [case["id"] for case in cases]

    assert len(document_ids) == len(set(document_ids))
    assert len(case_ids) == len(set(case_ids))

    known_ids = set(document_ids)
    for case in cases:
        assert set(case["expected_document_ids"]) <= known_ids


def test_dataset_covers_multiple_domains():
    dataset = load_dataset()

    domains = {doc["domain"] for doc in dataset["documents"]}
    case_types = {case["type"] for case in dataset["cases"]}

    assert len(domains) >= 5
    assert "unsupported" in case_types
    assert "cross_document" in case_types


def test_recall_supports_multiple_expected_documents():
    cases = [{
        "id": "multi",
        "expected_document_ids": ["doc_001", "doc_006"],
    }]
    results = {
        "multi": [
            {"document_id": "doc_001"},
            {"document_id": "doc_006"},
            {"document_id": "doc_002"},
        ]
    }

    assert calculate_document_recall_at_k(cases, results, k=2) == 1.0


def test_recall_detects_missing_expected_document():
    cases = [{
        "id": "multi",
        "expected_document_ids": ["doc_001", "doc_006"],
    }]
    results = {"multi": [{"document_id": "doc_001"}]}

    assert calculate_document_recall_at_k(cases, results) == 0.5


def test_unsupported_question_with_no_results():
    cases = [{
        "id": "unknown",
        "expected_document_ids": [],
    }]

    assert calculate_unsupported_source_rate(
        cases, {"unknown": []}
    ) == 0.0


def test_unsupported_question_with_irrelevant_results():
    cases = [{
        "id": "unknown",
        "expected_document_ids": [],
    }]
    results = {"unknown": [{"document_id": "doc_003"}]}

    assert calculate_unsupported_source_rate(cases, results) == 1.0
