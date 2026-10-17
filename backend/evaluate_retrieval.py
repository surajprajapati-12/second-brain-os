
import json
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


BASE_URL = "http://127.0.0.1:8000"
DATASET_PATH = (
    Path(__file__).parent
    / "tests"
    / "retrieval_eval_cases.json"
)
TOP_K = 5


def load_cases():
    with DATASET_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def search_documents(question):
    payload = json.dumps({
        "query": question,
        "limit": TOP_K
    }).encode("utf-8")

    request = Request(
        f"{BASE_URL}/search",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urlopen(request, timeout=60) as response:
        data = json.loads(response.read().decode("utf-8"))

    return data.get("results", [])


def main():
    cases = load_cases()
    evaluated = []
    total_latency = 0.0

    print(f"Evaluating {len(cases)} questions against {BASE_URL}")
    print("-" * 65)

    for case in cases:
        question = case["question"]

        try:
            started = time.perf_counter()
            results = search_documents(question)
            latency = time.perf_counter() - started
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            print(f"\nEvaluation failed for {case['id']}: {exc}")
            print("Check that the local backend is running.")
            sys.exit(1)

        total_latency += latency
        expected = case["expected_filename"]

        filenames = [
            result.get("filename")
            for result in results
        ]

        hit = (
            expected is not None
            and expected in filenames[:TOP_K]
        )

        unsupported = not case["expected_to_find_source"]
        unsupported_has_sources = unsupported and bool(results)

        evaluated.append({
            "id": case["id"],
            "expected_to_find_source": case[
                "expected_to_find_source"
            ],
            "hit": hit,
            "has_sources": bool(results),
            "unsupported_has_sources": unsupported_has_sources,
        })

        status = (
            "HIT" if hit
            else "NO_SOURCE_EXPECTED" if unsupported and not results
            else "MISS"
        )

        print(f"\n[{status}] {case['id']}")
        print(f"Question: {question}")
        print(f"Expected: {expected or 'no source'}")
        print(f"Retrieved: {filenames or 'no results'}")
        print(f"Latency: {latency:.2f}s")

    relevant_cases = [
        item for item in evaluated
        if item["expected_to_find_source"]
    ]
    unsupported_cases = [
        item for item in evaluated
        if not item["expected_to_find_source"]
    ]

    recall_at_k = (
        sum(item["hit"] for item in relevant_cases)
        / len(relevant_cases)
        if relevant_cases else 0.0
    )

    unsupported_source_rate = (
        sum(item["unsupported_has_sources"] for item in unsupported_cases)
        / len(unsupported_cases)
        if unsupported_cases else 0.0
    )

    print("\n" + "=" * 65)
    print("EVALUATION SUMMARY")
    print("=" * 65)
    print(
        f"Recall@{TOP_K}: "
        f"{recall_at_k:.1%} "
        f"({sum(item['hit'] for item in relevant_cases)}"
        f"/{len(relevant_cases)} relevant questions)"
    )
    print(
        "Unsupported-source rate: "
        f"{unsupported_source_rate:.1%} "
        f"({sum(item['unsupported_has_sources'] for item in unsupported_cases)}"
        f"/{len(unsupported_cases)} unsupported questions)"
    )
    print(f"Average search latency: {total_latency / len(evaluated):.2f}s")


if __name__ == "__main__":
    main()
