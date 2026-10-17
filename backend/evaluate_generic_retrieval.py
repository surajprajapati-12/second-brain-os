
import json
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from statistics import mean

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "tests" / "fixtures" / "rag_eval_dataset.json"
API_URL = "http://127.0.0.1:8001"
LIMIT = 5
DATASET_VERSION = "v2"


def request_json(url, data=None, headers=None):
    body = None if data is None else json.dumps(data).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers=headers or {},
        method="GET" if data is None else "POST",
    )
    with urllib.request.urlopen(req, timeout=120) as response:
        payload = response.read()
        return json.loads(payload) if payload else {}


def upload_text(filename, content):
    boundary = "----SecondBrainEval" + uuid.uuid4().hex
    parts = [
        f"--{boundary}\r\n".encode(),
        (
            f'Content-Disposition: form-data; name="file"; '
            f'filename="{filename}"\r\n'
        ).encode(),
        b"Content-Type: text/plain; charset=utf-8\r\n\r\n",
        content.encode("utf-8"),
        b"\r\n",
        f"--{boundary}--\r\n".encode(),
    ]
    req = urllib.request.Request(
        f"{API_URL}/documents",
        data=b"".join(parts),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as response:
        return json.loads(response.read())


def get_field(item, names, default=None):
    for name in names:
        if name in item:
            return item[name]
    return default


def expected_ids(case):
    value = get_field(
        case,
        (
            "expected_document_ids",
            "expected_doc_ids",
            "expected_documents",
            "relevant_doc_ids",
            "relevant_documents",
            "expected_ids",
        ),
        [],
    )
    if value is None:
        return set()
    if isinstance(value, str):
        return {value}
    return set(value)


def main():
    if not DATASET_PATH.is_file():
        raise SystemExit(f"Dataset not found: {DATASET_PATH}")

    with DATASET_PATH.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    documents = dataset.get("documents", [])
    cases = dataset.get("cases", [])

    if not documents or not cases:
        raise SystemExit(
            "Dataset must contain non-empty 'documents' and 'cases' lists."
        )

    # Validate the dataset before making any API requests that modify data.
    document_ids = [doc.get("id") for doc in documents]

    if any(not isinstance(doc_id, str) or not doc_id for doc_id in document_ids):
        raise SystemExit("Every document must have a non-empty string ID.")

    if len(document_ids) != len(set(document_ids)):
        raise SystemExit("Dataset contains duplicate document IDs.")

    for document in documents:
        if not isinstance(document.get("text"), str) or not document["text"].strip():
            raise SystemExit(
                f"Document {document['id']} has missing or empty text."
            )

    known_ids = set(document_ids)

    for case in cases:
        missing_ids = expected_ids(case) - known_ids
        if missing_ids:
            raise SystemExit(
                f"Case {case.get('id', '<unknown>')} references "
                f"unknown document IDs: {sorted(missing_ids)}"
            )

    print(
        f"Validated dataset: {len(documents)} documents, "
        f"{len(cases)} cases."
    )

    health = request_json(f"{API_URL}/health")
    if health.get("status") != "healthy":
        raise SystemExit(f"Test API is not healthy: {health}")

    existing = request_json(f"{API_URL}/documents")
    by_filename = {row["filename"]: row for row in existing}

    id_map = {}

    for document in documents:
        source_id = document["id"]
        # Versioned filenames prevent collisions with older dataset text.
        filename = f"eval_{DATASET_VERSION}_{source_id}.txt"
        content = document["text"]

        if filename in by_filename:
            existing_id = by_filename[filename]["id"]
            stored = request_json(f"{API_URL}/documents/{existing_id}")

            if stored.get("extracted_text") != content:
                raise SystemExit(
                    f"{filename} exists but its text differs from the dataset. "
                    "No existing document was changed. Inspect the test DB."
                )

            if stored.get("indexing_status") != "indexed":
                raise SystemExit(
                    f"{filename} is not indexed; status="
                    f"{stored.get('indexing_status')}"
                )

            id_map[source_id] = existing_id
            print(f"Reusing {filename}")
        else:
            print(f"Uploading {filename}...")
            result = upload_text(filename, content)

            if result.get("indexing_status") != "indexed":
                raise SystemExit(
                    f"Indexing failed for {filename}: "
                    f"{result.get('indexing_status')}"
                )

            id_map[source_id] = result["id"]
            print(f"  Indexed chunks: {result.get('indexed_chunks')}")

    recalls = []
    reciprocal_ranks = []
    latencies = []
    unsupported_results = 0
    unsupported_cases = 0
    failed_cases = 0

    print("\n--- Retrieval evaluation ---")

    for case in cases:
        query = get_field(case, ("query", "question", "text"))

        if not isinstance(query, str) or not query.strip():
            raise SystemExit(f"Case has no valid query: {case}")

        expected_source_ids = expected_ids(case)
        expected_database_ids = {
            id_map[source_id] for source_id in expected_source_ids
        }

        start = time.perf_counter()

        try:
            result = request_json(
                f"{API_URL}/search",
                {"query": query, "limit": LIMIT},
                {"Content-Type": "application/json"},
            )
        except (urllib.error.URLError, TimeoutError) as exc:
            failed_cases += 1
            print(f"\nERROR: {query}\n  {exc}")
            continue

        latency = time.perf_counter() - start
        latencies.append(latency)
        results = result.get("results", [])
        retrieved_ids = [row.get("document_id") for row in results]

        if expected_database_ids:
            hits = sum(
                1 for doc_id in retrieved_ids
                if doc_id in expected_database_ids
            )
            recall = hits / len(expected_database_ids)

            first_relevant_rank = next(
                (
                    rank
                    for rank, doc_id in enumerate(retrieved_ids, start=1)
                    if doc_id in expected_database_ids
                ),
                None,
            )
            reciprocal_rank = (
                1 / first_relevant_rank if first_relevant_rank else 0.0
            )

            recalls.append(recall)
            reciprocal_ranks.append(reciprocal_rank)

            print(
                f"\nCase: {case.get('id', '<unknown>')}\n"
                f"Query: {query}\n"
                f"  Recall@{LIMIT}: {recall:.2f}\n"
                f"  Reciprocal rank: {reciprocal_rank:.2f}\n"
                f"  Retrieved: {[row.get('filename') for row in results]}\n"
                f"  Latency: {latency:.2f}s"
            )
        else:
            unsupported_cases += 1
            returned = bool(results)
            unsupported_results += int(returned)

            print(
                f"\nCase: {case.get('id', '<unknown>')}\n"
                f"Query: {query}\n"
                f"  Unsupported query returned results: {returned}\n"
                f"  Result count: {len(results)}\n"
                f"  Latency: {latency:.2f}s"
            )

    print("\n--- Summary ---")
    print(f"Dataset documents: {len(documents)}")
    print(f"Evaluation cases: {len(cases)}")
    print(f"Cases with request errors: {failed_cases}")

    if recalls:
        print(f"Mean Recall@{LIMIT}: {mean(recalls):.2%}")
        print(f"Mean reciprocal rank: {mean(reciprocal_ranks):.3f}")

    if unsupported_cases:
        print(f"Unsupported cases: {unsupported_cases}")
        print(
            "Unsupported-result rate: "
            f"{unsupported_results / unsupported_cases:.2%}"
        )

    if latencies:
        print(f"Average latency: {mean(latencies):.2f}s")


if __name__ == "__main__":
    main()
