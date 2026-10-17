import pytest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

import backend.main as main
from backend.main import app, filter_and_deduplicate_rows


# ---------------------------------------------------------------------------
# Test configuration and reusable fixtures
# ---------------------------------------------------------------------------

client = TestClient(app)

FAKE_EMBEDDING = [0.1] * 384

FAKE_ROW = (
    "test-document-id",
    "phoenix-test.txt",
    0,
    "Project Phoenix uses a blue comet as its secret project code.",
    0.2456,
    None,
    0.016129,
)


def make_fake_database(rows):
    """Create a mock PostgreSQL connection returning the supplied rows."""
    connection = MagicMock()
    connection.__enter__.return_value.execute.return_value.fetchall.return_value = rows
    return connection


def make_search_row(
    document_id,
    filename,
    chunk_index,
    text,
    distance,
    keyword_rank,
    hybrid_score=0.01,
):
    """Create a row matching the hybrid retrieval query's output schema."""
    return (
        document_id,
        filename,
        chunk_index,
        text,
        distance,
        keyword_rank,
        hybrid_score,
    )


# ---------------------------------------------------------------------------
# Health and request validation
# ---------------------------------------------------------------------------

def test_health_returns_200():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_search_rejects_empty_query():
    response = client.post("/search", json={"query": "", "limit": 5})

    assert response.status_code == 422


def test_search_rejects_limit_above_maximum():
    response = client.post(
        "/search",
        json={"query": "test query", "limit": 11},
    )

    assert response.status_code == 422


def test_ask_rejects_empty_question():
    response = client.post("/ask", json={"question": "", "limit": 5})

    assert response.status_code == 422


def test_ask_rejects_limit_above_maximum():
    response = client.post(
        "/ask",
        json={"question": "test question", "limit": 11},
    )

    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Search endpoint
# ---------------------------------------------------------------------------

@patch("backend.main.get_postgres_connection")
@patch("backend.main.generate_embeddings")
def test_search_returns_relevant_document(mock_embeddings, mock_db):
    mock_embeddings.return_value = [FAKE_EMBEDDING]
    mock_db.return_value = make_fake_database([FAKE_ROW])

    response = client.post(
        "/search",
        json={
            "query": "What does phoenix-test.txt say?",
            "limit": 5,
        },
    )

    assert response.status_code == 200

    data = response.json()
    assert data["count"] == 1
    assert data["results"][0]["filename"] == "phoenix-test.txt"
    assert "blue comet" in data["results"][0]["text"]
    assert data["results"][0]["similarity_score"] is not None
    assert data["results"][0]["hybrid_score"] == FAKE_ROW[6]


# ---------------------------------------------------------------------------
# Ask endpoint
# ---------------------------------------------------------------------------

@patch("backend.main.genai.Client")
@patch("backend.main.get_postgres_connection")
@patch("backend.main.generate_embeddings")
def test_ask_returns_answer_and_sources(
    mock_embeddings,
    mock_db,
    mock_genai_client,
    monkeypatch,
):
    monkeypatch.setenv("GEMINI_API_KEY", "test-api-key")

    mock_embeddings.return_value = [FAKE_EMBEDDING]
    mock_db.return_value = make_fake_database([FAKE_ROW])

    fake_client = MagicMock()
    fake_client.models.generate_content.return_value.text = (
        "Project Phoenix uses a blue comet as its secret project code."
    )
    mock_genai_client.return_value = fake_client

    response = client.post(
        "/ask",
        json={
            "question": "What does phoenix-test.txt say?",
            "limit": 5,
        },
    )

    assert response.status_code == 200

    data = response.json()
    assert data["question"] == "What does phoenix-test.txt say?"
    assert "blue comet" in data["answer"]
    assert len(data["sources"]) >= 1
    assert data["sources"][0]["filename"] == "phoenix-test.txt"


def test_ask_deduplicates_displayed_sources_but_keeps_all_context(
    monkeypatch,
):
    """One source card per document, while Gemini receives both chunks."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-api-key")

    rows = [
        make_search_row(
            "doc-1",
            "orion.txt",
            0,
            "Project Orion uses a silver eagle.",
            0.10,
            1,
            0.03,
        ),
        make_search_row(
            "doc-1",
            "orion.txt",
            1,
            "The silver eagle is the internal launch code.",
            0.15,
            2,
            0.02,
        ),
    ]

    monkeypatch.setattr(
        main,
        "embed_query_or_http_error",
        lambda _: FAKE_EMBEDDING,
    )
    monkeypatch.setattr(
        main,
        "retrieve_hybrid_candidates",
        lambda *_args, **_kwargs: rows,
    )

    captured = {}

    fake_client = MagicMock()

    def fake_generate_content(*, model, contents):
        captured["prompt"] = contents
        response = MagicMock()
        response.text = "Project Orion's code is a silver eagle."
        return response

    fake_client.models.generate_content.side_effect = fake_generate_content
    monkeypatch.setattr(main.genai, "Client", lambda **_: fake_client)

    response = client.post(
        "/ask",
        json={"question": "What is Project Orion's code?", "limit": 5},
    )

    assert response.status_code == 200

    data = response.json()

    # Both chunks are available to Gemini as answer context.
    assert "Project Orion uses a silver eagle." in captured["prompt"]
    assert "The silver eagle is the internal launch code." in captured["prompt"]

    # The displayed source list contains the document only once.
    assert len(data["sources"]) == 1
    assert data["sources"][0]["document_id"] == "doc-1"
    assert data["sources"][0]["filename"] == "orion.txt"


# ---------------------------------------------------------------------------
# Retrieval filtering and deduplication helpers
# ---------------------------------------------------------------------------

def test_deduplication_removes_repeated_chunks_within_document():
    rows = [
        make_search_row("doc-1", "notes.txt", 0, "Same text", 0.1, 1),
        make_search_row("doc-1", "notes.txt", 1, "  Same   text ", 0.2, 2),
    ]

    results = filter_and_deduplicate_rows(rows)

    assert len(results) == 1
    assert results[0][2] == 0


def test_deduplication_preserves_identical_text_from_different_documents():
    rows = [
        make_search_row("doc-1", "notes.txt", 0, "Same text", 0.1, 1),
        make_search_row("doc-2", "notes.txt", 0, "Same text", 0.2, 2),
    ]

    results = filter_and_deduplicate_rows(rows)

    assert len(results) == 2
    assert {row[0] for row in results} == {"doc-1", "doc-2"}


def test_deduplication_filters_weak_vector_only_matches():
    rows = [
        make_search_row("doc-1", "notes.txt", 0, "Weak match", 0.8, None),
    ]

    results = filter_and_deduplicate_rows(rows)

    assert results == []


def test_deduplication_keeps_keyword_match_without_vector_distance():
    rows = [
        make_search_row("doc-1", "notes.txt", 0, "Keyword match", None, 1),
    ]

    results = filter_and_deduplicate_rows(rows)

    assert len(results) == 1
    assert results[0][3] == "Keyword match"
def test_deduplication_filters_unrelated_vector_only_match():
    row = make_search_row(
        "doc-rahul",
        "My name is Rahul!.txt",
        0,
        "I like learning new things every day.",
        0.3655,
        None,
    )

    results = filter_and_deduplicate_rows([row])

    assert results == []


def test_deduplication_keeps_relevant_paraphrase_match():
    row = make_search_row(
        "doc-orion",
        "neon-upload-test.txt",
        0,
        "Project Orion uses a silver eagle as its internal launch code.",
        0.1396,
        None,
    )

    results = filter_and_deduplicate_rows([row])

    assert len(results) == 1
    assert results[0][1] == "neon-upload-test.txt"


def test_deduplication_keeps_keyword_match_above_threshold():
    row = make_search_row(
        "doc-contact",
        "Suraj_Prajapati_Resume.pdf",
        0,
        "Contact information is available in the resume.",
        0.3655,
        1,
    )

    results = filter_and_deduplicate_rows([row])

    assert len(results) == 1
# ---------------------------------------------------------------------------
# Document upload regression tests
# ---------------------------------------------------------------------------

def test_upload_rejects_unsupported_extension():
    response = client.post(
        "/documents",
        files={"file": ("notes.exe", b"some content", "application/octet-stream")},
    )

    assert response.status_code == 415


def test_upload_rejects_empty_file():
    response = client.post(
        "/documents",
        files={"file": ("empty.txt", b"", "text/plain")},
    )

    assert response.status_code == 400


def test_upload_rejects_oversized_file():
    oversized_content = b"x" * (main.MAX_FILE_SIZE + 1)

    response = client.post(
        "/documents",
        files={"file": ("large.txt", oversized_content, "text/plain")},
    )

    assert response.status_code == 413


def test_upload_closes_file_after_validation_error(monkeypatch):
    original_extract_text = main.extract_text
    captured = {}

    def capture_file_and_raise(filename, file_bytes):
        raise main.HTTPException(
            status_code=400,
            detail="Simulated extraction failure",
        )

    # Capture the UploadFile passed to the endpoint by wrapping its read method.
    from starlette.datastructures import UploadFile

    original_read = UploadFile.read

    async def tracked_read(upload_file, *args, **kwargs):
        captured["file"] = upload_file
        return await original_read(upload_file, *args, **kwargs)

    monkeypatch.setattr(UploadFile, "read", tracked_read)
    monkeypatch.setattr(main, "extract_text", capture_file_and_raise)

    try:
        response = client.post(
            "/documents",
            files={"file": ("notes.txt", b"valid text", "text/plain")},
        )
    finally:
        monkeypatch.setattr(main, "extract_text", original_extract_text)

    assert response.status_code == 400
    assert captured["file"].file.closed
# ---------------------------------------------------------------------------
# Indexing failure recovery
# ---------------------------------------------------------------------------

@patch("backend.main.get_postgres_connection")
def test_upload_keeps_document_pending_when_indexing_fails(
    mock_db,
    monkeypatch,
):
    mock_db.return_value = make_fake_database([])

    def fail_indexing(*args, **kwargs):
        raise RuntimeError("Simulated embedding service failure")

    monkeypatch.setattr(main, "index_document", fail_indexing)

    response = client.post(
        "/documents",
        files={
            "file": (
                "pending-test.txt",
                b"This document should remain saved if indexing fails.",
                "text/plain",
            )
        },
    )

    assert response.status_code == 201

    data = response.json()
    assert data["filename"] == "pending-test.txt"
    assert data["indexing_status"] == "pending"
    assert data["indexed_chunks"] == 0

    # Metadata storage and indexing-status update are both attempted.
    assert mock_db.call_count == 2
@patch("backend.main.get_postgres_connection")
def test_upload_returns_503_when_metadata_storage_fails(mock_db):
    mock_db.return_value.__enter__.return_value.execute.side_effect = (
        main.psycopg.OperationalError("Simulated database outage")
    )

    response = client.post(
        "/documents",
        files={
            "file": (
                "database-failure.txt",
                b"This document cannot be stored.",
                "text/plain",
            )
        },
    )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Document storage is temporarily unavailable."
    )

    # Indexing must not run if metadata storage fails.
    assert mock_db.call_count == 1
def test_index_document_rejects_embedding_count_mismatch(monkeypatch):
    chunks = ["First chunk", "Second chunk"]

    monkeypatch.setattr(main, "split_text", lambda _: chunks)
    monkeypatch.setattr(
        main,
        "generate_embeddings",
        lambda _: [[0.1] * main.EMBEDDING_DIMENSIONS],
    )

    with pytest.raises(RuntimeError, match="Embedding count mismatch"):
        main.index_document("test-document-id", "test.txt", "Some text")
@patch("backend.main.get_postgres_connection")
def test_upload_successfully_saves_and_indexes_document(
    mock_db,
    monkeypatch,
):
    mock_db.return_value = make_fake_database([])

    indexed = {}

    def fake_index_document(document_id, filename, text):
        indexed["document_id"] = document_id
        indexed["filename"] = filename
        indexed["text"] = text
        return 2

    monkeypatch.setattr(main, "index_document", fake_index_document)

    content = b"Second Brain OS integration test document."

    response = client.post(
        "/documents",
        files={"file": ("integration.txt", content, "text/plain")},
    )

    assert response.status_code == 201

    data = response.json()
    assert data["filename"] == "integration.txt"
    assert data["file_type"] == "txt"
    assert data["size_bytes"] == len(content)
    assert data["indexing_status"] == "indexed"
    assert data["indexed_chunks"] == 2

    # The indexing function receives the extracted document text.
    assert indexed["document_id"] == data["id"]
    assert indexed["filename"] == "integration.txt"
    assert indexed["text"] == content.decode("utf-8")

    # One database operation saves metadata; another updates its status.
    assert mock_db.call_count == 2
def test_ask_returns_fallback_when_no_relevant_chunks_exist(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-api-key")

    monkeypatch.setattr(
        main,
        "embed_query_or_http_error",
        lambda _: FAKE_EMBEDDING,
    )
    monkeypatch.setattr(
        main,
        "retrieve_hybrid_candidates",
        lambda *_args, **_kwargs: [],
    )

    with patch("backend.main.genai.Client") as mock_genai:
        response = client.post(
            "/ask",
            json={
                "question": "What is the secret launch code?",
                "limit": 5,
            },
        )

    assert response.status_code == 200

    data = response.json()
    assert data["question"] == "What is the secret launch code?"
    assert data["answer"] == (
        "I couldn't find any indexed document content to answer this question."
    )
    assert data["sources"] == []

    # Gemini must not be called when retrieval returns no results.
    mock_genai.assert_not_called()
def test_ask_returns_502_when_gemini_fails(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-api-key")

    monkeypatch.setattr(
        main,
        "embed_query_or_http_error",
        lambda _: FAKE_EMBEDDING,
    )
    monkeypatch.setattr(
        main,
        "retrieve_hybrid_candidates",
        lambda *_args, **_kwargs: [FAKE_ROW],
    )

    fake_client = MagicMock()
    fake_client.models.generate_content.side_effect = RuntimeError(
        "Internal Gemini failure details"
    )
    monkeypatch.setattr(main.genai, "Client", lambda **_: fake_client)

    response = client.post(
        "/ask",
        json={"question": "What is the secret code?", "limit": 5},
    )

    assert response.status_code == 502
    assert response.json()["detail"] == (
        "Gemini could not generate an answer. Please retry shortly."
    )
    assert "Internal Gemini failure details" not in response.text