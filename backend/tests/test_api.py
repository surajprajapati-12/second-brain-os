
import pytest
from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


def test_health_returns_200():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_search_rejects_empty_query():
    response = client.post(
        "/search",
        json={"query": "", "limit": 5},
    )

    assert response.status_code == 422


def test_search_rejects_limit_above_maximum():
    response = client.post(
        "/search",
        json={
            "query": "What does phoenix-test.txt say?",
            "limit": 11,
        },
    )

    assert response.status_code == 422


def test_ask_rejects_empty_question():
    response = client.post(
        "/ask",
        json={"question": "", "limit": 5},
    )

    assert response.status_code == 422


def test_ask_rejects_limit_above_maximum():
    response = client.post(
        "/ask",
        json={
            "question": "What does phoenix-test.txt say?",
            "limit": 11,
        },
    )

    assert response.status_code == 422

from unittest.mock import MagicMock, patch


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
    connection = MagicMock()
    connection.__enter__.return_value.execute.return_value.fetchall.return_value = rows
    return connection


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
    assert "blue comet" in data["answer"]
    assert len(data["sources"]) >= 1
    assert data["sources"][0]["filename"] == "phoenix-test.txt"
from backend.main import filter_and_deduplicate_rows


def make_search_row(
    document_id,
    filename,
    chunk_index,
    text,
    distance,
    keyword_rank,
    hybrid_score=0.01,
):
    return (
        document_id,
        filename,
        chunk_index,
        text,
        distance,
        keyword_rank,
        hybrid_score,
    )


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