
import getpass
import sqlite3
from pathlib import Path

import psycopg
import pymupdf
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
SQLITE_PATH = BASE_DIR / "documents.db"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def split_text(text: str) -> list[str]:
    """Split text into overlapping character-based chunks."""
    text = text.strip()

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - CHUNK_OVERLAP

    return chunks


def main():
    if not SQLITE_PATH.exists():
        raise FileNotFoundError(
            f"SQLite database not found: {SQLITE_PATH}"
        )

    with sqlite3.connect(SQLITE_PATH) as sqlite_conn:
        sqlite_conn.row_factory = sqlite3.Row
        documents = sqlite_conn.execute("""
            SELECT id, filename, extracted_text
            FROM documents
            ORDER BY created_at
        """).fetchall()

    print(f"Found {len(documents)} documents in SQLite.")

    if not documents:
        print("Nothing to process.")
        return

    password = getpass.getpass("PostgreSQL password: ")

    model = SentenceTransformer(MODEL_NAME, device="cpu")
    print("Embedding model loaded.")

    prepared_chunks = []

    for document in documents:
        chunks = split_text(document["extracted_text"])

        print(f"{document['filename']}: {len(chunks)} chunk(s)")

        for chunk_index, chunk_text in enumerate(chunks):
            prepared_chunks.append({
                "document_id": document["id"],
                "filename": document["filename"],
                "chunk_index": chunk_index,
                "chunk_text": chunk_text,
            })

    if not prepared_chunks:
        print("No non-empty text chunks found.")
        return

    texts = [item["chunk_text"] for item in prepared_chunks]

    embeddings = model.encode(
        texts,
        batch_size=16,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    for item, embedding in zip(prepared_chunks, embeddings):
        item["embedding"] = embedding

    with psycopg.connect(
        host="localhost",
        port=5432,
        dbname="second_brain",
        user="postgres",
        password=password,
    ) as conn:
        register_vector(conn)

        with conn.cursor() as cursor:
            for item in prepared_chunks:
                cursor.execute(
                    """
                    INSERT INTO document_chunks (
                        document_id,
                        filename,
                        chunk_index,
                        chunk_text,
                        embedding
                    )
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (document_id, chunk_index)
                    DO UPDATE SET
                        filename = EXCLUDED.filename,
                        chunk_text = EXCLUDED.chunk_text,
                        embedding = EXCLUDED.embedding,
                        created_at = NOW()
                    """,
                    (
                        item["document_id"],
                        item["filename"],
                        item["chunk_index"],
                        item["chunk_text"],
                        item["embedding"],
                    ),
                )

        count = conn.execute(
            "SELECT COUNT(*) FROM document_chunks"
        ).fetchone()[0]

    print(f"Processed chunks: {len(prepared_chunks)}")
    print(f"Total chunks in PostgreSQL: {count}")
    print("Backfill completed successfully.")


if __name__ == "__main__":
    main()
