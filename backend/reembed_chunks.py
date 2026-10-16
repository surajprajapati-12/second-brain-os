
from pgvector import Vector

from backend.main import (
    EMBEDDING_DIMENSIONS,
    generate_embeddings,
    get_postgres_connection,
)


def main():
    # Read existing chunks without modifying the database.
    with get_postgres_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, chunk_text
                FROM public.document_chunks
                ORDER BY id
                """
            )
            rows = cursor.fetchall()

    if not rows:
        print("No chunks found. Nothing to re-embed.")
        return

    print(f"Found {len(rows)} chunks.")
    print("Generating Gemini embeddings...")

    # Generate every vector before making any database changes.
    updates = []
    for chunk_id, chunk_text in rows:
        vector = generate_embeddings([chunk_text])[0]

        if len(vector) != EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"Chunk {chunk_id}: expected {EMBEDDING_DIMENSIONS} "
                f"dimensions, got {len(vector)}."
            )

        updates.append((Vector(vector), chunk_id))
        print(f"Prepared chunk {chunk_id}")

    # Update existing rows in one transaction.
    with get_postgres_connection() as connection:
        with connection.cursor() as cursor:
            cursor.executemany(
                """
                UPDATE public.document_chunks
                SET embedding = %s
                WHERE id = %s
                """,
                updates,
            )

            cursor.execute(
                """
                SELECT COUNT(*)
                FROM public.document_chunks
                WHERE embedding IS NOT NULL
                  AND vector_dims(embedding) = %s
                """,
                (EMBEDDING_DIMENSIONS,),
            )
            valid_count = cursor.fetchone()[0]

            if valid_count != len(rows):
                raise RuntimeError(
                    f"Verification failed: expected {len(rows)} "
                    f"valid embeddings, found {valid_count}."
                )

    print(f"Success: updated {len(updates)} chunks with Gemini embeddings.")
    print(f"Verified {valid_count} embeddings with {EMBEDDING_DIMENSIONS} dimensions.")


if __name__ == "__main__":
    main()
