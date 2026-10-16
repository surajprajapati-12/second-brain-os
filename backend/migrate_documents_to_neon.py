
import sqlite3

from main import DB_PATH, get_postgres_connection


def main():
    # Read existing SQLite records without deleting or modifying them.
    with sqlite3.connect(DB_PATH) as sqlite_connection:
        sqlite_connection.row_factory = sqlite3.Row
        documents = sqlite_connection.execute("""
            SELECT id, filename, file_type, size_bytes,
                   extracted_text, created_at
            FROM documents
            ORDER BY created_at
        """).fetchall()

    print(f"SQLite documents found: {len(documents)}")

    inserted = 0
    already_present = 0

    with get_postgres_connection() as connection:
        for document in documents:
            document_id = document["id"]

            existing = connection.execute("""
                SELECT filename, file_type, size_bytes, extracted_text
                FROM public.documents
                WHERE id = %s
            """, (document_id,)).fetchone()

            expected = (
                document["filename"],
                document["file_type"],
                document["size_bytes"],
                document["extracted_text"],
            )

            if existing is not None:
                if tuple(existing) != expected:
                    raise RuntimeError(
                        f"ID conflict for {document_id}; "
                        "existing Neon record differs from SQLite."
                    )
                already_present += 1
                continue

            chunk_count = connection.execute("""
                SELECT COUNT(*)
                FROM public.document_chunks
                WHERE document_id = %s
            """, (document_id,)).fetchone()[0]

            status = "indexed" if chunk_count > 0 else "pending"

            connection.execute("""
                INSERT INTO public.documents (
                    id, filename, file_type, size_bytes,
                    extracted_text, created_at, indexing_status
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                document_id,
                document["filename"],
                document["file_type"],
                document["size_bytes"],
                document["extracted_text"],
                document["created_at"],
                status,
            ))

            inserted += 1

        # Verify every SQLite ID now exists in Neon.
        for document in documents:
            found = connection.execute("""
                SELECT 1
                FROM public.documents
                WHERE id = %s
            """, (document["id"],)).fetchone()

            if found is None:
                raise RuntimeError(
                    f"Migration verification failed: {document['id']}"
                )

        total = connection.execute("""
            SELECT COUNT(*) FROM public.documents
        """).fetchone()[0]

        print(f"Inserted: {inserted}")
        print(f"Already present and matching: {already_present}")
        print(f"Total Neon documents: {total}")
        print("SUCCESS: All SQLite document IDs verified.")


if __name__ == "__main__":
    main()
