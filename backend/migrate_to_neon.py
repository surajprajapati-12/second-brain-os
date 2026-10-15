
import os
from getpass import getpass
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

load_dotenv(Path(__file__).resolve().parent / ".env")


def main():
    required = ["PGHOST", "PGDATABASE", "PGUSER", "PGPASSWORD"]
    missing = [key for key in required if not os.getenv(key)]

    if missing:
        raise RuntimeError(
            f"Missing local PostgreSQL settings in backend/.env: {missing}"
        )

    # Connect to your existing local PostgreSQL database.
    with psycopg.connect(
        host=os.environ["PGHOST"],
        port=int(os.getenv("PGPORT", "5432")),
        dbname=os.environ["PGDATABASE"],
        user=os.environ["PGUSER"],
        password=os.environ["PGPASSWORD"],
        connect_timeout=10,
    ) as local:
        register_vector(local)

        # Enter the Neon connection string privately in the terminal.
        neon_url = getpass("Paste your Neon connection string (hidden): ").strip()
        if not neon_url:
            raise ValueError("Neon connection string cannot be empty.")

        # Connect to Neon and preserve its existing data.
        with psycopg.connect(
            neon_url,
            connect_timeout=15,
            sslmode="require",
        ) as neon:
            register_vector(neon)

            source_rows = local.execute(
                """
                SELECT document_id, filename, chunk_index,
                       chunk_text, embedding, created_at
                FROM document_chunks
                ORDER BY document_id, chunk_index
                """
            ).fetchall()

            print(f"Found {len(source_rows)} local chunks.")

            with neon.cursor() as cursor:
                for row in source_rows:
                    cursor.execute(
                        """
                        INSERT INTO document_chunks (
                            document_id, filename, chunk_index,
                            chunk_text, embedding, created_at
                        )
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (document_id, chunk_index)
                        DO UPDATE SET
                            filename = EXCLUDED.filename,
                            chunk_text = EXCLUDED.chunk_text,
                            embedding = EXCLUDED.embedding,
                            created_at = EXCLUDED.created_at
                        """,
                        row,
                    )

            # The Neon transaction commits when its connection context exits.

    # Verify the destination after the migration transaction has committed.
    with psycopg.connect(
        neon_url,
        connect_timeout=15,
        sslmode="require",
    ) as neon:
        destination_count = neon.execute(
            "SELECT COUNT(*) FROM document_chunks"
        ).fetchone()[0]

        migrated_count = neon.execute(
            "SELECT COUNT(*) FROM document_chunks WHERE (document_id, chunk_index) IN (SELECT document_id, chunk_index FROM document_chunks)"
        ).fetchone()[0]

        print(f"Neon now contains {destination_count} chunks.")
        print(f"Local chunks processed: {len(source_rows)}.")

        if destination_count < len(source_rows):
            raise RuntimeError(
                "Neon contains fewer chunks than were processed. "
                "Check the migration before switching databases."
            )

        print("Migration completed. Local PostgreSQL was not modified.")


if __name__ == "__main__":
    main()
