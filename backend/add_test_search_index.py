import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv

backend_dir = Path(__file__).resolve().parent
load_dotenv(backend_dir / ".env.test", override=True)

required = ["PGHOST", "PGDATABASE", "PGUSER", "PGPASSWORD"]
missing = [key for key in required if not os.getenv(key)]

if missing:
    raise RuntimeError(
        f"Missing test database settings: {', '.join(missing)}"
    )

with psycopg.connect(
    host=os.environ["PGHOST"],
    dbname=os.environ["PGDATABASE"],
    user=os.environ["PGUSER"],
    password=os.environ["PGPASSWORD"],
    sslmode="require",
    connect_timeout=5,
) as connection:
    database = connection.execute(
        "SELECT current_database()"
    ).fetchone()[0]

    print(f"Connected to database: {database}")

    connection.execute("""
        CREATE INDEX IF NOT EXISTS idx_chunks_fts
        ON document_chunks
        USING GIN (to_tsvector('english', chunk_text))
    """)

    connection.execute("ANALYZE document_chunks")

    print("Full-text search index created or already exists.")
    print("Table statistics updated.")