import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env.test", override=True)

with psycopg.connect(
    host=os.environ["PGHOST"],
    dbname=os.environ["PGDATABASE"],
    user=os.environ["PGUSER"],
    password=os.environ["PGPASSWORD"],
    sslmode="require",
    connect_timeout=5,
) as connection:
    print("Full-text index:")
    indexes = connection.execute("""
        SELECT indexname, indexdef
        FROM pg_indexes
        WHERE tablename = 'document_chunks'
          AND indexdef ILIKE '%USING gin%'
    """).fetchall()

    for name, definition in indexes:
        print(name)
        print(definition)

    print("\nKeyword query plan:")
    plan = connection.execute("""
        EXPLAIN
        SELECT id
        FROM document_chunks
        WHERE to_tsvector('english', chunk_text)
              @@ plainto_tsquery('english', %s)
    """, ("Atlas database recovery",)).fetchall()

    for row in plan:
        print(row[0])