
import getpass
import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    password = getpass.getpass("PostgreSQL password: ")

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    question = input("\nAsk your Second Brain a question: ").strip()

    if not question:
        print("Please enter a question.")
        return

    query_embedding = model.encode(
        question,
        normalize_embeddings=True,
    )

    with psycopg.connect(
        host="localhost",
        port=5432,
        dbname="second_brain",
        user="postgres",
        password=password,
    ) as conn:
        register_vector(conn)

        results = conn.execute(
            """
            SELECT
                filename,
                chunk_text,
                embedding <=> %s AS distance
            FROM document_chunks
            ORDER BY embedding <=> %s
            LIMIT 5
            """,
            (query_embedding, query_embedding),
        ).fetchall()

    print("\nMost relevant document chunks:\n")

    for rank, (filename, chunk_text, distance) in enumerate(
        results, start=1
    ):
        print(f"{rank}. File: {filename}")
        print(f"   Cosine distance: {distance:.4f}")
        print(f"   Text: {chunk_text[:500]}")
        print()


if __name__ == "__main__":
    main()
