from math import sqrt

from pgvector import Vector
from pgvector.psycopg import register_vector

from app.chunking import chunk_text
from app.config import EMBEDDING_DIMENSIONS, EVALUATION_QUESTIONS
from app.database import _connect_to_database
from app.embeddings import generate_embeddings


def compare_chunk_sizes(text):
    configurations = [
        (100, 20),
        (150, 20),
    ]

    for chunk_size, overlap in configurations:
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        print(
            f"Chunk size: {chunk_size}, overlap: {overlap}, "
            f"chunk count: {len(chunks)}"
        )

        for index, chunk in enumerate(chunks):
            print(f"Chunk {index}:")
            print(chunk)
            print()


def cosine_similarity(vector_a, vector_b):
    if len(vector_a) != len(vector_b):
        raise ValueError("vectors must have the same number of values")

    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    magnitude_a = sqrt(sum(value * value for value in vector_a))
    magnitude_b = sqrt(sum(value * value for value in vector_b))

    if magnitude_a == 0 or magnitude_b == 0:
        raise ValueError("cosine similarity is undefined for a zero vector")

    return dot_product / (magnitude_a * magnitude_b)


def search(query, chunks, embeddings, top_k=3):
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must have the same length")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    query_embedding = generate_embeddings([query])[0]
    results = []

    for index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        score = cosine_similarity(query_embedding, embedding)
        results.append(
            {
                "chunk_index": index,
                "text": chunk,
                "score": score,
            }
        )

    results.sort(key=lambda result: result["score"], reverse=True)
    return results[:top_k]


def search_database(query, top_k=3):
    if type(top_k) is not int or top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    query_embedding = generate_embeddings([query])[0]
    if len(query_embedding) != EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"query embedding must have {EMBEDDING_DIMENSIONS} dimensions"
        )
    vector = Vector(query_embedding)

    with _connect_to_database() as connection:
        register_vector(connection)
        rows = connection.execute(
            """
            SELECT
                source_document,
                chunk_index,
                content,
                -- Convert pgvector cosine distance into cosine similarity.
                1 - (embedding <=> %s) AS similarity
            FROM document_chunks
            -- Smaller cosine distance means the vectors are more similar.
            ORDER BY embedding <=> %s ASC
            -- LIMIT returns only the top_k closest rows.
            LIMIT %s
            """,
            (vector, vector, top_k),
        ).fetchall()

    return [
        {
            "source_document": row[0],
            "chunk_index": row[1],
            "content": row[2],
            "score": row[3],
        }
        for row in rows
    ]


def evaluate_retrieval():
    found_count = 0

    for item in EVALUATION_QUESTIONS:
        question = item["question"]
        expected_chunk_index = item["expected_chunk_index"]
        results = search_database(question, top_k=3)
        retrieved_chunk_indexes = [
            result["chunk_index"] for result in results
        ]
        found = expected_chunk_index in retrieved_chunk_indexes

        print(f"Question: {question}")
        print(f"Expected chunk: {expected_chunk_index}")
        print(f"Retrieved chunks: {retrieved_chunk_indexes}")
        print(f"Expected chunk found: {found}\n")

        if found:
            found_count += 1

    recall_at_3 = found_count / len(EVALUATION_QUESTIONS)
    print(f"Recall@3: {recall_at_3:.2%} ({found_count}/{len(EVALUATION_QUESTIONS)})")
    return recall_at_3
