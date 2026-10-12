from app.chunking import chunk_text
from app.database import store_chunks
from app.embeddings import generate_embeddings


def ingest_text(text, source_document):
    if not text.strip():
        raise ValueError("The document contains no extractable text")
    if not source_document or not source_document.startswith("s3://"):
        raise ValueError("source_document must be an S3 URI")

    chunks = [chunk for chunk in chunk_text(text) if chunk.strip()]
    if not chunks:
        raise ValueError("The document produced no usable text chunks")

    embeddings = generate_embeddings(chunks)
    if len(embeddings) != len(chunks):
        raise ValueError("embedding count does not match chunk count")
    store_chunks(chunks, embeddings, source_document=source_document)
    return chunks, embeddings
