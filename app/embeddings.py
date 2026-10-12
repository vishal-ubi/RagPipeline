from fastembed import TextEmbedding

from app.config import MODEL_NAME


def generate_embeddings(chunks):
    if not chunks:
        return []

    model = TextEmbedding(model_name=MODEL_NAME)
    vectors = model.embed(chunks)
    return [vector.tolist() for vector in vectors]
