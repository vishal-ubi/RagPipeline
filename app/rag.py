import json

import boto3

from app.config import BEDROCK_MODEL_ID, aws_region
from app.retrieval import search_database


def generate_answer(query):
    retrieved_chunks = search_database(query, top_k=3)
    print("Retrieved chunks for answer generation:")
    for result in retrieved_chunks:
        print(f"Chunk {result['chunk_index']}: {result['content']}\n")

    context = "\n\n".join(
        f"Chunk {result['chunk_index']}: {result['content']}"
        for result in retrieved_chunks
    )
    prompt = (
        "Answer the question using only the provided context. "
        "Do not invent information. If the answer is not contained in the "
        "context, say that the information is not available.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}"
    )

    bedrock = boto3.client(
        "bedrock-runtime",
        region_name=aws_region(),
    )
    response = bedrock.invoke_model(
        modelId=BEDROCK_MODEL_ID,
        body=json.dumps(
            {
                "schemaVersion": "messages-v1",
                "messages": [
                    {
                        "role": "user",
                        "content": [{"text": prompt}],
                    }
                ],
                "inferenceConfig": {
                    "max_new_tokens": 512,
                    "temperature": 0.2,
                },
            }
        ),
        contentType="application/json",
        accept="application/json",
    )
    response_body = json.loads(response["body"].read())
    answer = "".join(
        block["text"] for block in response_body["output"]["message"]["content"]
        if "text" in block
    ).strip()
    return {
        "answer": answer,
        "retrieved_chunks": [
            {
                "chunk_index": result["chunk_index"],
                "score": result["score"],
                "content": result["content"],
            }
            for result in retrieved_chunks
        ],
    }
