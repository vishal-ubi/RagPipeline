import sys

from app.config import S3_BUCKET_NAME
from app.document import list_s3_documents, read_s3_document
from app.ingestion import ingest_text


def main():
    if not S3_BUCKET_NAME:
        print("S3_BUCKET_NAME is not configured. Set it in the project .env file.")
        return 1

    try:
        keys = list_s3_documents(S3_BUCKET_NAME)
    except (PermissionError, RuntimeError) as exc:
        print(f"Could not list S3 objects: {exc}")
        return 1
    if not keys:
        print("No .txt or .pdf objects were found in the configured bucket.")
        return 1

    print("Available S3 documents:")
    for key in keys:
        print(f"  {key}")

    key = input("Enter the exact S3 object key to ingest: ").strip()
    if key not in keys:
        print("That key is not in the listed .txt/.pdf objects.")
        return 1

    source_document = f"s3://{S3_BUCKET_NAME}/{key}"
    try:
        text = read_s3_document(S3_BUCKET_NAME, key)
    except (FileNotFoundError, PermissionError, RuntimeError, ValueError) as exc:
        print(f"S3 document could not be read: {exc}")
        return 1

    try:
        chunks, embeddings = ingest_text(text, source_document=source_document)
    except ValueError as exc:
        print(f"S3 document was not ingested: {exc}")
        return 1

    print(f"Read {source_document}")
    print(f"Created {len(chunks)} text chunks.")
    print(f"Generated {len(embeddings)} embeddings.")
    print(f"Replaced the active document with {source_document}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
