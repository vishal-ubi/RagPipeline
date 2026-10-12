# S3-Only RAG Project

This beginner-friendly project lets you choose one `.txt` or `.pdf` object
from an S3 bucket, extracts its text, creates chunks and embeddings, and
stores it in PostgreSQL with pgvector. You can then ask questions about the
active document through a FastAPI endpoint and AWS Bedrock.

Only S3 objects are used for ingestion. There is no local-file ingestion
command. Ingestion is explicit: starting FastAPI does not load or replace a
document.

## Project structure

```text
.
├── app/
│   ├── api.py          # POST /ask endpoint
│   ├── chunking.py     # Split text into overlapping chunks
│   ├── config.py       # Read settings from environment and .env
│   ├── database.py     # Schema checks, migration helper, and storage
│   ├── document.py     # List and read S3 TXT/PDF objects
│   ├── embeddings.py   # FastEmbed document and question embeddings
│   ├── ingestion.py    # Validate, chunk, embed, and store selected text
│   ├── rag.py          # Retrieve context and call Bedrock
│   └── retrieval.py    # pgvector search and retrieval evaluation
├── scripts/
│   └── migrate_document_chunks.py
├── tests/
│   └── test_database_migration_safety.py
├── ingest_s3_text.py  # S3 object selection and ingestion command
├── requirements.txt
├── .gitignore
└── .env               # Private settings; do not commit or share
```

## Setup

In PowerShell, from the project folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create or update the local `.env` file with your own settings:

```dotenv
DB_HOST=your-postgres-host
DB_PORT=5432
DB_NAME=your-database-name
DB_USER=your-database-user
DB_PASSWORD=your-database-password

AWS_REGION=your-aws-region
S3_BUCKET_NAME=your-s3-bucket
BEDROCK_MODEL_ID=your-enabled-bedrock-model-id
```

Keep real credentials only in your private `.env` or your AWS credential
provider. Do not paste them into source code, documentation, or chat. Boto3
uses its default AWS credential chain (for example, an AWS CLI profile).

The project uses FastAPI, FastEmbed with `BAAI/bge-small-en-v1.5`, Psycopg,
PostgreSQL with pgvector, PyMuPDF for PDFs, and AWS Bedrock Runtime.

## Database preparation

The PostgreSQL database must have the pgvector extension available. For an
existing legacy `document_chunks` table, run the standalone migration once
before ingestion:

```powershell
python scripts\migrate_document_chunks.py
```

The migration uses the configured `DB_*` settings and runs transactionally.
It preserves existing rows while adding/backfilling `source_document` and
changing the chunk uniqueness to `(source_document, chunk_index)`. If it
reports a connection error, resolve database host/network connectivity before
retrying. Do not run ingestion against the legacy schema; the application
checks the schema and will stop with a migration instruction.

The schema migration and document ingestion are separate operations. Running
the migration alone does not clear or replace stored chunks.

## Select and ingest one S3 object

Make sure the configured AWS identity can list objects (`s3:ListBucket`) and
read the selected object (`s3:GetObject`). Then run:

```powershell
python ingest_s3_text.py
```

The command lists `.txt` and `.pdf` keys from the configured bucket. Enter one
listed key exactly when prompted, for example:

```text
Available S3 documents:
  policies/refunds.txt
  manuals/employee-guide.pdf
Enter the exact S3 object key to ingest: policies/refunds.txt
```

The application reads the selected object, extracts its text, splits it into
chunks, creates embeddings, and only then opens the database transaction.
TXT files must be UTF-8. PDFs are parsed with PyMuPDF; scanned image-only PDFs
need OCR, which is not included, and will be rejected if no text can be
extracted.

### One active document and replacement behavior

Each successful ingestion replaces the rows in `document_chunks` with the
selected object's chunks. The source identifier is stored as
`s3://<bucket>/<key>`. This means old chunks and chunk indexes from a previous
document, including a document with a different key, do not remain after a
successful replacement.

Text extraction and embedding happen before storage. Empty, unreadable, or
unusable documents are rejected without changing the current database rows.
Deletion and insertion are performed in the same PostgreSQL transaction: if
storage fails, it rolls back and preserves the previous document. This
replacement affects only rows in `document_chunks`; it does not drop tables,
extensions, or indexes.

Because ingestion replaces the active document, do not select a document
unless you intend it to become the one used for retrieval.

## Ask questions through FastAPI

Start the API in a separate PowerShell window:

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn app.api:app --reload
```

Open <http://127.0.0.1:8000/docs>, choose `POST /ask`, and send:

```json
{
  "question": "What is the refund policy?"
}
```

The endpoint embeds the question, retrieves the three nearest chunks from
pgvector, sends their text and the question to the configured Bedrock model,
and returns the answer plus supporting chunks and similarity scores. Scores
indicate vector similarity; they are not confidence probabilities.

The API does not ingest documents at startup. Ingest an S3 object explicitly
before asking questions.

## Chunking, embeddings, and retrieval evaluation

Chunking uses a default size of 100 characters with 20 characters of overlap.
It prefers sentence-ending punctuation when possible and tries to avoid
starting a chunk in the middle of a word. These are simple starting settings,
not automatically optimized values.

FastEmbed uses `BAAI/bge-small-en-v1.5` for both document chunks and question
vectors. Each vector has 384 dimensions, matching the PostgreSQL `VECTOR(384)`
column. Database retrieval orders by cosine distance and returns up to three
chunks.

The project includes `evaluate_retrieval()`, which checks whether the
expected chunk for each of three hand-written sample questions appears in
the top three. This Recall@3 check is only a small retrieval sanity check. It
does not automatically score Bedrock answer correctness or establish quality
for arbitrary documents.

## Run offline mock tests

The migration and ingestion safety tests use mocks; they do not need a live
PostgreSQL database or S3 bucket:

```powershell
python -m unittest discover -s tests -p test_database_migration_safety.py -v
```
