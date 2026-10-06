# RAG Pipeline Learning Project

A learning project for document-based question answering. The current version
extracts text from local TXT and PDF files and splits it into overlapping,
source-aware chunks. Embeddings, vector storage, retrieval, and answer
generation are not implemented yet.

## Requirements

- Python 3.13

## Setup on Windows (PowerShell)

From the project directory, create and use a virtual environment:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Activation is optional. Running commands through `.venv\Scripts\python.exe`
avoids PowerShell execution-policy issues.

## Run

```powershell
.\.venv\Scripts\python.exe -m rag_pipeline
```

## Test

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## Document text extraction

Use `DocumentExtractor` to extract a local `.txt` or `.pdf` file:

```python
from rag_pipeline.extraction import DocumentExtractor

document = DocumentExtractor().extract("path/to/document.pdf")
print(document.source.filename)
print(document.text)

# PDF pages retain one-based source page numbers.
for page in document.pages:
    print(page.page_number, page.text)
```

Empty documents, missing files, unsupported formats, and unreadable/corrupted
documents raise specific `DocumentExtractionError` subclasses.

## Text chunking

`TextChunker` produces deterministic character-based chunks. Its default target
size is 1000 characters with 200 characters of overlap:

```python
from rag_pipeline.chunking import TextChunker

chunks = TextChunker(chunk_size=1000, chunk_overlap=200).split(document)
```

Chunk IDs and indexes follow source order. PDF chunks include `page_numbers`;
`page_number` is set only when a chunk belongs to exactly one page. The chunker
preserves source text and uses whitespace as a preferred split boundary.

## Configuration

`PipelineConfig.from_env()` reads configuration from process environment
variables. Defaults are `APP_ENV=development` and `LOG_LEVEL=INFO`. Optional
integration placeholders are `LLM_PROVIDER`, `LLM_MODEL`, `EMBEDDING_PROVIDER`,
`EMBEDDING_MODEL`, `VECTOR_STORE_PROVIDER`, and `VECTOR_STORE_LOCATION`.
Provider settings are unset by default and no credentials are required.

`.env.example` documents the variable names; it is not automatically loaded.
Set variables in the shell or your development environment. Never put real
credentials in `.env.example` or commit a local `.env` file.
