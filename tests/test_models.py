import pytest

from rag_pipeline.models import DocumentMetadata, ExtractedDocument, TextChunk


def test_document_metadata_creation():
    document = DocumentMetadata(
        document_id="doc-1",
        filename="guide.pdf",
        media_type="application/pdf",
        source_uri="file:///documents/guide.pdf",
    )

    assert document.document_id == "doc-1"
    assert document.filename == "guide.pdf"


def test_extracted_document_creation():
    source = DocumentMetadata(document_id="doc-1", filename="notes.txt")

    document = ExtractedDocument(source=source, text="Extracted source text.")

    assert document.source == source
    assert document.text == "Extracted source text."


def test_text_chunk_creation_with_optional_page():
    chunk = TextChunk(
        chunk_id="doc-1-0",
        document_id="doc-1",
        text="A section of source text.",
        chunk_index=0,
        page_number=2,
    )

    assert chunk.chunk_index == 0
    assert chunk.page_number == 2


@pytest.mark.parametrize(
    "factory",
    [
        lambda: DocumentMetadata(filename="notes.txt"),
        lambda: ExtractedDocument(
            source=DocumentMetadata(document_id="doc-1", filename="notes.txt")
        ),
        lambda: TextChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            text="content",
        ),
    ],
)
def test_models_require_mandatory_fields(factory):
    with pytest.raises(TypeError):
        factory()


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (
            lambda: DocumentMetadata(document_id="", filename="notes.txt"),
            "document_id",
        ),
        (
            lambda: DocumentMetadata(document_id="doc-1", filename=" "),
            "filename",
        ),
        (
            lambda: ExtractedDocument(
                source=DocumentMetadata(document_id="doc-1", filename="notes.txt"),
                text="  ",
            ),
            "text",
        ),
        (
            lambda: ExtractedDocument(source="not metadata", text="content"),
            "source",
        ),
        (
            lambda: TextChunk(
                chunk_id="chunk-1",
                document_id="doc-1",
                text="content",
                chunk_index=-1,
            ),
            "chunk_index",
        ),
        (
            lambda: TextChunk(
                chunk_id="chunk-1",
                document_id="doc-1",
                text="content",
                chunk_index=0,
                page_number=0,
            ),
            "page_number",
        ),
    ],
)
def test_models_reject_invalid_data(factory, message):
    with pytest.raises(ValueError, match=message):
        factory()
