import pytest

from rag_pipeline.chunking import TextChunker
from rag_pipeline.models import (
    DocumentMetadata,
    ExtractedDocument,
    ExtractedPage,
)


def _document(text, document_id="doc-1", pages=()):
    return ExtractedDocument(
        source=DocumentMetadata(
            document_id=document_id,
            filename="source.txt",
            media_type="text/plain",
            source_uri="file:///source.txt",
        ),
        text=text,
        pages=pages,
    )


def _reconstruct(chunks, overlap):
    return chunks[0].text + "".join(chunk.text[overlap:] for chunk in chunks[1:])


def test_short_document_produces_one_chunk_with_source_metadata():
    document = _document("Short document.")

    chunks = TextChunker(chunk_size=100, chunk_overlap=10).split(document)

    assert len(chunks) == 1
    assert chunks[0].text == document.text
    assert chunks[0].document_id == "doc-1"
    assert chunks[0].chunk_id == "doc-1-chunk-0"
    assert chunks[0].chunk_index == 0
    assert chunks[0].page_number is None
    assert chunks[0].page_numbers == ()


def test_exact_chunk_size_boundary_produces_one_chunk():
    document = _document("12345678")

    chunks = TextChunker(chunk_size=8, chunk_overlap=2).split(document)

    assert [chunk.text for chunk in chunks] == ["12345678"]


def test_multiple_chunks_use_configured_character_overlap():
    document = _document("abcdefghijklmnop")

    chunks = TextChunker(chunk_size=8, chunk_overlap=2).split(document)

    assert [chunk.text for chunk in chunks] == [
        "abcdefgh",
        "ghijklmn",
        "mnop",
    ]
    assert all(len(chunk.text) <= 8 for chunk in chunks)
    assert chunks[0].text[-2:] == chunks[1].text[:2]
    assert chunks[1].text[-2:] == chunks[2].text[:2]


def test_custom_chunk_size_and_zero_overlap():
    document = _document("abcdefghij")

    chunks = TextChunker(chunk_size=4, chunk_overlap=0).split(document)

    assert [chunk.text for chunk in chunks] == ["abcd", "efgh", "ij"]
    assert _reconstruct(chunks, overlap=0) == document.text


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap", "message"),
    [
        (0, 0, "chunk_size"),
        (-1, 0, "chunk_size"),
        (4, -1, "chunk_overlap"),
        (4, 4, "less than"),
        (4, 5, "less than"),
        (True, 0, "chunk_size"),
        (4, False, "chunk_overlap"),
    ],
)
def test_invalid_chunk_configuration_is_rejected(
    chunk_size, chunk_overlap, message
):
    with pytest.raises(ValueError, match=message):
        TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def test_whitespace_boundaries_are_preferred_without_text_loss():
    document = _document("alpha beta gamma delta")

    chunks = TextChunker(chunk_size=12, chunk_overlap=2).split(document)

    assert chunks[0].text.endswith(" ")
    assert _reconstruct(chunks, overlap=2) == document.text


def test_chunking_preserves_text_with_whitespace_and_overlap():
    text = "  First line.\n\nSecond\tline with spaces.  "
    document = _document(text)

    chunks = TextChunker(chunk_size=14, chunk_overlap=3).split(document)

    assert len(chunks) > 1
    assert _reconstruct(chunks, overlap=3) == text


def test_chunk_indexes_and_ids_are_stable():
    document = _document("A repeated deterministic input.", document_id="stable-doc")
    chunker = TextChunker(chunk_size=10, chunk_overlap=2)

    first_run = chunker.split(document)
    second_run = chunker.split(document)

    assert [chunk.chunk_index for chunk in first_run] == list(range(len(first_run)))
    assert [chunk.chunk_id for chunk in first_run] == [
        f"stable-doc-chunk-{index}" for index in range(len(first_run))
    ]
    assert first_run == second_run


def test_chunking_preserves_all_source_text_after_removing_overlap():
    document = _document(
        "A deterministic sentence that must be fully retained in every run."
    )

    chunks = TextChunker(chunk_size=17, chunk_overlap=5).split(document)

    assert _reconstruct(chunks, overlap=5) == document.text


def test_pdf_page_metadata_is_assigned_to_single_page_chunks():
    page = ExtractedPage(page_number=3, text="Text from page three.")
    document = _document(page.text, pages=(page,))

    chunks = TextChunker(chunk_size=100, chunk_overlap=10).split(document)

    assert chunks[0].page_number == 3
    assert chunks[0].page_numbers == (3,)


def test_chunk_spanning_pdf_pages_lists_all_pages_without_single_page_claim():
    first_page = ExtractedPage(page_number=1, text="abcdefgh")
    second_page = ExtractedPage(page_number=2, text="ijklmnop")
    document = _document(
        "\n\n".join((first_page.text, second_page.text)),
        pages=(first_page, second_page),
    )

    chunks = TextChunker(chunk_size=10, chunk_overlap=5).split(document)

    spanning_chunk = next(chunk for chunk in chunks if chunk.page_numbers == (1, 2))
    assert spanning_chunk.page_number is None


def test_invalid_document_input_is_rejected():
    with pytest.raises(TypeError, match="ExtractedDocument"):
        TextChunker().split("not an extracted document")
