import pymupdf
import pytest

from rag_pipeline.extraction import (
    DocumentExtractionError,
    DocumentExtractor,
    DocumentNotFoundError,
    EmptyDocumentError,
    UnsupportedDocumentTypeError,
)
from rag_pipeline.models import ExtractedDocument


def _write_pdf(path, page_texts):
    document = pymupdf.open()
    for text in page_texts:
        page = document.new_page()
        page.insert_text((72, 72), text)
    document.save(path)
    document.close()


def test_extract_txt_returns_text_and_source_metadata(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("Text from the document.", encoding="utf-8")

    result = DocumentExtractor().extract(path)

    assert isinstance(result, ExtractedDocument)
    assert result.text == "Text from the document."
    assert result.source.filename == "notes.txt"
    assert result.source.media_type == "text/plain"
    assert result.source.source_uri == path.resolve().as_uri()
    assert result.pages == ()


def test_extract_pdf_preserves_page_and_source_metadata(tmp_path):
    path = tmp_path / "guide.pdf"
    _write_pdf(path, ["Text on the source page."])

    result = DocumentExtractor().extract(path)

    assert result.source.filename == "guide.pdf"
    assert result.source.media_type == "application/pdf"
    assert result.text == "Text on the source page.\n"
    assert len(result.pages) == 1
    assert result.pages[0].page_number == 1
    assert "Text on the source page." in result.pages[0].text


def test_extract_multiple_page_pdf_keeps_each_page_separate(tmp_path):
    path = tmp_path / "multi-page.pdf"
    _write_pdf(path, ["First page content.", "Second page content."])

    result = DocumentExtractor().extract(path)

    assert len(result.pages) == 2
    assert [page.page_number for page in result.pages] == [1, 2]
    assert "First page content." in result.pages[0].text
    assert "Second page content." in result.pages[1].text
    assert "First page content." in result.text
    assert "Second page content." in result.text


def test_empty_txt_raises_clear_application_error(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("", encoding="utf-8")

    with pytest.raises(EmptyDocumentError, match="No extractable text"):
        DocumentExtractor().extract(path)


def test_missing_file_raises_application_error(tmp_path):
    with pytest.raises(DocumentNotFoundError, match="Document not found"):
        DocumentExtractor().extract(tmp_path / "missing.txt")


def test_unsupported_extension_raises_application_error(tmp_path):
    path = tmp_path / "document.docx"
    path.write_text("Not a supported format.", encoding="utf-8")

    with pytest.raises(UnsupportedDocumentTypeError, match="Unsupported document"):
        DocumentExtractor().extract(path)


def test_invalid_pdf_raises_application_error(tmp_path):
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"This is not a valid PDF")

    with pytest.raises(DocumentExtractionError, match="Unable to extract text from PDF"):
        DocumentExtractor().extract(path)


def test_pdf_without_extractable_text_raises_empty_document_error(tmp_path):
    path = tmp_path / "blank.pdf"
    _write_pdf(path, [""])

    with pytest.raises(EmptyDocumentError, match="No extractable text"):
        DocumentExtractor().extract(path)
