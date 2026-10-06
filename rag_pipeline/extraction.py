"""Local text extraction for plain-text and PDF documents."""

from pathlib import Path
from uuid import uuid4

import pymupdf

from rag_pipeline.models import (
    DocumentMetadata,
    ExtractedDocument,
    ExtractedPage,
)


class DocumentExtractionError(Exception):
    """Base error for expected document extraction failures."""


class DocumentNotFoundError(DocumentExtractionError):
    """Raised when the requested document does not exist or is not a file."""


class UnsupportedDocumentTypeError(DocumentExtractionError):
    """Raised when a document has an unsupported file extension."""


class EmptyDocumentError(DocumentExtractionError):
    """Raised when a document contains no extractable text."""


class DocumentExtractor:
    """Extract text and source metadata from local TXT and PDF files."""

    def extract(self, file_path: str | Path) -> ExtractedDocument:
        path = Path(file_path)
        extension = path.suffix.lower()
        if extension not in {".txt", ".pdf"}:
            raise UnsupportedDocumentTypeError(
                f"Unsupported document type: {extension or '(no extension)'}"
            )

        try:
            resolved_path = path.resolve(strict=True)
        except FileNotFoundError as error:
            raise DocumentNotFoundError(f"Document not found: {path}") from error
        except OSError as error:
            raise DocumentExtractionError(
                f"Unable to access document: {path}"
            ) from error

        if not resolved_path.is_file():
            raise DocumentNotFoundError(f"Document is not a file: {path}")

        if extension == ".txt":
            text = self._extract_txt(resolved_path)
            pages: tuple[ExtractedPage, ...] = ()
            media_type = "text/plain"
        else:
            pages = self._extract_pdf(resolved_path)
            text = "\n\n".join(page.text for page in pages if page.text.strip())
            media_type = "application/pdf"

        if not text.strip():
            raise EmptyDocumentError(f"No extractable text in document: {path.name}")

        source = DocumentMetadata(
            document_id=uuid4().hex,
            filename=resolved_path.name,
            media_type=media_type,
            source_uri=resolved_path.as_uri(),
        )
        return ExtractedDocument(source=source, text=text, pages=pages)

    @staticmethod
    def _extract_txt(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError as error:
            raise DocumentExtractionError(
                f"Text document is not valid UTF-8: {path.name}"
            ) from error
        except OSError as error:
            raise DocumentExtractionError(
                f"Unable to read text document: {path.name}"
            ) from error

    @staticmethod
    def _extract_pdf(path: Path) -> tuple[ExtractedPage, ...]:
        try:
            with pymupdf.open(path) as pdf:
                return tuple(
                    ExtractedPage(page_number=index, text=page.get_text())
                    for index, page in enumerate(pdf, start=1)
                )
        except (pymupdf.FileDataError, pymupdf.EmptyFileError, RuntimeError) as error:
            raise DocumentExtractionError(
                f"Unable to extract text from PDF: {path.name}"
            ) from error
        except OSError as error:
            raise DocumentExtractionError(
                f"Unable to read PDF document: {path.name}"
            ) from error
