"""Provider-neutral data models for documents and extracted text."""

from dataclasses import dataclass


def _validate_non_empty_string(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class DocumentMetadata:
    document_id: str
    filename: str
    media_type: str | None = None
    source_uri: str | None = None

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.document_id, "document_id")
        _validate_non_empty_string(self.filename, "filename")
        if self.media_type is not None:
            _validate_non_empty_string(self.media_type, "media_type")
        if self.source_uri is not None:
            _validate_non_empty_string(self.source_uri, "source_uri")


@dataclass(frozen=True, slots=True)
class ExtractedPage:
    page_number: int
    text: str

    def __post_init__(self) -> None:
        if type(self.page_number) is not int or self.page_number < 1:
            raise ValueError("page_number must be a positive integer")
        if not isinstance(self.text, str):
            raise ValueError("text must be a string")


@dataclass(frozen=True, slots=True)
class ExtractedDocument:
    source: DocumentMetadata
    text: str
    pages: tuple[ExtractedPage, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.source, DocumentMetadata):
            raise ValueError("source must be a DocumentMetadata instance")
        _validate_non_empty_string(self.text, "text")
        if not isinstance(self.pages, tuple) or any(
            not isinstance(page, ExtractedPage) for page in self.pages
        ):
            raise ValueError("pages must be a tuple of ExtractedPage instances")


@dataclass(frozen=True, slots=True)
class TextChunk:
    chunk_id: str
    document_id: str
    text: str
    chunk_index: int
    page_number: int | None = None
    page_numbers: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.chunk_id, "chunk_id")
        _validate_non_empty_string(self.document_id, "document_id")
        _validate_non_empty_string(self.text, "text")
        if type(self.chunk_index) is not int or self.chunk_index < 0:
            raise ValueError("chunk_index must be a non-negative integer")
        if self.page_number is not None and (
            type(self.page_number) is not int or self.page_number < 1
        ):
            raise ValueError("page_number must be a positive integer when provided")
        if not isinstance(self.page_numbers, tuple) or any(
            type(page_number) is not int or page_number < 1
            for page_number in self.page_numbers
        ):
            raise ValueError("page_numbers must be a tuple of positive integers")
        if tuple(sorted(set(self.page_numbers))) != self.page_numbers:
            raise ValueError("page_numbers must be unique and in ascending order")
        if self.page_number is not None and self.page_numbers not in (
            (),
            (self.page_number,),
        ):
            raise ValueError("page_number must match page_numbers when both are set")
