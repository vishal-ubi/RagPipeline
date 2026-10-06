"""Deterministic character-based chunking for extracted documents."""

from dataclasses import dataclass

from rag_pipeline.models import ExtractedDocument, TextChunk


@dataclass(frozen=True, slots=True)
class _PageSpan:
    start: int
    end: int
    page_number: int


class TextChunker:
    """Split extracted text into overlapping chunks without dropping characters."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        if type(chunk_size) is not int or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        if type(chunk_overlap) is not int or chunk_overlap < 0:
            raise ValueError("chunk_overlap must be a non-negative integer")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, document: ExtractedDocument) -> list[TextChunk]:
        """Return chunks in source order with deterministic ids and page metadata."""
        if not isinstance(document, ExtractedDocument):
            raise TypeError("document must be an ExtractedDocument instance")
        if not document.text.strip():
            raise ValueError("document text must contain non-whitespace characters")

        page_spans = self._page_spans(document)
        chunks: list[TextChunk] = []
        start = 0

        while start < len(document.text):
            end = self._next_end(document.text, start)
            chunk_text = document.text[start:end]
            page_numbers = self._pages_for_span(page_spans, start, end)
            chunks.append(
                TextChunk(
                    chunk_id=f"{document.source.document_id}-chunk-{len(chunks)}",
                    document_id=document.source.document_id,
                    text=chunk_text,
                    chunk_index=len(chunks),
                    page_number=page_numbers[0] if len(page_numbers) == 1 else None,
                    page_numbers=page_numbers,
                )
            )

            if end == len(document.text):
                break
            start = end - self.chunk_overlap

        return chunks

    def _next_end(self, text: str, start: int) -> int:
        target = min(start + self.chunk_size, len(text))
        if target == len(text):
            return target

        earliest_boundary = min(
            target,
            start + max(self.chunk_overlap + 1, self.chunk_size // 2),
        )
        boundary = next(
            (
                index + 1
                for index in range(target - 1, earliest_boundary - 1, -1)
                if text[index].isspace()
            ),
            target,
        )

        if text[start:boundary].strip():
            return boundary

        # Keep long whitespace runs with neighboring text rather than emitting
        # a whitespace-only chunk that the application model rejects.
        next_text = next(
            (index + 1 for index in range(boundary, len(text)) if not text[index].isspace()),
            len(text),
        )
        return min(next_text, len(text))

    @staticmethod
    def _page_spans(document: ExtractedDocument) -> tuple[_PageSpan, ...]:
        text_pages = [page for page in document.pages if page.text.strip()]
        if not text_pages:
            return ()

        expected_text = "\n\n".join(page.text for page in text_pages)
        if expected_text != document.text:
            return ()

        spans: list[_PageSpan] = []
        offset = 0
        for page in text_pages:
            spans.append(
                _PageSpan(
                    start=offset,
                    end=offset + len(page.text),
                    page_number=page.page_number,
                )
            )
            offset += len(page.text) + 2
        return tuple(spans)

    @staticmethod
    def _pages_for_span(
        page_spans: tuple[_PageSpan, ...], start: int, end: int
    ) -> tuple[int, ...]:
        return tuple(
            span.page_number
            for span in page_spans
            if start < span.end and end > span.start
        )
