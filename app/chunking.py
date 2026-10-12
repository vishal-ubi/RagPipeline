def chunk_text(text, chunk_size=100, overlap=20):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and less than chunk_size")

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            for index in range(start + overlap, end):
                if text[index] in ".!?":
                    end = index + 1

        chunks.append(text[start:end])
        if end == len(text):
            break
        next_start = end - overlap
        if (
            start < next_start < end
            and text[next_start - 1].isalnum()
            and text[next_start].isalnum()
        ):
            boundaries = []
            left = next_start
            while left > start + 1 and not text[left - 1].isspace():
                left -= 1
            if left > start and text[left - 1].isspace():
                boundaries.append(left)

            right = next_start
            while right < end and not text[right].isspace():
                right += 1
            while right < end and text[right].isspace():
                right += 1
            if next_start < right < end:
                boundaries.append(right)

            if boundaries:
                next_start = min(
                    boundaries, key=lambda boundary: abs(boundary - next_start)
                )
        start = next_start

    return chunks
