import pymupdf
import pytest

from ingest import chunk_text, extract_pdf


def make_pdf(pages: list[str]) -> bytes:
    document = pymupdf.open()
    for text in pages:
        page = document.new_page()
        if text:
            page.insert_text((72, 72), text)
    content = document.tobytes()
    document.close()
    return content


def test_extract_pdf_preserves_page_metadata_and_reports_empty_pages():
    result = extract_pdf(
        make_pdf(["First page content.", "", "Third page content."]),
        "../example.pdf",
        chunk_size=100,
        overlap=10,
    )

    assert result.filename == "example.pdf"
    assert result.total_pages == 3
    assert result.empty_pages == (2,)
    assert [chunk.metadata["page"] for chunk in result.chunks] == [1, 3]
    assert all(chunk.metadata["file_hash"] == result.file_hash for chunk in result.chunks)


def test_chunk_text_creates_overlap_and_respects_approximate_size():
    text = " ".join(f"word-{index}" for index in range(100))
    chunks = chunk_text(text, chunk_size=120, overlap=20)

    assert len(chunks) > 1
    assert all(1 <= len(chunk) <= 120 for chunk in chunks)
    assert set(chunks[0].split()).intersection(chunks[1].split())


def test_duplicate_content_produces_deterministic_ids():
    content = make_pdf(["Stable content"])

    first = extract_pdf(content, "first-name.pdf")
    second = extract_pdf(content, "second-name.pdf")

    assert first.file_hash == second.file_hash
    assert [chunk.id for chunk in first.chunks] == [
        chunk.id for chunk in second.chunks
    ]


def test_image_only_pdf_is_rejected_with_ocr_message():
    with pytest.raises(ValueError, match="requires OCR"):
        extract_pdf(make_pdf([""]), "scan.pdf")
