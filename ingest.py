"""PDF extraction and deterministic, page-aware text chunking."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pymupdf

from config import CHUNK_OVERLAP, CHUNK_SIZE


@dataclass(frozen=True)
class DocumentChunk:
    id: str
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ExtractionResult:
    filename: str
    file_hash: str
    total_pages: int
    empty_pages: tuple[int, ...]
    chunks: tuple[DocumentChunk, ...]


def hash_file(content: bytes) -> str:
    """Return a stable content hash used for duplicate detection."""
    return hashlib.sha256(content).hexdigest()


def normalize_text(text: str) -> str:
    """Collapse PDF layout whitespace while preserving readable paragraphs."""
    paragraphs = []
    for paragraph in re.split(r"\n\s*\n", text):
        cleaned = re.sub(r"\s+", " ", paragraph).strip()
        if cleaned:
            paragraphs.append(cleaned)
    return "\n\n".join(paragraphs)


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping chunks, preferring whitespace boundaries."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        proposed_end = min(start + chunk_size, len(text))
        end = proposed_end

        if proposed_end < len(text):
            minimum_break = start + int(chunk_size * 0.6)
            whitespace = text.rfind(" ", minimum_break, proposed_end)
            if whitespace > start:
                end = whitespace

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break
        start = max(end - overlap, start + 1)

    return chunks


def extract_pdf(
    content: bytes,
    filename: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> ExtractionResult:
    """Extract text chunks and source metadata from an in-memory PDF."""
    if not content:
        raise ValueError("The uploaded PDF is empty")

    safe_filename = Path(filename).name or "document.pdf"
    file_hash = hash_file(content)

    try:
        document = pymupdf.open(stream=content, filetype="pdf")
    except Exception as exc:
        raise ValueError(f"{safe_filename} is not a readable PDF") from exc

    try:
        if document.needs_pass:
            raise ValueError(f"{safe_filename} is password-protected")

        total_pages = document.page_count
        empty_pages: list[int] = []
        chunks: list[DocumentChunk] = []

        for page_index in range(total_pages):
            page_number = page_index + 1
            text = normalize_text(document.load_page(page_index).get_text("text"))
            if not text:
                empty_pages.append(page_number)
                continue

            for chunk_index, chunk in enumerate(
                chunk_text(text, chunk_size=chunk_size, overlap=overlap)
            ):
                chunk_id = hashlib.sha256(
                    f"{file_hash}:{page_number}:{chunk_index}".encode("utf-8")
                ).hexdigest()
                chunks.append(
                    DocumentChunk(
                        id=chunk_id,
                        text=chunk,
                        metadata={
                            "source": safe_filename,
                            "page": page_number,
                            "chunk": chunk_index,
                            "file_hash": file_hash,
                            "total_pages": total_pages,
                        },
                    )
                )
    finally:
        document.close()

    if not chunks:
        raise ValueError(
            f"{safe_filename} contains no selectable text. "
            "It may be a scanned PDF that requires OCR."
        )

    return ExtractionResult(
        filename=safe_filename,
        file_hash=file_hash,
        total_pages=total_pages,
        empty_pages=tuple(empty_pages),
        chunks=tuple(chunks),
    )
