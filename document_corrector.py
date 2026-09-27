"""Extract, conservatively correct, and export OCR text with local Ollama."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Iterator

import ollama
import pymupdf
from docx import Document
from docx.document import Document as DocxDocument
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph

from config import OLLAMA_HOST


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".md", ".markdown", ".txt"}
MAX_GROUP_CHARS = 6_000

CORRECTION_SYSTEM_PROMPT = """You are a conservative OCR proofreader.
Correct only clear OCR, spelling, punctuation, spacing, merged-word, and
line-break errors in the supplied text blocks.

Mandatory rules:
- Preserve the original language, meaning, order, names, dates, and numbers.
- Do not summarize, translate, explain, expand, censor, or invent anything.
- Preserve Markdown symbols when present.
- Return every block exactly once with the same numeric id.
- Return valid JSON only, with this shape:
  {"corrections": [{"id": 0, "text": "corrected text"}]}"""


@dataclass(frozen=True)
class TextBlock:
    text: str
    style: str = "Normal"
    page_break_before: bool = False


@dataclass(frozen=True)
class ExtractedTextDocument:
    filename: str
    blocks: tuple[TextBlock, ...]

    @property
    def text(self) -> str:
        return "\n\n".join(block.text for block in self.blocks if block.text)


@dataclass(frozen=True)
class CorrectionResult:
    original: ExtractedTextDocument
    corrected_blocks: tuple[TextBlock, ...]
    warnings: tuple[str, ...]

    @property
    def corrected_text(self) -> str:
        return "\n\n".join(
            block.text for block in self.corrected_blocks if block.text
        )


@dataclass(frozen=True)
class _WorkItem:
    id: int
    block_index: int
    part_index: int
    text: str


def _decode_text(content: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-16", "cp1252"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("The text file encoding could not be detected")


def _blocks_from_plain_text(text: str, markdown: bool = False) -> list[TextBlock]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks: list[TextBlock] = []

    for raw_paragraph in re.split(r"\n\s*\n", text):
        paragraph = raw_paragraph.strip()
        if not paragraph:
            continue

        style = "Normal"
        if markdown:
            heading = re.match(r"^(#{1,6})\s+(.+)$", paragraph, flags=re.DOTALL)
            if heading:
                level = min(len(heading.group(1)), 3)
                style = f"Heading {level}"
                paragraph = heading.group(2).strip()

        blocks.append(TextBlock(text=paragraph, style=style))

    return blocks


def _iter_docx_blocks(document: DocxDocument) -> Iterator[Paragraph | Table]:
    for child in document.element.body.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, document)
        elif isinstance(child, CT_Tbl):
            yield Table(child, document)


def _extract_docx(content: bytes) -> list[TextBlock]:
    try:
        document = Document(BytesIO(content))
    except Exception as exc:
        raise ValueError("The DOCX file could not be read") from exc

    blocks: list[TextBlock] = []
    for item in _iter_docx_blocks(document):
        if isinstance(item, Paragraph):
            text = item.text.strip()
            if not text:
                continue
            style = item.style.name if item.style is not None else "Normal"
            page_break = bool(item.paragraph_format.page_break_before)
            blocks.append(
                TextBlock(
                    text=text,
                    style=style,
                    page_break_before=page_break,
                )
            )
        else:
            for row in item.rows:
                cells = [re.sub(r"\s+", " ", cell.text).strip() for cell in row.cells]
                row_text = " | ".join(cell for cell in cells if cell)
                if row_text:
                    blocks.append(TextBlock(text=row_text))
    return blocks


def _extract_pdf(content: bytes, filename: str) -> list[TextBlock]:
    try:
        document = pymupdf.open(stream=content, filetype="pdf")
    except Exception as exc:
        raise ValueError(f"{filename} is not a readable PDF") from exc

    try:
        if document.needs_pass:
            raise ValueError(f"{filename} is password-protected")

        blocks: list[TextBlock] = []
        for page_index in range(document.page_count):
            page_blocks = document.load_page(page_index).get_text("blocks", sort=True)
            first_on_page = True
            for raw_block in page_blocks:
                text = str(raw_block[4]).replace("\r", "\n")
                lines = [
                    re.sub(r"[ \t]+", " ", line).strip()
                    for line in text.splitlines()
                    if line.strip()
                ]
                cleaned = " ".join(lines).strip()
                if not cleaned:
                    continue
                blocks.append(
                    TextBlock(
                        text=cleaned,
                        page_break_before=page_index > 0 and first_on_page,
                    )
                )
                first_on_page = False
    finally:
        document.close()

    return blocks


def extract_editable_text(content: bytes, filename: str) -> ExtractedTextDocument:
    """Extract a common paragraph structure from a supported document."""
    if not content:
        raise ValueError("The uploaded file is empty")

    safe_filename = Path(filename).name
    extension = Path(safe_filename).suffix.casefold()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            "Unsupported format. Use PDF, DOCX, Markdown, or TXT."
        )

    if extension == ".pdf":
        blocks = _extract_pdf(content, safe_filename)
    elif extension == ".docx":
        blocks = _extract_docx(content)
    else:
        blocks = _blocks_from_plain_text(
            _decode_text(content),
            markdown=extension in {".md", ".markdown"},
        )

    if not blocks or not any(block.text.strip() for block in blocks):
        if extension == ".pdf":
            raise ValueError(
                f"{safe_filename} contains no selectable OCR text. "
                "Run OCR on the scanned PDF before correcting it."
            )
        raise ValueError(f"{safe_filename} contains no text to correct")

    return ExtractedTextDocument(filename=safe_filename, blocks=tuple(blocks))


def _split_long_text(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]

    parts: list[str] = []
    remaining = text
    while len(remaining) > limit:
        minimum_break = int(limit * 0.6)
        candidates = [
            remaining.rfind("\n", minimum_break, limit),
            remaining.rfind(". ", minimum_break, limit),
            remaining.rfind(" ", minimum_break, limit),
        ]
        end = max(candidates)
        if end < minimum_break:
            end = limit
        elif remaining[end : end + 2] == ". ":
            end += 1

        parts.append(remaining[:end].strip())
        remaining = remaining[end:].strip()

    if remaining:
        parts.append(remaining)
    return parts


def _response_content(response: Any) -> str:
    message = response.message if hasattr(response, "message") else response["message"]
    content = message.content if hasattr(message, "content") else message["content"]
    return str(content).strip()


def _parse_corrections(content: str) -> dict[int, str]:
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        start, end = content.find("{"), content.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("The model did not return valid JSON")
        data = json.loads(content[start : end + 1])

    values = data.get("corrections")
    if not isinstance(values, list):
        raise ValueError("The model response has no corrections list")

    corrections: dict[int, str] = {}
    for value in values:
        if not isinstance(value, dict) or "id" not in value or "text" not in value:
            continue
        corrections[int(value["id"])] = str(value["text"]).strip()
    return corrections


def _is_safe_correction(original: str, corrected: str) -> tuple[bool, str]:
    if not corrected:
        return False, "empty response"

    ratio = len(corrected) / max(len(original), 1)
    if ratio < 0.5 or ratio > 1.8:
        return False, "unexpected length change"

    number_pattern = r"(?<!\w)\d+(?:[.,]\d+)?(?!\w)"
    if re.findall(number_pattern, original) != re.findall(number_pattern, corrected):
        return False, "numbers changed"

    return True, ""


class DocumentCorrector:
    def __init__(
        self,
        model: str,
        ollama_host: str = OLLAMA_HOST,
        ollama_client: Any | None = None,
        group_char_limit: int = MAX_GROUP_CHARS,
    ) -> None:
        if group_char_limit < 100:
            raise ValueError("group_char_limit must be at least 100 characters")
        self.model = model
        self.ollama = ollama_client or ollama.Client(host=ollama_host)
        self.group_char_limit = group_char_limit

    def _work_items(self, document: ExtractedTextDocument) -> list[_WorkItem]:
        items: list[_WorkItem] = []
        item_id = 0
        for block_index, block in enumerate(document.blocks):
            for part_index, part in enumerate(
                _split_long_text(block.text, self.group_char_limit)
            ):
                items.append(
                    _WorkItem(
                        id=item_id,
                        block_index=block_index,
                        part_index=part_index,
                        text=part,
                    )
                )
                item_id += 1
        return items

    def _groups(self, items: list[_WorkItem]) -> list[list[_WorkItem]]:
        groups: list[list[_WorkItem]] = []
        current: list[_WorkItem] = []
        current_size = 0

        for item in items:
            item_size = len(item.text)
            if current and current_size + item_size > self.group_char_limit:
                groups.append(current)
                current = []
                current_size = 0
            current.append(item)
            current_size += item_size

        if current:
            groups.append(current)
        return groups

    def correct(
        self,
        document: ExtractedTextDocument,
        progress: Callable[[int, int], None] | None = None,
    ) -> CorrectionResult:
        items = self._work_items(document)
        groups = self._groups(items)
        accepted: dict[int, str] = {}
        warnings: list[str] = []

        for group_index, group in enumerate(groups, start=1):
            payload = [{"id": item.id, "text": item.text} for item in group]
            response = self.ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": CORRECTION_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            "Correct these OCR text blocks and return JSON only:\n"
                            + json.dumps(payload, ensure_ascii=False)
                        ),
                    },
                ],
                format="json",
                options={"temperature": 0},
            )

            try:
                corrections = _parse_corrections(_response_content(response))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                warnings.append(
                    f"Group {group_index}: invalid model response; "
                    f"original text kept ({exc})."
                )
                corrections = {}

            for item in group:
                corrected = corrections.get(item.id, "")
                safe, reason = _is_safe_correction(item.text, corrected)
                if safe:
                    accepted[item.id] = corrected
                else:
                    accepted[item.id] = item.text
                    warnings.append(
                        f"Block {item.id + 1}: original text kept ({reason or 'missing'})."
                    )

            if progress:
                progress(group_index, len(groups))

        parts_by_block: dict[int, list[tuple[int, str]]] = {}
        for item in items:
            parts_by_block.setdefault(item.block_index, []).append(
                (item.part_index, accepted.get(item.id, item.text))
            )

        corrected_blocks: list[TextBlock] = []
        for block_index, block in enumerate(document.blocks):
            parts = sorted(parts_by_block.get(block_index, []))
            corrected_blocks.append(
                TextBlock(
                    text=" ".join(text for _, text in parts).strip() or block.text,
                    style=block.style,
                    page_break_before=block.page_break_before,
                )
            )

        return CorrectionResult(
            original=document,
            corrected_blocks=tuple(corrected_blocks),
            warnings=tuple(dict.fromkeys(warnings)),
        )


def export_docx(result: CorrectionResult) -> bytes:
    """Create a new DOCX without modifying the uploaded source."""
    document = Document()
    available_styles = {style.name for style in document.styles}

    for index, block in enumerate(result.corrected_blocks):
        if block.page_break_before and index > 0:
            document.add_page_break()
        style = block.style if block.style in available_styles else "Normal"
        document.add_paragraph(block.text, style=style)

    output = BytesIO()
    document.save(output)
    return output.getvalue()


def corrected_filename(filename: str) -> str:
    stem = Path(filename).stem.strip() or "document"
    return f"{stem}_corrige.docx"
