import json
from io import BytesIO

import pymupdf
import pytest
from docx import Document

from document_corrector import (
    DocumentCorrector,
    corrected_filename,
    export_docx,
    extract_editable_text,
)


def make_docx() -> bytes:
    document = Document()
    document.add_heading("Rapport OCR", level=1)
    document.add_paragraph("Le texte est b1en lisible en 2024.")
    output = BytesIO()
    document.save(output)
    return output.getvalue()


def make_pdf(text: str = "") -> bytes:
    document = pymupdf.open()
    page = document.new_page()
    if text:
        page.insert_text((72, 72), text)
    content = document.tobytes()
    document.close()
    return content


class CorrectingClient:
    def chat(self, **kwargs):
        payload = json.loads(kwargs["messages"][1]["content"].split("\n", 1)[1])
        corrections = [
            {"id": item["id"], "text": item["text"].replace("b1en", "bien")}
            for item in payload
        ]
        return {"message": {"content": json.dumps({"corrections": corrections})}}


class NumberChangingClient:
    def chat(self, **kwargs):
        payload = json.loads(kwargs["messages"][1]["content"].split("\n", 1)[1])
        corrections = [
            {"id": item["id"], "text": item["text"].replace("2024", "2025")}
            for item in payload
        ]
        return {"message": {"content": json.dumps({"corrections": corrections})}}


class InvalidResponseClient:
    def chat(self, **kwargs):
        return {"message": {"content": "not valid JSON"}}


def test_extracts_txt_and_markdown_structure():
    text_document = extract_editable_text(
        "Premier paragraphe.\n\nDeuxième paragraphe.".encode(),
        "notes.txt",
    )
    markdown_document = extract_editable_text(
        "# Titre\n\nTexte avec errreur.".encode(),
        "notes.md",
    )

    assert [block.text for block in text_document.blocks] == [
        "Premier paragraphe.",
        "Deuxième paragraphe.",
    ]
    assert markdown_document.blocks[0].text == "Titre"
    assert markdown_document.blocks[0].style == "Heading 1"


def test_extracts_docx_paragraphs_and_styles():
    document = extract_editable_text(make_docx(), "rapport.docx")

    assert document.blocks[0].text == "Rapport OCR"
    assert document.blocks[0].style == "Heading 1"
    assert document.blocks[1].text.endswith("2024.")


def test_extracts_pdf_text_and_rejects_image_only_pdf():
    document = extract_editable_text(
        make_pdf("Texte OCR de la page."),
        "scan.pdf",
    )

    assert "Texte OCR" in document.text
    with pytest.raises(ValueError, match="Run OCR"):
        extract_editable_text(make_pdf(), "image-only.pdf")


def test_corrects_blocks_and_exports_a_new_docx():
    original = extract_editable_text(make_docx(), "rapport.docx")
    result = DocumentCorrector(
        model="test",
        ollama_client=CorrectingClient(),
        group_char_limit=500,
    ).correct(original)

    assert "bien lisible" in result.corrected_text
    assert "2024" in result.corrected_text
    assert result.warnings == ()

    exported = Document(BytesIO(export_docx(result)))
    assert exported.paragraphs[0].style.name == "Heading 1"
    assert "bien lisible" in "\n".join(p.text for p in exported.paragraphs)
    assert corrected_filename("rapport.pdf") == "rapport_corrige.docx"


def test_guardrail_keeps_original_when_model_changes_numbers():
    original = extract_editable_text(
        "Le total officiel est 2024.".encode(),
        "total.txt",
    )
    result = DocumentCorrector(
        model="test",
        ollama_client=NumberChangingClient(),
        group_char_limit=500,
    ).correct(original)

    assert result.corrected_text == original.text
    assert any("numbers changed" in warning for warning in result.warnings)


def test_invalid_model_response_never_deletes_original_text():
    original = extract_editable_text(
        "Texte original indispensable.".encode(),
        "original.txt",
    )
    result = DocumentCorrector(
        model="test",
        ollama_client=InvalidResponseClient(),
        group_char_limit=500,
    ).correct(original)

    assert result.corrected_text == original.text
    assert result.warnings
