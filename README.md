# Local Document AI

A private, local question-answering application for text-based PDF files. It
uses Ollama for embeddings and answer generation, ChromaDB for persistent
retrieval, PyMuPDF for text extraction, and Streamlit for the interface.
It can also use a local Ollama model to clean OCR errors from PDF, DOCX,
Markdown, and text files and export the result as a new DOCX.

The application keeps the vector index on your computer. PDF contents are sent
only to the locally running Ollama service.

## Prerequisites

- Python 3.10 or newer
- Ollama installed and running
- A local chat model and embedding model

The default chat model matches an already installed model:

```powershell
ollama pull qwen2.5-coder:14b
ollama pull nomic-embed-text
```

The Qwen model generates answers. `nomic-embed-text` is a separate model used
to turn document chunks and questions into vectors for similarity search.

## Install and run

From this directory on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run main.py
```

Upload one or more PDFs in the sidebar, select **Index selected PDFs**, and
then ask questions in the chat. Indexed documents remain available after an
application restart.

Use the sidebar's **Settings** section to choose any installed Ollama
text-generation model. The choice is saved locally in `data/settings.json` and
restored the next time the application starts. Select **Refresh installed
models** after pulling another model while the application is open.

## Correct OCR text

Open the **OCR correction** tab and upload one PDF, DOCX, Markdown, or TXT
file. The selected Ollama chat model processes the text locally in small
groups and applies conservative spelling, punctuation, word-boundary, and
common OCR corrections. The interface shows the original and corrected text
before offering a new DOCX for download. The source file is never overwritten.

This feature cleans an existing text layer. For a scanned PDF, first run OCR
with Adobe Acrobat or another OCR program. A PDF containing only page images
will be rejected with an explanatory message. Simple paragraphs, headings,
and PDF page boundaries are preserved where possible; complex Word layouts,
tables, images, footnotes, and exact typography are not reproduced.

Language, numbers, and meaning are protected by the correction prompt and
validation checks, but model output should still be reviewed before replacing
an authoritative document.

## Configuration

These optional environment variables override the defaults:

| Variable | Default | Purpose |
| --- | --- | --- |
| `RAG_CHAT_MODEL` | `qwen2.5-coder:14b` | Ollama model used for answers |
| `RAG_EMBEDDING_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server address |
| `RAG_TOP_K` | `5` | Chunks retrieved for each question |
| `RAG_CHUNK_SIZE` | `1200` | Approximate characters per chunk |
| `RAG_CHUNK_OVERLAP` | `200` | Characters shared by adjacent chunks |
| `RAG_DATA_DIR` | `./data` | Persistent local application data |

If you change the embedding model, delete `data/chroma` and re-index the PDFs.
Embedding models generally produce incompatible vector dimensions.

## Current scope

- Supports multiple persistent PDF documents.
- Displays filename, page number, and retrieved passage for each answer.
- Avoids indexing the same file content twice.
- Supports deleting indexed documents.
- Supports PDFs containing selectable text.
- Corrects OCR text from PDF, DOCX, Markdown, and TXT into a new DOCX.

Scanned or image-only PDFs need OCR, which is not part of this first version.
Pages without extractable text are reported during indexing.

## Tests

```powershell
pytest
```
