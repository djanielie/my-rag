from uuid import uuid4

import chromadb

from ingest import DocumentChunk, ExtractionResult
from vector_store import PDFVectorStore


class FakeEmbeddingClient:
    def embed(self, model, input):
        return {
            "embeddings": [
                [
                    float(len(text)),
                    float(sum(ord(character) for character in text) % 101),
                    1.0,
                ]
                for text in input
            ]
        }


def make_result() -> ExtractionResult:
    file_hash = "abc123"
    chunks = (
        DocumentChunk(
            id="chunk-1",
            text="The launch date is June 4.",
            metadata={
                "source": "project.pdf",
                "page": 1,
                "chunk": 0,
                "file_hash": file_hash,
                "total_pages": 2,
            },
        ),
        DocumentChunk(
            id="chunk-2",
            text="The budget is five thousand dollars.",
            metadata={
                "source": "project.pdf",
                "page": 2,
                "chunk": 0,
                "file_hash": file_hash,
                "total_pages": 2,
            },
        ),
    )
    return ExtractionResult(
        filename="project.pdf",
        file_hash=file_hash,
        total_pages=2,
        empty_pages=(),
        chunks=chunks,
    )


def make_store(tmp_path) -> PDFVectorStore:
    return PDFVectorStore(
        persist_directory=tmp_path,
        collection_name=f"test_{uuid4().hex}",
        chroma_client=chromadb.Client(),
        ollama_client=FakeEmbeddingClient(),
    )


def test_document_lifecycle_and_duplicate_detection(tmp_path):
    store = make_store(tmp_path)
    result = make_result()

    assert store.add_document(result) is True
    assert store.add_document(result) is False

    documents = store.list_documents()
    assert len(documents) == 1
    assert documents[0].filename == "project.pdf"
    assert documents[0].pages == 2
    assert documents[0].chunks == 2

    retrieved = store.query("When is the launch?", top_k=1)
    assert len(retrieved) == 1
    assert retrieved[0].filename == "project.pdf"
    assert retrieved[0].page in {1, 2}

    store.delete_document(result.file_hash)
    assert store.list_documents() == []
    assert store.query("When is the launch?") == []
