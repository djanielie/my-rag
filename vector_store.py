"""Persistent Chroma storage backed by embeddings from local Ollama."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import chromadb
import ollama

from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    OLLAMA_HOST,
    TOP_K,
)
from ingest import ExtractionResult


@dataclass(frozen=True)
class IndexedDocument:
    file_hash: str
    filename: str
    pages: int
    chunks: int


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    filename: str
    page: int
    chunk: int
    file_hash: str
    distance: float


def _batched(values: list[Any], size: int = 100) -> Iterable[list[Any]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]


class PDFVectorStore:
    """Store, retrieve, list, and delete PDF chunks."""

    def __init__(
        self,
        persist_directory: Path | str = CHROMA_DIR,
        collection_name: str = COLLECTION_NAME,
        embedding_model: str = EMBEDDING_MODEL,
        ollama_host: str = OLLAMA_HOST,
        chroma_client: Any | None = None,
        ollama_client: Any | None = None,
    ) -> None:
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        self.embedding_model = embedding_model
        self.ollama = ollama_client or ollama.Client(host=ollama_host)
        self.client = chroma_client or chromadb.PersistentClient(
            path=str(persist_directory)
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        embeddings: list[list[float]] = []
        for batch in _batched(texts, size=32):
            response = self.ollama.embed(model=self.embedding_model, input=batch)
            values = (
                response.embeddings
                if hasattr(response, "embeddings")
                else response["embeddings"]
            )
            embeddings.extend([list(vector) for vector in values])
        return embeddings

    def contains(self, file_hash: str) -> bool:
        existing = self.collection.get(
            where={"file_hash": file_hash},
            limit=1,
            include=["metadatas"],
        )
        return bool(existing["ids"])

    def add_document(self, result: ExtractionResult) -> bool:
        """Index a document. Return False when its content already exists."""
        if self.contains(result.file_hash):
            return False

        chunks = list(result.chunks)
        embeddings = self._embed([chunk.text for chunk in chunks])
        if len(embeddings) != len(chunks):
            raise RuntimeError("The embedding model returned an unexpected result count")

        try:
            for batch_start in range(0, len(chunks), 100):
                batch = chunks[batch_start : batch_start + 100]
                batch_embeddings = embeddings[batch_start : batch_start + 100]
                self.collection.add(
                    ids=[chunk.id for chunk in batch],
                    documents=[chunk.text for chunk in batch],
                    metadatas=[chunk.metadata for chunk in batch],
                    embeddings=batch_embeddings,
                )
        except Exception:
            self.collection.delete(where={"file_hash": result.file_hash})
            raise

        return True

    def list_documents(self) -> list[IndexedDocument]:
        stored = self.collection.get(include=["metadatas"])
        grouped: dict[str, dict[str, Any]] = {}

        for metadata in stored.get("metadatas") or []:
            if not metadata:
                continue
            file_hash = str(metadata["file_hash"])
            item = grouped.setdefault(
                file_hash,
                {
                    "filename": str(metadata["source"]),
                    "pages": int(metadata.get("total_pages", 0)),
                    "chunks": 0,
                },
            )
            item["chunks"] += 1

        return sorted(
            (
                IndexedDocument(
                    file_hash=file_hash,
                    filename=item["filename"],
                    pages=item["pages"],
                    chunks=item["chunks"],
                )
                for file_hash, item in grouped.items()
            ),
            key=lambda document: document.filename.casefold(),
        )

    def delete_document(self, file_hash: str) -> None:
        self.collection.delete(where={"file_hash": file_hash})

    def query(self, question: str, top_k: int = TOP_K) -> list[RetrievedChunk]:
        question = question.strip()
        if not question or self.collection.count() == 0:
            return []

        result_count = min(top_k, self.collection.count())
        question_embedding = self._embed([question])[0]
        results = self.collection.query(
            query_embeddings=[question_embedding],
            n_results=result_count,
            include=["documents", "metadatas", "distances"],
        )

        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]

        retrieved: list[RetrievedChunk] = []
        for text, metadata, distance in zip(documents, metadatas, distances):
            if text is None or metadata is None:
                continue
            retrieved.append(
                RetrievedChunk(
                    text=str(text),
                    filename=str(metadata["source"]),
                    page=int(metadata["page"]),
                    chunk=int(metadata["chunk"]),
                    file_hash=str(metadata["file_hash"]),
                    distance=float(distance),
                )
            )
        return retrieved
