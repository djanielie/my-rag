"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("RAG_DATA_DIR", BASE_DIR / "data")).resolve()
CHROMA_DIR = DATA_DIR / "chroma"

CHAT_MODEL = os.getenv("RAG_CHAT_MODEL", "qwen2.5-coder:14b")
EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
COLLECTION_NAME = os.getenv("RAG_COLLECTION", "pdf_chunks")

CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("RAG_TOP_K", "5"))

if CHUNK_SIZE <= 0:
    raise ValueError("RAG_CHUNK_SIZE must be greater than zero")
if CHUNK_OVERLAP < 0 or CHUNK_OVERLAP >= CHUNK_SIZE:
    raise ValueError(
        "RAG_CHUNK_OVERLAP must be non-negative and smaller than RAG_CHUNK_SIZE"
    )
if TOP_K <= 0:
    raise ValueError("RAG_TOP_K must be greater than zero")
