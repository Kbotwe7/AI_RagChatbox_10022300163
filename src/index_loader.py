"""
CS4241 | Load persisted FAISS + BM25 for the running app.
Student identity: see src/config.py (STUDENT_NAME, STUDENT_INDEX).
"""
from __future__ import annotations

import pickle
from pathlib import Path

from src.config import DATA_PROCESSED
from src.embeddings import EmbeddingPipeline
from src.retrieval import HybridRetriever
from src.vector_store import FaissVectorStore


def load_retriever(index_dir: Path | None = None) -> tuple[FaissVectorStore, HybridRetriever, EmbeddingPipeline]:
    directory = index_dir or (DATA_PROCESSED / "index_store")
    store = FaissVectorStore.load(directory)
    with open(directory / "bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)
    embedder = EmbeddingPipeline()
    retriever = HybridRetriever(store, bm25, embedder)
    return store, retriever, embedder
