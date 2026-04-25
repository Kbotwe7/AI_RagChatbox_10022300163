"""
CS4241 | FAISS vector store (manual): add, search, persist.
Student identity: see src/config.py (STUDENT_NAME, STUDENT_INDEX).
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np


class FaissVectorStore:
    """Inner product on L2-normalized vectors == cosine similarity."""

    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: list[dict[str, Any]] = []

    def add(self, vectors: np.ndarray, chunks: list[dict[str, Any]]) -> None:
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, k: int) -> tuple[list[float], list[int]]:
        if query_vector.dtype != np.float32:
            query_vector = query_vector.astype(np.float32)
        scores, ids = self.index.search(query_vector, k)
        return scores[0].tolist(), ids[0].tolist()

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "index.faiss"))
        meta = {"dim": self.dim, "n_chunks": len(self.chunks)}
        (directory / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        with open(directory / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

    @classmethod
    def load(cls, directory: Path) -> "FaissVectorStore":
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        store = cls(int(meta["dim"]))
        store.index = faiss.read_index(str(directory / "index.faiss"))
        with open(directory / "chunks.pkl", "rb") as f:
            store.chunks = pickle.load(f)
        return store
