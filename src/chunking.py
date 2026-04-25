"""
CS4241 | Character-based chunking with overlap (manual, no frameworks).
Student identity: see src/config.py (STUDENT_NAME, STUDENT_INDEX).
"""
from __future__ import annotations

from typing import Any


def chunk_text(
    text: str,
    chunk_size: int,
    overlap: int,
    metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Sliding windows over characters. Overlap preserves boundary context for retrieval.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    metadata = metadata or {}
    chunks: list[dict[str, Any]] = []
    start = 0
    n = len(text)
    chunk_id = 0
    while start < n:
        end = min(start + chunk_size, n)
        piece = text[start:end].strip()
        if piece:
            md = {**metadata, "chunk_id": chunk_id, "char_start": start, "char_end": end}
            chunks.append({"text": piece, "metadata": md})
            chunk_id += 1
        if end >= n:
            break
        start = end - overlap
    return chunks


def chunk_documents(
    documents: list[dict[str, Any]],
    chunk_size: int,
    overlap: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for doc in documents:
        base_meta = doc.get("metadata") or {}
        for c in chunk_text(doc["text"], chunk_size, overlap, base_meta):
            out.append(c)
    return out
