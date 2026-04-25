"""
CS4241 | Top-k retrieval, cosine (IP on normalized vectors), hybrid BM25+dense.
Student identity: see src/config.py (STUDENT_NAME, STUDENT_INDEX).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from src.bm25 import BM25Index, tokenize
from src.config import HYBRID_ALPHA
from src.vector_store import FaissVectorStore

if TYPE_CHECKING:
    from src.embeddings import EmbeddingPipeline


def _minmax(x: list[float]) -> list[float]:
    if not x:
        return []
    lo, hi = min(x), max(x)
    if hi - lo < 1e-9:
        return [1.0 for _ in x]
    return [(v - lo) / (hi - lo) for v in x]


def expand_query_synonyms(query: str) -> str:
    """Lightweight query expansion for domain terms (fix for keyword misses)."""
    # Longer keys first so "winner" matches before the substring "won".
    expansions: list[tuple[str, str]] = [
        ("winner", "NPP NDC presidential candidate votes"),
        ("won", "NPP NDC presidential candidate votes"),
        ("gdp", "gross domestic product"),
        ("inflation", "consumer prices cpi"),
        ("election", "presidential parliamentary votes"),
        ("tax", "revenue mobilisation"),
    ]
    q = query.lower()
    extra: list[str] = []
    for k, v in expansions:
        if k in q:
            extra.append(v)
    if not extra:
        return query
    return query + " " + " ".join(extra)


class HybridRetriever:
    def __init__(
        self,
        store: FaissVectorStore,
        bm25: BM25Index,
        embedder: EmbeddingPipeline,
        alpha: float | None = None,
    ):
        self.store = store
        self.bm25 = bm25
        self.embedder = embedder
        self.alpha = HYBRID_ALPHA if alpha is None else alpha

    def retrieve(
        self,
        query: str,
        k: int,
        use_hybrid: bool = True,
        use_query_expansion: bool = False,
    ) -> list[dict[str, Any]]:
        # Always apply light synonym hints (helps BM25 + dense for short questions).
        q = expand_query_synonyms(query)
        if use_query_expansion:
            q = f"{q} aggregated totals national regional constituency"
        q_emb = self.embedder.encode_query(q)

        n = min(len(self.store.chunks), max(k * 4, k))
        dense_scores, ids = self.store.search(q_emb, n)
        dense_map = {int(i): float(s) for i, s in zip(ids, dense_scores) if i >= 0}

        if not use_hybrid:
            ranked = sorted(dense_map.items(), key=lambda x: x[1], reverse=True)[:k]
            return self._pack_results(ranked, "dense")

        sparse = self.bm25.scores(q)
        sparse_ids = list(range(len(sparse)))
        sparse_pairs = sorted(
            zip(sparse_ids, sparse), key=lambda x: x[1], reverse=True
        )[: max(n, k)]
        sparse_map = {i: s for i, s in sparse_pairs}

        union_ids = set(dense_map) | set(sparse_map)
        dense_list = [dense_map.get(i, 0.0) for i in union_ids]
        sparse_list = [sparse_map.get(i, 0.0) for i in union_ids]
        dn = _minmax(dense_list)
        sn = _minmax(sparse_list)
        fused: list[tuple[int, float]] = []
        for j, i in enumerate(union_ids):
            score = self.alpha * dn[j] + (1 - self.alpha) * sn[j]
            fused.append((i, score))
        fused.sort(key=lambda x: x[1], reverse=True)
        return self._pack_results(fused[:k], "hybrid")

    def _pack_results(
        self, ranked: list[tuple[int, float]], mode: str
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for rank, (chunk_id, score) in enumerate(ranked):
            chunk = self.store.chunks[chunk_id]
            out.append(
                {
                    "rank": rank,
                    "chunk_id": chunk_id,
                    "score": float(score),
                    "text": chunk["text"],
                    "metadata": chunk.get("metadata", {}),
                    "mode": mode,
                }
            )
        return out


def build_bm25_from_chunks(chunks: list[dict[str, Any]]) -> BM25Index:
    tokens = [tokenize(c["text"]) for c in chunks]
    return BM25Index(tokens)
