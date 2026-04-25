"""
CS4241 | Embedding pipeline using sentence-transformers (manual orchestration).
Student identity: see src/config.py (STUDENT_NAME, STUDENT_INDEX).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL_NAME


class EmbeddingPipeline:
    def __init__(self, model_name: str | None = None):
        name = model_name or EMBEDDING_MODEL_NAME
        self.model = SentenceTransformer(name)
        self.model_name = name

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        emb = self.model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return emb.astype(np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        v = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(v, dtype=np.float32).reshape(1, -1)
