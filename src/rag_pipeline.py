"""
CS4241 | Full RAG pipeline with stage logging (Part D).
Student identity: see src/config.py (STUDENT_NAME, STUDENT_INDEX).
"""
from __future__ import annotations

import time
from typing import Any

from src.config import MAX_CONTEXT_CHARS, TOP_K
from src.embeddings import EmbeddingPipeline
from src.feedback_store import FeedbackStore, apply_feedback_boost
from src.llm import chat_complete
from src.prompts import build_messages
from src.retrieval import HybridRetriever


class StageLog(dict[str, Any]):
    """Small helper for consistent stage trace records."""

    def __init__(self, *, name: str, **payload: Any) -> None:
        super().__init__(name=name, **payload)


class RagPipeline:
    def __init__(
        self,
        retriever: HybridRetriever,
        embedder: EmbeddingPipeline,
        feedback: FeedbackStore | None = None,
    ):
        self.retriever = retriever
        self.embedder = embedder
        self.feedback = feedback

    def run(
        self,
        query: str,
        *,
        use_rag: bool = True,
        use_hybrid: bool = True,
        use_query_expansion: bool = False,
        prompt_variant: str = "strict",
        apply_feedback: bool = True,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        k = top_k or TOP_K
        trace: dict[str, Any] = {"query": query, "stages": []}

        t0 = time.perf_counter()

        if not use_rag:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": query},
            ]
            trace["stages"].append(
                StageLog(
                    name="retrieval_skipped",
                    detail="pure_llm_mode",
                    retrieved=[],
                )
            )
            trace["stages"].append(
                StageLog(
                    name="prompt",
                    messages=messages,
                    context_blocks=[],
                )
            )
            gen = chat_complete(messages)
            trace["stages"].append(StageLog(name="generation", output=gen["text"], model=gen["model"]))
            trace["final_answer"] = gen["text"]
            trace["latency_s"] = time.perf_counter() - t0
            return trace

        retrieved = self.retriever.retrieve(
            query,
            k,
            use_hybrid=use_hybrid,
            use_query_expansion=use_query_expansion,
        )
        trace["stages"].append(
            StageLog(
                name="retrieval",
                top_k=k,
                use_hybrid=use_hybrid,
                use_query_expansion=use_query_expansion,
                hits=retrieved,
            )
        )

        if self.feedback and apply_feedback:
            retrieved = apply_feedback_boost(retrieved, self.feedback)
            trace["stages"].append(
                StageLog(name="feedback_rerank", hits=retrieved)
            )

        messages, used_blocks = build_messages(
            query, retrieved, MAX_CONTEXT_CHARS, variant=prompt_variant
        )
        trace["stages"].append(
            StageLog(
                name="context_selection",
                n_blocks=len(used_blocks),
                max_chars=MAX_CONTEXT_CHARS,
                blocks=used_blocks,
            )
        )
        trace["stages"].append(
            StageLog(
                name="prompt",
                prompt_variant=prompt_variant,
                messages=messages,
            )
        )

        gen = chat_complete(messages)
        trace["stages"].append(
            StageLog(name="generation", output=gen["text"], model=gen["model"])
        )
        trace["final_answer"] = gen["text"]
        trace["latency_s"] = time.perf_counter() - t0
        return trace
