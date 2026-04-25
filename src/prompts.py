"""
CS4241 | Prompt templates, context packing, anti-hallucination instructions.
Student identity: see src/config.py (STUDENT_NAME, STUDENT_INDEX).
"""
from __future__ import annotations

from typing import Any


SYSTEM_BASE = """You are an assistant for Academic City University (Ghana).
Answer using ONLY the provided CONTEXT when it is sufficient.
You may combine facts from multiple CONTEXT blocks when each fact appears explicitly (e.g. comparing vote numbers on different rows).
If the context does not contain the answer, say you do not have enough information in the provided documents — do not invent facts.
Cite which part of the context you used when possible (short paraphrase, not long quotes)."""


def build_user_prompt(
    query: str,
    context_blocks: list[dict[str, Any]],
    variant: str = "strict",
) -> str:
    """
    variant:
      - strict: strongest grounding
      - concise: shorter answers
      - exploratory: allows slightly more synthesis but still grounded
    """
    ctx_parts: list[str] = []
    for i, block in enumerate(context_blocks, start=1):
        src = block.get("metadata", {})
        tag = f"[{i}] source={src.get('source', '?')} type={src.get('type', '?')}"
        ctx_parts.append(f"{tag}\n{block['text']}")
    context_str = "\n\n---\n\n".join(ctx_parts)

    if variant == "strict":
        rules = (
            "Rules: Use only CONTEXT. You may synthesize from multiple rows "
            "(e.g. regional presidential vote tables) only using numbers and names "
            "that appear in CONTEXT. If CONTEXT supports a partial answer "
            "(e.g. who led in the regions shown), give that and state what is not "
            "in CONTEXT (e.g. nationwide totals). Reply exactly "
            "'I cannot answer from the provided documents.' only when CONTEXT "
            "contains no usable facts for the question at all."
        )
    elif variant == "concise":
        rules = "Rules: Be brief (max 6 sentences). Use only CONTEXT."
    else:
        rules = (
            "Rules: Ground every claim in CONTEXT. If uncertain, state uncertainty explicitly."
        )

    return f"""QUESTION:
{query}

CONTEXT:
{context_str}

{rules}
"""


def build_messages(
    query: str,
    retrieved: list[dict[str, Any]],
    max_chars: int,
    variant: str = "strict",
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    """
    Rank by retrieval score, greedily add chunks until max_chars.
    Returns OpenAI-style messages and the list of context blocks actually used.
    """
    ranked = sorted(retrieved, key=lambda r: r.get("score", 0.0), reverse=True)
    used: list[dict[str, Any]] = []
    total = 0
    for r in ranked:
        piece = {"text": r["text"], "metadata": {**r.get("metadata", {}), "score": r.get("score")}}
        add_len = len(r["text"]) + 40
        if total + add_len > max_chars and used:
            break
        used.append(piece)
        total += add_len

    user = build_user_prompt(query, used, variant=variant)
    messages = [
        {"role": "system", "content": SYSTEM_BASE},
        {"role": "user", "content": user},
    ]
    return messages, used
