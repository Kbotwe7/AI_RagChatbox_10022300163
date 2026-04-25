"""
CS4241 | Chat generation via xAI Grok (OpenAI-compatible client, no LangChain).
Set XAI_API_KEY (or GROK_API_KEY) in .env — see .env.example.
"""
from __future__ import annotations

from typing import Any

from openai import OpenAI

from src.config import GROK_MODEL, XAI_API_KEY, XAI_BASE_URL


def chat_complete(messages: list[dict[str, str]], temperature: float = 0.2) -> dict[str, Any]:
    if not (XAI_API_KEY or "").strip():
        raise RuntimeError(
            "XAI_API_KEY (or GROK_API_KEY) is not set. Add your xAI key to .env — https://console.x.ai/"
        )
    client = OpenAI(api_key=XAI_API_KEY, base_url=XAI_BASE_URL)
    resp = client.chat.completions.create(
        model=GROK_MODEL,
        messages=messages,
        temperature=temperature,
    )
    text = resp.choices[0].message.content or ""
    return {"text": text, "model": GROK_MODEL, "raw": resp.model_dump()}
