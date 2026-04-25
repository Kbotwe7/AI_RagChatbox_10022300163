"""
CS4241 | Chat generation via xAI Grok (OpenAI-compatible client, no LangChain).
Set XAI_API_KEY (or GROK_API_KEY) in .env — see .env.example.
"""
from __future__ import annotations

from typing import Any

from openai import APIStatusError, OpenAI

from src.config import GROK_MODEL, XAI_API_KEY, XAI_BASE_URL


def _api_error_detail(err: APIStatusError) -> str:
    try:
        body = err.response.json() if err.response is not None else None
        if isinstance(body, dict):
            e = body.get("error")
            if isinstance(e, dict) and e.get("message"):
                return str(e["message"])
            if isinstance(e, str):
                return e
    except Exception:
        pass
    return str(err)


def chat_complete(messages: list[dict[str, str]], temperature: float = 0.2) -> dict[str, Any]:
    if not (XAI_API_KEY or "").strip():
        raise RuntimeError(
            "XAI_API_KEY (or GROK_API_KEY) is not set. Add your xAI key to .env — https://console.x.ai/"
        )
    client = OpenAI(api_key=XAI_API_KEY, base_url=XAI_BASE_URL)
    try:
        resp = client.chat.completions.create(
            model=GROK_MODEL,
            messages=messages,
            temperature=temperature,
        )
    except APIStatusError as e:
        detail = _api_error_detail(e)
        if e.status_code in (400, 404):
            raise RuntimeError(
                f"xAI chat request failed ({e.status_code}) with model {GROK_MODEL!r}. "
                f"{detail} "
                "Set **GROK_MODEL** in `.env` or Streamlit **Secrets** to a model your team can use "
                "(**Console → Models**, or https://docs.x.ai/docs/models). "
                "Examples that often work: `grok-3-mini`, `grok-3`, `grok-4-latest`."
            ) from e
        raise
    text = resp.choices[0].message.content or ""
    return {"text": text, "model": GROK_MODEL, "raw": resp.model_dump()}
