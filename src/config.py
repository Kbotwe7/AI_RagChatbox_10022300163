"""
CS4241 | Configure student identity and paths here or via .env (see README).
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Load from project root so keys work even if the process cwd is elsewhere (e.g. Streamlit).
# override=True: if Windows has an empty XAI_* user env var, still apply values from .env.
load_dotenv(PROJECT_ROOT / ".env", override=True)
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

STUDENT_NAME = os.getenv("STUDENT_NAME", "Kojo Baafi Botwe")
STUDENT_INDEX = os.getenv("STUDENT_INDEX", "10022300163")

CSV_URL = (
    "https://raw.githubusercontent.com/GodwinDansoAcity/acitydataset/main/"
    "Ghana_Election_Result.csv"
)
# Exam PDF link had a line break that merged into "andEconomic"; official path uses hyphens.
BUDGET_PDF_URL = (
    "https://mofep.gov.gh/sites/default/files/budget-statements/"
    "2025-Budget-Statement-and-Economic-Policy_v4.pdf"
)
BUDGET_PDF_URL_FALLBACKS = [
    "https://mofep.gov.gh/sites/default/files/budget-statements/"
    "2025-Budget-Statement-and-Economic-Policy.pdf",
    "https://mofep.gov.gh/sites/default/files/budget-statements/"
    "2025-Budget-Statement-andEconomic-Policy_v4.pdf",
]

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Chat: xAI Grok (OpenAI-compatible HTTP API — see https://docs.x.ai/docs/guides/chat)
XAI_API_KEY = os.getenv("XAI_API_KEY", "").strip() or os.getenv("GROK_API_KEY", "").strip()
GROK_MODEL = os.getenv("GROK_MODEL", "grok-beta")
XAI_BASE_URL = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")

# Chunking defaults (justify in README / docs)
DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "450"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "90"))

# Retrieval
TOP_K = int(os.getenv("TOP_K", "8"))
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.65"))  # weight on dense vs sparse

# Context window (approximate max chars for injected context)
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "12000"))


def student_header() -> str:
    return f"Student: {STUDENT_NAME} | Index: {STUDENT_INDEX}"
