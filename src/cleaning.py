"""
CS4241 | Data cleaning for CSV election results and extracted PDF text.
Student identity: see src/config.py (STUDENT_NAME, STUDENT_INDEX).
"""
from __future__ import annotations

import re
from typing import Any

import pandas as pd


def clean_election_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names, strip strings, drop fully empty rows."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()
    df = df.dropna(how="all")
    df = df[df.astype(str).apply(lambda r: any(v and v != "nan" for v in r), axis=1)]
    return df.reset_index(drop=True)


def text_normalize(s: str) -> str:
    s = s.replace("\x00", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def clean_pdf_text(raw: str) -> str:
    """Remove repeated headers/footers heuristics, collapse whitespace."""
    raw = text_normalize(raw)
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    # Drop very short isolated lines that are often page numbers
    kept: list[str] = []
    for ln in lines:
        if len(ln) <= 2 and ln.isdigit():
            continue
        kept.append(ln)
    return text_normalize("\n".join(kept))


def load_and_clean_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return clean_election_csv(df)


def dataframe_to_documents(df: pd.DataFrame, source_id: str) -> list[dict[str, Any]]:
    """Turn tabular rows into textual 'documents' for chunking."""
    docs: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        parts = [f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col])]
        text = "; ".join(parts)
        docs.append(
            {
                "text": text,
                "metadata": {
                    "source": source_id,
                    "row_index": int(idx),
                    "type": "election_csv",
                },
            }
        )
    return docs
