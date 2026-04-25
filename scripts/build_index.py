"""
CS4241 | Download sources, clean, chunk, embed, save FAISS index.
Student identity: see src/config.py (STUDENT_NAME, STUDENT_INDEX).
Run from project root: python scripts/build_index.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import requests
from pypdf import PdfReader

from src.chunking import chunk_documents
from src.cleaning import clean_pdf_text, dataframe_to_documents, load_and_clean_csv
from src.config import (
    BUDGET_PDF_URL,
    BUDGET_PDF_URL_FALLBACKS,
    CSV_URL,
    DATA_PROCESSED,
    DATA_RAW,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
)
from src.embeddings import EmbeddingPipeline
from src.retrieval import build_bm25_from_chunks
from src.vector_store import FaissVectorStore


def download(urls: list[str], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    last_err: Exception | None = None
    for url in urls:
        try:
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            dest.write_bytes(r.content)
            return
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not download PDF from any URL: {urls}") from last_err


def extract_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    parts: list[str] = []
    for page in reader.pages:
        t = page.extract_text() or ""
        parts.append(t)
    return clean_pdf_text("\n".join(parts))


def load_local_docs(raw_dir: Path, skip_paths: set[Path]) -> list[dict]:
    """
    Load any extra local datasets from data/raw (CSV/PDF) so users can extend
    retrieval simply by dropping files there.
    """
    docs: list[dict] = []
    for p in sorted(raw_dir.iterdir()):
        if not p.is_file() or p in skip_paths:
            continue
        suffix = p.suffix.lower()
        if suffix == ".csv":
            df = load_and_clean_csv(str(p))
            docs.extend(dataframe_to_documents(df, source_id=f"local_csv:{p.stem}"))
        elif suffix == ".pdf":
            text = extract_pdf(p)
            docs.append(
                {
                    "text": text,
                    "metadata": {"source": f"local_pdf:{p.stem}", "type": "local_pdf"},
                }
            )
    return docs


def build_index() -> None:
    """Download sources, embed, write FAISS + BM25 under data/processed/index_store/."""
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    csv_path = DATA_RAW / "Ghana_Election_Result.csv"
    pdf_path = DATA_RAW / "2025-Budget-Statement.pdf"
    download([CSV_URL], csv_path)
    download([BUDGET_PDF_URL, *BUDGET_PDF_URL_FALLBACKS], pdf_path)

    df = load_and_clean_csv(str(csv_path))
    election_docs = dataframe_to_documents(df, source_id="ghana_election_csv")
    budget_text = extract_pdf(pdf_path)
    budget_docs = [
        {
            "text": budget_text,
            "metadata": {"source": "mofep_2025_budget_pdf", "type": "budget"},
        }
    ]
    local_docs = load_local_docs(raw_dir=DATA_RAW, skip_paths={csv_path, pdf_path})
    all_docs = election_docs + budget_docs + local_docs

    chunks = chunk_documents(
        all_docs, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
    )
    embedder = EmbeddingPipeline()
    texts = [c["text"] for c in chunks]
    vectors = embedder.encode(texts)

    store = FaissVectorStore(vectors.shape[1])
    store.add(vectors, chunks)
    out_dir = DATA_PROCESSED / "index_store"
    store.save(out_dir)

    import pickle

    bm25 = build_bm25_from_chunks(store.chunks)
    with open(out_dir / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)

    print(f"Indexed {len(chunks)} chunks into {out_dir}")
    if local_docs:
        print(f"Included {len(local_docs)} extra local documents from {DATA_RAW}")


def main() -> None:
    build_index()


if __name__ == "__main__":
    # Avoid UnicodeEncodeError when printing paths under non-ASCII dirs (Windows cp1252).
    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, "reconfigure"):
            try:
                _stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
    main()
