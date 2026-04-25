"""
CS4241 | Part A helper: compare two chunking configs on overlap / mean chunk length.
Run: python scripts/compare_chunking.py
Student metadata: src/config.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.chunking import chunk_documents
from src.cleaning import load_and_clean_csv
from src.config import DATA_RAW
from src.embeddings import EmbeddingPipeline
from src.retrieval import HybridRetriever, build_bm25_from_chunks
from src.vector_store import FaissVectorStore


def build_temp_store(chunks, embedder):
    texts = [c["text"] for c in chunks]
    v = embedder.encode(texts)
    store = FaissVectorStore(v.shape[1])
    store.add(v, chunks)
    bm25 = build_bm25_from_chunks(chunks)
    return HybridRetriever(store, bm25, embedder)


def main() -> None:
    csv_path = DATA_RAW / "Ghana_Election_Result.csv"
    if not csv_path.exists():
        print("Run scripts/build_index.py first to download data.")
        return
    df = load_and_clean_csv(str(csv_path))
    from src.cleaning import dataframe_to_documents

    docs = dataframe_to_documents(df, source_id="ghana_election_csv")

    embedder = EmbeddingPipeline()
    queries = [
        "Which region had the highest valid votes in the dataset?",
        "What are the vote totals mentioned?",
    ]

    configs = [
        ("small_overlap", 300, 40),
        ("large_overlap", 450, 120),
    ]
    for name, size, ov in configs:
        chunks = chunk_documents(docs, size, ov)
        lens = [len(c["text"]) for c in chunks]
        print(f"\n{name}: chunk_size={size} overlap={ov}")
        print(f"  num_chunks={len(chunks)} mean_len={sum(lens)/len(lens):.1f}")
        retr = build_temp_store(chunks, embedder)
        for q in queries:
            hits = retr.retrieve(q, k=3, use_hybrid=False)
            print(f"  query={q!r}")
            for h in hits:
                print(f"    score={h['score']:.4f} preview={h['text'][:80]!r}")


if __name__ == "__main__":
    main()
