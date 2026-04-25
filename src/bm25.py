"""
CS4241 | Manual BM25 scoring (hybrid retrieval keyword leg).
Student identity: see src/config.py (STUDENT_NAME, STUDENT_INDEX).
"""
from __future__ import annotations

import math
import re
from collections import Counter


_TOKEN = re.compile(r"[a-z0-9]+", re.I)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


class BM25Index:
    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_tokens = corpus_tokens
        self.N = len(corpus_tokens)
        self.doc_len = [len(t) for t in corpus_tokens]
        self.avgdl = sum(self.doc_len) / self.N if self.N else 0.0
        self.df: dict[str, int] = {}
        for tokens in corpus_tokens:
            for term in set(tokens):
                self.df[term] = self.df.get(term, 0) + 1
        self.idf: dict[str, float] = {}
        for term, df in self.df.items():
            self.idf[term] = math.log(1 + (self.N - df + 0.5) / (df + 0.5))

    def scores(self, query: str) -> list[float]:
        q_tokens = tokenize(query)
        if not q_tokens or not self.N:
            return [0.0] * self.N
        scores = [0.0] * self.N
        for i, doc_tokens in enumerate(self.corpus_tokens):
            tf = Counter(doc_tokens)
            dl = self.doc_len[i]
            denom_norm = self.k1 * (1 - self.b + self.b * dl / self.avgdl) if self.avgdl else self.k1
            for term in q_tokens:
                if term not in tf:
                    continue
                idf = self.idf.get(term, 0.0)
                f = tf[term]
                score = idf * (f * (self.k1 + 1)) / (f + denom_norm)
                scores[i] += score
        return scores
