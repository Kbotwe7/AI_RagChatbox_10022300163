"""
CS4241 | Innovation (Part G): simple feedback-weighted re-ranking for chunk_ids.
Student identity: see src/config.py (STUDENT_NAME, STUDENT_INDEX).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class FeedbackStore:
    """Persist thumbs up/down per chunk_id; boosts scores in retrieval re-ranking."""

    def __init__(self, path: Path):
        self.path = path
        self.scores: dict[str, float] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            self.scores = json.loads(self.path.read_text(encoding="utf-8"))

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.scores, indent=2), encoding="utf-8")

    def record(self, chunk_id: int, label: str) -> None:
        key = str(chunk_id)
        delta = 0.15 if label == "up" else -0.15
        self.scores[key] = self.scores.get(key, 0.0) + delta
        self._save()

    def boost(self, chunk_id: int) -> float:
        return float(self.scores.get(str(chunk_id), 0.0))


def apply_feedback_boost(
    results: list[dict[str, Any]], store: FeedbackStore, weight: float = 0.25
) -> list[dict[str, Any]]:
    adjusted = []
    for r in results:
        b = store.boost(int(r["chunk_id"]))
        new_score = float(r["score"]) + weight * b
        adjusted.append({**r, "score": new_score, "feedback_boost": b})
    adjusted.sort(key=lambda x: x["score"], reverse=True)
    for i, r in enumerate(adjusted):
        r["rank"] = i
    return adjusted
