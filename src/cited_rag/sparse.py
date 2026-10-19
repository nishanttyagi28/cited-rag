"""BM25 keyword retrieval (rank_bm25.BM25Okapi)."""

from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi

_TOKEN = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")
STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "does",
        "do",
        "for",
        "from",
        "has",
        "have",
        "how",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "this",
        "to",
        "was",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "should",
    ]
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


class BM25Index:
    def __init__(self, texts: list[str]) -> None:
        self._bm25 = BM25Okapi([tokenize(t) or ["_"] for t in texts])
        self.size = len(texts)

    def scores(self, query: str) -> np.ndarray:
        q = tokenize(query)
        if not q:
            return np.zeros(self.size, dtype=np.float32)
        return np.asarray(self._bm25.get_scores(q), dtype=np.float32)
