"""Retrieval and answer metrics. Relevance is binary: right doc and overlapping pages."""

from __future__ import annotations

import math
from collections.abc import Sequence


def is_relevant(doc_id: str, pages: Sequence[int], exp_doc: str, exp_pages: Sequence[int]) -> bool:
    return doc_id == exp_doc and bool(set(pages) & set(exp_pages))


def hit_at_k(rels: Sequence[bool], k: int) -> float:
    return float(any(rels[:k]))


def mrr(rels: Sequence[bool], k: int | None = None) -> float:
    for i, r in enumerate(rels[:k] if k else rels, start=1):
        if r:
            return 1.0 / i
    return 0.0


def ndcg_at_k(rels: Sequence[bool], k: int, n_relevant: int | None = None) -> float:
    """Binary NDCG. ``n_relevant`` = relevant items that exist (for the ideal ranking);
    defaults to the number found in ``rels``."""
    dcg = sum(1.0 / math.log2(i + 2) for i, r in enumerate(rels[:k]) if r)
    n = sum(rels) if n_relevant is None else n_relevant
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, n)))
    return dcg / ideal if ideal > 0 else 0.0


def key_fact_recall(answer: str, facts: Sequence[Sequence[str] | str]) -> float:
    """Share of key facts present in the answer. A fact is a string or a list of
    acceptable alternatives; matching is case-insensitive substring."""
    if not facts:
        return 1.0
    a = " ".join(answer.lower().split())
    found = 0
    for f in facts:
        alts = [f] if isinstance(f, str) else list(f)
        found += any(alt.lower() in a for alt in alts)
    return found / len(facts)


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0
