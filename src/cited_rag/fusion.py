"""Rank fusion."""

from __future__ import annotations

from collections.abc import Sequence


def rrf(rankings: Sequence[Sequence[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion (Cormack et al. 2009).

    ``rankings`` are lists of item ids, best first. Score = sum 1 / (k + rank), rank 1-based.
    Returns (id, score) sorted by score desc; ties keep first-seen order.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])


def top_indices(scores, n: int, mask=None) -> list[int]:
    """Indices of the ``n`` highest scores, optionally restricted to ``mask``."""
    import numpy as np

    s = np.asarray(scores, dtype=np.float64).copy()
    if mask is not None:
        s[~mask] = -np.inf
    n = min(n, int(np.isfinite(s).sum()))
    if n <= 0:
        return []
    idx = np.argpartition(-s, n - 1)[:n]
    return [int(i) for i in idx[np.argsort(-s[idx], kind="stable")]]
