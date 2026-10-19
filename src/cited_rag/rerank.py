"""Rerankers: cross-encoder and ColBERT-style late interaction (MaxSim)."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .embed import Embedder


class Reranker(Protocol):
    name: str

    def score(self, query: str, texts: list[str]) -> np.ndarray: ...


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class CrossEncoderReranker:
    """Scores (query, passage) pairs jointly. Output is a probability-like value in [0, 1]."""

    def __init__(self, model_name: str, batch_size: int = 16) -> None:
        from sentence_transformers import CrossEncoder

        self.name = model_name
        self._model = CrossEncoder(model_name, device="cpu", max_length=512)
        self.batch_size = batch_size

    def score(self, query: str, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros(0, dtype=np.float32)
        raw = self._model.predict(
            [(query, t) for t in texts],
            batch_size=self.batch_size,
            activation_fn=_identity(),
            convert_to_numpy=True,
        )
        return _sigmoid(np.asarray(raw, dtype=np.float32))


def _identity():
    import torch

    return torch.nn.Identity()


def maxsim(query_tokens: np.ndarray, doc_tokens: np.ndarray) -> float:
    """Mean over query tokens of the max cosine similarity to any document token.

    ColBERT sums instead of averaging; averaging keeps the score in [-1, 1] so a single
    confidence threshold works across query lengths.
    """
    if query_tokens.size == 0 or doc_tokens.size == 0:
        return 0.0
    sim = query_tokens @ doc_tokens.T
    return float(sim.max(axis=1).mean())


class MaxSimReranker:
    """Late interaction over the bi-encoder's own token embeddings.

    This is an honest approximation, not ColBERT: bge-small was trained for pooled
    embeddings, not token-level matching. Document token vectors are cached on disk.
    """

    def __init__(self, embedder: Embedder) -> None:
        self.name = f"maxsim:{embedder.name}"
        self._emb = embedder

    def score(self, query: str, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros(0, dtype=np.float32)
        q = self._emb.token_embeddings([query])[0]
        docs = self._emb.token_embeddings(texts)
        return np.array([maxsim(q, d) for d in docs], dtype=np.float32)
