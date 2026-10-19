"""Embedding backends. All return L2-normalised float32 vectors."""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np

from .cache import EmbeddingCache, content_key

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    name: str

    def embed_docs(self, texts: list[str]) -> np.ndarray: ...
    def embed_queries(self, texts: list[str]) -> np.ndarray: ...
    def token_embeddings(self, texts: list[str]) -> list[np.ndarray]: ...


def _normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return (x / np.maximum(n, 1e-12)).astype(np.float32)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, batch_size: int = 16) -> None:
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name, device="cpu")
        self._query_prefix = BGE_QUERY_PREFIX if "bge" in model_name.lower() else ""

    def embed_docs(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        v = self._model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True)
        return np.asarray(v, dtype=np.float32)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self.embed_docs([self._query_prefix + t for t in texts])

    def token_embeddings(self, texts: list[str]) -> list[np.ndarray]:
        out = self._model.encode(texts, batch_size=self.batch_size, output_value="token_embeddings")
        return [_normalize(np.asarray(t.cpu().numpy() if hasattr(t, "cpu") else t)) for t in out]

    @property
    def dim(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())


_TOKEN = re.compile(r"[a-z0-9]+")


class HashingEmbedder:
    """Deterministic bag-of-words hashing embedder. For tests only."""

    def __init__(self, dim: int = 256) -> None:
        self.name = f"hashing-{dim}"
        self.dim = dim
        self.calls = 0

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in _TOKEN.findall(text.lower()):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        return v

    def embed_docs(self, texts: list[str]) -> np.ndarray:
        self.calls += len(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _normalize(np.stack([self._vec(t) for t in texts]))

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self.embed_docs(texts)

    def token_embeddings(self, texts: list[str]) -> list[np.ndarray]:
        self.calls += len(texts)
        out = []
        for t in texts:
            toks = _TOKEN.findall(t.lower()) or [""]
            out.append(_normalize(np.stack([self._vec(tok) for tok in toks])))
        return out


class CachedEmbedder:
    """Wraps an embedder with the on-disk cache. Query vectors are not cached by default
    so query latency numbers stay honest."""

    def __init__(self, inner: Embedder, cache: EmbeddingCache, cache_queries: bool = False) -> None:
        self.inner = inner
        self.cache = cache
        self.name = inner.name
        self.cache_queries = cache_queries

    @property
    def dim(self) -> int:
        return int(getattr(self.inner, "dim", 0))

    def _cached(self, kind: str, texts: list[str], fn) -> list[np.ndarray]:
        keys = [content_key(self.name, kind, t) for t in texts]
        found = self.cache.get_many(keys)
        todo = [i for i, k in enumerate(keys) if k not in found]
        if todo:
            fresh = fn([texts[i] for i in todo])
            new = {keys[i]: np.asarray(fresh[j]) for j, i in enumerate(todo)}
            self.cache.put_many(new)
            found.update(new)
        return [found[k] for k in keys]

    def embed_docs(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack(self._cached("doc", texts, self.inner.embed_docs))

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        if not self.cache_queries:
            return self.inner.embed_queries(texts)
        return np.stack(self._cached("query", texts, self.inner.embed_queries))

    def token_embeddings(self, texts: list[str]) -> list[np.ndarray]:
        return self._cached("tok", texts, self.inner.token_embeddings)
