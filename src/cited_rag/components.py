"""Wire up models, cache and index for a config. Models are loaded lazily and shared."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

from .cache import EmbeddingCache
from .chunking import HFTokenizer
from .config import Config
from .embed import CachedEmbedder, SentenceTransformerEmbedder
from .index import Index, build_index
from .rerank import CrossEncoderReranker, MaxSimReranker, Reranker
from .retriever import Retriever


@dataclass
class Paths:
    data: Path = Path("data")
    index_root: Path = Path("index")
    cache: Path = field(
        default_factory=lambda: Path(os.environ.get("CITED_RAG_CACHE", ".cache/embeddings.sqlite"))
    )


@cache
def _st_embedder(name: str) -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(name)


@cache
def _tokenizer(name: str) -> HFTokenizer:
    return HFTokenizer(name)


@cache
def _cross_encoder(name: str) -> CrossEncoderReranker:
    return CrossEncoderReranker(name)


_caches: dict[Path, EmbeddingCache] = {}


def embedder_for(cfg: Config, paths: Paths) -> CachedEmbedder:
    if paths.cache not in _caches:
        _caches[paths.cache] = EmbeddingCache(paths.cache)
    return CachedEmbedder(_st_embedder(cfg.embed_model), _caches[paths.cache])


def get_index(cfg: Config, paths: Paths, rebuild: bool = False) -> tuple[Index, dict]:
    out = paths.index_root / cfg.index_key
    if out.exists() and not rebuild:
        return Index.load(out), {}
    index, timings = build_index(
        paths.data, cfg, embedder_for(cfg, paths), _tokenizer(cfg.embed_model)
    )
    index.save(out)
    return index, timings


def reranker_for(cfg: Config, paths: Paths) -> Reranker | None:
    if cfg.rerank == "cross-encoder":
        return _cross_encoder(cfg.rerank_model)
    if cfg.rerank == "maxsim":
        return MaxSimReranker(embedder_for(cfg, paths))
    return None


def retriever_for(cfg: Config, paths: Paths, llm=None, web=None) -> Retriever:
    index, _ = get_index(cfg, paths)
    if cfg.web_fallback and web is None:
        from .websearch import get_provider

        web = get_provider(cfg.web_provider)
    return Retriever(index, cfg, embedder_for(cfg, paths), reranker_for(cfg, paths), llm, web)
