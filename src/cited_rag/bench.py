"""Embedding cache latency measurement: cold (empty cache) vs warm (all hits)."""

from __future__ import annotations

import shutil
import statistics
import time
from pathlib import Path

from .cache import EmbeddingCache
from .components import Paths, _st_embedder, _tokenizer
from .config import Config
from .embed import CachedEmbedder
from .index import build_index


def measure_cache(cfg: Config, paths: Paths, tmp: Path, repeats: int = 3) -> dict:
    inner = _st_embedder(cfg.embed_model)
    tok = _tokenizer(cfg.embed_model)
    inner.embed_docs(["warm up the model so load time is not counted"])
    cold, warm = [], []
    n_chunks = 0
    for r in range(repeats):
        if tmp.exists():
            shutil.rmtree(tmp)
        cache = EmbeddingCache(tmp / f"cache-{r}.sqlite")
        emb = CachedEmbedder(inner, cache)
        index, t_cold = build_index(paths.data, cfg, emb, tok)
        _, t_warm = build_index(paths.data, cfg, emb, tok)
        n_chunks = len(index.chunks)
        cold.append(t_cold)
        warm.append(t_warm)
        cache.close()
    shutil.rmtree(tmp, ignore_errors=True)

    def med(rows, key):
        return statistics.median(row[key] for row in rows)

    ec, ew = med(cold, "embed_s"), med(warm, "embed_s")
    tc, tw = med(cold, "total_s"), med(warm, "total_s")
    return {
        "config": cfg.index_key,
        "chunks": n_chunks,
        "repeats": repeats,
        "device": "cpu",
        "embed_cold_s": round(ec, 3),
        "embed_warm_s": round(ew, 3),
        "embed_reduction_pct": round(100 * (1 - ew / ec), 1),
        "ingest_cold_s": round(tc, 3),
        "ingest_warm_s": round(tw, 3),
        "ingest_reduction_pct": round(100 * (1 - tw / tc), 1),
        "raw": {"cold": cold, "warm": warm},
        "measured_at": time.strftime("%Y-%m-%d"),
    }
