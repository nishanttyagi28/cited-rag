"""The retrieval pipeline: rewrite -> expand -> dense + BM25 -> RRF -> rerank -> parents."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from . import filters as flt
from .config import Config
from .embed import Embedder
from .fusion import rrf, top_indices
from .index import Index
from .query import TextLLM, expand, is_ambiguous, llm_expand, rewrite
from .rerank import Reranker
from .types import Chunk, Hit
from .websearch import WebSearchProvider


@dataclass
class RetrievalResult:
    query: str
    queries: list[str]
    hits: list[Hit]
    confidence: float
    low_confidence: bool
    used_web: bool = False
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def refused(self) -> bool:
        return self.low_confidence and not self.used_web


class Retriever:
    def __init__(
        self,
        index: Index,
        cfg: Config,
        embedder: Embedder,
        reranker: Reranker | None = None,
        llm: TextLLM | None = None,
        web: WebSearchProvider | None = None,
    ) -> None:
        if cfg.rerank != "none" and reranker is None:
            raise ValueError(f"config wants rerank={cfg.rerank!r} but no reranker given")
        if cfg.web_fallback and web is None:
            raise ValueError("web_fallback is on but no provider given")
        self.index, self.cfg, self.embedder = index, cfg, embedder
        self.reranker, self.llm, self.web = reranker, llm, web

    def _queries(self, query: str) -> tuple[str, list[str]]:
        q = query
        if self.cfg.query_rewrite and self.llm is not None and is_ambiguous(query):
            q = rewrite(query, self.llm, [d["title"] for d in self.index.docs])
        queries = [q]
        if self.cfg.query_expansion:
            queries += expand(q, self.index.glossary)
            if self.llm is not None:
                queries += llm_expand(q, self.llm)
        return q, list(dict.fromkeys(queries))

    def retrieve(self, query: str, filters: dict[str, Any] | None = None) -> RetrievalResult:
        cfg, t = self.cfg, {}
        t0 = time.perf_counter()
        main_q, queries = self._queries(query)
        t["query_s"] = time.perf_counter() - t0
        m = flt.mask(self.index.chunks, filters)

        t1 = time.perf_counter()
        rankings: list[list[int]] = []
        dense_best: dict[int, float] = {}
        if cfg.dense:
            qv = self.embedder.embed_queries(queries)
            for row in qv:
                sims = self.index.dense @ row
                ranked = top_indices(sims, cfg.candidates, m)
                rankings.append(ranked)
                for i in ranked:
                    dense_best[i] = max(dense_best.get(i, -1.0), float(sims[i]))
        if cfg.sparse:
            for q in queries:
                rankings.append(top_indices(self.index.bm25.scores(q), cfg.candidates, m))
        t["search_s"] = time.perf_counter() - t1

        if len(rankings) == 1 and cfg.dense:
            fused = [(i, dense_best[i]) for i in rankings[0]]
        else:
            fused = rrf(rankings, k=cfg.rrf_k)

        t2 = time.perf_counter()
        scored: list[tuple[int, float, dict[str, float]]] = []
        if cfg.rerank != "none" and self.reranker is not None and fused:
            pool = [i for i, _ in fused[: cfg.rerank_candidates]]
            rs = self.reranker.score(main_q, [self.index.chunks[i].text for i in pool])
            fused_score = dict(fused)
            for i, s in zip(pool, rs, strict=True):
                scored.append((i, float(s), {"fused": fused_score[i], "rerank": float(s)}))
            scored.sort(key=lambda x: -x[1])
            confidence = scored[0][1]
        else:
            for i, s in fused:
                scored.append((i, float(s), {"fused": float(s), "dense": dense_best.get(i, 0.0)}))
            confidence = max((dense_best.get(i, 0.0) for i, _ in fused[:1]), default=0.0)
        t["rerank_s"] = time.perf_counter() - t2

        hits = [self._hit(self.index.chunks[i], s, sc) for i, s, sc in scored[: cfg.top_k]]
        low = (not hits) or confidence < cfg.min_confidence
        used_web = False
        if low and cfg.web_fallback and self.web is not None:
            hits = self._web_hits(main_q) + hits
            used_web = bool(hits)
        t["total_s"] = time.perf_counter() - t0
        return RetrievalResult(query, queries, hits, float(confidence), low, used_web, t)

    def _hit(self, chunk: Chunk, score: float, scores: dict[str, float]) -> Hit:
        ctx = self.index.parent_text(chunk) if self.cfg.parent_context else chunk.text
        return Hit(chunk=chunk, score=score, scores=scores, context=ctx)

    def _web_hits(self, query: str) -> list[Hit]:
        assert self.web is not None
        out = []
        for n, r in enumerate(self.web.search(query, k=3)):
            c = Chunk(f"web:{n}", "web", r.snippet, 0, 0, r.title, f"web:{n}", {"url": r.url})
            out.append(Hit(c, 0.0, {}, context=r.snippet, source="web", url=r.url))
        return out
