"""Ingest PDFs into an on-disk index (chunks, parents, dense vectors, acronym glossary)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .chunking import Parent, Tokenizer, chunk_pages
from .config import Config
from .embed import Embedder
from .pdf import load_pdf
from .query import mine_acronyms
from .sparse import BM25Index
from .types import Chunk


@dataclass
class Index:
    chunks: list[Chunk]
    parents: dict[str, Parent]
    dense: np.ndarray
    glossary: dict[str, str]
    docs: list[dict]
    key: str
    _bm25: BM25Index | None = field(default=None, repr=False)

    @property
    def bm25(self) -> BM25Index:
        if self._bm25 is None:
            self._bm25 = BM25Index([c.text for c in self.chunks])
        return self._bm25

    def parent_text(self, chunk: Chunk) -> str:
        p = self.parents.get(chunk.parent_id)
        return p.text if p else chunk.text

    def save(self, out: Path) -> None:
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "chunks.jsonl", "w") as f:
            for c in self.chunks:
                f.write(json.dumps(c.to_dict()) + "\n")
        with open(out / "parents.jsonl", "w") as f:
            for p in self.parents.values():
                f.write(json.dumps(p.to_dict()) + "\n")
        np.save(out / "dense.npy", self.dense)
        meta = {"key": self.key, "docs": self.docs, "glossary": self.glossary}
        (out / "meta.json").write_text(json.dumps(meta, indent=1))

    @classmethod
    def load(cls, path: Path) -> Index:
        meta = json.loads((path / "meta.json").read_text())
        lines = (path / "chunks.jsonl").read_text().splitlines()
        chunks = [Chunk.from_dict(json.loads(line)) for line in lines]
        parents = {}
        for line in (path / "parents.jsonl").read_text().splitlines():
            d = json.loads(line)
            d["page_breaks"] = [tuple(b) for b in d.get("page_breaks", [])]
            p = Parent(**d)
            parents[p.parent_id] = p
        dense = np.load(path / "dense.npy")
        return cls(chunks, parents, dense, meta["glossary"], meta["docs"], meta["key"])


def load_manifest(data_dir: Path) -> list[dict]:
    return json.loads((data_dir / "manifest.json").read_text())


def build_index(
    data_dir: Path, cfg: Config, embedder: Embedder, tokenizer: Tokenizer
) -> tuple[Index, dict[str, float]]:
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    docs = load_manifest(data_dir)
    chunks: list[Chunk] = []
    parents: list[Parent] = []
    full_text: list[str] = []
    for d in docs:
        pages = load_pdf(data_dir / d["file"], d["doc_id"])
        full_text.extend(p.text for p in pages)
        meta = {"year": d["year"], "topic": d.get("topic", ""), "title": d["title"]}
        c, p = chunk_pages(
            pages, tokenizer, cfg.chunk_size, cfg.chunk_overlap, cfg.parent_size, meta
        )
        chunks.extend(c)
        parents.extend(p)
    timings["parse_chunk_s"] = time.perf_counter() - t0
    t1 = time.perf_counter()
    dense = embedder.embed_docs([c.text for c in chunks])
    timings["embed_s"] = time.perf_counter() - t1
    glossary = mine_acronyms("\n".join(full_text))
    index = Index(chunks, {p.parent_id: p for p in parents}, dense, glossary, docs, cfg.index_key)
    timings["total_s"] = time.perf_counter() - t0
    return index, timings
