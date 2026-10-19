"""Pipeline configuration and named presets."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Config:
    # chunking
    chunk_size: int = 500  # tokens (embedding model tokenizer)
    chunk_overlap: int = 50
    parent_size: int = 1500  # tokens; 0 disables parent windows
    # models
    embed_model: str = "BAAI/bge-small-en-v1.5"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # retrieval
    dense: bool = True
    sparse: bool = True
    candidates: int = 50  # per retriever, before fusion
    rrf_k: int = 60
    rerank: str = "cross-encoder"  # "none" | "cross-encoder" | "maxsim"
    rerank_candidates: int = 30
    top_k: int = 10
    query_expansion: bool = True
    query_rewrite: bool = True  # only acts when an LLM is available
    llm_expansion: bool = False  # extra LLM paraphrases; costs one call per query
    parent_context: bool = True
    # refusal / fallback
    # refusal threshold in the units of the final scorer (cosine, CE probability, MaxSim).
    # Preset values come from `eval/calibrate.py` on half of the golden set, see README.
    min_confidence: float = 0.0
    web_fallback: bool = False
    web_provider: str = "fake"
    # generation
    context_chunks: int = 5  # max context blocks
    context_budget_chars: int = 8000  # total, same for every preset so comparisons are fair
    max_block_chars: int = 3200
    grounding_threshold: float = 0.6
    strict_grounding: bool = False  # drop unsupported sentences instead of only flagging

    def with_(self, **kw: Any) -> Config:
        return replace(self, **kw)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def index_key(self) -> str:
        """Ingest-time settings; two configs with the same key share an index."""
        model = self.embed_model.split("/")[-1]
        return f"{model}-c{self.chunk_size}-o{self.chunk_overlap}-p{self.parent_size}"


PRESETS: dict[str, Config] = {
    # dense only, fixed chunks without overlap, no rerank, no expansion, chunk text as context
    "naive": Config(
        chunk_overlap=0,
        parent_size=0,
        sparse=False,
        rerank="none",
        query_expansion=False,
        query_rewrite=False,
        parent_context=False,
        min_confidence=0.585,
    ),
    "hybrid": Config(rerank="none", min_confidence=0.651),
    "full": Config(min_confidence=0.065),
    "full-maxsim": Config(rerank="maxsim", min_confidence=0.642),
}


def load_config(name_or_path: str, **overrides: Any) -> Config:
    if name_or_path in PRESETS:
        cfg = PRESETS[name_or_path]
    else:
        cfg = Config(**json.loads(Path(name_or_path).read_text()))
    return cfg.with_(**overrides) if overrides else cfg
