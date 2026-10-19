"""End-to-end checks with the real CPU models. Run with: pytest -m slow"""

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def csf_retriever(tmp_path_factory):
    from cited_rag.cache import EmbeddingCache
    from cited_rag.chunking import HFTokenizer
    from cited_rag.config import load_config
    from cited_rag.embed import CachedEmbedder, SentenceTransformerEmbedder
    from cited_rag.index import build_index
    from cited_rag.rerank import CrossEncoderReranker
    from cited_rag.retriever import Retriever

    data = tmp_path_factory.mktemp("data")
    manifest = [
        d
        for d in json.loads((ROOT / "data/manifest.json").read_text())
        if d["doc_id"] == "nist-csf-2.0"
    ]
    (data / "manifest.json").write_text(json.dumps(manifest))
    (data / manifest[0]["file"]).write_bytes((ROOT / "data" / manifest[0]["file"]).read_bytes())
    cfg = load_config("full")
    emb = CachedEmbedder(
        SentenceTransformerEmbedder(cfg.embed_model), EmbeddingCache(data / "c.sqlite")
    )
    index, _ = build_index(data, cfg, emb, HFTokenizer(cfg.embed_model))
    return Retriever(index, cfg, emb, CrossEncoderReranker(cfg.rerank_model))


def test_real_pipeline_finds_tiers_page(csf_retriever):
    res = csf_retriever.retrieve("What are the four CSF Tiers?")
    pages = {p for h in res.hits[:3] for p in h.chunk.pages}
    assert 12 in pages  # section 3.2 "CSF Tiers"
    assert not res.refused


def test_real_pipeline_refuses_out_of_scope(csf_retriever):
    assert csf_retriever.retrieve("Who won the 2022 FIFA World Cup?").refused
