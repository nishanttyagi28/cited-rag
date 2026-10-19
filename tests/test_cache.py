import numpy as np

from cited_rag.cache import EmbeddingCache, content_key
from cited_rag.embed import CachedEmbedder, HashingEmbedder


def test_content_key_depends_on_all_parts():
    base = content_key("m", "doc", "hello")
    assert base == content_key("m", "doc", "hello")
    assert base != content_key("m2", "doc", "hello")
    assert base != content_key("m", "query", "hello")
    assert base != content_key("m", "doc", "hello!")


def test_cache_roundtrip_and_persistence(tmp_path):
    path = tmp_path / "c.sqlite"
    c = EmbeddingCache(path)
    c.put_many({"a": np.arange(4, dtype=np.float32), "b": np.ones((2, 3), dtype=np.float32)})
    c.close()
    c2 = EmbeddingCache(path)
    got = c2.get_many(["a", "b", "zzz"])
    assert np.array_equal(got["a"], np.arange(4))
    assert got["b"].shape == (2, 3)
    assert (c2.hits, c2.misses) == (2, 1)
    assert len(c2) == 2


def test_cached_embedder_only_computes_misses(tmp_path):
    inner = HashingEmbedder(64)
    emb = CachedEmbedder(inner, EmbeddingCache(tmp_path / "c.sqlite"))
    first = emb.embed_docs(["alpha beta", "gamma"])
    assert inner.calls == 2
    second = emb.embed_docs(["gamma", "alpha beta", "delta"])
    assert inner.calls == 3  # only "delta" was new
    assert np.allclose(first[0], second[1])
    toks = emb.token_embeddings(["one two three"])
    assert toks[0].shape == (3, 64)
    emb.token_embeddings(["one two three"])
    assert inner.calls == 4
