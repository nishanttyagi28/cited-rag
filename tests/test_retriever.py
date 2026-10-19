import pytest

from cited_rag.config import Config
from cited_rag.query import expand, is_ambiguous, mine_acronyms
from cited_rag.retriever import Retriever
from cited_rag.websearch import FakeWebSearch, WebResult


class FakeReranker:
    name = "fake"

    def score(self, query, texts):
        import numpy as np

        q = set(query.lower().split())
        return np.array([len(q & set(t.lower().split())) / (len(q) or 1) for t in texts])


def cfg(**kw):
    base = dict(rerank="none", candidates=10, top_k=3, query_rewrite=False)
    base.update(kw)
    return Config(**base)


def test_mine_acronyms_and_expand():
    g = mine_acronyms(
        "the Secure Software Development Framework (SSDF) and Test, Evaluation, "
        "Verification, and Validation (TEVV) processes"
    )
    assert g["SSDF"] == "Secure Software Development Framework"
    assert g["TEVV"].startswith("Test")
    assert expand("what is SSDF", g) == ["what is SSDF (Secure Software Development Framework)"]
    assert expand("secure software development framework goals", g)[0].endswith("SSDF")
    assert expand("nothing here", g) == []


def test_is_ambiguous():
    assert is_ambiguous("what about it?")
    assert is_ambiguous("what does the framework say about tiers")
    assert not is_ambiguous("What are the four CSF Tiers?")
    assert not is_ambiguous("How does NIST define confabulation in generative models?")


def test_hybrid_retrieval_finds_keyword_chunk(tiny_index, embedder):
    r = Retriever(tiny_index, cfg(), embedder)
    res = r.retrieve("software bill of materials")
    assert res.hits[0].chunk.page_start == 2
    assert res.hits[0].chunk.cite() == "sec, p. 2"


def test_filters_scope_results(tiny_index, embedder):
    r = Retriever(tiny_index, cfg(), embedder)
    res = r.retrieve("risk", filters={"doc": "ai"})
    assert res.hits and all(h.chunk.doc_id == "ai" for h in res.hits)
    res = r.retrieve("risk", filters={"year": {"lte": 2022}})
    assert all(h.chunk.doc_id == "sec" for h in res.hits)


def test_rerank_sets_confidence_and_refusal(tiny_index, embedder):
    r = Retriever(
        tiny_index,
        cfg(rerank="cross-encoder", min_confidence=0.5),
        embedder,
        reranker=FakeReranker(),
    )
    good = r.retrieve("fuzz testing")
    assert good.confidence == pytest.approx(1.0) and not good.refused
    bad = r.retrieve("best pizza in naples")
    assert bad.low_confidence and bad.refused and not bad.used_web


def test_rerank_requires_reranker(tiny_index, embedder):
    with pytest.raises(ValueError):
        Retriever(tiny_index, cfg(rerank="cross-encoder"), embedder)


def test_web_fallback_only_on_low_confidence(tiny_index, embedder):
    web = FakeWebSearch([WebResult("Naples", "https://example.org/pizza", "Pizza info")])
    r = Retriever(
        tiny_index,
        cfg(rerank="cross-encoder", min_confidence=0.5, web_fallback=True),
        embedder,
        reranker=FakeReranker(),
        web=web,
    )
    hi = r.retrieve("fuzz testing")
    assert not hi.used_web and web.queries == []
    lo = r.retrieve("best pizza in naples")
    assert lo.used_web and not lo.refused
    assert lo.hits[0].source == "web" and lo.hits[0].url == "https://example.org/pizza"
    assert web.queries == ["best pizza in naples"]


def test_web_fallback_off_by_default(tiny_index, embedder):
    assert Config().web_fallback is False
    with pytest.raises(ValueError):
        Retriever(tiny_index, cfg(web_fallback=True), embedder)


class RewriteLLM:
    def __init__(self):
        self.calls = []

    def complete(self, system, user, max_tokens=512):
        self.calls.append(user)
        if "vague" in system.lower():
            return "What is confabulation in generative AI?"
        return "confabulation definition\nfalse content models"


def test_rewrite_only_for_ambiguous_queries(tiny_index, embedder):
    llm = RewriteLLM()
    r = Retriever(tiny_index, cfg(query_rewrite=True, llm_expansion=True), embedder, llm=llm)
    res = r.retrieve("what is it?")
    assert res.queries[0] == "What is confabulation in generative AI?"
    assert "confabulation definition" in res.queries
    assert res.hits[0].chunk.doc_id == "ai"
    n = len(llm.calls)
    r.retrieve("Which practice covers fuzz testing in SSDF?")
    assert len(llm.calls) == n + 1  # expansion only, no rewrite


def test_acronym_mining_handles_embedded_caps():
    g = mine_acronyms("profile for Generative AI (GAI) and the Cybersecurity Framework (CSF)")
    assert g == {"GAI": "Generative AI", "CSF": "Cybersecurity Framework"}
    assert expand("AI risks", {"AI": "Artificial intelligence"}) == []


def test_expand_does_not_double_expand():
    g = {"NIST": "National Institute of Standards and Technology"}
    assert expand("NIST (National Institute of Standards and Technology) guidance", g) == []
