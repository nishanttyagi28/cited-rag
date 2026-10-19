from cited_rag.answer import RAG, REFUSAL_TEXT, build_blocks
from cited_rag.config import Config
from cited_rag.llm import FakeLLM, RequestTooLarge
from cited_rag.retriever import Retriever
from cited_rag.websearch import FakeWebSearch, WebResult


class OverlapReranker:
    name = "overlap"

    def score(self, query, texts):
        import numpy as np

        q = set(query.lower().replace("?", "").split())
        return np.array([len(q & set(t.lower().split())) / len(q) for t in texts])


def make(tiny_index, embedder, llm, **kw):
    base = dict(top_k=3, query_rewrite=False, rerank="cross-encoder", min_confidence=0.3)
    base.update(kw)
    cfg = Config(**base)
    web = FakeWebSearch([WebResult("W", "https://example.org/w", "web text")])
    r = Retriever(tiny_index, cfg, embedder, OverlapReranker(), web=web)
    return RAG(r, llm, embedder)


def first_source_answer(system, user):
    return "Use fuzz testing to find input handling issues [1]."


def test_answer_cites_pages(tiny_index, embedder):
    rag = make(tiny_index, embedder, FakeLLM(first_source_answer))
    a = rag.answer("fuzz testing input handling")
    assert not a.refused
    assert [b.label() for b in a.cited_blocks] == ["sec, p. 3"]
    assert a.grounding.fully_grounded
    d = a.to_dict()
    assert d["citations"][0]["pages"] == [3, 3]


def test_refuses_without_llm_call_when_confidence_low(tiny_index, embedder):
    llm = FakeLLM(first_source_answer)
    a = make(tiny_index, embedder, llm).answer("best pizza dough recipe")
    assert a.refused and a.text == REFUSAL_TEXT and a.meta["refusal"] == "retrieval"
    assert llm.calls == []


def test_llm_no_answer_becomes_refusal(tiny_index, embedder):
    a = make(tiny_index, embedder, FakeLLM("NO_ANSWER")).answer("fuzz testing")
    assert a.refused and a.meta["refusal"] == "llm"


def test_unsupported_sentence_flagged_and_strict_mode_drops_it(tiny_index, embedder):
    def reply(s, u):
        return (
            "Use fuzz testing to find input handling issues [1]. "
            "Fuzzing is mandated by federal law for all vendors [1]."
        )

    a = make(tiny_index, embedder, FakeLLM(reply)).answer("fuzz testing input handling")
    assert len(a.grounding.flagged) == 1
    strict = make(tiny_index, embedder, FakeLLM(reply), strict_grounding=True)
    b = strict.answer("fuzz testing input handling")
    assert "federal law" not in b.text and b.meta["dropped_sentences"] == 1


def test_request_too_large_shrinks_context(tiny_index, embedder):
    calls = []

    def reply(s, u):
        calls.append(u.count("\n["))
        if len(calls) == 1:
            raise RequestTooLarge("too big")
        return "Use fuzz testing to find input handling issues [1]."

    class Raising(FakeLLM):
        def complete(self, system, user, max_tokens=512):
            return reply(system, user)

    a = make(tiny_index, embedder, Raising()).answer("fuzz testing input handling")
    assert not a.refused and calls[1] < calls[0] + 1


def test_web_fallback_answer_cites_url(tiny_index, embedder):
    rag = make(tiny_index, embedder, FakeLLM("web text says so [1]."), web_fallback=True)
    a = rag.answer("best pizza dough recipe")
    assert not a.refused
    assert a.cited_blocks[0].url == "https://example.org/w"


def test_blocks_dedupe_parents_and_respect_budget(tiny_index, embedder):
    cfg = Config(context_budget_chars=100, max_block_chars=60, parent_context=False)
    r = Retriever(tiny_index, cfg.with_(rerank="none"), embedder)
    hits = r.retrieve("software").hits
    blocks = build_blocks(hits, tiny_index, cfg)
    assert sum(len(b.text) for b in blocks) <= 100
    assert all(len(b.text) <= 60 for b in blocks)
