import pytest

from cited_rag.grounding import (
    check_grounding,
    citations,
    lexical_recall,
    split_sentences,
)
from cited_rag.llm import FakeLLM, TokenPacer

SRC = {
    1: "Confabulation is the production of confidently stated but erroneous or false content.",
    2: "The CSF Tiers are Partial, Risk Informed, Repeatable, and Adaptive.",
}


def test_split_sentences_keeps_citations_attached():
    s = split_sentences("Tiers are four [2]. Confabulation is false content. [1] Done here [1][2].")
    assert s == ["Tiers are four [2].", "Confabulation is false content [1].", "Done here [1][2]."]
    assert citations(s[2]) == [1, 2]


def test_supported_sentence_passes():
    r = check_grounding(
        "The CSF Tiers are Partial, Risk Informed, Repeatable and Adaptive [2].", SRC
    )
    assert r.fully_grounded and r.supported_rate == 1.0


def test_unsupported_claim_is_flagged():
    ans = (
        "Confabulation is confidently stated but false content [1]. "
        "The CSF requires annual penetration tests by certified auditors [2]."
    )
    r = check_grounding(ans, SRC)
    assert len(r.sentences) == 2
    assert r.sentences[0].supported
    assert not r.sentences[1].supported
    assert r.supported_rate == pytest.approx(0.5)


def test_uncited_and_unknown_citations_are_unsupported():
    r = check_grounding("Confabulation is false content. Tiers are adaptive [9].", SRC)
    assert [s.reason for s in r.sentences] == ["no citation", "unknown citation"]
    assert not r.fully_grounded


def test_grounding_with_embedder(embedder):
    r = check_grounding("Confabulation means false content stated confidently [1].", SRC, embedder)
    assert r.sentences[0].supported


def test_lexical_recall():
    assert lexical_recall("false content", SRC[1]) == 1.0
    assert lexical_recall("pizza oven", SRC[1]) == 0.0


def test_fake_llm_records_calls():
    llm = FakeLLM(lambda s, u: u.upper())
    assert llm.complete("sys", "hi") == "HI"
    assert llm.calls == [("sys", "hi")]


def test_token_pacer_waits_when_window_full():
    now = [0.0]
    slept = []

    def sleep(d):
        slept.append(d)
        now[0] += d

    p = TokenPacer(1000, clock=lambda: now[0], sleep=sleep)
    assert p.wait(600) == 0
    now[0] = 10
    assert p.wait(300) == 0
    waited = p.wait(300)  # 1200 > 1000 -> must wait until first event leaves the window
    assert waited == pytest.approx(50.05)
    assert slept
