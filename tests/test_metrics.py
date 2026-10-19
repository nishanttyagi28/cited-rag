import math

import pytest

from cited_rag.config import Config
from cited_rag.eval.golden import GoldenCase, load_golden
from cited_rag.eval.metrics import hit_at_k, is_relevant, key_fact_recall, mrr, ndcg_at_k
from cited_rag.eval.runner import eval_retrieval, refusal_summary
from cited_rag.retriever import Retriever


def test_hit_and_mrr():
    rels = [False, False, True, False]
    assert hit_at_k(rels, 2) == 0.0
    assert hit_at_k(rels, 3) == 1.0
    assert mrr(rels) == pytest.approx(1 / 3)
    assert mrr(rels, 2) == 0.0
    assert mrr([False] * 3) == 0.0


def test_ndcg():
    assert ndcg_at_k([True, False], 10) == 1.0
    assert ndcg_at_k([False, True], 10) == pytest.approx(1 / math.log2(3))
    # two relevant exist, only one found at rank 1
    assert ndcg_at_k([True, False], 10, n_relevant=2) == pytest.approx(1 / (1 + 1 / math.log2(3)))
    assert ndcg_at_k([False], 10, n_relevant=0) == 0.0


def test_is_relevant():
    assert is_relevant("a", [3, 4], "a", [4])
    assert not is_relevant("a", [3], "a", [4])
    assert not is_relevant("b", [4], "a", [4])


def test_key_fact_recall_with_alternatives():
    ans = "The tiers are Partial and Adaptive."
    assert key_fact_recall(ans, ["partial", ["adaptive", "tier 4"], "repeatable"]) == 2 / 3
    assert key_fact_recall(ans, []) == 1.0


def test_refusal_summary():
    rows = [
        {"answerable": False, "refused": True},
        {"answerable": False, "refused": False},
        {"answerable": True, "refused": False},
        {"answerable": True, "refused": True},
    ]
    s = refusal_summary(rows)
    assert s["refusal_accuracy"] == 0.5
    assert s["out_of_scope_refused"] == 0.5
    assert s["answerable_wrongly_refused"] == 0.5


def test_eval_retrieval_end_to_end(tiny_index, embedder):
    r = Retriever(tiny_index, Config(rerank="none", query_rewrite=False), embedder)
    cases = [
        GoldenCase("q1", "software bill of materials", "sec", [2]),
        GoldenCase("q2", "what is confabulation", "ai", [4]),
        GoldenCase("q3", "pizza", None, kind="adversarial"),
    ]
    out = eval_retrieval(r, cases)
    s = out["summary"]
    assert s["n_answerable"] == 2 and s["hit@10"] == 1.0 and s["mrr@10"] > 0.5
    assert out["rows"][0]["first_relevant_rank"] == 1


def test_load_golden_rejects_duplicates(tmp_path):
    p = tmp_path / "g.jsonl"
    p.write_text(
        '{"id": "a", "question": "q", "doc": null}\n{"id": "a", "question": "q", "doc": null}\n'
    )
    with pytest.raises(ValueError):
        load_golden(p)


def test_calibrate_threshold_separates_scores():
    from cited_rag.eval.calibrate import accuracy, best_threshold, calibrate

    rows = [
        {"confidence": c, "answerable": a}
        for c, a in [
            (0.9, True),
            (0.8, True),
            (0.1, False),
            (0.2, False),
            (0.7, True),
            (0.6, True),
            (0.05, False),
            (0.15, False),
        ]
    ]
    t = best_threshold(rows)
    assert 0.2 < t < 0.6
    assert accuracy(rows, t) == 1.0
    out = calibrate(rows)
    assert out["held_out_accuracy"] == 1.0 and out["n_held_out"] == 4


def test_key_fact_matching_ignores_dash_style():
    assert key_fact_recall("Risk\u2011Informed (Tier 2)", ["risk informed"]) == 1.0
    assert key_fact_recall("non sector specific", ["non-sector-specific"]) == 1.0


def test_key_fact_matching_ignores_soft_hyphens():
    assert (
        key_fact_recall("explainable and interpret\u00adable", ["explainable and interpretable"])
        == 1.0
    )


def test_eval_answers_rescores_cached_rows(tmp_path, tiny_index, embedder):
    import json as _json

    from cited_rag.answer import RAG
    from cited_rag.eval.runner import eval_answers
    from cited_rag.llm import FakeLLM

    case = GoldenCase("c1", "fuzz testing", "sec", [3], ["fuzz testing"])
    cache = tmp_path / "a.jsonl"
    row = {
        "id": "c1",
        "kind": "factual",
        "answerable": True,
        "refused": False,
        "answer": "Use fuzz testing [1].",
        "grounding": None,
        "latency_s": 0.1,
        "citations": [{"n": 1, "doc_id": "sec", "pages": [3, 3], "url": ""}],
        "key_fact_recall": 0.0,
    }
    cache.write_text(_json.dumps(row) + "\n")
    llm = FakeLLM("should not be called")
    rag = RAG(Retriever(tiny_index, Config(rerank="none"), embedder), llm)
    out = eval_answers(rag, [case], cache)
    assert llm.calls == []
    r = out["rows"][0]
    assert r["key_fact_recall"] == 1.0 and r["citation_precision"] == 1.0
