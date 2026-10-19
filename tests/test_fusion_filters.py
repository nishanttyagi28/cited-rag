import numpy as np
import pytest

from cited_rag.embed import HashingEmbedder
from cited_rag.filters import mask, matches, parse_cli_filters
from cited_rag.fusion import rrf, top_indices
from cited_rag.rerank import MaxSimReranker, maxsim
from cited_rag.sparse import BM25Index, tokenize
from cited_rag.types import Chunk


def test_rrf_scores_match_formula():
    fused = dict(rrf([[1, 2, 3], [3, 1]], k=60))
    assert fused[1] == pytest.approx(1 / 61 + 1 / 62)
    assert fused[3] == pytest.approx(1 / 63 + 1 / 61)
    assert fused[2] == pytest.approx(1 / 62)


def test_rrf_rewards_agreement():
    order = [i for i, _ in rrf([[5, 1, 2], [1, 7, 8], [1, 9]])]
    assert order[0] == 1


def test_top_indices_respects_mask_and_order():
    s = np.array([0.1, 0.9, 0.5, 0.7])
    assert top_indices(s, 2) == [1, 3]
    m = np.array([True, False, True, False])
    assert top_indices(s, 5, m) == [2, 0]
    assert top_indices(s, 3, np.zeros(4, dtype=bool)) == []


def test_bm25_prefers_keyword_match():
    idx = BM25Index(["the cat sat", "software bill of materials sbom", "dogs bark"])
    assert int(np.argmax(idx.scores("what is an SBOM"))) == 1
    assert tokenize("PW.1.1 and GV.SC") == ["pw.1.1", "gv.sc"]
    assert idx.scores("the of").sum() == 0


def _chunk(doc, section, year, p0, p1, topic="ai"):
    return Chunk(f"{doc}:0", doc, "x", p0, p1, section, "p", {"year": year, "topic": topic})


def test_filters():
    c = _chunk("nist-csf-2.0", "3.2. CSF Tiers", 2024, 12, 13, "security")
    assert matches(c, None)
    assert matches(c, {"doc": "nist-csf-2.0"})
    assert matches(c, {"doc": ["a", "nist-csf-2.0"]})
    assert not matches(c, {"doc": "other"})
    assert matches(c, {"section": "tiers"})
    assert matches(c, {"year": {"gte": 2023}})
    assert not matches(c, {"year": {"lte": 2022}})
    assert matches(c, {"year": 2024, "topic": "security"})
    assert matches(c, {"pages": [13, 20]})
    assert not matches(c, {"pages": [1, 11]})
    with pytest.raises(ValueError):
        matches(c, {"author": "x"})
    other = _chunk("b", "s", 2020, 1, 1)
    assert mask([c, other], {"year": {"gte": 2023}}).tolist() == [True, False]
    assert mask([c], {}) is None


def test_parse_cli_filters():
    f = parse_cli_filters(["doc=a,b", "year>=2023", "pages=10-12", "section=Tiers"])
    assert f == {"doc": ["a", "b"], "year": {"gte": 2023}, "pages": [10, 12], "section": "Tiers"}


def test_maxsim_basic():
    q = np.array([[1.0, 0.0], [0.0, 1.0]])
    d = np.array([[1.0, 0.0], [0.6, 0.8]])
    assert maxsim(q, d) == pytest.approx((1.0 + 0.8) / 2)
    assert maxsim(np.zeros((0, 2)), d) == 0.0


def test_maxsim_reranker_prefers_overlap():
    r = MaxSimReranker(HashingEmbedder(512))
    s = r.score("secure software development", ["cooking pasta at home", "secure software"])
    assert s[1] > s[0]
