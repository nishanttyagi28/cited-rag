"""The golden set is checked against the PDFs: every key fact must appear on the stated pages."""

import json
import re
from functools import cache
from pathlib import Path

import pytest

from cited_rag.eval.golden import load_golden
from cited_rag.filters import matches
from cited_rag.pdf import load_pdf
from cited_rag.types import Chunk

ROOT = Path(__file__).resolve().parents[1]
CASES = load_golden(ROOT / "eval" / "golden.jsonl")
MANIFEST = {d["doc_id"]: d for d in json.loads((ROOT / "data" / "manifest.json").read_text())}


@cache
def pages_of(doc_id: str):
    return {p.page: p for p in load_pdf(ROOT / "data" / MANIFEST[doc_id]["file"], doc_id)}


def page_text(doc_id: str) -> dict[int, str]:
    return {n: re.sub(r"\s+", " ", p.text.lower()) for n, p in pages_of(doc_id).items()}


def test_size_and_mix():
    assert len(CASES) >= 50
    kinds = {c.kind for c in CASES}
    assert {"factual", "adversarial", "filtered", "ambiguous", "acronym"} <= kinds
    assert sum(not c.answerable for c in CASES) >= 10


@pytest.mark.parametrize("case", [c for c in CASES if c.answerable], ids=lambda c: c.id)
def test_key_facts_are_on_expected_pages(case):
    text = " ".join(page_text(case.doc)[p] for p in case.pages)
    for fact in case.key_facts:
        alts = [fact] if isinstance(fact, str) else fact
        assert any(a.lower() in text for a in alts), f"{case.id}: {fact!r} not on {case.pages}"


@pytest.mark.parametrize("case", [c for c in CASES if c.filters], ids=lambda c: c.id)
def test_filters_keep_the_expected_document(case):
    d = MANIFEST[case.doc]
    page = pages_of(case.doc)[case.pages[0]]
    meta = {"year": d["year"], "topic": d["topic"]}
    chunk = Chunk("x", case.doc, "", page.page, page.page, page.section, "x", meta)
    assert matches(chunk, case.filters)
