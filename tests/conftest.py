import pytest

from cited_rag.chunking import WhitespaceTokenizer, chunk_pages
from cited_rag.embed import HashingEmbedder
from cited_rag.index import Index
from cited_rag.query import mine_acronyms
from cited_rag.types import Page

PAGES = [
    Page("sec", 1, "The Secure Software Development Framework (SSDF) lists practices.", "Intro"),
    Page("sec", 2, "Produce a software bill of materials SBOM for every release.", "Protect"),
    Page("sec", 3, "Use fuzz testing to find input handling issues before release.", "Produce"),
    Page("ai", 4, "Confabulation is confidently stated but false content from models.", "Risks"),
    Page("ai", 5, "Risk tolerance is readiness to bear risk to achieve objectives.", "Framing"),
    Page("ai", 6, "Trustworthy AI is valid and reliable, safe, secure and resilient.", "Trust"),
]
META = {"sec": {"year": 2022, "topic": "security"}, "ai": {"year": 2024, "topic": "ai"}}


@pytest.fixture
def embedder():
    return HashingEmbedder(512)


@pytest.fixture
def tiny_index(embedder):
    chunks, parents = [], []
    for doc in ("sec", "ai"):
        pages = [p for p in PAGES if p.doc_id == doc]
        c, p = chunk_pages(pages, WhitespaceTokenizer(), 12, 2, 0, META[doc])
        chunks += c
        parents += p
    dense = embedder.embed_docs([c.text for c in chunks])
    docs = [{"doc_id": d, "title": d, **META[d]} for d in META]
    gl = mine_acronyms(" ".join(p.text for p in PAGES))
    return Index(chunks, {p.parent_id: p for p in parents}, dense, gl, docs, "test")
