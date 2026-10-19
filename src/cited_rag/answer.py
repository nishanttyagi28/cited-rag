"""Answer generation with numbered, page-cited context blocks and a grounding pass."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .config import Config
from .embed import Embedder
from .grounding import GroundingReport, check_grounding, citations
from .index import Index
from .llm import RequestTooLarge
from .query import TextLLM
from .retriever import RetrievalResult, Retriever
from .types import Hit

NO_ANSWER = "NO_ANSWER"
REFUSAL_TEXT = "I could not find this in the indexed documents."

SYSTEM_PROMPT = f"""You answer questions using only the numbered sources provided.
Rules:
- Every sentence in your answer must end with one or more citations like [1] or [2][3],
  pointing at the sources that support it. Use exactly that bracket format, nothing else.
- Use only facts stated in the sources. Do not add outside knowledge.
- Be concise: at most 5 sentences, plain prose, no headings or bullet lists.
- If the sources do not contain the answer, reply with exactly {NO_ANSWER} and nothing else."""


@dataclass
class Block:
    n: int
    doc_id: str
    page_start: int
    page_end: int
    section: str
    text: str
    chunk_ids: list[str]
    url: str = ""

    def label(self) -> str:
        if self.doc_id == "web":
            return f"web: {self.url}"
        pages = (
            f"p. {self.page_start}"
            if self.page_start == self.page_end
            else f"pp. {self.page_start}-{self.page_end}"
        )
        return f"{self.doc_id}, {pages}"


@dataclass
class Answer:
    question: str
    text: str
    refused: bool
    blocks: list[Block]
    cited: list[int]
    grounding: GroundingReport | None
    retrieval: RetrievalResult
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def cited_blocks(self) -> list[Block]:
        by_n = {b.n: b for b in self.blocks}
        return [by_n[n] for n in self.cited if n in by_n]

    def to_dict(self) -> dict[str, Any]:
        g = self.grounding
        return {
            "question": self.question,
            "answer": self.text,
            "refused": self.refused,
            "citations": [
                {"n": b.n, "doc_id": b.doc_id, "pages": [b.page_start, b.page_end], "url": b.url}
                for b in self.cited_blocks
            ],
            "grounding": None
            if g is None
            else {
                "supported_rate": g.supported_rate,
                "flagged": [s.text for s in g.flagged],
                "sentences": len(g.sentences),
            },
            "confidence": self.retrieval.confidence,
            "used_web": self.retrieval.used_web,
            **self.meta,
        }


def _window(text: str, anchor: str, size: int) -> tuple[int, int]:
    """A ``size``-char window of ``text`` centred on where ``anchor`` starts."""
    if len(text) <= size:
        return 0, len(text)
    pos = max(0, text.find(anchor[:200]))
    start = max(0, min(pos - size // 4, len(text) - size))
    return start, start + size


def build_blocks(hits: list[Hit], index: Index | None, cfg: Config) -> list[Block]:
    blocks: list[Block] = []
    seen_parent: dict[str, Block] = {}
    budget = cfg.context_budget_chars
    for h in hits:
        if len(blocks) >= cfg.context_chunks or budget <= 200:
            break
        c = h.chunk
        if h.source == "web":
            text = h.context[: min(cfg.max_block_chars, budget)]
            blocks.append(Block(len(blocks) + 1, "web", 0, 0, c.section, text, [c.chunk_id], h.url))
            budget -= len(text)
            continue
        size = min(cfg.max_block_chars, budget)
        parent = index.parents.get(c.parent_id) if (index and cfg.parent_context) else None
        if parent is not None:
            if c.parent_id in seen_parent:
                seen_parent[c.parent_id].chunk_ids.append(c.chunk_id)
                continue
            lo, hi = _window(parent.text, c.text, size)
            text = parent.text[lo:hi]
            p0, p1 = parent.page_at(lo), parent.page_at(max(lo, hi - 1))
        else:
            text, p0, p1 = c.text[:size], c.page_start, c.page_end
        b = Block(len(blocks) + 1, c.doc_id, p0, p1, c.section, text, [c.chunk_id])
        blocks.append(b)
        if parent is not None:
            seen_parent[c.parent_id] = b
        budget -= len(text)
    return blocks


def format_context(blocks: list[Block]) -> str:
    return "\n\n".join(f"[{b.n}] ({b.label()}; {b.section})\n{b.text}" for b in blocks)


class RAG:
    def __init__(
        self,
        retriever: Retriever,
        llm: TextLLM,
        embedder: Embedder | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.embedder = embedder
        self.cfg = retriever.cfg

    def answer(self, question: str, filters: dict[str, Any] | None = None) -> Answer:
        res = self.retriever.retrieve(question, filters)
        if res.refused or not res.hits:
            return Answer(question, REFUSAL_TEXT, True, [], [], None, res, {"refusal": "retrieval"})
        blocks = build_blocks(res.hits, self.retriever.index, self.cfg)
        text = ""
        while blocks:
            prompt = f"Sources:\n{format_context(blocks)}\n\nQuestion: {question}"
            try:
                text = self.llm.complete(SYSTEM_PROMPT, prompt, max_tokens=900)
                break
            except RequestTooLarge:
                blocks = blocks[:-1]  # shrink and retry
        text = normalize_citations(text)
        if not text or NO_ANSWER in text:
            return Answer(question, REFUSAL_TEXT, True, blocks, [], None, res, {"refusal": "llm"})
        sources = {b.n: b.text for b in blocks}
        report = check_grounding(text, sources, self.embedder, self.cfg.grounding_threshold)
        meta: dict[str, Any] = {}
        if self.cfg.strict_grounding and report.flagged:
            bad = {s.text for s in report.flagged}
            kept = [s for s in report.sentences if s.text not in bad]
            if not kept:
                return Answer(
                    question, REFUSAL_TEXT, True, blocks, [], report, res, {"refusal": "grounding"}
                )
            text = " ".join(f"{s.text} " + "".join(f"[{n}]" for n in s.cited) for s in kept)
            meta["dropped_sentences"] = len(bad)
        cited = sorted({n for n in citations(text) if n in sources})
        return Answer(question, _tidy(text), False, blocks, cited, report, res, meta)


def normalize_citations(text: str) -> str:
    """Map other citation styles the model sometimes uses onto [n]."""
    text = re.sub(r"【\s*(\d+)[^】]*】", r"[\1]", text)  # 【1†L3-L9】
    text = re.sub(r"\[(\d+)†[^\]]*\]", r"[\1]", text)  # [1†L3]
    text = re.sub(
        r"\[(\d+(?:\s*,\s*\d+)+)\]",
        lambda m: "".join(f"[{n.strip()}]" for n in m.group(1).split(",")),
        text,
    )  # [1, 2]
    return text


def _tidy(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\u00ad", "")).strip()
