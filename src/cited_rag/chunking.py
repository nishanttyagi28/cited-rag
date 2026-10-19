"""Token-window chunking with parent windows.

Pages are grouped into section blocks (a run of pages under the same TOC heading).
Each block is cut into parent windows of ``parent_size`` tokens, and each parent is cut
into child chunks of ``chunk_size`` tokens with ``chunk_overlap`` tokens of overlap.
Children are what gets embedded and searched; parents are what the LLM reads.
Every chunk keeps the PDF page range its characters came from.
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass, field
from typing import Protocol

from .types import Chunk, Page


class Tokenizer(Protocol):
    def spans(self, text: str) -> list[tuple[int, int]]:
        """Character (start, end) span of every token in ``text``."""
        ...


class WhitespaceTokenizer:
    """Cheap tokenizer for tests: one token per non-space run."""

    _re = re.compile(r"\S+")

    def spans(self, text: str) -> list[tuple[int, int]]:
        return [m.span() for m in self._re.finditer(text)]


class HFTokenizer:
    """Wraps a Hugging Face fast tokenizer so chunk sizes match the embedding model."""

    def __init__(self, model_name: str) -> None:
        from transformers import AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(model_name)
        self._tok.model_max_length = 10**9  # we window ourselves; silence length warnings

    def spans(self, text: str) -> list[tuple[int, int]]:
        enc = self._tok(text, add_special_tokens=False, return_offsets_mapping=True)
        return [tuple(o) for o in enc["offset_mapping"]]


@dataclass
class Parent:
    parent_id: str
    doc_id: str
    text: str
    page_start: int
    page_end: int
    section: str
    page_breaks: list[tuple[int, int]] = field(default_factory=list)  # (char offset, page)

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    def page_at(self, offset: int) -> int:
        page = self.page_start
        for off, pg in self.page_breaks:
            if off <= offset:
                page = pg
            else:
                break
        return page


def windows(n: int, size: int, overlap: int) -> list[tuple[int, int]]:
    """Token index windows [start, end) covering ``n`` tokens."""
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be in [0, size)")
    if n == 0:
        return []
    out, start, step = [], 0, size - overlap
    while True:
        end = min(start + size, n)
        out.append((start, end))
        if end >= n:
            return out
        start += step


def _blocks(pages: list[Page]) -> list[list[Page]]:
    blocks: list[list[Page]] = []
    for p in pages:
        if blocks and blocks[-1][-1].section == p.section and blocks[-1][-1].doc_id == p.doc_id:
            blocks[-1].append(p)
        else:
            blocks.append([p])
    return blocks


def chunk_pages(
    pages: list[Page],
    tokenizer: Tokenizer,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    parent_size: int = 1500,
    meta: dict | None = None,
) -> tuple[list[Chunk], list[Parent]]:
    chunks: list[Chunk] = []
    parents: list[Parent] = []
    for block in _blocks(pages):
        doc_id, section = block[0].doc_id, block[0].section
        text, starts = "", []
        for p in block:
            if text:
                text += "\n\n"
            starts.append(len(text))
            text += p.text
        page_nums = [p.page for p in block]

        def page_at(char: int, _s=starts, _p=page_nums) -> int:
            return _p[max(0, bisect.bisect_right(_s, char) - 1)]

        spans = tokenizer.spans(text)
        if not spans:
            continue
        p_wins = windows(len(spans), parent_size, 0) if parent_size > 0 else [(0, len(spans))]
        for ps, pe in p_wins:
            pid = f"{doc_id}:p{len(parents):04d}"
            p_lo, p_hi = spans[ps][0], spans[pe - 1][1]
            if parent_size > 0:
                breaks = [(max(0, st - p_lo), pg) for st, pg in zip(starts, page_nums, strict=True)]
                breaks = [(o, pg) for o, pg in breaks if o < p_hi - p_lo]
                parents.append(
                    Parent(
                        pid,
                        doc_id,
                        text[p_lo:p_hi],
                        page_at(p_lo),
                        page_at(p_hi - 1),
                        section,
                        [
                            b
                            for i, b in enumerate(breaks)
                            if i == len(breaks) - 1 or b[0] != breaks[i + 1][0]
                        ],
                    )
                )
            for cs, ce in windows(pe - ps, chunk_size, chunk_overlap):
                lo, hi = spans[ps + cs][0], spans[ps + ce - 1][1]
                cid = f"{doc_id}:{len(chunks):04d}"
                chunks.append(
                    Chunk(
                        chunk_id=cid,
                        doc_id=doc_id,
                        text=text[lo:hi],
                        page_start=page_at(lo),
                        page_end=page_at(hi - 1),
                        section=section,
                        parent_id=pid if parent_size > 0 else cid,
                        meta=dict(meta or {}),
                    )
                )
    return chunks, parents
