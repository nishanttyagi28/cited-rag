"""Core data types shared across modules."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Page:
    doc_id: str
    page: int  # 1-based physical PDF page
    text: str
    section: str


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    page_start: int
    page_end: int
    section: str
    parent_id: str
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def pages(self) -> list[int]:
        return list(range(self.page_start, self.page_end + 1))

    def cite(self) -> str:
        p = (
            f"p. {self.page_start}"
            if self.page_start == self.page_end
            else f"pp. {self.page_start}-{self.page_end}"
        )
        return f"{self.doc_id}, {p}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Chunk:
        return cls(**d)


@dataclass
class Hit:
    """A retrieved chunk plus the scores collected along the pipeline."""

    chunk: Chunk
    score: float
    scores: dict[str, float] = field(default_factory=dict)
    context: str = ""  # text handed to the LLM (parent section when enabled)
    source: str = "corpus"  # "corpus" or "web"
    url: str = ""
