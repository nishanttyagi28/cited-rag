"""Golden test cases (JSONL)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class GoldenCase:
    id: str
    question: str
    doc: str | None  # None => out of scope, must return no answer
    pages: list[int] = field(default_factory=list)
    key_facts: list[Any] = field(default_factory=list)
    kind: str = "factual"  # factual | acronym | ambiguous | filtered | adversarial
    filters: dict[str, Any] | None = None

    @property
    def answerable(self) -> bool:
        return self.doc is not None


def load_golden(path: str | Path) -> list[GoldenCase]:
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("//"):
            out.append(GoldenCase(**json.loads(line)))
    ids = [c.id for c in out]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate golden ids")
    return out
