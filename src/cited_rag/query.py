"""Query expansion (recall) and query rewriting (ambiguous questions)."""

from __future__ import annotations

import re
from typing import Protocol

_ACRO = re.compile(r"\b([A-Z][A-Za-z,\-/ ]{3,80}?)\s+\(([A-Z][A-Za-z0-9\-]{1,9})\)")
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]*")

PRONOUNS = {"it", "this", "that", "they", "them", "these", "those", "its", "there"}
VAGUE = {"framework", "document", "doc", "profile", "standard", "thing", "guidance"}


class TextLLM(Protocol):
    def complete(self, system: str, user: str, max_tokens: int = 512) -> str: ...


def mine_acronyms(text: str) -> dict[str, str]:
    """Find 'Long Form Words (LFW)' patterns whose initials match the acronym."""
    glossary: dict[str, str] = {}
    for long, acro in _ACRO.findall(text.replace("\n", " ")):
        words = _WORD.findall(long)
        letters = [c for c in acro if c.isupper()]
        n = len(letters)
        if n < 2 or len(words) < n:
            continue
        cand = words[-n:]
        # allow small words like "and"/"of" inside: try widening by up to 2 words
        for extra in range(0, 3):
            cand = words[-(n + extra) :]
            initials = [w[0].upper() for w in cand if w.lower() not in {"and", "of", "the", "for"}]
            if initials == letters:
                glossary.setdefault(acro, " ".join(cand))
                break
    return glossary


def expand(query: str, glossary: dict[str, str]) -> list[str]:
    """Return extra query variants: acronyms spelled out and long forms abbreviated."""
    variants: list[str] = []
    expanded = query
    for acro, long in glossary.items():
        if re.search(rf"\b{re.escape(acro)}\b", query):
            expanded = re.sub(rf"\b{re.escape(acro)}\b", f"{acro} ({long})", expanded)
        elif long.lower() in query.lower():
            expanded = f"{expanded} {acro}"
    if expanded != query:
        variants.append(expanded)
    return variants


def llm_expand(query: str, llm: TextLLM, n: int = 2) -> list[str]:
    out = llm.complete(
        "You write alternative search queries. Reply with one query per line, nothing else.",
        f"Write {n} differently-worded search queries for: {query}",
        max_tokens=200,
    )
    lines = [ln.strip(" -*0123456789.\t") for ln in out.splitlines()]
    return [ln for ln in lines if ln and ln.lower() != query.lower()][:n]


def is_ambiguous(query: str) -> bool:
    words = [w.lower() for w in _WORD.findall(query)]
    content = [w for w in words if len(w) > 2]
    if len(content) < 3:
        return True
    has_pronoun = any(w in PRONOUNS for w in words[:4])
    has_entity = any(t.isupper() and len(t) > 1 for t in _WORD.findall(query)) or any(
        w[0].isdigit() for w in words
    )
    vague_ref = any(w in VAGUE for w in words) and not has_entity
    return (has_pronoun and not has_entity) or vague_ref


def rewrite(query: str, llm: TextLLM, doc_titles: list[str]) -> str:
    titles = "\n".join(f"- {t}" for t in doc_titles)
    out = llm.complete(
        "You rewrite vague questions into one specific, self-contained search question. "
        "Keep the user's intent. Do not answer it. Reply with the question only.",
        f"Documents available:\n{titles}\n\nQuestion: {query}",
        max_tokens=120,
    ).strip()
    return out.splitlines()[0].strip() if out else query
