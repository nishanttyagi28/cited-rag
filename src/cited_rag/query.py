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


def _subsequence(needle: str, hay: str) -> bool:
    it = iter(hay)
    return all(ch in it for ch in needle)


def _matches(acro: str, words: list[str]) -> bool:
    letters = acro.lower().rstrip("s") if acro.endswith("s") and len(acro) > 2 else acro.lower()
    content = [w for w in words if w.lower() not in _SMALL]
    if not content or content[0][0].lower() != letters[0]:
        return False
    initials = "".join(w if (w.isupper() and len(w) > 1) else w[0] for w in content).lower()
    return _subsequence(initials, letters) and _subsequence(letters, "".join(words).lower())


_SMALL = {"and", "of", "the", "for", "to", "in", "on"}


def mine_acronyms(text: str) -> dict[str, str]:
    """Find 'Long Form Words (LFW)' patterns. The long form is the shortest run of words
    before the parenthesis whose initials fit the acronym (handles 'Generative AI (GAI)'
    and 'Cybersecurity Framework (CSF)')."""
    glossary: dict[str, str] = {}
    for long, acro in _ACRO.findall(text.replace("\n", " ")):
        words = _WORD.findall(long)
        if len(acro) < 2:
            continue
        for n in range(1, min(len(words), len(acro) + 3) + 1):
            cand = words[-n:]
            if cand[0].lower() in _SMALL:
                continue
            if _matches(acro, cand):
                glossary.setdefault(acro, " ".join(cand))
                break
    return glossary


def expand(query: str, glossary: dict[str, str]) -> list[str]:
    """Return extra query variants: acronyms spelled out and long forms abbreviated."""
    variants: list[str] = []
    expanded = query
    for acro, long in glossary.items():
        if len(acro.rstrip("s")) < 3:  # two-letter codes (AI, SP, ID) are too ambiguous
            continue
        has_acro = re.search(rf"\b{re.escape(acro)}\b", query) is not None
        if long.lower() in query.lower():
            if not has_acro:
                expanded = f"{expanded} {acro}"
        elif has_acro:
            expanded = re.sub(rf"\b{re.escape(acro)}\b", f"{acro} ({long})", expanded, count=1)
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
        "You rewrite a vague question into one short, specific search question. Replace "
        "pronouns like 'it' or 'the framework' with the most likely topic or document. "
        "Name at most one document, and only if the question points to it. Keep the user's "
        "intent and key terms. Do not answer it. Reply with the question only.",
        f"Documents available:\n{titles}\n\nQuestion: {query}",
        max_tokens=120,
    ).strip()
    return out.splitlines()[0].strip() if out else query
