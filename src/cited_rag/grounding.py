"""Citation grounding: is every answer sentence supported by a chunk it cites?

For each sentence we take the chunks it cites ([1], [2] ...), split their text into
sentence windows, and score the best window by
    score = 0.5 * cosine(sentence, window) + 0.5 * lexical recall
where lexical recall is the share of the sentence's content words found in the window.
A sentence is supported when score >= threshold. Uncited sentences are unsupported.
This is a cheap proxy, not an entailment model; see DESIGN.md.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .embed import Embedder
from .sparse import tokenize

_CITE = re.compile(r"\[(\d+)\]")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    # move citations that trail the full stop back inside: "foo. [1] Bar" -> "foo [1]. Bar"
    text = re.sub(r"([.!?])((?:\s*\[\d+\])+)", r"\2\1", text)
    return [p.strip() for p in _SENT.split(text) if p.strip()]


def citations(sentence: str) -> list[int]:
    return [int(n) for n in _CITE.findall(sentence)]


def strip_citations(sentence: str) -> str:
    return _CITE.sub("", sentence).strip()


def lexical_recall(claim: str, evidence: str) -> float:
    c = set(tokenize(claim))
    if not c:
        return 1.0
    return len(c & set(tokenize(evidence))) / len(c)


def _windows(text: str, size: int = 2) -> list[str]:
    sents = split_sentences(text) or [text]
    if len(sents) <= size:
        return [" ".join(sents)]
    return [" ".join(sents[i : i + size]) for i in range(len(sents) - size + 1)]


@dataclass
class SentenceCheck:
    text: str
    cited: list[int]
    score: float
    supported: bool
    reason: str = ""


@dataclass
class GroundingReport:
    sentences: list[SentenceCheck] = field(default_factory=list)

    @property
    def supported_rate(self) -> float:
        if not self.sentences:
            return 1.0
        return sum(s.supported for s in self.sentences) / len(self.sentences)

    @property
    def flagged(self) -> list[SentenceCheck]:
        return [s for s in self.sentences if not s.supported]

    @property
    def fully_grounded(self) -> bool:
        return not self.flagged


def check_grounding(
    answer: str,
    sources: dict[int, str],
    embedder: Embedder | None = None,
    threshold: float = 0.6,
) -> GroundingReport:
    """``sources`` maps citation number -> text of that source."""
    report = GroundingReport()
    for sent in split_sentences(answer):
        claim = strip_citations(sent)
        if len(tokenize(claim)) < 2:
            continue  # headings, list markers, "Sources:" and the like
        cited = citations(sent)
        valid = [n for n in cited if n in sources]
        if not cited:
            report.sentences.append(SentenceCheck(claim, [], 0.0, False, "no citation"))
            continue
        if not valid:
            report.sentences.append(SentenceCheck(claim, cited, 0.0, False, "unknown citation"))
            continue
        windows = [w for n in valid for w in _windows(sources[n])]
        lex = np.array([lexical_recall(claim, w) for w in windows])
        if embedder is not None:
            vecs = embedder.embed_docs([claim, *windows])
            cos = vecs[1:] @ vecs[0]
            scores = 0.5 * cos + 0.5 * lex
        else:
            scores = lex
        best = float(scores.max())
        ok = best >= threshold
        report.sentences.append(
            SentenceCheck(claim, cited, best, ok, "" if ok else "weak support in cited source")
        )
    return report
