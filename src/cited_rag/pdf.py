"""PDF loading. Page numbers come straight from the PDF (1-based physical pages)."""

from __future__ import annotations

import re
from pathlib import Path

import pymupdf

from .types import Page

_WS = re.compile(r"[ \t]+")


def clean_text(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # join hyphenated line breaks
    text = _WS.sub(" ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def section_for_pages(toc: list[list], n_pages: int) -> list[str]:
    """Map each page to the deepest TOC heading that starts on or before it."""
    sections = [""] * (n_pages + 1)
    entries = sorted(
        ((int(p), lvl, str(t).strip()) for lvl, t, p, *_ in toc if int(p) >= 1),
        key=lambda e: (e[0], e[1]),
    )
    current = ""
    i = 0
    for page in range(1, n_pages + 1):
        while i < len(entries) and entries[i][0] <= page:
            title = entries[i][2]
            if re.search(r"[A-Za-z]{2,}", title):  # skip junk entries like "1."
                current = title
            i += 1
        sections[page] = current
    return sections[1:]


def load_pdf(path: Path, doc_id: str) -> list[Page]:
    doc = pymupdf.open(path)
    try:
        sections = section_for_pages(doc.get_toc(), doc.page_count)
        pages = []
        for i, pg in enumerate(doc):
            text = clean_text(pg.get_text())
            if text:
                pages.append(Page(doc_id=doc_id, page=i + 1, text=text, section=sections[i]))
        return pages
    finally:
        doc.close()
