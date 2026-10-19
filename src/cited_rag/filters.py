"""Metadata filters for scoped retrieval.

A filter is a dict. Supported keys:
  doc      : str | list[str]        exact doc_id match
  topic    : str | list[str]        manifest topic (ai, security, privacy)
  section  : str                    case-insensitive substring of the section heading
  year     : int | {"gte": int, "lte": int}
  pages    : [lo, hi]               chunk page range must overlap [lo, hi]
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .types import Chunk

KNOWN_KEYS = {"doc", "topic", "section", "year", "pages"}


def _as_list(v: Any) -> list:
    return list(v) if isinstance(v, (list, tuple, set)) else [v]


def matches(chunk: Chunk, flt: dict[str, Any] | None) -> bool:
    if not flt:
        return True
    unknown = set(flt) - KNOWN_KEYS
    if unknown:
        raise ValueError(f"unknown filter keys: {sorted(unknown)}")
    if "doc" in flt and chunk.doc_id not in _as_list(flt["doc"]):
        return False
    if "topic" in flt and chunk.meta.get("topic") not in _as_list(flt["topic"]):
        return False
    if "section" in flt and str(flt["section"]).lower() not in chunk.section.lower():
        return False
    if "year" in flt:
        year = chunk.meta.get("year")
        cond = flt["year"]
        if year is None:
            return False
        if isinstance(cond, dict):
            if "gte" in cond and year < cond["gte"]:
                return False
            if "lte" in cond and year > cond["lte"]:
                return False
        elif year not in _as_list(cond):
            return False
    if "pages" in flt:
        lo, hi = flt["pages"]
        if chunk.page_end < lo or chunk.page_start > hi:
            return False
    return True


def mask(chunks: list[Chunk], flt: dict[str, Any] | None) -> np.ndarray | None:
    if not flt:
        return None
    return np.array([matches(c, flt) for c in chunks], dtype=bool)


def parse_cli_filters(items: list[str]) -> dict[str, Any]:
    """Parse ``key=value`` strings, e.g. doc=nist-csf-2.0 year>=2023 pages=10-20."""
    out: dict[str, Any] = {}
    for item in items:
        if ">=" in item:
            k, v = item.split(">=", 1)
            out.setdefault(k, {})["gte"] = int(v)
        elif "<=" in item:
            k, v = item.split("<=", 1)
            out.setdefault(k, {})["lte"] = int(v)
        else:
            k, v = item.split("=", 1)
            if k == "pages":
                lo, _, hi = v.partition("-")
                out[k] = [int(lo), int(hi or lo)]
            elif k == "year":
                out[k] = int(v)
            else:
                out[k] = v.split(",") if "," in v else v
    return out
