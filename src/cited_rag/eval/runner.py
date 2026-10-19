"""Run golden cases through the retriever (no LLM) or the full RAG (with LLM)."""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..answer import RAG
from ..retriever import Retriever
from .golden import GoldenCase
from .metrics import hit_at_k, is_relevant, key_fact_recall, mean, mrr, ndcg_at_k

KS = (1, 3, 5, 10)


def _pct(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def refusal_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    adv = [r for r in rows if not r["answerable"]]
    ans = [r for r in rows if r["answerable"]]
    correct = sum(r["refused"] for r in adv) + sum(not r["refused"] for r in ans)
    return {
        "refusal_accuracy": correct / len(rows) if rows else 0.0,
        "out_of_scope_refused": mean([float(r["refused"]) for r in adv]),
        "answerable_wrongly_refused": mean([float(r["refused"]) for r in ans]),
    }


def eval_retrieval(retriever: Retriever, cases: list[GoldenCase]) -> dict[str, Any]:
    index = retriever.index
    rows = []
    for case in cases:
        t0 = time.perf_counter()
        res = retriever.retrieve(case.question, case.filters)
        latency = time.perf_counter() - t0
        row: dict[str, Any] = {
            "id": case.id,
            "kind": case.kind,
            "question": case.question,
            "answerable": case.answerable,
            "expected": {"doc": case.doc, "pages": case.pages},
            "queries": res.queries,
            "refused": res.refused,
            "confidence": round(res.confidence, 4),
            "latency_s": round(latency, 4),
            "hits": [
                {
                    "chunk_id": h.chunk.chunk_id,
                    "doc": h.chunk.doc_id,
                    "pages": [h.chunk.page_start, h.chunk.page_end],
                    "section": h.chunk.section,
                    "score": round(h.score, 4),
                }
                for h in res.hits
            ],
        }
        if case.answerable:
            rels = [
                is_relevant(h.chunk.doc_id, h.chunk.pages, case.doc, case.pages) for h in res.hits
            ]
            n_rel = sum(is_relevant(c.doc_id, c.pages, case.doc, case.pages) for c in index.chunks)
            row["first_relevant_rank"] = next((i + 1 for i, r in enumerate(rels) if r), None)
            for k in KS:
                row[f"hit@{k}"] = hit_at_k(rels, k)
            row["mrr@10"] = mrr(rels, 10)
            row["ndcg@10"] = ndcg_at_k(rels, 10, n_rel)
        rows.append(row)
    ans = [r for r in rows if r["answerable"]]
    lat = [r["latency_s"] for r in rows]
    summary: dict[str, Any] = {f"hit@{k}": mean([r[f"hit@{k}"] for r in ans]) for k in KS}
    summary["mrr@10"] = mean([r["mrr@10"] for r in ans])
    summary["ndcg@10"] = mean([r["ndcg@10"] for r in ans])
    summary.update(refusal_summary(rows))
    summary["latency_p50_s"] = statistics.median(lat) if lat else 0.0
    summary["latency_p95_s"] = _pct(lat, 0.95)
    summary["n_cases"] = len(rows)
    summary["n_answerable"] = len(ans)
    by_kind: dict[str, dict[str, float]] = {}
    for kind in sorted({r["kind"] for r in ans}):
        rs = [r for r in ans if r["kind"] == kind]
        by_kind[kind] = {
            "n": len(rs),
            "hit@5": mean([r["hit@5"] for r in rs]),
            "mrr@10": mean([r["mrr@10"] for r in rs]),
        }
    summary["by_kind"] = by_kind
    return {"summary": summary, "rows": rows}


def _rescore(row: dict[str, Any], case: GoldenCase) -> dict[str, Any]:
    """(Re)compute answer metrics from a stored answer, so metric fixes apply to cached runs."""
    if "error" in row:
        return row
    rels = [
        is_relevant(c["doc_id"], range(c["pages"][0], c["pages"][1] + 1), case.doc, case.pages)
        if case.doc
        else False
        for c in row["citations"]
    ]
    row["citation_precision"] = mean([float(x) for x in rels]) if rels else 0.0
    row["citation_hit"] = float(any(rels))
    row["key_fact_recall"] = (
        key_fact_recall(row["answer"], case.key_facts) if not row["refused"] else 0.0
    )
    return row


def eval_answers(
    rag: RAG,
    cases: list[GoldenCase],
    cache_path: Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    done: dict[str, dict] = {}
    if cache_path and cache_path.exists():
        for line in cache_path.read_text().splitlines():
            r = json.loads(line)
            done[r["id"]] = r
    rows = []
    for case in cases:
        if case.id in done:
            rows.append(_rescore(done[case.id], case))
            continue
        t0 = time.perf_counter()
        try:
            a = rag.answer(case.question, case.filters)
        except Exception as e:  # keep going; record the failure honestly
            row = {
                "id": case.id,
                "error": f"{type(e).__name__}: {e}",
                "answerable": case.answerable,
            }
            rows.append(row)
            if progress:
                progress(f"{case.id} ERROR {row['error'][:120]}")
            continue
        d = a.to_dict()
        row = {
            "id": case.id,
            "kind": case.kind,
            "answerable": case.answerable,
            "refused": a.refused,
            "answer": a.text,
            "citations": d["citations"],
            "grounding": d["grounding"],
            "latency_s": round(time.perf_counter() - t0, 3),
        }
        row = _rescore(row, case)
        rows.append(row)
        if cache_path:
            with open(cache_path, "a") as f:
                f.write(json.dumps(row) + "\n")
        if progress:
            progress(f"{case.id} refused={a.refused} facts={row['key_fact_recall']:.2f}")
    ok = [r for r in rows if "error" not in r]
    answered = [r for r in ok if r["answerable"] and not r["refused"]]
    grounded = [r for r in answered if r["grounding"]]
    summary = {
        "n_cases": len(rows),
        "errors": len(rows) - len(ok),
        "key_fact_recall": mean([r["key_fact_recall"] for r in ok if r["answerable"]]),
        "citation_precision": mean([r["citation_precision"] for r in answered]),
        "citation_hit": mean([r["citation_hit"] for r in answered]),
        "sentence_support_rate": mean([r["grounding"]["supported_rate"] for r in grounded]),
        "fully_grounded_answers": mean([float(not r["grounding"]["flagged"]) for r in grounded]),
        **refusal_summary(ok),
    }
    return {"summary": summary, "rows": rows}
