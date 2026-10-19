"""Retrieval quality dashboard.

    streamlit run dashboard/app.py

Reads results/*.json written by `cited-rag bench`. The "Live query" tab loads the
models and runs the retriever (no LLM) against the local index.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

RESULTS = Path(__file__).resolve().parents[1] / "results"
METRICS = ["hit@1", "hit@3", "hit@5", "hit@10", "mrr@10", "ndcg@10"]


@st.cache_data
def load(name: str) -> dict | None:
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def overview(bench: dict) -> None:
    st.subheader("Retrieval metrics per config (no LLM)")
    rows = []
    for name, s in bench["retrieval"].items():
        rows.append(
            {
                "config": name,
                **{m: s[m] for m in METRICS},
                "refusal acc": s["refusal_accuracy"],
                "out-of-scope refused": s["out_of_scope_refused"],
                "p50 latency (s)": s["latency_p50_s"],
            }
        )
    df = pd.DataFrame(rows).set_index("config")
    st.dataframe(df.style.format("{:.3f}"), use_container_width=True)
    st.bar_chart(df[["hit@1", "hit@5", "mrr@10", "ndcg@10"]])
    if bench.get("answers"):
        st.subheader("Answer-level metrics (Groq)")
        adf = pd.DataFrame(bench["answers"]).T
        st.dataframe(adf, use_container_width=True)
    cache = load("cache_latency.json")
    if cache:
        st.subheader("Embedding cache, cold vs warm")
        st.write(
            f"{cache['chunks']} chunks: embed {cache['embed_cold_s']}s cold vs "
            f"{cache['embed_warm_s']}s warm ({cache['embed_reduction_pct']}% less); full ingest "
            f"{cache['ingest_cold_s']}s vs {cache['ingest_warm_s']}s "
            f"({cache['ingest_reduction_pct']}% less), median of {cache['repeats']} runs."
        )


def drilldown(configs: list[str]) -> None:
    name = st.selectbox("config", configs)
    ret = load(f"retrieval_{name}.json")
    ans = load(f"answers_{name}.json")
    rows = ret["rows"]
    kinds = sorted({r["kind"] for r in rows})
    kind = st.multiselect("kind", kinds, default=kinds)
    rows = [r for r in rows if r["kind"] in kind]
    table = pd.DataFrame(
        [
            {
                "id": r["id"],
                "kind": r["kind"],
                "question": r["question"],
                "first relevant rank": r.get("first_relevant_rank"),
                "refused": r["refused"],
                "confidence": r["confidence"],
            }
            for r in rows
        ]
    )
    st.dataframe(table, use_container_width=True, hide_index=True)
    qid = st.selectbox("question", [r["id"] for r in rows])
    r = next(x for x in rows if x["id"] == qid)
    st.markdown(f"**{r['question']}**")
    exp = r["expected"]
    if exp["doc"]:
        st.write(f"Expected: `{exp['doc']}` pages {exp['pages']}")
    else:
        st.write("Expected: no answer (out of scope)")
    st.write(f"Queries used: {r['queries']}")
    hits = []
    for i, h in enumerate(r["hits"], 1):
        rel = exp["doc"] == h["doc"] and bool(
            set(range(h["pages"][0], h["pages"][1] + 1)) & set(exp["pages"])
        )
        hits.append(
            {
                "rank": i,
                "doc": h["doc"],
                "pages": f"{h['pages'][0]}-{h['pages'][1]}",
                "section": h["section"],
                "score": h["score"],
                "relevant": rel,
            }
        )
    st.dataframe(pd.DataFrame(hits), use_container_width=True, hide_index=True)
    if ans:
        a = next((x for x in ans["rows"] if x["id"] == qid), None)
        if a and "answer" in a:
            st.markdown("**Answer**")
            st.write(a["answer"])
            for c in a["citations"]:
                st.write(f"[{c['n']}] {c['doc_id']}, pages {c['pages'][0]}-{c['pages'][1]}")
            if a["grounding"] and a["grounding"]["flagged"]:
                st.warning("Unsupported sentences: " + " | ".join(a["grounding"]["flagged"]))


def compare(configs: list[str]) -> None:
    data = {
        c: {
            r["id"]: r.get("first_relevant_rank")
            for r in load(f"retrieval_{c}.json")["rows"]
            if r["answerable"]
        }
        for c in configs
    }
    df = pd.DataFrame(data)
    st.write("First relevant rank per question (empty = not in top 10)")
    st.dataframe(df, use_container_width=True)


def live() -> None:
    from cited_rag.components import Paths, retriever_for
    from cited_rag.config import PRESETS, load_config
    from cited_rag.filters import parse_cli_filters

    name = st.selectbox("preset", sorted(PRESETS), index=sorted(PRESETS).index("full"))
    q = st.text_input("question", "What are the four CSF Tiers?")
    f = st.text_input("filters (space separated, e.g. doc=nist-csf-2.0 year>=2023)", "")
    if st.button("search"):
        r = retriever_for(load_config(name), Paths()).retrieve(q, parse_cli_filters(f.split()))
        st.write(f"confidence {r.confidence:.3f}, refused: {r.refused}")
        for i, h in enumerate(r.hits, 1):
            st.markdown(f"**{i}. {h.chunk.cite()}** ({h.score:.3f}) - {h.chunk.section}")
            st.caption(h.chunk.text[:500])


def main() -> None:
    st.set_page_config(page_title="cited-rag", layout="wide")
    st.title("cited-rag retrieval quality")
    bench = load("benchmark.json")
    if not bench:
        st.error("No results yet. Run `cited-rag bench` first.")
        return
    configs = list(bench["retrieval"])
    tabs = st.tabs(["Overview", "Per question", "Compare configs", "Live query"])
    with tabs[0]:
        overview(bench)
    with tabs[1]:
        drilldown(configs)
    with tabs[2]:
        compare(configs)
    with tabs[3]:
        live()


main()
