"""Summarise answer evals from the cached per-question rows.

The Groq free tier caps tokens per day, so the naive answer run may cover only part of
the golden set. This compares configs on the questions every config has answered and
merges the summaries into results/benchmark.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from cited_rag.eval.golden import load_golden
from cited_rag.eval.metrics import mean
from cited_rag.eval.runner import _rescore, refusal_summary

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def summarise(rows: list[dict]) -> dict:
    ok = [r for r in rows if "error" not in r]
    answered = [r for r in ok if r["answerable"] and not r["refused"]]
    grounded = [r for r in answered if r["grounding"]]
    return {
        "n_cases": len(rows),
        "n_answerable": sum(r["answerable"] for r in ok),
        "key_fact_recall": mean([r["key_fact_recall"] for r in ok if r["answerable"]]),
        "citation_precision": mean([r["citation_precision"] for r in answered]),
        "citation_hit": mean([r["citation_hit"] for r in answered]),
        "sentence_support_rate": mean([r["grounding"]["supported_rate"] for r in grounded]),
        "fully_grounded_answers": mean([float(not r["grounding"]["flagged"]) for r in grounded]),
        "errors": len(rows) - len(ok),
        **refusal_summary(ok),
    }


def main(configs: list[str]) -> None:
    cases = {c.id: c for c in load_golden(ROOT / "eval" / "golden.jsonl")}
    rows = {}
    for name in configs:
        lines = (RES / f"answers_{name}.jsonl").read_text().splitlines()
        rows[name] = {r["id"]: _rescore(r, cases[r["id"]]) for r in map(json.loads, lines)}
    common = sorted(set.intersection(*(set(r) for r in rows.values())), key=list(cases).index)
    out = {
        "full_set": {n: summarise(list(r.values())) for n, r in rows.items()},
        "common_subset": {
            "ids": common,
            **{n: summarise([rows[n][i] for i in common]) for n in configs},
        },
    }
    (RES / "answers_summary.json").write_text(json.dumps(out, indent=1))
    bench = json.loads((RES / "benchmark.json").read_text())
    bench["answers"] = out["full_set"]
    bench["answers_common_subset"] = out["common_subset"]
    (RES / "benchmark.json").write_text(json.dumps(bench, indent=1))
    print(json.dumps({k: v for k, v in out["common_subset"].items() if k != "ids"}, indent=1))
    print("common ids:", len(common))
    print(json.dumps(out["full_set"], indent=1))


if __name__ == "__main__":
    main(sys.argv[1:] or ["naive", "full"])
