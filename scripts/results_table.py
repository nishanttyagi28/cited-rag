"""Print the README results tables (markdown) from results/*.json."""

from __future__ import annotations

import json
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / "results"


def f(x: float) -> str:
    return f"{x:.3f}"


def main() -> None:
    b = json.loads((RES / "benchmark.json").read_text())
    print(
        "| config | hit@1 | hit@3 | hit@5 | hit@10 | MRR@10 | NDCG@10 | refusal acc (all) "
        "| refusal acc (held-out half) | out-of-scope refused | answerable refused "
        "| p50 latency |"
    )
    print("|---|" + "---|" * 11)
    for name, s in b["retrieval"].items():
        c = s["calibration"]
        print(
            f"| {name} | {f(s['hit@1'])} | {f(s['hit@3'])} | {f(s['hit@5'])} | "
            f"{f(s['hit@10'])} | {f(s['mrr@10'])} | {f(s['ndcg@10'])} | "
            f"{f(s['refusal_accuracy'])} | {f(c['held_out_accuracy'])} | "
            f"{f(s['out_of_scope_refused'])} | {f(s['answerable_wrongly_refused'])} | "
            f"{s['latency_p50_s'] * 1000:.0f} ms |"
        )
    if b.get("answers"):
        print()
        print(
            "| config | key-fact recall | citation precision | answers with a correct citation "
            "| sentence support rate | fully grounded answers | refusal acc "
            "| out-of-scope refused | answerable refused | errors |"
        )
        print("|---|" + "---|" * 9)
        for name, s in b["answers"].items():
            print(
                f"| {name} | {f(s['key_fact_recall'])} | {f(s['citation_precision'])} | "
                f"{f(s['citation_hit'])} | {f(s['sentence_support_rate'])} | "
                f"{f(s['fully_grounded_answers'])} | {f(s['refusal_accuracy'])} | "
                f"{f(s['out_of_scope_refused'])} | {f(s['answerable_wrongly_refused'])} | "
                f"{s['errors']} |"
            )
    p = RES / "cache_latency.json"
    if p.exists():
        c = json.loads(p.read_text())
        print()
        print(
            f"cache: {c['chunks']} chunks, embed cold {c['embed_cold_s']} s, warm "
            f"{c['embed_warm_s']} s ({c['embed_reduction_pct']}%), ingest cold "
            f"{c['ingest_cold_s']} s, warm {c['ingest_warm_s']} s "
            f"({c['ingest_reduction_pct']}%), median of {c['repeats']}"
        )


if __name__ == "__main__":
    main()
