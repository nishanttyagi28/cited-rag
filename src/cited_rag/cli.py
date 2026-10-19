"""Command line: ingest, query, eval, bench, cache-bench."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .config import PRESETS, load_config
from .filters import parse_cli_filters


def _paths(args):
    from .components import Paths

    p = Paths(data=Path(args.data), index_root=Path(args.index_dir))
    if args.cache:
        p.cache = Path(args.cache)
    return p


def cmd_ingest(args) -> int:
    from .components import get_index

    cfg = load_config(args.config)
    index, t = get_index(cfg, _paths(args), rebuild=True)
    print(
        f"indexed {len(index.chunks)} chunks / {len(index.parents)} parents "
        f"from {len(index.docs)} docs -> {args.index_dir}/{cfg.index_key}"
    )
    print(json.dumps({k: round(v, 2) for k, v in t.items()}))
    return 0


def _llm(args):
    from .llm import GroqLLM

    return GroqLLM(model=args.model)


def cmd_query(args) -> int:
    from .answer import RAG
    from .components import embedder_for, retriever_for

    overrides = {"web_fallback": True} if args.web else {}
    cfg = load_config(args.config, **overrides)
    paths = _paths(args)
    llm = None if args.retrieve_only else _llm(args)
    retriever = retriever_for(cfg, paths, llm=llm)
    filters = parse_cli_filters(args.filter or [])
    if args.retrieve_only:
        res = retriever.retrieve(args.question, filters)
        print(f"confidence={res.confidence:.3f} refused={res.refused} queries={res.queries}")
        for i, h in enumerate(res.hits, 1):
            snippet = " ".join(h.chunk.text.split())[:160]
            print(f"{i:2d}. [{h.chunk.cite()}] {h.score:.3f} {h.chunk.section[:40]!r}: {snippet}")
        return 0
    a = RAG(retriever, llm, embedder_for(cfg, paths)).answer(args.question, filters)
    print(a.text)
    if a.cited_blocks:
        print("\nSources:")
        for b in a.cited_blocks:
            print(f"  [{b.n}] {b.label()} - {b.section}")
    if a.grounding and a.grounding.flagged:
        print("\nUnsupported sentences (flagged):")
        for s in a.grounding.flagged:
            print(f"  - {s.text} ({s.reason}, score {s.score:.2f})")
    return 0


def _write(out: Path, name: str, obj) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    p = out / name
    p.write_text(json.dumps(obj, indent=1))
    return p


def run_retrieval_eval(name: str, args):
    from .components import retriever_for
    from .eval.golden import load_golden
    from .eval.runner import eval_retrieval

    cfg = load_config(name)
    retriever = retriever_for(cfg, _paths(args))
    retriever.retrieve("warm up")  # load models before timing
    result = eval_retrieval(retriever, load_golden(args.golden))
    result["config_name"] = name
    result["config"] = cfg.to_dict()
    return result


def run_answer_eval(name: str, args):
    from .answer import RAG
    from .components import embedder_for, retriever_for
    from .eval.golden import load_golden
    from .eval.runner import eval_answers

    cfg = load_config(name)
    paths = _paths(args)
    llm = _llm(args)
    rag = RAG(retriever_for(cfg, paths, llm=llm), llm, embedder_for(cfg, paths))
    cases = load_golden(args.golden)
    if args.limit:
        cases = cases[: args.limit]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result = eval_answers(
        rag, cases, out / f"answers_{name}.jsonl", progress=lambda m: print(f"  {name}: {m}")
    )
    result["config_name"] = name
    result["model"] = llm.model
    result["llm_usage"] = llm.usage
    return result


def _fmt(summary: dict) -> str:
    keys = ["hit@1", "hit@5", "hit@10", "mrr@10", "ndcg@10", "refusal_accuracy", "latency_p50_s"]
    return "  ".join(f"{k}={summary[k]:.3f}" for k in keys if k in summary)


def cmd_eval(args) -> int:
    res = run_retrieval_eval(args.config, args)
    p = _write(Path(args.out), f"retrieval_{args.config}.json", res)
    print(f"{args.config}: {_fmt(res['summary'])}\n-> {p}")
    if args.answers:
        ans = run_answer_eval(args.config, args)
        p = _write(Path(args.out), f"answers_{args.config}.json", ans)
        print(json.dumps(ans["summary"], indent=1), f"\n-> {p}")
    return 0


def cmd_bench(args) -> int:
    from .eval.calibrate import calibrate

    names = args.configs.split(",")
    table = {}
    for name in names:
        res = run_retrieval_eval(name, args)
        res["summary"]["calibration"] = calibrate(res["rows"])
        _write(Path(args.out), f"retrieval_{name}.json", res)
        table[name] = res["summary"]
        print(f"{name:12s} {_fmt(res['summary'])}")
    bench = {"retrieval": table, "golden": args.golden, "date": time.strftime("%Y-%m-%d")}
    for name in [n for n in (args.answers or "").split(",") if n]:
        ans = run_answer_eval(name, args)
        _write(Path(args.out), f"answers_{name}.json", ans)
        bench.setdefault("answers", {})[name] = ans["summary"] | {"model": ans["model"]}
        print(f"{name:12s} answers: {json.dumps(ans['summary'])}")
    _write(Path(args.out), "benchmark.json", bench)
    return 0


def cmd_cache_bench(args) -> int:
    from .bench import measure_cache

    res = measure_cache(load_config(args.config), _paths(args), Path(args.tmp), args.repeats)
    p = _write(Path(args.out), "cache_latency.json", res)
    print(json.dumps(res, indent=1), f"\n-> {p}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="cited-rag")
    ap.add_argument("--data", default="data")
    ap.add_argument("--index-dir", default="index")
    ap.add_argument("--cache", default=None, help="embedding cache file")
    sub = ap.add_subparsers(dest="cmd", required=True)
    choices = sorted(PRESETS)

    p = sub.add_parser("ingest", help="parse, chunk and embed the PDFs")
    p.add_argument("--config", default="full")
    p.set_defaults(fn=cmd_ingest)

    p = sub.add_parser("query", help="ask a question")
    p.add_argument("question")
    p.add_argument("--config", default="full")
    p.add_argument("--filter", action="append", help="e.g. doc=nist-csf-2.0, year>=2023")
    p.add_argument("--retrieve-only", action="store_true", help="no LLM, show ranked chunks")
    p.add_argument("--web", action="store_true", help="enable web fallback for this query")
    p.add_argument("--model", default=None)
    p.set_defaults(fn=cmd_query)

    for name, fn in (("eval", cmd_eval), ("bench", cmd_bench)):
        p = sub.add_parser(name)
        if name == "eval":
            p.add_argument("--config", default="full", help=f"preset {choices} or json path")
            p.add_argument("--answers", action="store_true", help="also run answer eval (LLM)")
        else:
            p.add_argument("--configs", default="naive,hybrid,full,full-maxsim")
            p.add_argument("--answers", default="", help="configs to answer-eval, e.g. naive,full")
        p.add_argument("--golden", default="eval/golden.jsonl")
        p.add_argument("--out", default="results")
        p.add_argument("--limit", type=int, default=0)
        p.add_argument("--model", default=None)
        p.set_defaults(fn=fn)

    p = sub.add_parser("cache-bench", help="measure cold vs warm embedding latency")
    p.add_argument("--config", default="full")
    p.add_argument("--tmp", default=".cache/bench")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--out", default="results")
    p.set_defaults(fn=cmd_cache_bench)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
