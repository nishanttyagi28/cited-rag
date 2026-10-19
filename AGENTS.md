# AGENTS.md

Guidance for anyone (human or tool) changing this repo.

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev,dashboard,charts]"
```

## Commands

| what | command |
|---|---|
| lint | `ruff check . && ruff format --check .` |
| unit tests (fast, no models) | `pytest -m "not slow"` |
| model tests (downloads ~150 MB) | `pytest -m slow` |
| build index | `cited-rag ingest --config full` |
| retrieval-only query | `cited-rag query "..." --retrieve-only` |
| answer with citations | `cited-rag query "..."` (needs `GROQ_API_KEY`) |
| retrieval eval | `cited-rag eval --config full` |
| benchmark | `cited-rag bench --configs naive,hybrid,full,full-maxsim` |
| answer benchmark | `cited-rag bench --answers full,naive` (slow, rate limited) |
| cache latency | `cited-rag cache-bench` |
| charts | `python scripts/make_charts.py` |
| dashboard | `streamlit run dashboard/app.py` |

## Layout

`src/cited_rag/` holds one concern per module: `pdf`, `chunking`, `embed`, `cache`,
`sparse`, `fusion`, `filters`, `rerank`, `query`, `index`, `retriever`, `answer`,
`grounding`, `llm`, `websearch`, `eval/`. `components.py` is the only place that loads
real models.

## PR rules

* One logical change per PR, with tests. CI (ruff + `pytest -m "not slow"` + slow model
  tests) must pass.
* Any change that can move metrics (chunking, fusion, rerank, thresholds, prompts) must
  re-run `cited-rag bench` and update `results/` and the README table in the same PR.
* Plain lowercase commit messages describing the change.

## Always

* Keep unit tests model-free: use `HashingEmbedder`, `FakeLLM`, `FakeWebSearch`.
* Keep page numbers as 1-based physical PDF pages.
* Report only numbers produced by a real run, and say how they were produced.
* Pace LLM calls (`TokenPacer`); the Groq free tier is ~8k tokens/minute.

## Ask first

* Adding a dependency, especially anything that pulls a GPU stack.
* Changing the golden set (`eval/golden.jsonl`) or the calibrated thresholds.
* Adding documents to `data/` (license must be clear; update `SOURCES.md`).
* Turning web fallback on by default.

## Never

* Commit API keys, `.env` files, the `index/` folder or the embedding cache.
* Edit golden cases to make a metric go up.
* Print or log `GROQ_API_KEY` / `TAVILY_API_KEY`.
* Call an LLM from retrieval evaluation.
