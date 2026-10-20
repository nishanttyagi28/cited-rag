# cited-rag

RAG over a handful of NIST PDFs that cites page numbers and says "no answer" when it
doesn't know.

## Why

I wanted to see what each common RAG improvement buys on real documents, measured
rather than assumed. So there is a naive baseline (dense search, fixed chunks, no
rerank) and a full pipeline (hybrid search, reranking, grounding checks, refusal),
both scored on the same 77 golden questions.

Retrieval runs locally on CPU. Answers come from Groq.

## Quickstart

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev,dashboard,charts]"

python scripts/fetch_data.py        # download the PDFs and check sha256
cited-rag ingest --config full      # parse, chunk, embed

cited-rag query "What are the four CSF Tiers?" --retrieve-only
cited-rag query "What does PW.1.1 suggest?" --filter doc=nist-sp-800-218-ssdf

export GROQ_API_KEY=...             # only needed for answers
cited-rag query "How does NIST define confabulation?"

cited-rag bench --configs naive,hybrid,full,full-maxsim   # retrieval eval, no LLM
cited-rag bench --answers full,naive                      # answer eval, slow
cited-rag cache-bench
streamlit run dashboard/app.py
```

The default model is `openai/gpt-oss-120b`. Set `CITED_RAG_MODEL` to change it and
`CITED_RAG_TPM` to change request pacing (default 7000 tokens per minute).

## Results

All numbers are from `results/`. Retrieval evals don't call an LLM.

Configs:

- naive: dense only, 500-token chunks, no overlap, no BM25, no rerank, no expansion
- hybrid: dense + BM25 with RRF, 500/50 chunks, acronym expansion, no rerank
- full: hybrid + cross-encoder rerank + parent context
- full-maxsim: hybrid + MaxSim rerank instead of the cross-encoder

Retrieval, 77 cases (65 answerable, 12 out of scope):

| config | hit@1 | hit@3 | hit@5 | hit@10 | MRR@10 | NDCG@10 | refusal acc | refusal acc (held-out half) | out-of-scope refused | answerable refused | p50 latency |
|---|---|---|---|---|---|---|---|---|---|---|---|
| naive | 0.462 | 0.723 | 0.831 | 0.892 | 0.621 | 0.540 | 0.922 | 0.921 | 0.500 | 0.000 | 7 ms |
| hybrid | 0.538 | 0.723 | 0.862 | 0.923 | 0.659 | 0.589 | 0.935 | 0.947 | 0.750 | 0.031 | 10 ms |
| full | 0.631 | 0.862 | 0.938 | 0.969 | 0.749 | 0.632 | 0.961 | 0.974 | 0.750 | 0.000 | 515 ms |
| full-maxsim | 0.554 | 0.785 | 0.846 | 0.908 | 0.684 | 0.594 | 0.961 | 0.947 | 0.750 | 0.000 | 147 ms |

A hit is a chunk from the expected document whose pages overlap the expected pages.
Refusal thresholds were picked on the even-numbered half of the golden set. The held-out
column is the other half. There are only 12 out-of-scope questions, so take the refusal
numbers as rough.

The cross-encoder is the biggest win and the biggest cost: it takes median latency
from 7 ms to about half a second on CPU. MaxSim beats no rerank but loses to the
cross-encoder.

Answers with `openai/gpt-oss-120b`, full pipeline, all 77 cases:

| key-fact recall | citation precision | answers with a correct citation | sentence support rate | fully grounded | refusal acc | out-of-scope refused | answerable refused |
|---|---|---|---|---|---|---|---|
| 0.777 | 0.750 | 0.879 | 0.909 | 0.862 | 0.909 | 12/12 | 7/65 |

Naive vs full answers on the same 26 questions. The naive run hit Groq's free-tier
daily token limit after 26 of 77 questions, so the comparison only covers those 26.
They are all answerable, so naive refusal of out-of-scope questions is only measured at
the retrieval level above.

| config | key-fact recall | citation precision | answers with a correct citation | sentence support rate | fully grounded | answerable refused |
|---|---|---|---|---|---|---|
| naive | 0.635 | 0.810 | 0.857 | 0.768 | 0.667 | 5/26 |
| full | 0.846 | 0.800 | 0.880 | 0.927 | 0.880 | 1/26 |

Metric notes:

- key-fact recall: share of expected facts found in the answer (case-insensitive
  substring, so a correct paraphrase can miss)
- citation precision: share of citations pointing at an expected page
- sentence support rate: share of sentences that pass the grounding check
- fully grounded: answers where every sentence passed

Embedding cache, full ingest of 323 chunks on CPU, median of 3 runs:

| | cold | warm |
|---|---|---|
| embedding step | 13.06 s | 0.003 s |
| whole ingest | 13.69 s | 0.60 s (95.6% less) |

Warm ingest time is PDF parsing and tokenizing. This ran while an answer eval was
running in the background, so the cold numbers may be a bit high.

Charts: `docs/chart_retrieval.png`, `docs/chart_by_kind.png`, `docs/chart_answers.png`.

## How it works

1. PDFs are split into pages with PyMuPDF. Pages are grouped by PDF bookmark section.
2. Each section is cut into 1500-token parent windows, and each parent into 500-token
   chunks with 50 overlap.
3. Chunks are embedded with bge-small-en-v1.5 and cached on disk by content hash.
4. A query is expanded with acronyms mined from the corpus. If it looks ambiguous and
   an LLM is set up, it gets rewritten first.
5. Dense and BM25 results are fused with RRF, then the top 30 are reranked with
   ms-marco-MiniLM-L-6-v2 and the top 10 are kept.
6. If the top rerank score is below the threshold, it refuses (or falls back to web
   search, if that's turned on, which it isn't by default).
7. Parent windows go to the LLM as numbered sources. Each answer sentence must cite one.
8. Each sentence is checked against the source it cites. Weak ones are flagged.

DESIGN.md has the reasoning and tradeoffs.

## Layout

```
src/cited_rag/   pdf, chunking, embed, cache, sparse, fusion, filters, rerank, query,
                 index, retriever, answer, grounding, llm, websearch, cli
src/cited_rag/eval/   golden loader, metrics, runners, threshold calibration
eval/golden.jsonl     77 golden cases
data/            manifest.json and SOURCES.md (PDFs are downloaded, not committed)
results/         eval and benchmark output
dashboard/       Streamlit app
scripts/         fetch data, charts, results tables
tests/           pytest; `-m slow` runs the tests that need real models
```

Tests: `pytest -m "not slow"` runs 130 tests with no models. `pytest -m slow` runs 2
end-to-end tests with the real models. CI runs ruff and both sets.

## Limitations

- Small corpus: 5 NIST documents, 223 pages, all written in a similar style.
- I wrote the golden set myself. It has only 3 ambiguous and 2 acronym questions.
- Query rewriting needs an LLM, so the retrieval benchmark doesn't use it.
- The grounding check is similarity-based, not entailment. It can miss a flipped meaning
  and flag a correct paraphrase.
- MaxSim uses bge-small token vectors, which weren't trained for late interaction.
- Tables in the PDFs are flattened to text. Page numbers are physical PDF pages, not the
  printed page numbers.
- The naive answer comparison covers 26 of 77 questions.
- Web fallback has only been tested with the fake provider.

## License

MIT for the code. The PDFs are U.S. government works, see `data/SOURCES.md`.
