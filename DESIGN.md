# Design notes

Why the pipeline looks like this, what each piece costs, and where it falls short.
Numbers come from `results/`. The README has the full tables.

## Pipeline

```
question
  -> rewrite (only if ambiguous and an LLM is configured)
  -> expansion (acronym glossary mined from the corpus; optional LLM paraphrases)
  -> metadata filter mask (doc, topic, section, year, pages)
  -> dense search (bge-small, cosine)  +  BM25 search      (top 50 each, per query variant)
  -> Reciprocal Rank Fusion (k = 60)
  -> rerank top 30 with a cross-encoder (or MaxSim), keep top 10
  -> refuse if top score < calibrated threshold (or fall back to web search, if enabled)
  -> build context: parent windows around the matched chunks, 8,000 char budget
  -> LLM answer with [n] citations -> per-sentence grounding check
```

## Chunking

Default: 500 tokens, 50 overlap, measured with the embedding model's own tokenizer
(`BAAI/bge-small-en-v1.5`, WordPiece). Chunks never cross a section boundary: pages are
first grouped into blocks by the deepest PDF bookmark (TOC entry) that starts on or before
the page, then each block is cut into parent windows (1,500 tokens) and each parent into
child chunks.

Why this works for these documents:

* NIST frameworks are written as numbered sections and tables of outcomes (GOVERN 1.1,
  PW.1.1, ID.RA-P3). A section is the natural unit a reader cites, so keeping chunks inside
  one section means a chunk's page range is tight and its section label is meaningful.
* 500 tokens sits just under bge-small's 512-token limit, so no chunk is silently
  truncated by the encoder. Bigger chunks would be cut off; smaller chunks lose the
  surrounding definition that most questions need.
* 50 tokens of overlap (10%) means a sentence that straddles a window edge appears whole
  in at least one chunk, at the cost of ~15% more chunks (323 vs 281 for zero overlap).

Tradeoffs:

* Fixed token windows ignore sentence boundaries, so a chunk can start mid-sentence. The
  parent window given to the LLM smooths this over; the search side tolerates it.
* Section blocks depend on the PDF having bookmarks. All five corpus PDFs do; a PDF
  without them degrades to one block per document (still correct, page ranges looser).
* Tables are flattened to text by PyMuPDF. Row/column structure is lost, which matters for
  the long subcategory tables (e.g. AI RMF Tables 1-4).
* Page numbers are 1-based physical PDF pages, not the printed folio. That is what a
  viewer jumps to, but it differs from the page number printed in the footer.

All of it is configurable (`Config.chunk_size`, `chunk_overlap`, `parent_size`); the
`naive` preset uses 500/0 with no parents.

## Parent document retrieval

Children (500 tokens) are embedded and searched. When `parent_context=True`, the context
block given to the LLM is a window of the child's parent (up to 3,200 chars) centred on
where the child starts, and several children from the same parent collapse into one
block. Each block's page range is recomputed from per-page offsets stored on the parent,
so citations still point at the pages the LLM actually saw.

Note: with 500-token children and an 8,000-char context budget (kept small for the
Groq free tier), the parent window is only modestly larger than the child. The main
practical win is de-duplication (two overlapping hits from the same section cost one
block, not two). Smaller children (e.g. 200 tokens) would make the parent step matter more.

## Hybrid search and fusion

Dense retrieval handles paraphrase ("when should systems be tested" -> "tested before
their deployment"). BM25 handles exact identifiers that embeddings blur: `GV.SC`,
`PW.1.1`, `MG-2.4-004`, `14110`. The tokenizer keeps dotted/hyphenated identifiers as one
token for that reason.

RRF (`score = sum 1/(60 + rank)`) is used instead of score blending because cosine
similarities and BM25 scores are on unrelated scales, and RRF needs no tuning. Each query
variant contributes its own dense and BM25 ranking to the fusion.

## Reranking

`cross-encoder/ms-marco-MiniLM-L-6-v2` reads (question, chunk) jointly. It reranks the top
30 fused candidates and the top 10 are kept. This is the single largest quality gain in
the benchmark and also the largest latency cost (~0.5 s/query on CPU vs ~10 ms without).

The cross-encoder's sigmoid output doubles as the refusal signal: it separates
out-of-scope questions far better than cosine similarity does (most out-of-scope
questions score < 0.01).

## Late interaction (MaxSim)

`rerank="maxsim"` scores candidates with ColBERT-style MaxSim over bge-small's own
token embeddings, averaged over query tokens so the score stays in [-1, 1]. Document
token matrices are cached on disk with the same content-hash cache.

Limits: this is not ColBERT. bge-small was trained to produce a pooled
sentence vector, not token vectors meant for MaxSim, and there is no compression or
PLAID index. A real ColBERT checkpoint (e.g. via PyLate) would be the proper version; it
was left out to keep the dependency set and CPU footprint small. In the benchmark the
MaxSim reranker is better than no reranker on MRR but worse than the cross-encoder.

## Query expansion and rewriting

* Expansion (recall): an acronym glossary is mined from the corpus at ingest
  ("Secure Software Development Framework (SSDF)", "Generative AI (GAI)"). A query with
  `SSDF` gets a variant with the long form, and vice versa. Two-letter codes (AI, SP, ID)
  are skipped because they are ambiguous. Optional LLM paraphrases
  (`llm_expansion=True`) cost one LLM call per query and are off by default.
* Rewriting (ambiguous questions): `is_ambiguous` flags short questions, leading
  pronouns without a named entity, and vague references ("the framework") with no
  identifier. Only those are sent to the LLM for a rewrite. Early versions asked the LLM
  to resolve "it" to a document and it guessed the wrong one; the rewrite now drops
  unresolved pronouns instead of guessing, which retrieved the right section in all three
  ambiguous golden cases during development. Without chat history there is nothing better
  to resolve against.

## Metadata filters

Filters are applied as a boolean mask before both dense and BM25 ranking, so scoped
queries cannot be crowded out by other documents. Supported: `doc`, `topic`, `section`
(substring), `year` (exact or `gte`/`lte`), `pages` (overlap).

## Refusal and threshold calibration

Each preset has a `min_confidence` in the units of its final scorer. The value was chosen
by `eval/calibrate.py` on the even-numbered half of the golden set and the README reports
accuracy on the odd-numbered (held-out) half as well. With 77 cases, of which 12 are
out-of-scope, both halves are small, so treat the refusal numbers as rough.

A second refusal layer is the LLM itself: it is told to reply `NO_ANSWER` when the
sources do not contain the answer.

## Citation grounding

After generation, every sentence is checked against the blocks it cites:

```
score = 0.5 * cosine(sentence, best 2-sentence window of cited block)
      + 0.5 * share of the sentence's content words found in that window
```

A sentence is supported if `score >= 0.6`. Uncited sentences and citations to blocks that
do not exist are unsupported. Unsupported sentences are flagged; with
`strict_grounding=True` they are removed, and an answer with nothing left is refused.

This is a cheap proxy, not entailment. It catches invented specifics and uncited claims
but it will pass a sentence that reuses the source's words with the meaning flipped
("does not prescribe" vs "prescribes"), and it can flag a correct paraphrase that shares
few words with the source. An NLI model would be the next step.

## Web fallback

`web_fallback=True` sends low-confidence questions to a `WebSearchProvider` and adds its
results as citable `web:` blocks. Off by default because these NIST documents are the
whole point of the tool and a silent web answer would hide a retrieval failure. Providers
are pluggable (`fake` for tests, `tavily` with `TAVILY_API_KEY`). Only the fake provider
is exercised in tests.

## Embedding cache

SQLite file, key = `sha256(model name, kind, text)`. Content-hash keys mean re-ingesting
after changing only the parent size or a single PDF reuses every unchanged chunk.
Query vectors are not cached by default so query latency numbers stay honest.

## Evaluation

* Golden set: `eval/golden.jsonl`, 77 cases written by reading the PDFs. Every key fact
  is checked by `tests/test_golden.py` to appear on the stated pages, so the set cannot
  drift from the documents.
* Relevance is binary: a retrieved chunk is relevant if it is from the expected document
  and its page range overlaps the expected pages.
* NDCG@10 uses the number of relevant chunks that exist in the index for the ideal DCG,
  so it penalises retrieving one relevant chunk out of several.
* Retrieval evaluation never calls an LLM. Query rewriting therefore does not run in the
  retrieval benchmark; the ambiguous cases are scored on their raw wording there.
* Answer evaluation uses Groq (`openai/gpt-oss-120b` by default): key-fact recall
  (substring match), citation precision (share of citations pointing at an expected
  page), sentence support rate (grounding), and refusal accuracy.
