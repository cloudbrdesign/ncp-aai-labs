# Module 4: Knowledge integration and RAG

Version 4 of the course project. The support desk learns to answer product questions from
its own product manuals and to read order facts from a real database. The manuals are
cleaned, deduplicated, chunked, embedded and stored in a local Milvus Lite vector database;
questions are answered with dense, keyword (BM25) or hybrid search, filtered by product.
Orders move from a JSON file into SQLite, read through a read-only SQL tool. The M3 desk
graph then routes every request: SQL for order facts, RAG for manual questions, and both
when only the order says which product's manual to search.
Runs in free mode on a laptop: no GPU, no Docker, no AWS.

## Install

From the repo root, with the Module 0 venv active:

```bash
pip install -r m04/requirements.txt
ollama pull embeddinggemma          # the embedding model (llama3.2:3b from Module 0 stays the chat model)
```

Chat defaults to local Ollama `llama3.2:3b`; the NVIDIA variables from the Module 1 README
work for chat as before. Embeddings always come from local Ollama `embeddinggemma`
(`M04_EMBED_MODEL` picks another Ollama embedding model; re-run step 2 after changing it).
The same model has to embed the chunks and the questions. The NVIDIA embedding models
(`NVIDIAEmbeddings` in `langchain-nvidia-ai-endpoints`) are not wired in: untested with our key.

A 3B model is small, so it only fills small schemas: the route (step types, order IDs,
product), one reply draft, one groundedness grade, and the optional SQL query in step 5.
Cleaning, dedup, chunking, search, fusion, SQL and the checks are plain Python or Milvus,
and every model step has a fallback the log names (`keyword router`, `template`).

Everything the lab builds lives in `m04/state/` (`clean/`, `manuals.db`, `threads.db`,
`memory.db`) and `m04/data/orders.db`, none of it in Git. To start from scratch:
`rm -rf m04/state m04/data/orders.db`.

| File | What it is |
|---|---|
| `data/manuals/` | four product manuals (made up for the course) plus four planted problem files |
| `data/questions.jsonl` | 20 questions, each labelled with the manual section that answers it |
| `clean.py` | step 1: normalise, quality filters, exact and near-duplicate removal |
| `ingest.py`, `vector_store.py` | step 2: load, chunk, embed, index; the Milvus Lite schema and indexes |
| `retrieve.py`, `compare_retrieval.py` | step 3: dense, keyword and hybrid search; hit@3 per mode |
| `make_orders_db.py`, `orders_sql.py` | step 5: the orders database and the read-only SQL tool |
| `router.py`, `desk_steps.py`, `desk_graph.py` | step 6: the desk with routing, SQL and RAG |
| `llm_calls.py` | the one place that calls the chat and embedding models |

## 1. Clean, filter and deduplicate

```bash
python m04/clean.py --jaccard
```

`m04/data/manuals/` holds the four real documents (H200 headset, D300 dock, M270 monitor,
warranty and returns FAQ) and four planted problems: an exact copy of the dock manual, an
older revision of the headset manual, a garbled scanned page and a near-empty page. The
script runs the stages of a NeMo Curator text pipeline in plain Python and says what each
one did:

```
== 1. Normalise
[normalise] d300_dock.md: NFKC ﬁ->fi; 2 boilerplate lines removed
[normalise] m270_monitor.md: NFKC ３->3, ７->7, Ｅ->E; 2 boilerplate lines removed
== 2. Quality filters
[filter] DROP d300_quickstart.md: too short (11 words < 50)
[filter] DROP m270_monitor_scan.md: garbled (80% symbols > 25%)
== 3. Exact dedup (MD5 of the normalised text)
[exact] DROP d300_dock_copy.md: same MD5 as d300_dock.md (b8817852)
== 4. Near dedup (5-character shingles, Jaccard >= 0.8)
[near-dup] 0.98  h200_headset.md ~ h200_headset_rev1.md
[near-dup] DROP h200_headset_rev1.md: Jaccard 0.98 with h200_headset.md; revision 2026-01 < 2026-06
[INFO] kept 4 of 8: d300_dock.md, h200_headset.md, m270_monitor.md, warranty_returns_faq.md
```

The copy differs from the original only in spaces and blank lines, so its raw bytes have a
different MD5, and the normalised text the same one: normalise first, then hash. The
monitor manual writes its error code in full-width characters (`Ｅ７３`); after NFKC it is
`E73`, which keyword search can find. Curator does fuzzy dedup at scale with MinHash and
LSH on a GPU; comparing every pair, as here, only works for a handful of files.

## 2. Chunk, embed and index

```bash
python m04/ingest.py --chunks-only                              # statistics only
python m04/ingest.py --chunks-only --chunk-size 600 --overlap 120
python m04/ingest.py                                            # size 300, overlap 60, HNSW
```

The loader joins hard-wrapped lines back into paragraphs and reads the product code from
each file. `MarkdownHeaderTextSplitter` cuts at the `## Section` headings (the section
becomes metadata), then `RecursiveCharacterTextSplitter` cuts long sections into chunks
that share `--overlap` characters with their neighbour. Sizes are characters, not tokens.

```
[chunk] 37 chunks (size 300, overlap 60): D300 10, FAQ 6, H200 11, M270 10
[chunk] 7 neighbouring pairs overlap
[chunk] D300-setup-1 -> D300-setup-2 share: "when the dock works normally. Windows and ma..."
```

With `--chunk-size 600` you get 24 chunks and no overlapping pairs: most sections now fit
in one chunk. The full run embeds the 37 chunks with `embeddinggemma` and writes the
collection `manuals` to `m04/state/manuals.db`:

```
[embed] 37 chunks with ollama embeddinggemma: dimension ..., ... s
[milvus] index on dense: HNSW, metric COSINE (M=16, efConstruction=200)
[milvus] index on sparse: SPARSE_INVERTED_INDEX, metric BM25 (Milvus computes it from text)
[milvus] inserted 37 rows; the collection has 37; indexes: dense, sparse
```

The dimension is read from the model's first vector. `sparse` is filled by Milvus itself:
a BM25 function in the schema turns `text` into term weights. The vectors are normalised
to length 1, so COSINE and IP (inner product, which GPU indexes need) rank the same.
`--index FLAT` (exact, the baseline) and `--index IVF_FLAT` (`nlist=8` clusters) build the
other index types. Milvus Lite builds FLAT, HNSW, HNSW_SQ, IVF_FLAT and IVF_SQ8; it accepts
DISKANN or GPU_CAGRA without an error but builds nothing for them, so the lab offers only
the first three.

**What duplicates do to search.** Index the uncleaned files next to the clean ones:

```bash
python m04/ingest.py --raw
python m04/retrieve.py --collection manuals_raw --mode keyword "What does error E42 mean?"
```

```
 1  5.2741  D300-troubleshooting-1
   ...E42: the power adapter gives too little power. This ha...
 2  5.2741  D300-troubleshooting-1@d300_dock_copy
   ...E42: the power adapter gives too little power. This ha...
```

The copy takes a second place in the top 3 and brings nothing new.

## 3. Dense, keyword and hybrid search

```bash
python m04/retrieve.py --mode keyword "What does error E42 mean?"
python m04/retrieve.py --mode dense "How do I stop the headphones from blocking outside noise?"
python m04/retrieve.py "My second screen just copies the first one"          # hybrid, RRF
python m04/retrieve.py --ranker weighted --weights 0.7 0.3 "E42 on my dock"
python m04/retrieve.py --product D300 "the light blinks orange"               # filtered
```

Each result shows its score, chunk ID and the text around the first matching word. For
hybrid results you also see where the chunk ranked in each list:

```
[INFO] mode hybrid (RRF k=60) | k=3 | filter: none
Question: My second screen just copies the first one
 1  0.0...  D300-connecting-displays-2  (dense ., kw 1)
   ...second display shows the same picture as the first (mi...
```

RRF scores each chunk by 1/(60 + rank) in each list, so a chunk near the top of both lists
wins. `--product D300` adds the filter `product == "D300"`: only the dock's chunks are
candidates. `--ef` sets how many candidates HNSW keeps while it searches.

Then measure all three modes on the 20 labelled questions:

```bash
python m04/compare_retrieval.py
python m04/compare_retrieval.py --ranker weighted --weights 0.3 0.7
```

```
hit@3 (hybrid: RRF k=60)
mode         all  paraphrase   exact  ms/query
dense      ../20       ../10    ../10     ...
keyword    ../20       ../10    ../10     ...
hybrid     ../20       ../10    ../10     ...
[INFO] dense misses: ...
[INFO] HNSW vs FLAT (exact): HNSW found ...% of FLAT's top-3 chunks (... vs ... ms/query)
[INFO] hybrid: right chunk in the top 20 for ../20, in the top 3 for ../20
```

A question is a hit when a chunk from the right product and section is in the top 3.
Expect keyword search to win on the exact codes (E42, MST, VESA), dense search on the
everyday wording ("headphones", "hang the screen on the wall"), and hybrid to cover most of
both. HNSW should find the same chunks as FLAT here: with 37 vectors there is nothing to
approximate. Your numbers depend on the embedding model; `--verbose` shows each
question's top chunk per mode.

## 4. Reranking (not run)

The last line of `compare_retrieval.py` is the room a reranker has: a reranker re-scores
the retrieved chunks (here, the hybrid top 20) and keeps the best 3, so it can only fix
questions whose right chunk is in the 20 but not in the 3. The lesson covers the RAG
Blueprint's reranker settings; the lab doesn't download a reranking model.

## 5. A read-only SQL tool

```bash
python m04/make_orders_db.py                     # orders.db: 8 orders (A1001-A1003 from M1)
python m04/orders_sql.py A1003
python m04/orders_sql.py --customer "Tom B."
python m04/orders_sql.py --free-sql "Which orders are still processing?"
```

The desk only uses named, parameterised queries (`order_status`, `order_product`,
`orders_for_customer`): the model supplies an order ID, SQLite binds it with `?`, and the
query itself never changes. A 3B model can't damage anything that way.

```
[sql] order_status('A1003') -> 6 fields
   order_id=A1003, customer=Lena K., status=processing, note=awaiting stock
   product=M270, item=4K monitor
[sql] order_product('A1003') -> M270
```

`--free-sql` shows the other way: the model writes the SELECT from the table schema, a
validator allows one SELECT on the two known tables with a LIMIT, and the connection is
opened read-only (`file:...?mode=ro`). Then a planted `DELETE` meets both guards:

```
[INFO] planted query: DELETE FROM orders WHERE order_id = 'A1001'
[guard] BLOCKED by the validator: not a SELECT; write keyword DELETE
[INFO] now skip the validator and send it straight to the read-only connection
[guard] BLOCKED by SQLite: attempt to write a readonly database
[INFO] order A1001 is still there
```

## 6. The desk: SQL, RAG or both

```bash
python m04/desk_graph.py --input "Where is order A1003?"
python m04/desk_graph.py --input "My D300 dock shows E42. What does it mean?"
python m04/desk_graph.py --input "The dock from order A1002 won't drive my second screen."
python m04/desk_graph.py --bad-draft --input "My D300 dock shows E42. What does it mean?"
```

The M3 graph (recall, plan, execute, draft, critique, remember) with a new step type,
`manual_search`. The plan line names each step's source. The third request needs both:
the plan carries the order ID instead of a product, SQL says the order was a D300, and the
manual search is filtered to the dock:

```
[plan] 1 step (model): manual_search A1002 (rag)
[execute] step 1/1 manual_search A1002 (rag)
[sql]   order_product(A1002) -> D300
[rag]   hybrid search, filter product == "D300"
[rag]   -> D300-..., D300-..., D300-...
[draft] first draft (model): ... [D300-...]
[critique] PASS: order IDs answered, none invented, 1 citation, all retrieved, grounded
```

The draft sees the order facts and the passages, each with its chunk ID, and must cite
the IDs it uses. `critique` keeps the M3 checks and adds three: every cited ID was retrieved
in this turn, nothing but passage IDs goes in square brackets (a bare `[D300]` is sent
back), and a reply that uses passages cites at least one. `--bad-draft` cites a
headset chunk that was never retrieved:

```
[critique] FAIL unknown_chunk: [H200-specifications-1] was not retrieved; cite only the passage IDs given
[critique] lesson saved (1/3 kept): Cite only the passage IDs you were given.
[draft] revision 1 (model): ...
```

If the model's plan breaks a rule (a manual question without `manual_search`, an order ID
that isn't in the request), the keyword router plans instead and the log says so. Threads,
long-term memory and lessons work as in M3 (`--thread`, `--customer`), stored in
`m04/state/`. `--draw` writes the graph to `m04/graph.mmd`.

## Check

```bash
python m04/check.py
```

It rebuilds the lab's state and checks: both models answer; clean drops the four planted
files and keeps the four manuals; the collection has one row per chunk, the model's
dimension, both indexes and the metadata; neighbouring chunks overlap; the product filter
returns only that product; keyword search ranks the E42 chunk first; hybrid hit@3 is at
least the lower of dense and keyword; the SQL tool answers A1003 and refuses a write; the
router sends order, manual and mixed requests to the right sources and an invalid plan
falls back; the desk filters the manual search by the order's product and cites only
retrieved chunks, with nothing else in square brackets; a planted citation is caught.

Small local models don't follow instructions every time. If a check fails, run it again;
if it keeps failing, try NVIDIA mode for chat or a larger local model.

Offline self-test (course maintainers only): `M04_FAKE_LLM=1 python m04/check.py` swaps
the models for `m04/tests/fake_llm.py`, a scripted chat model and a hashed bag-of-words
embedder. You never need it.

## Tested

2026-09-29, Python 3.11.15: pymilvus 2.6.9, milvus-lite 3.2.1, langchain-text-splitters
1.1.2, ollama 0.6.2 (Python client), langgraph 1.2.12, langgraph-checkpoint-sqlite 3.1.1,
langchain-core 1.6.5, nvidia-nat 1.9.0, pydantic 2.13.5. Checked with the offline
self-test on Linux; the run with Ollama `llama3.2:3b` and `embeddinggemma` on a 16 GB Mac
comes next.

The manuals, orders and customers are made up for the course.
