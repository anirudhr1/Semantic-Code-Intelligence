# Architecture

## Overview

Semantic Code Intelligence is a hybrid code-search system that combines dense
vector retrieval (semantic similarity) with sparse keyword retrieval (BM25)
and a symbol-name match bonus to produce a single ranked list of code chunks.

```
┌─────────────────────────────────────────────────────────────────┐
│                         Client layer                             │
│   Browser UI (index.html / app.js)  ·  CLI (build_index.py)    │
└────────────────────────┬────────────────────────────────────────┘
                         │  HTTP / REST
┌────────────────────────▼────────────────────────────────────────┐
│                       FastAPI app  (main.py)                     │
│   POST /api/index          GET /api/search        GET /api/health│
│   routes_index.py          routes_search.py                      │
└──────┬──────────────────────────────────────────────────────────┘
       │                                      │
┌──────▼─────────┐                  ┌─────────▼──────────────────┐
│  Ingestion     │                  │  Hybrid Ranker             │
│  repo_loader   │                  │  hybrid_ranker.py          │
│  ast_extractor │                  │  fused = w_sem·sem         │
│  chunking      │                  │         + w_bm25·bm25      │
└──────┬─────────┘                  │         + w_sym·symbol     │
       │ CodeChunk[]                └─────────┬──────────────────┘
┌──────▼─────────┐                  ┌─────────▼──────────────────┐
│  Embedder      │                  │  VectorStore (FAISS)       │
│  embedder.py   │──embeddings──►   │  vector_store.py           │
│  sentence-     │                  │  IndexFlatIP + chunks.pkl  │
│  transformers  │                  ├────────────────────────────┤
└────────────────┘                  │  KeywordIndex (BM25)       │
                                    │  keyword_index.py          │
                                    │  BM25Okapi + bm25.pkl      │
                                    └────────────────────────────┘
                                              │ persist
                                    ┌─────────▼──────────────────┐
                                    │  data/index/               │
                                    │    faiss.index             │
                                    │    chunks.pkl              │
                                    │    bm25.pkl                │
                                    └────────────────────────────┘
```

---

## Components

### Ingestion pipeline

| Module | Responsibility |
|--------|---------------|
| `repo_loader.py` | Accepts a local path or remote git URL, clones if necessary, walks the directory tree filtering by file extension, returns a `RepoLoadResult` with a sorted list of source file paths. |
| `ast_extractor.py` | Parses each source file with tree-sitter and emits a `CodeChunk` per function, method, or class definition. Falls back to a single whole-file chunk when tree-sitter is unavailable or no definitions are found. |
| `chunking.py` | Post-processing utilities: splits oversized chunks (> 3 000 chars) into overlapping line-boundary sub-chunks, deduplicates identical content, and filters blank chunks. |

### Embedding

`Embedder` (sentence-transformers) loads a model once, encodes text in batches
of 64, and L2-normalises all output vectors so that inner product equals cosine
similarity. The text fed to the model is:

```
<symbol_name>

<docstring>

<code>
```

The symbol name and docstring are prepended so the model can exploit them even
when the query is a natural-language description rather than a code pattern.

### Indexing

| Index | Storage | Operation |
|-------|---------|-----------|
| **FAISS `IndexFlatIP`** | `data/index/faiss.index` + `chunks.pkl` | Exact inner-product search over L2-normalised vectors. No quantisation, so recall is 100 % at the cost of linear scan time — acceptable for up to ~500 k chunks. |
| **BM25Okapi** | `data/index/bm25.pkl` | Bag-of-words ranking using code-aware tokenisation (camelCase splitting, delimiter splitting, lowercase). Symbol names are repeated 3× in the document corpus to up-weight API-name matches. |

Both indexes are loaded into memory at startup and persisted to disk after
every successful index run.

### Hybrid Ranker

```
fused_score = w_semantic  × cosine_score   (FAISS inner product, [0, 1])
            + w_bm25      × bm25_score     (BM25 normalised to [0, 1])
            + w_symbol    × symbol_score   (tiered name match, [0, 1])
```

Default weights: `0.6 / 0.3 / 0.1` (configurable via `.env`).

**Symbol score tiers:**

| Condition | Score |
|-----------|-------|
| Query == symbol name (case-insensitive) | 1.0 |
| Query is a substring of symbol name | 0.6 |
| Symbol name is a substring of query | 0.4 |
| No match | 0.0 |

The ranker fetches `top_k × 10` candidates from each index to build a large
enough fusion pool before truncating to `top_k`.

### API

Built with FastAPI. Both indexes live on `app.state` (set during the
`lifespan` context manager), making them accessible to all request handlers
without global variables or additional dependency injection boilerplate.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/index` | POST | Full ingestion pipeline for one repo. |
| `/api/search` | GET | Hybrid search with optional language filter. |
| `/api/health` | GET | Liveness check + index size stats. |

### Frontend

Vanilla HTML/CSS/JS (no bundler). The UI is served as a static directory
mounted at `/frontend`; the root `/` route redirects to `index.html`.
`app.js` communicates with the same-origin API using `fetch`, requiring
no CORS configuration in production.

---

## Data flow: indexing a repository

```
POST /api/index { "repo_url": "https://github.com/..." }
  │
  ├─ repo_loader.load_repo()
  │     git clone → data/repos/<name>/
  │     walk tree → [Path, …]
  │
  ├─ ast_extractor.extract_chunks(file)  × N files
  │     tree-sitter parse → [CodeChunk, …]
  │
  ├─ chunking.split_oversized_chunks()
  ├─ chunking.filter_empty_chunks()
  │
  ├─ embedder.encode_chunks()            batch size 64
  │     sentence-transformers → float32 ndarray (N, dim)
  │
  ├─ vector_store.add(chunks, embeddings)
  │     faiss.IndexFlatIP.add()
  │
  ├─ keyword_index.add(chunks)
  │     BM25Okapi rebuild
  │
  ├─ vector_store.save()  →  data/index/faiss.index + chunks.pkl
  ├─ keyword_index.save() →  data/index/bm25.pkl
  │
  └─ IndexResponse { repo_name, chunks_indexed, files_processed, … }
```

## Data flow: searching

```
GET /api/search?q=calculate+distance&top_k=10
  │
  ├─ embedder.encode_one(query)
  │     → float32 vector (dim,)
  │
  ├─ vector_store.search(vector, top_k=100)
  │     faiss inner-product → [(CodeChunk, cosine_score), …]
  │
  ├─ keyword_index.search(query, top_k=100)
  │     BM25Okapi.get_scores() → [(CodeChunk, bm25_score), …]
  │
  ├─ union by chunk_id
  │
  ├─ for each chunk:
  │     symbol_score = _symbol_score(query, chunk.symbol_name)
  │     fused = w_sem*sem + w_bm25*bm25 + w_sym*symbol
  │
  ├─ sort descending, truncate to top_k
  │
  └─ SearchResponse { query, results: [SearchResult, …] }
```

---

## Scalability notes

- **Up to ~500 k chunks**: `IndexFlatIP` is fast enough (< 50 ms) and fits
  comfortably in RAM. Beyond that, swap to `IndexIVFFlat` or `IndexHNSWFlat`
  for sub-linear retrieval.
- **Incremental indexing**: currently the entire BM25 model is rebuilt on
  every `add()` call (rank-bm25 limitation). For very large corpora, consider
  a streaming BM25 implementation or Elasticsearch.
- **GPU embedding**: change `device="cpu"` → `device="cuda"` in the Embedder
  for 10–20× throughput improvement on large batch indexing jobs.
- **Multi-repo namespacing**: `CodeChunk.repo_name` is stored but not yet used
  as an index partition. A future namespace filter would avoid cross-repo noise.

---

## Key dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `fastapi` | 0.111 | Web framework |
| `sentence-transformers` | 3.0 | Code embedding |
| `faiss-cpu` | 1.8 | Approximate nearest-neighbour search |
| `rank-bm25` | 0.2 | BM25Okapi keyword index |
| `tree-sitter` | 0.22 | Language-agnostic AST parsing |
| `gitpython` | 3.1 | Remote repo cloning |
| `pydantic` / `pydantic-settings` | 2.x | Schema validation and config |
