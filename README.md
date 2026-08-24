# Semantic Code Intelligence

A hybrid semantic + keyword code search engine that indexes source code repositories and lets you query them with natural language or symbol names.

---

## Features

- **AST-based chunking** — tree-sitter extracts functions, classes, and top-level blocks per language
- **Semantic search** — sentence-transformers embeddings stored in FAISS
- **Keyword search** — BM25 full-text index over raw source tokens
- **Hybrid ranking** — configurable fusion of semantic score, BM25 score, and symbol-name match
- **REST API** — FastAPI backend with `/api/index` and `/api/search` endpoints
- **Web UI** — lightweight vanilla JS frontend

---

## Project Structure

```
semantic-code-intelligence/
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI entrypoint
│   │   ├── config.py                # Settings (env-driven)
│   │   ├── models/schemas.py        # Pydantic models
│   │   ├── ingestion/
│   │   │   ├── repo_loader.py       # Load local path or git clone
│   │   │   └── ast_extractor.py     # tree-sitter → CodeChunk list
│   │   ├── embeddings/embedder.py   # sentence-transformers wrapper
│   │   ├── indexing/
│   │   │   ├── vector_store.py      # FAISS index
│   │   │   └── keyword_index.py     # BM25 index
│   │   ├── ranking/hybrid_ranker.py # Score fusion
│   │   ├── api/
│   │   │   ├── routes_index.py      # POST /api/index
│   │   │   └── routes_search.py     # GET /api/search
│   │   └── utils/chunking.py        # Chunk helpers
│   ├── scripts/build_index.py       # CLI indexing tool
│   └── tests/
│       ├── test_ast_extractor.py
│       └── test_ranking.py
├── frontend/
│   ├── index.html
│   ├── app.js
│   └── style.css
├── data/
│   ├── repos/     # cloned/local repos (gitignored)
│   └── index/     # persisted FAISS + BM25 + chunks (gitignored)
└── docs/
    └── architecture.md
```

---

## API Reference

| Method | Endpoint       | Body / Params                                              | Description                          |
|--------|----------------|------------------------------------------------------------|--------------------------------------|
| POST   | `/api/index`   | `{ "path": "...", "repo_url": "...", "language": "..." }` | Index a local path or remote git URL |
| GET    | `/api/search`  | `?q=...&top_k=10&language=python`                         | Hybrid search over indexed chunks    |
| GET    | `/api/health`  | —                                                          | Liveness check                       |

### IndexRequest body

```json
{
  "path": "/data/repos/my-project",
  "repo_url": "https://github.com/owner/repo",
  "language": "python"
}
```

Provide either `path` (local) or `repo_url` (will be cloned into `data/repos/`).

### SearchResult schema

```json
{
  "chunk_id": "uuid",
  "file_path": "src/utils.py",
  "language": "python",
  "symbol_name": "calculate_distance",
  "chunk_type": "function",
  "code": "def calculate_distance(a, b): ...",
  "start_line": 42,
  "end_line": 55,
  "score": 0.873
}
```

---

## Setup

### Prerequisites

- Python 3.10+
- Git (for remote repo cloning)
- (Optional) Docker & Docker Compose

### Local

```bash
git clone https://github.com/yourname/semantic-code-intelligence.git
cd semantic-code-intelligence

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
# edit .env as needed

uvicorn backend.app.main:app --reload --port 8000
```

Open `http://localhost:8000` for the API, or open `frontend/index.html` in a browser.

### Docker

```bash
docker-compose up --build
```

---

## CLI Indexing

```bash
python backend/scripts/build_index.py --path ./data/repos/my-project --language python
# or
python backend/scripts/build_index.py --repo-url https://github.com/owner/repo
```

---

## Environment Variables

| Variable               | Default                        | Description                             |
|------------------------|--------------------------------|-----------------------------------------|
| `EMBEDDING_MODEL`      | `all-MiniLM-L6-v2`             | sentence-transformers model name        |
| `INDEX_DIR`            | `data/index`                   | Where FAISS + BM25 index files land     |
| `REPOS_DIR`            | `data/repos`                   | Where remote repos are cloned           |
| `SEMANTIC_WEIGHT`      | `0.6`                          | Weight for semantic score in fusion     |
| `BM25_WEIGHT`          | `0.3`                          | Weight for BM25 score in fusion         |
| `SYMBOL_WEIGHT`        | `0.1`                          | Weight for symbol name match in fusion  |
| `DEFAULT_TOP_K`        | `10`                           | Default number of results returned      |

---

## Roadmap

- [ ] Multi-language support (Python, JS/TS, Go, Rust, Java)
- [ ] Incremental re-indexing (only changed files)
- [ ] VS Code extension
- [ ] Authentication + multi-tenant index namespaces
- [ ] LLM-powered result summarisation

---

## License

MIT — see [LICENSE](LICENSE).
