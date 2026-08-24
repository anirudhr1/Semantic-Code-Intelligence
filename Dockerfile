# ── Build stage ───────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install git (needed by GitPython for repo cloning)
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies into a dedicated prefix so we can copy them
# cleanly into the runtime stage.
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --prefix=/install --no-cache-dir -r requirements.txt

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Runtime system deps
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Create data directories expected at runtime
RUN mkdir -p data/repos data/index

# Expose API port
EXPOSE 8000

# Default environment (overridden by docker-compose or .env)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    EMBEDDING_MODEL=all-MiniLM-L6-v2 \
    INDEX_DIR=data/index \
    REPOS_DIR=data/repos \
    SEMANTIC_WEIGHT=0.6 \
    BM25_WEIGHT=0.3 \
    SYMBOL_WEIGHT=0.1 \
    DEFAULT_TOP_K=10 \
    API_HOST=0.0.0.0 \
    API_PORT=8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
