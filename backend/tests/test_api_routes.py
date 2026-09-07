"""
test_api_routes.py
------------------
Unit tests for FastAPI endpoints:
  - GET /api/health
  - GET /api/repos
  - DELETE /api/repos/{repo_name}
  - GET /api/search
  - POST /api/index
"""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.main import create_app
from backend.app.models.schemas import CodeChunk, Language, ChunkType
from backend.app.indexing.vector_store import VectorStore
from backend.app.indexing.keyword_index import KeywordIndex


@pytest.fixture
def temp_index_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def client(temp_index_dir):
    with patch("backend.app.config.settings.index_dir", temp_index_dir):
        app = create_app()

        # Initialize mock state for tests
        vs = VectorStore(index_dir=temp_index_dir, model_name="all-MiniLM-L6-v2")
        kw = KeywordIndex(index_dir=temp_index_dir)
        app.state.vector_store = vs
        app.state.keyword_index = kw

        with TestClient(app) as test_client:
            yield test_client


def _sample_chunk(repo_name="demo_repo", symbol_name="func_a") -> CodeChunk:
    return CodeChunk(
        file_path="src/main.py",
        language=Language.python,
        symbol_name=symbol_name,
        chunk_type=ChunkType.function,
        code="def func_a(): return 42",
        start_line=1,
        end_line=2,
        repo_name=repo_name,
    )


class TestHealthEndpoint:
    def test_health_check(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "vector_store_size" in data
        assert "bm25_index_size" in data


class TestReposEndpoint:
    def test_list_repos_empty(self, client):
        response = client.get("/api/repos")
        assert response.status_code == 200
        data = response.json()
        assert data["total_chunks"] == 0
        assert data["repos"] == []

    def test_list_and_delete_repo(self, client, temp_index_dir):
        vs = client.app.state.vector_store
        kw = client.app.state.keyword_index

        chunk = _sample_chunk(repo_name="my_repo")
        vec = np.ones((1, 384), dtype=np.float32)
        vs.add([chunk], vec)
        kw.add([chunk])

        # List repos
        res = client.get("/api/repos")
        assert res.status_code == 200
        data = res.json()
        assert data["total_chunks"] == 1
        assert len(data["repos"]) == 1
        assert data["repos"][0]["repo_name"] == "my_repo"
        assert data["repos"][0]["chunk_count"] == 1

        # Delete repo
        del_res = client.delete("/api/repos/my_repo?delete_clone=false")
        assert del_res.status_code == 200
        del_data = del_res.json()
        assert del_data["repo_name"] == "my_repo"
        assert del_data["chunks_removed"] == 1

        # Verify list is empty
        res_after = client.get("/api/repos")
        assert res_after.json()["total_chunks"] == 0

    def test_delete_non_existent_repo_returns_404(self, client):
        res = client.delete("/api/repos/non_existent_repo")
        assert res.status_code == 404


class TestSearchEndpoint:
    def test_search_no_index_404(self, client):
        res = client.get("/api/search?q=test")
        assert res.status_code == 404

    @patch("backend.app.embeddings.embedder.get_embedder")
    def test_search_with_indexed_data(self, mock_get_embedder, client):
        mock_embedder = MagicMock()
        mock_embedder.encode_one.return_value = np.ones(384, dtype=np.float32)
        mock_get_embedder.return_value = mock_embedder

        vs = client.app.state.vector_store
        kw = client.app.state.keyword_index

        chunk = _sample_chunk(repo_name="test_repo", symbol_name="target_func")
        vec = np.ones((1, 384), dtype=np.float32)
        vs.add([chunk], vec)
        kw.add([chunk])

        res = client.get("/api/search?q=target_func&repo_name=test_repo")
        assert res.status_code == 200
        data = res.json()
        assert data["query"] == "target_func"
        assert len(data["results"]) > 0
        assert data["results"][0]["symbol_name"] == "target_func"
