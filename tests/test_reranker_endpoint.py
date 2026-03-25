# SPDX-License-Identifier: Apache-2.0
"""Tests for the /v1/rerank endpoint validation in server.py.

Tests the input validation logic (empty query, empty docs, too many docs)
and document normalization (string vs dict formats) using FastAPI TestClient.
"""

import platform
import sys

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or platform.machine() != "arm64",
    reason="Requires Apple Silicon",
)


@pytest.fixture
def client():
    """Create a TestClient with auth disabled and no reranker loaded."""
    from unittest.mock import patch

    from vmlx_engine.server import app

    with patch("vmlx_engine.server._api_key", None), \
         patch("vmlx_engine.server._reranker", None):
        from starlette.testclient import TestClient
        yield TestClient(app, raise_server_exceptions=False)


class TestRerankerValidation:
    """Tests for reranker endpoint input validation."""

    def test_rerank_empty_query_rejected(self, client):
        """Empty query should return 400."""
        resp = client.post("/v1/rerank", json={
            "query": "",
            "documents": ["doc1", "doc2"],
            "model": "some-model",
        })
        assert resp.status_code == 400
        assert "query" in resp.json()["detail"].lower()

    def test_rerank_empty_documents_rejected(self, client):
        """Empty documents list should return 400."""
        resp = client.post("/v1/rerank", json={
            "query": "test query",
            "documents": [],
            "model": "some-model",
        })
        assert resp.status_code == 400
        assert "documents" in resp.json()["detail"].lower()

    def test_rerank_too_many_documents_rejected(self, client):
        """More than 1000 documents should return 400."""
        resp = client.post("/v1/rerank", json={
            "query": "test query",
            "documents": [f"doc-{i}" for i in range(1001)],
            "model": "some-model",
        })
        assert resp.status_code == 400
        assert "1000" in resp.json()["detail"]

    def test_rerank_document_normalization_str(self, client):
        """String documents should pass validation (fails later at reranker load, not at validation).

        We test that the validation logic accepts string documents — the actual
        reranker model load will fail since we have no model, but the validation
        passes (no 400 for the document format).
        """
        resp = client.post("/v1/rerank", json={
            "query": "test query",
            "documents": ["document one", "document two"],
            "model": "nonexistent-model",
        })
        # Should not be 400 (validation passed) — will be 400 (model required)
        # or 500 (reranker load fails) but NOT a document format error
        if resp.status_code == 400:
            # The only acceptable 400 is "model is required" not a document error
            detail = resp.json()["detail"].lower()
            assert "document" not in detail or "model" in detail

    def test_rerank_document_normalization_dict(self, client):
        """Dict documents with 'text' key should pass validation."""
        resp = client.post("/v1/rerank", json={
            "query": "test query",
            "documents": [{"text": "document one"}, {"text": "document two"}],
            "model": "nonexistent-model",
        })
        # Same as above — validation should pass, may fail at reranker load
        if resp.status_code == 400:
            detail = resp.json()["detail"].lower()
            assert "document" not in detail or "model" in detail
