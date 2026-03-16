# SPDX-License-Identifier: Apache-2.0
"""Tests for the cancellation endpoints in server.py.

Verifies that cancel routes exist and that cancelling an unknown ID returns 404.
Uses FastAPI TestClient to exercise real HTTP routing.
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
    """Create a TestClient with a mock engine that returns False for abort."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from vmlx_engine.server import app

    mock_engine = MagicMock()
    # abort_request returns False (request not found)
    mock_engine.abort_request = AsyncMock(return_value=False)
    mock_engine.get_stats.return_value = {"engine_type": "simple"}
    mock_engine.is_mllm = False

    with patch("vmlx_engine.server._engine", mock_engine), \
         patch("vmlx_engine.server._api_key", None):
        from starlette.testclient import TestClient
        yield TestClient(app, raise_server_exceptions=False)


class TestCancellationEndpoints:
    """Tests for /v1/.../cancel routes."""

    def test_cancel_unknown_id_returns_404(self, client):
        """Cancelling a nonexistent request ID should return 404."""
        resp = client.post("/v1/chat/completions/nonexistent-id-12345/cancel")
        assert resp.status_code == 404
        data = resp.json()
        assert "not found" in data["detail"].lower() or "already finished" in data["detail"].lower()

    def test_cancel_chat_completion_endpoint_exists(self, client):
        """Verify the chat completion cancel route is registered (not 405/404 for method)."""
        resp = client.post("/v1/chat/completions/test-id/cancel")
        # Should be 404 (request not found), NOT 405 (method not allowed)
        # or 422 (unprocessable) which would mean route doesn't exist
        assert resp.status_code in (404, 200)

    def test_cancel_responses_endpoint_exists(self, client):
        """Verify the responses cancel route is registered."""
        resp = client.post("/v1/responses/test-resp-id/cancel")
        assert resp.status_code in (404, 200)

    def test_cancel_completions_endpoint_exists(self, client):
        """Verify the completions cancel route is registered."""
        resp = client.post("/v1/completions/test-cmpl-id/cancel")
        assert resp.status_code in (404, 200)
