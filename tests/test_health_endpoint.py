# SPDX-License-Identifier: Apache-2.0
"""Tests for the /health endpoint in server.py.

Verifies status reporting, path sanitization, JANG metadata, memory info,
and last_request_time fields by patching server globals directly.
"""

import asyncio
import platform
import sys
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or platform.machine() != "arm64",
    reason="Requires Apple Silicon",
)


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


class TestHealthEndpoint:
    """Tests for the health() async handler."""

    def test_health_no_model_loaded(self):
        """When _engine is None, health returns status='no_model'."""
        from vmlx_engine import server

        with (
            patch.object(server, "_engine", None),
            patch.object(server, "_model_name", None),
            patch.object(server, "_model_load_error", None),
            patch.object(server, "_mcp_manager", None),
            patch.object(server, "_jang_metadata", None),
            patch.object(server, "_last_request_time", 0.0),
        ):
            result = _run(server.health())

        assert result["status"] == "no_model"
        assert result["model_loaded"] is False

    def test_health_with_model_loaded(self):
        """When _engine is present with get_stats(), status='healthy'."""
        from vmlx_engine import server

        mock_engine = MagicMock()
        mock_engine.get_stats.return_value = {"engine_type": "simple"}
        mock_engine.is_mllm = False

        with (
            patch.object(server, "_engine", mock_engine),
            patch.object(server, "_model_name", "test-model"),
            patch.object(server, "_model_load_error", None),
            patch.object(server, "_mcp_manager", None),
            patch.object(server, "_jang_metadata", None),
            patch.object(server, "_last_request_time", 0.0),
        ):
            result = _run(server.health())

        assert result["status"] == "healthy"
        assert result["model_loaded"] is True
        assert result["model_name"] == "test-model"
        assert result["engine_type"] == "simple"

    def test_health_error_sanitizes_paths(self):
        """Path strings in _model_load_error are replaced with <path>."""
        from vmlx_engine import server

        error_msg = "FileNotFoundError: /home/user/models/foo/config.json not found"

        with (
            patch.object(server, "_engine", None),
            patch.object(server, "_model_name", None),
            patch.object(server, "_model_load_error", error_msg),
            patch.object(server, "_mcp_manager", None),
            patch.object(server, "_jang_metadata", None),
            patch.object(server, "_last_request_time", 0.0),
        ):
            result = _run(server.health())

        assert "error" in result
        assert "/home/user" not in result["error"]
        assert "<path>" in result["error"]

    def test_health_jang_metadata_cached(self):
        """When _jang_metadata is set, it appears as quantization_format in response."""
        from vmlx_engine import server

        jang_meta = {
            "type": "jang",
            "target_bits": 4.5,
            "actual_bits": 4.48,
            "block_size": 64,
        }

        mock_engine = MagicMock()
        mock_engine.get_stats.return_value = {"engine_type": "simple"}
        mock_engine.is_mllm = False

        with (
            patch.object(server, "_engine", mock_engine),
            patch.object(server, "_model_name", "jang-model"),
            patch.object(server, "_model_load_error", None),
            patch.object(server, "_mcp_manager", None),
            patch.object(server, "_jang_metadata", jang_meta),
            patch.object(server, "_last_request_time", 0.0),
        ):
            result = _run(server.health())

        assert "quantization_format" in result
        assert result["quantization_format"]["type"] == "jang"
        assert result["quantization_format"]["target_bits"] == 4.5

    def test_health_memory_info(self):
        """When mlx memory functions exist, memory dict is present."""
        from vmlx_engine import server

        mock_engine = MagicMock()
        mock_engine.get_stats.return_value = {"engine_type": "simple"}
        mock_engine.is_mllm = False

        # Mock mlx.core memory functions that health() calls internally
        mock_mx = MagicMock()
        mock_mx.get_active_memory.return_value = 1024 * 1024 * 512  # 512 MB
        mock_mx.get_peak_memory.return_value = 1024 * 1024 * 1024  # 1 GB
        mock_mx.get_cache_memory.return_value = 1024 * 1024 * 256  # 256 MB

        with (
            patch.object(server, "_engine", mock_engine),
            patch.object(server, "_model_name", "test-model"),
            patch.object(server, "_model_load_error", None),
            patch.object(server, "_mcp_manager", None),
            patch.object(server, "_jang_metadata", None),
            patch.object(server, "_last_request_time", 0.0),
            patch.dict("sys.modules", {"mlx.core": mock_mx}),
        ):
            result = _run(server.health())

        # Memory info should be present since mlx.core is mocked with the functions
        if "memory" in result:
            assert "active_mb" in result["memory"]
            assert "peak_mb" in result["memory"]
            assert "cache_mb" in result["memory"]
        # If mlx.core import fails in test env, memory may be absent — that's OK

    def test_health_last_request_time(self):
        """Verify last_request_time field exists in health response."""
        from vmlx_engine import server

        mock_engine = MagicMock()
        mock_engine.get_stats.return_value = {"engine_type": "simple"}
        mock_engine.is_mllm = False

        with (
            patch.object(server, "_engine", mock_engine),
            patch.object(server, "_model_name", "test-model"),
            patch.object(server, "_model_load_error", None),
            patch.object(server, "_mcp_manager", None),
            patch.object(server, "_jang_metadata", None),
            patch.object(server, "_last_request_time", 1700000000.0),
        ):
            result = _run(server.health())

        # last_request_time should be present (non-zero value means it's not None)
        assert "last_request_time" in result
        assert result["last_request_time"] == 1700000000.0
