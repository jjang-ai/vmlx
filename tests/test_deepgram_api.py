# SPDX-License-Identifier: Apache-2.0
"""Deepgram compatibility surface tests."""

import asyncio
import inspect

import pytest
from fastapi import HTTPException

import vmlx_engine.server as server


def test_deepgram_routes_exist_in_server_source():
    source = inspect.getsource(server)
    for route in (
        "/deepgram/vl/listen",
        "/deepgram/vl/speak",
        "/deepgram/vl/read",
        "/deepgram/vl/models",
        "/deepgram/vl/models/{model_id}",
    ):
        assert route in source


def test_auth_header_token_supports_token_scheme():
    assert server._auth_header_token("Token abc123") == "abc123"
    assert server._auth_header_token("Bearer abc123") == "abc123"
    assert server._auth_header_token("Basic xyz") is None


def test_deepgram_model_catalog_shape():
    catalog = server._deepgram_model_catalog()
    assert "stt" in catalog
    assert "tts" in catalog
    assert isinstance(catalog["stt"], list)
    assert isinstance(catalog["tts"], list)
    assert "canonical_name" in catalog["stt"][0]
    assert "target_model" in catalog["tts"][0]


def test_deepgram_listen_response_has_required_metadata_fields():
    payload = server._deepgram_listen_response(
        transcript="hello",
        language="en",
        model="mlx-community/whisper-large-v3-mlx",
        request_id="req_123",
        sha256_hex="abc",
    )
    assert payload["metadata"]["request_id"] == "req_123"
    assert payload["metadata"]["sha256"] == "abc"
    assert payload["metadata"]["channels"] == 1
    assert payload["metadata"]["models"] == ["mlx-community/whisper-large-v3-mlx"]


def test_deepgram_model_detail_returns_entry():
    model = asyncio.run(server.deepgram_model_detail_alias("nova-3"))
    assert model["name"] == "nova-3"
    assert model["provider"] == "local-mlx"


def test_deepgram_model_detail_missing_raises_404():
    with pytest.raises(HTTPException) as exc:
        asyncio.run(server.deepgram_model_detail_alias("does-not-exist"))
    assert exc.value.status_code == 404
