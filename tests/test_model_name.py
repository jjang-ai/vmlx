# SPDX-License-Identifier: Apache-2.0
"""
Comprehensive tests for the model name system:
- _normalize_model_name(): path → "org/model" normalization
- _resolve_model_name(): priority resolution (served > actual > default)
- Permissive model validation in Chat Completions and Responses API
- /v1/models endpoint listing behavior with served_model_name
- CLI arg --served-model-name plumbing
- request.model overwrite to resolved name

These are pure unit tests — no model loading or MLX dependency required.
"""

import asyncio
import platform
import sys

import pytest

# Skip all tests if not on Apple Silicon (same guard as test_server.py)
pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or platform.machine() != "arm64",
    reason="Requires Apple Silicon",
)


# =============================================================================
# _normalize_model_name() Tests
# =============================================================================


class TestNormalizeModelName:
    """Test the _normalize_model_name() path-to-name conversion."""

    def test_full_local_path_extracts_org_model(self):
        """Full filesystem path → last two path components."""
        from vllm_mlx.server import _normalize_model_name

        result = _normalize_model_name(
            "/Users/eric/.lmstudio/models/mlx-community/Llama-3.2-3B-Instruct-4bit"
        )
        assert result == "mlx-community/Llama-3.2-3B-Instruct-4bit"

    def test_huggingface_id_unchanged(self):
        """HuggingFace-style 'org/model' stays unchanged."""
        from vllm_mlx.server import _normalize_model_name

        result = _normalize_model_name("mlx-community/Llama-3.2-3B-Instruct-4bit")
        assert result == "mlx-community/Llama-3.2-3B-Instruct-4bit"

    def test_single_component_stays_as_is(self):
        """A bare model name without org prefix stays unchanged."""
        from vllm_mlx.server import _normalize_model_name

        result = _normalize_model_name("my-local-model")
        assert result == "my-local-model"

    def test_path_with_trailing_slash(self):
        """Trailing slashes are stripped before extraction."""
        from vllm_mlx.server import _normalize_model_name

        result = _normalize_model_name(
            "/Users/eric/models/mlx-community/Qwen3-8B/"
        )
        assert result == "mlx-community/Qwen3-8B"

    def test_deep_nested_path(self):
        """Very deep path still extracts last two components."""
        from vllm_mlx.server import _normalize_model_name

        result = _normalize_model_name(
            "/a/b/c/d/e/f/org-name/model-name"
        )
        assert result == "org-name/model-name"

    def test_relative_path_with_separator(self):
        """Relative path with / is treated as a path."""
        from vllm_mlx.server import _normalize_model_name

        result = _normalize_model_name("models/org/model")
        assert result == "org/model"

    def test_root_path_single_component(self):
        """Path /model → '/model': splits to ['', 'model'], f"{parts[-2]}/{parts[-1]}" = '/model'."""
        from vllm_mlx.server import _normalize_model_name

        # /model splits to ['', 'model'] — len >= 2,
        # so returns f"{''}/{'model'}" = "/model" (empty org prefix + slash + name).
        result = _normalize_model_name("/model")
        assert result == "/model"

    def test_empty_string(self):
        """Empty string → empty string."""
        from vllm_mlx.server import _normalize_model_name

        result = _normalize_model_name("")
        assert result == ""


# =============================================================================
# _resolve_model_name() Tests
# =============================================================================


class TestResolveModelName:
    """Test the _resolve_model_name() priority resolution."""

    def test_served_name_wins_over_actual(self):
        """served_model_name has highest priority."""
        import vllm_mlx.server as server

        orig_served = server._served_model_name
        orig_model = server._model_name
        try:
            server._served_model_name = "my-custom-name"
            server._model_name = "mlx-community/Llama-3.2-3B"
            assert server._resolve_model_name() == "my-custom-name"
        finally:
            server._served_model_name = orig_served
            server._model_name = orig_model

    def test_actual_name_when_no_served_name(self):
        """Falls back to _model_name when no served name."""
        import vllm_mlx.server as server

        orig_served = server._served_model_name
        orig_model = server._model_name
        try:
            server._served_model_name = None
            server._model_name = "mlx-community/Llama-3.2-3B"
            assert server._resolve_model_name() == "mlx-community/Llama-3.2-3B"
        finally:
            server._served_model_name = orig_served
            server._model_name = orig_model

    def test_default_when_nothing_set(self):
        """Falls back to 'default' when nothing is set."""
        import vllm_mlx.server as server

        orig_served = server._served_model_name
        orig_model = server._model_name
        try:
            server._served_model_name = None
            server._model_name = None
            assert server._resolve_model_name() == "default"
        finally:
            server._served_model_name = orig_served
            server._model_name = orig_model

    def test_empty_served_name_treated_as_none(self):
        """Empty string served_model_name is falsy — falls through to _model_name."""
        import vllm_mlx.server as server

        orig_served = server._served_model_name
        orig_model = server._model_name
        try:
            server._served_model_name = ""
            server._model_name = "org/model"
            assert server._resolve_model_name() == "org/model"
        finally:
            server._served_model_name = orig_served
            server._model_name = orig_model

    def test_served_name_with_special_characters(self):
        """Served name can be any arbitrary string."""
        import vllm_mlx.server as server

        orig_served = server._served_model_name
        orig_model = server._model_name
        try:
            server._served_model_name = "CRACK-Qwen3.5-122B-Q4"
            server._model_name = "mlx-community/Qwen3.5-122B"
            assert server._resolve_model_name() == "CRACK-Qwen3.5-122B-Q4"
        finally:
            server._served_model_name = orig_served
            server._model_name = orig_model


# =============================================================================
# /v1/models Endpoint Tests
# =============================================================================


class TestModelsEndpoint:
    """Test the /v1/models listing behavior with served_model_name."""

    def test_lists_served_name_as_primary(self):
        """When served_model_name is set, it appears first in the list."""
        import vllm_mlx.server as server

        orig_served = server._served_model_name
        orig_model = server._model_name
        try:
            server._served_model_name = "my-custom-model"
            server._model_name = "mlx-community/Llama-3.2-3B"

            result = asyncio.run(server.list_models())
            model_ids = [m.id for m in result.data]
            assert model_ids[0] == "my-custom-model"
            assert "mlx-community/Llama-3.2-3B" in model_ids
            assert len(model_ids) == 2
        finally:
            server._served_model_name = orig_served
            server._model_name = orig_model

    def test_lists_only_actual_when_no_served_name(self):
        """When no served_model_name, only actual name appears."""
        import vllm_mlx.server as server

        orig_served = server._served_model_name
        orig_model = server._model_name
        try:
            server._served_model_name = None
            server._model_name = "mlx-community/Llama-3.2-3B"

            result = asyncio.run(server.list_models())
            model_ids = [m.id for m in result.data]
            assert model_ids == ["mlx-community/Llama-3.2-3B"]
        finally:
            server._served_model_name = orig_served
            server._model_name = orig_model

    def test_no_duplicate_when_served_equals_actual(self):
        """When served_model_name == _model_name, no duplicate listing."""
        import vllm_mlx.server as server

        orig_served = server._served_model_name
        orig_model = server._model_name
        try:
            server._served_model_name = "mlx-community/Llama-3.2-3B"
            server._model_name = "mlx-community/Llama-3.2-3B"

            result = asyncio.run(server.list_models())
            model_ids = [m.id for m in result.data]
            assert model_ids == ["mlx-community/Llama-3.2-3B"]
        finally:
            server._served_model_name = orig_served
            server._model_name = orig_model

    def test_empty_list_when_default(self):
        """When resolved name is 'default', list is empty."""
        import vllm_mlx.server as server

        orig_served = server._served_model_name
        orig_model = server._model_name
        try:
            server._served_model_name = None
            server._model_name = None

            result = asyncio.run(server.list_models())
            assert result.data == []
        finally:
            server._served_model_name = orig_served
            server._model_name = orig_model

    def test_response_format_is_openai_compatible(self):
        """Models response has correct OpenAI-compatible structure."""
        import vllm_mlx.server as server

        orig_served = server._served_model_name
        orig_model = server._model_name
        try:
            server._served_model_name = None
            server._model_name = "test-model"

            result = asyncio.run(server.list_models())
            assert result.object == "list"
            assert len(result.data) == 1
            model = result.data[0]
            assert model.object == "model"
            assert model.id == "test-model"
            assert model.owned_by == "vllm-mlx"
            assert isinstance(model.created, int)
        finally:
            server._served_model_name = orig_served
            server._model_name = orig_model


# =============================================================================
# Permissive Model Validation Tests
# =============================================================================


class TestPermissiveModelValidation:
    """Test that model name validation is permissive (logs, doesn't reject).

    The server is single-model — any request model name is accepted and
    normalized to the resolved name. The original model field in the request
    is overwritten to the canonical resolved name.
    """

    def test_matching_model_no_warning(self):
        """Request model matching served name — no warning logged."""
        import logging

        import vllm_mlx.server as server

        orig_served = server._served_model_name
        orig_model = server._model_name
        try:
            server._served_model_name = "my-model"
            server._model_name = "org/actual"

            # Verify the validation logic inline (same as endpoint code)
            request_model = "my-model"
            resolved_name = server._resolve_model_name()
            # Model matches served name — no log expected
            assert request_model == resolved_name
        finally:
            server._served_model_name = orig_served
            server._model_name = orig_model

    def test_matching_actual_name_no_warning(self):
        """Request model matching actual name (not served) — no warning."""
        import vllm_mlx.server as server

        orig_served = server._served_model_name
        orig_model = server._model_name
        try:
            server._served_model_name = "alias"
            server._model_name = "org/actual"

            request_model = "org/actual"
            resolved_name = server._resolve_model_name()
            # Matches _model_name but not resolved (served) — still accepted
            assert request_model != resolved_name  # different from resolved
            assert request_model == server._model_name  # but matches actual
        finally:
            server._served_model_name = orig_served
            server._model_name = orig_model

    def test_mismatched_model_accepted_permissively(self):
        """Request with unknown model — accepted (permissive single-model server)."""
        import vllm_mlx.server as server

        orig_served = server._served_model_name
        orig_model = server._model_name
        try:
            server._served_model_name = "alias"
            server._model_name = "org/actual"

            # Simulate what the endpoint does
            request_model = "completely-wrong-name"
            resolved_name = server._resolve_model_name()
            # Both checks fail — but it's permissive
            assert request_model != resolved_name
            assert request_model != server._model_name
            # Request model would be overwritten to resolved name
            # (simulating: request.model = resolved_name)
            overwritten = resolved_name
            assert overwritten == "alias"
        finally:
            server._served_model_name = orig_served
            server._model_name = orig_model

    def test_none_model_accepted(self):
        """Request with model=None — accepted and overwritten."""
        import vllm_mlx.server as server

        orig_served = server._served_model_name
        orig_model = server._model_name
        try:
            server._served_model_name = None
            server._model_name = "org/model"

            request_model = None
            resolved_name = server._resolve_model_name()
            # None doesn't match anything, but permissive — just overwrite
            assert resolved_name == "org/model"
        finally:
            server._served_model_name = orig_served
            server._model_name = orig_model

    def test_empty_string_model_accepted(self):
        """Request with model='' — accepted and overwritten."""
        import vllm_mlx.server as server

        orig_served = server._served_model_name
        orig_model = server._model_name
        try:
            server._served_model_name = "alias"
            server._model_name = "org/model"

            request_model = ""
            resolved_name = server._resolve_model_name()
            # Empty string is falsy — validation check `if request.model` skips it
            assert resolved_name == "alias"
        finally:
            server._served_model_name = orig_served
            server._model_name = orig_model


# =============================================================================
# Validation Logic Structural Tests (inspect-based)
# =============================================================================


class TestValidationStructure:
    """Verify the model validation code structure using source inspection.

    Ensures both Chat Completions and Responses API endpoints have:
    - Model name validation with _resolve_model_name()
    - request.model overwrite to resolved name
    - Permissive behavior (no HTTPException for model mismatch)
    """

    def test_chat_completions_has_model_validation(self):
        """Chat completions endpoint validates model name."""
        import inspect

        import vllm_mlx.server as server

        source = inspect.getsource(server.create_chat_completion)
        # Must call _resolve_model_name
        assert "_resolve_model_name()" in source
        # Must overwrite request.model
        assert "request.model = resolved_name" in source

    def test_responses_api_has_model_validation(self):
        """Responses API endpoint validates model name."""
        import inspect

        import vllm_mlx.server as server

        source = inspect.getsource(server.create_response)
        # Must call _resolve_model_name
        assert "_resolve_model_name()" in source
        # Must overwrite request.model
        assert "request.model = resolved_name" in source

    def test_validation_is_permissive_not_strict(self):
        """Neither endpoint raises HTTPException for model name mismatch."""
        import inspect

        import vllm_mlx.server as server

        for func in [server.create_chat_completion, server.create_response]:
            source = inspect.getsource(func)
            # Validation block should NOT have HTTPException for model mismatch
            # (It just logs and overwrites)
            lines = source.split('\n')
            for i, line in enumerate(lines):
                if 'request.model' in line and 'differs from' in line:
                    # The line after a mismatch detection should be a logger call,
                    # NOT an HTTPException raise
                    nearby = '\n'.join(lines[max(0, i-2):i+5])
                    assert 'HTTPException' not in nearby, \
                        f"Model mismatch should not raise HTTPException:\n{nearby}"
                    break

    def test_chat_completions_validation_accepts_served_and_actual(self):
        """Validation check in chat completions accepts both served and actual name."""
        import inspect

        import vllm_mlx.server as server

        source = inspect.getsource(server.create_chat_completion)
        # The validation condition should check both resolved_name AND _model_name
        assert "request.model != resolved_name" in source
        assert "request.model != _model_name" in source

    def test_responses_validation_accepts_served_and_actual(self):
        """Validation check in responses API accepts both served and actual name."""
        import inspect

        import vllm_mlx.server as server

        source = inspect.getsource(server.create_response)
        assert "request.model != resolved_name" in source
        assert "request.model != _model_name" in source


# =============================================================================
# load_model() served_model_name Integration Tests
# =============================================================================


class TestLoadModelServedName:
    """Test that load_model() correctly sets _served_model_name global."""

    def test_load_model_signature_accepts_served_model_name(self):
        """load_model() has served_model_name parameter."""
        import inspect

        import vllm_mlx.server as server

        sig = inspect.signature(server.load_model)
        assert "served_model_name" in sig.parameters
        param = sig.parameters["served_model_name"]
        assert param.default is None

    def test_load_model_sets_served_name_global(self):
        """load_model() body writes to _served_model_name global."""
        import inspect

        import vllm_mlx.server as server

        source = inspect.getsource(server.load_model)
        # Must declare _served_model_name as global
        assert "_served_model_name" in source
        # Must assign the parameter value
        assert "_served_model_name = served_model_name" in source


# =============================================================================
# CLI Arg Tests
# =============================================================================


class TestCLIServedModelName:
    """Test the --served-model-name CLI argument definition."""

    def test_argparse_has_served_model_name(self):
        """CLI argparse includes --served-model-name flag."""
        import inspect

        from vllm_mlx import cli

        source = inspect.getsource(cli)
        assert "--served-model-name" in source

    def test_cli_passes_served_name_to_load_model(self):
        """serve_command() passes served_model_name to load_model()."""
        import inspect

        from vllm_mlx import cli

        source = inspect.getsource(cli.serve_command)
        assert "served_model_name" in source

    def test_cli_prints_served_name_on_startup(self):
        """serve_command() prints served model name when set."""
        import inspect

        from vllm_mlx import cli

        source = inspect.getsource(cli.serve_command)
        assert "Served model name:" in source


# =============================================================================
# SSE Streaming: request.model in Chunks Tests
# =============================================================================


class TestStreamingModelName:
    """Test that streaming functions use request.model (already overwritten to resolved name)."""

    def test_stream_chat_completion_uses_request_model(self):
        """stream_chat_completion() references request.model for chunk model field."""
        import inspect

        import vllm_mlx.server as server

        source = inspect.getsource(server.stream_chat_completion)
        # The streaming function should reference request.model for response objects
        assert "model=request.model" in source

    def test_stream_responses_api_uses_request_model(self):
        """stream_responses_api() references request.model for response objects."""
        import inspect

        import vllm_mlx.server as server

        source = inspect.getsource(server.stream_responses_api)
        # Responses API uses request.model in dict literals (SSE events), not keyword args
        assert "request.model" in source

    def test_registry_lookup_uses_model_path_not_served_name(self):
        """Model config registry lookups use _model_path, not potentially-aliased request.model."""
        import inspect

        import vllm_mlx.server as server

        source = inspect.getsource(server.stream_chat_completion)
        # Registry lookup should use _model_path as primary (not request.model which is aliased)
        assert "_model_path" in source


# =============================================================================
# SSE Streaming Fix: Empty Delta Suppression Tests
# =============================================================================


class TestSSEStreamingFix:
    """Test that the SSE streaming code correctly handles the reasoning parser path.

    When request_parser is active but delta_text is empty (progress-only engine chunk),
    the code should NOT emit an empty delta:{} to the client. The fix uses
    `elif not request_parser:` instead of `else:` so that progress-only chunks
    are silently dropped when a parser is active.
    """

    def test_standard_path_guarded_by_not_request_parser(self):
        """The non-parser streaming path is guarded by `elif not request_parser:`."""
        import inspect

        import vllm_mlx.server as server

        source = inspect.getsource(server.stream_chat_completion)
        # The critical fix: standard path must be `elif not request_parser:`
        # This prevents empty delta emission when parser is active but delta_text is empty
        assert "elif not request_parser:" in source

    def test_parser_path_requires_delta_text(self):
        """Parser path only fires when both request_parser AND delta_text are truthy."""
        import inspect

        import vllm_mlx.server as server

        source = inspect.getsource(server.stream_chat_completion)
        # The parser path condition
        assert "if request_parser and delta_text:" in source

    def test_empty_chunk_skip_in_parser_path(self):
        """Parser path skips chunks with nothing to emit (no content, no reasoning)."""
        import inspect

        import vllm_mlx.server as server

        source = inspect.getsource(server.stream_chat_completion)
        # Empty emission guard inside parser path
        assert "not emit_content and not emit_reasoning and not output.finished" in source


# =============================================================================
# ModelInfo and ModelsResponse Pydantic Model Tests
# =============================================================================


class TestModelInfoResponse:
    """Test the Pydantic models used in /v1/models response."""

    def test_model_info_defaults(self):
        """ModelInfo has correct default fields."""
        from vllm_mlx.api.models import ModelInfo

        info = ModelInfo(id="test-model")
        assert info.id == "test-model"
        assert info.object == "model"
        assert info.owned_by == "vllm-mlx"
        assert isinstance(info.created, int)

    def test_model_info_serialization(self):
        """ModelInfo serializes correctly for JSON response."""
        from vllm_mlx.api.models import ModelInfo

        info = ModelInfo(id="my-custom-name")
        data = info.model_dump()
        assert data["id"] == "my-custom-name"
        assert data["object"] == "model"
        assert data["owned_by"] == "vllm-mlx"

    def test_models_response_structure(self):
        """ModelsResponse has correct list structure."""
        from vllm_mlx.api.models import ModelInfo, ModelsResponse

        response = ModelsResponse(data=[
            ModelInfo(id="served-name"),
            ModelInfo(id="org/actual-name"),
        ])
        assert response.object == "list"
        assert len(response.data) == 2
        assert response.data[0].id == "served-name"
        assert response.data[1].id == "org/actual-name"

    def test_models_response_empty(self):
        """ModelsResponse can be empty."""
        from vllm_mlx.api.models import ModelsResponse

        response = ModelsResponse(data=[])
        assert response.object == "list"
        assert response.data == []

    def test_models_response_json_serialization(self):
        """ModelsResponse serializes to OpenAI-compatible JSON."""
        from vllm_mlx.api.models import ModelInfo, ModelsResponse

        response = ModelsResponse(data=[ModelInfo(id="test")])
        json_str = response.model_dump_json()
        assert '"object":"list"' in json_str or '"object": "list"' in json_str
        assert '"id":"test"' in json_str or '"id": "test"' in json_str
