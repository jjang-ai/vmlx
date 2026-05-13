"""MLLM generation stream lifecycle contracts."""

from __future__ import annotations

import inspect


def test_reset_generation_streams_clears_module_and_class_handles():
    """Deep sleep/reload must not keep thread-local MLX streams from old workers."""
    import vmlx_engine.mllm_batch_generator as gen

    old_module_stream = gen._GENERATION_STREAM
    old_class_stream = gen.MLLMBatchGenerator._stream
    try:
        gen._GENERATION_STREAM = object()
        gen.MLLMBatchGenerator._stream = object()

        gen.reset_generation_streams()

        assert gen._GENERATION_STREAM is None
        assert gen.MLLMBatchGenerator._stream is None
    finally:
        gen._GENERATION_STREAM = old_module_stream
        gen.MLLMBatchGenerator._stream = old_class_stream


def test_server_resets_mllm_streams_when_replacing_or_unloading_engine():
    """Model switch and deep sleep both tear down MLLM stream ownership."""
    import vmlx_engine.server as srv

    load_body = inspect.getsource(srv.load_model)
    deep_sleep_body = inspect.getsource(srv.admin_deep_sleep)

    assert "_reset_mllm_generation_streams()" in load_body
    assert "_reset_mllm_generation_streams()" in deep_sleep_body
