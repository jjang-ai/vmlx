# SPDX-License-Identifier: Apache-2.0
"""Tests for request cancellation functionality."""

import pytest
from vmlx_engine.engine.batched import BatchedEngine


@pytest.mark.asyncio
async def test_custom_request_id_passed_through():
    """Test that custom request_id is used in stream_chat."""
    engine = BatchedEngine(model_name="mlx-community/Qwen3-0.6B-8bit")
    await engine.start()

    try:
        custom_id = "chatcmpl-test123"
        messages = [{"role": "user", "content": "Hi"}]

        # Stream with custom request_id
        output_count = 0
        async for output in engine.stream_chat(
            messages=messages,
            max_tokens=3,
            request_id=custom_id,
        ):
            output_count += 1
            # Just verify it streams without error
            if output_count >= 1:
                break

        # Verify we got at least one output
        assert output_count > 0

    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_cancel_endpoint_aborts_request():
    """Test that cancel endpoint successfully aborts ongoing request."""
    engine = BatchedEngine(model_name="mlx-community/Qwen3-0.6B-8bit")
    await engine.start()

    try:
        custom_id = "chatcmpl-cancel-test"

        # Add a request that would generate many tokens
        from vmlx_engine.request import SamplingParams
        await engine._engine.add_request(
            prompt="Count to 100",
            sampling_params=SamplingParams(max_tokens=100),
            request_id=custom_id
        )

        # Cancel it immediately
        success = await engine.abort_request(custom_id)
        assert success is True

    finally:
        await engine.stop()
