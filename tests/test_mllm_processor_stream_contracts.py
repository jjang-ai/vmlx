# SPDX-License-Identifier: Apache-2.0
"""Contracts for VLM processor and simple-stream compatibility fixes."""

from pathlib import Path


def test_processor_direct_bypasses_noncallable_process_attr():
    from vmlx_engine.mllm_batch_generator import _call_processor_direct

    class _Processor:
        process = object()

        def __call__(self, **kwargs):
            return {
                "input_ids": [1, 2, 3],
                "images": kwargs.get("images"),
                "text": kwargs.get("text"),
            }

    out = _call_processor_direct(
        _Processor(),
        prompts="look",
        images=["/tmp/image.png"],
        add_special_tokens=False,
    )
    assert out["input_ids"] == [1, 2, 3]
    assert out["images"] == ["/tmp/image.png"]
    assert out["text"] == "look"


def test_simple_mllm_stream_generate_runs_inside_stream_context():
    source = Path("./vmlx_engine/models/mllm.py").read_text()
    stream_generate_idx = source.index("def stream_generate(")
    stream_chat_idx = source.index("def stream_chat(")
    first_stream_body = source[stream_generate_idx:stream_chat_idx]
    chat_body = source[stream_chat_idx: source.index("def describe_image(", stream_chat_idx)]

    assert "with _MaybeVLMStream():" in first_stream_body
    assert "for chunk in stream_generate(" in first_stream_body
    assert "with _MaybeVLMStream():" in chat_body
    assert "RuntimeError: There is no Stream" in source
