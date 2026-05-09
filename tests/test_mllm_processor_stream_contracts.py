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


class TestProcessorRoutingDecision:
    """vmlx#145 hardening: route around mlx_vlm.process_inputs's TokenizerWrapper trap.

    Coverage matrix — (has_image_literal × has_images × processor shape):
      A. images + literal + callable .process       → prepare_inputs (fast path)
      B. images + literal + non-callable .process   → safe path (vmlx#145 fix v1.5.26)
      C. images + no literal                        → safe path (Gemma 4 etc.)
      D. images + literal + no .process attr +
         processor itself not callable              → safe path (this hardening)
      E. images + literal + no .process attr +
         processor itself IS callable               → prepare_inputs (works)
      F. no images                                  → prepare_inputs (text-only)
    """

    def test_case_A_callable_process_with_literal_uses_prepare_inputs(self):
        from vmlx_engine.mllm_batch_generator import _should_use_safe_processor_path

        class _Processor:
            def process(self, *a, **k):
                return {}

            def __call__(self, **k):
                return {}

        assert _should_use_safe_processor_path(
            _Processor(), has_image_literal=True, has_images=True
        ) is False

    def test_case_B_noncallable_process_with_literal_uses_safe_path(self):
        from vmlx_engine.mllm_batch_generator import _should_use_safe_processor_path

        class _Processor:
            process = object()  # non-callable sentinel — TokenizerWrapper-style

            def __call__(self, **k):
                return {}

        assert _should_use_safe_processor_path(
            _Processor(), has_image_literal=True, has_images=True
        ) is True

    def test_case_C_no_literal_uses_safe_path(self):
        from vmlx_engine.mllm_batch_generator import _should_use_safe_processor_path

        class _Processor:
            def process(self, *a, **k):
                return {}

            def __call__(self, **k):
                return {}

        assert _should_use_safe_processor_path(
            _Processor(), has_image_literal=False, has_images=True
        ) is True

    def test_case_D_missing_process_attr_uncallable_processor_uses_safe_path(self):
        """The Case D hole this hardening closes — TokenizerWrapper-only bundles."""
        from vmlx_engine.mllm_batch_generator import _should_use_safe_processor_path

        class _BareTokenizerWrapper:
            # No .process attribute at all, and not callable.
            pass

        assert _should_use_safe_processor_path(
            _BareTokenizerWrapper(), has_image_literal=True, has_images=True
        ) is True

    def test_case_E_missing_process_attr_callable_processor_uses_prepare_inputs(self):
        from vmlx_engine.mllm_batch_generator import _should_use_safe_processor_path

        class _Processor:
            def __call__(self, **k):
                return {}

        assert _should_use_safe_processor_path(
            _Processor(), has_image_literal=True, has_images=True
        ) is False

    def test_case_F_no_images_uses_prepare_inputs(self):
        from vmlx_engine.mllm_batch_generator import _should_use_safe_processor_path

        class _Processor:
            def __call__(self, **k):
                return {}

        assert _should_use_safe_processor_path(
            _Processor(), has_image_literal=False, has_images=False
        ) is False

    def test_routing_call_site_uses_helper(self):
        """Pin: _preprocess_request actually calls the helper, not inline logic."""
        source = Path("./vmlx_engine/mllm_batch_generator.py").read_text()
        assert "_should_use_safe_processor_path(" in source
        leaky = 'hasattr(self.processor, "process") and not callable(_process_attr)'
        assert leaky not in source, (
            "Routing condition still has Case D hole. The hasattr/callable inline "
            "expression should be replaced by _should_use_safe_processor_path()."
        )


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
