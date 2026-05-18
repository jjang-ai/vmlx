# SPDX-License-Identifier: Apache-2.0
"""Contracts for VLM processor and simple-stream compatibility fixes."""

from pathlib import Path


def test_gemma4_processor_accepts_tiny_rgb_image():
    """Gemma4 image preprocessing must not misread 1x1 RGB as channel-first."""
    import vmlx_engine  # noqa: F401 - installs runtime patches
    from PIL import Image
    from mlx_vlm.models.gemma4.processing_gemma4 import Gemma4ImageProcessor

    processor = Gemma4ImageProcessor()
    data, soft_tokens = processor([Image.new("RGB", (1, 1), (255, 0, 0))])

    pixel_values = data["pixel_values"]
    assert pixel_values.shape[0] == 1
    assert pixel_values.shape[1] == 3
    assert pixel_values.shape[-2] > 0
    assert pixel_values.shape[-1] > 0
    assert len(soft_tokens) == 1
    assert soft_tokens[0] > 0


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


class TestHybridOomGuardWrapperTraversal:
    """OOM-guard num_attention_heads detection must walk VLM wrappers.

    The hybrid auto-chunk decision in MLLMBatchGenerator predicts attention
    matmul bytes as ``n_heads * seq_len^2 * 2`` and chunks if above 8 GB.
    For VLM-wrapped models (Kimi K2.6, Mistral 4 wrappers, glm_moe_dsa),
    ``language_model.config`` / ``.args`` may not expose
    ``num_attention_heads`` directly — the wrapper exposes it via
    ``text_config`` or via an inner ``.model`` candidate. The earlier
    inline check fell back to the 32-head default and made the chunking
    decision against the wrong attention shape.
    """

    def test_detects_kimi_style_config_text_config_heads(self):
        from types import SimpleNamespace
        from vmlx_engine.mllm_batch_generator import (
            _infer_attention_heads_for_hybrid_oom_guard,
        )

        # Local Kimi-K2.6 config shape: top-level VLM config has no
        # num_attention_heads, text_config is the DeepseekV3 text backbone.
        language_model = SimpleNamespace(
            config=SimpleNamespace(
                model_type="kimi_k25",
                text_config=SimpleNamespace(
                    model_type="kimi_k2",
                    num_attention_heads=64,
                ),
            )
        )

        assert _infer_attention_heads_for_hybrid_oom_guard(language_model) == 64

    def test_detects_inner_model_config_heads(self):
        from types import SimpleNamespace
        from vmlx_engine.mllm_batch_generator import (
            _infer_attention_heads_for_hybrid_oom_guard,
        )

        language_model = SimpleNamespace(
            config=SimpleNamespace(model_type="wrapper"),
            model=SimpleNamespace(
                config=SimpleNamespace(
                    model_type="inner_text",
                    num_attention_heads=80,
                )
            ),
        )

        assert _infer_attention_heads_for_hybrid_oom_guard(language_model) == 80

    def test_valid_32_head_config_does_not_fall_through_to_inner_default_sentinel(self):
        from types import SimpleNamespace
        from vmlx_engine.mllm_batch_generator import (
            _infer_attention_heads_for_hybrid_oom_guard,
        )

        language_model = SimpleNamespace(
            config=SimpleNamespace(model_type="text", num_attention_heads=32),
            model=SimpleNamespace(
                config=SimpleNamespace(model_type="nested", num_attention_heads=64)
            ),
        )

        assert _infer_attention_heads_for_hybrid_oom_guard(language_model) == 32

    def test_dict_text_config_shape_is_supported(self):
        from types import SimpleNamespace
        from vmlx_engine.mllm_batch_generator import (
            _infer_attention_heads_for_hybrid_oom_guard,
        )

        language_model = SimpleNamespace(
            config={
                "model_type": "kimi_k25",
                "text_config": {
                    "model_type": "kimi_k2",
                    "num_attention_heads": 64,
                },
            }
        )

        assert _infer_attention_heads_for_hybrid_oom_guard(language_model) == 64

    def test_oom_guard_call_sites_use_shared_helper(self):
        source = Path("./vmlx_engine/mllm_batch_generator.py").read_text()

        assert source.count("_infer_attention_heads_for_hybrid_oom_guard(") >= 3
        assert "cfg = getattr(self.language_model, \"config\", None) or getattr(\n" not in source


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
