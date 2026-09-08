"""vmlx#254: the auto prompt cap over-charged interval-hybrid families ~4x.

Qwen3.5/3.6 stacks keep standard K+V only on every Nth layer
(`full_attention_interval`); the other layers hold a fixed-size recurrent
state that does not grow with context. The generic estimator charged every
layer as full attention, so the derived cap landed at ~20k tokens on a box
that fits far more — the reporter saw "tokenized VLM text prompt has 20771
tokens, max prompt/context tokens is 20119".

Same shape as the existing dots3_note and deepseek_v4 special cases.
"""

from vmlx_engine.utils.memory_limits import (
    estimate_kv_bytes_per_token_from_config as est,
)

_COMMON = {"num_key_value_heads": 4, "head_dim": 256, "torch_dtype": "bfloat16"}


def _cfg(**kw):
    return {"text_config": {**_COMMON, **kw}}


def test_interval_hybrid_charges_only_full_attention_layers():
    hybrid = est(_cfg(model_type="qwen3_5", num_hidden_layers=48, full_attention_interval=4))
    # 12 full-attention layers x 2 (K+V) x 4 heads x 256 dim x 2 bytes
    assert hybrid == 12 * 2 * 4 * 256 * 2


def test_dense_model_is_unchanged():
    dense = est(_cfg(model_type="qwen3", num_hidden_layers=48))
    assert dense == 48 * 2 * 4 * 256 * 2


def test_hybrid_is_four_times_cheaper_than_the_old_all_layer_charge():
    hybrid = est(_cfg(model_type="qwen3_5", num_hidden_layers=48, full_attention_interval=4))
    dense = est(_cfg(model_type="qwen3", num_hidden_layers=48))
    assert dense == 4 * hybrid


def test_explicit_layer_types_win_over_interval():
    cfg = _cfg(
        model_type="qwen3_5",
        num_hidden_layers=8,
        layer_types=["linear_attention"] * 6 + ["full_attention"] * 2,
    )
    assert est(cfg) == 2 * 2 * 4 * 256 * 2


def test_interval_of_one_is_not_hybrid():
    # interval 1 means every layer is full attention: must not take the
    # hybrid branch and must match the dense charge.
    cfg = _cfg(model_type="qwen3_5", num_hidden_layers=48, full_attention_interval=1)
    assert est(cfg) == 48 * 2 * 4 * 256 * 2


def test_all_full_layer_types_fall_through_to_generic():
    cfg = _cfg(
        model_type="qwen3_5",
        num_hidden_layers=4,
        layer_types=["full_attention"] * 4,
    )
    assert est(cfg) == 4 * 2 * 4 * 256 * 2


def test_head_dim_derived_from_hidden_size_when_absent():
    cfg = {
        "text_config": {
            "model_type": "qwen3_5",
            "num_hidden_layers": 48,
            "full_attention_interval": 4,
            "num_key_value_heads": 4,
            "num_attention_heads": 8,
            "hidden_size": 2048,
            "torch_dtype": "bfloat16",
        }
    }
    assert est(cfg) == 12 * 2 * 4 * (2048 // 8) * 2


def test_dots3_note_special_case_still_wins():
    cfg = {
        "text_config": {
            "model_type": "dots3_note",
            "num_hidden_layers": 46,
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64,
            "index_head_dim": 128,
            "layer_types": ["sliding_attention"] * 33 + ["full_attention"] * 13,
            "torch_dtype": "bfloat16",
        }
    }
    assert est(cfg) == 13 * (512 + 64 + 128) * 2


def test_garbage_config_does_not_crash():
    assert est({"text_config": {"model_type": "qwen3_5", "full_attention_interval": "nope"}}) >= 0
    assert est({}) >= 0


def test_prompt_too_long_sources_name_the_prompt_not_the_engine_lane():
    """The typed 413 says what was measured. "VLM text prompt" read as a video/image
    budget error to a reader of the app's rejection bubble (Codex review 2026-09-07);
    the MLLM lane now names the text prompt and, for media, says the media tokens
    are included — no engine-lane jargon in a user-facing message."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "vmlx_engine"
    sources = [
        s for s in re.findall(r'source="([^"]+)"', (root / "mllm_scheduler.py").read_text() + (root / "mllm_batch_generator.py").read_text())
        if "prompt" in s
    ]
    assert sources, "the MLLM lane raises PromptTooLongError with a named source"
    assert all("VLM" not in s for s in sources), sources
    assert "text prompt (tokenized)" in sources
    assert any("media tokens included" in s for s in sources)
