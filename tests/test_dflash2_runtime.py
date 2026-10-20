from __future__ import annotations

import inspect
import json
from types import SimpleNamespace
from unittest.mock import patch


def test_dflash2_effective_width_comes_from_checkpoint_config(tmp_path):
    from vmlx_engine.speculative import resolve_num_draft_tokens

    (tmp_path / "config.json").write_text(json.dumps({
        "architectures": ["DFlash2DraftModel"],
        "block_size": 5,
    }))

    assert resolve_num_draft_tokens(str(tmp_path), requested=3) == 5


def test_standard_draft_preserves_requested_width():
    from vmlx_engine.speculative import resolve_num_draft_tokens

    assert resolve_num_draft_tokens("ordinary-draft", requested=3) == 3


def test_custom_drafter_health_does_not_report_configured_count_as_live_depth():
    import vmlx_engine.speculative as spec

    # Both DFlash2 and MiMo DFlash use this serving kind, regardless of the
    # legacy value held in SpeculativeConfig.
    for configured in (5, 8):
        config = spec.SpeculativeConfig(model="bundled-drafter", num_tokens=configured)
        with patch.object(spec, "_spec_config", config), \
             patch.object(spec, "_spec_kind", "dflash2"), \
             patch.object(spec, "_draft_model", object()):
            stats = spec.get_spec_stats()["speculative_decoding"]
            assert stats["enabled"]
            assert stats["num_draft_tokens"] is None
            assert stats["num_draft_tokens_scope"] == "runtime_selected"
            assert stats["configured_num_draft_tokens"] == configured
            assert spec.get_num_draft_tokens() == configured


def test_standard_drafter_health_identifies_fixed_count():
    import vmlx_engine.speculative as spec

    config = spec.SpeculativeConfig(model="external-drafter", num_tokens=4)
    with patch.object(spec, "_spec_config", config), \
         patch.object(spec, "_spec_kind", "standard"), \
         patch.object(spec, "_draft_model", object()):
        stats = spec.get_spec_stats()["speculative_decoding"]
        assert stats["num_draft_tokens"] == 4
        assert stats["num_draft_tokens_scope"] == "fixed"


def test_simple_engine_counts_multi_token_mllm_chunks_cumulatively():
    from vmlx_engine.engine.simple import _advance_mllm_completion_tokens

    assert _advance_mllm_completion_tokens(
        1, SimpleNamespace(text="four tokens", completion_tokens=5)
    ) == 5
    assert _advance_mllm_completion_tokens(
        5, SimpleNamespace(text="", completion_tokens=5)
    ) == 5
    assert _advance_mllm_completion_tokens(
        5, SimpleNamespace(text="legacy", completion_tokens=0)
    ) == 6


def test_qwen_mtp_model_wrapper_accepts_upstream_capture_contract():
    from vmlx_engine.patches.mlx_vlm_mtp.qwen35_vl import _patch_qwen_model

    class FakeQwenModel:
        def __call__(
            self,
            inputs,
            inputs_embeds=None,
            mask=None,
            cache=None,
            position_ids=None,
        ):
            return "original"

    qlang = SimpleNamespace(Qwen3_5Model=FakeQwenModel)
    _patch_qwen_model(qlang)

    signature = inspect.signature(FakeQwenModel.__call__)
    assert "capture_layer_ids" in signature.parameters
    assert "hidden_sink" in signature.parameters
    assert FakeQwenModel()(None, capture_layer_ids=None, hidden_sink=None) == "original"


def test_stream_chat_routes_text_only_dflash2_before_mlx_vlm_generator():
    from concurrent.futures import Future
    from vmlx_engine.models.mllm import MLXMultimodalLM

    model = object.__new__(MLXMultimodalLM)
    model._loaded = True
    model.model_name = "target"
    model.model = object()
    model.processor = SimpleNamespace(tokenizer=object())
    model.config = {}
    model._extract_multimodal_messages = lambda _messages: (
        [{"role": "user", "content": "hello"}],
        [],
        [],
        [],
    )
    model._normalize_text_only_messages_for_processor = (
        lambda messages, has_media: messages
    )
    model._apply_chat_template = lambda messages, enable_thinking, tools=None, reasoning_effort=None: "PROMPT"

    receipt = Future()
    chunks = [
        SimpleNamespace(
            text="one",
            tokens=[1],
            accepted=None,
            prompt_tokens=7,
            generation_tps=0.0,
            finish_reason=None,
        ),
        SimpleNamespace(
            text=" two",
            tokens=[2, 3],
            accepted=2,
            prompt_tokens=7,
            generation_tps=55.0,
            finish_reason="stop",
            persistence_future=receipt,
        ),
    ]

    with (
        patch("vmlx_engine.speculative.is_dflash2_enabled", return_value=True),
        patch("vmlx_engine.speculative.get_draft_model", return_value=object()),
        patch(
            "vmlx_engine.dflash2_runtime.stream_dflash2_generate",
            return_value=iter(chunks),
        ) as bridge,
        patch("mlx_vlm.stream_generate") as ordinary,
    ):
        outputs = list(
            model.stream_chat(
                [{"role": "user", "content": "hello"}],
                max_tokens=10,
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                min_p=.1,
                logit_bias={"10": -20},
                repetition_penalty=1.1,
                frequency_penalty=.2,
                presence_penalty=.3,
                stop=["END"],
            )
        )

    bridge.assert_called_once()
    ordinary.assert_not_called()
    assert [output.text for output in outputs] == ["one", " two"]
    assert outputs[-1].completion_tokens == 3
    assert outputs[-1].finish_reason == "stop"
    assert outputs[-1].persistence_future is receipt
    assert bridge.call_args.kwargs["min_p"] == .1
    assert bridge.call_args.kwargs["logit_bias"] == {"10": -20}
    assert bridge.call_args.kwargs["repetition_penalty"] == 1.1
    assert bridge.call_args.kwargs["frequency_penalty"] == .2
    assert bridge.call_args.kwargs["presence_penalty"] == .3
    assert bridge.call_args.kwargs["stop"] == ["END"]


def test_target_adapter_media_plan_slices_prompt_and_offsets_text_rows():
    """Image/video prompts: prompt chunks take their slice of the VLM's merged
    embeddings and M-RoPE ids; every later row (decode, verify) is text at
    cache offset + rope_delta, through the 2-D text-RoPE path."""
    import mlx.core as mx

    from vmlx_engine.dflash2_runtime import _TargetAdapter

    calls = []

    class Inner:
        fa_idx = 1

        def __call__(self, inputs, cache=None, **kwargs):
            calls.append(kwargs)
            return mx.zeros((1, inputs.shape[1], 4))

        def norm(self, h):
            return h

    lm = SimpleNamespace(model=Inner(), lm_head=lambda h: h)
    adapter = _TargetAdapter(lm)
    cache = [None, SimpleNamespace(offset=0)]
    embeds = mx.arange(10 * 4).reshape(1, 10, 4)
    positions = mx.broadcast_to(mx.arange(10)[None, None], (3, 1, 10))
    adapter.media_embeds, adapter.media_positions, adapter.rope_delta = embeds, positions, -6

    cache[1].offset = 2
    adapter(mx.zeros((1, 3), dtype=mx.int32), cache)
    assert mx.array_equal(calls[-1]["inputs_embeds"], embeds[:, 2:5]).item()
    assert calls[-1]["position_ids"].shape == (3, 1, 3)

    cache[1].offset = 10  # past the prompt: verify rows
    adapter(mx.zeros((1, 4), dtype=mx.int32), cache)
    assert "inputs_embeds" not in calls[-1]
    assert calls[-1]["position_ids"].tolist() == [[4, 5, 6, 7]]

    adapter.media_embeds = adapter.media_positions = None
    adapter.rope_delta = 0
    adapter(mx.zeros((1, 2), dtype=mx.int32), cache)
    assert "position_ids" not in calls[-1] and "inputs_embeds" not in calls[-1]


def _media_chat_model(plan):
    from vmlx_engine.models.mllm import MLXMultimodalLM

    model = object.__new__(MLXMultimodalLM)
    model._loaded = True
    model.model_name = "target"
    model.model = object()
    model.processor = SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda s: [9]))
    model.config = {}
    model._extract_multimodal_messages = lambda _messages: (
        [{"role": "user", "content": "describe"}], ["img://1"], [], [],
    )
    model._normalize_text_only_messages_for_processor = lambda messages, has_media: messages
    model._prepare_images = lambda urls: ["IMAGE"]
    model._apply_chat_template = lambda messages, enable_thinking, tools=None, reasoning_effort=None: "PROMPT"
    model._dflash2_media_plan = lambda prompt, images, kwargs: plan
    model.guard_calls = []
    model._guard_simple_image_prefill = lambda prompt, has_images, **kw: model.guard_calls.append(has_images)
    return model


def test_stream_chat_routes_media_dflash2_through_vlm_prefill_plan():
    plan = {"prompt_tokens": [1, 2, 3], "embeds": "E", "positions": "P", "rope_delta": -3,
            "first_media_index": 1, "salt": "abc"}
    model = _media_chat_model(plan)
    chunk = SimpleNamespace(text="cat", tokens=[5], accepted=None, prompt_tokens=3,
                            generation_tps=0.0, finish_reason="stop",
                            prefill_usage={"tokens": 3, "seconds": 0.5, "scope": "model_prefill_and_prompt_state", "path": "dflash2"})
    with (
        patch("vmlx_engine.speculative.is_dflash2_enabled", return_value=True),
        patch("vmlx_engine.speculative.get_draft_model", return_value=object()),
        patch("vmlx_engine.dflash2_runtime.stream_dflash2_generate", return_value=iter([chunk])) as bridge,
        patch("mlx_vlm.stream_generate") as ordinary,
    ):
        outputs = list(model.stream_chat([{"role": "user", "content": "describe"}], max_tokens=4, temperature=0.0))
    bridge.assert_called_once()
    ordinary.assert_not_called()
    assert bridge.call_args.kwargs["media"] is plan
    assert bridge.call_args.kwargs["prompt_tokens"] == [1, 2, 3]
    assert outputs[-1].text == "cat"
    assert model.guard_calls == [True]
    # pp/s: the engine-measured prefill reaches the output (-> usage.vmlx_prefill)
    assert outputs[-1].prefill_usage == {"tokens": 3, "seconds": 0.5, "scope": "model_prefill_and_prompt_state", "path": "dflash2"}  # the native path's image-prefill guard also runs on the DFlash2 media route


def test_stream_chat_media_without_vlm_prefill_plan_keeps_native_path():
    model = _media_chat_model(None)
    with (
        patch("vmlx_engine.speculative.is_dflash2_enabled", return_value=True),
        patch("vmlx_engine.speculative.get_draft_model", return_value=object()),
        patch("vmlx_engine.dflash2_runtime.stream_dflash2_generate") as bridge,
        patch("mlx_vlm.stream_generate", return_value=iter([])) as ordinary,
    ):
        try:
            list(model.stream_chat([{"role": "user", "content": "describe"}], max_tokens=4, temperature=0.0))
        except Exception:
            pass  # the stub model cannot run the native path; only the routing matters
    bridge.assert_not_called()


def test_native_vlm_stream_counts_cumulative_generation_tokens_not_chunks():
    """mlx-vlm ends with a flush result repeating the last generation_tokens: max_tokens=2 must report 2, not 3."""
    model = _media_chat_model(None)
    results = [
        SimpleNamespace(text="a", generation_tokens=1, prompt_tokens=4),
        SimpleNamespace(text="b", generation_tokens=2, prompt_tokens=4),
        SimpleNamespace(text="", generation_tokens=2, prompt_tokens=4),  # finalize flush
    ]
    model._extract_multimodal_messages = lambda _messages: ([{"role": "user", "content": "hi"}], [], [], [])
    model._guard_simple_image_prefill = lambda *a, **k: None
    model.processor = SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda s: [9]), _patched_detok=True)
    model._cache_manager = None
    with (
        patch("vmlx_engine.speculative.is_dflash2_enabled", return_value=False),
        patch("mlx_vlm.stream_generate", return_value=iter(results)),
        patch("mlx_vlm.models.cache.make_prompt_cache", return_value=None),
    ):
        outputs = list(model.stream_chat([{"role": "user", "content": "hi"}], max_tokens=2, temperature=0.0, use_cache=False))
    assert outputs[-1].completion_tokens == 2
    assert outputs[-1].finish_reason == "length"


def test_clone_cache_shells_does_not_alias_arrays_cache_lists():
    """A snapshot must not see later in-place GDN state writes on the live cache."""
    import mlx.core as mx
    from mlx_lm.models.cache import ArraysCache, KVCache

    from vmlx_engine.dflash2_runtime import _clone_cache_shells

    live = [ArraysCache(2), KVCache()]
    live[0][0] = mx.zeros((1, 3, 4)); live[0][1] = mx.zeros((1, 2, 4, 4))
    live[1].update_and_fetch(mx.zeros((1, 2, 5, 8)), mx.zeros((1, 2, 5, 8)))
    snap = _clone_cache_shells(live, lambda: [ArraysCache(2), KVCache()])
    live[0][1] = mx.ones((1, 2, 4, 4))  # in-place GDN update after the snapshot
    live[1].update_and_fetch(mx.ones((1, 2, 3, 8)), mx.ones((1, 2, 3, 8)))
    assert snap[0].state is not live[0].state
    assert mx.array_equal(snap[0][1], mx.zeros((1, 2, 4, 4))).item()
    assert snap[1].offset == 5


def test_text_request_clears_a_stale_media_plan_on_the_shared_adapter():
    """An abandoned media generator may not have run its finally: the next request must reset the plan at entry."""
    import inspect
    from vmlx_engine.dflash2_runtime import _stream_generate_resumable
    src = inspect.getsource(_stream_generate_resumable)
    entry = src.index("adapter.media_embeds = media[\"embeds\"] if media is not None else None")
    assert entry < src.index("    try:\n        tic = time.perf_counter()")


def test_dflash2_media_key_tokens_are_per_item_and_position_aware():
    import mlx.core as mx
    from vmlx_engine.models.mllm import _dflash2_media_key_tokens

    IMG = 9
    run = [IMG] * 6
    red, blue = mx.ones((4, 3)), mx.full((4, 3), 2.0)
    grid1 = mx.array([[1, 2, 2]])
    k_red = _dflash2_media_key_tokens([1, 2, *run, 3], {IMG}, grid1, red)
    k_blue = _dflash2_media_key_tokens([1, 2, *run, 3], {IMG}, grid1, blue)
    assert k_red[:2] == [1, 2] and k_red[-1] == 3
    assert all(t >= 0x80000000 for t in k_red[2:8])
    assert len(set(k_red[2:8])) > 1                    # per-position ids (30 bits each), not one 30-bit id
    assert k_red[2:8] != k_blue[2:8]                   # different image -> different key
    two = _dflash2_media_key_tokens([1, 2, *run, 3, *run], {IMG}, mx.array([[1, 2, 2], [1, 2, 2]]),
                                    mx.concatenate([red, blue]))
    assert two[:9] == k_red and two[9:] == k_blue[2:8]  # red prefix keys unchanged; new blue item diverges
    assert _dflash2_media_key_tokens([1, IMG, 2, IMG], {IMG}, grid1, red) is None       # unpairable -> fail closed
    assert _dflash2_media_key_tokens([1, IMG, IMG, 3], {IMG}, grid1, red) is None       # run < 5 tokens -> fail closed


def test_video_frames_take_their_prompt_position_among_images():
    from vmlx_engine.models.mllm import _order_images_and_video_frames

    msgs = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "a"}]},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": [{"type": "video"}, {"type": "text", "text": "b"}]},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "c"}]},
    ]
    assert _order_images_and_video_frames(msgs, ["shot", "blue"], [["f1", "f2"]]) == ["shot", "f1", "f2", "blue"]
    assert _order_images_and_video_frames(msgs, ["shot"], [["f1"]]) is None  # mismatch fails closed


def test_step3p7_language_model_accepts_mlx_vlm_inputs_keyword():
    import inspect
    from vmlx_engine.models.step3p7_mlx_vlm import LanguageModel

    params = inspect.signature(LanguageModel.__call__).parameters
    assert "inputs" in params and params["input_ids"].default is None


def test_dflash2_prefill_receipt_uses_the_panel_pp_scope():
    """panel shared/chatMetrics.ts calculatePrefillTps accepts only this scope; anything else hides pp/s."""
    import inspect
    from vmlx_engine.dflash2_runtime import _stream_generate_resumable
    assert '"scope": "model_prefill_and_prompt_state"' in inspect.getsource(_stream_generate_resumable)


def test_media_order_mismatch_stops_both_chat_routes_before_generation():
    import pytest

    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "video"}, {"type": "image"},
    ]}]
    for method in ("chat", "stream_chat"):
        model = _media_chat_model(None)
        model.config = {"model_type": "qwen3_5"}
        # One failed image has been omitted by preprocessing. Do not shift
        # the surviving image/video into another input's placeholder.
        model._extract_multimodal_messages = lambda _: (messages, ["bad", "good"], ["clip"], [])
        model._prepare_images = lambda _: ["good"]
        model._prepare_video = lambda *a, **kw: ["frame"]
        with pytest.raises(ValueError, match="media items.*prompt order"):
            result = getattr(model, method)(messages)
            if method == "stream_chat":
                list(result)
