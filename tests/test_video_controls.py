"""Per-request video controls: validation, effective defaults, cache-key
identity, loader element, fallback sizing, and transport through every
API dialect and engine hop."""
import inspect
import math

import pytest

from vmlx_engine.video_controls import (
    MLX_VLM_VIDEO_MAX_PIXELS,
    VIDEO_CONTROL_FIELDS,
    VideoControls,
    bound_video_frames,
    pop_video_control_kwargs,
    validate_video_controls,
    video_control_kwargs,
    video_controls_from_kwargs,
)


class TestValidation:
    @pytest.mark.parametrize(
        "field, value",
        [
            ("video_fps", 0), ("video_fps", -1), ("video_fps", float("nan")), ("video_fps", float("inf")), ("video_fps", "2"), ("video_fps", True),
            ("video_max_frames", 0), ("video_max_frames", 2.5), ("video_max_frames", -3),
            ("video_max_pixels", 0), ("video_max_pixels", -1), ("video_max_pixels", 1.5),
            ("video_min_pixels", 0), ("video_total_pixels", 0),
            ("video_resized_height", 0), ("video_resized_width", -8),
        ],
    )
    def test_rejects_non_finite_zero_negative_and_non_integer(self, field, value):
        with pytest.raises(ValueError, match=field):
            validate_video_controls({field: value})

    def test_resized_dimensions_come_together(self):
        with pytest.raises(ValueError, match="together"):
            validate_video_controls({"video_resized_height": 224})
        with pytest.raises(ValueError, match="together"):
            validate_video_controls({"video_resized_width": 224})
        ok = validate_video_controls({"video_resized_height": 224, "video_resized_width": 336})
        assert (ok.resized_height, ok.resized_width) == (224, 336)

    def test_contradictions(self):
        with pytest.raises(ValueError, match="video_min_pixels must not exceed"):
            validate_video_controls({"video_min_pixels": 500_000, "video_max_pixels": 100_000})
        with pytest.raises(ValueError, match="exceeds video_max_pixels"):
            validate_video_controls({"video_resized_height": 1000, "video_resized_width": 1000, "video_max_pixels": 100_000})

    def test_accepts_and_normalizes(self):
        c = validate_video_controls({"video_fps": 4, "video_max_frames": 16.0, "video_max_pixels": 200_000})
        assert c == VideoControls(fps=4.0, max_frames=16, max_pixels=200_000)
        assert validate_video_controls({}).is_unset
        assert validate_video_controls(None).is_unset


class TestEffectiveAndKeys:
    def test_unset_equals_spelled_out_defaults(self):
        from vmlx_engine.models.mllm import DEFAULT_FPS, MAX_FRAMES
        unset = VideoControls()
        explicit = VideoControls(fps=DEFAULT_FPS, max_frames=MAX_FRAMES)
        assert unset.effective_fps() == DEFAULT_FPS and unset.effective_max_frames() == MAX_FRAMES
        assert unset.cache_key_fragment() == explicit.cache_key_fragment()

    def test_every_control_changes_the_key(self):
        base = VideoControls(fps=2.0, max_frames=8).cache_key_fragment()
        for change in (
            dict(fps=4.0, max_frames=8), dict(fps=2.0, max_frames=16), dict(fps=2.0, max_frames=8, max_pixels=100_000),
            dict(fps=2.0, max_frames=8, min_pixels=50_000), dict(fps=2.0, max_frames=8, total_pixels=5_000_000),
            dict(fps=2.0, max_frames=8, resized_height=224, resized_width=224),
        ):
            assert VideoControls(**change).cache_key_fragment() != base, change

    def test_forwarding_kwargs_only_carry_set_fields_plus_the_object(self):
        class Req:
            video_fps = 3.0
            video_max_frames = None
            video_max_pixels = 150_000
            video_min_pixels = None
            video_total_pixels = None
            video_resized_height = None
            video_resized_width = None
        kw = video_control_kwargs(Req())
        assert kw == {"video_fps": 3.0, "video_max_pixels": 150_000, "video_controls": VideoControls(fps=3.0, max_pixels=150_000)}
        assert video_control_kwargs({"video_fps": None}) == {}
        assert video_controls_from_kwargs({"foo": 1}) is None
        kwargs = {"video_fps": 5, "video_max_pixels": 1000, "other": 1}
        c = pop_video_control_kwargs(kwargs)
        assert c == VideoControls(fps=5, max_pixels=1000) and kwargs == {"other": 1}

    def test_loader_element_and_clamp_note(self):
        c = VideoControls(fps=1.0, max_frames=4, max_pixels=200_000, min_pixels=50_000, total_pixels=2_000_000, resized_height=224, resized_width=336)
        ele = c.fetch_video_element("/tmp/x.mp4")
        assert ele == {"video": "/tmp/x.mp4", "fps": 1.0, "max_frames": 4, "min_pixels": 50_000, "max_pixels": 200_000, "total_pixels": 2_000_000, "resized_height": 224, "resized_width": 336}
        assert VideoControls(max_pixels=MLX_VLM_VIDEO_MAX_PIXELS).clamp_note() is None
        assert "clamps" in VideoControls(max_pixels=MLX_VLM_VIDEO_MAX_PIXELS + 1).clamp_note()
        assert set(ele) - {"video"} <= {f[len("video_"):] for f in VIDEO_CONTROL_FIELDS}


class TestFallbackFrames:
    def test_pixel_budget_scales_aspect_preserved_and_never_upscales(self):
        np = pytest.importorskip("numpy")
        pytest.importorskip("cv2")
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        out = bound_video_frames([frame], max_pixels=100_000)[0]
        h, w = out.shape[:2]
        assert h * w <= 100_000
        assert abs((w / h) - (1280 / 720)) < 0.02
        assert bound_video_frames([frame], max_pixels=10_000_000)[0].shape == frame.shape
        # both caps: the tighter one wins
        out2 = bound_video_frames([frame], max_long_edge=640, max_pixels=100_000)[0]
        assert max(out2.shape[:2]) <= 640 and out2.shape[0] * out2.shape[1] <= 100_000

    def test_explicit_size_wins(self):
        np = pytest.importorskip("numpy")
        pytest.importorskip("cv2")
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        out = bound_video_frames([frame], max_long_edge=256, max_pixels=100, resize=(224, 336))[0]
        assert out.shape[:2] == (224, 336)

    def test_bounds_from_controls(self):
        b = VideoControls(max_pixels=123_456).fallback_bounds(default_long_edge=768)
        assert (b.max_long_edge, b.max_pixels, b.resize) == (768, 123_456, None)
        b = VideoControls(resized_height=224, resized_width=224).fallback_bounds(default_long_edge=768)
        assert b.resize == (224, 224)


class TestApiTransport:
    def test_chat_and_responses_models_validate_and_carry_the_fields(self):
        from vmlx_engine.api.models import ChatCompletionRequest, ResponsesRequest
        for cls, base in ((ChatCompletionRequest, {"model": "m", "messages": [{"role": "user", "content": "hi"}]}), (ResponsesRequest, {"model": "m", "input": "hi"})):
            req = cls(**base, video_fps=3, video_max_frames=12, video_max_pixels=200_000, video_resized_height=224, video_resized_width=224)
            assert (req.video_fps, req.video_max_frames, req.video_max_pixels, req.video_resized_height, req.video_resized_width) == (3, 12, 200_000, 224, 224)
            with pytest.raises(ValueError):
                cls(**base, video_fps=-2)
            with pytest.raises(ValueError):
                cls(**base, video_resized_height=224)
            with pytest.raises(ValueError):
                cls(**base, video_max_frames=0)

    def test_anthropic_request_forwards_every_control(self):
        from vmlx_engine.api.anthropic_adapter import AnthropicRequest, to_chat_completion
        req = AnthropicRequest(model="m", messages=[{"role": "user", "content": "hi"}], max_tokens=10, video_fps=2.5, video_max_frames=6, video_max_pixels=150_000, video_min_pixels=50_000, video_total_pixels=900_000, video_resized_height=224, video_resized_width=336)
        chat = to_chat_completion(req)
        assert video_control_kwargs(chat)["video_controls"] == VideoControls(fps=2.5, max_frames=6, max_pixels=150_000, min_pixels=50_000, total_pixels=900_000, resized_height=224, resized_width=336)
        with pytest.raises(ValueError):
            AnthropicRequest(model="m", messages=[{"role": "user", "content": "hi"}], max_tokens=10, video_fps=0)

    def test_ollama_chat_forwards_top_level_and_options_controls(self):
        from vmlx_engine.api.ollama_adapter import ollama_chat_to_openai
        body = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "video_fps": 4, "options": {"video_max_pixels": 120_000}}
        req = ollama_chat_to_openai(body)
        assert req["video_fps"] == 4 and req["video_max_pixels"] == 120_000
        assert "video_max_frames" not in req

    def test_server_forwards_on_all_four_dialect_sites(self):
        import vmlx_engine.server as server
        src = inspect.getsource(server)
        # chat + responses + anthropic + ollama streaming
        assert src.count("video_control_kwargs(request)") >= 2
        assert "_msg_kwargs.update(video_control_kwargs(chat_req))" in src
        assert "chat_kwargs.update(video_control_kwargs(chat_req))" in src
        # the truthiness forwarding that dropped explicit values is gone
        assert 'if request.video_fps:\n        chat_kwargs["video_fps"]' not in src


class TestEngineHops:
    def test_scheduler_and_batch_request_carry_the_object(self):
        from vmlx_engine.mllm_scheduler import MLLMRequest
        from vmlx_engine.mllm_batch_generator import MLLMBatchRequest
        assert "video_controls" in MLLMRequest.__dataclass_fields__
        assert "video_controls" in MLLMBatchRequest.__dataclass_fields__

    def test_media_digests_change_with_pixel_controls(self):
        from vmlx_engine.mllm_batch_generator import _mllm_media_item_digest, _mllm_media_cache_extra_keys

        class Req:
            images = []
            videos = ["data:video/mp4;base64,AAAA"]
            audio = None
            image_token_budget = None
            video_fps = 2.0
            video_max_frames = 8
            video_controls = None
            image_grid_thw = None
            video_grid_thw = None
            audio_codes = None
            pixel_values = None
            pixel_values_videos = None
            video_pixel_values = None
        a = Req()
        b = Req(); b.video_controls = VideoControls(fps=2.0, max_frames=8, max_pixels=100_000)
        c = Req(); c.video_controls = VideoControls(fps=2.0, max_frames=8)  # same as the plain fields
        assert _mllm_media_item_digest(a, "video", a.videos[0]) != _mllm_media_item_digest(b, "video", b.videos[0])
        assert _mllm_media_item_digest(a, "video", a.videos[0]) == _mllm_media_item_digest(c, "video", c.videos[0])
        ka, kb = _mllm_media_cache_extra_keys(a), _mllm_media_cache_extra_keys(b)
        assert ka is not None and kb is not None and ka != kb

    def test_pixel_cache_key_carries_pixel_controls(self):
        from vmlx_engine.mllm_batch_generator import _pixel_cache_prompt_key
        base = _pixel_cache_prompt_key("p", has_videos=True, video_fps=2.0, video_max_frames=8)
        with_px = _pixel_cache_prompt_key("p", has_videos=True, video_fps=2.0, video_max_frames=8, video_controls=VideoControls(fps=2.0, max_frames=8, max_pixels=100_000))
        same = _pixel_cache_prompt_key("p", has_videos=True, video_controls=VideoControls(fps=2.0, max_frames=8))
        assert with_px != base and same == base

    def test_native_fetch_builds_the_loader_element_from_controls(self, monkeypatch):
        import types, sys
        import vmlx_engine.mllm_batch_generator as g
        seen = {}
        fake = types.ModuleType("mlx_vlm.video_generate")
        def fetch_video(ele, return_video_sample_fps=False):
            seen.update(ele); return ([1, 2], 1.5)
        fake.fetch_video = fetch_video
        monkeypatch.setitem(sys.modules, "mlx_vlm.video_generate", fake)
        frames, fps = g._fetch_video_for_processor("/tmp/v.mp4", fps=2.0, max_frames=8, controls=VideoControls(fps=2.0, max_frames=8, max_pixels=100_000, resized_height=224, resized_width=224))
        assert frames == [1, 2] and fps == 1.5
        assert seen == {"video": "/tmp/v.mp4", "fps": 2.0, "max_frames": 8, "max_pixels": 100_000, "resized_height": 224, "resized_width": 224}

    def test_frame_fallback_and_simple_engine_use_the_controls(self):
        import vmlx_engine.engine.batched as b
        import vmlx_engine.models.mllm as m
        bs, ms = inspect.getsource(b), inspect.getsource(m)
        assert "controls.fallback_bounds(" in bs and "controls.cache_key_fragment()" in bs
        assert "bound_video_frames(" in bs
        assert "pop_video_control_kwargs(kwargs)" in ms and "controls.pixel_fragment()" in ms


class TestVideoTelemetry:
    def test_frame_count_and_size_helpers(self):
        np = pytest.importorskip("numpy")
        from vmlx_engine.mllm_batch_generator import _video_frame_count, _video_frame_size
        arr = np.zeros((6, 3, 224, 336), dtype=np.uint8)
        assert (_video_frame_count(arr), _video_frame_size(arr)) == (6, "224x336")
        frames = [np.zeros((180, 320, 3), dtype=np.uint8)] * 4
        assert (_video_frame_count(frames), _video_frame_size(frames)) == (4, "180x320")
        assert _video_frame_count(object()) == -1 and _video_frame_size(object()) == "?"

    def test_batch_generator_logs_video_input_and_pixel_cache_outcome(self):
        import vmlx_engine.mllm_batch_generator as g
        src = inspect.getsource(g)
        assert '"Video input for %s: sampled_frames=%d sample_fps=%.3f frame=%s timestamps=%s controls=%s"' in src
        from vmlx_engine.mllm_batch_generator import _format_timestamps
        assert _format_timestamps([0.0, 0.5, 1.0]) == "[0.00,0.50,1.00]"
        assert _format_timestamps(list(range(12))).endswith("] n=12") and _format_timestamps(None) == "[]"
        assert '"Vision pixel cache %s for %s: %d media item(s) key=%s%s"' in src
        assert "_dump_pixel_cache_miss(request.request_id, media_cache_sources, pixel_cache_prompt, pixel_cache_key)" in src


class TestVideoTokenBudget:
    def test_token_budget_derives_the_clip_pixel_budget(self):
        from vmlx_engine.video_controls import VIDEO_TOKEN_IMAGE_FACTOR
        c = validate_video_controls({"video_token_budget": 1024})
        assert c.token_budget == 1024
        assert c.effective_total_pixels() == 1024 * VIDEO_TOKEN_IMAGE_FACTOR ** 2
        assert c.fetch_video_element("/v.mp4")["total_pixels"] == 1024 * 784
        # an explicit total_pixels wins over the derivation
        both = validate_video_controls({"video_token_budget": 1024, "video_total_pixels": 5000})
        assert both.fetch_video_element("/v.mp4")["total_pixels"] == 5000
        with pytest.raises(ValueError, match="video_token_budget"):
            validate_video_controls({"video_token_budget": 0})

    def test_token_budget_is_part_of_cache_identity_and_fallback_bounds(self):
        a = VideoControls(fps=2.0, max_frames=8)
        b = VideoControls(fps=2.0, max_frames=8, token_budget=512)
        assert a.cache_key_fragment() != b.cache_key_fragment() and b.has_pixel_controls
        bounds = b.fallback_bounds(default_long_edge=768)
        assert bounds.max_pixels == (512 * 784) // 8

    def test_api_models_accept_token_budget(self):
        from vmlx_engine.api.models import ChatCompletionRequest, ResponsesRequest
        from vmlx_engine.api.anthropic_adapter import AnthropicRequest, to_chat_completion
        assert ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "hi"}], video_token_budget=2048).video_token_budget == 2048
        assert ResponsesRequest(model="m", input="hi", video_token_budget=2048).video_token_budget == 2048
        a = AnthropicRequest(model="m", messages=[{"role": "user", "content": "hi"}], max_tokens=5, video_token_budget=2048)
        assert to_chat_completion(a).video_token_budget == 2048


def test_frame_fallback_summary_logs_bounds_and_controls():
    import vmlx_engine.engine.batched as b
    src = inspect.getsource(b)
    assert '"max_long_edge=%s, max_pixels=%s, resize=%s, controls=%s)"' in src


class TestPixelCacheKeyStability:
    def test_store_with_the_lookup_key_survives_a_removed_temp_file(self):
        import os, tempfile
        import mlx.core as mx
        from vmlx_engine.vision_embedding_cache import VisionEmbeddingCache
        c = VisionEmbeddingCache(max_pixel_entries=4)
        a = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False); a.write(b"same bytes" * 50); a.close()
        key = c.make_key([a.name], "p")
        assert c.get_pixel_cache([a.name], "p") is None
        os.unlink(a.name)  # the temp file is gone before the store
        c.set_pixel_cache([a.name], "p", pixel_values=None, video_pixel_values=mx.array([1.0]), input_ids=mx.array([1]), key=key)
        b = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False); b.write(b"same bytes" * 50); b.close()
        assert c.get_pixel_cache([b.name], "p") is not None  # same bytes, new path: HIT
        # and the unstable way (key rebuilt after removal) would have missed
        c2 = VisionEmbeddingCache(max_pixel_entries=4)
        d = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False); d.write(b"other" * 50); d.close(); os.unlink(d.name)
        c2.set_pixel_cache([d.name], "p", pixel_values=None, video_pixel_values=mx.array([1.0]), input_ids=mx.array([1]))
        e = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False); e.write(b"other" * 50); e.close()
        assert c2.get_pixel_cache([e.name], "p") is None

    def test_batch_generator_stores_under_the_lookup_key_and_logs_the_outcome(self):
        import vmlx_engine.mllm_batch_generator as g
        src = inspect.getsource(g)
        assert "pixel_cache_key = (" in src and "key=pixel_cache_key," in src
        assert '"Vision pixel cache STORE for %s: %d media item(s), %.2fs of processing"' in src
        assert '"Vision pixel cache NOT STORED for %s: %s"' in src
        assert '"vision memory cache disabled (--no-vision-memory-cache); media reuse relies on the prefix cache"' in src
        assert '"DISABLED" if not self.vision_cache.enabled else "MISS"' in src
        # a disabled cache never claims a STORE
        assert "and self.vision_cache.enabled\n            and media_cache_sources" in src


class TestCleanMediaBoundaryMatchesFetchContract:
    """The fetch side admits a media hit only when it covers every placeholder
    or ends before the first one. The store side must not capture a boundary
    inside the media span (declined next time, store wasted)."""

    def _gen(self, placeholder=99, block=64):
        import types
        from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator
        fake = types.SimpleNamespace(
            _ssm_block_aligned_boundary=lambda n: (n // block) * block,
            block_aware_cache=types.SimpleNamespace(block_size=block),
            _media_placeholder_token_ids=lambda: {placeholder},
        )
        fake._media_placeholder_span = lambda ids: MLLMBatchGenerator._media_placeholder_span(fake, ids)
        fake._media_placeholder_run_at = lambda ids, b: MLLMBatchGenerator._media_placeholder_run_at(fake, ids, b)
        fake._model_type = "qwen4_exp"
        return fake, MLLMBatchGenerator._media_clean_cache_boundary_for

    def test_boundary_after_the_media_when_it_fits(self):
        import types
        fake, fn = self._gen()
        tokens = [1] * 100 + [99] * 500 + [2] * 200   # media 100..600, N=800, N-1=799 -> 768 >= 600
        assert fn(fake, types.SimpleNamespace(request_id="r"), tokens) == 768

    def test_aligned_boundary_inside_media_falls_back_to_the_exact_n_minus_1(self):
        import types
        fake, fn = self._gen()
        # media 100..780, N-1=799 -> aligned 768 cuts the media; the exact N-1 sits after it and is what
        # the paged fetch matches on a repeat or a longer next turn (live: 1771 vs media 4..1764)
        tokens = [1] * 100 + [99] * 680 + [2] * 20
        assert fn(fake, types.SimpleNamespace(request_id="r"), tokens) == 799
        # a tail that leaves an aligned boundary after the media keeps it
        tokens = [1] * 100 + [99] * 650 + [2] * 20    # media 100..750, N-1=769 -> 768 >= 750
        assert fn(fake, types.SimpleNamespace(request_id="r"), tokens) == 768

    def test_no_boundary_only_when_nothing_lies_outside_the_media(self):
        import types
        fake, fn = self._gen()
        tokens = [1] * 10 + [99] * 700 + [2] * 20     # media 10..710, N-1=729 -> 704 cuts; exact 729 is after it
        assert fn(fake, types.SimpleNamespace(request_id="r"), tokens) == 729
        tokens = [1] * 10 + [99] * 720                # media runs to the end and starts in the first block -> none
        assert fn(fake, types.SimpleNamespace(request_id="r"), tokens) == 0

    def test_qwen_family_refuses_any_boundary_inside_the_media_span(self):
        import types
        fake, fn = self._gen()
        fake._model_type = "qwen4_exp"
        # two videos with text between; N-1 aligned boundary (768) lands between them → still inside the span for Qwen
        tokens = [1] * 100 + [99] * 300 + [3] * 500 + [99] * 40 + [2] * 20    # span 100..940, N-1=959 -> 896 inside the span -> exact 959
        assert fn(fake, types.SimpleNamespace(request_id="r"), tokens) == 959
        tokens = [1] * 100 + [99] * 300 + [3] * 500 + [99] * 60               # media to the end -> pre-media 64
        assert fn(fake, types.SimpleNamespace(request_id="r"), tokens) == 64

    def test_other_families_keep_a_boundary_between_whole_media_items(self):
        import types
        fake, fn = self._gen()
        fake._model_type = "muse_glimmer"
        tokens = [1] * 100 + [99] * 300 + [3] * 500 + [99] * 40 + [2] * 20    # 896 lies in the text between the two runs -> kept
        assert fn(fake, types.SimpleNamespace(request_id="r"), tokens) == 896

    def test_text_only_prompt_keeps_the_terminal_boundary(self):
        import types
        fake, fn = self._gen()
        tokens = [1] * 300
        assert fn(fake, types.SimpleNamespace(request_id="r"), tokens) == 256

    def test_capture_site_skips_when_no_boundary(self):
        import vmlx_engine.mllm_batch_generator as g
        src = inspect.getsource(g)
        assert '"MLLM media prefix cache: no boundary outside the "' in src and '"media span for %s (N-1=%d); nothing stored for "' in src
        assert "if clean_media_cache is None and _clean_media_len > 0:" in src


def test_processed_video_grid_is_logged_with_media_tokens():
    import types
    import vmlx_engine.mllm_batch_generator as g
    src = inspect.getsource(g)
    assert '"Video processed for %s: grid_thw=%s media_tokens=%s input_ids=%d token_budget=%s"' in src
    class Grid:
        def tolist(self): return [[16, 26, 46]]
    assert g._format_grid(Grid()) == "[16x26x46]"
    proc = types.SimpleNamespace(video_processor=types.SimpleNamespace(merge_size=2))
    assert g._grid_media_tokens(Grid(), proc) == f"{16*26*46//4} (merge=2)"


class TestProcessorSideClipBudget:
    """Measured live (video-tokens-194316, Flash-Next 4S): budgets 512/256/128 all produced 728 media tokens because the
    loader's per-frame cap floors at its own minimum (224x420). The transformers Qwen3-VL video processor sizes the
    whole clip from size.longest_edge, so the budget is enforced there, request-locally, with the processor's real
    pixels-per-token factor (patch 16 x merge 2 squared x temporal 2 = 2,048; the legacy 784 asked for 2.6x too few)."""

    def _proc(self, patch=16, merge=2, temporal=2, short=100352, long=12582912):
        import types
        return types.SimpleNamespace(video_processor=types.SimpleNamespace(patch_size=patch, merge_size=merge, temporal_patch_size=temporal, size={"shortest_edge": short, "longest_edge": long}))

    def test_pixels_per_token_come_from_the_processor(self):
        from vmlx_engine.video_controls import video_token_pixels, DEFAULT_VIDEO_TOKEN_PIXELS
        assert video_token_pixels(self._proc()) == 2048
        assert video_token_pixels(self._proc(patch=14)) == 1568  # Qwen2.5-VL geometry
        assert video_token_pixels(None) == DEFAULT_VIDEO_TOKEN_PIXELS == 784
        assert video_token_pixels(object()) == 784

    def test_budget_becomes_a_request_local_size_for_the_processor(self):
        from vmlx_engine.video_controls import VideoControls
        c = VideoControls(token_budget=512).with_processor(self._proc())
        assert c.token_pixels == 2048 and c.effective_total_pixels() == 512 * 2048
        assert c.processor_video_kwargs(self._proc()) == {"size": {"shortest_edge": 100352, "longest_edge": 1048576}}
        # a budget below the processor's own minimum lowers the minimum too, so the size dict stays valid
        tiny = VideoControls(token_budget=32).with_processor(self._proc())
        assert tiny.processor_video_kwargs(self._proc()) == {"size": {"shortest_edge": 65536, "longest_edge": 65536}}
        # explicit total pixels win over the budget; no clip budget -> nothing is sent
        assert VideoControls(total_pixels=300000).processor_video_kwargs(self._proc())["size"]["longest_edge"] == 300000
        assert VideoControls(fps=1, max_frames=4).processor_video_kwargs(self._proc()) is None
        assert VideoControls(max_pixels=100352).processor_video_kwargs(self._proc()) is None  # per-frame cap stays loader-side

    def test_call_site_forwards_the_size_without_touching_the_processor(self):
        import inspect
        import vmlx_engine.mllm_batch_generator as g
        src = inspect.getsource(g)
        assert "videos_kwargs=(" in src and ".with_processor(self.processor).processor_video_kwargs(self.processor)" in src
        assert 'kwargs["videos_kwargs"] = dict(videos_kwargs)' in src
        assert 'token_budget=%s"' in src  # the processed-grid line names the budget it was asked to meet


class TestClipBudgetAppliedToTheSampledClip:
    """The engine's mlx-vlm Qwen3-VL processor ignores per-call videos_kwargs (probe on the box: size and
    do_sample_frames changed nothing, 1760 tokens every time), so the clip budget is applied to the sampled frames
    with the processor's own rounding before they reach it."""

    def test_dims_match_the_transformers_smart_resize_result(self):
        from vmlx_engine.video_controls import clip_budget_dims
        # measured: 16 frames 364x644 under 512 tokens x 2048 px -> 192x320 -> grid 8x12x20 -> 480 tokens
        assert clip_budget_dims(16, 364, 644, total_pixels=512 * 2048, factor=32, temporal=2) == (192, 320)
        assert clip_budget_dims(16, 364, 644, total_pixels=256 * 2048, factor=32, temporal=2) == (128, 224)  # 8x4x7 -> 224 tokens
        # already within budget: unchanged rounding only
        assert clip_budget_dims(16, 364, 644, total_pixels=16 * 364 * 644 * 4, factor=32, temporal=2) == (352, 640)

    def test_the_loader_site_applies_it_and_the_helper_resizes_channels_first_clips(self):
        import inspect, numpy as np, types
        import vmlx_engine.mllm_batch_generator as g
        from vmlx_engine.video_controls import VideoControls
        src = inspect.getsource(g)
        assert "video_input = _apply_clip_pixel_budget(" in src and "strict=bool(getattr(request, \"media_controls_strict\", False))" in src
        proc = types.SimpleNamespace(video_processor=types.SimpleNamespace(patch_size=16, merge_size=2, temporal_patch_size=2))
        clip = np.random.randint(0, 255, size=(16, 3, 364, 644)).astype(np.float32)
        out = g._apply_clip_pixel_budget(clip, VideoControls(token_budget=512).with_processor(proc), proc, "r")
        assert out.shape == (16, 3, 192, 320) and out.dtype == np.float32
        # no budget -> the same object comes back
        assert g._apply_clip_pixel_budget(clip, VideoControls(fps=2), proc, "r") is clip


class TestFrameFallbackPlan:
    """27B (qwen3_5) frame fallback: every sampled frame is one IMAGE with the image processor's own pixel floor
    (Qwen2VLImageProcessorFast size.shortest_edge=65536 -> >=64 tokens/frame at 1024 px/token). Live at 8a664585 the
    per-frame budget was total // frame CAP (128), so a 16-frame clip got an eighth of its budget and the floor made
    every budget land at ~1101-1277 prompt tokens; 32-frame rows failed admission as '32 images' (limit 20)."""

    GEOM = dict(frame_height=364, frame_width=644, token_pixels=1024, pixel_floor=65536, pixel_ceiling=16_777_216)

    def plan(self, budget, *, frames=16, cap=20, **kw):
        from vmlx_engine.video_controls import VideoControls, plan_fallback_frames

        c = VideoControls(token_budget=budget, token_pixels=2048)
        return plan_fallback_frames(c, frames_available=frames, frame_cap=cap, **{**self.GEOM, **kw})

    def test_budget_reduces_frames_at_the_processor_floor_and_expected_total_fits(self):
        p = self.plan(512)
        assert p.num_frames == 7 and p.expected_tokens_per_frame == 66 and p.expected_total == 462 and p.met is True
        assert "reduced 16->7" in p.reason
        p = self.plan(1024)
        assert p.num_frames == 15 and p.expected_total == 990 and p.met is True

    def test_generous_budget_keeps_every_sampled_frame(self):
        p = self.plan(2048)
        assert p.num_frames == 16 and p.expected_total == 1920 and p.met is True and p.reason == "budget met"

    def test_impossible_budget_keeps_one_frame_and_reports_not_met(self):
        p = self.plan(4)
        assert p.num_frames == 1 and p.met is False and "below the image processor floor" in p.reason
        assert p.expected_tokens_per_frame >= 64  # the floor, never a fabricated 4

    def test_frame_cap_from_the_image_limit_bounds_the_plan(self):
        p = self.plan(8192, frames=32, cap=10)
        assert p.num_frames == 10 and p.frame_cap == 10
        p = self.plan(None, frames=32, cap=10)
        assert p.num_frames == 10 and p.expected_total is None and p.met is None and p.reason == "no budget"

    def test_explicit_max_pixels_wins_over_the_budget_split(self):
        from vmlx_engine.video_controls import VideoControls, plan_fallback_frames

        c = VideoControls(max_pixels=100_000, token_budget=512, token_pixels=2048)
        p = plan_fallback_frames(c, frames_available=16, frame_cap=20, **self.GEOM)
        assert p.num_frames == 16 and p.per_frame_max_pixels == 100_000 and p.reason == "explicit max_pixels"

    def test_frame_tokens_model_the_processor_floor_after_our_bound(self):
        from vmlx_engine.video_controls import fallback_frame_tokens, smart_resize_dims

        # bounded to 3136 px (the old 512-token/128-cap split) the processor upsizes to >= 65536 px
        assert fallback_frame_tokens(364, 644, per_frame_max_pixels=3136, token_pixels=1024, pixel_floor=65536) >= 64
        assert smart_resize_dims(364, 644, factor=32, min_pixels=65536, max_pixels=16_777_216) == (352, 640)
        assert smart_resize_dims(30, 50, factor=32, min_pixels=65536, max_pixels=None)[0] % 32 == 0

    def test_even_subsample_keeps_first_and_last(self):
        from vmlx_engine.video_controls import subsample_frames_evenly

        assert subsample_frames_evenly(list(range(32)), 10) == [0, 3, 7, 10, 14, 17, 21, 24, 28, 31]
        assert subsample_frames_evenly(list(range(5)), 5) == [0, 1, 2, 3, 4]
        assert subsample_frames_evenly(list(range(5)), 1) == [0]
        assert subsample_frames_evenly([], 3) == []

    def test_fallback_bounds_spread_the_budget_over_the_sampled_frames_not_the_cap(self):
        from vmlx_engine.video_controls import VideoControls

        c = VideoControls(token_budget=512, token_pixels=784)  # cap 128 (default), 16 sampled
        assert c.fallback_bounds(default_long_edge=768).max_pixels == 512 * 784 // 128  # legacy split (cap)
        assert c.fallback_bounds(default_long_edge=768, num_frames=16).max_pixels == 512 * 784 // 16
        assert c.fallback_bounds(default_long_edge=768, per_frame_max_pixels=70_000).max_pixels == 70_000

    def test_engine_fallback_splits_the_image_limit_across_videos_and_plans_per_video(self):
        import inspect

        from vmlx_engine.engine import batched

        src = inspect.getsource(batched.BatchedEngine._video_frame_fallback_messages)
        assert "max_images_per_request" in src and "_count_video_and_image_parts(messages)" in src
        assert "per_video_frame_cap = max(1, (int(image_limit) - n_images) // max(1, n_videos))" in src
        assert "plan_fallback_frames(" in src and "subsample_frames_evenly(frames, plan.num_frames)" in src
        assert "per_frame_max_pixels=plan.per_frame_max_pixels" in src and "|cap={per_video_frame_cap}" in src
        assert "video frame fallback plan" in src
        videos, images = batched.BatchedEngine._count_video_and_image_parts(
            [{"role": "user", "content": [{"type": "video_url", "video_url": {"url": "a"}}, {"type": "image_url", "image_url": {"url": "b"}}, {"type": "text", "text": "q"}]},
             {"role": "user", "content": "plain"},
             {"role": "user", "content": [{"type": "input_video", "video": "c"}]}]
        )
        assert (videos, images) == (2, 1)


class TestEffectiveSettingsReporting:
    """Best-effort contract: when a control cannot be honoured as sent, the response carries a `video_controls:` warning
    naming the effective settings. Nothing is reported when the request was honoured exactly."""

    GEOM = dict(frame_height=364, frame_width=644, token_pixels=1024, pixel_floor=65536, pixel_ceiling=16_777_216)

    def plan(self, budget, frames=16, cap=20):
        from vmlx_engine.video_controls import VideoControls, plan_fallback_frames

        return plan_fallback_frames(VideoControls(token_budget=budget, token_pixels=2048), frames_available=frames, frame_cap=cap, **self.GEOM)

    def test_fallback_reduced_frames_are_reported_with_effective_values(self):
        from vmlx_engine.video_controls import fallback_plan_diagnostics

        msgs = fallback_plan_diagnostics(self.plan(512), sampled=16, pixel_floor=65536, token_pixels=1024)
        assert len(msgs) == 1 and msgs[0].startswith("video_controls: frame fallback kept 7 of 16 sampled frames")
        assert "462 media tokens" in msgs[0] and "74898 px" in msgs[0]

    def test_fallback_impossible_budget_reports_best_effort_not_rejection(self):
        from vmlx_engine.video_controls import fallback_plan_diagnostics

        msgs = fallback_plan_diagnostics(self.plan(4), sampled=16, pixel_floor=65536, token_pixels=1024)
        assert len(msgs) == 1 and "video_token_budget=4 cannot be met" in msgs[0] and "not rejected" in msgs[0]
        assert "77 media tokens" in msgs[0]

    def test_fallback_frame_cap_and_explicit_size_below_floor_are_reported(self):
        from vmlx_engine.video_controls import fallback_plan_diagnostics

        msgs = fallback_plan_diagnostics(self.plan(8192, frames=32, cap=10), sampled=32, pixel_floor=65536, token_pixels=1024, resize=(224, 224))
        assert any("32 sampled frames exceed the per-video frame cap 10" in m for m in msgs)
        assert any("explicit size 224x224 is below the image processor floor" in m and "effective 256x256, 64 tokens/frame" in m for m in msgs)

    def test_fallback_honoured_budget_reports_nothing(self):
        from vmlx_engine.video_controls import fallback_plan_diagnostics

        assert fallback_plan_diagnostics(self.plan(2048), sampled=16, pixel_floor=65536, token_pixels=1024) == []

    def test_native_min_pixels_override_and_impossible_budget_are_reported(self):
        from vmlx_engine.video_controls import VideoControls, clip_budget_diagnostics, clip_budget_dims

        c = VideoControls(token_budget=256, min_pixels=100352, token_pixels=2048)
        h, w = clip_budget_dims(16, 364, 644, total_pixels=c.effective_total_pixels(), min_total_pixels=100352 * 16, factor=32, temporal=2)
        msgs = clip_budget_diagnostics(c, num_frames=16, height=364, width=644, resized=(h, w), factor=32, temporal=2)
        assert len(msgs) == 1 and "video_min_pixels=100352 x 16 frames" in msgs[0] and "the budget wins" in msgs[0]
        c4 = VideoControls(token_budget=4, token_pixels=2048)
        h, w = clip_budget_dims(16, 364, 644, total_pixels=c4.effective_total_pixels(), factor=32, temporal=2)
        msgs = clip_budget_diagnostics(c4, num_frames=16, height=364, width=644, resized=(h, w), factor=32, temporal=2)
        assert (h, w) == (32, 32) and len(msgs) == 1 and "video_token_budget=4 cannot be met" in msgs[0] and "8 media tokens" in msgs[0]

    def test_native_honoured_budget_reports_nothing(self):
        from vmlx_engine.video_controls import VideoControls, clip_budget_diagnostics, clip_budget_dims

        c = VideoControls(token_budget=512, token_pixels=2048)
        h, w = clip_budget_dims(16, 364, 644, total_pixels=c.effective_total_pixels(), factor=32, temporal=2)
        assert clip_budget_diagnostics(c, num_frames=16, height=364, width=644, resized=(h, w), factor=32, temporal=2) == []


def test_request_diagnostics_registry_drains_into_the_context_bucket():
    from vmlx_engine import request_diagnostics as rd

    rd.DIAGNOSTICS.set(None)  # another test may have left a capture open in this context
    with rd._REGISTRY_LOCK:
        rd._REGISTRY.clear()
    assert rd.record("x") is False  # no capture running: dropped, never raised
    rd.record_for("chatcmpl-a", "video_controls: one")
    rd.record_for("chatcmpl-a", "video_controls: two")
    assert rd.pending_for("chatcmpl-a") == ["video_controls: one", "video_controls: two"]
    rd.begin_capture()
    assert rd.drain_into_context("chatcmpl-a") == 2 and rd.pending_for("chatcmpl-a") == []
    assert rd.drain_into_context("chatcmpl-missing") == 0
    assert rd.take() == ["video_controls: one", "video_controls: two"] and rd.take() == []
    # bounded: the oldest request's entries are dropped past the cap
    for i in range(rd._REGISTRY_MAX_REQUESTS + 5):
        rd.record_for(f"r{i}", "m")
    assert rd.pending_for("r0") == [] and rd.pending_for(f"r{rd._REGISTRY_MAX_REQUESTS + 4}") == ["m"]


def test_effective_video_settings_reach_the_response_warnings_in_both_lanes():
    """Wiring pin: the fallback plan and the native clip budget record by request id; the engine drains the registry
    into the request context after generation (non-stream) and before the terminal chunk (stream); the server's
    warnings bucket IS the shared context variable, so the existing take() delivers them on every lane."""
    import inspect

    from vmlx_engine import server
    from vmlx_engine.engine import batched
    from vmlx_engine import mllm_batch_generator, request_diagnostics

    assert server._TOOL_CALL_DROP_DIAGNOSTICS is request_diagnostics.DIAGNOSTICS
    # 1e9214e5 aliased the bucket but the server's take still read it directly (live: 17 recorded lines, empty warnings)
    request_diagnostics.DIAGNOSTICS.set(None); request_diagnostics.note_request_id("chatcmpl-take")
    request_diagnostics.record_for("chatcmpl-take", "video_controls: via server take")
    assert server._take_tool_call_drop_diagnostics() == ["video_controls: via server take"]
    request_diagnostics.note_request_id(None)
    fb = inspect.getsource(batched.BatchedEngine._video_frame_fallback_messages)
    assert "fallback_plan_diagnostics(" in fb and "record_for(request_id, _msg)" in fb
    gen = inspect.getsource(batched.BatchedEngine.generate)
    assert gen.index("self._mllm_scheduler.generate(") < gen.index("_drain_request_diagnostics(request_id)")
    st = inspect.getsource(batched.BatchedEngine.stream_generate)
    assert "if output.finished:" in st and st.index("_drain_request_diagnostics(request_id)") < st.index("yield GenerationOutput(")
    clip = inspect.getsource(mllm_batch_generator._apply_clip_pixel_budget)
    assert "clip_budget_diagnostics(" in clip and "record_for(request_id, _msg)" in clip


def test_take_drains_the_registry_for_the_noted_request_id_even_without_an_open_bucket():
    """Live at 670b5e58: every video row had the right effective settings in the engine log and an EMPTY warnings
    array — the JSON chat handler opens its capture bucket only AFTER the engine returns, and the engine's drain
    dropped the entries. Now the handler notes the request id up front and take() drains the registry for it; the
    engine-side drain leaves entries alone when no capture is open."""
    from vmlx_engine import request_diagnostics as rd

    rd.DIAGNOSTICS.set(None); rd.CURRENT_REQUEST_ID.set(None)
    with rd._REGISTRY_LOCK:
        rd._REGISTRY.clear()
    rd.record_for("chatcmpl-v", "video_controls: effective")
    assert rd.drain_into_context("chatcmpl-v") == 0 and rd.pending_for("chatcmpl-v") == ["video_controls: effective"]  # kept, not dropped
    rd.note_request_id("chatcmpl-v")
    assert rd.take() == ["video_controls: effective"] and rd.pending_for("chatcmpl-v") == []
    # another request's entries never leak into this one
    rd.record_for("chatcmpl-other", "video_controls: other")
    assert rd.take() == [] and rd.pending_for("chatcmpl-other") == ["video_controls: other"]
    with rd._REGISTRY_LOCK:
        rd._REGISTRY.clear()


def test_server_notes_the_engine_request_id_before_every_engine_hand_off():
    import re

    from vmlx_engine import server
    from pathlib import Path

    src = Path(server.__file__).read_text()
    hand_offs = [m.start() for m in re.finditer(r"request_id=response_id[,)]?\n", src)]
    assert hand_offs, "no engine hand-off found"
    for pos in hand_offs:
        window = src[max(0, pos - 4000):pos]
        assert "_note_request_diagnostics_id(response_id)" in window, f"hand-off at {pos} without a noted request id"
