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

    def test_boundary_inside_media_falls_back_to_pre_media(self):
        import types
        fake, fn = self._gen()
        tokens = [1] * 100 + [99] * 680 + [2] * 20    # media 100..780, N-1=799 -> 768 cuts the media -> 64 (before)
        assert fn(fake, types.SimpleNamespace(request_id="r"), tokens) == 64
        # a tail that leaves an aligned boundary after the media keeps it
        tokens = [1] * 100 + [99] * 650 + [2] * 20    # media 100..750, N-1=769 -> 768 >= 750
        assert fn(fake, types.SimpleNamespace(request_id="r"), tokens) == 768

    def test_no_boundary_when_media_starts_in_the_first_block(self):
        import types
        fake, fn = self._gen()
        tokens = [1] * 10 + [99] * 700 + [2] * 20     # media 10..710, N-1=729 -> 704 cuts; before = 0 -> none
        assert fn(fake, types.SimpleNamespace(request_id="r"), tokens) == 0

    def test_qwen_family_refuses_any_boundary_inside_the_media_span(self):
        import types
        fake, fn = self._gen()
        fake._model_type = "qwen4_exp"
        # two videos with text between; N-1 aligned boundary (768) lands between them → still inside the span for Qwen
        tokens = [1] * 100 + [99] * 300 + [3] * 500 + [99] * 40 + [2] * 20    # span 100..940, N-1=959 -> 896 inside the span -> before = 64
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
    assert '"Video processed for %s: grid_thw=%s media_tokens=%s input_ids=%d"' in src
    class Grid:
        def tolist(self): return [[16, 26, 46]]
    assert g._format_grid(Grid()) == "[16x26x46]"
    proc = types.SimpleNamespace(video_processor=types.SimpleNamespace(merge_size=2))
    assert g._grid_media_tokens(Grid(), proc) == f"{16*26*46//4} (merge=2)"
