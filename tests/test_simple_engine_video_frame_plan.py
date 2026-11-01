"""SimpleEngine plans video frames like the batched frame fallback and never refuses on the image limit.

Audit 2026-10-07 (ISSUES I-24/I-33): a 20 s clip at the default 2 fps (40 frames) returned HTTP 400 "Total image count
(40, including video frames) exceeds limit of 20" on 27B (SimpleEngine) -- every pixel/budget control was unusable for
clips over 10 s -- while the batched engine planned the same clip to the per-video cap. SimpleEngine also kept frames at
full resolution (~882 tokens/frame vs ~292 batched).
"""
import inspect
from types import SimpleNamespace

import numpy as np
import pytest

import vmlx_engine.models.mllm as mllm
from vmlx_engine.models.mllm import MLXMultimodalLM, _simple_video_frame_cap
from vmlx_engine.video_controls import VideoControls


def _lm(monkeypatch, n_frames=40, h=720, w=1280):
    lm = MLXMultimodalLM.__new__(MLXMultimodalLM)
    lm.processor = None
    frames = [np.full((h, w, 3), i, dtype=np.uint8) for i in range(n_frames)]
    monkeypatch.setattr(mllm, "process_video_input", lambda v: v)
    monkeypatch.setattr(mllm, "extract_video_frames_smart", lambda path, fps, max_frames: list(frames))
    monkeypatch.setattr(mllm, "save_frames_to_temp", lambda fr: fr)
    return lm


def test_frame_cap_splits_the_image_limit_across_videos():
    assert _simple_video_frame_cap({}, 1, 0) == 20
    assert _simple_video_frame_cap({}, 2, 2) == 9
    assert _simple_video_frame_cap({"max_images_per_request": 64}, 1, 0) == 64
    assert _simple_video_frame_cap({}, 1, 30) == 1


def test_default_clip_is_planned_to_the_cap_and_bounded_not_refused(monkeypatch):
    lm = _lm(monkeypatch)
    out = lm._prepare_video("clip.mp4", fps=2.0, max_frames=128, controls=None, frame_cap=20)
    assert len(out) == 20
    assert int(out[0][0, 0, 0]) == 0 and int(out[-1][0, 0, 0]) == 39  # evenly spread, first and last kept
    assert max(out[0].shape[:2]) <= 768  # batched default long edge


def test_explicit_max_frames_below_the_cap_is_honoured(monkeypatch):
    lm = _lm(monkeypatch)
    out = lm._prepare_video("clip.mp4", fps=2.0, max_frames=4, controls=VideoControls(max_frames=4), frame_cap=20)
    assert len(out) == 4


def test_explicit_resize_wins(monkeypatch):
    lm = _lm(monkeypatch)
    ctl = VideoControls(resized_height=224, resized_width=224)
    out = lm._prepare_video("clip.mp4", fps=2.0, max_frames=128, controls=ctl, frame_cap=20)
    assert out[0].shape[:2] == (224, 224)


def test_image_limit_is_advisory_on_every_simple_engine_path():
    src = inspect.getsource(MLXMultimodalLM)
    assert "including video frames) \"\n" not in src
    assert src.count("above the advisory") == 4
    assert "exceeds limit of" not in src


def test_sampled_timestamps_follow_frames_after_budget_subsampling(monkeypatch):
    lm = _lm(monkeypatch, n_frames=5, h=28, w=28)
    frames = [np.full((28, 28, 3), i, dtype=np.uint8) for i in range(5)]
    metadata = [{"frame_index": i, "timestamp_seconds": t}
                for i, t in enumerate([0.0, 0.2, 1.3, 2.8, 5.0])]
    monkeypatch.setattr(mllm, "extract_video_frames_smart",
                        lambda *a, **kw: (frames, metadata))
    selected, times = lm._prepare_video("clip.mp4", frame_cap=3, return_metadata=True)
    assert [int(f[0, 0, 0]) for f in selected] == [0, 2, 4]
    assert [m["timestamp_seconds"] for m in times] == [0.0, 1.3, 5.0]


def test_glm_frame_fallback_keeps_clip_boundaries_and_original_timestamps():
    earlier = {"role": "user", "content": [{"type": "image"}]}
    current = {"role": "user", "content": [
        {"type": "video"}, {"type": "text", "text": "then"}, {"type": "video"},
    ]}
    out = mllm._expand_video_placeholders_to_image_frames(
        [earlier, current], [2, 1], glm_timestamps=[[0.0, 1.75], [4.2]],
    )
    assert out[0] == earlier
    text = [p["text"] for p in out[1]["content"] if p["type"] == "text"]
    assert text == ["<|begin_of_video|>", "0.0 seconds", "1.8 seconds", "<|end_of_video|>",
                    "then", "<|begin_of_video|>", "4.2 seconds", "<|end_of_video|>"]
    assert sum(p["type"] == "image" for p in out[1]["content"]) == 3
    assert current["content"][0]["type"] == "video"


@pytest.mark.parametrize("times", [[None], [float("nan")], [-1.0], []])
def test_glm_frame_fallback_does_not_invent_missing_timestamps(times):
    with pytest.raises(ValueError, match="timestamps"):
        mllm._expand_video_placeholders_to_image_frames(
            [{"role": "user", "content": [{"type": "video"}]}], [1], glm_timestamps=[times],
        )


@pytest.mark.parametrize("native_video_pixels", [False, True])
def test_glm_timeline_markers_keep_dflash_image_plan_but_native_tensors_do_not(monkeypatch, native_video_pixels):
    import mlx.core as mx
    import mlx_vlm.utils
    import vmlx_engine.glm_dflash2

    lm = MLXMultimodalLM.__new__(MLXMultimodalLM)
    lm.model_name = "glm"
    lm.processor = object()
    lm.model = SimpleNamespace(
        language_model=SimpleNamespace(),
        get_input_embeddings=lambda *a, **kw: pytest.fail("media materialization must stay lazy"),
        config=SimpleNamespace(image_token_id=10, video_token_id=11, video_start_token_id=12),
    )
    inputs = {"input_ids": mx.array([[12, 10, 10, 10, 10, 13, 42]]),
              "pixel_values": mx.zeros((4, 8)), "image_grid_thw": mx.array([[1, 2, 2]])}
    if native_video_pixels:
        inputs["pixel_values_videos"] = mx.zeros((4, 8))
        inputs["video_grid_thw"] = mx.array([[1, 2, 2]])
    monkeypatch.setattr(mlx_vlm.utils, "prepare_inputs", lambda *a, **kw: inputs)
    monkeypatch.setattr(vmlx_engine.glm_dflash2, "is_glm5_next", lambda model: True)
    plan = lm._dflash2_media_plan("timeline", ["frame"], {})
    if native_video_pixels:
        assert plan is None
    else:
        assert plan["prompt_tokens"] == [12, 10, 10, 10, 10, 13, 42]
        assert plan["first_media_index"] == 1
        assert plan["media_end"] == 5
