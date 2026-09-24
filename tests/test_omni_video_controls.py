"""Video preprocessing must obey request controls and never reuse stale frames."""
import os

import numpy as np
import pytest
from PIL import Image

from vmlx_engine import omni_multimodal as omni
from vmlx_engine.video_controls import VideoControls


def test_same_video_path_with_replaced_bytes_cannot_reuse_old_frames(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"old")
    frame = np.full((8, 8, 3), 20, dtype=np.uint8)
    monkeypatch.setattr("vmlx_engine.models.mllm.extract_video_frames_smart", lambda *a, **kw: [frame])
    before = omni._extract_omni_video_frames(video, tmp_path)
    old_stat = video.stat()
    video.write_bytes(b"new")
    os.utime(video, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
    frame[:] = 220
    after = omni._extract_omni_video_frames(video, tmp_path)
    assert after != before
    with Image.open(after[0]) as image:
        assert np.array(image).mean() > 200


def test_video_environment_controls_are_part_of_frame_identity(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    observed = []

    def sample(*args, **kwargs):
        observed.append(kwargs)
        return [np.full((8, 8, 3), int(kwargs['fps']) * 50, dtype=np.uint8)]

    monkeypatch.setattr("vmlx_engine.models.mllm.extract_video_frames_smart", sample)
    monkeypatch.setenv("VMLINUX_OMNI_VIDEO_FPS", "1")
    before = omni._extract_omni_video_frames(video, tmp_path)
    monkeypatch.setenv("VMLINUX_OMNI_VIDEO_FPS", "3")
    after = omni._extract_omni_video_frames(video, tmp_path)
    assert observed[0]['fps'] == 1 and observed[1]['fps'] == 3
    assert before != after


def test_video_request_controls_reach_frame_sampler(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    observed = []
    monkeypatch.setattr("vmlx_engine.models.mllm.extract_video_frames_smart",
                        lambda *a, **kw: observed.append(kw) or [np.zeros((8, 8, 3), dtype=np.uint8)])
    omni._extract_omni_video_frames(video, tmp_path, video_controls=VideoControls(fps=3, max_frames=2))
    assert observed == [{"fps": 3, "max_frames": 2}]


def test_native_media_identity_detects_content_change_with_same_size_and_mtime(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"old")
    part = {"type": "video_url", "video_url": {"url": str(video)}}
    before = omni._media_part_identity(part)
    old_stat = video.stat()
    video.write_bytes(b"new")
    os.utime(video, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
    assert omni._media_part_identity(part) != before


def test_radio_frame_cap_wins_over_shared_sampler_temporal_minimum(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    monkeypatch.setenv("VMLINUX_OMNI_VIDEO_CONTACT_SHEET", "0")
    monkeypatch.setattr("vmlx_engine.models.mllm.extract_video_frames_smart",
                        lambda *a, **kw: [np.full((8, 8, 3), i * 50, dtype=np.uint8) for i in range(4)])
    result = omni._extract_omni_video_frames(video, tmp_path, video_controls=VideoControls(fps=1, max_frames=1))
    assert len(result) == 1


def test_failed_frame_sampling_cannot_fall_back_to_fixed_native_defaults(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"invalid")
    monkeypatch.setattr("vmlx_engine.models.mllm.extract_video_frames_smart", lambda *a, **kw: [])
    with pytest.raises(ValueError, match="no readable frames"):
        omni._extract_omni_video_frames(video, tmp_path, video_controls=VideoControls(fps=3, max_frames=2))


def test_video_policy_invalidates_native_state_only_after_video_prefix():
    text = [{"role": "user", "content": "hello"}]
    video = text + [{"role": "user", "content": [{"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA"}}]}]
    low = omni._omni_video_policy(VideoControls(fps=1, max_frames=2))
    high = omni._omni_video_policy(VideoControls(fps=3, max_frames=4))
    assert omni._conversation_signature(text, False, video_policy=low) == omni._conversation_signature(text, False, video_policy=high)
    assert omni._conversation_signature(video, False, video_policy=low) != omni._conversation_signature(video, False, video_policy=high)


@pytest.mark.asyncio
@pytest.mark.parametrize("control", ["video_max_pixels", "video_total_pixels", "video_token_budget"])
async def test_native_video_does_not_silently_ignore_pixel_or_token_budget(monkeypatch, control):
    from fastapi import HTTPException
    from vmlx_engine.api.models import ChatCompletionRequest
    monkeypatch.setattr(omni, "omni_multimodal_component_status", lambda _: {"modalities": ["text", "video"]})
    monkeypatch.setattr(omni.OmniMultimodalDispatcher, "get", lambda *a, **kw: pytest.fail("unsupported control reached model load"))
    request = ChatCompletionRequest(model="omni", messages=[{"role": "user", "content": [
        {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA"}},
    ]}], **{control: 4096})
    with pytest.raises(HTTPException) as exc:
        await omni.dispatch_omni_chat_completion(request, "/unused")
    assert exc.value.status_code == 400
    assert "not supported" in exc.value.detail
