"""Legacy MiniMax video processors must accept the current RGB dispatch."""

from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("legacy", [True, False])
def test_video_rgb_dispatch_preserves_processor_default_and_override(monkeypatch, legacy):
    from transformers import AutoProcessor
    from vmlx_engine.models.minimax_m3 import m3_vl_preprocess as media

    class LegacyVideo:
        do_convert_rgb = True

        def _preprocess(self, videos, do_convert_rgb):
            return videos, do_convert_rgb

    class CurrentVideo:
        do_convert_rgb = True

        def _preprocess(self, videos, do_convert_rgb=True):
            return videos, do_convert_rgb

    video = LegacyVideo() if legacy else CurrentVideo()
    processor = SimpleNamespace(video_processor=video)
    monkeypatch.setattr(AutoProcessor, "from_pretrained", lambda *a, **kw: processor)
    monkeypatch.setattr(media, "_PROCESSOR_CACHE", {})
    loaded = media._get_processor("test-video")
    # Current Transformers has already consumed the RGB option at this point.
    assert loaded.video_processor._preprocess(["frame"]) == (["frame"], True)
    # Older dispatchers still explicitly pass it; their choice must win.
    assert loaded.video_processor._preprocess(["frame"], do_convert_rgb=False) == (
        ["frame"], False
    )
    assert media._get_processor("test-video") is loaded
