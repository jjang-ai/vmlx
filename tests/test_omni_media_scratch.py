"""Decoded request media must not become an unbounded persistent cache."""
import base64
import threading
from pathlib import Path

import pytest

from vmlx_engine import omni_multimodal as omni


@pytest.mark.parametrize("outcome", ["complete", "encode_error", "cancelled"])
def test_request_media_is_alive_during_turn_and_cleaned_on_every_exit(tmp_path, outcome):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    other_request = scratch / "other-request"
    other_request.mkdir()
    other_file = other_request / "keep.png"
    other_file.write_bytes(b"other request")
    external = tmp_path / "external.png"
    external.write_bytes(b"external image")
    observed = []

    class Session:
        def reset(self):
            pass

        def turn(self, **kwargs):
            observed.extend(kwargs["images"])
            assert [p.read_bytes() for p in observed] == [b"request image", b"external image"]
            if outcome == "encode_error":
                raise ValueError("encoder failed")
            if outcome == "cancelled":
                raise omni._OmniStreamCancelled("cancelled")
            return "answer"

    d = omni.OmniMultimodalDispatcher.__new__(omni.OmniMultimodalDispatcher)
    d._session = Session()
    d._backend = "stage1"
    d._lock = threading.Lock()
    d._scratch_dir = scratch
    d._last_signature = None
    messages = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(b"request image").decode()}},
        {"type": "image_url", "image_url": {"url": str(external)}},
        {"type": "text", "text": "Describe."},
    ]}]
    if outcome == "complete":
        assert d.chat(messages, enable_thinking=False)["content"] == "answer"
    else:
        with pytest.raises(ValueError if outcome == "encode_error" else omni._OmniStreamCancelled):
            d.chat(messages, enable_thinking=False)
    assert observed and not observed[0].exists()
    assert external.read_bytes() == b"external image"
    assert other_file.read_bytes() == b"other request"
    assert list(scratch.iterdir()) == [other_request]


def test_cold_history_media_uses_same_request_lifetime(tmp_path, monkeypatch):
    class Session:
        def reset(self):
            pass

    observed = []
    def history(session, messages, *, scratch_dir, extract_parts, **kwargs):
        _, images, _, _ = extract_parts(messages[:1], scratch_dir)
        observed.extend(images)
        assert images[0].read_bytes() == b"image"
        return "answer"

    monkeypatch.setattr(omni, "_run_omni_full_history", history)
    d = omni.OmniMultimodalDispatcher.__new__(omni.OmniMultimodalDispatcher)
    d._session = Session()
    d._backend = "stage1"
    d._lock = threading.Lock()
    d._scratch_dir = tmp_path
    d._last_signature = None
    d.chat([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,aW1hZ2U="}}]},
            {"role": "assistant", "content": "old answer"}, {"role": "user", "content": "Recall."}], enable_thinking=False)
    assert observed and not observed[0].exists()
    assert list(tmp_path.iterdir()) == []
