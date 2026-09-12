"""GLM opt-in fingerprints observe, never rewrite, native feature placement."""
import json
from types import SimpleNamespace

import mlx.core as mx
import pytest

from vmlx_engine.models.glm5_next.vlm import Model, _trace_media_scatter


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
@pytest.mark.parametrize("modality", ["image", "video"])
def test_trace_matches_each_actual_group_without_changing_output(monkeypatch, caplog, dtype, modality):
    ids = mx.array([[1, 10, 10, 2, 10, 10, 3]])
    original = mx.arange(28).reshape(1, 7, 4).astype(dtype)
    features = (mx.arange(16).reshape(4, 4) * 0.125 - 0.75).astype(dtype)
    grid = mx.array([[2, 2, 2]]) if modality == "video" else mx.array([[1, 2, 2], [1, 2, 2]])
    monkeypatch.delenv("VMLX_GLM5_MEDIA_TRACE", raising=False)
    baseline = Model._scatter_features(original, ids, features, 10, modality, grid=grid)
    mx.eval(baseline)
    assert not caplog.records
    monkeypatch.setenv("VMLX_GLM5_MEDIA_TRACE", "1")
    with caplog.at_level("INFO"):
        traced = Model._scatter_features(original, ids, features, 10, modality, grid=grid)
    assert bool(mx.array_equal(baseline, traced).item())
    payload = json.loads(caplog.records[-1].message.split("GLM media scatter trace: ", 1)[1])
    assert payload["exact_placement"] and payload["finite"]
    assert payload["features_dtype"] == payload["merged_dtype"] == str(dtype)
    assert [group["rows"] for group in payload["groups"]] == [2, 2]
    assert len({group["source_sha256"] for group in payload["groups"]}) == 2
    assert all(group["source_sha256"] == group["placed_sha256"] for group in payload["groups"])


def test_large_trace_is_metadata_only_without_materializing(monkeypatch, caplog):
    monkeypatch.setenv("VMLX_GLM5_MEDIA_TRACE", "1")
    features = SimpleNamespace(shape=(4097, 4096), size=4097 * 4096, dtype=mx.bfloat16)
    with caplog.at_level("INFO"):
        _trace_media_scatter(features, SimpleNamespace(dtype=mx.bfloat16), None, "video", None, mx.bfloat16)
    payload = json.loads(caplog.records[-1].message.split("GLM media scatter trace: ", 1)[1])
    assert payload["fingerprints"] == "skipped_element_limit"
    assert "groups" not in payload
