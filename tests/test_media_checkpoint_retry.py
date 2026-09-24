"""Aborted native checkpoints cannot outlive an outer prefill retry."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("failures,fatal", [(1, False), (2, False), (3, False), (1, True)])
@pytest.mark.parametrize("media", [None, "images", "video_pixel_values", "audio_features", "history"])
def test_outer_prefill_retry_discards_native_snapshots(monkeypatch, failures, fatal, media):
    import vmlx_engine.mllm_batch_generator as module

    monkeypatch.setattr(module.MLLMBatchGenerator, "_stream", module.mx.default_stream(module.mx.gpu))

    class LM:
        layers = []

        def __call__(self, ids, cache=None):
            return module.mx.zeros((1, 8, 4))

        def make_cache(self):
            return []

    gen = module.MLLMBatchGenerator.__new__(module.MLLMBatchGenerator)
    gen.language_model = gen._cache_model = LM()
    gen._stats = SimpleNamespace(prompt_tokens=0, prompt_time=0)
    gen._is_hybrid = gen._ssm_companion_enabled = False
    gen._prefix_cache_enabled = gen._decode_trace = False
    gen.block_aware_cache = gen.memory_aware_cache = None
    gen.prefix_cache = gen.disk_cache = None
    gen._hybrid_kv_positions = []
    gen._prefill_errors = []
    gen._drain_tight_memory_allocator = lambda *args: None
    gen._media_scoped_cache_extra_keys = lambda *args: {}
    gen._tokens_contain_media_placeholders = lambda ids: 99 in ids
    gen._media_prefix_cache_allowed = lambda *args: False
    gen._prepare_native_mtp_prompt_priming = lambda *args: None
    gen._seed_native_mtp_from_prefill = lambda *args: None
    gen._make_request_sampler = lambda req: lambda logits: module.mx.array([2])
    req = module.MLLMBatchRequest(uid=1, request_id="retry", prompt="one")
    if media and media != "history":
        setattr(req, media, ["image"] if media == "images" else module.mx.zeros((1,)))
    sibling = module.MLLMBatchRequest(uid=2, request_id="sibling", prompt="two")
    sibling._media_clean_prefix_cache = ["sibling-owned"]
    forwards = []
    retry_snapshots = []
    block_cache = SimpleNamespace(release_cache=Mock(), clear=Mock())

    def snapshots(request):
        return (
            getattr(request, "_media_clean_prefix_cache", None),
            getattr(request, "_media_clean_prefix_len", 0),
            getattr(request, "_media_clean_native", False),
            getattr(request, "_media_repair_ssm_checkpoint", None),
            getattr(request, "_media_clean_capture_boundaries", ()),
        )

    def preprocess(request):
        request.input_ids = module.mx.arange(8)[None, :]
        request._original_token_ids = list(range(8))
        if request is req and media == "history":
            request._original_token_ids[0] = 99
            request.input_ids = module.mx.array([request._original_token_ids])

    def forward(request, cache):
        forwards.append(request.request_id)
        if request is req:
            attempt = forwards.count("retry")
            if attempt > 1:
                retry_snapshots.append(snapshots(request))
            if attempt <= failures:
                request._media_clean_prefix_cache = [object()]
                request._media_clean_prefix_len = 6
                request._media_clean_native = True
                request._media_repair_ssm_checkpoint = (2, [0, 1], [object()])
                request._media_clean_capture_boundaries = (2, 6)
                gen.block_aware_cache = block_cache
                raise RuntimeError("invalid media input" if fatal else "broadcast stale shape")
        return module.mx.zeros((1, 8, 4))

    gen._preprocess_request = preprocess
    gen._run_vision_encoding = forward
    batch = gen._process_prompts([req, sibling])
    succeeded = not fatal and not media and failures < 3
    assert batch.request_ids == (["retry", "sibling"] if succeeded else ["sibling"])
    assert len(gen._prefill_errors) == (0 if succeeded else 1)
    assert forwards == ["retry"] * (1 if fatal or media else min(failures + 1, 3)) + ["sibling"]
    assert retry_snapshots == [(None, 0, False, None, ())] * len(retry_snapshots)
    assert snapshots(req) == (None, 0, False, None, ())
    assert sibling._media_clean_prefix_cache == ["sibling-owned"]
    assert block_cache.clear.call_count == (1 if not fatal and not media and failures >= 2 else 0)
