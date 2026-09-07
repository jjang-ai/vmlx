import asyncio
from types import SimpleNamespace

import mlx.core as mx
import numpy as np


def test_turboquant_config_threads_compress_after_into_plain_cache(monkeypatch):
    from vmlx_engine.utils.turboquant_config import (
        TurboQuantConfig,
        make_turboquant_cache,
    )

    cfg = TurboQuantConfig.from_jang_config(
        {"turboquant": {"enabled": True, "compress_after": 17}}, 1
    )
    cache = make_turboquant_cache(cfg, 1, [8], [8], ["attention"])
    assert cache[0].compress_after == 17


def test_turboquant_real_compress_records_tokens_and_resident_delta():
    from jang_tools.turboquant.cache import TurboQuantKVCache

    from vmlx_engine.utils.turboquant_config import (
        TurboQuantConfig,
        make_turboquant_cache,
    )

    cache = make_turboquant_cache(
        TurboQuantConfig(
            n_layers=1, compress_after=2, sink_tokens=0, critical_layers=[]
        ),
        1,
        [8],
        [8],
        ["attention"],
    )[0]
    keys = mx.zeros((1, 1, 3, 8), dtype=mx.float16)
    cache.update_and_fetch(keys, keys)
    telemetry = TurboQuantKVCache._vmlx_last_compress
    assert telemetry["compressed_tokens_after"] == 2
    assert telemetry["resident_after_bytes"] > 0
    assert telemetry["resident_delta_bytes"] > 0


def test_hybrid_turboquant_threads_compress_after_and_preserves_companion():
    from mlx_lm.models.cache import ArraysCache, KVCache

    from vmlx_engine.utils.hybrid_tq_cache import build_hybrid_turboquant_make_cache
    from vmlx_engine.utils.turboquant_config import TurboQuantConfig

    cfg = TurboQuantConfig(n_layers=2, compress_after=23, critical_layers=[])
    make_cache = build_hybrid_turboquant_make_cache(
        lambda: [ArraysCache(size=2), KVCache()],
        cfg,
        8,
        8,
        ["ssm", "attention"],
    )
    cache = make_cache()
    assert isinstance(cache[0], ArraysCache)
    assert type(cache[1]).__name__ == "TurboQuantKVCache"
    assert cache[1].compress_after == 23


def test_seeded_sampler_is_repeatable_and_request_local():
    from vmlx_engine.sampling import make_sampler

    logprobs = mx.log(mx.array([[0.1, 0.2, 0.3, 0.4]]))

    def draw(seed):
        sampler = make_sampler(temp=0.9, top_p=1.0, seed=seed)
        return [int(sampler(logprobs).item()) for _ in range(20)]

    assert draw(7) == draw(7)
    assert draw(7) != draw(8)


def test_reasoning_effort_enables_supported_family_only(monkeypatch):
    from vmlx_engine import server

    class Registry:
        config = SimpleNamespace(
            family_name="gemma4",
            model_type="gemma4",
            supports_thinking=True,
            architecture_hints={},
        )

        def lookup(self, _key):
            return self.config

    monkeypatch.setattr(
        "vmlx_engine.model_config_registry.get_model_config_registry",
        lambda: Registry(),
    )
    assert server._resolve_enable_thinking(
        None, {}, False, "gemma", reasoning_effort="high"
    ) is True
    Registry.config.supports_thinking = False
    assert server._resolve_enable_thinking(
        None, {}, False, "non-thinking", reasoning_effort="high"
    ) is False


def test_reasoning_effort_never_overrides_explicit_off(monkeypatch):
    from vmlx_engine import server

    class Registry:
        config = SimpleNamespace(
            family_name="gemma4",
            model_type="gemma4",
            supports_thinking=True,
            architecture_hints={},
        )

        def lookup(self, _key):
            return self.config

    monkeypatch.setattr(
        "vmlx_engine.model_config_registry.get_model_config_registry",
        lambda: Registry(),
    )
    monkeypatch.setattr(server, "_default_enable_thinking", None)
    assert server._resolve_enable_thinking(
        False, {}, False, "gemma", reasoning_effort="high"
    ) is False
    assert server._resolve_enable_thinking(
        None, {"enable_thinking": False}, False, "gemma", reasoning_effort="high"
    ) is False
    assert server._resolve_enable_thinking(
        None, {}, False, "gemma", reasoning_effort="none"
    ) is None


def test_ollama_streaming_applies_hy3_reasoning_dialect():
    import inspect

    from vmlx_engine import server

    source = inspect.getsource(server.ollama_chat)
    policy_index = source.index("_apply_hy3_reasoning_policy(")
    stream_index = source.index("async def ndjson_stream()")
    assert policy_index < stream_index


def test_capabilities_alias_uses_active_model(monkeypatch):
    from vmlx_engine import server

    monkeypatch.setattr(server, "_resolve_model_name", lambda: "loaded-model")

    async def fake(model_id):
        return {"model": model_id}

    monkeypatch.setattr(server, "model_capabilities", fake)
    assert asyncio.run(server.active_model_capabilities()) == {
        "model": "loaded-model"
    }


def test_nemotron_omni_is_selective_attention_kv_allowlisted():
    from vmlx_engine.utils.hybrid_tq_cache import is_selective_hybrid_tq_supported

    assert is_selective_hybrid_tq_supported(
        {"model_type": "nemotron_h"}, ["ssm", "attention"]
    )
    assert not is_selective_hybrid_tq_supported(
        {"model_type": "unknown_hybrid"}, ["ssm", "attention"]
    )
    selective_tq = open("vmlx_engine/utils/hybrid_tq_cache.py").read()
    assert '"nemotron_h",' in selective_tq


def test_gemma4_mixed_swa_tq_uses_global_attention_head_dim():
    from vmlx_engine.utils.model_inspector import _detect_turboquant_layer_types

    cfg = {
        "model_type": "gemma4_unified_text",
        "head_dim": 256,
        "global_head_dim": 512,
        "layer_types": ["sliding_attention", "full_attention"],
    }
    layer_types, key_dim, value_dim = _detect_turboquant_layer_types(cfg, 2)
    assert layer_types == ["attention", "attention"]
    assert (key_dim, value_dim) == (512, 512)


def test_ui_defaults_prefix_on_paged_off_and_hy3_mtp_native_type_visible():
    form = open(
        "panel/src/renderer/src/components/sessions/SessionConfigForm.tsx"
    ).read()
    sessions = open("panel/src/main/sessions.ts").read()
    registry = open("panel/src/main/model-config-registry.ts").read()
    assert "enablePrefixCache: true" in form
    # DEFAULT_CONFIG pre-detection fallback stays false; the effective default now
    # comes from the detection layer (see registry assertion below).
    assert "usePagedCache: false" in form
    # 2026-08-23: in-RAM paged cache is OFF for EVERY family, DSV4 included.
    # SSD block-disk L2 is the only cache tier, so the launch default is a flat
    # false and no per-family capability feeds it.
    assert "const defaultUsePagedCache = false" in sessions
    assert "dsv4Active ? true" not in sessions
    assert "detectedUsePaged" not in sessions
    assert "dsv4PrefixOptIn" not in sessions
    # The detection fallback no longer lets an UNDECLARED text family acquire a
    # RAM tier by omission -- that used to default every plain-KV text model
    # paged-ON.
    assert "usePagedCache: config.usePagedCache ?? false" in registry
    assert "config.isMultimodal ? false : true" not in registry
    assert "'hy_v3'" in registry
    # Both HY3 and ERNIE expose plain attention state through native MTP.
    assert "nativeCacheType: hy3 || ernie45 ? 'plain_kv_v1'" in " ".join(registry.split())
    # The "native cache: <type>" note renders from the locale catalog since the
    # i18n pass. The invariant is what this test is named for — the DETECTED
    # native cache type (hy3's plain_kv_v1) stays visible in the UI — so assert
    # the form interpolates the detected type into the note and the English
    # template still prints it.
    en_catalog = open("panel/src/renderer/src/i18n/locales/en.json").read()
    assert "t('sessions.config.nativeMtpDetectedNote'" in form
    assert (
        "cache: detectedNativeMtp?.nativeCacheType || detectedCacheSubtype || detectedCacheType || 'unknown'"
        in form
    )
    assert "native cache: {cache}" in en_catalog


def test_laguna_dedicated_loader_reaches_mixed_swa_patch():
    source = open("vmlx_engine/loaders/load_laguna.py").read()
    assert "_patch_turboquant_make_cache(model, jang_cfg, model.config)" in source


def test_memory_prefix_q4_storage_is_stream_independent_numpy_and_restores():
    from mlx_lm.models.cache import KVCache

    from vmlx_engine.scheduler import Scheduler

    scheduler = Scheduler.__new__(Scheduler)
    scheduler._kv_cache_bits = 4
    scheduler._kv_cache_group_size = 64
    scheduler.config = SimpleNamespace(use_paged_cache=False)
    cache = KVCache()
    cache.keys = mx.ones((1, 2, 8, 64), dtype=mx.float16)
    cache.values = mx.ones((1, 2, 8, 64), dtype=mx.float16)
    cache.offset = 8

    stored = scheduler._quantize_cache_for_storage([cache])
    assert all(isinstance(part, np.ndarray) for part in stored[0].keys)
    restored = scheduler._dequantize_cache_for_use(stored)
    mx.eval(restored[0].keys, restored[0].values)
    assert restored[0].keys.shape == cache.keys.shape
